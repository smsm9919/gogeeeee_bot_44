# -*- coding: utf-8 -*-
"""
RF Futures Bot — DOGE/USDT (BingX Perp, CCXT)
Entry: Pure Range Filter (closed-candle)
Post-entry profit engine: EMA-9 Touch + EMA-20 Gate + ADX/RSI + ATR trailing
No scalping. Single-file, production-ready (Flask + logs + strict close).
"""

import os, time, math, random, threading, traceback, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, InvalidOperation

import pandas as pd
import ccxt
from flask import Flask, jsonify

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# ========= ENV / MODE =========
API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)

SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
KEEPALIVE_SECONDS = 50
PORT = int(os.getenv("PORT", 5000))

# ========= MARKET & STRATEGY =========
SYMBOL   = "DOGE/USDT:USDT"
INTERVAL = "15m"
LEVERAGE = 10               # هنرفعه 15x بعد ما البوت يثبت نجاحه (يدويًا)
RISK_ALLOC = 0.60           # نسبة من الرصيد × ليفريج علشان الحجم

# --- Range Filter (إشارة الدخول) ---
RF_SOURCE = "close"
RF_PERIOD = 20
RF_MULT   = 3.5

# --- مؤشرات القوة/الخروج ---
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

# --- إدارة ربح/تريل أساسية ---
TP1_PCT          = 0.40     # أول هدف ربح كنسبة %
TP1_CLOSE_FRAC   = 0.40     # إغلاق جزئي عند TP1
BREAKEVEN_AFTER  = 0.30     # تحريك الستوب لنقطة الدخول بعد الربح ده
TRAIL_ACTIVATE   = 0.60     # تفعيل التريل بعد ربح معيّن
ATR_MULT_TRAIL   = 1.6      # معامل ATR للتريل العادي

# --- EMA Touch Engine (الجديد) ---
EMA_FAST = 9
EMA_BASE = 20
TOUCH_TOL_BPS = 15          # 0.15% سماحية للمس EMA-9
EMA_RSI_THR   = 55
EMA_ADX_MIN   = 25
EMA_ADX_DELTA = 1.5
DEFERRAL_BARS = 2           # عدد الشموع لتأجيل TP
ATR_MULT_THRUST = 2.0       # تريل أقوى بعد تأكيد اللمس

# --- حمايات تنفيذ ---
SPREAD_GUARD_BPS = 6
STRICT_EXCHANGE_CLOSE = True
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0
MIN_RESIDUAL_TO_FORCE = 1.0

BASE_SLEEP = 8
NEAR_CLOSE_SLEEP = 1

STATE_FILE = "bot_state.json"

print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • {SYMBOL} • {INTERVAL}", "yellow"))

# ========= LOGGING =========
def setup_logs():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        root.addHandler(fh)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(colored("🗂️ logging ready", "cyan"))

setup_logs()

# ========= EXCHANGE =========
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY, "secret": API_SECRET,
        "enableRateLimit": True, "timeout": 20000,
        "options": {"defaultType": "swap"}
    })

ex = make_exchange()

MARKET, AMT_PREC, LOT_STEP, LOT_MIN = {}, 0, None, None
def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int(MARKET.get("precision", {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits", {}).get("amount", {}) or {}).get("step", None)
        LOT_MIN  = (MARKET.get("limits", {}).get("amount", {}) or {}).get("min", None)
        print(colored(f"📊 specs: prec={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ load specs: {e}", "yellow"))

def ensure_leverage():
    try:
        ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
        print(colored(f"✅ leverage {LEVERAGE}x", "green"))
    except Exception as e:
        print(colored(f"⚠️ set_leverage: {e}", "yellow"))

try:
    load_market_specs()
    if MODE_LIVE: ensure_leverage()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}", "yellow"))

def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        if AMT_PREC>=0:
            d = d.quantize(Decimal(1).scaleb(-AMT_PREC), rounding=ROUND_DOWN)
        if LOT_MIN and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except Exception:
        return max(0.0, float(q or 0.0))

def safe_qty(q):
    q = _round_amt(q)
    if q<=0: print(colored(f"⚠️ qty normalize→{q}", "yellow"))
    return q

# ========= HELPERS =========
def fmt(v,d=6):
    try:
        if v is None: return "—"
        return f"{float(v):.{d}f}"
    except Exception:
        return "—"

def _interval_seconds(iv: str) -> int:
    iv = iv.lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 900

def fetch_ohlcv(limit=600):
    rows = ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = ex.fetch_ticker(SYMBOL)
        return t.get("last") or t.get("close")
    except Exception:
        return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = ex.fetch_balance(params={"type":"swap"})
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception:
        return None

def orderbook_spread_bps():
    try:
        ob = ex.fetch_order_book(SYMBOL, limit=5)
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

def compute_next_sleep(df):
    try:
        tf = _interval_seconds(INTERVAL)
        cur_start = int(df["time"].iloc[-1])
        now_ms = int(time.time()*1000)
        next_close = cur_start + tf*1000
        while next_close <= now_ms: next_close += tf*1000
        left = (next_close - now_ms)//1000
        if left <= 10 or (tf-left)<=8: return NEAR_CLOSE_SLEEP
        return BASE_SLEEP
    except Exception:
        return BASE_SLEEP

# ========= INDICATORS =========
def wilder(s, n): return s.ewm(alpha=1/n, adjust=False).mean()
def ema(s, n):    return s.ewm(span=n, adjust=False).mean()

def indicators(df: pd.DataFrame):
    c = df["close"].astype(float); h = df["high"].astype(float); l = df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder(tr, ATR_LEN)

    delta = c.diff()
    up = delta.clip(lower=0); dn = (-delta).clip(lower=0)
    rs = wilder(up, RSI_LEN) / wilder(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))

    up_move = h.diff(); down_move = l.shift(1) - l
    plus_dm  = up_move.where((up_move>down_move)&(up_move>0), 0.0)
    minus_dm = down_move.where((down_move>up_move)&(down_move>0), 0.0)
    plus_di  = 100 * (wilder(plus_dm, ADX_LEN) / atr.replace(0,1e-12))
    minus_di = 100 * (wilder(minus_dm, ADX_LEN) / atr.replace(0,1e-12))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,1e-12)).fillna(0.0)
    adx = wilder(dx, ADX_LEN)

    e9  = ema(c, EMA_FAST)
    e20 = ema(c, EMA_BASE)

    i = len(df)-1; prev_i = max(0,i-1)
    return {
        "rsi": float(rsi.iloc[i]), "adx": float(adx.iloc[i]),
        "adx_prev": float(adx.iloc[prev_i]),
        "atr": float(atr.iloc[i]),
        "ema9": float(e9.iloc[i]), "ema20": float(e20.iloc[i])
    }

# ========= Range Filter (إشارة الدخول فقط) =========
def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src, qty, n):
    avrng = _ema((src - src.shift(1)).abs(), n); wper = (n*2)-1
    return _ema(avrng, wper)*qty
def _rng_filter(src, rsize):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x-r>prev: cur=x-r
        if x+r<prev: cur=x+r
        rf.append(cur)
    filt = pd.Series(rf, index=src.index, dtype="float64")
    return filt+rsize, filt-rsize, filt

def rf_signal(df: pd.DataFrame):
    if len(df)<RF_PERIOD+3:
        i=-1; p=float(df["close"].iloc[i]); t=int(df["time"].iloc[i])
        return {"time":t,"price":p,"long":False,"short":False,"filter":p,"hi":p,"lo":p}
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    dfilt = filt - filt.shift(1)
    fdir = pd.Series(0.0,index=filt.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0.0)

    upward=(fdir==1).astype(int); downward=(fdir==-1).astype(int)
    src_gt_f=(src>filt); src_lt_f=(src<filt)
    src_gt_p=(src>src.shift(1)); src_lt_p=(src<src.shift(1))
    longCond=(src_gt_f & ((src_gt_p)|(src_lt_p)) & (upward>0))
    shortCond=(src_lt_f & ((src_lt_p)|(src_gt_p)) & (downward>0))

    CondIni=pd.Series(0,index=src.index)
    for i in range(1,len(src)):
        if bool(longCond.iloc[i]): CondIni.iloc[i]=1
        elif bool(shortCond.iloc[i]): CondIni.iloc[i]=-1
        else: CondIni.iloc[i]=CondIni.iloc[i-1]
    longSignal = longCond & (CondIni.shift(1)==-1)
    shortSignal= shortCond & (CondIni.shift(1)==1)

    i=len(df)-1
    return {
        "time": int(df["time"].iloc[i]),
        "price": float(df["close"].iloc[i]),
        "long":  bool(longSignal.iloc[i]),
        "short": bool(shortSignal.iloc[i]),
        "filter": float(filt.iloc[i]),
        "hi": float(hi.iloc[i]), "lo": float(lo.iloc[i]),
        "fdir": float(fdir.iloc[i])
    }

# ========= STATE =========
state = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "bars": 0, "pnl": 0.0, "trail": None,
    "tp1_done": False, "breakeven": None,
    "highest_profit_pct": 0.0,
    # EMA Touch engine
    "deferral_left": 0, "last_touch_ok": False
}
compound_pnl = 0.0
_last_bar_ts = None

def save_state():
    try:
        import json, os
        tmp = STATE_FILE + ".tmp"
        with open(tmp,"w",encoding="utf-8") as f:
            json.dump({"state":state,"compound_pnl":compound_pnl,"ts":time.time()}, f, ensure_ascii=False)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        logging.error(f"save_state: {e}")

def load_state():
    global state, compound_pnl
    try:
        import json, os
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE,"r",encoding="utf-8") as f:
                data=json.load(f)
            state.update(data.get("state",{}))
            compound_pnl = data.get("compound_pnl", 0.0)
            print(colored("✅ state restored", "green"))
    except Exception as e:
        logging.error(f"load_state: {e}")

# ========= SIZING =========
def compute_size(balance, price):
    eff = (balance or 0.0) + (compound_pnl or 0.0)
    capital = eff * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

# ========= ORDERS =========
def _params_open(side: str):
    return {"positionSide":"BOTH","reduceOnly":False}

def _params_close():
    return {"positionSide":"BOTH","reduceOnly":True}

def _read_pos():
    try:
        poss = ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym=(p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0] not in sym: continue
            qty=abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty<=0: return 0.0,None,None
            entry=float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side_raw=(p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side="long" if "long" in side_raw or float(p.get("cost",0))>0 else "short"
            return qty,side,entry
    except Exception as e:
        logging.error(f"_read_pos: {e}")
    return 0.0,None,None

def open_market(side, qty, price):
    if qty<=0: return
    if MODE_LIVE:
        try:
            ex.create_order(SYMBOL,"market",side,qty,None,_params_open(side))
        except Exception as e:
            print(colored(f"❌ open: {e}", "red")); return
    state.update({
        "open": True, "side": "long" if side=="buy" else "short",
        "entry": float(price), "qty": float(qty),
        "bars": 0, "trail": None, "tp1_done": False,
        "breakeven": None, "highest_profit_pct": 0.0,
        "deferral_left": 0, "last_touch_ok": False
    })
    print(colored(f"✅ OPEN {side.upper()} qty={fmt(qty,4)} @ {fmt(price)}", "green" if side=="buy" else "red"))
    save_state()

def close_market_strict(reason):
    global compound_pnl
    exch_qty, exch_side, exch_entry = _read_pos()
    if exch_qty<=0 and not MODE_LIVE:
        exch_qty = state.get("qty",0.0); exch_side = state.get("side"); exch_entry = state.get("entry")
    if exch_qty<=0:
        _reset(reason); return
    side_to_close = "sell" if exch_side=="long" else "buy"
    qty_to_close  = safe_qty(exch_qty)
    attempts, last_error = 0, None
    while attempts < CLOSE_RETRY_ATTEMPTS:
        try:
            if MODE_LIVE:
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,_params_close())
            time.sleep(CLOSE_VERIFY_WAIT_S)
            left,_,_ = _read_pos()
            if left<=0: 
                px = price_now() or state.get("entry")
                pnl = (px - exch_entry)*exch_qty*(1 if exch_side=="long" else -1)
                compound_pnl += pnl
                print(colored(f"🔚 STRICT CLOSE {exch_side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                _reset(reason); return
            qty_to_close = safe_qty(left); attempts +=1
            if qty_to_close < MIN_RESIDUAL_TO_FORCE: time.sleep(CLOSE_VERIFY_WAIT_S)
        except Exception as e:
            last_error=e; attempts+=1; time.sleep(CLOSE_VERIFY_WAIT_S)
    print(colored(f"❌ strict close failed: {last_error}","red"))

def close_partial(frac, reason):
    global compound_pnl
    if not state["open"]: return
    qty_close = safe_qty(state["qty"]*min(max(frac,0.0),1.0))
    if qty_close < 1:
        print(colored(f"⚠️ skip partial ({fmt(qty_close,4)} < 1)", "yellow")); return
    px = price_now() or state["entry"]
    side = "sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try:
            ex.create_order(SYMBOL,"market",side,qty_close,None,_params_close())
        except Exception as e:
            print(colored(f"❌ partial: {e}", "red")); return
    pnl = (px-state["entry"])*qty_close*(1 if state["side"]=="long" else -1)
    compound_pnl += pnl
    state["qty"] -= qty_close
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(state['qty'],4)}","magenta"))
    if state["qty"]<=0: _reset("fully_exited"); return
    save_state()

def _reset(reason):
    state.update({"open":False,"side":None,"entry":None,"qty":0.0,"bars":0,"pnl":0.0,
                  "trail":None,"tp1_done":False,"breakeven":None,
                  "highest_profit_pct":0.0,"deferral_left":0,"last_touch_ok":False})
    print(colored(f"🔚 FLAT • {reason}", "cyan"))
    save_state()

# ========= EMA Touch Logic =========
def ema_touch_confirmation(side, ind, ohlc_row):
    """
    Returns True if we got a strong EMA-9 touch/confirm with momentum.
    For long: low touches <= ema9*(1+tol), and close >= ema9; momentum RSI/ADX.
    """
    ema9 = ind["ema9"]; ema20 = ind["ema20"]
    rsi  = ind["rsi"];  adx = ind["adx"]; adx_prev = ind["adx_prev"]
    o = float(ohlc_row["open"]); h=float(ohlc_row["high"]); l=float(ohlc_row["low"]); c=float(ohlc_row["close"])
    tol = TOUCH_TOL_BPS / 10000.0

    if side == "long":
        touched = (l <= ema9*(1+tol)) and (c >= ema9)
        momentum = (rsi >= EMA_RSI_THR) and (adx >= EMA_ADX_MIN or (adx - adx_prev) >= EMA_ADX_DELTA)
        # gate: الاتجاه العام فوق ema20 أفضل
        gate_ok = c >= ema20
        return bool(touched and momentum and gate_ok)
    else:
        touched = (h >= ema9*(1-tol)) and (c <= ema9)
        momentum = (rsi <= 100-EMA_RSI_THR) and (adx >= EMA_ADX_MIN or (adx - adx_prev) >= EMA_ADX_DELTA)
        gate_ok = c <= ema20
        return bool(touched and momentum and gate_ok)

# ========= PROFIT ENGINE =========
def post_entry_profit_engine(df: pd.DataFrame, ind: dict, info: dict):
    """يُدار بعد الدخول فقط. لا سكالبنج — ركوب الترند وتريل ذكي."""
    if not state["open"] or state["qty"]<=0: return

    # تحديث PnL
    px = info["price"]; e = state["entry"]; side = state["side"]
    rr = (px - e)/e*100.0 * (1 if side=="long" else -1)
    atr = ind["atr"]

    # أقصى ربح وصلناه
    if rr > state["highest_profit_pct"]:
        state["highest_profit_pct"] = rr

    # --- EMA Touch deferral / thrust trail ---
    # بنستخدم آخر شمعة مغلقة (df.iloc[-1]) لأن المنطق على شموع مُغلقة
    last = df.iloc[-1]
    if ema_touch_confirmation(side, ind, last):
        state["deferral_left"] = max(state["deferral_left"], DEFERRAL_BARS)
        state["last_touch_ok"] = True
        # تريل أقوى
        gap = atr * ATR_MULT_THRUST
        if side=="long":
            state["trail"] = max(state["trail"] or (px-gap), px-gap)
        else:
            state["trail"] = min(state["trail"] or (px+gap), px+gap)
        print(colored(f"🔒 EMA9 TOUCH CONFIRMED → deferral {state['deferral_left']} bars • thrust trail set", "green"))
    else:
        state["last_touch_ok"] = False

    # --- TP1 & Breakeven (يتأثر بالتأجيل) ---
    if not state["tp1_done"]:
        if rr >= TP1_PCT:
            if state["deferral_left"]>0:
                # أأجل TP1 لكن أرفع التريل + أضمن Breakeven
                if rr >= BREAKEVEN_AFTER and state["breakeven"] is None:
                    state["breakeven"] = e
                pass
            else:
                close_partial(TP1_CLOSE_FRAC, f"TP1 @{TP1_PCT:.2f}%")
                state["tp1_done"] = True
                if rr >= BREAKEVEN_AFTER and state["breakeven"] is None:
                    state["breakeven"] = e

    # --- تفعيل التريل العادي لو مافيش thrust ---
    if rr >= TRAIL_ACTIVATE and state["deferral_left"]==0:
        gap = atr * ATR_MULT_TRAIL
        if side=="long":
            state["trail"] = max(state["trail"] or (px-gap), px-gap)
        else:
            state["trail"] = min(state["trail"] or (px+gap), px+gap)

    # --- Breakeven guard ---
    if state["breakeven"] is not None:
        if side=="long":
            state["trail"] = max(state["trail"] or state["breakeven"], state["breakeven"])
        else:
            state["trail"] = min(state["trail"] or state["breakeven"], state["breakeven"])

    # --- تنفيذ التريل لو اتضرب ---
    if state["trail"] is not None and atr>0:
        if side=="long" and px < state["trail"]:
            close_market_strict(f"TRAIL_HIT ({'thrust' if state['last_touch_ok'] else 'std'})")
            return
        if side=="short" and px > state["trail"]:
            close_market_strict(f"TRAIL_HIT ({'thrust' if state['last_touch_ok'] else 'std'})")
            return

    # --- منطق حماية في حالة عكس واضح عبر EMA-20 ---
    adx = ind["adx"]; adx_prev = ind["adx_prev"]; rsi = ind["rsi"]
    broke_gate_weak = (adx_prev - adx) >= 2.0
    if side=="long" and last["close"] < ind["ema20"] and rsi < 50 and broke_gate_weak:
        close_partial(0.35, "EMA20 breakdown (weakening)")
    if side=="short" and last["close"] > ind["ema20"] and rsi > 50 and broke_gate_weak:
        close_partial(0.35, "EMA20 break-up (weakening)")

    # --- تناقص مؤقت للتأجيل ---
    if state["deferral_left"]>0:
        state["deferral_left"] -= 1

# ========= BAR COUNTER =========
def update_bar_counters(df):
    global _last_bar_ts
    if len(df)==0: return False
    last_ts = int(df["time"].iloc[-1])
    if _last_bar_ts is None:
        _last_bar_ts = last_ts; return False
    if last_ts != _last_bar_ts:
        _last_bar_ts = last_ts
        if state["open"]: state["bars"] += 1
        return True
    return False

# ========= LOOP =========
def trade_loop():
    load_state()
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            new_bar = update_bar_counters(df)

            info = rf_signal(df)
            ind  = indicators(df)
            spread_bps = orderbook_spread_bps()

            # تحديث PnL
            if state["open"] and px:
                state["pnl"] = (px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]

            # إدارة الصفقة (لو مفتوحة)
            if state["open"]:
                post_entry_profit_engine(df, ind, info)

            # دخول جديد: Range Filter فقط — على الشمعة المُغلقة (إشارات info من آخر شمية مغلقة)
            if (not state["open"]) and (spread_bps is None or spread_bps <= SPREAD_GUARD_BPS):
                sig = "buy" if info["long"] else ("sell" if info["short"] else None)
                if sig:
                    qty = compute_size(bal, info["price"])
                    if qty>0:
                        open_market(sig, qty, info["price"])

            snapshot(bal, info, ind, spread_bps)
            save_state()
            time.sleep(compute_next_sleep(df))
        except Exception as e:
            print(colored(f"❌ loop: {e}\n{traceback.format_exc()}", "red"))
            time.sleep(BASE_SLEEP)

# ========= SNAPSHOT =========
def snapshot(bal, info, ind, spread_bps):
    print(colored("—"*100,"cyan"))
    print(colored(f"{SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(f"Price={fmt(info['price'])}  RF={fmt(info['filter'])}  L={info['long']} S={info['short']}  |  spread_bps={fmt(spread_bps,2)}")
    print(f"RSI={fmt(ind['rsi'])}  ADX={fmt(ind['adx'])}  (Δ={fmt(ind['adx']-ind['adx_prev'],2)})  ATR={fmt(ind['atr'])}  EMA9={fmt(ind['ema9'])}  EMA20={fmt(ind['ema20'])}")
    if state["open"]:
        lamp = "🟩 LONG" if state["side"]=="long" else "🟥 SHORT"
        rr = (info["price"]-state["entry"])/state["entry"]*100*(1 if state["side"]=="long" else -1)
        print(f"{lamp} entry={fmt(state['entry'])} qty={fmt(state['qty'],4)} pnl={fmt(state['pnl'])} rr={fmt(rr,2)}% bars={state['bars']}")
        print(f"trail={fmt(state['trail'])}  tp1_done={state['tp1_done']}  breakeven={fmt(state['breakeven'])}  highRR={fmt(state['highest_profit_pct'],2)}%  deferral={state['deferral_left']}")
    else:
        print("FLAT — waiting RF closed signal")
    eff = (bal or 0.0) + compound_pnl
    print(f"Balance={fmt(bal,2)}  CompPnL={fmt(compound_pnl)}  EffectiveEq={fmt(eff,2)} USDT")
    print(colored("—"*100,"cyan"))

# ========= KEEPALIVE / API =========
app = Flask(__name__)
import logging as flask_logging
flask_logging.getLogger("werkzeug").setLevel(flask_logging.ERROR)

@app.route("/")
def home():
    mode = 'LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ RF Bot — {SYMBOL} {INTERVAL} — {mode} — RF Entry + EMA9 Touch Profit Engine"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC,
        "price": price_now(), "position": state, "compound_pnl": compound_pnl,
        "touch_engine": {
            "ema9": True, "ema20": True,
            "rsi_thr": EMA_RSI_THR, "adx_min": EMA_ADX_MIN, "adx_delta": EMA_ADX_DELTA,
            "deferral_bars": DEFERRAL_BARS, "atr_mult_thrust": ATR_MULT_THRUST
        }
    })

def keepalive_loop():
    url = (SELF_URL or "").strip().rstrip("/")
    if not url: return
    import requests
    s = requests.Session(); s.headers.update({"User-Agent":"rf-bot/keepalive"})
    while True:
        try: s.get(url, timeout=6)
        except Exception: pass
        time.sleep(max(KEEPALIVE_SECONDS,15))

# ========= BOOT =========
if __name__ == "__main__":
    print(colored("🚀 starting RF bot (no scalping) — EMA9 Touch integrated", "green"))
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
