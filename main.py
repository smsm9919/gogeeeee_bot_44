# -*- coding: utf-8 -*-
"""
DOGE Bot Z — Trend-Only (BingX Perp, CCXT) with EMA9-sensitive Profit Harvest
- دخول بالـ Range Filter على شمعة مُغلقة فقط.
- بعد الدخول: جني أرباح ديناميكي (ATR% + إجماع RSI/ADX/DI/ST + حساسية EMA9/EMA20).
- بدون Scalping. تركيز كامل على الركوب الذكي للترند ثم الحصاد.
"""

import os, time, math, threading, json, random, signal, sys, logging, traceback
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pandas as pd
import ccxt
from flask import Flask, jsonify

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV / SETTINGS ===================
API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)

SYMBOL      = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL    = os.getenv("INTERVAL", "15m")
LEVERAGE    = int(os.getenv("LEVERAGE", "10"))
RISK_ALLOC  = float(os.getenv("RISK_ALLOC", "0.60"))      # 60% من الرصيد × الرافعة
BINGX_POSITION_MODE = os.getenv("BINGX_POSITION_MODE", "oneway")

# Range Filter params
RF_SOURCE  = "close"
RF_PERIOD  = 20
RF_MULT    = 3.5
USE_TV_BAR = True  # نحترم الشمعة المغلقة

# Indicators
RSI_LEN, ADX_LEN, ATR_LEN = 14, 14, 14

# Profit & trail (Trend-only)
TP1_BASE_PCT         = 0.45     # حد أدنى TP1 كنسبة مئوية
TP1_CLOSE_FRAC       = 0.35
BREAKEVEN_AFTER      = 0.30     # حجز الدخول بعد أول حصاد
TRAIL_ACTIVATE_PCT   = 0.70     # تفعيل التريل بعد هذا الربح (قابِل للتكيّف)
ATR_TRAIL_MULT       = 1.6

# EMA حساس:
EMA_TOUCH_LOCK_FRAC  = 0.25     # إغلاق جزئي عند لمس/الكسر ضد الاتجاه على EMA9
EMA_STRONG_LOCK_FRAC = 0.40     # إغلاق أكبر عند كسر EMA20
EMA_SLOPE_BARS       = 5        # لاشتقاق ميل EMA9
EMA_SLOPE_MIN        = 0.0      # الميل <= 0 يعتبر ضعف للاتجاه

# Supertrend (للتأكيد فقط)
ST_PERIOD = 10
ST_MULT   = 3.0

# Spread & sleep
SPREAD_GUARD_BPS = 6
BASE_SLEEP       = 5
NEAR_CLOSE_SLEEP = 1

# Logging HUD
LOG_STYLE = os.getenv("LOG_STYLE", "full")  # full | lite

# State persistence
STATE_FILE = "bot_state.json"

# Web keepalive (اختياري)
SELF_URL          = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
KEEPALIVE_SECONDS = int(os.getenv("KEEPALIVE_SECONDS", "50"))
PORT              = int(os.getenv("PORT", "5000"))

# =================== INIT LOGGING ===================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ file logging ready", "cyan"))
setup_file_logging()

print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • SYMBOL={SYMBOL} • {INTERVAL}", "yellow"))

# =================== EXCHANGE ===================
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY, "secret": API_SECRET,
        "enableRateLimit": True, "timeout": 20000,
        "options": {"defaultType": "swap"}
    })
ex = make_exchange()

MARKET = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN  = None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int(MARKET.get("precision", {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits", {}).get("amount", {}) or {}).get("step", None)
        LOT_MIN  = (MARKET.get("limits", {}).get("amount", {}) or {}).get("min", None)
        print(colored(f"📊 Market specs: prec={AMT_PREC} step={LOT_STEP} min={LOT_MIN}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}", "yellow"))
load_market_specs()

def ensure_leverage():
    try:
        ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
        print(colored(f"✅ leverage set {LEVERAGE}x", "green"))
    except Exception as e:
        print(colored(f"⚠️ set_leverage warn: {e}", "yellow"))
ensure_leverage()

# =================== HELPERS ===================
from decimal import Decimal, ROUND_DOWN, InvalidOperation

def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP, (int, float)) and LOT_STEP > 0:
            step = Decimal(str(LOT_STEP))
            d = (d / step).to_integral_value(rounding=ROUND_DOWN) * step
        prec = int(AMT_PREC) if AMT_PREC and AMT_PREC >= 0 else 0
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and float(d) < LOT_MIN:
            return 0.0
        return float(d)
    except Exception:
        try: return max(0.0, float(q))
        except: return 0.0

def safe_qty(q): 
    q = _round_amt(q)
    if q <= 0: print(colored(f"⚠️ qty invalid after normalize: {q}", "yellow"))
    return q

def fmt(x, d=6):
    try:
        if x is None: return "—"
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)

def _interval_seconds(iv: str) -> int:
    iv = iv.lower()
    if iv.endswith("m"): return int(iv[:-1])*60
    if iv.endswith("h"): return int(iv[:-1])*3600
    if iv.endswith("d"): return int(iv[:-1])*86400
    return 900

def time_to_candle_close(df: pd.DataFrame) -> int:
    tf = _interval_seconds(INTERVAL)
    if len(df)==0: return tf
    cur_start_ms = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    next_close = cur_start_ms + tf*1000
    while next_close <= now_ms:
        next_close += tf*1000
    return max(0, (next_close - now_ms)//1000)

def with_retry(fn, attempts=3, base_wait=0.4):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i==attempts-1: raise
            time.sleep(base_wait*(2**i)+random.random()*0.2)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception:
        return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception:
        return None

def orderbook_spread_bps():
    try:
        ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not bid or not ask: return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

def compute_size(balance, price):
    eff = (balance or 0.0)
    capital = eff * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

# =================== INDICATORS ===================
def wilder_ema(s: pd.Series, n: int): return s.ewm(alpha=1/n, adjust=False).mean()
def ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    i = len(df)-1
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)

    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    # RSI
    delta = c.diff()
    up = delta.clip(lower=0.0); dn = (-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0, 1e-12)
    rsi = 100 - (100 / (1+rs))

    # DMI/ADX
    up_move = h.diff(); down_move = l.shift(1) - l
    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di  = 100 * (wilder_ema(plus_dm, ADX_LEN) / atr.replace(0,1e-12))
    minus_di = 100 * (wilder_ema(minus_dm, ADX_LEN) / atr.replace(0,1e-12))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,1e-12)).fillna(0.0)
    adx = wilder_ema(dx, ADX_LEN)

    # EMA9/EMA20
    ema9  = ema(c, 9)
    ema20 = ema(c, 20)
    # ميل EMA9 (كنسبة من السعر) على آخر EMA_SLOPE_BARS
    slope = 0.0
    if len(c) > EMA_SLOPE_BARS+1:
        e_old = float(ema9.iloc[-EMA_SLOPE_BARS-1])
        e_new = float(ema9.iloc[-1])
        base  = max(abs(e_old), 1e-9)
        slope = (e_new - e_old) / base

    # Supertrend مبسّط
    try:
        hl2 = (h + l) / 2.0
        atr_st = wilder_ema(tr, ST_PERIOD)
        upperband = hl2 + ST_MULT * atr_st
        lowerband = hl2 - ST_MULT * atr_st
        st = [float('nan')]; dirv = [0]
        for k in range(1, len(df)):
            prev_st = st[-1]; prev_dir = dirv[-1]
            cur = float(c.iloc[k]); ub = float(upperband.iloc[k]); lb = float(lowerband.iloc[k])
            if math.isnan(prev_st):
                st.append(lb if cur > lb else ub)
                dirv.append(1 if cur > lb else -1); continue
            if prev_dir == 1:
                ub = min(ub, prev_st); st_val = lb if cur > ub else ub; dir_now = 1 if cur > ub else -1
            else:
                lb = max(lb, prev_st); st_val = ub if cur < lb else lb; dir_now = -1 if cur < lb else 1
            st.append(st_val); dirv.append(dir_now)
        st_series  = pd.Series(st[1:], index=df.index[1:]).reindex(df.index, method='pad')
        dir_series = pd.Series(dirv[1:], index=df.index[1:]).reindex(df.index, method='pad').fillna(0)
        st_val = float(st_series.iloc[-1]); st_dir = int(dir_series.iloc[-1])
    except Exception:
        st_val, st_dir = None, 0

    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]), "minus_di": float(minus_di.iloc[i]),
        "dx": float(dx.iloc[i]), "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i]),
        "ema9": float(ema9.iloc[i]), "ema20": float(ema20.iloc[i]), "ema9_slope": float(slope),
        "st": st_val, "st_dir": st_dir
    }

# ============ RANGE FILTER (closed-bar) ============
def _ema(s: pd.Series, n:int): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n); wper = (n*2)-1
    return _ema(avrng, wper) * qty
def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf = [float(src.iloc[0])]
    for i in range(1, len(src)):
        prev = rf[-1]; x = float(src.iloc[i]); r = float(rsize.iloc[i]); cur = prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rf.append(cur)
    filt = pd.Series(rf, index=src.index, dtype="float64")
    return filt + rsize, filt - rsize, filt

def compute_tv_signals(df: pd.DataFrame):
    if len(df) < RF_PERIOD + 3:
        i = -1
        price = float(df["close"].iloc[i]) if len(df) else None
        return {"time": int(df["time"].iloc[i]) if len(df) else int(time.time()*1000),
                "price": price or 0.0, "long": False, "short": False,
                "filter": price or 0.0, "hi": price or 0.0, "lo": price or 0.0}
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    dfilt = filt - filt.shift(1)
    fdir = pd.Series(0.0, index=filt.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0.0)
    src_gt_f=(src>filt); src_lt_f=(src<filt); src_gt_p=(src>src.shift(1)); src_lt_p=(src<src.shift(1))
    upward=(fdir==1).astype(int); downward=(fdir==-1).astype(int)
    longCond=(src_gt_f&((src_gt_p)|(src_lt_p))&(upward>0))
    shortCond=(src_lt_f&((src_lt_p)|(src_gt_p))&(downward>0))
    CondIni=pd.Series(0,index=src.index)
    for i in range(1,len(src)):
        if bool(longCond.iloc[i]): CondIni.iloc[i]=1
        elif bool(shortCond.iloc[i]): CondIni.iloc[i]=-1
        else: CondIni.iloc[i]=CondIni.iloc[i-1]
    longSignal=longCond&(CondIni.shift(1)==-1)
    shortSignal=shortCond&(CondIni.shift(1)==1)
    i = len(df)-1
    return {
        "time": int(df["time"].iloc[i]),
        "price": float(df["close"].iloc[i]),
        "long": bool(longSignal.iloc[i]),
        "short": bool(shortSignal.iloc[i]),
        "filter": float(filt.iloc[i]),
        "hi": float(hi.iloc[i]), "lo": float(lo.iloc[i])
    }

# =================== STATE ===================
state = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None,
    "tp1_done": False, "breakeven": None,
    "highest_profit_pct": 0.0, "last_action": None, "action_reason": None
}
compound_pnl = 0.0

def save_state():
    try:
        data = {"state": state, "compound_pnl": compound_pnl, "ts": time.time()}
        with open(STATE_FILE, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False)
    except Exception as e:
        logging.error(f"save_state err: {e}")

def load_state():
    global compound_pnl
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            state.update(data.get("state", {}))
            compound_pnl = data.get("compound_pnl", 0.0)
            print(colored("✅ state restored", "green"))
    except Exception as e:
        logging.error(f"load_state err: {e}")
load_state()

# =================== ORDERS ===================
def _params_open(side):
    if BINGX_POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if side=="buy" else "SHORT", "reduceOnly": False}
    return {"positionSide": "BOTH", "reduceOnly": False}
def _params_close():
    if BINGX_POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if state.get("side")=="long" else "SHORT", "reduceOnly": True}
    return {"positionSide": "BOTH", "reduceOnly": True}

def open_market(side, qty, price):
    if qty<=0: return
    if MODE_LIVE:
        try:
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
        except Exception as e:
            print(colored(f"❌ open: {e}", "red")); return
    state.update({
        "open": True, "side": "long" if side=="buy" else "short",
        "entry": price, "qty": qty, "pnl": 0.0, "bars": 0, "trail": None,
        "tp1_done": False, "breakeven": None, "highest_profit_pct": 0.0,
        "last_action": "OPEN", "action_reason": "RF closed signal"
    })
    print(colored(f"✅ OPEN {side.upper()} qty={fmt(qty,4)} @ {fmt(price)}", "green" if side=="buy" else "red"))
    save_state()

def close_partial(frac, reason):
    global compound_pnl
    if not state["open"]: return
    frac = min(max(frac,0.0), 1.0)
    qty_close = safe_qty(state["qty"] * frac)
    if qty_close <= 0: return
    px = price_now() or state["entry"]
    side = "sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try:
            ex.create_order(SYMBOL, "market", side, qty_close, None, _params_close())
        except Exception as e:
            print(colored(f"❌ partial: {e}", "red")); return
    pnl=(px-state["entry"])*qty_close*(1 if state["side"]=="long" else -1)
    compound_pnl += pnl
    state["qty"] -= qty_close
    state["last_action"] = "SCALE_OUT"; state["action_reason"] = reason
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(state['qty'],4)}", "magenta"))
    if state["qty"]<=0: reset_after_close("fully_exited")
    save_state()

def close_market(reason):
    global compound_pnl
    if not state["open"]: return
    px  = price_now() or state["entry"]
    qty = state["qty"]; side = "sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try:
            ex.create_order(SYMBOL, "market", side, qty, None, _params_close())
        except Exception as e:
            print(colored(f"❌ close: {e}", "red")); return
    pnl=(px-state["entry"])*qty*(1 if state["side"]=="long" else -1)
    compound_pnl += pnl
    print(colored(f"🔚 CLOSE {state['side']} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}", "magenta"))
    reset_after_close(reason)

def reset_after_close(reason):
    state.update({
        "open": False, "side": None, "entry": None, "qty": 0.0, "pnl": 0.0,
        "bars": 0, "trail": None, "tp1_done": False, "breakeven": None,
        "highest_profit_pct": 0.0, "last_action": "CLOSE", "action_reason": reason
    })
    save_state()

# =================== PROFIT ENGINE (EMA9-sensitive) ===================
def indicator_consensus(ind, side, rf_filter, price):
    score = 0.0
    if (side=="long" and ind["plus_di"]>ind["minus_di"]) or (side=="short" and ind["minus_di"]>ind["plus_di"]): score += 1
    if (side=="long" and ind["rsi"]>=55) or (side=="short" and ind["rsi"]<=45): score += 1
    if ind["adx"]>=28: score += 1
    if price is not None and rf_filter is not None:
        if (side=="long" and price>rf_filter) or (side=="short" and price<rf_filter): score += 0.5
    # EMA9/20 ترتيب وميل
    if side=="long":
        if ind["ema9"]>=ind["ema20"]: score += 0.5
        if ind["ema9_slope"]>EMA_SLOPE_MIN: score += 0.5
    else:
        if ind["ema9"]<=ind["ema20"]: score += 0.5
        if ind["ema9_slope"]< -EMA_SLOPE_MIN: score += 0.5
    return float(score)  # من 0 إلى ~4

def build_dynamic_targets(ind, side, price, rf_filter):
    atr = ind["atr"]; atr_pct = (atr / max(price,1e-9)) * 100.0
    base = max(TP1_BASE_PCT, round(atr_pct*1.2, 2))  # TP1 يعتمد على ATR%
    score = indicator_consensus(ind, side, rf_filter, price)
    # سلّم أهداف بسيط 3 درجات
    if score >= 3.5:
        mults = [1.8, 3.2, 5.0]
        fracs = [0.30, 0.30, 0.40]
    elif score >= 2.5:
        mults = [1.6, 2.8, 4.2]
        fracs = [0.35, 0.30, 0.35]
    else:
        mults = [1.3, 2.2, 3.2]
        fracs = [0.40, 0.30, 0.30]
    targets = [round(base*m, 2) for m in mults]
    return targets, fracs, score, atr_pct

def ema_touch_signals(ind, side, close_price):
    """يرجع (weak_touch, strong_break, ema_bias_ok)
       weak_touch: لمس/كسر EMA9 ضد الاتجاه → حصاد جزئي
       strong_break: كسر EMA20 ضد الاتجاه → حصاد أكبر/تشديد التريل
       ema_bias_ok: ترتيب EMAs لصالحنا
    """
    ema9, ema20 = ind["ema9"], ind["ema20"]
    ema_bias_ok = (ema9>=ema20) if side=="long" else (ema9<=ema20)
    weak_touch = (close_price < ema9) if side=="long" else (close_price > ema9)
    strong_break = (close_price < ema20) if side=="long" else (close_price > ema20)
    return weak_touch, strong_break, ema_bias_ok

def manage_profit(info, ind):
    """منطق إدارة الصفقة بعد الدخول"""
    if not state["open"] or state["qty"]<=0: return
    side = state["side"]; price = info["price"]; entry = state["entry"]
    rr = (price - entry)/entry * 100.0 * (1 if side=="long" else -1)
    targets, fracs, score, atr_pct = build_dynamic_targets(ind, side, info.get("filter"), price)

    # تحديث أعلى ربح
    if rr > state["highest_profit_pct"]: state["highest_profit_pct"] = rr

    # TP1 ديناميكي + حجز دخول
    tp1 = targets[0]
    if (not state["tp1_done"]) and rr >= tp1:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{tp1:.2f}% (dyn)")
        state["tp1_done"] = True
        if rr >= BREAKEVEN_AFTER: state["breakeven"] = entry

    # تفعيل تريل بالـ ATR بعد مستوى معين
    if rr >= max(TRAIL_ACTIVATE_PCT, tp1*0.9) and ind["atr"]>0:
        gap = ind["atr"] * ATR_TRAIL_MULT
        if side=="long":
            new_trail = price - gap
            state["trail"] = max(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None: state["trail"] = max(state["trail"], state["breakeven"])
            if price < state["trail"]: return close_market(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")
        else:
            new_trail = price + gap
            state["trail"] = min(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None: state["trail"] = min(state["trail"], state["breakeven"])
            if price > state["trail"]: return close_market(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")

    # حساسية EMA9/EMA20
    weak_touch, strong_break, ema_bias_ok = ema_touch_signals(ind, side, price)

    # لو الاتجاه قوي (score عالي و ema_bias_ok) نتحمل لمس EMA9 بدون خروج كامل، فقط حصاد خفيف
    if weak_touch and rr>0:
        frac = EMA_TOUCH_LOCK_FRAC if state["tp1_done"] else min(0.20, EMA_TOUCH_LOCK_FRAC)
        close_partial(frac, "EMA9 touch against position")
        # tighten trail قليلًا
        if ind["atr"]>0:
            gap = ind["atr"] * max(1.2, ATR_TRAIL_MULT-0.2)
            if side=="long":
                state["trail"] = max(state["trail"] or (price-gap), price-gap)
            else:
                state["trail"] = min(state["trail"] or (price+gap), price+gap)

    # كسر EMA20 ضد الاتجاه = ضعف واضح
    if strong_break:
        if not state["tp1_done"]:
            close_partial(0.30, "EMA20 break pre-TP1")
            state["tp1_done"] = True
        else:
            close_partial(EMA_STRONG_LOCK_FRAC, "EMA20 break")
        # لو ميل EMA9 انقلب ضدنا + ADX ضعيف → إغلاق نهائي
        if ((side=="long" and ind["ema9_slope"]<=0) or (side=="short" and ind["ema9_slope"]>=0)) and ind["adx"]<20:
            return close_market("EMA20 break + slope flip (weak ADX)")

    # الوصول لباقي الأهداف الديناميكية
    achieved = 1 if state["tp1_done"] else 0
    for k in range(achieved, len(targets)):
        if rr >= targets[k]:
            close_partial(fracs[k], f"TP{k+1}@{targets[k]:.2f}% (dyn)")
            state["tp1_done"] = True

    # ختام ترند: لو ADX يضعف جدًا وتبادل DI ضدنا مع كسر EMA9 وST قريب → اخرج
    di_flip = (side=="long" and ind["minus_di"] > ind["plus_di"]) or (side=="short" and ind["plus_di"] > ind["minus_di"])
    st_flip = (ind["st"] is not None) and ((side=="long" and ind["st_dir"]==-1) or (side=="short" and ind["st_dir"]==1))
    broke_ema9 = weak_touch
    if di_flip and broke_ema9 and ind["adx"]<20:
        return close_market("Trend end: DI flip + EMA9 break + weak ADX")
    if st_flip and broke_ema9:
        close_partial(0.25, "ST flip + EMA9 break")

# =================== SNAPSHOT / HUD ===================
def _ema_now(series, n):
    return float(pd.Series(series).astype(float).ewm(span=n, adjust=False).mean().iloc[-1])

def print_indicator_block(df, ind, info, spread_bps):
    price = info.get("price")
    rf    = info.get("filter")
    print(f"Price={fmt(price)} | RF={fmt(rf)} | L={info['long']} S={info['short']} | spread_bps={fmt(spread_bps,2)}")
    print(f"RSI({RSI_LEN})={ind['rsi']:.2f} | +DI={ind['plus_di']:.2f}  -DI={ind['minus_di']:.2f} | "
          f"DX={ind['dx']:.2f}  ADX({ADX_LEN})={ind['adx']:.2f} | ATR={ind['atr']:.6f}")
    if ind.get("st") is not None:
        print(f"Supertrend={ind['st']:.6f} (dir={'UP' if ind['st_dir']==1 else 'DOWN' if ind['st_dir']==-1 else 'FLAT'})")
    print(f"EMA9={ind['ema9']:.6f} | EMA20={ind['ema20']:.6f} | EMA9_slope={ind['ema9_slope']:.4f}")

def snapshot(bal, info, ind, spread_bps, df):
    print(colored("─"*100,"cyan"))
    print(colored(f"{SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    if LOG_STYLE.lower()=="full":
        print("📈 INDICATORS")
        print_indicator_block(df, ind, info, spread_bps)
    left = time_to_candle_close(df)
    print(f"⏱️ Candle closes in ~ {left}s")
    print("🧭 POSITION")
    print(f"   Balance={fmt(bal,2)} USDT  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x")
    if state["open"]:
        lamp = '🟩 LONG' if state['side']=='long' else '🟥 SHORT'
        rr = (info["price"]-state["entry"])/state["entry"]*100.0*(1 if state["side"]=="long" else -1)
        print(f"   {lamp}  Entry={fmt(state['entry'])} Qty={fmt(state['qty'],4)}  RR={fmt(rr,2)}%  "
              f"Trail={fmt(state['trail'])}  TP1_done={state['tp1_done']}")
    else:
        print("   FLAT — waiting RF closed signal")
    eff = (bal or 0.0) + compound_pnl
    print(f"💰 CompPnL={fmt(compound_pnl)}  EffectiveEq={fmt(eff,2)} USDT")
    print(colored("─"*100,"cyan"))

# =================== LOOP ===================
_last_bar_ts = None
def update_bar_counter(df):
    global _last_bar_ts
    if len(df)==0: return False
    ts = int(df["time"].iloc[-1])
    if _last_bar_ts is None:
        _last_bar_ts = ts; return False
    if ts != _last_bar_ts:
        _last_bar_ts = ts
        if state["open"]: state["bars"] += 1
        return True
    return False

def trade_loop():
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            update_bar_counter(df)

            info = compute_tv_signals(df)
            ind  = compute_indicators(df)
            spread_bps = orderbook_spread_bps()

            if state["open"] and px:
                state["pnl"] = (px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]

            # إدارة الصفقة إن وجدت
            if state["open"]:
                manage_profit(info, ind)

            # دخول جديد
            else:
                reason_block = None
                if spread_bps is not None and spread_bps > SPREAD_GUARD_BPS:
                    reason_block = f"spread too high {spread_bps:.2f}bps"
                sig = "buy" if info["long"] else ("sell" if info["short"] else None)
                if sig and not reason_block:
                    qty = compute_size(bal, info["price"])
                    if qty > 0: open_market(sig, qty, info["price"])

            snapshot(bal, info, ind, spread_bps, df)
            time.sleep(NEAR_CLOSE_SLEEP if time_to_candle_close(df)<=10 else BASE_SLEEP)

        except Exception as e:
            print(colored(f"❌ loop: {e}\n{traceback.format_exc()}", "red"))
            time.sleep(BASE_SLEEP)

# =================== HTTP / KEEPALIVE ===================
app = Flask(__name__)
@app.route("/")
def home():
    mode = 'LIVE' if MODE_LIVE else 'PAPER'
    return f"OK — {SYMBOL} {INTERVAL} — {mode} — TrendOnly + EMA9-sensitive harvest"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "price": price_now(), "position": state, "compound_pnl": compound_pnl,
        "settings": {
            "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC,
            "tp1_base_pct": TP1_BASE_PCT, "trail_activate_pct": TRAIL_ACTIVATE_PCT,
            "atr_trail_mult": ATR_TRAIL_MULT, "ema_touch_frac": EMA_TOUCH_LOCK_FRAC
        }
    })

@app.route("/health")
def health():
    return jsonify({"ok": True, "open": state["open"], "side": state["side"], "qty": state["qty"],
                    "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat()}), 200

def keepalive_loop():
    url = (SELF_URL or "").strip().rstrip("/")
    if not url: return
    import requests
    sess = requests.Session()
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(max(KEEPALIVE_SECONDS, 15))

# =================== BOOT ===================
def _graceful_exit(signum, frame):
    print(colored(f"🛑 signal {signum} → exiting", "red"))
    save_state(); sys.exit(0)
signal.signal(signal.SIGTERM, _graceful_exit)
signal.signal(signal.SIGINT,  _graceful_exit)

if __name__ == "__main__":
    print(colored("🚀 Starting DOGE Bot Z (Trend-Only, EMA9-Sensitive Harvest)", "green"))
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
