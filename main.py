# -*- coding: utf-8 -*-
"""
RF Futures Bot — Trend-Only Pro (BingX Perp, CCXT) — RF-LIVE EDITION
- Entry: Range Filter (TV-like) على الشمعة الحيّة (live bar)
- Post-entry: Supertrend + ADX/RSI/ATR + EMA9/20 + فوبوناتشي مبسّط
- Dynamic TP ladder + ATR trail + Impulse/Wick harvesting
- سلوك قمم/قيعان: تأكيد وانتظار شمعة تالية عند الشك + دفاع ذكي
- Trap/Stop-hunt guard + FVG/OB مبسّط لقراءة السيولة
- Strict close + Final-Chunk rule + Spread guard
- Flask /metrics & /health + bot.log rotation
"""

import os, time, math, random, signal, sys, traceback, logging
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

# =================== ENV ===================
API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)

SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.getenv("PORT", 5000))

# =================== SETTINGS ===================
SYMBOL     = os.getenv("BOT_SYMBOL", "DOGE/USDT:USDT")
INTERVAL   = os.getenv("BOT_INTERVAL", "15m")

# ريسك/رافعة (زي ما اتفقنا 60% × 10x)
RISK_ALLOC = float(os.getenv("RISK_ALLOC", "0.60"))
LEVERAGE   = int(os.getenv("LEVERAGE", "10"))
POSITION_MODE = os.getenv("BINGX_POSITION_MODE","oneway").lower()  # "oneway" | "hedge"

# Range Filter
RF_SOURCE = "close"
RF_PERIOD = int(os.getenv("RF_PERIOD","20"))
RF_MULT   = float(os.getenv("RF_MULT","3.5"))
RF_LIVE   = True                  # ← دخول على الشمعة الحيّة

# مؤشرات أساسية
RSI_LEN, ADX_LEN, ATR_LEN = 14, 14, 14

# Spread guard
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS","6.0"))

# إدارة أرباح أساسية
TP0_PROFIT_PCT   = 0.2
TP0_CLOSE_FRAC   = 0.10
TP0_MAX_USDT     = 1.5

TP1_BASE_PCT     = 0.40     # بيتعدل ديناميكياً
TP1_CLOSE_FRAC   = 0.50
BREAKEVEN_AFTER  = 0.30
TRAIL_ACTIVATE   = 1.20
ATR_MULT_TRAIL   = 1.6

# سلوك ترند
TREND_TARGETS     = [0.5, 1.0, 1.8]
TREND_CLOSE_FRACS = [0.30, 0.30, 0.20]
MIN_TREND_HOLD_ADX = 25

# حصاد اندفاع/ذيول
IMPULSE_X_ATR = 1.8
WICK_PCT_HARVEST = 45.0  # % من مدى الشمعة

# تأكيد قمم/قيعان (انتظار شمعة تالية عند الشك)
EXTREME_LEFT_RIGHT = 2  # fractal(2,2)
EXTREME_RSI_WEAK = (45,55)   # تشبع غير حاسم → انتظر
EXTREME_ADX_MIN  = 18
EXTREME_WAIT_ONE_BAR = True

# Trap / Stop-hunt
TRAP_WICK_MIN = 60.0
TRAP_BODY_MAX = 25.0
TRAP_VOL_SPIKE = 1.30
TRAP_HOLD_BARS = 3

# FVG/OB مبسّط
FVG_MAX_ATR = 2.0

# Final chunk rule — يغلق الصفقة لو الباقي ≤ هذا العدد (في عملة العقد)
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY","50"))

# Pacing
BASE_SLEEP = 5
NEAR_CLOSE_SLEEP = 1

# Logging ===================
def setup_file_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h,"baseFilename","").endswith("bot.log") for h in root.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        root.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready","cyan"))
setup_file_logging()

# Exchange ===================
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY, "secret": API_SECRET,
        "enableRateLimit": True, "timeout": 20000,
        "options": {"defaultType":"swap"}
    })

ex = make_exchange()
MARKET, AMT_PREC, LOT_STEP, LOT_MIN = {}, 0, None, None
def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        prec = (MARKET.get("precision",{}) or {}).get("amount", 0) or 0
        AMT_PREC = int(prec) if isinstance(prec,int) else 0
        LOT_STEP = ((MARKET.get("limits",{}) or {}).get("amount",{}) or {}).get("step")
        LOT_MIN  = ((MARKET.get("limits",{}) or {}).get("amount",{}) or {}).get("min") or 0.0
        print(colored(f"🔢 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}","cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_markets: {e}","yellow"))
load_market_specs()

def ensure_leverage_mode():
    try:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
        except Exception as e:
            print(colored(f"set_leverage warn: {e}","yellow"))
        print(colored(f"position mode: {POSITION_MODE}","cyan"))
    except Exception as e:
        print(colored(f"ensure_leverage: {e}","yellow"))
ensure_leverage_mode()

# Helpers ===================
def safe_amt(q: float) -> float:
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and float(LOT_STEP)>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN) * step
        if AMT_PREC and AMT_PREC>0:
            d = d.quantize(Decimal(1).scaleb(-AMT_PREC), rounding=ROUND_DOWN)
        if LOT_MIN and float(LOT_MIN)>0 and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except (InvalidOperation, ValueError, TypeError):
        return max(0.0, float(q))

def fmt(x, d=6):
    try: return f"{float(x):.{d}f}"
    except Exception: return "N/A"

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

def _interval_seconds(iv: str) -> int:
    iv=(iv or "").lower()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 900

def fetch_ohlcv(limit=600):
    rows = ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"})
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

# State ===================
STATE_FILE = "bot_state.json"
state = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "trail": None, "breakeven": None,
    "bars": 0, "pnl": 0.0,
    "tp1_done": False, "_tp0_done": False,
    "profit_targets_achieved": 0,
    "highest_pct": 0.0,
    "extreme_wait": False, "extreme_set_at": None,
}
compound_pnl = 0.0
_last_bar_ts = None
def save_state():
    try:
        import json, os as _os
        tmp=STATE_FILE+".tmp"
        with open(tmp,"w",encoding="utf-8") as f:
            json.dump({"state":state,"compound_pnl":compound_pnl}, f, ensure_ascii=False)
        os.replace(tmp, STATE_FILE)
    except Exception as e:
        logging.error(f"save_state: {e}")
def load_state():
    global compound_pnl
    try:
        import json, os as _os
        if _os.path.exists(STATE_FILE):
            data=json.load(open(STATE_FILE,"r",encoding="utf-8"))
            state.update(data.get("state",{}))
            compound_pnl = data.get("compound_pnl",0.0)
            print(colored("✅ state restored","green"))
    except Exception as e:
        logging.error(f"load_state: {e}")
load_state()

# Indicators ===================
def wilder_ema(s: pd.Series, n:int): return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df)<max(ATR_LEN,RSI_LEN,ADX_LEN)+3:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"adx":0.0,"atr":0.0,"st":None,"st_dir":0,
                "ema9":df["close"].iloc[-1] if len(df) else None,
                "ema20":df["close"].iloc[-1] if len(df) else None}
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)

    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN).fillna(method="bfill")

    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))

    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    plus_di=100*(wilder_ema(plus_dm, ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(wilder_ema(minus_dm, ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=wilder_ema(dx, ADX_LEN)

    # Supertrend مبسط
    st_val, st_dir = None, 0
    try:
        st_period=10; st_mult=3.0
        hl2=(h+l)/2.0
        atr_st=wilder_ema(tr, st_period)
        upper=hl2+st_mult*atr_st; lower=hl2-st_mult*atr_st
        st=[float('nan')]; d=[0]
        for i in range(1,len(df)):
            prev=st[-1]; prevd=d[-1]; cur=float(c.iloc[i]); ub=float(upper.iloc[i]); lb=float(lower.iloc[i])
            if math.isnan(prev):
                st.append(lb if cur>lb else ub); d.append(1 if cur>lb else -1); continue
            if prevd==1:
                ub=min(ub, prev); stn = lb if cur>ub else ub; dn = 1 if cur>ub else -1
            else:
                lb=max(lb, prev); stn = ub if cur<lb else lb; dn = -1 if cur<lb else 1
            st.append(stn); d.append(dn)
        st_val=float(pd.Series(st[1:], index=df.index[1:]).iloc[-1])
        st_dir=int(pd.Series(d[1:], index=df.index[1:]).iloc[-1])
    except Exception:
        pass

    ema9  = df["close"].ewm(span=9, adjust=False).mean().iloc[-1]
    ema20 = df["close"].ewm(span=20, adjust=False).mean().iloc[-1]

    return {"rsi":float(rsi.iloc[-1]), "plus_di":float(plus_di.iloc[-1]),
            "minus_di":float(minus_di.iloc[-1]), "adx":float(adx.iloc[-1]),
            "atr":float(atr.iloc[-1]), "st":st_val, "st_dir":st_dir,
            "ema9":float(ema9), "ema20":float(ema20)}

# Range Filter (TV-like) ===============
def _ema(s: pd.Series, n:int): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng = _ema((src - src.shift(1)).abs(), n); wper = (n*2)-1
    return _ema(avrng, wper) * qty
def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rf.append(cur)
    filt=pd.Series(rf, index=src.index, dtype="float64")
    return filt + rsize, filt - rsize, filt

def rf_live_signal(df: pd.DataFrame):
    """
    نولد إشارة RF على الشمعة الحية:
    - long لو السعر الحي فوق الفلتر + اتجاه الفلتر لأعلى
    - short لو السعر الحي تحت الفلتر + اتجاه الفلتر لأسفل
    - مع هيستريسيز بسيط لمنع التذبذب.
    """
    if len(df) < RF_PERIOD+3:
        return {"time": int(df["time"].iloc[-1]) if len(df) else int(time.time()*1000),
                "price": float(df["close"].iloc[-1]) if len(df) else None,
                "filter": None, "long": False, "short": False, "fdir": 0}
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    dfilt = filt.diff()
    fdir = (dfilt.apply(lambda x: 1 if x>0 else (-1 if x<0 else 0))).fillna(method="ffill").iloc[-1]
    price = float(df["close"].iloc[-1])
    f     = float(filt.iloc[-1])
    # هيستريسيز 8 bps
    bps = abs((price - f)/max(f,1e-12))*10000.0
    long_sig  = price > f and fdir==1 and bps >= 8.0
    short_sig = price < f and fdir==-1 and bps >= 8.0
    return {"time": int(df["time"].iloc[-1]), "price": price, "filter": f,
            "long": bool(long_sig), "short": bool(short_sig), "fdir": int(fdir)}

# Extremes (tops/bottoms) ===============
def _find_fractals(df: pd.DataFrame, left=2, right=2):
    if len(df) < left+right+3: return None, None
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    ph=[None]*len(df); pl=[None]*len(df)
    for i in range(left, len(df)-right):
        if all(h[i] >= h[j] for j in range(i-left, i+right+1)): ph[i]=h[i]
        if all(l[i] <= l[j] for j in range(i-left, i+right+1)): pl[i]=l[i]
    return ph, pl

def extreme_guard_wait_next(df_closed: pd.DataFrame, ind: dict, side: str)->bool:
    """
    عند الشك إننا قريبين من قمة/قاع + زخم ضعيف → نطلب انتظار شمعة تالية ونمنع الإغلاق الكامل.
    """
    if len(df_closed)<EXTREME_LEFT_RIGHT*2+5: return False
    ph, pl = _find_fractals(df_closed, EXTREME_LEFT_RIGHT, EXTREME_LEFT_RIGHT)
    last_ph = max([i for i,v in enumerate(ph) if v is not None], default=None)
    last_pl = max([i for i,v in enumerate(pl) if v is not None], default=None)
    if last_ph is None or last_pl is None: return False
    c = float(df_closed["close"].iloc[-1])
    top = float(ph[last_ph]); bot = float(pl[last_pl])
    near_top = abs((c-top)/top)*10000.0 <= 10.0
    near_bot = abs((c-bot)/bot)*10000.0 <= 10.0
    adx = float(ind.get("adx") or 0.0)
    rsi = float(ind.get("rsi") or 50.0)
    weak = (EXTREME_RSI_WEAK[0] <= rsi <= EXTREME_RSI_WEAK[1]) or (adx < EXTREME_ADX_MIN)
    if side=="long" and near_top and weak: return True
    if side=="short" and near_bot and weak: return True
    return False

# Simple FVG / Trap ======================
def detect_trap(df: pd.DataFrame):
    if len(df)<3: return None
    o,h,l,c = [float(df[x].iloc[-1]) for x in ["open","high","low","close"]]
    rng=max(h-l,1e-12); body=abs(c-o)
    up_w=(h-max(o,c))/rng*100.0; lo_w=(min(o,c)-l)/rng*100.0; body_pct=body/rng*100.0
    vol=float(df["volume"].iloc[-1])
    vma=df["volume"].iloc[-21:-1].astype(float).mean() if len(df)>=21 else 0.0
    volspike=(vma>0 and vol/vma>=TRAP_VOL_SPIKE)
    if up_w>=TRAP_WICK_MIN and body_pct<=TRAP_BODY_MAX and volspike: return {"trap":"bear"}
    if lo_w>=TRAP_WICK_MIN and body_pct<=TRAP_BODY_MAX and volspike: return {"trap":"bull"}
    return None

# Orders ===================
def _open_params(side: str):
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if side=="buy" else "SHORT", "reduceOnly": False}
    return {"positionSide":"BOTH","reduceOnly":False}

def _close_params():
    if POSITION_MODE=="hedge":
        return {"positionSide": "LONG" if state.get("side")=="long" else "SHORT", "reduceOnly": True}
    return {"positionSide":"BOTH","reduceOnly":True}

def compute_size(balance, price):
    eq = (balance or 0.0) + (compound_pnl or 0.0)
    capital = eq * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(price or 1e-9, 1e-9))
    return safe_amt(raw)

def open_market(side, qty, price):
    if qty <= 0: 
        print(colored("⛔ qty<=0 skip open","red")); 
        return
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception: pass
            ex.create_order(SYMBOL,"market",side,qty,None,_open_params(side))
        except Exception as e:
            print(colored(f"❌ open: {e}","red")); return
    state.update({
        "open": True, "side": "long" if side=="buy" else "short",
        "entry": price, "qty": qty, "pnl": 0.0, "bars": 0,
        "trail": None, "breakeven": None,
        "tp1_done": False, "_tp0_done": False,
        "profit_targets_achieved": 0, "highest_pct": 0.0,
        "extreme_wait": False, "extreme_set_at": None,
    })
    logging.info(f"OPEN {side} qty={qty} @ {price}")
    print(colored(f"✅ OPEN {side.upper()} qty={fmt(qty,4)} @ {fmt(price)}","green" if side=="buy" else "red"))
    save_state()

def _read_position():
    try:
        poss = ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym = (p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0] not in sym: continue
            qty = abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty<=0: return 0.0, None, None
            entry=float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side_raw = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side = "long" if ("long" in side_raw or float(p.get("cost",0))>0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position: {e}")
    return 0.0, None, None

def close_market_strict(reason="STRICT"):
    global compound_pnl
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty <= 0:
        if state["open"]: reset_after_close(reason)
        return
    side_to_close = "sell" if exch_side=="long" else "buy"
    qty_to_close = safe_amt(exch_qty)
    try:
        if MODE_LIVE:
            ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,_close_params())
        px = price_now() or state.get("entry")
        side = exch_side or state.get("side")
        pnl = ((px - exch_entry) * (1 if side=="long" else -1)) * qty_to_close
        compound_pnl += pnl
        print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
    except Exception as e:
        print(colored(f"❌ strict close: {e}","red"))
    reset_after_close(reason)

def close_partial(frac, reason="PARTIAL"):
    if not state["open"] or state["qty"]<=0: return
    frac = max(0.0, min(1.0, float(frac)))
    qty_close = safe_amt(state["qty"] * frac)
    # احترام الحدّ الأدنى للكمية
    min_unit = max(LOT_MIN or 0.0, 0.0)
    if qty_close <= 0 or qty_close < min_unit:
        print(colored(f"⏸ skip partial (amount={fmt(qty_close,4)} < min_unit={fmt(min_unit,1)})","yellow"))
        return
    side = "sell" if state["side"]=="long" else "buy"
    px = price_now() or state["entry"]
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_close_params())
        except Exception as e: 
            print(colored(f"❌ partial: {e}","red")); 
            return
    pnl = (px - state["entry"])*(1 if state["side"]=="long" else -1)*qty_close
    state["qty"] = max(0.0, state["qty"] - qty_close)
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(state['qty'],4)}","magenta"))
    # Final chunk rule
    if state["qty"] > 0 and state["qty"] <= max(FINAL_CHUNK_QTY, LOT_MIN or 0.0):
        print(colored("🔒 Final chunk ≤ threshold → strict close","cyan"))
        close_market_strict("FINAL_CHUNK_RULE")
    save_state()

def reset_after_close(reason="CLOSE"):
    state.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "trail": None, "breakeven": None, "bars": 0, "pnl": 0.0,
        "tp1_done": False, "_tp0_done": False, "profit_targets_achieved": 0,
        "highest_pct": 0.0, "extreme_wait": False, "extreme_set_at": None,
    })
    save_state()
    logging.info(f"RESET {reason}")

# Post-entry management ===================
def tp0_quick(ind):
    if state["_tp0_done"] or not state["open"]: return
    px = ind.get("price") or price_now() or state["entry"]
    if not px or not state["entry"]: return
    rr = (px - state["entry"])/state["entry"]*100*(1 if state["side"]=="long" else -1)
    if rr >= TP0_PROFIT_PCT:
        # ما نتجاوزش 1.5 USDT تقريبًا
        usdt_val = px * state["qty"]
        frac = min(TP0_CLOSE_FRAC, (TP0_MAX_USDT / usdt_val) if usdt_val>0 else TP0_CLOSE_FRAC)
        close_partial(frac, f"TP0 {rr:.2f}%")
        state["_tp0_done"]=True

def indicator_consensus(ind, info, side):
    score=0.0
    if (side=="long" and ind.get("st_dir")==1) or (side=="short" and ind.get("st_dir")==-1): score+=1
    if (side=="long" and ind["plus_di"]>ind["minus_di"]) or (side=="short" and ind["minus_di"]>ind["plus_di"]): score+=1
    if (side=="long" and ind["rsi"]>=55) or (side=="short" and ind["rsi"]<=45): score+=1
    if ind["adx"]>=28: score+=1
    if info.get("filter") is not None and info.get("price") is not None:
        if (side=="long" and info["price"]>info["filter"]) or (side=="short" and info["price"]<info["filter"]): score+=0.5
    return float(score)

def build_tp_ladder(ind, info, side):
    px = info.get("price") or state.get("entry")
    atr = float(ind.get("atr") or 0.0)
    if not px or atr<=0:
        return TREND_TARGETS[:], TREND_CLOSE_FRACS[:]
    atr_pct = (atr/max(px,1e-9))*100.0
    c = indicator_consensus(ind, info, side)
    if c >= 4.0: mults=[1.8,3.2,5.0]
    elif c >= 3.0: mults=[1.6,2.8,4.5]
    else: mults=[1.2,2.4,4.0]
    tps=[round(m*atr_pct,2) for m in mults]
    fracs=[0.25,0.30,0.45]
    return tps, fracs

def impulse_wick_harvest(df, ind):
    if not state["open"] or state["qty"]<=0: return
    o,h,l,c = [float(df[x].iloc[-1]) for x in ["open","high","low","close"]]
    rng=max(h-l,1e-12); body=abs(c-o)
    atr=float(ind.get("atr") or 0.0)
    if atr>0 and body >= IMPULSE_X_ATR*atr:
        close_partial(0.5, f"Impulse x{body/atr:.2f} ATR")
    up_w=(h-max(o,c))/rng*100.0; lo_w=(min(o,c)-l)/rng*100.0
    if state["side"]=="long" and up_w>=WICK_PCT_HARVEST: close_partial(0.25, f"Upper wick {up_w:.0f}%")
    if state["side"]=="short" and lo_w>=WICK_PCT_HARVEST: close_partial(0.25, f"Lower wick {lo_w:.0f}%")

def ema_defense(ind, info):
    if not state["open"] or state["qty"]<=0: return
    side=state["side"]; px = info.get("price") or price_now() or state["entry"]
    e9, e20 = ind.get("ema9"), ind.get("ema20")
    if px is None or e9 is None or e20 is None: return
    weak_touch  = (px < e9)  if side=="long" else (px > e9)
    strong_break= (px < e20) if side=="long" else (px > e20)
    if weak_touch:  close_partial(0.20 if not state["tp1_done"] else 0.25, "EMA9 touch")
    if strong_break:
        close_partial(0.30 if not state["tp1_done"] else 0.40, "EMA20 break")
        state["breakeven"] = state.get("breakeven") or state["entry"]

def apply_trail(ind, info):
    if not state["open"] or state["qty"]<=0: return
    px = info.get("price") or price_now() or state["entry"]
    e  = state["entry"]; side=state["side"]
    rr = (px - e)/e*100.0*(1 if side=="long" else -1)
    atr = float(ind.get("atr") or 0.0)
    # TP1 وتثبيت التعادل
    # بيتعدل بالـ ADX
    if not state["tp1_done"]:
        tp1 = TP1_BASE_PCT * (2.2 if ind.get("adx",0)>=35 else 1.8 if ind.get("adx",0)>=28 else 1.0)
        if rr >= tp1:
            close_partial(TP1_CLOSE_FRAC, f"TP1@{tp1:.2f}%")
            state["tp1_done"]=True
            if rr >= BREAKEVEN_AFTER: state["breakeven"]=e
    # تفعيل تريل ATR
    activate = TRAIL_ACTIVATE * (0.7 if ind.get("adx",0)>=35 else 0.8 if ind.get("adx",0)>=28 else 1.0)
    if rr >= activate and atr>0:
        gap = atr * ATR_MULT_TRAIL
        if side=="long":
            new = px - gap
            state["trail"] = max(state["trail"] or new, new)
            if state["breakeven"] is not None:
                state["trail"] = max(state["trail"], state["breakeven"])
            if px < state["trail"]: close_market_strict(f"TRAIL_ATR({ATR_MULT_TRAIL}x)")
        else:
            new = px + gap
            state["trail"] = min(state["trail"] or new, new)
            if state["breakeven"] is not None:
                state["trail"] = min(state["trail"], state["breakeven"])
            if px > state["trail"]: close_market_strict(f"TRAIL_ATR({ATR_MULT_TRAIL}x)")

def dynamic_targets_take(ind, info):
    """سلم TP ديناميكي آمن (بدون أخطاء index)."""
    if not state["open"] or state["qty"]<=0: return
    px = info.get("price") or price_now() or state["entry"]
    e  = state["entry"]; side=state["side"]
    rr = (px - e)/e*100.0*(1 if side=="long" else -1)
    tps, fracs = build_tp_ladder(ind, info, side)
    k = int(state.get("profit_targets_achieved", 0))
    # احرص إننا ما نتجاوزش الطول
    for idx,(tp,fr) in enumerate(list(zip(tps, fracs))):
        if k==idx and rr >= tp:
            close_partial(fr, f"TP_dyn@{tp:.2f}%")
            state["profit_targets_achieved"]=k+1

def wait_candle_logic(df_closed, ind, info):
    """تفعيل وضع انتظار شمعة واحدة لو قريبين من قمة/قاع بضعف زخم."""
    if not state["open"]: return
    if state["extreme_wait"]:
        # انتهت شمعة جديدة؟ ارفع العلم
        state["extreme_wait"]=False
        state["extreme_set_at"]=None
        print(colored("⏱️ EXTREME wait finished — resume normal","cyan"))
        return
    if EXTREME_WAIT_ONE_BAR and extreme_guard_wait_next(df_closed, ind, state["side"]):
        state["extreme_wait"]=True
        state["extreme_set_at"]=int(info["time"])
        print(colored("⏳ POSSIBLE extreme — wait next candle (no full close)","yellow"))

# Loop utils ===================
def time_to_close(df):
    if len(df)==0: return 10
    tf=_interval_seconds(INTERVAL)
    start=int(df["time"].iloc[-1]); now=int(time.time()*1000)
    nxt = start+tf*1000
    while nxt<=now: nxt += tf*1000
    return int(max(0,nxt-now)/1000)

def update_bar_counter(df):
    global _last_bar_ts
    if len(df)==0: return False
    ts=int(df["time"].iloc[-1])
    if _last_bar_ts is None:
        _last_bar_ts=ts; return False
    if ts != _last_bar_ts:
        _last_bar_ts=ts
        if state["open"]: state["bars"] += 1
        return True
    return False

# HUD ===================
def snapshot(bal, info, ind, spread_bps, reason=None):
    print(colored("─"*100,"cyan"))
    print(colored(f"{SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(f"💲 Price {fmt(info.get('price'))} | RF={fmt(info.get('filter'))} | spread_bps={fmt(spread_bps,2)} | close_in~{time_to_close(fetch_ohlcv(2))}s")
    print(f"RSI={fmt(ind.get('rsi'),2)}  +DI={fmt(ind.get('plus_di'),2)}  -DI={fmt(ind.get('minus_di'),2)}  ADX={fmt(ind.get('adx'),2)}  ATR={fmt(ind.get('atr'))}  ST_dir={ind.get('st_dir')}")
    print(f"Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x")
    if state["open"]:
        lam='🟩 LONG' if state['side']=='long' else '🟥 SHORT'
        print(f"{lam} Entry={fmt(state['entry'])} Qty={fmt(state['qty'],4)} Bars={state['bars']} Trail={fmt(state['trail'])} BE={fmt(state['breakeven'])}")
        print(f"TP1={state['tp1_done']}  Highest%={fmt(state['highest_pct'],2)}  ExtremeWait={state['extreme_wait']}")
    else:
        print("⚪ FLAT")
    if reason: print(colored(f"ℹ️ {reason}","yellow"))
    print(colored("─"*100,"cyan"))

# Main loop ===================
def trade_loop():
    loop=0
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            new_bar = update_bar_counter(df)
            ind = compute_indicators(df)
            info_live = rf_live_signal(df)    # ← RF على الشمعة الحيّة
            spread_bps = orderbook_spread_bps()

            # تحديث PnL
            if state["open"] and px:
                state["pnl"] = (px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]

            # Trap guard info
            trap = detect_trap(df)

            # ENTRY (flat) — guard السبريد
            sig = "buy" if info_live["long"] else ("sell" if info_live["short"] else None)
            if not state["open"] and sig and (spread_bps is None or spread_bps<=MAX_SPREAD_BPS):
                qty = compute_size(bal, px or info_live["price"])
                if qty>0:
                    open_market(sig, qty, px or info_live["price"])
                else:
                    snapshot(bal, info_live, ind, spread_bps, "qty<=0")
                    time.sleep(BASE_SLEEP); continue

            # إدارة بعد الدخول
            if state["open"]:
                # Quick cash صغير
                tp0_quick({"price": px or info_live["price"]})

                # حصاد اندفاع/ذيول
                impulse_wick_harvest(df, {**ind, "price": px or info_live["price"]})

                # EMA دفاع
                ema_defense(ind, {"price": px or info_live["price"]})

                # أهداف ديناميكية
                dynamic_targets_take(ind, {"price": px or info_live["price"], "filter": info_live.get("filter")})

                # Trail ATR + BE
                apply_trail(ind, {"price": px or info_live["price"]})

                # منطق انتظار شمعة عند القمم/القيعان
                df_closed = df.iloc[:-1] if len(df)>=2 else df.copy()
                wait_candle_logic(df_closed, ind, info_live)

                # أعلى ربح
                e = state["entry"]; side=state["side"]
                if px and e:
                    rr = (px-e)/e*100*(1 if side=="long" else -1)
                    state["highest_pct"]=max(state["highest_pct"], rr)

                # إغلاق صارم لو التريل أو الـ BE ضرب
                if state["trail"] is not None and px is not None:
                    if (state["side"]=="long" and px < state["trail"]) or (state["side"]=="short" and px > state["trail"]):
                        close_market_strict("TRAIL_HIT")

                if state["breakeven"] is not None and px is not None:
                    if (state["side"]=="long" and px < state["breakeven"]) or (state["side"]=="short" and px > state["breakeven"]):
                        close_market_strict("BREAKEVEN_HIT")

                # Trap → جزئي دفاعي وتعادل
                if trap:
                    close_partial(0.15, f"Trap {trap['trap']}")
                    state["breakeven"]=state.get("breakeven") or state["entry"]

            # HUD
            snapshot(bal, info_live, ind, spread_bps)

            # pacing
            left = time_to_close(df)
            time.sleep(NEAR_CLOSE_SLEEP if left<=10 else BASE_SLEEP)

        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}","red"))
            logging.error(f"loop error: {e}")
            time.sleep(2)

# Web ===================
app = Flask(__name__)
@app.route("/")
def home():
    return f"✅ RF-LIVE Trend Bot • {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC,
        "price": price_now(), "position": state, "compound_pnl": compound_pnl,
        "rf_live": True, "max_spread_bps": MAX_SPREAD_BPS,
        "final_chunk_qty": FINAL_CHUNK_QTY
    })

@app.route("/health")
def health():
    return jsonify({"ok": True, "open": state["open"], "qty": state["qty"], "side": state["side"],
                    "strict_close_ready": True, "timestamp": datetime.utcnow().isoformat()}), 200

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (no SELF_URL)","yellow")); 
        return
    import requests
    sess=requests.Session()
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

def graceful_exit(signum, frame):
    print(colored("🛑 signal → saving state & exit","red"))
    save_state(); sys.exit(0)

signal.signal(signal.SIGTERM, graceful_exit)
signal.signal(signal.SIGINT, graceful_exit)

if __name__ == "__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}","yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  RF_LIVE=True","yellow"))
    from threading import Thread
    Thread(target=trade_loop, daemon=True).start()
    Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
