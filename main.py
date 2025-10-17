# -*- coding: utf-8 -*-
"""
RF Futures Bot — Trend-Only Pro (BingX Perp, CCXT) — FINAL
• Entry: Closed-candle Range Filter signals only
• Post-entry: Supertrend + ADX/RSI/ATR dynamic harvesting
• Dynamic TP ladder (ATR% + consensus) + Ratchet Lock
• Impulse & Wick harvesting + Thrust Lock (Chandelier exit)
• Emergency protection layer + Breakout Engine (with exits)
• Strict exchange close + Patience Guard (no early full exits)
• Flask metrics/health + keepalive + rotated logging

NOTE: No scalping at all — trend-only management.
"""

import os, time, math, threading, requests, traceback, random, signal, sys, logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV ===================
API_KEY = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE = bool(API_KEY and API_SECRET)

SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.getenv("PORT", 5000))

# =================== SETTINGS ===================
SYMBOL   = "DOGE/USDT:USDT"
INTERVAL = "15m"
LEVERAGE = 10
RISK_ALLOC = 0.60
BINGX_POSITION_MODE = "oneway"  # "hedge" أو "oneway"

# Range Filter (TradingView-like)
RF_SOURCE = "close"
RF_PERIOD = 20
RF_MULT   = 3.5
USE_TV_BAR = True  # pacing فقط

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

# Strategy core (Trend-Only)
STRATEGY = "smart"
TREND_ONLY = True
USE_SMART_EXIT = True

# Targets & trailing (base)
TP1_PCT = 0.40          # % ربح للهدف الأول (سيتعدل ديناميكيًا)
TP1_CLOSE_FRAC = 0.50   # نسبة الإغلاق عند TP1
BREAKEVEN_AFTER = 0.30  # تثبيت التعادل بعد ربح >= 0.30%
TRAIL_ACTIVATE = 0.60   # تفعيل التريل عند تجاوز هذا الربح
ATR_MULT_TRAIL = 1.6    # معامل ATR للتريل الافتراضي

# لا سكالبنج — كل شيء ترند فقط
TREND_TARGETS       = [0.50, 1.00, 1.80]
TREND_CLOSE_FRACS   = [0.30, 0.30, 0.20]
MIN_TREND_HOLD_ADX  = 25
END_TREND_ADX_DROP  = 5.0
END_TREND_RSI_NEUTRAL = (45, 55)
DI_FLIP_BUFFER = 1.0

# Impulse/Wick/Ratchet
IMPULSE_HARVEST_THRESHOLD = 1.2  # body >= 1.2×ATR
LONG_WICK_HARVEST_THRESHOLD = 0.60  # 60% من مدى الشمعة
RATCHET_RETRACE_THRESHOLD = 0.40     # إقفال جزئي عند ارتداد 40% من أعلى ربح

# Breakout Engine
BREAKOUT_ATR_SPIKE = 1.8
BREAKOUT_ADX_THRESHOLD = 25
BREAKOUT_LOOKBACK_BARS = 20
BREAKOUT_CALM_THRESHOLD = 1.1
BREAKOUT_CONFIRM_BARS = 1
BREAKOUT_HARD_ATR_SPIKE = 1.8
BREAKOUT_SOFT_ATR_SPIKE = 1.4
BREAKOUT_VOLUME_SPIKE = 1.3
BREAKOUT_VOLUME_MED   = 1.1

# Emergency Layer
EMERGENCY_PROTECTION_ENABLED = True
EMERGENCY_ADX_MIN = 40
EMERGENCY_ATR_SPIKE_RATIO = 1.6
EMERGENCY_RSI_PUMP  = 72
EMERGENCY_RSI_CRASH = 28
EMERGENCY_POLICY = "tp_then_close"  # "close_always" أو "tp_then_close"
EMERGENCY_HARVEST_FRAC = 0.60
EMERGENCY_FULL_CLOSE_PROFIT = 1.0
EMERGENCY_TRAIL_ATR_MULT = 1.2

# TP0 quick cash (صغير وآمن)
TP0_PROFIT_PCT = 0.2
TP0_CLOSE_FRAC = 0.10
TP0_MAX_USDT   = 1.0

# Thrust Lock (Chandelier)
THRUST_ATR_BARS = 3
THRUST_VOLUME_FACTOR = 1.3
CHANDELIER_ATR_MULT  = 3.0
CHANDELIER_LOOKBACK  = 20

# Adaptive trail (Smart Alpha)
TRAIL_MULT_STRONG_ALPHA   = 2.4
TRAIL_MULT_CAUTIOUS_ALPHA = 2.0

# Pacing
ADAPTIVE_PACING   = True
BASE_SLEEP        = 10
NEAR_CLOSE_SLEEP  = 1
JUST_CLOSED_WINDOW= 8

# Strict close
STRICT_EXCHANGE_CLOSE = True
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0
MIN_RESIDUAL_TO_FORCE= 1.0

# Exit protection layer
MIN_HOLD_BARS        = 3
MIN_HOLD_SECONDS     = 120
TRAIL_ONLY_AFTER_TP1 = True
RF_HYSTERESIS_BPS    = 8
NO_FULL_CLOSE_BEFORE_TP1 = True
WICK_MAX_BEFORE_TP1  = 0.25

# Patience Guard (reverse consensus)
ENABLE_PATIENCE       = True
PATIENCE_MIN_BARS     = 2
PATIENCE_NEED_CONSENSUS = 2
REV_ADX_DROP          = 3.0
REV_RSI_LEVEL         = 50
REV_RF_CROSS          = True

# Dynamic TP & ST proximity
HOLD_SCORE            = 3.0
STRONG_HOLD_SCORE     = 4.0
DEFER_TP_UNTIL_IDX    = 2
ST_NEAR_ATR = 0.6
ST_FAR_ATR  = 1.5

# Fearless Hold
HOLD_TCI = 65
HOLD_STRONG_TCI = 75
CHOP_MAX_FOR_HOLD = 0.35
HOLD_MAX_PARTIAL_FRAC = 0.25
HOLD_RATCHET_LOCK_PCT_ON_HOLD = 0.80

# Keepalive
KEEPALIVE_SECONDS = 50

# =================== LOGGING ===================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ file logging with rotation enabled", "cyan"))

setup_file_logging()

# =================== EXCHANGE ===================
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
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
        amt_prec = MARKET.get("precision", {}).get("amount", 0)
        AMT_PREC = int(amt_prec) if isinstance(amt_prec, int) else int(amt_prec or 0)
        LOT_STEP = (MARKET.get("limits", {}).get("amount", {}) or {}).get("step", None)
        LOT_MIN  = (MARKET.get("limits", {}).get("amount", {}) or {}).get("min",  None)
        print(colored(f"📊 Market specs: precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}", "yellow"))

def ensure_leverage_and_mode():
    try:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            print(colored(f"✅ leverage set: {LEVERAGE}x", "green"))
        except Exception as e:
            print(colored(f"⚠️ set_leverage warn: {e}", "yellow"))
        print(colored(f"ℹ️ position mode target: {BINGX_POSITION_MODE}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_leverage_and_mode: {e}", "yellow"))

try:
    load_market_specs()
    ensure_leverage_and_mode()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}", "yellow"))

# =================== STATE & PERSIST ===================
STATE_FILE = "bot_state.json"
_consec_err = 0
last_loop_ts = time.time()
last_open_fingerprint = None

state = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None,
    "breakeven": None, "scale_ins": 0, "scale_outs": 0,
    "last_action": None, "action_reason": None,
    "highest_profit_pct": 0.0,
    "trade_mode": None,
    "profit_targets_achieved": 0,
    "entry_time": None,
    "tp1_done": False, "tp2_done": False,
    "_tp0_done": False,
    "tci": None, "chop01": None,
    "_hold_trend": False,
    # Breakout
    "breakout_active": False, "breakout_direction": None,
    "breakout_entry_price": None, "breakout_score": 0.0,
    "breakout_votes_detail": {},
    "opened_by_breakout": False,
    # Dynamic TP
    "_tp_ladder": None, "_tp_fracs": None,
    "_consensus_score": None, "_atr_pct": None,
    # Idempotency/wait
    "opened_at": None
}
compound_pnl = 0.0
last_signal_id = None
post_close_cooldown = 0
wait_for_next_signal_side = None
last_close_signal_time = None

_state_lock = threading.Lock()

def save_state():
    try:
        data = {"state": state, "compound_pnl": compound_pnl, "last_signal_id": last_signal_id, "timestamp": time.time()}
        tmp = STATE_FILE + ".tmp"
        with _state_lock:
            import json
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, STATE_FILE)
        logging.info("State saved")
    except Exception as e:
        logging.error(f"save_state error: {e}")

def load_state():
    global state, compound_pnl, last_signal_id
    try:
        import json, os as _os
        if _os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            state.update(data.get("state", {}))
            compound_pnl = data.get("compound_pnl", 0.0)
            last_signal_id = data.get("last_signal_id")
            print(colored("✅ state restored from disk", "green"))
    except Exception as e:
        logging.error(f"load_state error: {e}")

def _graceful_exit(signum, frame):
    print(colored(f"🛑 signal {signum} → saving state & exiting", "red"))
    save_state(); sys.exit(0)

signal.signal(signal.SIGTERM, _graceful_exit)
signal.signal(signal.SIGINT,  _graceful_exit)

# =================== HELPERS ===================
from decimal import Decimal, ROUND_DOWN, InvalidOperation
def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP, (int,float)) and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = int(AMT_PREC) if AMT_PREC and AMT_PREC>0 else 0
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except (InvalidOperation, ValueError, TypeError):
        return max(0.0, float(q))

def safe_qty(q):
    q = _round_amt(q)
    if q <= 0: print(colored(f"⚠️ qty invalid after normalize → {q}", "yellow"))
    return q

def fmt(v, d=6, na="N/A"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception: return na

def with_retry(fn, attempts=3, base_wait=0.4):
    global _consec_err
    for i in range(attempts):
        try:
            r = fn()
            _consec_err = 0
            return r
        except Exception:
            _consec_err += 1
            if i == attempts-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.2)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None

def orderbook_spread_bps():
    try:
        ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

def _interval_seconds(iv: str) -> int:
    iv = (iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame, use_tv_bar: bool) -> int:
    tf = _interval_seconds(INTERVAL)
    if len(df) == 0: return tf
    cur_start_ms = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    next_close_ms = cur_start_ms + tf*1000
    while next_close_ms <= now_ms:
        next_close_ms += tf*1000
    left = max(0, next_close_ms - now_ms)
    return int(left/1000)

def compute_next_sleep(df):
    if not ADAPTIVE_PACING: return BASE_SLEEP
    try:
        left_s = time_to_candle_close(df, USE_TV_BAR)
        tf = _interval_seconds(INTERVAL)
        if left_s <= 10: return NEAR_CLOSE_SLEEP
        if (tf - left_s) <= JUST_CLOSED_WINDOW: return NEAR_CLOSE_SLEEP
        return BASE_SLEEP
    except Exception:
        return BASE_SLEEP

# =================== INDICATORS ===================
def wilder_ema(s: pd.Series, n: int): return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0,"adx_prev":0.0,"st":None,"st_dir":0,"chop":50.0}
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

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

    # Supertrend
    st_val, st_dir = None, 0
    try:
        st_period = 10; st_mult = 3.0
        hl2=(h+l)/2.0
        atr_st = wilder_ema(pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1), st_period)
        upper=hl2+st_mult*atr_st; lower=hl2-st_mult*atr_st
        st=[float('nan')]; dirv=[0]
        for i in range(1, len(df)):
            prev_st=st[-1]; prev_dir=dirv[-1]
            cur=float(c.iloc[i]); ub=float(upper.iloc[i]); lb=float(lower.iloc[i])
            if math.isnan(prev_st):
                st.append(lb if cur>lb else ub); dirv.append(1 if cur>lb else -1); continue
            if prev_dir==1:
                ub=min(ub, prev_st); st_val_now = lb if cur>ub else ub; dir_now = 1 if cur>ub else -1
            else:
                lb=max(lb, prev_st); st_val_now = ub if cur<lb else lb; dir_now = -1 if cur<lb else 1
            st.append(st_val_now); dirv.append(dir_now)
        st_series = pd.Series(st[1:], index=df.index[1:]).reindex(df.index, method='pad')
        dir_series= pd.Series(dirv[1:], index=df.index[1:]).reindex(df.index, method='pad').fillna(0)
        st_val=float(st_series.iloc[-1]); st_dir=int(dir_series.iloc[-1])
    except Exception:
        pass

    # Choppiness-like (scaled 0..100)
    try:
        n=14; tr_sum = tr.rolling(n).sum(); hh=h.rolling(n).max(); ll=l.rolling(n).min()
        chop = 100.0 * ((tr_sum / (hh-ll).replace(0,1e-12)).apply(lambda x: math.log10(x+1e-12)) / math.log10(n))
        chop = chop.fillna(50.0)
        chop_val = float(chop.iloc[-1])
    except Exception:
        chop_val = 50.0

    # --- EMA9 / EMA20 + slope ---
    ema9  = df["close"].astype(float).ewm(span=9, adjust=False).mean()
    ema20 = df["close"].astype(float).ewm(span=20, adjust=False).mean()
    slope = 0.0
    if len(ema9) > 6:
        e_old = float(ema9.iloc[-6]); e_new = float(ema9.iloc[-1])
        base  = max(abs(e_old), 1e-9)
        slope = (e_new - e_old) / base

    i=len(df)-1; prev_i=max(0,i-1)
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i]),
        "adx_prev": float(adx.iloc[prev_i]),
        "st": st_val, "st_dir": st_dir, "chop": chop_val,
        "ema9": float(ema9.iloc[-1]),
        "ema20": float(ema20.iloc[-1]),
        "ema9_slope": float(slope)
    }

# =================== RANGE FILTER ===================
def _ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()
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

def compute_tv_signals(df: pd.DataFrame):
    """Signals from CLOSED candles only — pass df_closed (excludes live bar)."""
    if len(df) < RF_PERIOD + 3:
        i = -1
        price = float(df["close"].iloc[i]) if len(df) else None
        return {"time": int(df["time"].iloc[i]) if len(df) else int(time.time()*1000),
                "price": price or 0.0, "long": False, "short": False,
                "filter": price or 0.0, "hi": price or 0.0, "lo": price or 0.0, "fdir": 0.0}
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    dfilt = filt - filt.shift(1)
    fdir = pd.Series(0.0, index=filt.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0.0)
    upward=(fdir==1).astype(int); downward=(fdir==-1).astype(int)
    src_gt_f=(src>filt); src_lt_f=(src<filt); src_gt_p=(src>src.shift(1)); src_lt_p=(src<src.shift(1))
    longCond=(src_gt_f&((src_gt_p)|(src_lt_p))&(upward>0))
    shortCond=(src_lt_f&((src_lt_p)|(src_gt_p))&(downward>0))
    CondIni=pd.Series(0,index=src.index)
    for i in range(1,len(src)):
        if bool(longCond.iloc[i]): CondIni.iloc[i]=1
        elif bool(shortCond.iloc[i]): CondIni.iloc[i]=-1
        else: CondIni.iloc[i]=CondIni.iloc[i-1]
    longSignal = longCond & (CondIni.shift(1)==-1)
    shortSignal= shortCond & (CondIni.shift(1)==1)
    i=len(df)-1
    return {
        "time": int(df["time"].iloc[i]), "price": float(df["close"].iloc[i]),
        "long": bool(longSignal.iloc[i]), "short": bool(shortSignal.iloc[i]),
        "filter": float(filt.iloc[i]), "hi": float(hi.iloc[i]), "lo": float(lo.iloc[i]),
        "fdir": float(fdir.iloc[i])
    }

# =================== TREND CONVICTION / CHOP ===================
def _clamp01(x):
    try:
        x=float(x); 
        return 0.0 if x<0 else 1.0 if x>1 else x
    except Exception: return 0.0

def _slope(series, k=4):
    s=pd.Series(series).astype(float)
    if len(s)<k+1: return 0.0
    base=abs(s.iloc[-1-k]) or 1e-12
    return (s.iloc[-1]-s.iloc[-1-k]) / base

def compute_tci_and_chop(df: pd.DataFrame, ind: dict, side: str):
    src=df[RF_SOURCE].astype(float)
    _,_,filt=_rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    adx=float(ind.get("adx") or 0.0)
    pdi=float(ind.get("plus_di") or 0.0)
    mdi=float(ind.get("minus_di") or 0.0)
    rsi=float(ind.get("rsi") or 50.0)
    price=float(df["close"].iloc[-1])
    atr=float(ind.get("atr") or 0.0)
    s_adx  = _clamp01((adx - 18.0)/22.0)
    s_di   = _clamp01(((pdi - mdi) if side=="long" else (mdi - pdi))/30.0)
    s_rsi  = _clamp01(((rsi - 50.0) if side=="long" else (50.0 - rsi))/20.0)
    s_slope= _clamp01(_slope(filt, 4) / ((price or 1e-9)*0.004))
    tci = 100.0*(0.40*s_adx + 0.30*s_di + 0.20*s_rsi + 0.10*s_slope)
    # chop01 (0..1): كثافة تقاطعات + ADX ضعيف + ATR% منخفض
    N=8; crosses=0; sgn_prev=None
    for i in range(max(1,len(src)-N), len(src)):
        sgn=1 if src.iloc[i]>=filt.iloc[i] else -1
        if sgn_prev is not None and sgn != sgn_prev: crosses += 1
        sgn_prev=sgn
    atr_pct=(atr / max(price,1e-9))*100.0
    chop = min(1.0, (crosses/max(N-1,1))*0.6 + (1.0 if adx<20 else 0.0)*0.3 + (1.0 if atr_pct<0.7 else 0.0)*0.1)
    return {"tci":float(tci), "chop01":float(chop),
            "hold_mode": (tci>=HOLD_TCI and chop<=CHOP_MAX_FOR_HOLD),
            "strong_hold": (tci>=HOLD_STRONG_TCI and chop<=CHOP_MAX_FOR_HOLD*1.1)}

# =================== DYNAMIC TP LADDER ===================
def _indicator_consensus(info: dict, ind: dict, side: str) -> float:
    score=0.0
    try:
        st_dir=int(ind.get("st_dir") or 0)
        adx=float(ind.get("adx") or 0.0)
        rsi=float(ind.get("rsi") or 50.0)
        pdi=float(ind.get("plus_di") or 0.0)
        mdi=float(ind.get("minus_di") or 0.0)
        rf=info.get("filter"); px=info.get("price")
        if (side=="long" and st_dir==1) or (side=="short" and st_dir==-1): score += 1.0
        if (side=="long" and pdi>mdi) or (side=="short" and mdi>pdi): score += 1.0
        if (side=="long" and rsi>=55) or (side=="short" and rsi<=45): score += 1.0
        if adx>=28: score += 1.0
        elif adx>=20: score += 0.5
        if px is not None and rf is not None:
            if (side=="long" and px>rf) or (side=="short" and px<rf): score += 0.5
        chop01 = state.get("chop01", None)
        if chop01 is not None and chop01 <= CHOP_MAX_FOR_HOLD: score += 0.5
    except Exception:
        pass
    return float(score)

def _build_tp_ladder(info: dict, ind: dict, side: str):
    px = info.get("price") or state.get("entry")
    atr = float(ind.get("atr") or 0.0)
    atr_pct = (atr / max(float(px or 0.0), 1e-9)) * 100.0 if px else 0.5
    score = _indicator_consensus(info, ind, side)
    if score >= STRONG_HOLD_SCORE: mults = [1.8, 3.2, 5.0]
    elif score >= HOLD_SCORE:      mults = [1.6, 2.8, 4.5]
    else:                          mults = [1.2, 2.4, 4.0]
    targets = [round(m*atr_pct, 2) for m in mults]
    close_fracs = [0.25, 0.30, 0.45]
    return targets, close_fracs, atr_pct, score

# =================== ORDERS ===================
def _position_params_for_open(side: str):
    if BINGX_POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if side=="buy" else "SHORT", "reduceOnly": False}
    return {"positionSide": "BOTH", "reduceOnly": False}

def _position_params_for_close():
    if BINGX_POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if state.get("side")=="long" else "SHORT", "reduceOnly": True}
    return {"positionSide": "BOTH", "reduceOnly": True}

def _read_exchange_position():
    try:
        poss = ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym = (p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0] not in sym: continue
            qty = abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty <= 0: return 0.0, None, None
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side_raw = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side = "long" if ("long" in side_raw or float(p.get("cost",0))>0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_exchange_position error: {e}")
    return 0.0, None, None

def compute_size(balance, price):
    effective_balance = (balance or 0.0) + (compound_pnl or 0.0)
    capital = effective_balance * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def open_market(side, qty, price):
    global state
    if qty<=0: 
        print(colored("❌ qty<=0 skip open","red")); 
        return
    params = _position_params_for_open(side)
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception as e: print(colored(f"⚠️ set_leverage: {e}", "yellow"))
            ex.create_order(SYMBOL,"market",side,qty,None,params)
        except Exception as e:
            print(colored(f"❌ open: {e}","red")); logging.error(f"open_market error: {e}"); return
    state.update({
        "open": True, "side": "long" if side=="buy" else "short",
        "entry": price, "qty": qty, "pnl": 0.0, "bars": 0,
        "trail": None, "breakeven": None,
        "scale_ins": 0, "scale_outs": 0,
        "last_action": "OPEN", "action_reason": "Initial position",
        "highest_profit_pct": 0.0,
        "trade_mode": "TREND",  # تثبيت ترند فقط
        "profit_targets_achieved": 0,
        "entry_time": time.time(),
        "tp1_done": False, "tp2_done": False,
        "_tp0_done": False,
        "tci": None, "chop01": None,
        "_hold_trend": False,
        "breakout_active": False, "breakout_direction": None,
        "breakout_entry_price": None, "breakout_score": 0.0,
        "breakout_votes_detail": {}, "opened_by_breakout": False,
        "_tp_ladder": None, "_tp_fracs": None,
        "_consensus_score": None, "_atr_pct": None,
        "opened_at": time.time()
    })
    print(colored(f"✅ OPEN {side.upper()} qty={fmt(qty,4)} @ {fmt(price)}","green" if side=="buy" else "red"))
    logging.info(f"OPEN {side} qty={qty} price={price}")
    save_state()

def close_market_strict(reason):
    global state, compound_pnl, wait_for_next_signal_side, last_close_signal_time
    exch_qty, exch_side, exch_entry = _read_exchange_position()
    if exch_qty <= 0:
        if state.get("open"):
            reset_after_full_close("strict_close_already_zero", state.get("side"))
        return
    side_to_close = "sell" if (exch_side=="long") else "buy"
    qty_to_close  = safe_qty(exch_qty)
    attempts=0; last_error=None
    while attempts < CLOSE_RETRY_ATTEMPTS:
        try:
            if MODE_LIVE:
                params = _position_params_for_close(); params["reduceOnly"]=True
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            time.sleep(CLOSE_VERIFY_WAIT_S)
            left_qty, _, _ = _read_exchange_position()
            if left_qty <= 0:
                px = price_now() or state.get("entry"); entry_px = state.get("entry") or exch_entry or px
                side = state.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
                qty  = exch_qty
                pnl  = (px - entry_px) * qty * (1 if side=="long" else -1)
                compound_pnl += pnl
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                logging.info(f"STRICT_CLOSE {side} pnl={pnl} total={compound_pnl}")
                reset_after_full_close(reason, side)
                return
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            print(colored(f"⚠️ strict close retry {attempts}/{CLOSE_RETRY_ATTEMPTS} — residual={fmt(left_qty,4)}","yellow"))
            if qty_to_close < MIN_RESIDUAL_TO_FORCE: time.sleep(CLOSE_VERIFY_WAIT_S)
        except Exception as e:
            last_error = e; logging.error(f"close_market_strict attempt {attempts+1}: {e}"); attempts += 1; time.sleep(CLOSE_VERIFY_WAIT_S)
    print(colored(f"❌ STRICT CLOSE FAILED after {CLOSE_RETRY_ATTEMPTS} attempts — manual check needed. Last error: {last_error}", "red"))
    logging.critical(f"STRICT CLOSE FAILED — last_error={last_error}")

def close_partial(frac, reason):
    global state, compound_pnl
    if not state["open"]: return
    qty_close = safe_qty(max(0.0, state["qty"] * min(max(frac,0.0),1.0)))
    if qty_close < 1:
        print(colored(f"⚠️ skip partial close (amount={fmt(qty_close,4)} < 1 DOGE)", "yellow"))
        return
    px = price_now() or state["entry"]
    side = "sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_position_params_for_close())
        except Exception as e: print(colored(f"❌ partial close: {e}","red")); logging.error(f"close_partial error: {e}"); return
    pnl=(px-state["entry"])*qty_close*(1 if state["side"]=="long" else -1)
    compound_pnl += pnl
    state["qty"] -= qty_close
    state["scale_outs"] += 1
    state["last_action"]="SCALE_OUT"; state["action_reason"]=reason
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem_qty={fmt(state['qty'],4)}","magenta"))
    logging.info(f"PARTIAL_CLOSE {reason} qty={qty_close} pnl={pnl} remaining={state['qty']}")
    if state["qty"] < 60:
        print(colored(f"⚠️ Remaining qty={fmt(state['qty'],2)} < 60 DOGE → full close", "yellow"))
        close_market_strict("auto_full_close_small_qty"); return
    save_state()

def reset_after_full_close(reason, prev_side=None):
    global state, post_close_cooldown, wait_for_next_signal_side, last_close_signal_time
    print(colored(f"🔚 CLOSE {reason} totalCompounded now={fmt(compound_pnl)}","magenta"))
    logging.info(f"FULL_CLOSE {reason} total_compounded={compound_pnl}")
    if prev_side is None: prev_side = state.get("side")
    state.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None,
        "breakeven": None, "scale_ins": 0, "scale_outs": 0,
        "last_action": "CLOSE", "action_reason": reason,
        "highest_profit_pct": 0.0,
        "trade_mode": None, "profit_targets_achieved": 0,
        "entry_time": None, "tp1_done": False, "tp2_done": False,
        "_tp0_done": False, "tci": None, "chop01": None, "_hold_trend": False,
        "breakout_active": False, "breakout_direction": None, "breakout_entry_price": None,
        "breakout_score": 0.0, "breakout_votes_detail": {}, "opened_by_breakout": False,
        "_tp_ladder": None, "_tp_fracs": None, "_consensus_score": None, "_atr_pct": None,
        "opened_at": None
    })
    if prev_side == "long":  wait_for_next_signal_side = "sell"
    elif prev_side == "short": wait_for_next_signal_side = "buy"
    else: wait_for_next_signal_side = None
    last_close_signal_time = None
    post_close_cooldown = 0
    save_state()

# =================== MANAGEMENT ===================
def detect_candle_pattern(df: pd.DataFrame):
    if len(df)<3: return {"pattern":"NONE","name_ar":"لا شيء","name_en":"NONE","strength":0}
    idx=-1
    o2,h2,l2,c2 = map(float,(df["open"].iloc[idx-2], df["high"].iloc[idx-2], df["low"].iloc[idx-2], df["close"].iloc[idx-2]))
    o1,h1,l1,c1 = map(float,(df["open"].iloc[idx-1], df["high"].iloc[idx-1], df["low"].iloc[idx-1], df["close"].iloc[idx-1]))
    o0,h0,l0,c0 = map(float,(df["open"].iloc[idx], df["high"].iloc[idx], df["low"].iloc[idx], df["close"].iloc[idx]))
    def _candle_stats(o,c,h,l):
        rng=max(h-l,1e-12); body=abs(c-o); upper=h-max(o,c); lower=min(o,c)-l
        return {"range":rng,"body":body,"body_pct":(body/rng)*100,"upper_pct":(upper/rng)*100,"lower_pct":(lower/rng)*100,"bull":c>o,"bear":c<o}
    s2=_candle_stats(o2,c2,h2,l2); s1=_candle_stats(o1,c1,h1,l1); s0=_candle_stats(o0,c0,h0,l0)
    if (s2["bear"] and s2["body_pct"]>=60 and s1["body_pct"]<=25 and l1>l2 and s0["bull"] and s0["body_pct"]>=50 and c0>(o1+c1)/2):
        return {"pattern":"MORNING_STAR","name_ar":"نجمة الصباح","name_en":"Morning Star","strength":4}
    if (s2["bull"] and s2["body_pct"]>=60 and s1["body_pct"]<=25 and h1<h2 and s0["bear"] and s0["body_pct"]>=50 and c0<(o1+c1)/2):
        return {"pattern":"EVENING_STAR","name_ar":"نجمة المساء","name_en":"Evening Star","strength":4}
    if s0["body_pct"]<=30 and s0["lower_pct"]>=60 and s0["upper_pct"]<=10 and s0["bull"]:
        return {"pattern":"HAMMER","name_ar":"المطرقة","name_en":"Hammer","strength":2}
    if s0["body_pct"]<=30 and s0["upper_pct"]>=60 and s0["lower_pct"]<=10 and s0["bear"]:
        return {"pattern":"SHOOTING_STAR","name_ar":"النجمة الهاوية","name_en":"Shooting Star","strength":2}
    if s0["body_pct"]<=10: return {"pattern":"DOJI","name_ar":"دوجي","name_en":"Doji","strength":1}
    if s0["body_pct"]>=85 and s0["upper_pct"]<=7 and s0["lower_pct"]<=7:
        direction="BULL" if s0["bull"] else "BEAR"
        name_ar="المربوزو الصاعد" if s0["bull"] else "المربوزو الهابط"
        name_en=f"Marubozu {direction}"
        return {"pattern":f"MARUBOZU_{direction}","name_ar":name_ar,"name_en":name_en,"strength":3}
    return {"pattern":"NONE","name_ar":"لا شيء","name_en":"NONE","strength":0}

def tp0_quick_cash(ind: dict) -> bool:
    if not state["open"] or state["qty"]<=0 or state.get("_tp0_done", False): return False
    try:
        price = ind.get("price") or price_now() or state["entry"]
        entry = state["entry"]; side = state["side"]
        if not (price and entry): return False
        rr = (price-entry)/entry*100*(1 if side=="long" else -1)
        if rr >= TP0_PROFIT_PCT:
            usdt_value = state["qty"]*price
            close_frac = min(TP0_CLOSE_FRAC, TP0_MAX_USDT/usdt_value) if usdt_value>0 else TP0_CLOSE_FRAC
            if close_frac>0:
                close_partial(close_frac, f"TP0 Quick Cash @ {rr:.2f}%")
                state["_tp0_done"]=True
                return True
    except Exception as e:
        logging.error(f"tp0_quick_cash error: {e}")
    return False

def handle_impulse_and_long_wicks(df: pd.DataFrame, ind: dict):
    if not state["open"] or state["qty"]<=0: return None
    try:
        idx=-1
        o0=float(df["open"].iloc[idx]); h0=float(df["high"].iloc[idx])
        l0=float(df["low"].iloc[idx]);  c0=float(df["close"].iloc[idx])
        current_price = ind.get("price") or c0
        entry = state["entry"]; side = state["side"]
        rr = (current_price - entry) / entry * 100 * (1 if side=="long" else -1)
        candle_range = h0 - l0; body = abs(c0 - o0)
        upper_wick = h0 - max(o0,c0); lower_wick = min(o0,c0) - l0
        body_pct = (body / candle_range) * 100 if candle_range>0 else 0
        upper_wick_pct = (upper_wick / candle_range) * 100 if candle_range>0 else 0
        lower_wick_pct = (lower_wick / candle_range) * 100 if candle_range>0 else 0
        atr = ind.get("atr", 0)
        if atr>0 and body >= IMPULSE_HARVEST_THRESHOLD*atr:
            candle_direction = 1 if c0>o0 else -1
            trade_direction  = 1 if side=="long" else -1
            if candle_direction == trade_direction:
                harvest_frac = 0.50 if body >= 2.0*atr else 0.33
                close_partial(harvest_frac, f"Impulse x{body/atr:.2f} ATR")
                if not state["breakeven"] and rr >= BREAKEVEN_AFTER: state["breakeven"]=entry
                gap = atr * (state.get("_adaptive_trail_mult") or ATR_MULT_TRAIL or 1.0)
                if side=="long":
                    state["trail"] = max(state.get("trail") or (current_price-gap), current_price-gap)
                else:
                    state["trail"] = min(state.get("trail") or (current_price+gap), current_price+gap)
                return "IMPULSE_HARVEST"
        if side=="long" and upper_wick_pct >= LONG_WICK_HARVEST_THRESHOLD*100:
            close_partial(0.25, f"Upper wick {upper_wick_pct:.1f}%"); return "WICK_HARVEST"
        if side=="short" and lower_wick_pct >= LONG_WICK_HARVEST_THRESHOLD*100:
            close_partial(0.25, f"Lower wick {lower_wick_pct:.1f}%"); return "WICK_HARVEST"
    except Exception as e:
        logging.error(f"handle_impulse_and_long_wicks error: {e}")
    return None

def ratchet_protection(ind: dict):
    if not state["open"] or state["qty"]<=0: return None
    current_price = ind.get("price") or price_now() or state["entry"]
    entry = state["entry"]; side = state["side"]
    current_rr = (current_price - entry)/entry*100*(1 if side=="long" else -1)
    if current_rr > state["highest_profit_pct"]: state["highest_profit_pct"]=current_rr
    if (state["highest_profit_pct"]>=20 and current_rr < state["highest_profit_pct"]*(1-RATCHET_RETRACE_THRESHOLD)):
        close_partial(0.5, f"Ratchet {state['highest_profit_pct']:.1f}%→{current_rr:.1f}%")
        state["highest_profit_pct"]=current_rr
        return "RATCHET"
    return None

def get_dynamic_tp_params(adx: float) -> tuple:
    if adx >= 35: return 2.2, 0.7
    elif adx >= 28: return 1.8, 0.8
    else: return 1.0, 1.0

def get_adaptive_trail_multiplier(breakout_score: float) -> float:
    if breakout_score >= 3.0: return TRAIL_MULT_STRONG_ALPHA
    elif breakout_score >= 2.0: return TRAIL_MULT_CAUTIOUS_ALPHA
    else: return ATR_MULT_TRAIL

def thrust_lock(df: pd.DataFrame, ind: dict) -> bool:
    if not state["open"] or state["qty"]<=0 or state.get("thrust_locked", False): return False
    try:
        need = max(THRUST_ATR_BARS + 20, ATR_LEN + 5)
        if len(df) < need: return False
        c=df["close"].astype(float); h=df["high"].astype(float); l=df["low"].astype(float)
        tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
        atr_series = wilder_ema(tr, ATR_LEN).dropna()
        if len(atr_series) < THRUST_ATR_BARS + 1: return False
        tail = atr_series.iloc[-(THRUST_ATR_BARS+1):]
        atr_increasing = all(tail.iloc[i] < tail.iloc[i+1] for i in range(len(tail)-1))
        current_volume = float(df["volume"].iloc[-1]) if len(df) else 0.0
        volume_ma = df["volume"].iloc[-21:-1].astype(float).mean() if len(df)>=21 else 0.0
        volume_spike = (current_volume > volume_ma*THRUST_VOLUME_FACTOR) if (volume_ma>0) else False
        if atr_increasing and volume_spike and state.get("trade_mode")=="TREND":
            state["thrust_locked"]=True; state["breakeven"]=state["entry"]
            atr_now=float(ind.get("atr") or (atr_series.iloc[-1] if len(atr_series) else 0.0))
            if state["side"]=="long":
                lookback_lows = df["low"].iloc[-CHANDELIER_LOOKBACK:].astype(float)
                chandelier_stop = lookback_lows.min() - atr_now*CHANDELIER_ATR_MULT
                state["trail"] = max(state.get("trail") or chandelier_stop, chandelier_stop)
            else:
                lookback_highs = df["high"].iloc[-CHANDELIER_LOOKBACK:].astype(float)
                chandelier_stop = lookback_highs.max() + atr_now*CHANDELIER_ATR_MULT
                state["trail"] = min(state.get("trail") or chandelier_stop, chandelier_stop)
            print(colored("🔒 THRUST LOCK: Activated (Chandelier)", "green"))
            return True
    except Exception as e:
        logging.error(f"thrust_lock error: {e}")
    return False

# =================== EMA-BASED HARVESTING ===================
def ema_touch_harvest(info: dict, ind: dict):
    """Partial harvest on EMA9/EMA20 violations against the position + tighten trail."""
    if not state["open"] or state["qty"] <= 0: return None
    side = state["side"]; px = info.get("price") or price_now() or state["entry"]
    ema9 = ind.get("ema9"); ema20 = ind.get("ema20"); atr = ind.get("atr") or 0.0
    if px is None or ema9 is None or ema20 is None: return None

    # لمس/كسر ضد الاتجاه
    weak_touch  = (px < ema9)  if side=="long" else (px > ema9)
    strong_break= (px < ema20) if side=="long" else (px > ema20)

    acted = None
    if weak_touch:
        frac = 0.25 if state.get("tp1_done") else 0.20
        close_partial(frac, "EMA9 touch against position")
        acted = "EMA9_TOUCH"
    if strong_break:
        frac = 0.40 if state.get("tp1_done") else 0.30
        close_partial(frac, "EMA20 break against position")
        # تشديد التريل لو ATR متاح
        if atr > 0:
            gap = atr * max(state.get("_adaptive_trail_mult") or ATR_MULT_TRAIL, 1.2)
            if side=="long":
                state["trail"] = max(state.get("trail") or (px-gap), px-gap)
            else:
                state["trail"] = min(state.get("trail") or (px+gap), px+gap)
        acted = (acted or "") + "+EMA20_BREAK"
        # لو الميل انقلب وADX ضعيف → إغلاق نهائي
        if ((side=="long" and (ind.get("ema9_slope",0.0) <= 0)) or
            (side=="short" and (ind.get("ema9_slope",0.0) >= 0))) and (ind.get("adx",0.0) < 20):
            close_market_strict("EMA20 break + slope flip (weak ADX)")
    return acted

# =================== EXIT / POST-ENTRY ===================
def trend_end_confirmed(ind: dict, candle_info: dict, info: dict) -> bool:
    adx = float(ind.get("adx") or 0.0)
    adx_prev = float(ind.get("adx_prev") or adx)
    plus_di = float(ind.get("plus_di") or 0.0)
    minus_di = float(ind.get("minus_di") or 0.0)
    rsi = float(ind.get("rsi") or 50.0)
    adx_weak = (adx_prev - adx) >= END_TREND_ADX_DROP or adx < 20
    di_flip = (minus_di - plus_di) > DI_FLIP_BUFFER if state.get("side")=="long" else (plus_di - minus_di) > DI_FLIP_BUFFER
    rsi_neutral = END_TREND_RSI_NEUTRAL[0] <= rsi <= END_TREND_RSI_NEUTRAL[1]
    rf_opposite = (state.get("side")=="long" and info.get("short")) or (state.get("side")=="short" and info.get("long"))
    return adx_weak or di_flip or rsi_neutral or rf_opposite

def trend_profit_taking(ind: dict, info: dict, df_cached: pd.DataFrame):
    if not state["open"] or state["qty"]<=0: return None
    price = ind.get("price") or price_now() or state["entry"]
    entry = state["entry"]; side = state["side"]
    rr = (price - entry)/entry*100*(1 if side=="long" else -1)
    targets = state.get("_tp_ladder", TREND_TARGETS)
    fracs   = state.get("_tp_fracs",  TREND_CLOSE_FRACS)
    k = int(state.get("profit_targets_achieved", 0))
    cscore = float(state.get("_consensus_score", 0.0))
    if k < len(targets) and rr >= targets[k]:
        if cscore >= STRONG_HOLD_SCORE and k < max(DEFER_TP_UNTIL_IDX, len(targets)-1):
            return None  # تأجيل TP للترند القوي
        close_partial(fracs[k], f"TP{k+1}@{targets[k]:.2f}% (dyn)")
        state["profit_targets_achieved"] = k + 1
        return f"TREND_TP{k+1}"
    if float(ind.get("adx") or 0.0) >= MIN_TREND_HOLD_ADX:
        return None
    if state.get("profit_targets_achieved", 0) >= len(targets) or trend_end_confirmed(ind, detect_candle_pattern(df_cached), info):
        close_market_strict("TREND finished — full exit")
        return "TREND_COMPLETE"
    return None

def smart_post_entry_manager(df: pd.DataFrame, ind: dict, info: dict):
    if not state["open"] or state["qty"]<=0: return None
    now_ts = info.get("time") or time.time()
    state.setdefault("opened_at", now_ts)
    hold = compute_tci_and_chop(df, ind, state.get("side"))
    state["_hold_trend"]=bool(hold["hold_mode"])
    state["tci"]=hold["tci"]; state["chop01"]=hold["chop01"]
    if hold["strong_hold"]:
        state["_adaptive_trail_mult"]=max(state.get("_adaptive_trail_mult") or ATR_MULT_TRAIL, TRAIL_MULT_STRONG_ALPHA)
    if state.get("trade_mode") is None: state["trade_mode"]="TREND"; state["profit_targets_achieved"]=0
    # سلّم ديناميكي كل دورة
    tp_list, tp_fracs, atr_pct, cscore = _build_tp_ladder(info, ind, state.get("side"))
    state["_tp_ladder"]=tp_list; state["_tp_fracs"]=tp_fracs
    state["_consensus_score"]=cscore; state["_atr_pct"]=atr_pct
    if TRAIL_ONLY_AFTER_TP1 and not state.get("tp1_done"):
        state["_adaptive_trail_mult"]=0.0
    # Quick cash صغير
    if tp0_quick_cash(ind): return "TP0_QUICK_CASH"
    # Thrust Lock
    if state.get("trade_mode")=="TREND" and not state.get("thrust_locked"): 
        if thrust_lock(df, ind): return "THRUST_LOCK"
    # Harvests
    act = handle_impulse_and_long_wicks(df, ind)
    if act: return act

    # ✅ NEW: EMA-based harvesting - Integrated intelligently
    if not act:
        act = ema_touch_harvest({**info, "price": ind.get("price") or info.get("price")}, ind)
        if act: return act

    act = ratchet_protection(ind)
    if act: return act
    return trend_profit_taking(ind, info, df)

def smart_exit_check(info, ind, df_cached=None, prev_ind_cached=None):
    if not (STRATEGY=="smart" and USE_SMART_EXIT and state["open"]): return None
    now_ts = info.get("time") or time.time()
    opened_at = state.get("opened_at") or now_ts
    elapsed_s = max(0, now_ts - opened_at)
    elapsed_bar = int(state.get("bars", 0))

    def _allow_full_close(reason: str) -> bool:
        if elapsed_bar >= MIN_HOLD_BARS or elapsed_s >= MIN_HOLD_SECONDS: return True
        hard_reasons=("TRAIL","TRAIL_ATR","CHANDELIER","STRICT","FORCE","STOP","EMERGENCY")
        return any(tag in reason for tag in hard_reasons)

    def _safe_full_close(reason):
        if NO_FULL_CLOSE_BEFORE_TP1 and not state.get("tp1_done"):
            if not _allow_full_close(reason):
                logging.info(f"EXIT_BLOCKED (early): {reason}"); return False
        close_market_strict(reason); return True

    # Reverse consensus votes
    side=state.get("side")
    rsi=float(ind.get("rsi") or 50.0)
    adx=float(ind.get("adx") or 0.0); adx_prev=float(ind.get("adx_prev") or adx)
    rf=info.get("filter"); px=info.get("price")
    reverse_votes=0
    if side=="long" and rsi<REV_RSI_LEVEL: reverse_votes+=1
    if side=="short" and rsi>(100-REV_RSI_LEVEL): reverse_votes+=1
    if adx_prev and (adx_prev - adx) >= REV_ADX_DROP: reverse_votes+=1
    if REV_RF_CROSS and px is not None and rf is not None:
        try:
            bps_diff = abs((px - rf) / rf) * 10000
            if (side=="long" and px < rf - (RF_HYSTERESIS_BPS/10000.0)*rf) or \
               (side=="short" and px > rf + (RF_HYSTERESIS_BPS/10000.0)*rf): reverse_votes+=1
        except Exception: pass
    # Supertrend/DI flips
    try:
        st_dir=int(ind.get("st_dir") or 0)
        if (side=="long" and st_dir==-1) or (side=="short" and st_dir==1): reverse_votes+=1
    except Exception: pass
    try:
        pdi=float(ind.get("plus_di") or 0.0); mdi=float(ind.get("minus_di") or 0.0)
        if side=="long" and (mdi - pdi) > DI_FLIP_BUFFER: reverse_votes+=1
        if side=="short" and (pdi - mdi) > DI_FLIP_BUFFER: reverse_votes+=1
    except Exception: pass

    # Set soft block
    need=max(PATIENCE_NEED_CONSENSUS, 3)
    state["_exit_soft_block"] = not (elapsed_bar >= MIN_HOLD_BARS and elapsed_s >= MIN_HOLD_SECONDS and reverse_votes >= need)
    if ENABLE_PATIENCE and state.get("_exit_soft_block"): return None

    # Adaptive trail on breakout/thrust
    if state.get("breakout_active") or state.get("thrust_locked"):
        state["_adaptive_trail_mult"] = get_adaptive_trail_multiplier(state.get("breakout_score", 0.0))

    # Supertrend proximity logic
    try:
        st_val=ind.get("st"); st_dir=int(ind.get("st_dir") or 0); px=info.get("price"); atr=float(ind.get("atr") or 0.0)
        pdi=float(ind.get("plus_di") or 0.0); mdi=float(ind.get("minus_di") or 0.0)
        if state["open"] and st_val is not None and px is not None and atr>0:
            dist=abs(px - st_val); near = dist <= ST_NEAR_ATR*atr; far = dist >= ST_FAR_ATR*atr
            if far and state.get("_consensus_score",0) >= STRONG_HOLD_SCORE:
                state["_exit_soft_block"]=True
            if near:
                di_flip = (state["side"]=="long" and mdi>pdi) or (state["side"]=="short" and pdi>mdi)
                st_flip = (state["side"]=="long" and st_dir==-1) or (state["side"]=="short" and st_dir==1)
                if di_flip or st_flip:
                    if not state.get("tp1_done"):
                        close_partial(min(0.25, HOLD_MAX_PARTIAL_FRAC), "ST proximity warn")
                        state["tp1_done"]=True
                    else:
                        gap = atr * (state.get("_adaptive_trail_mult") or ATR_MULT_TRAIL)
                        if state["side"]=="long":
                            state["trail"] = max(state.get("trail") or (px-gap), px - gap)
                        else:
                            state["trail"] = min(state.get("trail") or (px+gap), px + gap)
    except Exception as e:
        logging.error(f"ST proximity error: {e}")

    # Trail enforcement
    px = info["price"]; e = state["entry"]; side=state["side"]
    if e is None or px is None or side is None or e == 0: return None
    rr = (px - e)/e * 100.0 * (1 if side=="long" else -1)
    atr = ind.get("atr") or 0.0
    adx = ind.get("adx") or 0.0
    rsi = ind.get("rsi") or 50.0
    tp_multiplier, trail_activate_multiplier = get_dynamic_tp_params(adx)
    current_tp1_pct = TP1_PCT * tp_multiplier
    current_trail_activate = TRAIL_ACTIVATE * trail_activate_multiplier
    trail_mult_to_use = state.get("_adaptive_trail_mult") or ATR_MULT_TRAIL

    # TP1 & breakeven
    if (not state.get("tp1_done")) and rr >= current_tp1_pct:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{current_tp1_pct:.2f}%")
        state["tp1_done"]=True
        if rr >= BREAKEVEN_AFTER: state["breakeven"]=e

    # Delay TP when trend strong
    if (side=="long" and adx>=30 and rsi>=55) or (side=="short" and adx>=30 and rsi<=45):
        print(colored("💎 HOLD-TP: strong trend continues, delaying TP", "cyan"))

    # Ratchet lock (tighter on hold)
    ratchet_lock_pct = HOLD_RATCHET_LOCK_PCT_ON_HOLD if state.get("_hold_trend") else 0.60
    if (state["highest_profit_pct"] >= current_trail_activate and
        rr < state["highest_profit_pct"] * ratchet_lock_pct):
        close_partial(0.5, f"Ratchet Lock @ {state['highest_profit_pct']:.2f}%")
        state["highest_profit_pct"] = rr

    # ATR trail
    if rr >= current_trail_activate and atr and (trail_mult_to_use or 0)>0:
        gap = atr * trail_mult_to_use
        if side=="long":
            new_trail = px - gap
            state["trail"] = max(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None: state["trail"] = max(state["trail"], state["breakeven"])
            if px < state["trail"]: _safe_full_close(f"TRAIL_ATR({trail_mult_to_use}x)")
        else:
            new_trail = px + gap
            state["trail"] = min(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None: state["trail"] = min(state["trail"], state["breakeven"])
            if px > state["trail"]: _safe_full_close(f"TRAIL_ATR({trail_mult_to_use}x)")

# =================== BREAKOUT ENGINE ===================
def breakout_votes(df: pd.DataFrame, ind: dict, prev_ind: dict) -> tuple:
    votes=0.0; vote_details={}
    try:
        if len(df) < max(BREAKOUT_LOOKBACK_BARS, 20)+2: return 0.0, {"error":"Insufficient data"}
        price=float(df["close"].iloc[-1])
        atr_now=float(ind.get("atr") or 0.0); atr_prev=float(prev_ind.get("atr") or atr_now)
        if atr_prev>0:
            atr_ratio=atr_now/atr_prev
            if atr_ratio>=BREAKOUT_HARD_ATR_SPIKE: votes+=1.0; vote_details["atr_spike"]=f"Hard ({atr_ratio:.2f}x)"
            elif atr_ratio>=BREAKOUT_SOFT_ATR_SPIKE: votes+=0.5; vote_details["atr_spike"]=f"Soft ({atr_ratio:.2f}x)"
            else: vote_details["atr_spike"]=f"Normal ({atr_ratio:.2f}x)"
        adx=float(ind.get("adx") or 0.0); adx_prev=float(prev_ind.get("adx") or adx)
        if adx>=BREAKOUT_ADX_THRESHOLD: votes+=1.0; vote_details["adx"]=f"Strong ({adx:.1f})"
        elif adx>=18 and adx>adx_prev: votes+=0.5; vote_details["adx"]=f"Building ({adx:.1f})"
        else: vote_details["adx"]=f"Weak ({adx:.1f})"
        if len(df)>=21:
            current_volume=float(df["volume"].iloc[-1]); volume_ma=df["volume"].iloc[-21:-1].astype(float).mean()
            if volume_ma>0:
                ratio=current_volume/volume_ma
                if ratio>=BREAKOUT_VOLUME_SPIKE: votes+=1.0; vote_details["volume"]=f"Spike ({ratio:.2f}x)"
                elif ratio>=BREAKOUT_VOLUME_MED: votes+=0.5; vote_details["volume"]=f"High ({ratio:.2f}x)"
                else: vote_details["volume"]=f"Normal ({ratio:.2f}x)"
        recent_highs=df["high"].iloc[-BREAKOUT_LOOKBACK_BARS:-1].astype(float)
        recent_lows =df["low"].iloc[-BREAKOUT_LOOKBACK_BARS:-1].astype(float)
        if len(recent_highs)>0 and len(recent_lows)>0:
            highest_high=recent_highs.max(); lowest_low=recent_lows.min()
            if price>highest_high: votes+=1.0; vote_details["price_break"]=f"New High (>{highest_high:.6f})"
            elif price<lowest_low: votes+=1.0; vote_details["price_break"]=f"New Low (<{lowest_low:.6f})"
            else: vote_details["price_break"]="Within Range"
        rsi=float(ind.get("rsi") or 50.0)
        if rsi>=60: votes+=0.5; vote_details["rsi"]=f"Bull ({rsi:.1f})"
        elif rsi<=40: votes+=0.5; vote_details["rsi"]=f"Bear ({rsi:.1f})"
        else: vote_details["rsi"]=f"Neutral ({rsi:.1f})"
        vote_details["total_score"]=f"{votes:.1f}/5.0"
    except Exception as e:
        logging.error(f"breakout_votes error: {e}"); vote_details["error"]=str(e)
    return votes, vote_details

def detect_breakout(df: pd.DataFrame, ind: dict, prev_ind: dict) -> str:
    try:
        if len(df) < BREAKOUT_LOOKBACK_BARS + 2: return None
        price = float(df["close"].iloc[-1])
        adx = float(ind.get("adx") or 0.0)
        atr_now = float(ind.get("atr") or 0.0)
        atr_prev= float(prev_ind.get("atr") or atr_now)
        atr_spike = atr_now > atr_prev * BREAKOUT_ATR_SPIKE
        strong_trend = adx >= BREAKOUT_ADX_THRESHOLD
        if not (atr_spike and strong_trend): return None
        recent_highs=df["high"].iloc[-BREAKOUT_LOOKBACK_BARS:-1].astype(float)
        recent_lows =df["low"].iloc[-BREAKOUT_LOOKBACK_BARS:-1].astype(float)
        if len(recent_highs)>0 and price>recent_highs.max(): return "BULL_BREAKOUT"
        if len(recent_lows)>0 and price<recent_lows.min():   return "BEAR_BREAKOUT"
    except Exception as e:
        logging.error(f"detect_breakout error: {e}")
    return None

def handle_breakout_entries(df: pd.DataFrame, ind: dict, prev_ind: dict, bal: float, spread_bps: float) -> bool:
    global state
    if state["open"]: return False
    breakout_signal = detect_breakout(df, ind, prev_ind)
    if not breakout_signal or state["breakout_active"]: return False
    price = ind.get("price") or float(df["close"].iloc[-1])
    breakout_score, vote_details = breakout_votes(df, ind, prev_ind)
    if breakout_score < 3.0: return False
    if spread_bps is not None and spread_bps > 6: return False
    qty = compute_size(bal, price)
    if qty < (LOT_MIN or 1): return False
    if breakout_signal == "BULL_BREAKOUT":
        open_market("buy", qty, price)
        state["breakout_active"]=True; state["breakout_direction"]="bull"
    elif breakout_signal == "BEAR_BREAKOUT":
        open_market("sell", qty, price)
        state["breakout_active"]=True; state["breakout_direction"]="bear"
    state["breakout_entry_price"]=price; state["opened_by_breakout"]=True
    state["breakout_score"]=breakout_score; state["breakout_votes_detail"]=vote_details
    return True

def handle_breakout_exits(df: pd.DataFrame, ind: dict, prev_ind: dict) -> bool:
    global state
    if not state["breakout_active"] or not state["open"]: return False
    atr_now=float(ind.get("atr") or 0.0); atr_prev=float(prev_ind.get("atr") or atr_now)
    volatility_calm = atr_now < atr_prev * BREAKOUT_CALM_THRESHOLD
    if volatility_calm:
        direction=state["breakout_direction"]; entry_price=state["breakout_entry_price"]
        current_price=float(df["close"].iloc[-1])
        pnl_pct=((current_price-entry_price)/entry_price*100*(1 if direction=="bull" else -1))
        close_market_strict(f"Breakout ended - {pnl_pct:.2f}% PnL")
        state["breakout_active"]=False; state["breakout_direction"]=None
        state["breakout_entry_price"]=None; state["opened_by_breakout"]=False
        return True
    return False

def breakout_emergency_protection(ind: dict, prev_ind: dict) -> bool:
    if not (EMERGENCY_PROTECTION_ENABLED and state.get("open")): return False
    try:
        adx=float(ind.get("adx") or 0.0); rsi=float(ind.get("rsi") or 50.0)
        atr_now=float(ind.get("atr") or 0.0); atr_prev=float(prev_ind.get("atr") or atr_now)
        price = ind.get("price") or price_now() or state.get("entry")
        atr_spike = atr_now > atr_prev * EMERGENCY_ATR_SPIKE_RATIO
        strong_trend = adx >= EMERGENCY_ADX_MIN
        if not (atr_spike and strong_trend): return False
        pump = rsi >= EMERGENCY_RSI_PUMP; crash = rsi <= EMERGENCY_RSI_CRASH
        if not (pump or crash): return False
        side=state.get("side"); entry=state.get("entry") or price
        rr_pct=(price-entry)/entry*100*(1 if side=="long" else -1)
        if (pump and side=="short") or (crash and side=="long"):
            close_market_strict("EMERGENCY opposite pump/crash"); return True
        if EMERGENCY_POLICY=="close_always": close_market_strict("EMERGENCY favorable — close all"); return True
        if rr_pct>=EMERGENCY_FULL_CLOSE_PROFIT:
            close_market_strict(f"EMERGENCY full close @ {rr_pct:.2f}%"); return True
        harvest=max(0.0, min(1.0, EMERGENCY_HARVEST_FRAC))
        if harvest>0: close_partial(harvest, f"EMERGENCY harvest {harvest*100:.0f}%")
        state["breakeven"]=entry
        if atr_now>0:
            if side=="long": state["trail"]=max(state.get("trail") or (price-atr_now*EMERGENCY_TRAIL_ATR_MULT), price-atr_now*EMERGENCY_TRAIL_ATR_MULT)
            else:            state["trail"]=min(state.get("trail") or (price+atr_now*EMERGENCY_TRAIL_ATR_MULT), price+atr_now*EMERGENCY_TRAIL_ATR_MULT)
        if EMERGENCY_POLICY=="tp_then_close":
            close_market_strict("EMERGENCY: harvest then full close")
            return True
        return True
    except Exception as e:
        logging.error(f"breakout_emergency_protection error: {e}")
        return False

# =================== HUD / SNAPSHOT ===================
def build_log_insights(df: pd.DataFrame, ind: dict, price: float):
    adx=float(ind.get("adx") or 0.0); pdi=float(ind.get("plus_di") or 0.0); mdi=float(ind.get("minus_di") or 0.0)
    atr=float(ind.get("atr") or 0.0); rsi=float(ind.get("rsi") or 0.0)
    ema9=float(ind.get("ema9") or 0.0); ema20=float(ind.get("ema20") or 0.0); slope=float(ind.get("ema9_slope") or 0.0)
    bias="UP" if pdi>mdi else ("DOWN" if mdi>pdi else "NEUTRAL")
    regime="TREND" if adx>=20 else "RANGE"
    atr_pct=(atr/max(price or 1e-9,1e-9))*100.0
    rsi_zone="RSI🔥 Overbought" if rsi>=70 else ("RSI❄️ Oversold" if rsi<=30 else "RSI⚖️ Neutral")
    candle_info=detect_candle_pattern(df)
    ema_status=f"EMA9={fmt(ema9)} EMA20={fmt(ema20)} Slope={fmt(slope,4)}"
    return {"regime":regime, "bias":bias, "atr_pct":atr_pct, "rsi_zone":rsi_zone, "candle":candle_info, "ema_status":ema_status}

def snapshot(bal,info,ind,spread_bps,reason=None, df=None):
    df = df if df is not None else fetch_ohlcv()
    left_s = time_to_candle_close(df, USE_TV_BAR)
    insights = build_log_insights(df, ind, info.get("price"))
    print(colored("─"*100,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*100,"cyan"))
    print("📈 INDICATORS & CANDLES")
    print(f"   💲 Price {fmt(info.get('price'))}  |  RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))}  lo={fmt(info.get('lo'))}")
    print(f"   🧮 RSI({RSI_LEN})={fmt(ind['rsi'])}   +DI={fmt(ind['plus_di'])}   -DI={fmt(ind['minus_di'])}   ADX({ADX_LEN})={fmt(ind['adx'])}   ATR={fmt(ind['atr'])} (~{fmt(insights['atr_pct'],2)}%)")
    print(f"   📊 EMA: {insights['ema_status']}")
    print(f"   🎯 Signal  ✅ BUY={info['long']}   ❌ SELL={info['short']}   |   🧮 spread_bps={fmt(spread_bps,2)}")
    candle_info = insights['candle']
    print(f"   🕯️ Candles = {candle_info['name_ar']} / {candle_info['name_en']} (Strength: {candle_info['strength']}/4)")
    if state.get("tci") is not None:
        hold_msg = "الترند قوي — امسك الصفقة" if state.get("_hold_trend") else "إدارة عادية"
        print(colored(f"   🧭 TCI={state['tci']:.0f}/100 • Chop01={state.get('chop01',0):.2f} → {hold_msg}", "cyan" if state.get("_hold_trend") else "blue"))
    ach = state.get("profit_targets_achieved", 0); lad = state.get("_tp_ladder") or []
    if ach < len(lad):
        nxt = lad[ach]
        print(colored(f"   🎯 Dynamic TP: next={nxt:.2f}% • ATR%≈{state.get('_atr_pct',0):.2f} • Consensus={state.get('_consensus_score',0):.1f}/5", "magenta"))
    if not state["open"] and wait_for_next_signal_side:
        print(colored(f"   ⏳ WAITING — need next {wait_for_next_signal_side.upper()} RF signal", "cyan"))
    if state["breakout_active"]:
        print(colored(f"   ⚡ BREAKOUT ACTIVE: {state['breakout_direction']}", "cyan"))
    print(f"   ⏱️ Candle closes in ~ {left_s}s")
    print("\n🧭 POSITION & MANAGEMENT")
    print(f"   💰 Balance {fmt(bal,2)} USDT   Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x")
    if state["open"]:
        lamp='🟩 LONG' if state['side']=='long' else '🟥 SHORT'
        print(f"   📌 {lamp}  Entry={fmt(state['entry'])}  Qty={fmt(state['qty'],4)}  Bars={state['bars']}  PnL={fmt(state['pnl'])}")
        print(f"   🎯 Scale-outs={state['scale_outs']}  Trail={fmt(state['trail'])}")
        print(f"   📊 TP1_done={state['tp1_done']}  Breakeven={fmt(state['breakeven'])}  HighestProfit={fmt(state['highest_profit_pct'],2)}%")
        if state.get("thrust_locked"): print(colored(f"   🔒 THRUST LOCK Active", "cyan"))
        if state.get("opened_by_breakout"): print(colored(f"   ⚡ OPENED BY BREAKOUT", "magenta"))
        if state['last_action']: print(f"   🔄 Last Action: {state['last_action']} - {state['action_reason']}")
    else:
        print("   ⚪ FLAT")
    print("\n📦 RESULTS")
    eff_eq = (bal or 0.0) + compound_pnl
    print(f"   🧮 CompoundPnL {fmt(compound_pnl)}   🚀 EffectiveEq {fmt(eff_eq)} USDT")
    if reason: print(colored(f"   ℹ️ WAIT — reason: {reason}","yellow"))
    print(colored("─"*100,"cyan"))

# =================== BARS / WATCHDOG ===================
_last_bar_ts = None
def update_bar_counters(df: pd.DataFrame):
    global _last_bar_ts
    if len(df)==0: return False
    last_ts=int(df["time"].iloc[-1])
    if _last_bar_ts is None:
        _last_bar_ts=last_ts; return False
    if last_ts != _last_bar_ts:
        _last_bar_ts=last_ts
        if state["open"]: state["bars"] += 1
        return True
    return False

def loop_heartbeat():
    global last_loop_ts
    last_loop_ts = time.time()

def watchdog_check(max_stall=180):
    while True:
        try:
            stall_time = time.time() - last_loop_ts
            if stall_time > max_stall:
                print(colored(f"🛑 WATCHDOG: main loop stall > {max_stall}s", "red"))
                logging.critical(f"WATCHDOG: main loop stalled for {stall_time:.0f}s")
            time.sleep(60)
        except Exception as e:
            logging.error(f"watchdog_check error: {e}")
            time.sleep(60)

# =================== MAIN LOOP ===================
def trade_loop():
    global state, post_close_cooldown, wait_for_next_signal_side, last_close_signal_time, last_open_fingerprint
    loop_counter=0
    while True:
        try:
            loop_heartbeat(); loop_counter += 1
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            new_bar = update_bar_counters(df)

            # CLOSED-CANDLE signals only
            df_closed = df.iloc[:-1] if len(df)>=2 else df.copy()
            info_closed = compute_tv_signals(df_closed)
            info_live   = {"time": int(df["time"].iloc[-1]) if len(df) else int(time.time()*1000),
                           "price": px or (float(df["close"].iloc[-1]) if len(df) else None),
                           "filter": info_closed.get("filter"), "hi": info_closed.get("hi"),
                           "lo": info_closed.get("lo")}
            ind  = compute_indicators(df)
            prev_ind = compute_indicators(df.iloc[:-1]) if len(df)>=2 else ind
            spread_bps = orderbook_spread_bps()

            # PnL snapshot
            if state["open"] and px:
                state["pnl"] = (px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]

            # Breakout lifecycle
            breakout_exited = handle_breakout_exits(df, ind, prev_ind)
            breakout_entered = False
            if not state["open"] and not breakout_exited:
                breakout_entered = handle_breakout_entries(df_closed, ind, prev_ind, bal, spread_bps)
            if breakout_entered:
                snapshot(bal, {**info_closed, "price": px or info_closed["price"]}, ind, spread_bps, "BREAKOUT ENTRY - skipping normal logic", df)
                time.sleep(compute_next_sleep(df)); continue
            if state["breakout_active"]:
                snapshot(bal, {**info_closed, "price": px or info_closed["price"]}, ind, spread_bps, "BREAKOUT ACTIVE - monitoring exit", df)
                time.sleep(compute_next_sleep(df)); continue

            # Emergency layer if open
            if state["open"]:
                if breakout_emergency_protection(ind, prev_ind):
                    snapshot(bal, {**info_closed, "price": px or info_closed["price"]}, ind, spread_bps, "EMERGENCY LAYER action", df)
                    time.sleep(compute_next_sleep(df)); continue

            # Post-entry smart management
            smart_exit_check({**info_closed, "price": px or info_closed["price"]}, ind, df_cached=df, prev_ind_cached=prev_ind)
            post_action = smart_post_entry_manager(df, ind, {**info_closed, "price": px or info_closed["price"]})
            if post_action:
                logging.info(f"POST_ENTRY_ACTION: {post_action}")

            # ENTRY on closed-candle RF signal only
            sig = "buy" if info_closed["long"] else ("sell" if info_closed["short"] else None)
            reason=None
            if not sig:
                reason="no signal"
            elif spread_bps is not None and spread_bps > 6:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > 6)"
            elif post_close_cooldown > 0:
                reason=f"cooldown {post_close_cooldown} bars"

            # If open and opposite signal → close strictly then wait next opposite signal
            if state["open"] and sig and (reason is None):
                desired = "long" if sig=="buy" else "short"
                if state["side"] != desired:
                    prev_side = state["side"]
                    close_market_strict("opposite_signal")
                    wait_for_next_signal_side = "sell" if prev_side=="long" else "buy"
                    last_close_signal_time = info_closed["time"]
                    snapshot(bal, {**info_closed, "price": px or info_closed["price"]}, ind, spread_bps, "waiting next opposite signal", df)
                    time.sleep(compute_next_sleep(df)); continue

            # Open when flat
            if not state["open"] and (reason is None) and sig:
                if wait_for_next_signal_side:
                    if sig != wait_for_next_signal_side:
                        reason=f"waiting opposite RF signal: need {wait_for_next_signal_side}"
                    else:
                        qty = compute_size(bal, px or info_closed["price"])
                        if qty>0:
                            open_market(sig, qty, px or info_closed["price"])
                            wait_for_next_signal_side=None; last_close_signal_time=None; last_open_fingerprint=None
                            last_signal_id=f"{info_closed['time']}:{sig}"
                        else:
                            reason="qty<=0"
                else:
                    qty = compute_size(bal, px or info_closed["price"])
                    if qty>0:
                        open_market(sig, qty, px or info_closed["price"])
                        last_open_fingerprint=None
                        last_signal_id=f"{info_closed['time']}:{sig}"
                    else:
                        reason="qty<=0"

            snapshot(bal, {**info_closed, "price": px or info_closed["price"]}, ind, spread_bps, reason, df)

            if post_close_cooldown>0 and not state["open"]: post_close_cooldown -= 1
            if loop_counter % 5 == 0: save_state()

            # pacing
            time.sleep(compute_next_sleep(df))

        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}", "red"))
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# =================== KEEPALIVE & API ===================
def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive: SELF_URL/RENDER_EXTERNAL_URL not set — skipping.", "yellow"))
        return
    sess=requests.Session(); sess.headers.update({"User-Agent":"rf-pro-bot/keepalive"})
    print(colored(f"KEEPALIVE start → every {KEEPALIVE_SECONDS}s → {url}", "cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(max(KEEPALIVE_SECONDS,15))

app = Flask(__name__)
import logging as flask_logging
flask_logging.getLogger('werkzeug').setLevel(flask_logging.ERROR)
_root_logged=False

@app.route("/")
def home():
    global _root_logged
    if not _root_logged: print("GET / HTTP/1.1 200"); _root_logged=True
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ RF Bot — {SYMBOL} {INTERVAL} — {mode} — TREND-ONLY — CLOSED-CANDLE RF — SMART HARVESTING — BREAKOUT ENGINE — EMERGENCY LAYER — STRICT CLOSE"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "position": state, "compound_pnl": compound_pnl, "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "strategy": STRATEGY, "trend_only": True, "bingx_mode": BINGX_POSITION_MODE,
        "strict_close_enabled": STRICT_EXCHANGE_CLOSE,
        "waiting_for_signal": wait_for_next_signal_side,
        "breakout_engine": {
            "active": state.get("breakout_active", False),
            "direction": state.get("breakout_direction"),
            "entry_price": state.get("breakout_entry_price"),
            "score": state.get("breakout_score", 0.0),
            "votes": state.get("breakout_votes_detail", {})
        },
        "emergency_protection": {
            "enabled": EMERGENCY_PROTECTION_ENABLED,
            "policy": EMERGENCY_POLICY,
            "harvest_frac": EMERGENCY_HARVEST_FRAC
        },
        "fearless_hold": {
            "tci": state.get("tci"), "chop01": state.get("chop01"),
            "hold_mode": state.get("_hold_trend", False),
            "hold_tci_threshold": HOLD_TCI, "strong_hold_tci_threshold": HOLD_STRONG_TCI
        },
        "dynamic_profit_taking": {
            "consensus_score": state.get("_consensus_score"),
            "atr_pct": state.get("_atr_pct"),
            "tp_ladder": state.get("_tp_ladder"),
            "tp_fracs": state.get("_tp_fracs")
        },
        "ema_indicators": {
            "ema9": state.get("ema9"), "ema20": state.get("ema20"), "ema9_slope": state.get("ema9_slope")
        }
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "loop_stall_s": time.time() - last_loop_ts,
        "mode": "live" if MODE_LIVE else "paper",
        "open": state["open"], "side": state["side"], "qty": state["qty"],
        "compound_pnl": compound_pnl, "consecutive_errors": _consec_err,
        "timestamp": datetime.utcnow().isoformat(),
        "strict_close_enabled": STRICT_EXCHANGE_CLOSE,
        "trade_mode": state.get("trade_mode"),
        "profit_targets_achieved": state.get("profit_targets_achieved", 0),
        "waiting_for_signal": wait_for_next_signal_side,
        "breakout_active": state.get("breakout_active", False),
        "emergency_protection_enabled": EMERGENCY_PROTECTION_ENABLED,
        "breakout_score": state.get("breakout_score", 0.0),
        "tp0_done": state.get("_tp0_done", False),
        "thrust_locked": state.get("thrust_locked", False),
        "fearless_hold": {"tci": state.get("tci"), "chop01": state.get("chop01"), "hold_mode": state.get("_hold_trend", False)},
        "dynamic_profit_taking": {"consensus_score": state.get("_consensus_score"), "atr_pct": state.get("_atr_pct"), "tp_ladder": state.get("_tp_ladder")},
        "ema_indicators": {"ema9": state.get("ema9"), "ema20": state.get("ema20"), "ema9_slope": state.get("ema9_slope")}
    }), 200

@app.route("/ping")
def ping(): return "pong", 200

# =================== BOOT ===================
def _validate_market_specs():
    assert AMT_PREC is None or (isinstance(AMT_PREC, int) and AMT_PREC >= 0), "AMT_PREC invalid"
    assert LOT_MIN is None or (isinstance(LOT_MIN, (int, float)) and LOT_MIN >= 0), "LOT_MIN invalid"
    assert LOT_STEP is None or (isinstance(LOT_STEP, (int, float)) and LOT_STEP >= 0), "LOT_STEP invalid"

if __name__ == "__main__":
    setup_file_logging()
    _validate_market_specs()
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • SYMBOL={SYMBOL} • {INTERVAL}", "yellow"))
    print(colored(f"STRATEGY: {STRATEGY.upper()} (TREND-ONLY) • SMART_EXIT={'ON' if USE_SMART_EXIT else 'OFF'}", "yellow"))
    load_state()
    print(colored("🛡️ Watchdog started", "cyan"))
    threading.Thread(target=watchdog_check, daemon=True).start()
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
