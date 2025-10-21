# -*- coding: utf-8 -*-
"""
RF Futures Bot — RF-LIVE ONLY (BingX Perp via CCXT) + ORCHESTRA + SMC
• Entry: Range Filter (TradingView-like) — LIVE CANDLE ONLY (RF فقط)
• Post-entry (Orchestra+SMC): EQH/EQL + FVG + Order Blocks + BOS/CHOCH + Premium/Discount + Sweeps
• Dynamic TP ladder + Breakeven + ATR-trailing + Impulse/Wick harvesting
• One-shot full exit عند تأكد "آخر قمة/آخر قاع" (لتقليل العمولات)
• Strict close مع التحقق من المنصّة + Dust/Final-chunk guard (<= 40 DOGE)
• Opposite-signal wait policy بعد أي إغلاق
• Flask /metrics + /health + rotated logging + HUD ملوّن

مهم: منطق الدخول كما هو — RF على الشمعة الحيّة فقط.
"""

import os, time, math, random, signal, sys, traceback, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pandas as pd
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV / MODE ===================
API_KEY = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE = bool(API_KEY and API_SECRET)

SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.getenv("PORT", 5000))

# =================== SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))   # 60%

POSITION_MODE = os.getenv("BINGX_POSITION_MODE", "oneway")  # oneway/hedge

# RF — live candle only
RF_SOURCE = "close"
RF_PERIOD = int(os.getenv("RF_PERIOD", 20))
RF_MULT   = float(os.getenv("RF_MULT", 3.5))
RF_LIVE_ONLY = True
RF_HYST_BPS  = 6.0  # هيستريسس للشمعة الحيّة

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

# دخول فقط من RF
ENTRY_RF_ONLY = True

# Spread guard
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 6.0))

# Dynamic TP / trail
TP1_PCT_BASE       = 0.40
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.6

# سلّم الأهداف الافتراضي
TREND_TPS       = [0.50, 1.00, 1.80]
TREND_TP_FRACS  = [0.30, 0.30, 0.20]

# Dust guard / final chunk
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 40.0))
RESIDUAL_MIN_QTY = float(os.getenv("RESIDUAL_MIN_QTY", 9.0))

# Strict close
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# ============ ORCHESTRA / SMC (Post-Entry Smart Layer) ============
ORCHESTRA_ENABLED = True

# EQH/EQL + FVG + Traps
EQ_TOLERANCE_PCT   = 0.05
LEVEL_PROX_BPS     = 10.0
FVG_MAX_GAP_ATR    = 2.0
TRAP_WICK_PCT      = 60.0
TRAP_BODY_MAX_PCT  = 25.0
VOL_SPIKE_X        = 1.30
REJECT_NEED_VOTES  = 2
CONFIRM_BARS       = 1
IMPULSE_X_ATR      = 1.2
WICK_HARVEST_PCT   = 45.0
HOLD_STRONG_ADX    = 28
ONE_SHOT_MIN_PCT   = 0.60
TRAIL_WIDE_MULT    = 2.0
RF_FAKE_GUARD      = True

# --- SMC specific (Order Blocks, BOS/CHOCH, PD arrays, Sweeps) ---
SMC_ENABLED         = True
OB_LOOKBACK         = 80          # عدد الشموع التي نبحث فيها عن آخر OB
OB_DISP_MULT_ATR    = 1.2         # دسبليسمِنت: حركة لاحقة ≥ 1.2×ATR
OB_TOLERANCE_BPS    = 12.0        # سماحية دخول/ملامسة داخل منطقة OB
BOS_LOOKBACK_SWINGS = 8           # كم سوينج نراجع لأقرب BOS/CHOCH
PD_MID_METHOD       = "fib50"     # fib50 أو fib62 (قابل للتبديل)
SWEEP_WICK_PCT      = 55.0        # ذيل ≥55% فوق/تحت سوينج = Sweep
CHOCH_NEED_BREAK    = True        # لازم كسر واضح (close)

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
    print(colored("🗂️ log rotation ready", "cyan"))

setup_file_logging()

# =================== EXCHANGE ===================
def make_ex():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap"}
    })

ex = make_ex()
MARKET = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN  = None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision", {}) or {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits", {}) or {}).get("amount", {}).get("step", None)
        LOT_MIN  = (MARKET.get("limits", {}) or {}).get("amount", {}).get("min",  None)
        print(colored(f"🔧 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}", "yellow"))

def ensure_leverage_mode():
    try:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            print(colored(f"✅ leverage set: {LEVERAGE}x", "green"))
        except Exception as e:
            print(colored(f"⚠️ set_leverage warn: {e}", "yellow"))
        print(colored(f"📌 position mode: {POSITION_MODE}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_leverage_mode: {e}", "yellow"))

try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}", "yellow"))

# =================== HELPERS ===================
_consec_err = 0
last_loop_ts = time.time()

def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = int(AMT_PREC) if AMT_PREC and AMT_PREC>=0 else 0
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except (InvalidOperation, ValueError, TypeError):
        return max(0.0, float(q))

def safe_qty(q):
    q = _round_amt(q)
    if q<=0: print(colored(f"⚠️ qty invalid after normalize → {q}", "yellow"))
    return q

def fmt(v, d=6, na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def with_retry(fn, tries=3, base_wait=0.4):
    global _consec_err
    for i in range(tries):
        try:
            r = fn()
            _consec_err = 0
            return r
        except Exception:
            _consec_err += 1
            if i == tries-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.25)

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
    iv=(iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame) -> int:
    tf = _interval_seconds(INTERVAL)
    if len(df) == 0: return tf
    cur_start_ms = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    next_close_ms = cur_start_ms + tf*1000
    while next_close_ms <= now_ms:
        next_close_ms += tf*1000
    left = max(0, next_close_ms - now_ms)
    return int(left/1000)

# =================== INDICATORS ===================
def wilder_ema(s: pd.Series, n: int):
    return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0}
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

    i=len(df)-1
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i])
    }

# =================== RANGE FILTER (TV-like) ===================
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

def rf_signal_live(df: pd.DataFrame):
    if len(df) < RF_PERIOD + 3:
        i = -1
        price = float(df["close"].iloc[i]) if len(df) else None
        return {"time": int(df["time"].iloc[i]) if len(df) else int(time.time()*1000),
                "price": price or 0.0, "long": False, "short": False,
                "filter": price or 0.0, "hi": price or 0.0, "lo": price or 0.0}
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    p_now = float(src.iloc[-1]); p_prev = float(src.iloc[-2])
    f_now = float(filt.iloc[-1]); f_prev = float(filt.iloc[-2])
    long_flip  = (p_prev <= f_prev and p_now > f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    short_flip = (p_prev >= f_prev and p_now < f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    return {
        "time": int(df["time"].iloc[-1]), "price": p_now,
        "long": bool(long_flip), "short": bool(short_flip),
        "filter": f_now, "hi": float(hi.iloc[-1]), "lo": float(lo.iloc[-1])
    }

# =================== STRUCTURE / LIQUIDITY (incl. SMC) ===================
def _find_swings(df: pd.DataFrame, left:int=2, right:int=2):
    if len(df) < left+right+3:
        return [None]*len(df), [None]*len(df)
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    ph = [None]*len(df); pl = [None]*len(df)
    for i in range(left, len(df)-right):
        if all(h[i] >= h[j] for j in range(i-left, i+right+1)): ph[i] = h[i]
        if all(l[i] <= l[j] for j in range(i-left, i+right+1)): pl[i] = l[i]
    return ph, pl

def _eq_levels(ph, pl):
    eqh_candidates = []; eql_candidates = []
    for arr, is_high in ((ph, True),(pl, False)):
        for i, price in enumerate(arr):
            if price is None: continue
            tol = price * (EQ_TOLERANCE_PCT/100.0)
            local = []
            for j in range(max(0,i-10), min(len(arr), i+10)):
                pj = arr[j]
                if pj is not None and abs(pj - price) <= tol:
                    local.append(pj)
            if len(local) >= 2:
                if is_high: eqh_candidates.append(max(local))
                else:       eql_candidates.append(min(local))
    eqh = max(eqh_candidates) if eqh_candidates else None
    eql = min(eql_candidates) if eql_candidates else None
    return eqh, eql

def _fvg_scan(df_closed: pd.DataFrame, atr_now: float):
    try:
        if len(df_closed) < 3 or atr_now <= 0: return None
        for i in range(len(df_closed)-3, max(len(df_closed)-20, 2), -1):
            prev_high = float(df_closed["high"].iloc[i-1])
            prev_low  = float(df_closed["low"].iloc[i-1])
            cur_low   = float(df_closed["low"].iloc[i])
            cur_high  = float(df_closed["high"].iloc[i])
            if cur_low > prev_high and (cur_low - prev_high) <= FVG_MAX_GAP_ATR*atr_now:
                return {"type":"BULL_FVG","bottom":prev_high,"top":cur_low}
            if cur_high < prev_low  and (prev_low - cur_high) <= FVG_MAX_GAP_ATR*atr_now:
                return {"type":"BEAR_FVG","bottom":cur_high,"top":prev_low}
    except Exception:
        pass
    return None

# ---- SMC helpers ----
def _pd_mid(high, low, method="fib50"):
    if high is None or low is None: return None
    if method=="fib62":
        return low + (high-low)*0.62
    return (high + low)/2.0  # fib50

def _last_two_swings(ph, pl):
    highs = [i for i,v in enumerate(ph) if v is not None]
    lows  = [i for i,v in enumerate(pl) if v is not None]
    last_high_idx = highs[-1] if highs else None
    prev_high_idx = highs[-2] if len(highs)>=2 else None
    last_low_idx  = lows[-1]  if lows else None
    prev_low_idx  = lows[-2]  if len(lows)>=2 else None
    return last_high_idx, prev_high_idx, last_low_idx, prev_low_idx

def _bos_choch(df_closed: pd.DataFrame, ph, pl):
    """بسيطة: BOS عندما نكسر آخر سوينج بنفس الاتجاه، CHOCH عندما نكسر عكس الاتجاه السابق."""
    if len(df_closed) < 5: return {"bos":None, "choch":False}
    lh, phv, ll, plv = _last_two_swings(ph, pl)
    if lh is None or ll is None: return {"bos":None, "choch":False}
    last_close = float(df_closed["close"].iloc[-1])
    bos = None; choch=False
    # كسرة فوق آخر قمة؟
    if ph[lh] is not None and last_close > ph[lh]:
        bos = "up"
        # لو قبلها كنا بكسر قيعان مؤخرًا → CHOCH
        if plv is not None and df_closed["close"].iloc[ll] < pl[plv]:
            choch=True
    # كسرة تحت آخر قاع؟
    if pl[ll] is not None and last_close < pl[ll]:
        bos = "down"
        if phv is not None and df_closed["close"].iloc[lh] > ph[phv]:
            choch=True
    return {"bos":bos, "choch":choch}

def _find_order_blocks(df_closed: pd.DataFrame, atr_now: float):
    """آخر Bullish/Bearish Order Block وفق SMC: آخر شمعة عكسية قبل دسبليسمِنت قوي."""
    bull = None; bear = None
    n = len(df_closed)
    if n < 10 or atr_now<=0: return {"bull":bull, "bear":bear}
    # مسح للخلف
    for i in range(n-3, max(3, n-OB_LOOKBACK)-1, -1):
        o = float(df_closed["open"].iloc[i])
        c = float(df_closed["close"].iloc[i])
        h = float(df_closed["high"].iloc[i])
        l = float(df_closed["low"].iloc[i])
        # Bullish OB: آخر شمعة هابطة قبل اندفاع صاعد ≥ OB_DISP_MULT_ATR * ATR
        if c < o:
            fwd_high = float(df_closed["high"].iloc[i+1:i+4].max())
            disp = fwd_high - h
            if disp >= OB_DISP_MULT_ATR*atr_now:
                bull = {"time": int(df_closed["time"].iloc[i]),
                        "low": l, "high": o, "type":"BULL_OB"}
                break
    for i in range(n-3, max(3, n-OB_LOOKBACK)-1, -1):
        o = float(df_closed["open"].iloc[i])
        c = float(df_closed["close"].iloc[i])
        h = float(df_closed["high"].iloc[i])
        l = float(df_closed["low"].iloc[i])
        # Bearish OB: آخر شمعة صاعدة قبل اندفاع هابط
        if c > o:
            fwd_low = float(df_closed["low"].iloc[i+1:i+4].min())
            disp = l - fwd_low
            if disp >= OB_DISP_MULT_ATR*atr_now:
                bear = {"time": int(df_closed["time"].iloc[i]),
                        "low": c, "high": h, "type":"BEAR_OB"}
                break
    return {"bull":bull, "bear":bear}

def _in_zone(px, low, high, bps):
    try:
        if low is None or high is None: return False
        if low>high: low,high = high,low
        if px < low or px > high: return False
        # قرب الحواف كمان
        near_low  = abs((px-low)/low)*10000.0 <= bps if low>0 else False
        near_high = abs((px-high)/high)*10000.0 <= bps if high>0 else False
        return True or near_low or near_high
    except Exception:
        return False

def _liquidity_sweep(df: pd.DataFrame, ph, pl):
    """سويب سيولة بسيط: ذيل يضرب آخر سوينج ويرجع يغلق داخله."""
    if len(df)<3: return None
    o = float(df["open"].iloc[-1]); h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1]);  c = float(df["close"].iloc[-1])
    rng = max(h-l, 1e-12)
    upper_pct = (h - max(o,c))/rng*100.0
    lower_pct = (min(o,c) - l)/rng*100.0
    # آخر سوينجات
    hs = [v for v in ph if v is not None]
    ls = [v for v in pl if v is not None]
    eqh = max(hs) if hs else None
    eql = min(ls) if ls else None
    if eqh and h>eqh and c<eqh and upper_pct>=SWEEP_WICK_PCT:
        return {"type":"SWEEP_HIGH", "level":eqh}
    if eql and l<eql and c>eql and lower_pct>=SWEEP_WICK_PCT:
        return {"type":"SWEEP_LOW", "level":eql}
    return None

def build_structure_snapshot(df_full: pd.DataFrame, ind: dict):
    df_closed = df_full.iloc[:-1] if len(df_full)>=2 else df_full.copy()
    ph, pl = _find_swings(df_closed)
    eqh, eql = _eq_levels(ph, pl)
    fvg = _fvg_scan(df_closed, float(ind.get("atr") or 0.0))
    smc = {"bos":None,"choch":False,"pd":None,"ob":None,"sweep":None}
    if SMC_ENABLED:
        # BOS/CHOCH
        smc.update(_bos_choch(df_closed, ph, pl))
        # PD array (premium/discount) من آخر هاي/لو
        highs = [v for v in ph if v is not None]
        lows  = [v for v in pl if v is not None]
        hi = max(highs) if highs else None
        lo = min(lows)  if lows  else None
        mid = _pd_mid(hi, lo, PD_MID_METHOD) if hi and lo else None
        last_close = float(df_closed["close"].iloc[-1]) if len(df_closed) else None
        if mid and last_close:
            smc["pd"] = "premium" if last_close>mid else "discount"
        # Order Blocks
        atr_now = float(ind.get("atr") or 0.0)
        smc["ob"] = _find_order_blocks(df_closed, atr_now)
        # Sweeps
        smc["sweep"] = _liquidity_sweep(df_full, ph, pl)
    return {"eqh":eqh, "eql":eql, "fvg":fvg, "ph":ph, "pl":pl, "smc":smc}

def _bps(a,b):
    try: return abs((a-b)/b)*10000.0
    except Exception: return 0.0

# =================== STATE ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0,
    # ORCHESTRA/SMC
    "struct": None,
    "rejection_votes": 0,
    "hold_votes": 0,
    "last_level_tag": None,
    "last_reject_bar_ts": None,
    "confirm_queue": 0,
}

compound_pnl = 0.0
wait_for_next_signal_side = None  # "buy"/"sell"

# =================== ORDERS ===================
def _params_open(side):
    if POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if side=="buy" else "SHORT", "reduceOnly": False}
    return {"positionSide": "BOTH", "reduceOnly": False}

def _params_close():
    if POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if STATE.get("side")=="long" else "SHORT", "reduceOnly": True}
    return {"positionSide": "BOTH", "reduceOnly": True}

def _read_position():
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
        logging.error(f"_read_position error: {e}")
    return 0.0, None, None

def compute_size(balance, price):
    effective = balance or 0.0
    capital = effective * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def open_market(side, qty, price):
    if qty<=0:
        print(colored("❌ skip open (qty<=0)", "red"))
        return False
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception: pass
            ex.create_order(SYMBOL, "market", side, qty, None, _params_open(side))
        except Exception as e:
            print(colored(f"❌ open: {e}", "red"))
            logging.error(f"open_market error: {e}")
            return False
    STATE.update({
        "open": True, "side": "long" if side=="buy" else "short", "entry": price,
        "qty": qty, "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "struct": None, "rejection_votes": 0, "hold_votes": 0,
        "last_level_tag": None, "last_reject_bar_ts": None, "confirm_queue": 0
    })
    print(colored(f"🚀 OPEN {('LONG' if side=='buy' else 'SHORT')} qty={fmt(qty,4)} @ {fmt(price)}", "green" if side=="buy" else "red"))
    logging.info(f"OPEN {side} qty={qty} price={price}")
    return True

def close_market_strict(reason="STRICT"):
    global compound_pnl, wait_for_next_signal_side
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty <= 0:
        if STATE.get("open"):
            _reset_after_close(reason)
        return
    side_to_close = "sell" if (exch_side=="long") else "buy"
    qty_to_close  = safe_qty(exch_qty)
    attempts=0; last_error=None
    while attempts < CLOSE_RETRY_ATTEMPTS:
        try:
            if MODE_LIVE:
                params = _params_close(); params["reduceOnly"]=True
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            time.sleep(CLOSE_VERIFY_WAIT_S)
            left_qty, _, _ = _read_position()
            if left_qty <= 0:
                px = price_now() or STATE.get("entry")
                entry_px = STATE.get("entry") or exch_entry or px
                side = STATE.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
                qty  = exch_qty
                pnl  = (px - entry_px) * qty * (1 if side=="long" else -1)
                compound_pnl += pnl
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                logging.info(f"STRICT_CLOSE {side} pnl={pnl} total={compound_pnl}")
                _reset_after_close(reason, prev_side=side)
                return
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            print(colored(f"⚠️ strict close retry {attempts}/{CLOSE_RETRY_ATTEMPTS} — residual={fmt(left_qty,4)}","yellow"))
            time.sleep(CLOSE_VERIFY_WAIT_S)
        except Exception as e:
            last_error = e; logging.error(f"close_market_strict attempt {attempts+1}: {e}"); attempts += 1; time.sleep(CLOSE_VERIFY_WAIT_S)
    print(colored(f"❌ STRICT CLOSE FAILED after {CLOSE_RETRY_ATTEMPTS} attempts — last error: {last_error}", "red"))
    logging.critical(f"STRICT CLOSE FAILED — last_error={last_error}")

def _reset_after_close(reason, prev_side=None):
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0, "profit_targets_achieved": 0,
        "struct": None, "rejection_votes": 0, "hold_votes": 0,
        "last_level_tag": None, "last_reject_bar_ts": None, "confirm_queue": 0
    })
    if prev_side == "long":  wait_for_next_signal_side = "sell"
    elif prev_side == "short": wait_for_next_signal_side = "buy"
    else: wait_for_next_signal_side = None
    logging.info(f"AFTER_CLOSE waiting_for={wait_for_next_signal_side}")

def close_partial(frac, reason):
    if not STATE["open"] or STATE["qty"]<=0: return
    qty_close = safe_qty(max(0.0, STATE["qty"] * min(max(frac,0.0),1.0)))
    px = price_now() or STATE["entry"]
    min_unit = max(RESIDUAL_MIN_QTY, LOT_MIN or RESIDUAL_MIN_QTY)
    if qty_close < min_unit:
        print(colored(f"⏸️ skip partial (amount={fmt(qty_close,4)} < min_unit={fmt(min_unit,4)})", "yellow"))
        return
    side = "sell" if STATE["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_params_close())
        except Exception as e: print(colored(f"❌ partial close: {e}", "red")); return
    pnl = (px - STATE["entry"]) * qty_close * (1 if STATE["side"]=="long" else -1)
    STATE["qty"] = safe_qty(STATE["qty"] - qty_close)
    logging.info(f"PARTIAL_CLOSE {reason} qty={qty_close} pnl={pnl} rem={STATE['qty']}")
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(STATE['qty'],4)}","magenta"))
    if STATE["qty"] <= FINAL_CHUNK_QTY and STATE["qty"]>0:
        print(colored(f"🧹 Final chunk ≤ {FINAL_CHUNK_QTY} DOGE → strict close", "yellow"))
        close_market_strict("FINAL_CHUNK_RULE")

# =================== DYNAMIC TP ===================
def _consensus(ind, info, side) -> float:
    score=0.0
    try:
        adx=float(ind.get("adx") or 0.0)
        rsi=float(ind.get("rsi") or 50.0)
        if (side=="long" and rsi>=55) or (side=="short" and rsi<=45): score += 1.0
        if adx>=28: score += 1.0
        elif adx>=20: score += 0.5
        if abs(info["price"]-info["filter"])/max(info["filter"],1e-9) >= (RF_HYST_BPS/10000.0): score += 0.5
    except Exception: pass
    return float(score)

def _tp_ladder(info, ind, side):
    px = info["price"]; atr = float(ind.get("atr") or 0.0)
    atr_pct = (atr / max(px,1e-9))*100.0 if px else 0.5
    score = _consensus(ind, info, side)
    if score >= 2.5: mults = [1.8, 3.2, 5.0]
    elif score >= 1.5: mults = [1.6, 2.8, 4.5]
    else: mults = [1.2, 2.4, 4.0]
    tps = [round(m*atr_pct, 2) for m in mults]
    frs = [0.25, 0.30, 0.45]
    return tps, frs

def manage_after_entry(df, ind, info):
    if not STATE["open"] or STATE["qty"]<=0: return
    px = info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr = (px - entry)/entry*100*(1 if side=="long" else -1)
    dyn_tps, dyn_fracs = _tp_ladder(info, ind, side)
    STATE.setdefault("_tp_cache", dyn_tps); STATE["_tp_cache"]=dyn_tps
    STATE.setdefault("_tp_fracs", dyn_fracs); STATE["_tp_fracs"]=dyn_fracs
    k = int(STATE.get("profit_targets_achieved", 0))
    tp1_now = TP1_PCT_BASE*(2.2 if ind.get("adx",0)>=35 else 1.8 if ind.get("adx",0)>=28 else 1.0)
    if (not STATE["tp1_done"]) and rr >= tp1_now:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{tp1_now:.2f}%")
        STATE["tp1_done"]=True
        if rr >= BREAKEVEN_AFTER: STATE["breakeven"]=entry
    if k < len(dyn_tps) and rr >= dyn_tps[k]:
        frac = dyn_fracs[k] if k < len(dyn_fracs) else 0.25
        close_partial(frac, f"TP_dyn@{dyn_tps[k]:.2f}%")
        STATE["profit_targets_achieved"] = k + 1
    if rr > STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=rr
    if rr >= TRAIL_ACTIVATE_PCT and ind.get("atr",0)>0:
        trail_mult = TRAIL_WIDE_MULT if ind.get("adx",0) >= HOLD_STRONG_ADX else ATR_TRAIL_MULT
        gap = ind["atr"] * trail_mult
        if side=="long":
            new_trail = px - gap
            STATE["trail"] = max(STATE["trail"] or new_trail, new_trail)
            if STATE["breakeven"] is not None: STATE["trail"] = max(STATE["trail"], STATE["breakeven"])
            if px < STATE["trail"]: close_market_strict(f"TRAIL_ATR({trail_mult}x)")
        else:
            new_trail = px + gap
            STATE["trail"] = min(STATE["trail"] or new_trail, new_trail)
            if STATE["breakeven"] is not None: STATE["trail"] = min(STATE["trail"], STATE["breakeven"])
            if px > STATE["trail"]: close_market_strict(f"TRAIL_ATR({trail_mult}x)")

# =================== ORCHESTRA (post-entry smart brain + SMC) ===================
def _candle_stats(df):
    if len(df)<1: return {}
    o = float(df["open"].iloc[-1])
    h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1])
    c = float(df["close"].iloc[-1])
    rng = max(h-l, 1e-12); body = abs(c-o)
    upper = h - max(o,c); lower = min(o,c) - l
    return {
        "range":rng, "body":body,
        "body_pct": (body/rng)*100.0 if rng>0 else 0.0,
        "upper_pct": (upper/rng)*100.0 if rng>0 else 0.0,
        "lower_pct": (lower/rng)*100.0 if rng>0 else 0.0,
        "bull": c>o, "bear": c<o
    }

def _vol_ratio(df):
    if len(df)<21: return 1.0
    v = float(df["volume"].iloc[-1])
    vma = df["volume"].iloc[-21:-1].astype(float).mean()
    return (v/max(vma,1e-9)) if vma>0 else 1.0

def near_level(px, lvl, bps):
    try: return abs((px-lvl)/lvl)*10000.0 <= bps
    except Exception: return False

def orchestra_post_entry(df, ind, info):
    if not (ORCHESTRA_ENABLED and STATE["open"] and STATE["qty"]>0): return
    px   = info["price"]; entry = STATE["entry"]; side = STATE["side"]
    rr   = (px - entry)/entry*100*(1 if side=="long" else -1)
    atr  = float(ind.get("atr") or 0.0)
    adx  = float(ind.get("adx") or 0.0)
    rsi  = float(ind.get("rsi") or 50.0)
    cndl = _candle_stats(df)
    volx = _vol_ratio(df)
    df_closed = df.iloc[:-1] if len(df)>=2 else df.copy()

    # Build snapshots (بما فيها SMC)
    STATE["struct"] = build_structure_snapshot(df, ind)
    eqh = STATE["struct"]["eqh"]; eql = STATE["struct"]["eql"]; fvg = STATE["struct"]["fvg"]
    smc = STATE["struct"]["smc"] if STATE["struct"] else {}

    # ---- Traps/Wicks عند مستويات السيولة ----
    trap = None
    if atr>0 and cndl:
        if cndl["upper_pct"]>=TRAP_WICK_PCT and cndl["body_pct"]<=TRAP_BODY_MAX_PCT and volx>=VOL_SPIKE_X:
            if eqh and near_level(float(df["high"].iloc[-1]), eqh, LEVEL_PROX_BPS): trap="bear_trap_at_EQH"
            elif fvg and fvg.get("type")=="BEAR_FVG" and near_level(float(df["high"].iloc[-1]), fvg["bottom"], LEVEL_PROX_BPS): trap="bear_trap_at_FVG"
        if cndl["lower_pct"]>=TRAP_WICK_PCT and cndl["body_pct"]<=TRAP_BODY_MAX_PCT and volx>=VOL_SPIKE_X:
            if eql and near_level(float(df["low"].iloc[-1]), eql, LEVEL_PROX_BPS): trap="bull_trap_at_EQL"
            elif fvg and fvg.get("type")=="BULL_FVG" and near_level(float(df["low"].iloc[-1]), fvg["top"], LEVEL_PROX_BPS): trap="bull_trap_at_FVG"

    if trap:
        close_partial(0.20, f"TRAP {trap}")
        if rr >= BREAKEVEN_AFTER and STATE.get("breakeven") is None:
            STATE["breakeven"]=entry
        if atr>0:
            gap = atr * max(ATR_TRAIL_MULT, 1.4)
            if side=="long":  STATE["trail"] = max(STATE["trail"] or (px-gap), px-gap)
            else:             STATE["trail"] = min(STATE["trail"] or (px+gap), px+gap)

    # ---- Impulse/Wick harvesting ----
    if atr>0 and cndl.get("body",0) >= IMPULSE_X_ATR*atr:
        if (cndl["bull"] and side=="long") or (cndl["bear"] and side=="short"):
            frac = 0.50 if cndl["body"] >= 2.0*atr else 0.33
            close_partial(frac, f"IMPULSE x{cndl['body']/atr:.2f} ATR")
            gap = atr * TRAIL_WIDE_MULT
            if side=="long":  STATE["trail"] = max(STATE["trail"] or (px-gap), px-gap)
            else:             STATE["trail"] = min(STATE["trail"] or (px+gap), px+gap)
    if side=="long" and cndl.get("upper_pct",0)>=WICK_HARVEST_PCT:
        close_partial(0.25, f"UPPER-WICK {cndl['upper_pct']:.0f}%")
    if side=="short" and cndl.get("lower_pct",0)>=WICK_HARVEST_PCT:
        close_partial(0.25, f"LOWER-WICK {cndl['lower_pct']:.0f}%")

    # ---- HOLD vs REJECT votes near EQH/EQL/FVG ----
    targeted_level = None; tagged = None
    if side=="long":
        if eqh and near_level(px, eqh, LEVEL_PROX_BPS): targeted_level = eqh; tagged="EQH"
        elif fvg and fvg.get("type")=="BEAR_FVG" and near_level(px, fvg["bottom"], LEVEL_PROX_BPS): targeted_level=fvg["bottom"]; tagged="FVG"
    else:
        if eql and near_level(px, eql, LEVEL_PROX_BPS): targeted_level = eql; tagged="EQL"
        elif fvg and fvg.get("type")=="BULL_FVG" and near_level(px, fvg["top"], LEVEL_PROX_BPS): targeted_level=fvg["top"]; tagged="FVG"

    # أصوات استمرار الترند (ADX + RSI)
    if (side=="long" and adx>=HOLD_STRONG_ADX and rsi>=55) or (side=="short" and adx>=HOLD_STRONG_ADX and rsi<=45):
        STATE["hold_votes"] = min(STATE.get("hold_votes",0)+1, 3)
    else:
        STATE["hold_votes"] = max(0, STATE.get("hold_votes",0)-1)

    # رفضات عند مستوى
    if targeted_level:
        STATE["last_level_tag"] = tagged
        adx_soft = adx < HOLD_STRONG_ADX or True
        is_reject = False
        if side=="long":
            is_reject = (float(df["high"].iloc[-1])>targeted_level and px<targeted_level and cndl["upper_pct"]>=40.0 and volx>=VOL_SPIKE_X and adx_soft)
        else:
            is_reject = (float(df["low"].iloc[-1])<targeted_level and px>targeted_level and cndl["lower_pct"]>=40.0 and volx>=VOL_SPIKE_X and adx_soft)
        if is_reject:
            STATE["rejection_votes"] = min(STATE.get("rejection_votes",0)+1, 3)
            STATE["last_reject_bar_ts"] = int(df["time"].iloc[-1])
        if STATE["hold_votes"]>=2 and not is_reject:
            STATE["rejection_votes"] = max(0, STATE.get("rejection_votes",0)-1)

    # ---- SMC reinforcement ----
    if SMC_ENABLED and smc:
        bos = smc.get("bos"); choch = smc.get("choch")
        pd  = smc.get("pd")
        ob  = smc.get("ob") or {}
        bull_ob = ob.get("bull"); bear_ob = ob.get("bear")
        sweep   = smc.get("sweep")

        # Premium/Discount bias → أصوات HOLD لو مع الاتجاه
        if side=="long" and pd=="discount": STATE["hold_votes"] = min(STATE["hold_votes"]+1, 3)
        if side=="short" and pd=="premium": STATE["hold_votes"] = min(STATE["hold_votes"]+1, 3)

        # BOS في صالحنا → HOLD
        if (side=="long" and bos=="up") or (side=="short" and bos=="down"):
            STATE["hold_votes"] = min(STATE["hold_votes"]+1, 3)
        # CHOCH ضدنا → زيادة رفض/حذر
        if choch and ((side=="long" and bos=="down") or (side=="short" and bos=="up")):
            STATE["rejection_votes"] = min(STATE["rejection_votes"]+1, 3)
            if rr>0.25: close_partial(0.20, "CHOCH guard")

        # Order Block touch/mitigation:
        if side=="long" and bull_ob and _in_zone(px, bull_ob["low"], bull_ob["high"], OB_TOLERANCE_BPS):
            # تواجد داخل BULL_OB + ذيل سفلي/حجم ⇒ HOLD قوي
            if cndl.get("lower_pct",0)>=35.0 and volx>=1.05:
                STATE["hold_votes"] = min(STATE["hold_votes"]+1, 3)
        if side=="short" and bear_ob and _in_zone(px, bear_ob["low"], bear_ob["high"], OB_TOLERANCE_BPS):
            if cndl.get("upper_pct",0)>=35.0 and volx>=1.05:
                STATE["hold_votes"] = min(STATE["hold_votes"]+1, 3)

        # لو لمسنا OB المعاكس وظهر رفض قوي ⇒ جني جزء/حذر
        if side=="long" and bear_ob and _in_zone(px, bear_ob["low"], bear_ob["high"], OB_TOLERANCE_BPS) and cndl.get("upper_pct",0)>=40.0:
            close_partial(0.25, "Opposite OB reject")
            STATE["rejection_votes"] = min(STATE["rejection_votes"]+1, 3)
        if side=="short" and bull_ob and _in_zone(px, bull_ob["low"], bull_ob["high"], OB_TOLERANCE_BPS) and cndl.get("lower_pct",0)>=40.0:
            close_partial(0.25, "Opposite OB reject")
            STATE["rejection_votes"] = min(STATE["rejection_votes"]+1, 3)

        # Liquidity sweep إشارة خادعة → جزءي + تشديد تريل
        if sweep:
            close_partial(0.20, f"{sweep['type']}")
            if atr>0:
                gap = atr * max(1.4, ATR_TRAIL_MULT)
                if side=="long":  STATE["trail"] = max(STATE["trail"] or (px-gap), px-gap)
                else:             STATE["trail"] = min(STATE["trail"] or (px+gap), px+gap)

    # ---- Fake-RF guard ----
    if RF_FAKE_GUARD:
        flipped_against = (side=="long" and info.get("short")) or (side=="short" and info.get("long"))
        if flipped_against and rr >= 0.20:
            close_partial(0.20, "Opposite RF (guard partial)")
            if atr>0:
                gap = atr * max(1.4, ATR_TRAIL_MULT)
                if side=="long":  STATE["trail"] = max(STATE["trail"] or (px-gap), px-gap)
                else:             STATE["trail"] = min(STATE["trail"] or (px+gap), px+gap)
            STATE["confirm_queue"] = max(STATE.get("confirm_queue",0), CONFIRM_BARS)

    # ---- ONE-SHOT decision ----
    targeted_level = None
    if side=="long": targeted_level = eqh
    else:            targeted_level = eql
    last_conf_bar_ok = STATE.get("confirm_queue",0) == 0
    if targeted_level and STATE.get("rejection_votes",0) >= REJECT_NEED_VOTES and rr >= ONE_SHOT_MIN_PCT and last_conf_bar_ok:
        # تعزيز القرار بـ SMC: وجود Sweep/ضرب OB معاكس أو اختفاء الزخم
        extra_ok = True
        if SMC_ENABLED and smc:
            bos = smc.get("bos")
            if (side=="long" and bos=="down") or (side=="short" and bos=="up"):
                extra_ok = True
        if extra_ok:
            print(colored(f"💥 ONE-SHOT EXIT @{fmt(rr,2)}% — last level confirmed (votes={STATE['rejection_votes']})", "magenta"))
            close_market_strict("ONE_SHOT_LAST_LEVEL")
            return

    if STATE.get("confirm_queue",0) > 0:
        STATE["confirm_queue"] = max(0, STATE["confirm_queue"]-1)

# =================== LOOP / LOG ===================
def pretty_snapshot(bal, info, ind, spread_bps, reason=None, df=None):
    left_s = time_to_candle_close(df) if df is not None else 0
    print(colored("─"*112,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*112,"cyan"))
    print("📈 RF & INDICATORS")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))}  📶 spread={fmt(spread_bps,2)}bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}")
    print(f"   🎯 ENTRY MODE: RF-LIVE ONLY  •  ⏱️ closes_in ≈ {left_s}s")

    if ORCHESTRA_ENABLED:
        st = STATE.get("struct") or {}
        eqh, eql, fvg = st.get("eqh"), st.get("eql"), st.get("fvg")
        fvg_txt = f"FVG[{(fvg or {}).get('type')}]({fmt((fvg or {}).get('bottom'))}-{fmt((fvg or {}).get('top'))})" if fvg else "—"
        smc = (st.get("smc") or {}) if st else {}
        ob  = smc.get("ob") or {}
        bull_ob = ob.get("bull"); bear_ob = ob.get("bear")
        smc_line = f"SMC: BOS={smc.get('bos')}  CHOCH={smc.get('choch')}  PD={smc.get('pd')}  SWEEP={(smc.get('sweep') or {}).get('type') if smc.get('sweep') else '—'}"
        ob_line  = f"OB: BULL[{fmt((bull_ob or {}).get('low'))}-{fmt((bull_ob or {}).get('high'))}]  BEAR[{fmt((bear_ob or {}).get('low'))}-{fmt((bear_ob or {}).get('high'))}]"
        print(colored(f"   🧠 STRUCT: EQH={fmt(eqh)}  EQL={fmt(eql)}  {fvg_txt}", "yellow"))
        print(colored(f"   🧩 {smc_line}", "yellow"))
        print(colored(f"   🧱 {ob_line}", "yellow"))
        print(colored(f"   🗳️ Votes: HOLD={STATE.get('hold_votes',0)}  REJECT={STATE.get('rejection_votes',0)}  confirm_wait={STATE.get('confirm_queue',0)}  tag={STATE.get('last_level_tag')}", "yellow"))

    print("\n🧭 POSITION & PNL")
    bal_line = f"💰 Balance={fmt(bal,2)}  Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x  CompoundPnL={fmt(compound_pnl)}  Eq~{fmt((bal or 0)+compound_pnl,2)}"
    print(colored(f"   {bal_line}", "white"))
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        rr = ((info.get("price") or STATE["entry"])-STATE["entry"])/max(STATE["entry"],1e-9)*100*(1 if STATE["side"]=="long" else -1)
        print(f"   {lamp}  Entry={fmt(STATE['entry'])}  Qty={fmt(STATE['qty'],4)}  Bars={STATE['bars']}  Trail={fmt(STATE['trail'])}  BE={fmt(STATE['breakeven'])}  RR%={fmt(rr,2)}")
        print(f"   🎯 TP_done={STATE['profit_targets_achieved']}  HP={fmt(STATE['highest_profit_pct'],2)}%  final_chunk={FINAL_CHUNK_QTY} DOGE")
    else:
        print("   ⚪ FLAT")
        if wait_for_next_signal_side:
            print(colored(f"   ⏳ Waiting for opposite RF: {wait_for_next_signal_side.upper()}", "cyan"))
    if reason: print(colored(f"   ℹ️ reason: {reason}", "white"))
    print(colored("─"*112,"cyan"))

def trade_loop():
    global wait_for_next_signal_side
    loop_i=0
    while True:
        try:
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            info = rf_signal_live(df)             # RF live
            ind  = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]

            manage_after_entry(df, ind, {"price": px or info["price"], **info})
            orchestra_post_entry(df, ind, {"price": px or info["price"], **info})

            reason=None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"

            sig = "buy" if (ENTRY_RF_ONLY and info["long"]) else ("sell" if (ENTRY_RF_ONLY and info["short"]) else None)

            if not STATE["open"] and sig and reason is None:
                if wait_for_next_signal_side and sig != wait_for_next_signal_side:
                    reason=f"waiting opposite RF: need {wait_for_next_signal_side.upper()}"
                else:
                    qty = compute_size(bal, px or info["price"])
                    if qty>0:
                        ok = open_market(sig, qty, px or info["price"])
                        if ok:
                            wait_for_next_signal_side = None
                    else:
                        reason="qty<=0"

            pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, reason, df)
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP
            time.sleep(sleep_s)
        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}", "red"))
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# =================== API / KEEPALIVE ===================
app = Flask(__name__)
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ RF-LIVE Bot — {SYMBOL} {INTERVAL} — {mode} — Entry: RF LIVE only — Dynamic TP — Strict Close — ORCHESTRA+SMC ON"

@app.route("/metrics")
def metrics():
    st = STATE.get("struct") or {}
    smc = st.get("smc") or {}
    ob  = smc.get("ob") or {}
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": {k: (v if k!='struct' else {
            "eqh": st.get("eqh"), "eql": st.get("eql"),
            "fvg": st.get("fvg"), "smc": {
                "bos": smc.get("bos"), "choch": smc.get("choch"), "pd": smc.get("pd"),
                "sweep": smc.get("sweep"), "ob": ob
            }}) for k,v in STATE.items()},
        "compound_pnl": compound_pnl,
        "entry_mode": "RF_LIVE_ONLY", "wait_for_next_signal": wait_for_next_signal_side,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "final_chunk_qty": FINAL_CHUNK_QTY},
        "orchestra_enabled": ORCHESTRA_ENABLED, "smc_enabled": SMC_ENABLED
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "entry_mode": "RF_LIVE_ONLY", "wait_for_next_signal": wait_for_next_signal_side,
        "orchestra_enabled": ORCHESTRA_ENABLED, "smc_enabled": SMC_ENABLED
    }), 200

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (SELF_URL not set)", "yellow"))
        return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"rf-live-bot/keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}", "cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# =================== BOOT ===================
if __name__ == "__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  RF_LIVE={RF_LIVE_ONLY}", "yellow"))
    print(colored(f"ENTRY: RF ONLY  •  FINAL_CHUNK_QTY={FINAL_CHUNK_QTY}  •  ORCHESTRA+SMC={'ON' if (ORCHESTRA_ENABLED and SMC_ENABLED) else 'OFF'}", "yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
