# -*- coding: utf-8 -*-
"""
RF Futures Bot — Smart Pro (BingX Perp, CCXT) - HARDENED EDITION
(UNIFIED bot.py WITH Smart Alpha Pack + Patience Guard & Smart Harvest v2
and reliability patches per user spec; core trading logic preserved)
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

# ------------ ENV ------------
API_KEY = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE = bool(API_KEY and API_SECRET)

# ------------ Hard-coded Settings ------------
SYMBOL = "DOGE/USDT:USDT"
INTERVAL = "15m"
LEVERAGE = 10
RISK_ALLOC = 0.60
SPREAD_GUARD_BPS = 6
COOLDOWN_AFTER_CLOSE_BARS = 0

RF_SOURCE = "close"
RF_PERIOD = 20
RF_MULT = 3.5
USE_TV_BAR = True

RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

STRATEGY = "smart"
USE_SMART_EXIT = True

TP1_PCT = 0.40
TP1_CLOSE_FRAC = 0.50
BREAKEVEN_AFTER = 0.30
TRAIL_ACTIVATE = 0.60
ATR_MULT_TRAIL = 1.6

SCALE_IN_MAX_STEPS = 0
SCALE_IN_STEP_PCT = 0.20
ADX_STRONG_THRESH = 28
RSI_TREND_BUY = 55
RSI_TREND_SELL = 45
TRAIL_MULT_STRONG = 2.0
TRAIL_MULT_MED = 1.5
TRAIL_MULT_CHOP = 1.0

ADX_TIER1 = 28
ADX_TIER2 = 35
ADX_TIER3 = 45
RATCHET_LOCK_PCT = 0.60

BINGX_POSITION_MODE = "oneway"

SCALP_TARGETS = [0.35, 0.70, 1.20]
SCALP_CLOSE_FRACS = [0.40, 0.30, 0.30]

TREND_TARGETS = [0.50, 1.00, 1.80]
TREND_CLOSE_FRACS = [0.30, 0.30, 0.20]
MIN_TREND_HOLD_ADX = 25
END_TREND_ADX_DROP = 5.0
END_TREND_RSI_NEUTRAL = (45, 55)
DI_FLIP_BUFFER = 1.0

IMPULSE_HARVEST_THRESHOLD = 1.2
LONG_WICK_HARVEST_THRESHOLD = 0.60
RATCHET_RETRACE_THRESHOLD = 0.40

BREAKOUT_ATR_SPIKE = 1.8
BREAKOUT_ADX_THRESHOLD = 25
BREAKOUT_LOOKBACK_BARS = 20
BREAKOUT_CALM_THRESHOLD = 1.1

EMERGENCY_PROTECTION_ENABLED = True
EMERGENCY_ADX_MIN = 40
EMERGENCY_ATR_SPIKE_RATIO = 1.6
EMERGENCY_RSI_PUMP = 72
EMERGENCY_RSI_CRASH = 28
EMERGENCY_POLICY = "tp_then_close"
EMERGENCY_HARVEST_FRAC = 0.60
EMERGENCY_FULL_CLOSE_PROFIT = 1.0
EMERGENCY_TRAIL_ATR_MULT = 1.2

BREAKOUT_CONFIRM_BARS = 1
BREAKOUT_HARD_ATR_SPIKE = 1.8
BREAKOUT_SOFT_ATR_SPIKE = 1.4
BREAKOUT_VOLUME_SPIKE = 1.3
BREAKOUT_VOLUME_MED = 1.1

TP0_PROFIT_PCT = 0.2
TP0_CLOSE_FRAC = 0.10
TP0_MAX_USDT = 1.0

THRUST_ATR_BARS = globals().get("THRUST_ATR_BARS", 3)
THRUST_VOLUME_FACTOR = globals().get("THRUST_VOLUME_FACTOR", 1.3)
CHANDELIER_ATR_MULT = globals().get("CHANDELIER_ATR_MULT", 3.0)
CHANDELIER_LOOKBACK = globals().get("CHANDELIER_LOOKBACK", 20)

TRAIL_MULT_STRONG_ALPHA = 2.4
TRAIL_MULT_CAUTIOUS_ALPHA = 2.0
TRAIL_MULT_SCALP_ALPHA = 1.6

ADAPTIVE_PACING = True
BASE_SLEEP = 10
NEAR_CLOSE_SLEEP = 1
JUST_CLOSED_WINDOW = 8

SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
KEEPALIVE_SECONDS = 50
PORT = int(os.getenv("PORT", 5000))

STRICT_EXCHANGE_CLOSE = True
CLOSE_RETRY_ATTEMPTS   = 6
CLOSE_VERIFY_WAIT_S    = 2.0
MIN_RESIDUAL_TO_FORCE  = 1.0

REQUIRE_NEW_BAR_AFTER_CLOSE = False
wait_for_next_signal_side = None
last_close_signal_time = None

STATE_FILE = "bot_state.json"
_consec_err = 0
last_open_fingerprint = None
last_loop_ts = time.time()

# ---- Smart Harvest defaults (added if not defined) ----
TP1_PCT_SCALED = globals().get("TP1_PCT_SCALED", 0.004)
TP2_PCT_SCALED = globals().get("TP2_PCT_SCALED", 0.008)
TP1_FRAC       = globals().get("TP1_FRAC", 0.25)
TP2_FRAC       = globals().get("TP2_FRAC", 0.35)

# ---- Patience Guard defaults ----
ENABLE_PATIENCE       = globals().get("ENABLE_PATIENCE", True)
PATIENCE_MIN_BARS     = globals().get("PATIENCE_MIN_BARS", 2)
PATIENCE_NEED_CONSENSUS = globals().get("PATIENCE_NEED_CONSENSUS", 2)
REV_ADX_DROP          = globals().get("REV_ADX_DROP", 3.0)
REV_RSI_LEVEL         = globals().get("REV_RSI_LEVEL", 50)
REV_RF_CROSS          = globals().get("REV_RF_CROSS", True)

# ---- Exit Protection Layer ----
MIN_HOLD_BARS        = 3        # أقل عدد شموع لازم يفضل فيها قبل أي إغلاق "اختياري"
MIN_HOLD_SECONDS     = 120      # أو زمن حد أدنى (احتياطي)
TRAIL_ONLY_AFTER_TP1 = True     # فعل التريل بعد TP1 فقط (إلا لو ضرب Chand/Thrust)
RF_HYSTERESIS_BPS    = 8        # بوفر للهسترة لما يحصل عكس مؤقت مع Range Filter
NO_FULL_CLOSE_BEFORE_TP1 = True # ممنوع قفل كامل قبل TP1 (يسمح بجزئي فقط)
WICK_MAX_BEFORE_TP1  = 0.25     # أقصى إغلاق جزئي مسموح من حصاد الذيل قبل TP1

# Optional debug flags (no change to existing logs)
DEBUG_PATIENCE = globals().get("DEBUG_PATIENCE", False)
DEBUG_HARVEST  = globals().get("DEBUG_HARVEST", False)
DEBUG_EXIT_GUARD = globals().get("DEBUG_EXIT_GUARD", False)

print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • SYMBOL={SYMBOL} • {INTERVAL}", "yellow"))
print(colored(f"STRATEGY: {STRATEGY.upper()} • SMART_EXIT={'ON' if USE_SMART_EXIT else 'OFF'}", "yellow"))
print(colored(f"EXIT PROTECTION: MIN_HOLD={MIN_HOLD_BARS} bars, {MIN_HOLD_SECONDS}s • RF_HYSTERESIS={RF_HYSTERESIS_BPS}bps", "cyan"))

# ------------ File Logging with Rotation (IDEMPOTENT) ------------
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ file logging with rotation enabled (idempotent)", "cyan"))

setup_file_logging()

# ------------ Graceful Exit ------------
_state_lock = threading.Lock()

def _graceful_exit(signum, frame):
    print(colored(f"🛑 signal {signum} → saving state & exiting", "red"))
    save_state()
    sys.exit(0)

signal.signal(signal.SIGTERM, _graceful_exit)
signal.signal(signal.SIGINT,  _graceful_exit)

# ------------ Exchange ------------
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap"}
    })

ex = make_exchange()

# ------------ Market Specs & Amount Normalization ------------
MARKET = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN = None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        MARKET = ex.markets.get(SYMBOL, {})
        amt_prec = MARKET.get("precision", {}).get("amount", 0)
        try:
            AMT_PREC = int(amt_prec)
        except (TypeError, ValueError):
            AMT_PREC = 0
        LOT_STEP = (MARKET.get("limits", {}).get("amount", {}) or {}).get("step", None)
        LOT_MIN = (MARKET.get("limits", {}).get("amount", {}) or {}).get("min", None)
        print(colored(f"📊 Market specs: precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}", "yellow"))

# ---- Decimal rounding (safe) ----
from decimal import Decimal, ROUND_DOWN, InvalidOperation

def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP, (int, float)) and LOT_STEP > 0:
            step = Decimal(str(LOT_STEP))
            d = (d / step).to_integral_value(rounding=ROUND_DOWN) * step
        prec = int(AMT_PREC) if AMT_PREC and AMT_PREC > 0 else 0
        if prec >= 0:
            d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN, (int, float)) and LOT_MIN > 0:
            if d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except (InvalidOperation, ValueError, TypeError):
        return max(0.0, float(q))

def safe_qty(q):
    q = _round_amt(q)
    if q <= 0:
        print(colored(f"⚠️ qty invalid after normalize → {q}", "yellow"))
    return q

def ensure_leverage_and_mode():
    try:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            print(colored(f"✅ leverage set: {LEVERAGE}x with side=BOTH", "green"))
        except Exception as e:
            print(colored(f"⚠️ set_leverage warn: {e}", "yellow"))
        print(colored(f"ℹ️ position mode target: {BINGX_POSITION_MODE}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_leverage_and_mode: {e}", "yellow"))

try:
    ex.load_markets()
    load_market_specs()
    ensure_leverage_and_mode()
except Exception as e:
    print(colored(f"⚠️ load_markets: {e}", "yellow"))

# ------------ State Persistence (ATOMIC) ------------
def save_state():
    try:
        data = {
            "state": state,
            "compound_pnl": compound_pnl,
            "last_signal_id": last_signal_id,
            "timestamp": time.time()
        }
        tmp = STATE_FILE + ".tmp"
        with _state_lock:
            with open(tmp, "w", encoding="utf-8") as f:
                import json
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, STATE_FILE)
        logging.info(f"State saved atomically: compound_pnl={compound_pnl}")
    except Exception as e:
        print(colored(f"⚠️ save_state: {e}", "yellow"))
        logging.error(f"save_state error: {e}")

def load_state():
    global state, compound_pnl, last_signal_id
    try:
        import json, os
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            state.update(data.get("state", {}))
            compound_pnl = data.get("compound_pnl", 0.0)
            ls = data.get("last_signal_id")
            if ls:
                globals()["last_signal_id"] = ls
            print(colored("✅ state restored from disk", "green"))
            logging.info(f"State loaded: compound_pnl={compound_pnl}, open={state['open']}")
    except Exception as e:
        print(colored(f"⚠️ load_state: {e}", "yellow"))
        logging.error(f"load_state error: {e}")

def net_guard(success):
    global _consec_err
    if success:
        _consec_err = 0
    else:
        _consec_err += 1
        if _consec_err in (3, 5, 8):
            wait = min(60, 5 * _consec_err)
            print(colored(f"🌐 network backoff: {_consec_err} errors → sleep {wait}s", "yellow"))
            logging.warning(f"Network backoff: {_consec_err} errors, sleeping {wait}s")
            time.sleep(wait)

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

def sanity_check_bar_clock(df):
    try:
        if len(df) < 2: return
        tf = _interval_seconds(INTERVAL)
        delta = (int(df["time"].iloc[-1]) - int(df["time"].iloc[-2]))/1000
        if abs(delta - tf) > tf*0.5:
            print(colored(f"⚠️ bar interval anomaly: {delta}s vs tf {tf}s", "yellow"))
            logging.warning(f"Bar interval anomaly: {delta}s vs tf {tf}s")
    except Exception as e:
        logging.error(f"sanity_check_bar_clock error: {e}")

# ------------ Idempotency Guard (improved) ------------
def can_open(sig, price):
    if not state.get("open"):
        return True
    global last_open_fingerprint
    try:
        pkey = f"{float(price or 0):.6f}"
    except Exception:
        pkey = str(price)
    fp = f"{sig}|{pkey}|{INTERVAL}|{SYMBOL}"
    if fp == last_open_fingerprint:
        return False
    last_open_fingerprint = fp
    return True

# ------------ Helpers ------------
def fmt(v, d=6, na="N/A"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception: return na

def with_retry(fn, attempts=3, base_wait=0.4):
    for i in range(attempts):
        try: 
            result = fn()
            net_guard(True)
            return result
        except Exception:
            net_guard(False)
            if i == attempts-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.2)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception as e:
        print(colored(f"❌ ticker: {e}", "red")); return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception as e:
        print(colored(f"❌ balance: {e}", "red")); return None

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
    if not ADAPTIVE_PACING:
        return BASE_SLEEP
    try:
        left_s = time_to_candle_close(df, USE_TV_BAR)
        tf = _interval_seconds(INTERVAL)
        if left_s <= 10:
            return NEAR_CLOSE_SLEEP
        if (tf - left_s) <= JUST_CLOSED_WINDOW:
            return NEAR_CLOSE_SLEEP
        return BASE_SLEEP
    except Exception:
        return BASE_SLEEP

# ------------ Indicators ------------
def wilder_ema(s: pd.Series, n: int): return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {"rsi": 50.0, "plus_di": 0.0, "minus_di": 0.0, "dx": 0.0,
                "adx": 0.0, "atr": 0.0, "adx_prev": 0.0}
    c, h, l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    delta = c.diff()
    up = delta.clip(lower=0.0); dn = (-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0, 1e-12)
    rsi = 100 - (100 / (1+rs))

    up_move = h.diff(); down_move = l.shift(1) - l
    plus_dm  = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    plus_di  = 100 * (wilder_ema(plus_dm, ADX_LEN) / atr.replace(0,1e-12))
    minus_di = 100 * (wilder_ema(minus_dm, ADX_LEN) / atr.replace(0,1e-12))
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0,1e-12)).fillna(0.0)
    adx = wilder_ema(dx, ADX_LEN)

    i = len(df)-1
    prev_i = max(0, i-1)
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i]),
        "adx_prev": float(adx.iloc[prev_i])
    }

# ------------ Range Filter (EXACT) ------------
def _ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()
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
        return {
            "time": int(df["time"].iloc[i]) if len(df) else int(time.time()*1000),
            "price": price or 0.0, "long": False, "short": False,
            "filter": price or 0.0, "hi": price or 0.0, "lo": price or 0.0, "fdir": 0.0
        }
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    dfilt = filt - filt.shift(1)
    fdir = pd.Series(0.0, index=filt.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0.0)
    upward = (fdir==1).astype(int)
    downward = (fdir == -1).astype(int)
    src_gt_f=(src>filt); src_lt_f=(src<filt); src_gt_p=(src>src.shift(1)); src_lt_p=(src<src.shift(1))
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
        "time": int(df["time"].iloc[i]), "price": float(df["close"].iloc[i]),
        "long": bool(longSignal.iloc[i]), "short": bool(shortSignal.iloc[i]),
        "filter": float(filt.iloc[i]), "hi": float(hi.iloc[i]), "lo": float(lo.iloc[i]),
        "fdir": float(fdir.iloc[i])
    }

# ------------ State & Sync ------------
state={
    "open": False, "side": None, "entry": None, "qty": 0.0, 
    "pnl": 0.0, "bars": 0, "trail": None, "tp1_done": False, 
    "breakeven": None, "scale_ins": 0, "scale_outs": 0,
    "last_action": None, "action_reason": None,
    "highest_profit_pct": 0.0,
    "trade_mode": None,
    "profit_targets_achieved": 0,
    "entry_time": None,
    "fakeout_pending": False,
    "fakeout_need_side": None,
    "fakeout_confirm_bars": 0,
    "fakeout_started_at": None,
    "breakout_active": False,
    "breakout_direction": None,
    "breakout_entry_price": None,
    "_tp0_done": False,
    "thrust_locked": False,
    "breakout_score": 0.0,
    "breakout_votes_detail": {},
    "opened_by_breakout": False,
    "opened_at": None,
    "_exit_soft_block": False
}
compound_pnl = 0.0
last_signal_id = None
post_close_cooldown = 0

def compute_size(balance, price):
    effective_balance = (balance or 0.0) + (compound_pnl or 0.0)
    capital = effective_balance * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def sync_from_exchange_once():
    try:
        poss = ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym = p.get("symbol") or p.get("info",{}).get("symbol") or ""
            if SYMBOL.split(":")[0] not in sym:
                continue
            qty = abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty<=0: continue
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            if side not in ("long","short"):
                cost = float(p.get("cost") or 0.0)
                side = "long" if cost>0 else "short"
            state.update({
                "open": True, "side": side, "entry": entry, "qty": qty, 
                "pnl": 0.0, "bars": 0, "trail": None, "tp1_done": False, 
                "breakeven": None, "scale_ins": 0, "scale_outs": 0,
                "last_action": "SYNC", "action_reason": "Position synced from exchange",
                "highest_profit_pct": 0.0,
                "trade_mode": None,
                "profit_targets_achieved": 0,
                "entry_time": time.time(),
                "fakeout_pending": False,
                "fakeout_need_side": None,
                "fakeout_confirm_bars": 0,
                "fakeout_started_at": None,
                "breakout_active": False,
                "breakout_direction": None,
                "breakout_entry_price": None,
                "_tp0_done": False,
                "thrust_locked": False,
                "breakout_score": 0.0,
                "breakout_votes_detail": {},
                "opened_by_breakout": False,
                "opened_at": time.time(),
                "_exit_soft_block": False
            })
            print(colored(f"✅ Synced position ⇒ {side.upper()} qty={fmt(qty,4)} @ {fmt(entry)}","green"))
            logging.info(f"Position synced: {side} qty={qty} entry={entry}")
            return
        print(colored("↔️  Sync: no open position on exchange.","yellow"))
    except Exception as e:
        print(colored(f"❌ sync error: {e}","red"))
        logging.error(f"sync_from_exchange_once error: {e}")

def _position_params_for_open(side: str):
    if BINGX_POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if side == "buy" else "SHORT", "reduceOnly": False}
    return {"positionSide": "BOTH", "reduceOnly": False}

def _position_params_for_close():
    if BINGX_POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if state.get("side")=="long" else "SHORT", "reduceOnly": True}
    return {"positionSide": "BOTH", "reduceOnly": True}

def _read_exchange_position():
    try:
        poss = ex.fetch_positions(params={"type": "swap"})
        for p in poss:
            sym = (p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0] not in sym:
                continue
            qty = abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty <= 0:
                return 0.0, None, None
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side_raw = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side = "long" if ("long" in side_raw or float(p.get("cost",0)) > 0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_exchange_position error: {e}")
    return 0.0, None, None

def close_market_strict(reason):
    global state, compound_pnl, wait_for_next_signal_side, last_close_signal_time
    prev_side_local = state.get("side")
    exch_qty, exch_side, exch_entry = _read_exchange_position()
    if exch_qty <= 0:
        if state.get("open"):
            reset_after_full_close("strict_close_already_zero", prev_side_local)
        return
    side_to_close = "sell" if (exch_side == "long") else "buy"
    qty_to_close  = safe_qty(exch_qty)
    attempts = 0
    last_error = None
    while attempts < CLOSE_RETRY_ATTEMPTS:
        try:
            if MODE_LIVE:
                params = _position_params_for_close()
                params["reduceOnly"] = True
                ex.create_order(SYMBOL, "market", side_to_close, qty_to_close, None, params)
            time.sleep(CLOSE_VERIFY_WAIT_S)
            left_qty, _, _ = _read_exchange_position()
            if left_qty <= 0:
                px = price_now() or state.get("entry")
                entry_px = state.get("entry") or exch_entry or px
                side = state.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
                qty = exch_qty
                pnl = (px - entry_px) * qty * (1 if side == "long" else -1)
                compound_pnl += pnl
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                logging.info(f"STRICT_CLOSE {side} pnl={pnl} total={compound_pnl}")
                reset_after_full_close(reason, prev_side_local)
                return
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            print(colored(f"⚠️ strict close retry {attempts}/{CLOSE_RETRY_ATTEMPTS} — residual={fmt(left_qty,4)}","yellow"))
            if qty_to_close < MIN_RESIDUAL_TO_FORCE:
                time.sleep(CLOSE_VERIFY_WAIT_S)
        except Exception as e:
            last_error = e
            logging.error(f"close_market_strict error attempt {attempts+1}: {e}")
            attempts += 1
            time.sleep(CLOSE_VERIFY_WAIT_S)
    print(colored(f"❌ STRICT CLOSE FAILED after {CLOSE_RETRY_ATTEMPTS} attempts — manual check needed. Last error: {last_error}", "red"))
    logging.critical(f"STRICT CLOSE FAILED — last_error={last_error}")

def sync_consistency_guard():
    if not state["open"]:
        return
    exch_qty, exch_side, exch_entry = _read_exchange_position()
    if exch_qty <= 0 and state["open"]:
        print(colored("🛠️  CONSISTENCY GUARD: Exchange shows no position but locally open → resetting", "yellow"))
        logging.warning("Consistency guard: resetting local state (exchange shows no position)")
        reset_after_full_close("consistency_guard_no_position", state.get("side"))
        return
    if exch_qty > 0 and state["open"]:
        diff_pct = abs(exch_qty - state["qty"]) / max(exch_qty, state["qty"])
        if diff_pct > 0.1:
            print(colored(f"🛠️  CONSISTENCY GUARD: Quantity mismatch local={state['qty']} vs exchange={exch_qty} → syncing", "yellow"))
            logging.warning(f"Consistency guard: quantity mismatch local={state['qty']} exchange={exch_qty}")
            state["qty"] = exch_qty
            state["entry"] = exch_entry or state["entry"]
            save_state()

def get_dynamic_scale_in_step(adx: float) -> tuple:
    if adx >= ADX_TIER3:
        return 0.25, f"ADX-tier3 step=25% (ADX≥{ADX_TIER3})"
    elif adx >= ADX_TIER2:
        return 0.20, f"ADX-tier2 step=20% (ADX≥{ADX_TIER2})"
    elif adx >= ADX_TIER1:
        return 0.15, f"ADX-tier1 step=15% (ADX≥{ADX_TIER1})"
    else:
        return 0.0, f"ADX below tier1 (ADX<{ADX_TIER1})"

def get_dynamic_tp_params(adx: float) -> tuple:
    if adx >= ADX_TIER3:
        return 2.2, 0.7
    elif adx >= ADX_TIER2:
        return 1.8, 0.8
    else:
        return 1.0, 1.0

def trend_end_confirmed(ind: dict, candle_info: dict, info: dict) -> bool:
    adx = float(ind.get("adx") or 0.0)
    adx_prev = float(ind.get("adx_prev") or adx)
    plus_di = float(ind.get("plus_di") or 0.0)
    minus_di = float(ind.get("minus_di") or 0.0)
    rsi = float(ind.get("rsi") or 50.0)
    adx_weak = (adx_prev - adx) >= END_TREND_ADX_DROP or adx < 20
    if state.get("side") == "long":
        di_flip = (minus_di - plus_di) > DI_FLIP_BUFFER
    else:
        di_flip = (plus_di - minus_di) > DI_FLIP_BUFFER
    rsi_neutral = END_TREND_RSI_NEUTRAL[0] <= rsi <= END_TREND_RSI_NEUTRAL[1]
    rf_opposite = (state.get("side") == "long" and info.get("short")) or \
                  (state.get("side") == "short" and info.get("long"))
    return adx_weak or di_flip or rsi_neutral or rf_opposite

def check_trend_confirmation(candle_info: dict, ind: dict, current_side: str) -> str:
    try:
        pattern = candle_info.get("pattern", "NONE")
        adx = float(ind.get("adx") or 0)
        plus_di = float(ind.get("plus_di") or 0)
        minus_di = float(ind.get("minus_di") or 0)
        rsi = float(ind.get("rsi") or 50)
        reversal_patterns = ["DOJI", "HAMMER", "SHOOTING_STAR", "EVENING_STAR", "MORNING_STAR"]
        if pattern in reversal_patterns:
            if adx >= 25:
                if current_side == "long" and plus_di > minus_di and rsi >= RSI_TREND_BUY:
                    return "CONFIRMED_CONTINUE"
                elif current_side == "short" and minus_di > plus_di and rsi <= RSI_TREND_SELL:
                    return "CONFIRMED_CONTINUE"
                else:
                    if current_side == "long" and minus_di > plus_di and rsi < 45:
                        return "CONFIRMED_REVERSAL"
                    elif current_side == "short" and plus_di > minus_di and rsi > 55:
                        return "CONFIRMED_REVERSAL"
                    else:
                        return "POSSIBLE_FAKEOUT"
            elif adx < 25:
                return "POSSIBLE_FAKEOUT"
            else:
                return "NO_SIGNAL"
        if adx >= 30:
            if current_side == "long" and plus_di > minus_di and rsi >= RSI_TREND_BUY:
                return "CONFIRMED_CONTINUE"
            elif current_side == "short" and minus_di > plus_di and rsi <= RSI_TREND_SELL:
                return "CONFIRMED_CONTINUE"
        return "NO_SIGNAL"
    except Exception as e:
        print(colored(f"⚠️ check_trend_confirmation error: {e}", "yellow"))
        return "NO_SIGNAL"

def should_scale_in(candle_info: dict, ind: dict, current_side: str) -> tuple:
    return False, 0.0, "Scale-in disabled"

def should_scale_out(candle_info: dict, ind: dict, current_side: str) -> tuple:
    if state["qty"] <= 0:
        return False, "No position to scale out"
    trend_signal = check_trend_confirmation(candle_info, ind, current_side)
    if trend_signal == "CONFIRMED_REVERSAL":
        return True, f"Confirmed reversal: {candle_info.get('name_en', 'NONE')}"
    adx = ind.get("adx", 0)
    rsi = ind.get("rsi", 50)
    if (current_side == "long" and candle_info.get("pattern") in ["SHOOTING_STAR", "EVENING_STAR"]) or \
       (current_side == "short" and candle_info.get("pattern") in ["HAMMER", "MORNING_STAR"]):
        return True, f"Warning pattern: {candle_info.get('name_en')}"
    if adx > 30 and ind.get("adx_prev", 0) > adx + 2:
        return True, f"ADX weakening: {ind.get('adx_prev', 0):.1f} → {adx:.1f}"
    if current_side == "long" and rsi > 75:
        return True, f"RSI overbought: {rsi:.1f}"
    if current_side == "short" and rsi < 25:
        return True, f"RSI oversold: {rsi:.1f}"
    return False, "No strong scale-out signal"

def get_trail_multiplier(ind: dict) -> float:
    adx = ind.get("adx", 0)
    atr_pct = (ind.get("atr", 0) / (ind.get("price", 1) or 1)) * 100
    if adx >= 30 and atr_pct > 1.0:
        return TRAIL_MULT_STRONG
    elif adx >= 20:
        return TRAIL_MULT_MED
    else:
        return TRAIL_MULT_CHOP

# ----- SMART ALPHA PACK pieces (unchanged core; kept as in original) -----
def breakout_votes(df: pd.DataFrame, ind: dict, prev_ind: dict) -> tuple:
    votes = 0.0
    vote_details = {}
    try:
        if len(df) < max(BREAKOUT_LOOKBACK_BARS, 20) + 2:
            return 0.0, {"error": "Insufficient data"}
        current_idx = -1
        price = float(df["close"].iloc[current_idx])
        atr_now = float(ind.get("atr") or 0.0)
        atr_prev = float(prev_ind.get("atr") or atr_now)
        if atr_prev > 0:
            atr_ratio = atr_now / atr_prev
            if atr_ratio >= BREAKOUT_HARD_ATR_SPIKE:
                votes += 1.0; vote_details["atr_spike"] = f"Hard ({atr_ratio:.2f}x)"
            elif atr_ratio >= BREAKOUT_SOFT_ATR_SPIKE:
                votes += 0.5; vote_details["atr_spike"] = f"Soft ({atr_ratio:.2f}x)"
            else:
                vote_details["atr_spike"] = f"Normal ({atr_ratio:.2f}x)"
        adx = float(ind.get("adx") or 0.0)
        adx_prev = float(prev_ind.get("adx") or adx)
        if adx >= BREAKOUT_ADX_THRESHOLD:
            votes += 1.0; vote_details["adx"] = f"Strong ({adx:.1f})"
        elif adx >= 18 and adx > adx_prev:
            votes += 0.5; vote_details["adx"] = f"Building ({adx:.1f})"
        else:
            vote_details["adx"] = f"Weak ({adx:.1f})"
        if len(df) >= 21:
            current_volume = float(df["volume"].iloc[current_idx])
            volume_ma = df["volume"].iloc[-21:-1].astype(float).mean()
            if volume_ma > 0:
                volume_ratio = current_volume / volume_ma
                if volume_ratio >= BREAKOUT_VOLUME_SPIKE:
                    votes += 1.0; vote_details["volume"] = f"Spike ({volume_ratio:.2f}x)"
                elif volume_ratio >= BREAKOUT_VOLUME_MED:
                    votes += 0.5; vote_details["volume"] = f"High ({volume_ratio:.2f}x)"
                else:
                    vote_details["volume"] = f"Normal ({volume_ratio:.2f}x)"
        recent_highs = df["high"].iloc[-BREAKOUT_LOOKBACK_BARS:-1].astype(float)
        recent_lows = df["low"].iloc[-BREAKOUT_LOOKBACK_BARS:-1].astype(float)
        if len(recent_highs) > 0 and len(recent_lows) > 0:
            highest_high = recent_highs.max()
            lowest_low = recent_lows.min()
            if price > highest_high:
                votes += 1.0; vote_details["price_break"] = f"New High (>{highest_high:.6f})"
            elif price > recent_highs.iloc[-10:].max() if len(recent_highs) >= 10 else price > highest_high:
                votes += 0.5; vote_details["price_break"] = f"Near High"
            elif price < lowest_low:
                votes += 1.0; vote_details["price_break"] = f"New Low (<{lowest_low:.6f})"
            elif price < recent_lows.iloc[-10:].min() if len(recent_lows) >= 10 else price < lowest_low:
                votes += 0.5; vote_details["price_break"] = f"Near Low"
            else:
                vote_details["price_break"] = "Within Range"
        rsi = float(ind.get("rsi") or 50.0)
        if rsi >= 60:
            votes += 0.5; vote_details["rsi"] = f"Strong Bull ({rsi:.1f})"
        elif rsi <= 40:
            votes += 0.5; vote_details["rsi"] = f"Strong Bear ({rsi:.1f})"
        else:
            vote_details["rsi"] = f"Neutral ({rsi:.1f})"
        vote_details["total_score"] = f"{votes:.1f}/5.0"
    except Exception as e:
        print(colored(f"⚠️ breakout_votes error: {e}", "yellow"))
        logging.error(f"breakout_votes error: {e}")
        vote_details["error"] = str(e)
    return votes, vote_details

def tp0_quick_cash(ind: dict) -> bool:
    if not state["open"] or state["qty"] <= 0 or state.get("_tp0_done", False):
        return False
    try:
        price = ind.get("price") or price_now() or state["entry"]
        entry = state["entry"]; side = state["side"]
        if not (price and entry):
            return False
        rr = (price - entry) / entry * 100 * (1 if side == "long" else -1)
        if rr >= TP0_PROFIT_PCT:
            usdt_value = state["qty"] * price
            close_frac = min(TP0_CLOSE_FRAC, TP0_MAX_USDT / usdt_value) if usdt_value > 0 else TP0_CLOSE_FRAC
            if close_frac > 0:
                close_partial(close_frac, f"TP0 Quick Cash @ {rr:.2f}%")
                state["_tp0_done"] = True
                print(colored(f"💰 TP0: Quick cash taken at {rr:.2f}% profit", "cyan"))
                logging.info(f"TP0_QUICK_CASH: {rr:.2f}% profit, closed {close_frac*100:.1f}%")
                return True
    except Exception as e:
        print(colored(f"⚠️ tp0_quick_cash error: {e}", "yellow"))
        logging.error(f"tp0_quick_cash error: {e}")
    return False

# ------------ Thrust Lock (IMPROVED with guards) ------------
def thrust_lock(df: pd.DataFrame, ind: dict) -> bool:
    if not state["open"] or state["qty"] <= 0 or state.get("thrust_locked", False):
        return False
    try:
        need = max(THRUST_ATR_BARS + 20, ATR_LEN + 5)
        if df is None or len(df) < need:
            return False

        c = df["close"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        tr = pd.concat([(h - l).abs(), (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
        atr_series = wilder_ema(tr, ATR_LEN).dropna()
        if len(atr_series) < THRUST_ATR_BARS + 1:
            return False
        tail = atr_series.iloc[-(THRUST_ATR_BARS + 1):]
        atr_increasing = all(tail.iloc[i] < tail.iloc[i+1] for i in range(len(tail)-1))

        current_volume = float(df["volume"].iloc[-1]) if len(df) else 0.0
        volume_ma = df["volume"].iloc[-21:-1].astype(float).mean() if len(df) >= 21 else 0.0
        volume_ma_ok = (isinstance(volume_ma, (int,float))) and (not math.isnan(volume_ma)) and (volume_ma > 0)
        volume_spike = (current_volume > volume_ma * THRUST_VOLUME_FACTOR) if volume_ma_ok else False

        if atr_increasing and volume_spike and state.get("trade_mode") == "TREND":
            state["thrust_locked"] = True
            state["breakeven"] = state["entry"]
            atr_now = float(ind.get("atr") or (atr_series.iloc[-1] if len(atr_series) else 0.0))
            if state["side"] == "long":
                lookback_lows = df["low"].iloc[-CHANDELIER_LOOKBACK:].astype(float)
                chandelier_stop = lookback_lows.min() - atr_now * CHANDELIER_ATR_MULT
                state["trail"] = max(state.get("trail") or chandelier_stop, chandelier_stop)
            else:
                lookback_highs = df["high"].iloc[-CHANDELIER_LOOKBACK:].astype(float)
                chandelier_stop = lookback_highs.max() + atr_now * CHANDELIER_ATR_MULT
                state["trail"] = min(state.get("trail") or chandelier_stop, chandelier_stop)
            print(colored("🔒 THRUST LOCK: Activated with Chandelier exit", "green"))
            logging.info("THRUST_LOCK: Activated - ATR rising & volume spike")
            return True
    except Exception as e:
        print(colored(f"⚠️ thrust_lock error: {e}", "yellow"))
        logging.error(f"thrust_lock error: {e}")
    return False

# ------------ Post-Entry Management ------------
def determine_trade_mode(df: pd.DataFrame, ind: dict) -> str:
    adx = ind.get("adx", 0)
    atr = ind.get("atr", 0)
    price = ind.get("price", 0)
    atr_pct = (atr / price) * 100 if price > 0 else 0
    if len(df) >= 3:
        idx = -1
        recent_ranges = []
        for i in range(max(0, idx-2), idx+1):
            high = float(df["high"].iloc[i]); low = float(df["low"].iloc[i])
            recent_ranges.append(high - low)
        avg_range = sum(recent_ranges) / len(recent_ranges) if recent_ranges else 0
        range_pct = (avg_range / price) * 100 if price > 0 else 0
        if (adx >= 25 and atr_pct >= 1.0 and range_pct >= 1.5 and
           (ind.get("plus_di", 0) > ind.get("minus_di", 0) if state["side"] == "long" else ind.get("minus_di", 0) > ind.get("plus_di", 0))):
            return "TREND"
    if adx < 20 or atr_pct < 0.8:
        return "SCALP"
    return "SCALP"

def handle_impulse_and_long_wicks(df: pd.DataFrame, ind: dict):
    if not state["open"] or state["qty"] <= 0:
        return None
        
    # Early exit protection context
    now_ts = time.time()
    opened_at = state.get("opened_at") or now_ts
    elapsed_bar = int(state.get("bars", 0))
    
    try:
        idx = -1
        o0 = float(df["open"].iloc[idx]); h0 = float(df["high"].iloc[idx])
        l0 = float(df["low"].iloc[idx]);  c0 = float(df["close"].iloc[idx])
        current_price = ind.get("price") or c0
        entry = state["entry"]; side = state["side"]
        rr = (current_price - entry) / entry * 100 * (1 if side == "long" else -1)
        candle_range = h0 - l0
        body = abs(c0 - o0)
        upper_wick = h0 - max(o0, c0)
        lower_wick = min(o0, c0) - l0
        body_pct = (body / candle_range) * 100 if candle_range > 0 else 0
        upper_wick_pct = (upper_wick / candle_range) * 100 if candle_range > 0 else 0
        lower_wick_pct = (lower_wick / candle_range) * 100 if candle_range > 0 else 0
        atr = ind.get("atr", 0)
        
        # Use guarded close for impulse harvesting
        if atr > 0 and body >= IMPULSE_HARVEST_THRESHOLD * atr:
            candle_direction = 1 if c0 > o0 else -1
            trade_direction = 1 if side == "long" else -1
            if candle_direction == trade_direction:
                if body >= 2.0 * atr:
                    harvest_frac = 0.50; reason = f"Strong Impulse x{body/atr:.2f} ATR"
                else:
                    harvest_frac = 0.33; reason = f"Impulse x{body/atr:.2f} ATR"
                
                # Apply early exit protection
                if elapsed_bar < MIN_HOLD_BARS and not state.get("_tp1_done"):
                    harvest_frac = min(harvest_frac, WICK_MAX_BEFORE_TP1)
                    reason += " (early guard reduced)"
                    
                close_partial(harvest_frac, reason)
                
                if not state["breakeven"] and rr >= BREAKEVEN_AFTER * 100:
                    state["breakeven"] = entry
                    
                if atr and ATR_MULT_TRAIL > 0:
                    gap = atr * ATR_MULT_TRAIL
                    if side == "long":
                        state["trail"] = max(state.get("trail") or (current_price - gap), current_price - gap)
                    else:
                        state["trail"] = min(state.get("trail") or (current_price + gap), current_price + gap)
                return "IMPULSE_HARVEST"
                
        # Use guarded close for wick harvesting  
        if side == "long" and upper_wick_pct >= LONG_WICK_HARVEST_THRESHOLD * 100:
            harvest_frac = 0.25
            if elapsed_bar < MIN_HOLD_BARS and not state.get("_tp1_done"):
                harvest_frac = min(harvest_frac, WICK_MAX_BEFORE_TP1)
                if DEBUG_EXIT_GUARD:
                    logging.info(f"EXIT_GUARD: Reduced wick harvest to {harvest_frac*100:.1f}% (early bars)")
            close_partial(harvest_frac, f"Upper wick (LONG) {upper_wick_pct:.1f}%")
            return "LONG_WICK_HARVEST"
            
        if side == "short" and lower_wick_pct >= LONG_WICK_HARVEST_THRESHOLD * 100:
            harvest_frac = 0.25
            if elapsed_bar < MIN_HOLD_BARS and not state.get("_tp1_done"):
                harvest_frac = min(harvest_frac, WICK_MAX_BEFORE_TP1)
                if DEBUG_EXIT_GUARD:
                    logging.info(f"EXIT_GUARD: Reduced wick harvest to {harvest_frac*100:.1f}% (early bars)")
            close_partial(harvest_frac, f"Lower wick (SHORT) {lower_wick_pct:.1f}%")
            return "LONG_WICK_HARVEST"
            
    except Exception as e:
        print(colored(f"⚠️ handle_impulse_and_long_wicks error: {e}", "yellow"))
        logging.error(f"handle_impulse_and_long_wicks error: {e}")
    return None

def ratchet_protection(ind: dict):
    if not state["open"] or state["qty"] <= 0:
        return None
    current_price = ind.get("price") or price_now() or state["entry"]
    entry = state["entry"]; side = state["side"]
    current_rr = (current_price - entry) / entry * 100 * (1 if side == "long" else -1)
    if current_rr > state["highest_profit_pct"]:
        state["highest_profit_pct"] = current_rr
    if (state["highest_profit_pct"] >= 20 and
        current_rr < state["highest_profit_pct"] * (1 - RATCHET_RETRACE_THRESHOLD)):
        close_partial(0.5, f"Ratchet protection: {state['highest_profit_pct']:.1f}% → {current_rr:.1f}%")
        state["highest_profit_pct"] = current_rr
        return "RATCHET_PROTECTION"
    return None

def scalp_profit_taking(ind: dict, info: dict):
    if not state["open"] or state["qty"] <= 0:
        return None
    price = ind.get("price") or price_now() or state["entry"]
    entry = state["entry"]; side = state["side"]
    rr = (price - entry) / entry * 100 * (1 if side == "long" else -1)
    targets = SCALP_TARGETS; fracs = SCALP_CLOSE_FRACS
    k = int(state.get("profit_targets_achieved", 0))
    if k < len(targets) and rr >= targets[k]:
        close_partial(fracs[k], f"SCALP TP{k+1}@{targets[k]:.2f}%")
        state["profit_targets_achieved"] = k + 1
        if state["profit_targets_achieved"] >= len(targets):
            close_market_strict("SCALP sequence complete")
            return "SCALP_COMPLETE"
        return f"SCALP_TP{k+1}"
    return None

def trend_profit_taking(ind: dict, info: dict):
    if not state["open"] or state["qty"] <= 0:
        return None
    price = ind.get("price") or price_now() or state["entry"]
    entry = state["entry"]; side = state["side"]
    rr = (price - entry) / entry * 100 * (1 if side == "long" else -1)
    adx = float(ind.get("adx") or 0.0)
    targets = TREND_TARGETS; fracs = TREND_CLOSE_FRACS
    k = int(state.get("profit_targets_achieved", 0))
    if k < len(targets) and rr >= targets[k]:
        close_partial(fracs[k], f"TREND TP{k+1}@{targets[k]:.2f}%")
        state["profit_targets_achieved"] = k + 1
        return f"TREND_TP{k+1}"
    if adx >= MIN_TREND_HOLD_ADX:
        return None
    if state.get("profit_targets_achieved", 0) >= len(targets) or trend_end_confirmed(ind, detect_candle_pattern(fetch_ohlcv()), info):
        close_market_strict("TREND finished — full exit")
        return "TREND_COMPLETE"
    return None

# ------------ Smart Alpha Pack extras (unchanged) ------------
def get_adaptive_trail_multiplier(breakout_score: float, trade_mode: str) -> float:
    if breakout_score >= 3.0:
        return TRAIL_MULT_STRONG_ALPHA
    elif breakout_score >= 2.0:
        return TRAIL_MULT_CAUTIOUS_ALPHA
    elif trade_mode == "SCALP":
        return TRAIL_MULT_SCALP_ALPHA
    else:
        return ATR_MULT_TRAIL

def smart_alpha_features(df: pd.DataFrame, ind: dict, prev_ind: dict, info: dict) -> str:
    if not state["open"] or state["qty"] <= 0:
        return None
    action = None
    try:
        if tp0_quick_cash(ind):
            action = "TP0_QUICK_CASH"
        breakout_score, vote_details = breakout_votes(df, ind, prev_ind)
        state["breakout_score"] = breakout_score
        state["breakout_votes_detail"] = vote_details
        if state.get("trade_mode") == "TREND" and not state.get("thrust_locked"):
            if thrust_lock(df, ind):
                action = "THRUST_LOCK_ACTIVATED"
        if state.get("breakout_active") or state.get("thrust_locked"):
            adaptive_mult = get_adaptive_trail_multiplier(
                breakout_score, 
                state.get("trade_mode", "SCALP")
            )
            state["_adaptive_trail_mult"] = adaptive_mult
        if breakout_score >= 3.0:
            print(colored(f"🎯 BREAKOUT SCORE: {breakout_score:.1f}/5.0 - Strong momentum", "cyan"))
            for factor, detail in vote_details.items():
                if factor != "total_score":
                    print(colored(f"   📊 {factor}: {detail}", "blue"))
    except Exception as e:
        print(colored(f"⚠️ smart_alpha_features error: {e}", "yellow"))
        logging.error(f"smart_alpha_features error: {e}")
    return action

# ------------ NEW: SMART HARVEST v2 integrated in smart_post_entry_manager ------------
def smart_post_entry_manager(df: pd.DataFrame, ind: dict, info: dict):
    if not state["open"] or state["qty"] <= 0:
        return None

    # ---- Early exit protection setup ----
    now_ts = info.get("time") or time.time()
    opened_at = state.get("opened_at") or now_ts
    state.setdefault("opened_at", opened_at)
    elapsed_s = max(0, now_ts - state["opened_at"])
    elapsed_bar = int(state.get("bars", 0))

    # TP0 guard - limit early closes
    def _tp0_guard_close(frac, label):
        max_frac = WICK_MAX_BEFORE_TP1 if not state.get("_tp1_done") else frac
        frac = min(frac, max_frac)
        if frac > 0:
            close_partial(frac, label)

    # ---- Smart alpha pack first ----
    prev_ind = compute_indicators(df.iloc[:-1]) if len(df) >= 2 else ind
    alpha_action = smart_alpha_features(df, ind, prev_ind, info)
    if alpha_action:
        return alpha_action

    if state.get("trade_mode") is None:
        trade_mode = determine_trade_mode(df, ind)
        state["trade_mode"] = trade_mode
        state["profit_targets_achieved"] = 0
        state["entry_time"] = time.time()
        print(colored(f"🎯 TRADE MODE DETECTED: {trade_mode}", "cyan"))
        logging.info(f"Trade mode detected: {trade_mode}")

    impulse_action = handle_impulse_and_long_wicks(df, ind)
    if impulse_action:
        return impulse_action

    ratchet_action = ratchet_protection(ind)
    if ratchet_action:
        return ratchet_action

    # Delay trail activation until TP1 if configured
    if TRAIL_ONLY_AFTER_TP1 and not state.get("_tp1_done"):
        state["_adaptive_trail_mult"] = 0.0  # Disable regular ATR trail until TP1

    # ---- Smart Harvest v2 with protection ----
    def smart_harvest(_df: pd.DataFrame, _ind: dict, _info: dict):
        if not state.get("open") or state.get("qty", 0) <= 0:
            return
        px = float(_ind.get("price") or _info.get("price") or 0.0)
        e  = float(state.get("entry") or 0.0)
        side = state.get("side")
        if not (px and e and side):
            return
        rr = (px - e) / e * 100.0 * (1 if side == "long" else -1)
        atr = float(_ind.get("atr") or 0.0)

        if not state.get("_tp1_done") and rr >= TP1_PCT_SCALED * 100.0:
            _tp0_guard_close(TP1_FRAC, f"TP1 @{TP1_PCT_SCALED*100:.2f}%")
            state["_tp1_done"] = True
            state["breakeven"] = e

        if not state.get("_tp2_done") and rr >= TP2_PCT_SCALED * 100.0:
            _tp0_guard_close(TP2_FRAC, f"TP2 @{TP2_PCT_SCALED*100:.2f}%")
            state["_tp2_done"] = True

        if rr > state.get("highest_profit_pct", 0.0):
            state["highest_profit_pct"] = rr
        hp = state.get("highest_profit_pct", 0.0)
        if hp and rr < hp * RATCHET_LOCK_PCT:
            _tp0_guard_close(0.50, f"Ratchet lock {hp:.2f}%→{rr:.2f}%")
            state["highest_profit_pct"] = rr

        # Trail management with protection
        if atr and rr >= (TP1_PCT_SCALED * 100.0):
            mult = _adaptive_trail_mult(_ind)
            if side == "long":
                new_trail = px - atr * mult
                state["trail"] = max(state.get("trail") or new_trail, new_trail, state.get("breakeven") or new_trail)
                if px < state["trail"]:
                    _safe_full_close(f"TRAIL_ATR({mult}x)")
            else:
                new_trail = px + atr * mult
                state["trail"] = min(state.get("trail") or new_trail, new_trail, state.get("breakeven") or new_trail)
                if px > state["trail"]:
                    _safe_full_close(f"TRAIL_ATR({mult}x)")
            if DEBUG_HARVEST:
                logging.info(f"HARV rr={rr:.2f}% trail_mult={mult} hp={state.get('highest_profit_pct',0):.2f}")

    smart_harvest(df, ind, info)

    if state["trade_mode"] == "SCALP":
        return scalp_profit_taking(ind, info)
    else:
        return trend_profit_taking(ind, info)

# ------------ Orders ------------
def open_market(side, qty, price):
    global state
    if qty<=0:
        print(colored("❌ qty<=0 skip open","red")); return
    params = _position_params_for_open(side)
    if MODE_LIVE:
        try:
            lev_params = {"side": "BOTH"}
            try: ex.set_leverage(LEVERAGE, SYMBOL, params=lev_params)
            except Exception as e: print(colored(f"⚠️ set_leverage: {e}", "yellow"))
            ex.create_order(SYMBOL, "market", side, qty, None, params)
        except Exception as e:
            print(colored(f"❌ open: {e}", "red"))
            logging.error(f"open_market error: {e}")
            return
    state.update({
        "open": True, "side": "long" if side=="buy" else "short", 
        "entry": price, "qty": qty, "pnl": 0.0, "bars": 0, 
        "trail": None, "tp1_done": False, "breakeven": None,
        "scale_ins": 0, "scale_outs": 0,
        "last_action": "OPEN", "action_reason": "Initial position",
        "highest_profit_pct": 0.0,
        "trade_mode": None,
        "profit_targets_achieved": 0,
        "entry_time": time.time(),
        "fakeout_pending": False,
        "fakeout_need_side": None,
        "fakeout_confirm_bars": 0,
        "fakeout_started_at": None,
        "breakout_active": False,
        "breakout_direction": None,
        "breakout_entry_price": None,
        "_tp0_done": False,
        "thrust_locked": False,
        "breakout_score": 0.0,
        "breakout_votes_detail": {},
        "opened_by_breakout": False,
        "_adaptive_trail_mult": None,
        "_tp1_done": state.get("_tp1_done", False),
        "_tp2_done": state.get("_tp2_done", False),
        "opened_at": time.time(),
        "_exit_soft_block": False
    })
    print(colored(f"✅ OPEN {side.upper()} qty={fmt(qty,4)} @ {fmt(price)}","green" if side=="buy" else "red"))
    logging.info(f"OPEN {side} qty={qty} price={price}")
    save_state()

def scale_in_position(scale_pct: float, reason: str):
    global state
    if not state["open"]: return
    current_price = price_now() or state["entry"]
    additional_qty = safe_qty(state["qty"] * scale_pct)
    side = "buy" if state["side"] == "long" else "sell"
    if MODE_LIVE:
        try: 
            ex.create_order(SYMBOL, "market", side, additional_qty, None, _position_params_for_open(side))
        except Exception as e: 
            print(colored(f"❌ scale_in: {e}","red")); 
            logging.error(f"scale_in_position error: {e}")
            return
    total_qty = state["qty"] + additional_qty
    state["entry"] = (state["entry"] * state["qty"] + current_price * additional_qty) / total_qty
    state["qty"] = total_qty
    state["scale_ins"] += 1
    state["last_action"] = "SCALE_IN"
    state["action_reason"] = reason
    print(colored(f"📈 SCALE IN +{scale_pct*100:.0f}% | Total qty={fmt(state['qty'],4)} | Avg entry={fmt(state['entry'])} | Reason: {reason}", "cyan"))
    logging.info(f"SCALE_IN +{scale_pct*100:.0f}% total_qty={state['qty']} avg_entry={state['entry']}")
    save_state()

def close_partial(frac, reason):
    global state, compound_pnl
    if not state["open"]: return
    qty_close = safe_qty(max(0.0, state["qty"]*min(max(frac,0.0),1.0)))
    if qty_close < 1:
        print(colored(f"⚠️ skip partial close (amount={fmt(qty_close,4)} < 1 DOGE)", "yellow"))
        return
    px = price_now() or state["entry"]
    side = "sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_position_params_for_close())
        except Exception as e: 
            print(colored(f"❌ partial close: {e}","red")); 
            logging.error(f"close_partial error: {e}")
            return
    pnl=(px-state["entry"])*qty_close*(1 if state["side"]=="long" else -1)
    compound_pnl+=pnl
    state["qty"]-=qty_close
    state["scale_outs"] += 1
    state["last_action"] = "SCALE_OUT"
    state["action_reason"] = reason
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem_qty={fmt(state['qty'],4)}","magenta"))
    logging.info(f"PARTIAL_CLOSE {reason} qty={qty_close} pnl={pnl} remaining={state['qty']}")
    if state["qty"] < 60:
        print(colored(f"⚠️ Remaining qty={fmt(state['qty'],2)} < 60 DOGE → full close triggered", "yellow"))
        logging.warning(f"Auto-full close triggered: remaining qty={state['qty']} < 60 DOGE")
        close_market_strict("auto_full_close_small_qty")
        return
    if state["qty"]<=0:
        reset_after_full_close("fully_exited")
    else:
        save_state()

def reset_after_full_close(reason, prev_side=None):
    global state, post_close_cooldown, wait_for_next_signal_side, last_close_signal_time
    print(colored(f"🔚 CLOSE {reason} totalCompounded now={fmt(compound_pnl)}","magenta"))
    logging.info(f"FULL_CLOSE {reason} total_compounded={compound_pnl}")
    if prev_side is None:
        prev_side = state.get("side")
    state.update({
        "open": False, "side": None, "entry": None, "qty": 0.0, 
        "pnl": 0.0, "bars": 0, "trail": None, "tp1_done": False, 
        "breakeven": None, "scale_ins": 0, "scale_outs": 0,
        "last_action": "CLOSE", "action_reason": reason,
        "highest_profit_pct": 0.0,
        "trade_mode": None,
        "profit_targets_achieved": 0,
        "entry_time": None,
        "fakeout_pending": False,
        "fakeout_need_side": None,
        "fakeout_confirm_bars": 0,
        "fakeout_started_at": None,
        "breakout_active": False,
        "breakout_direction": None,
        "breakout_entry_price": None,
        "_tp0_done": False,
        "thrust_locked": False,
        "breakout_score": 0.0,
        "breakout_votes_detail": {},
        "opened_by_breakout": False,
        "_adaptive_trail_mult": None,
        "_tp1_done": False,
        "_tp2_done": False,
        "opened_at": None,
        "_exit_soft_block": False
    })
    if prev_side == "long":
        wait_for_next_signal_side = "sell"
    elif prev_side == "short":
        wait_for_next_signal_side = "buy"
    else:
        wait_for_next_signal_side = None
    last_close_signal_time = None
    post_close_cooldown = COOLDOWN_AFTER_CLOSE_BARS
    save_state()

def close_market(reason):
    global state, compound_pnl, wait_for_next_signal_side, last_close_signal_time
    if not state["open"]: return
    prev_side_local = state.get("side")
    px=price_now() or state["entry"]; qty=state["qty"]
    side="sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty,None,_position_params_for_close())
        except Exception as e: 
            print(colored(f"❌ close: {e}","red")); 
            logging.error(f"close_market error: {e}")
            return
    pnl=(px-state["entry"])*qty*(1 if state["side"]=="long" else -1)
    compound_pnl+=pnl
    print(colored(f"🔚 CLOSE {state['side']} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
    logging.info(f"CLOSE_MARKET {state['side']} reason={reason} pnl={pnl} total={compound_pnl}")
    reset_after_full_close(reason, prev_side_local)

def advanced_position_management(candle_info: dict, ind: dict):
    if not state["open"]:
        return None
    current_side = state["side"]
    px = ind.get("price") or price_now() or state["entry"]
    trend_signal = check_trend_confirmation(candle_info, ind, current_side)
    if trend_signal == "CONFIRMED_CONTINUE":
        print(colored("📈 اتجاه مؤكد: استمرار في الترند", "green"))
        logging.info("Trend confirmed: continuing in the same direction")
    elif trend_signal == "CONFIRMED_REVERSAL":
        print(colored("⚠️ انعكاس مؤكد: خروج جزئي", "red"))
        logging.info("Reversal confirmed: partial exit")
        close_partial(0.3, "Reversal confirmed by trend analysis")
        return "SCALE_OUT_REVERSAL"
    should_scale, step_size, scale_reason = should_scale_in(candle_info, ind, current_side)
    if should_scale:
        scale_in_position(step_size, scale_reason)
        return "SCALE_IN"
    do_scale_out, scale_out_reason = should_scale_out(candle_info, ind, current_side)
    if do_scale_out:
        close_partial(0.3, scale_out_reason)
        return "SCALE_OUT"
    trail_mult = get_trail_multiplier({**ind, "price": px})
    if state["trail"] is not None and trail_mult != ATR_MULT_TRAIL:
        atr = ind.get("atr", 0)
        if current_side == "long":
            new_trail = px - atr * trail_mult
            state["trail"] = max(state["trail"], new_trail)
        else:
            new_trail = px + atr * trail_mult
            state["trail"] = min(state["trail"], new_trail)
        if trail_mult != ATR_MULT_TRAIL:
            state["last_action"] = "TRAIL_ADJUST"
            state["action_reason"] = f"Trail mult {ATR_MULT_TRAIL} → {trail_mult}"
            print(colored(f"🔄 TRAIL ADJUST: multiplier {ATR_MULT_TRAIL} → {trail_mult} ({'STRONG' if trail_mult==TRAIL_MULT_STRONG else 'MED' if trail_mult==TRAIL_MULT_MED else 'CHOP'})", "blue"))
            logging.info(f"TRAIL_ADJUST {ATR_MULT_TRAIL}→{trail_mult}")
    return None

# ------------ Smart Exit (signature updated with df cache) ------------
def smart_exit_check(info, ind, df_cached=None, prev_ind_cached=None):
    if not (STRATEGY=="smart" and USE_SMART_EXIT and state["open"]):
        return None

    df_local = df_cached if df_cached is not None else fetch_ohlcv()
    
    # ---- Early exit protection calculations ----
    now_ts = info.get("time") or time.time()
    opened_at = state.get("opened_at") or now_ts
    elapsed_s = max(0, now_ts - opened_at)
    elapsed_bar = int(state.get("bars", 0))

    def _allow_full_close(reason: str) -> bool:
        """Prevent early full closes unless hard conditions met"""
        if elapsed_bar >= MIN_HOLD_BARS or elapsed_s >= MIN_HOLD_SECONDS:
            return True
        # Early close allowed only for these hard reasons
        hard_reasons = ("TRAIL", "TRAIL_ATR", "CHANDELIER", "STRICT", "FORCE", "STOP", "EMERGENCY")
        return any(tag in reason.upper() for tag in hard_reasons)

    # ---- Enhanced Patience Guard ----
    side = state.get("side")
    rsi = float(ind.get("rsi") or 50.0)
    adx = float(ind.get("adx") or 0.0)
    adx_prev = float(ind.get("adx_prev") or adx)
    rf = info.get("filter")
    px = ind.get("price") or info.get("price")

    reverse_votes = 0
    vote_details = []

    # RSI vote
    if side == "long" and rsi < REV_RSI_LEVEL:
        reverse_votes += 1
        vote_details.append("RSI<50")
    if side == "short" and rsi > (100-REV_RSI_LEVEL):
        reverse_votes += 1  
        vote_details.append("RSI>50")

    # ADX vote
    if adx_prev and (adx_prev - adx) >= REV_ADX_DROP:
        reverse_votes += 1
        vote_details.append(f"ADX↓{adx_prev:.1f}->{adx:.1f}")

    # RF hysteresis vote
    if REV_RF_CROSS and px is not None and rf is not None:
        try:
            bps_diff = abs((px - rf) / rf) * 10000
            if (side == "long" and px < rf - (RF_HYSTERESIS_BPS/10000.0)*rf) or \
               (side == "short" and px > rf + (RF_HYSTERESIS_BPS/10000.0)*rf):
                reverse_votes += 1
                vote_details.append("RF cross")
        except Exception:
            pass

    # Candle confirmation
    try:
        closes = df_local["close"].astype(float)
        opens = df_local["open"].astype(float)
        last2_red = (closes.iloc[-1] < opens.iloc[-1]) and (closes.iloc[-2] < opens.iloc[-2])
        last2_green = (closes.iloc[-1] > opens.iloc[-1]) and (closes.iloc[-2] > opens.iloc[-2])
        candle_ok = (side == "long" and last2_red) or (side == "short" and last2_green)
    except Exception:
        candle_ok = False

    # Set exit block flag
    if (elapsed_bar < MIN_HOLD_BARS or elapsed_s < MIN_HOLD_SECONDS) and (reverse_votes < PATIENCE_NEED_CONSENSUS or not candle_ok):
        state["_exit_soft_block"] = True
        if DEBUG_PATIENCE:
            logging.info(f"PATIENCE: Blocking soft exits (bars:{elapsed_bar}, votes:{reverse_votes}, candle_ok:{candle_ok})")
    else:
        state["_exit_soft_block"] = False

    # Safe close functions
    def _safe_close_partial(frac, reason):
        if state.get("_exit_soft_block") and not state.get("_tp1_done"):
            frac = min(frac, WICK_MAX_BEFORE_TP1)
            if DEBUG_EXIT_GUARD:
                logging.info(f"EXIT_GUARD: Reduced partial close to {frac*100:.1f}% for: {reason}")
        close_partial(frac, reason)

    def _safe_full_close(reason):
        if NO_FULL_CLOSE_BEFORE_TP1 and not state.get("_tp1_done"):
            if not _allow_full_close(reason):
                if DEBUG_EXIT_GUARD:
                    logging.info(f"EXIT_BLOCKED (early): {reason} - bars:{elapsed_bar}, secs:{elapsed_s:.0f}")
                return False
                
        if state.get("_exit_soft_block") and not _allow_full_close(reason):
            if DEBUG_EXIT_GUARD:
                logging.info(f"EXIT_BLOCKED (patience): {reason} - votes:{reverse_votes}/{PATIENCE_NEED_CONSENSUS}")
            return False
            
        close_market_strict(reason)
        return True

    # ---- Rest of existing smart_exit_check logic with safe closes ----
    candle_info = detect_candle_pattern(df_local)
    management_action = advanced_position_management(candle_info, ind)
    if management_action:
        print(colored(f"🎯 MANAGEMENT: {management_action} - {state['action_reason']}", "yellow"))
        logging.info(f"MANAGEMENT_ACTION: {management_action} - {state['action_reason']}")

    df = df_local
    post_entry_action = smart_post_entry_manager(df_local, ind, info)
    if post_entry_action:
        print(colored(f"🎯 POST-ENTRY: {post_entry_action} - Trade Mode: {state.get('trade_mode', 'N/A')}", "cyan"))
        logging.info(f"POST_ENTRY_ACTION: {post_entry_action} - Trade Mode: {state.get('trade_mode', 'N/A')}")

    # Existing trend/fakeout logic...
    trend_signal = check_trend_confirmation(candle_info, ind, state["side"])
    if state["open"]:
        if (trend_signal == "POSSIBLE_FAKEOUT" and 
            not state["fakeout_pending"] and 
            state["fakeout_confirm_bars"] == 0):
            state["fakeout_pending"] = True
            state["fakeout_confirm_bars"] = 2
            state["fakeout_need_
