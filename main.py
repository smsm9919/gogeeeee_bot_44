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

THRUST_ATR_BARS = 3
THRUST_VOLUME_FACTOR = 1.3
CHANDELIER_ATR_MULT = 3.0
CHANDELIER_LOOKBACK = 20

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

# === EXIT PROTECTION LAYER CONFIG ===
MIN_HOLD_BARS        = 3
MIN_HOLD_SECONDS     = 120
TRAIL_ONLY_AFTER_TP1 = True
RF_HYSTERESIS_BPS    = 8
NO_FULL_CLOSE_BEFORE_TP1 = True
WICK_MAX_BEFORE_TP1  = 0.25

# ---- Smart Harvest defaults ----
TP1_PCT_SCALED = 0.004
TP2_PCT_SCALED = 0.008
TP1_FRAC       = 0.25
TP2_FRAC       = 0.35

# ---- Patience Guard defaults ----
ENABLE_PATIENCE       = True
PATIENCE_MIN_BARS     = 2
PATIENCE_NEED_CONSENSUS = 2
REV_ADX_DROP          = 3.0
REV_RSI_LEVEL         = 50
REV_RF_CROSS          = True

# Optional debug flags
DEBUG_PATIENCE = False
DEBUG_HARVEST  = False

print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • SYMBOL={SYMBOL} • {INTERVAL}", "yellow"))
print(colored(f"STRATEGY: {STRATEGY.upper()} • SMART_EXIT={'ON' if USE_SMART_EXIT else 'OFF'}", "yellow"))

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
    "opened_by_breakout": False
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
                "opened_by_breakout": False
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
        if atr > 0 and body >= IMPULSE_HARVEST_THRESHOLD * atr:
            candle_direction = 1 if c0 > o0 else -1
            trade_direction = 1 if side == "long" else -1
            if candle_direction == trade_direction:
                if body >= 2.0 * atr:
                    harvest_frac = 0.50; reason = f"Strong Impulse x{body/atr:.2f} ATR"
                else:
                    harvest_frac = 0.33; reason = f"Impulse x{body/atr:.2f} ATR"
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
        if side == "long" and upper_wick_pct >= LONG_WICK_HARVEST_THRESHOLD * 100:
            close_partial(0.25, f"Upper wick (LONG) {upper_wick_pct:.1f}%")
            return "LONG_WICK_HARVEST"
        if side == "short" and lower_wick_pct >= LONG_WICK_HARVEST_THRESHOLD * 100:
            close_partial(0.25, f"Lower wick (SHORT) {lower_wick_pct:.1f}%")
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

    # === EXIT PROTECTION LAYER: Initialize timing guards ===
    now_ts = info.get("time") or time.time()
    opened_at = state.get("opened_at") or now_ts
    state.setdefault("opened_at", opened_at)
    elapsed_s = max(0, now_ts - state["opened_at"])
    elapsed_bar = int(state.get("bars", 0))

    # TP0 guard: limit partial closes before TP1
    def _tp0_guard_close(frac, label):
        max_frac = WICK_MAX_BEFORE_TP1 if not state.get("_tp1_done") else frac
        frac = min(frac, max_frac)
        if frac > 0:
            close_partial(frac, label)

    # Trail management: disable normal trail until TP1 (except Thrust/Chandelier)
    if TRAIL_ONLY_AFTER_TP1 and not state.get("_tp1_done"):
        state["_adaptive_trail_mult"] = 0.0  # Disable normal ATR trail

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

    # Use protected close for wick/impulse harvest
    impulse_action = handle_impulse_and_long_wicks(df, ind)
    if impulse_action:
        return impulse_action

    ratchet_action = ratchet_protection(ind)
    if ratchet_action:
        return ratchet_action

    # ---- Smart Harvest v2 helpers (local only) ----
    def _adaptive_trail_mult(_ind: dict) -> float:
        adx = float(_ind.get("adx") or 0.0)
        rsi = float(_ind.get("rsi") or 50.0)
        side = state.get("side")
        strong = (adx >= 30 and ((side == "long" and rsi >= 55) or (side == "short" and rsi <= 45)))
        weak   = (adx < 20)
        return 2.4 if strong else (1.6 if weak else 2.0)

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

        if atr and rr >= (TP1_PCT_SCALED * 100.0):
            mult = _adaptive_trail_mult(_ind)
            if side == "long":
                new_trail = px - atr * mult
                state["trail"] = max(state.get("trail") or new_trail, new_trail, state.get("breakeven") or new_trail)
                if px < state["trail"]:
                    close_market_strict(f"TRAIL_ATR({mult}x)")
            else:
                new_trail = px + atr * mult
                state["trail"] = min(state.get("trail") or new_trail, new_trail, state.get("breakeven") or new_trail)
                if px > state["trail"]:
                    close_market_strict(f"TRAIL_ATR({mult}x)")
            if DEBUG_HARVEST:
                logging.info(f"HARV rr={rr:.2f}% trail_mult={mult} hp={state.get('highest_profit_pct',0):.2f}")

    # ---- invoke Smart Harvest v2 (non-intrusive) ----
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
        "opened_at": time.time()  # Track position opening time
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
        "opened_at": None
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

# ------------ Smart Exit (UPDATED with Exit Protection Layer) ------------
def smart_exit_check(info, ind, df_cached=None, prev_ind_cached=None):
    if not (STRATEGY=="smart" and USE_SMART_EXIT and state["open"]):
        return None

    # === EXIT PROTECTION LAYER: Enhanced Patience Guard ===
    now_ts = info.get("time") or time.time()
    opened_at = state.get("opened_at") or now_ts
    elapsed_s = max(0, now_ts - opened_at)
    elapsed_bar = int(state.get("bars", 0))

    def _allow_full_close(reason: str) -> bool:
        """Allow full close only for hard reasons during min hold period"""
        if elapsed_bar >= MIN_HOLD_BARS or elapsed_s >= MIN_HOLD_SECONDS:
            return True
        # Allowed early close reasons
        hard_reasons = ("TRAIL", "TRAIL_ATR", "CHANDELIER", "STRICT", "FORCE", "STOP", "EMERGENCY")
        return any(tag in reason for tag in hard_reasons)

    # Enhanced Patience Guard: Reverse consensus check
    side = state.get("side")
    rsi = float(ind.get("rsi") or 50.0)
    adx = float(ind.get("adx") or 0.0)
    adx_prev = float(ind.get("adx_prev") or adx)
    rf = ind.get("rfilt") or ind.get("rf")
    px = ind.get("price") or info.get("price")

    reverse_votes = 0
    vote_details = []
    
    # RSI reverse signal
    if side == "long" and rsi < REV_RSI_LEVEL: 
        reverse_votes += 1
        vote_details.append(f"RSI<{REV_RSI_LEVEL}")
    if side == "short" and rsi > (100-REV_RSI_LEVEL): 
        reverse_votes += 1
        vote_details.append(f"RSI>{100-REV_RSI_LEVEL}")
    
    # ADX drop signal
    if adx_prev and (adx_prev - adx) >= REV_ADX_DROP:
        reverse_votes += 1
        vote_details.append(f"ADX↓{adx_prev:.1f}->{adx:.1f}")
    
    # RF/MA cross with hysteresis
    if REV_RF_CROSS and px is not None and rf is not None:
        try:
            bps_diff = abs((px - rf) / rf) * 10000
            if (side == "long" and px < rf - (RF_HYSTERESIS_BPS/10000.0)*rf) or \
               (side == "short" and px > rf + (RF_HYSTERESIS_BPS/10000.0)*rf):
                reverse_votes += 1
                vote_details.append(f"RF_cross({bps_diff:.1f}bps)")
        except Exception:
            pass

    # Two confirming candles against position
    try:
        df_local = df_cached if df_cached is not None else fetch_ohlcv()
        closes = df_local["close"].astype(float)
        opens = df_local["open"].astype(float)
        last2_red = (closes.iloc[-1] < opens.iloc[-1]) and (closes.iloc[-2] < opens.iloc[-2])
        last2_green = (closes.iloc[-1] > opens.iloc[-1]) and (closes.iloc[-2] > opens.iloc[-2])
        candle_ok = (side == "long" and last2_red) or (side == "short" and last2_green)
    except Exception:
        candle_ok = False

    # Set soft block state
    if (elapsed_bar < MIN_HOLD_BARS or elapsed_s < MIN_HOLD_SECONDS) and (reverse_votes < PATIENCE_NEED_CONSENSUS or not candle_ok):
        state["_exit_soft_block"] = True
        if DEBUG_PATIENCE:
            logging.info(f"PAT: early-bars block (bars={elapsed_bar}, votes={reverse_votes}/{PATIENCE_NEED_CONSENSUS})")
    else:
        state["_exit_soft_block"] = False
        if DEBUG_PATIENCE and reverse_votes >= PATIENCE_NEED_CONSENSUS:
            logging.info(f"PAT: consensus={reverse_votes}/{PATIENCE_NEED_CONSENSUS} ({', '.join(vote_details)})")

    # Protected close functions
    def _safe_close_partial(frac, reason):
        if state.get("_exit_soft_block") and not state.get("_tp1_done"):
            frac = min(frac, WICK_MAX_BEFORE_TP1)
        close_partial(frac, reason)

    def _safe_full_close(reason):
        if NO_FULL_CLOSE_BEFORE_TP1 and not state.get("_tp1_done"):
            if not _allow_full_close(reason):
                logging.info(f"EXIT_BLOCKED (early): {reason}")
                return False
        if state.get("_exit_soft_block") and not _allow_full_close(reason):
            logging.info(f"EXIT_BLOCKED (patience): {reason}")
            return False
        close_market_strict(reason)
        return True

    # ---- cached df/prev_ind usage ----
    df_local = df_cached if df_cached is not None else fetch_ohlcv()
    candle_info = detect_candle_pattern(df_local)
    management_action = advanced_position_management(candle_info, ind)
    if management_action:
        print(colored(f"🎯 MANAGEMENT: {management_action} - {state['action_reason']}", "yellow"))
        logging.info(f"MANAGEMENT_ACTION: {management_action} - {state['action_reason']}")

    # Smart alpha post-entry
    df = df_local
    post_entry_action = smart_post_entry_manager(df_local, ind, info)
    if post_entry_action:
        print(colored(f"🎯 POST-ENTRY: {post_entry_action} - Trade Mode: {state.get('trade_mode', 'N/A')}", "cyan"))
        logging.info(f"POST_ENTRY_ACTION: {post_entry_action} - Trade Mode: {state.get('trade_mode', 'N/A')}")

    # Patience guard BEFORE any final close decisions
    if ENABLE_PATIENCE and state.get("_exit_soft_block"):
        return None

    px = info["price"]; e = state["entry"]; side = state["side"]
    if e is None or px is None or side is None or e == 0:
        return None

    trend_signal = check_trend_confirmation(candle_info, ind, state["side"])
    if state["open"]:
        if (trend_signal == "POSSIBLE_FAKEOUT" and 
            not state["fakeout_pending"] and 
            state["fakeout_confirm_bars"] == 0):
            state["fakeout_pending"] = True
            state["fakeout_confirm_bars"] = 2
            state["fakeout_need_side"] = "short" if state["side"] == "long" else "long"
            state["fakeout_started_at"] = info["time"]
            print(colored("🕒 WAITING — possible fake reversal detected, holding position...", "yellow"))
            logging.info("FAKEOUT PROTECTION: Possible fake reversal detected, waiting for confirmation")
            return None
        elif state["fakeout_pending"]:
            if (trend_signal == "CONFIRMED_REVERSAL" and 
                state["fakeout_need_side"] == ("short" if state["side"] == "long" else "long")):
                state["fakeout_confirm_bars"] -= 1
                print(colored(f"🕒 FAKEOUT CONFIRMATION — {state['fakeout_confirm_bars']} bars left", "yellow"))
                if state["fakeout_confirm_bars"] <= 0:
                    print(colored("⚠️ CONFIRMED REVERSAL — closing position", "red"))
                    logging.info("FAKEOUT PROTECTION: Confirmed reversal after fakeout delay")
                    _safe_full_close("CONFIRMED REVERSAL after fakeout delay")
                    state["fakeout_pending"] = False
                    state["fakeout_need_side"] = None
                    state["fakeout_confirm_bars"] = 0
                    state["fakeout_started_at"] = None
                    return True
            elif trend_signal == "CONFIRMED_CONTINUE":
                state["fakeout_pending"] = False
                state["fakeout_need_side"] = None
                state["fakeout_confirm_bars"] = 0
                state["fakeout_started_at"] = None
                print(colored("✅ CONTINUE — fakeout ignored, staying in trade", "green"))
                logging.info("FAKEOUT PROTECTION: Fakeout ignored, continuing in trade")

    rr = (px - e)/e * 100.0 * (1 if side=="long" else -1)
    atr = ind.get("atr") or 0.0
    adx = ind.get("adx") or 0.0
    rsi = ind.get("rsi") or 50.0
    tp_multiplier, trail_activate_multiplier = get_dynamic_tp_params(adx)
    current_tp1_pct = TP1_PCT * tp_multiplier
    current_trail_activate = TRAIL_ACTIVATE * trail_activate_multiplier
    current_tp1_pct_pct = current_tp1_pct * 100.0
    current_trail_activate_pct = current_trail_activate * 100.0
    trail_mult_to_use = state.get("_adaptive_trail_mult") or ATR_MULT_TRAIL

    if state.get("trade_mode") is None:
        if (not state["tp1_done"]) and rr >= current_tp1_pct_pct:
            _safe_close_partial(TP1_CLOSE_FRAC, f"TP1@{current_tp1_pct_pct:.2f}%")
            state["tp1_done"] = True

    if side == "long" and adx >= 30 and rsi >= RSI_TREND_BUY:
        print(colored("💎 HOLD-TP: strong uptrend continues, delaying TP", "cyan"))
    elif side == "short" and adx >= 30 and rsi <= RSI_TREND_SELL:
        print(colored("💎 HOLD-TP: strong downtrend continues, delaying TP", "cyan"))

    if state["bars"] < 2:
        return None

    if rr > state["highest_profit_pct"]:
        state["highest_profit_pct"] = rr
        if tp_multiplier > 1.0:
            print(colored(f"🎯 TREND AMPLIFIER: New high {rr:.2f}% • TP={current_tp1_pct_pct:.2f}% • TrailActivate={current_trail_activate_pct:.2f}%", "green"))
            logging.info(f"TREND_AMPLIFIER new_high={rr:.2f}% TP={current_tp1_pct_pct:.2f}%")

    if (not state["tp1_done"]) and rr >= current_tp1_pct_pct:
        _safe_close_partial(TP1_CLOSE_FRAC, f"TP1@{current_tp1_pct_pct:.2f}%")
        state["tp1_done"] = True
        if rr >= BREAKEVEN_AFTER * 100.0:
            state["breakeven"] = e

    if (state["highest_profit_pct"] >= current_trail_activate_pct and
        rr < state["highest_profit_pct"] * RATCHET_LOCK_PCT):
        _safe_close_partial(0.5, f"Ratchet Lock @ {state['highest_profit_pct']:.2f}%")
        state["highest_profit_pct"] = rr
        return None

    if rr >= current_trail_activate_pct and atr and trail_mult_to_use > 0:
        gap = atr * trail_mult_to_use
        if side == "long":
            new_trail = px - gap
            state["trail"] = max(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None:
                state["trail"] = max(state["trail"], state["breakeven"])
            if px < state["trail"]:
                _safe_full_close(f"TRAIL_ATR({trail_mult_to_use}x)")
                return True
        else:
            new_trail = px + gap
            state["trail"] = min(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None:
                state["trail"] = min(state["trail"], state["breakeven"])
            if px > state["trail"]:
                _safe_full_close(f"TRAIL_ATR({trail_mult_to_use}x)")
                return True
    return None

# ------------ BREAKOUT ENGINE (unchanged behavior) ------------
def detect_breakout(df: pd.DataFrame, ind: dict, prev_ind: dict) -> str:
    try:
        if len(df) < BREAKOUT_LOOKBACK_BARS + 2:
            return None
        current_idx = -1
        adx = float(ind.get("adx") or 0.0)
        atr_now = float(ind.get("atr") or 0.0)
        atr_prev = float(prev_ind.get("atr") or atr_now)
        price = float(df["close"].iloc[current_idx])
        atr_spike = atr_now > atr_prev * BREAKOUT_ATR_SPIKE
        strong_trend = adx >= BREAKOUT_ADX_THRESHOLD
        if not (atr_spike and strong_trend):
            return None
        recent_highs = df["high"].iloc[-BREAKOUT_LOOKBACK_BARS:-1].astype(float)
        new_high = price > recent_highs.max() if len(recent_highs) > 0 else False
        recent_lows = df["low"].iloc[-BREAKOUT_LOOKBACK_BARS:-1].astype(float)
        new_low = price < recent_lows.min() if len(recent_lows) > 0 else False
        if new_high: return "BULL_BREAKOUT"
        elif new_low: return "BEAR_BREAKOUT"
    except Exception as e:
        print(colored(f"⚠️ detect_breakout error: {e}", "yellow"))
        logging.error(f"detect_breakout error: {e}")
    return None

def handle_breakout_entries(df: pd.DataFrame, ind: dict, prev_ind: dict, bal: float, spread_bps: float) -> bool:
    global state
    breakout_signal = detect_breakout(df, ind, prev_ind)
    if not breakout_signal or state["breakout_active"]:
        return False
    price = ind.get("price") or float(df["close"].iloc[-1])
    breakout_score, vote_details = breakout_votes(df, ind, prev_ind)
    if breakout_score < 3.0:
        print(colored(f"⛔ BREAKOUT: Score too low ({breakout_score:.1f}/5.0) - skipping", "yellow"))
        logging.warning(f"BREAKOUT_ENGINE: Score filter blocked - {breakout_score:.1f}/5.0")
        return False
    if spread_bps is not None and spread_bps > SPREAD_GUARD_BPS:
        print(colored(f"⛔ BREAKOUT: Spread too high ({fmt(spread_bps,2)}bps) - skipping entry", "yellow"))
        logging.warning(f"BREAKOUT_ENGINE: Spread filter blocked entry - {spread_bps}bps")
        return False
    qty = compute_size(bal, price)
    if qty < (LOT_MIN or 1):
        print(colored(f"⛔ BREAKOUT: Quantity too small ({fmt(qty,4)} < {LOT_MIN or 1}) - skipping", "yellow"))
        logging.warning(f"BREAKOUT_ENGINE: Quantity below minimum - {qty} < {LOT_MIN or 1}")
        return False
    if not can_open(breakout_signal, price):
        print(colored("⛔ BREAKOUT: Idempotency guard blocked duplicate entry", "yellow"))
        logging.warning("BREAKOUT_ENGINE: Idempotency guard blocked entry")
        return False
    if breakout_signal == "BULL_BREAKOUT":
        open_market("buy", qty, price)
        state["breakout_active"] = True
        state["breakout_direction"] = "bull"
        state["breakout_entry_price"] = price
        state["opened_by_breakout"] = True
        state["breakout_score"] = breakout_score
        state["breakout_votes_detail"] = vote_details
        print(colored(f"⚡ BREAKOUT ENGINE: BULLISH EXPLOSION - ENTERING LONG (Score: {breakout_score:.1f}/5.0)", "green"))
        logging.info(f"BREAKOUT_ENGINE: Bullish explosion - LONG {qty} @ {price} - Score: {breakout_score:.1f}")
        return True
    elif breakout_signal == "BEAR_BREAKOUT":
        open_market("sell", qty, price)
        state["breakout_active"] = True  
        state["breakout_direction"] = "bear"
        state["breakout_entry_price"] = price
        state["opened_by_breakout"] = True
        state["breakout_score"] = breakout_score
        state["breakout_votes_detail"] = vote_details
        print(colored(f"⚡ BREAKOUT ENGINE: BEARISH CRASH - ENTERING SHORT (Score: {breakout_score:.1f}/5.0)", "red"))
        logging.info(f"BREAKOUT_ENGINE: Bearish crash - SHORT {qty} @ {price} - Score: {breakout_score:.1f}")
        return True
    return False

def handle_breakout_exits(df: pd.DataFrame, ind: dict, prev_ind: dict) -> bool:
    global state
    if not state["breakout_active"] or not state["open"]:
        return False
    current_idx = -1
    atr_now = float(ind.get("atr") or 0.0)
    atr_prev = float(prev_ind.get("atr") or atr_now)
    volatility_calm = atr_now < atr_prev * BREAKOUT_CALM_THRESHOLD
    if volatility_calm:
        direction = state["breakout_direction"]
        entry_price = state["breakout_entry_price"]
        current_price = ind.get("price") or float(df["close"].iloc[current_idx])
        pnl_pct = ((current_price - entry_price) / entry_price * 100 * 
                  (1 if direction == "bull" else -1))
        close_market_strict(f"Breakout ended - {pnl_pct:.2f}% PnL")
        print(colored(f"✅ BREAKOUT ENGINE: {direction.upper()} breakout ended - {pnl_pct:.2f}% PnL", "magenta"))
        logging.info(f"BREAKOUT_ENGINE: {direction} breakout ended - PnL: {pnl_pct:.2f}%")
        state["breakout_active"] = False
        state["breakout_direction"] = None  
        state["breakout_entry_price"] = None
        state["opened_by_breakout"] = False
        return True
    return False

def breakout_emergency_protection(ind: dict, prev_ind: dict) -> bool:
    if not (EMERGENCY_PROTECTION_ENABLED and state.get("open")):
        return False
    try:
        adx = float(ind.get("adx") or 0.0)
        rsi = float(ind.get("rsi") or 50.0)
        atr_now  = float(ind.get("atr") or 0.0)
        atr_prev = float(prev_ind.get("atr") or atr_now)
        price = ind.get("price") or price_now() or state.get("entry")
        atr_spike = atr_now > atr_prev * EMERGENCY_ATR_SPIKE_RATIO
        strong_trend = adx >= EMERGENCY_ADX_MIN
        if not (atr_spike and strong_trend):
            return False
        pump  = rsi >= EMERGENCY_RSI_PUMP
        crash = rsi <= EMERGENCY_RSI_CRASH
        if not (pump or crash):
            return False
        side  = state.get("side")
        entry = state.get("entry") or price
        if not (side and entry and price):
            return False
        rr_pct = (price - entry) / entry * 100.0 * (1 if side == "long" else -1)
        print(colored(f"🛡️ EMERGENCY LAYER DETECTED: {side.upper()} | RSI={rsi:.1f} | ADX={adx:.1f} | ATR Spike={atr_now/atr_prev:.2f}x | PnL={rr_pct:.2f}%", "yellow"))
        logging.info(f"EMERGENCY_LAYER: {side} RSI={rsi} ADX={adx} ATR_ratio={atr_now/atr_prev:.2f} PnL={rr_pct:.2f}%")
        if (pump and side == "short") or (crash and side == "long"):
            close_market_strict("EMERGENCY opposite pump/crash — close now")
            print(colored(f"🛑 EMERGENCY: AGAINST POSITION - FULL CLOSE", "red"))
            logging.warning(f"EMERGENCY_LAYER: Against position - full close")
            return True
        if EMERGENCY_POLICY == "close_always":
            close_market_strict("EMERGENCY favorable pump/crash — close all")
            print(colored(f"🟡 EMERGENCY: FAVORABLE - POLICY CLOSE ALL", "yellow"))
            logging.info(f"EMERGENCY_LAYER: Favorable - policy close all")
            return True
        if rr_pct >= EMERGENCY_FULL_CLOSE_PROFIT:
            close_market_strict(f"EMERGENCY full close @ {rr_pct:.2f}%")
            print(colored(f"🟢 EMERGENCY: PROFIT TARGET HIT - FULL CLOSE @ {rr_pct:.2f}%", "green"))
            logging.info(f"EMERGENCY_LAYER: Profit target hit - full close @ {rr_pct:.2f}%")
            return True
        harvest = max(0.0, min(1.0, EMERGENCY_HARVEST_FRAC))
        if harvest > 0:
            close_partial(harvest, f"EMERGENCY {'PUMP' if pump else 'CRASH'} harvest {harvest*100:.0f}%")
            print(colored(f"💰 EMERGENCY: HARVEST {harvest*100:.0f}% - PnL={rr_pct:.2f}%", "cyan"))
            logging.info(f"EMERGENCY_LAYER: Harvest {harvest*100:.0f}% - PnL={rr_pct:.2f}%")
        state["breakeven"] = entry
        if atr_now > 0:
            if side == "long":
                new_trail = price - atr_now * EMERGENCY_TRAIL_ATR_MULT
                state["trail"] = max(state.get("trail") or new_trail, new_trail)
            else:
                new_trail = price + atr_now * EMERGENCY_TRAIL_ATR_MULT
                state["trail"] = min(state.get("trail") or new_trail, new_trail)
            print(colored(f"🛡️ EMERGENCY: BREAKEVEN + TRAIL SET @ {new_trail:.6f}", "blue"))
            logging.info(f"EMERGENCY_LAYER: Breakeven + trail set @ {new_trail:.6f}")
        if EMERGENCY_POLICY == "tp_then_close":
            close_market_strict("EMERGENCY: harvest then full close")
            print(colored(f"🟡 EMERGENCY: TP_THEN_CLOSE POLICY - FULL CLOSE", "yellow"))
            logging.info(f"EMERGENCY_LAYER: tp_then_close policy - full close")
            return True
        print(colored(f"🟢 EMERGENCY: TP_THEN_TRAIL POLICY - RIDING THE MOVE", "green"))
        logging.info(f"EMERGENCY_LAYER: tp_then_trail policy - riding the move")
        return True
    except Exception as e:
        print(colored(f"⚠️ breakout_emergency_protection error: {e}", "yellow"))
        logging.error(f"breakout_emergency_protection error: {e}")
        return False

# ------------ HUD / snapshot (unchanged) ------------
def detect_candle_pattern(df: pd.DataFrame):
    if len(df) < 3:
        return {"pattern": "NONE", "name_ar": "لا شيء", "name_en": "NONE", "strength": 0}
    idx = -1
    o2, h2, l2, c2 = map(float, (df["open"].iloc[idx-2], df["high"].iloc[idx-2], df["low"].iloc[idx-2], df["close"].iloc[idx-2]))
    o1, h1, l1, c1 = map(float, (df["open"].iloc[idx-1], df["high"].iloc[idx-1], df["low"].iloc[idx-1], df["close"].iloc[idx-1]))
    o0, h0, l0, c0 = map(float, (df["open"].iloc[idx], df["high"].iloc[idx], df["low"].iloc[idx], df["close"].iloc[idx]))
    def _candle_stats(o, c, h, l):
        rng = max(h - l, 1e-12); body = abs(c - o); upper = h - max(o, c); lower = min(o, c) - l
        return {"range": rng,"body": body,"body_pct": (body / rng) * 100.0,"upper_pct": (upper / rng) * 100.0,"lower_pct": (lower / rng) * 100.0,"bull": c > o,"bear": c < o}
    s2 = _candle_stats(o2, c2, h2, l2); s1 = _candle_stats(o1, c1, h1, l1); s0 = _candle_stats(o0, c0, h0, l0)
    if (s2["bear"] and s2["body_pct"] >= 60 and s1["body_pct"] <= 25 and l1 > l2 and s0["bull"] and s0["body_pct"] >= 50 and c0 > (o1 + c1)/2):
        return {"pattern": "MORNING_STAR", "name_ar": "نجمة الصباح", "name_en": "Morning Star", "strength": 4}
    if (s2["bull"] and s2["body_pct"] >= 60 and s1["body_pct"] <= 25 and h1 < h2 and s0["bear"] and s0["body_pct"] >= 50 and c0 < (o1 + c1)/2):
        return {"pattern": "EVENING_STAR", "name_ar": "نجمة المساء", "name_en": "Evening Star", "strength": 4}
    if (s2["bull"] and s1["bull"] and s0["bull"] and c2 > o2 and c1 > o1 and c0 > o0 and c1 > c2 and c0 > c1 and s0["body_pct"] >= 50 and s1["body_pct"] >= 50 and s2["body_pct"] >= 50):
        return {"pattern": "THREE_WHITE_SOLDIERS", "name_ar": "الجنود الثلاث البيض", "name_en": "Three White Soldiers", "strength": 4}
    if (s2["bear"] and s1["bear"] and s0["bear"] and c2 < o2 and c1 < o1 and c0 < o0 and c1 < c2 and c0 < c1 and s0["body_pct"] >= 50 and s1["body_pct"] >= 50 and s2["body_pct"] >= 50):
        return {"pattern": "THREE_BLACK_CROWS", "name_ar": "الغربان الثلاث السود", "name_en": "Three Black Crows", "strength": 4}
    if (s1["bear"] and s0["bull"] and o0 <= c1 and c0 >= o1 and s0["body"] > s1["body"]):
        return {"pattern": "ENGULF_BULL", "name_ar": "الابتلاع الشرائي", "name_en": "Bullish Engulfing", "strength": 3}
    if (s1["bull"] and s0["bear"] and o0 >= c1 and c0 <= o1 and s0["body"] > s1["body"]):
        return {"pattern": "ENGULF_BEAR", "name_ar": "الابتلاع البيعي", "name_en": "Bearish Engulfing", "strength": 3}
    if (s0["body_pct"] <= 30 and s0["lower_pct"] >= 60 and s0["upper_pct"] <= 10 and s0["bull"]):
        return {"pattern": "HAMMER", "name_ar": "المطرقة", "name_en": "Hammer", "strength": 2}
    if (s0["body_pct"] <= 30 and s0["upper_pct"] >= 60 and s0["lower_pct"] <= 10 and s0["bear"]):
        return {"pattern": "SHOOTING_STAR", "name_ar": "النجمة الهاوية", "name_en": "Shooting Star", "strength": 2}
    if s0["body_pct"] <= 10:
        return {"pattern": "DOJI", "name_ar": "دوجي", "name_en": "Doji", "strength": 1}
    if s0["body_pct"] >= 85 and s0["upper_pct"] <= 7 and s0["lower_pct"] <= 7:
        direction = "BULL" if s0["bull"] else "BEAR"
        name_ar = "المربوزو الصاعد" if s0["bull"] else "المربوزو الهابط"
        name_en = f"Marubozu {direction}"
        return {"pattern": f"MARUBOZU_{direction}", "name_ar": name_ar, "name_en": name_en, "strength": 3}
    return {"pattern": "NONE", "name_ar": "لا شيء", "name_en": "NONE", "strength": 0}

def get_candle_emoji(pattern):
    emoji_map = {
        "MORNING_STAR": "🌅", "EVENING_STAR": "🌇",
        "THREE_WHITE_SOLDIERS": "💂‍♂️", "THREE_BLACK_CROWS": "🐦‍⬛",
        "ENGULF_BULL": "🟩", "ENGULF_BEAR": "🟥",
        "HAMMER": "🔨", "SHOOTING_STAR": "☄️",
        "DOJI": "➕", "MARUBOZU_BULL": "🚀", "MARUBOZU_BEAR": "💥",
        "NONE": "—"
    }
    return emoji_map.get(pattern, "—")

def build_log_insights(df: pd.DataFrame, ind: dict, price: float):
    adx = float(ind.get("adx") or 0.0)
    plus_di = float(ind.get("plus_di") or 0.0)
    minus_di = float(ind.get("minus_di") or 0.0)
    atr = float(ind.get("atr") or 0.0)
    rsi = float(ind.get("rsi") or 0.0)
    bias = "UP" if plus_di > minus_di else ("DOWN" if minus_di > plus_di else "NEUTRAL")
    regime = "TREND" if adx >= 20 else "RANGE"
    bias_emoji = "🟢" if bias=="UP" else ("🔴" if bias=="DOWN" else "⚪")
    regime_emoji = "📡" if regime=="TREND" else "〰️"
    atr_pct = (atr / max(price or 1e-9, 1e-9)) * 100.0
    if rsi >= 70: rsi_zone = "RSI🔥 Overbought"
    elif rsi <= 30: rsi_zone = "RSI❄️ Oversold"
    else: rsi_zone = "RSI⚖️ Neutral"
    candle_info = detect_candle_pattern(df)
    candle_emoji = get_candle_emoji(candle_info["pattern"])
    return {
        "regime": regime, "regime_emoji": regime_emoji,
        "bias": bias, "bias_emoji": bias_emoji,
        "atr_pct": atr_pct, "rsi_zone": rsi_zone,
        "candle": candle_info, "candle_emoji": candle_emoji
    }

def snapshot(bal,info,ind,spread_bps,reason=None, df=None):
    df = df if df is not None else fetch_ohlcv()
    left_s = time_to_candle_close(df, USE_TV_BAR)
    insights = build_log_insights(df, ind, info.get("price"))
    print(colored("─"*100,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*100,"cyan"))
    print("📈 INDICATORS & CANDLES")
    print(f"   💲 Price {fmt(info.get('price'))}  |  RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))}  lo={fmt(info.get('lo'))}")
    print(f"   🧮 RSI({RSI_LEN})={fmt(ind['rsi'])}   +DI={fmt(ind['plus_di'])}   -DI={fmt(ind['minus_di'])}   DX={fmt(ind['dx'])}   ADX({ADX_LEN})={fmt(ind['adx'])}   ATR={fmt(ind['atr'])} (~{fmt(insights['atr_pct'],2)}%)")
    print(f"   🎯 Signal  ✅ BUY={info['long']}   ❌ SELL={info['short']}   |   🧮 spread_bps={fmt(spread_bps,2)}")
    print(f"   {insights['regime_emoji']} Regime={insights['regime']}   {insights['bias_emoji']} Bias={insights['bias']}   |   {insights['rsi_zone']}")
    candle_info = insights['candle']
    print(f"   🕯️ Candles = {insights['candle_emoji']} {candle_info['name_ar']} / {candle_info['name_en']} (Strength: {candle_info['strength']}/4)")
    if not state["open"] and wait_for_next_signal_side:
        print(colored(f"   ⏳ WAITING — need next {wait_for_next_signal_side.upper()} signal from TradingView Range Filter", "cyan"))
    if state["open"] and state["fakeout_pending"]:
        print(colored(f"   🛡️ FAKEOUT PROTECTION — waiting {state['fakeout_confirm_bars']} bars for confirmation", "yellow"))
    if state["breakout_active"]:
        print(colored(f"   ⚡ BREAKOUT MODE ACTIVE: {state['breakout_direction'].upper()} - Monitoring volatility...", "cyan"))
    if state.get("breakout_score", 0) >= 3.0:
        print(colored(f"   🎯 BREAKOUT SCORE: {state['breakout_score']:.1f}/5.0", "cyan"))
    print(f"   ⏱️ Candle closes in ~ {left_s}s")
    print()
    print("🧭 POSITION & MANAGEMENT")
    print(f"   💰 Balance {fmt(bal,2)} USDT   Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x   PostCloseCooldown={post_close_cooldown}")
    if state["open"]:
        lamp = '🟩 LONG' if state['side']=='long' else '🟥 SHORT'
        trade_mode_display = state.get('trade_mode', 'DETECTING')
        targets_achieved = state.get('profit_targets_achieved', 0)
        print(f"   📌 {lamp}  Entry={fmt(state['entry'])}  Qty={fmt(state['qty'],4)}  Bars={state['bars']}  PnL={fmt(state['pnl'])}")
        print(f"   🎯 Management: Scale-ins={state['scale_ins']}/{SCALE_IN_MAX_STEPS}  Scale-outs={state['scale_outs']}  Trail={fmt(state['trail'])}")
        print(f"   📊 TP1_done={state['tp1_done']}  Breakeven={fmt(state['breakeven'])}  HighestProfit={fmt(state['highest_profit_pct'],2)}%")
        print(f"   🔧 Trade Mode: {trade_mode_display}  Targets Achieved: {targets_achieved}")
        if state.get("_tp0_done"):
            print(colored(f"   💰 TP0: Quick cash taken", "green"))
        if state.get("thrust_locked"):
            print(colored(f"   🔒 THRUST LOCK: Active", "cyan"))
        if state.get("opened_by_breakout"):
            print(colored(f"   ⚡ OPENED BY BREAKOUT", "magenta"))
        if state['last_action']:
            print(f"   🔄 Last Action: {state['last_action']} - {state['action_reason']}")
    else:
        print("   ⚪ FLAT")
    print()
    print("💡 ACTION INSIGHTS")
    if state["open"] and STRATEGY == "smart":
        current_side = state["side"]
        should_scale, step_size, scale_reason = should_scale_in(candle_info, ind, current_side)
        do_scale_out, scale_out_reason = should_scale_out(candle_info, ind, current_side)
        trend_signal = check_trend_confirmation(candle_info, ind, current_side)
        if trend_signal == "CONFIRMED_CONTINUE":
            print(colored(f"   📈 اتجاه مؤكد: استمرار في الترند", "green"))
        elif trend_signal == "POSSIBLE_FAKEOUT":
            print(colored(f"   🕒 اشتباه انعكاس وهمي: في انتظار التأكيد", "yellow"))
        elif trend_signal == "CONFIRMED_REVERSAL":
            print(colored(f"   ⚠️ انعكاس مؤكد: خروج جزئي", "red"))
        else:
            print(colored(f"   ℹ️ لا إشارة اتجاه واضحة", "blue"))
        if should_scale:
            print(colored(f"   ✅ SCALE-IN READY: {scale_reason}", "green"))
        elif do_scale_out:
            print(colored(f"   ⚠️ SCALE-OUT ADVISED: {scale_out_reason}", "yellow"))
        else:
            print(colored(f"   ℹ️ HOLD POSITION: {scale_reason}", "blue"))
        adx = ind.get("adx", 0)
        tp_multiplier, trail_multiplier = get_dynamic_tp_params(adx)
        if tp_multiplier > 1.0:
            current_tp1_pct = TP1_PCT * tp_multiplier * 100.0
            current_trail_activate = TRAIL_ACTIVATE * trail_multiplier * 100.0
            print(colored(f"   🚀 TREND AMPLIFIER ACTIVE: TP={current_tp1_pct:.2f}% • TrailActivate={current_trail_activate:.2f}%", "cyan"))
        trail_mult = get_trail_multiplier({**ind, "price": info.get("price")})
        trail_type = "STRONG" if trail_mult == TRAIL_MULT_STRONG else "MED" if trail_mult == TRAIL_MULT_MED else "CHOP"
        print(f"   🛡️ Trail Multiplier: {trail_mult} ({trail_type})")
        if state.get("_adaptive_trail_mult"):
            adaptive_type = "STRONG" if state["_adaptive_trail_mult"] == TRAIL_MULT_STRONG_ALPHA else "CAUTIOUS" if state["_adaptive_trail_mult"] == TRAIL_MULT_CAUTIOUS_ALPHA else "SCALP"
            print(colored(f"   🎯 ADAPTIVE TRAIL: {state['_adaptive_trail_mult']} ({adaptive_type})", "magenta"))
        trade_mode = state.get('trade_mode')
        if trade_mode:
            if trade_mode == "SCALP":
                targets = SCALP_TARGETS; fracs = SCALP_CLOSE_FRACS; mode_name = "SCALP"
            else:
                targets = TREND_TARGETS; fracs = TREND_CLOSE_FRACS; mode_name = "TREND"
            achieved = state.get('profit_targets_achieved', 0)
            if achieved < len(targets):
                next_target = targets[achieved]; next_frac = fracs[achieved] * 100
                print(colored(f"   🎯 {mode_name} MODE: {achieved}/{len(targets)} targets • Next: TP{achieved+1}@{next_target:.2f}% ({next_frac:.0f}%)", "magenta"))
            else:
                print(colored(f"   ✅ {mode_name} MODE: All targets achieved • Riding trend", "green"))
    else:
        print("   🔄 Waiting for trading signals...")
    print()
    print("📦 RESULTS")
    eff_eq = (bal or 0.0) + compound_pnl
    print(f"   🧮 CompoundPnL {fmt(compound_pnl)}   🚀 EffectiveEq {fmt(eff_eq)} USDT")
    if reason:
        print(colored(f"   ℹ️ WAIT — reason: {reason}","yellow"))
    print(colored("─"*100,"cyan"))

# ------------ Real bar counter ------------
_last_bar_ts = None
def update_bar_counters(df: pd.DataFrame):
    global _last_bar_ts
    if len(df) == 0: return False
    last_ts = int(df["time"].iloc[-1])
    if _last_bar_ts is None:
        _last_bar_ts = last_ts
        return False
    if last_ts != _last_bar_ts:
        _last_bar_ts = last_ts
        if state["open"]:
            state["bars"] += 1
        return True
    return False

# ------------ Decision Loop ------------
def trade_loop():
    global last_signal_id, state, post_close_cooldown, wait_for_next_signal_side, last_close_signal_time, last_open_fingerprint
    sync_from_exchange_once()
    loop_counter = 0
    while True:
        try:
            loop_heartbeat()
            loop_counter += 1
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            new_bar = update_bar_counters(df)
            sanity_check_bar_clock(df)
            info = compute_tv_signals(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            prev_ind = compute_indicators(df.iloc[:-1]) if len(df) >= 2 else ind
            if state["open"] and px:
                state["pnl"] = (px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]
            breakout_exited = handle_breakout_exits(df, ind, prev_ind)
            breakout_entered = False
            if not state["open"] and not breakout_exited:
                breakout_entered = handle_breakout_entries(df, ind, prev_ind, bal, spread_bps)
            if breakout_entered:
                snapshot(bal, info, ind, spread_bps, "BREAKOUT ENTRY - skipping normal logic", df)
                time.sleep(compute_next_sleep(df))
                continue
            if state["breakout_active"]:
                snapshot(bal, info, ind, spread_bps, "BREAKOUT ACTIVE - monitoring exit", df)
                time.sleep(compute_next_sleep(df))
                continue
            if state["open"]:
                if breakout_emergency_protection(ind, prev_ind):
                    snapshot(bal, info, ind, spread_bps, "EMERGENCY LAYER action", df)
                    time.sleep(compute_next_sleep(df))
                    continue

            smart_exit_check(info, ind, df_cached=df, prev_ind_cached=prev_ind)

            sig = "buy" if info["long"] else ("sell" if info["short"] else None)
            reason = None
            if not sig:
                reason = "no signal"
            elif spread_bps is not None and spread_bps > SPREAD_GUARD_BPS:
                reason = f"spread too high ({fmt(spread_bps,2)}bps > {SPREAD_GUARD_BPS})"
            elif post_close_cooldown > 0:
                reason = f"cooldown {post_close_cooldown} bars"

            if state["open"] and sig and (reason is None):
                desired = "long" if sig == "buy" else "short"
                if state["side"] != desired:
                    prev_side = state["side"]
                    close_market_strict("opposite_signal")
                    wait_for_next_signal_side = "sell" if prev_side == "long" else "buy"
                    last_close_signal_time = info["time"]
                    snapshot(bal, info, ind, spread_bps, "waiting next opposite signal", df)
                    time.sleep(compute_next_sleep(df))
                    continue

            if not state["open"] and (reason is None) and sig:
                if wait_for_next_signal_side:
                    if sig != wait_for_next_signal_side:
                        reason = f"waiting opposite signal from Range Filter: need {wait_for_next_signal_side}"
                    else:
                        qty = compute_size(bal, px or info["price"])
                        if qty > 0:
                            open_market(sig, qty, px or info["price"])
                            wait_for_next_signal_side = None
                            last_close_signal_time = None
                            last_open_fingerprint = None
                            last_signal_id = f"{info['time']}:{sig}"
                        else:
                            reason = "qty<=0"
                else:
                    qty = compute_size(bal, px or info["price"])
                    if qty > 0:
                        open_market(sig, qty, px or info["price"])
                        last_open_fingerprint = None
                        last_signal_id = f"{info['time']}:{sig}"
                    else:
                        reason = "qty<=0"

            snapshot(bal, info, ind, spread_bps, reason, df)

            if post_close_cooldown > 0 and not state["open"]:
                post_close_cooldown -= 1

            if loop_counter % 5 == 0:
                save_state()

            sync_consistency_guard()
            sleep_s = compute_next_sleep(df)
            time.sleep(sleep_s)

        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}", "red"))
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# ------------ Keepalive + API ------------
def keepalive_loop():
    url = (SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive: SELF_URL/RENDER_EXTERNAL_URL not set — skipping.", "yellow"))
        return
    sess = requests.Session()
    sess.headers.update({"User-Agent":"rf-pro-bot/keepalive"})
    print(colored(f"KEEPALIVE start → every {KEEPALIVE_SECONDS}s → {url}", "cyan"))
    first_result_printed = False
    while True:
        try:
            r = sess.get(url, timeout=8)
            if not first_result_printed:
                if r.status_code == 200:
                    print(colored("KEEPALIVE ok (first 200)", "green"))
                else:
                    print(colored(f"KEEPALIVE first status={r.status_code}", "yellow"))
                first_result_printed = True
        except Exception as e:
            if not first_result_printed:
                print(colored(f"KEEPALIVE first error: {e}", "red"))
                first_result_printed = True
        time.sleep(max(KEEPALIVE_SECONDS,15))

app = Flask(__name__)
import logging as flask_logging
log = flask_logging.getLogger('werkzeug')
log.setLevel(flask_logging.ERROR)
root_logged = False

@app.route("/")
def home():
    global root_logged
    if not root_logged:
        print("GET / HTTP/1.1 200")
        root_logged = True
    mode = 'LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ RF Bot — {SYMBOL} {INTERVAL} — {mode} — {STRATEGY.upper()} — ADVANCED — TREND AMPLIFIER — HARDENED — TREND CONFIRMATION — INSTANT ENTRY — PURE RANGE FILTER — STRICT EXCHANGE CLOSE — SMART POST-ENTRY MANAGEMENT — CLOSED CANDLE SIGNALS — WAIT FOR NEXT SIGNAL AFTER CLOSE — FAKEOUT PROTECTION — ADVANCED PROFIT TAKING — OPPOSITE SIGNAL WAITING — CORRECTED WICK HARVESTING — BREAKOUT ENGINE — EMERGENCY PROTECTION LAYER — SMART ALPHA PACK"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE,
        "risk_alloc": RISK_ALLOC,
        "price": price_now(),
        "position": state,
        "compound_pnl": compound_pnl,
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "strategy": STRATEGY,
        "bingx_mode": BINGX_POSITION_MODE,
        "advanced_features": {
            "scale_in_steps": state.get("scale_ins", 0),
            "scale_outs": state.get("scale_outs", 0),
            "last_action": state.get("last_action"),
            "action_reason": state.get("action_reason"),
            "highest_profit_pct": state.get("highest_profit_pct", 0),
            "trade_mode": state.get("trade_mode"),
            "profit_targets_achieved": state.get("profit_targets_achieved", 0),
            "breakout_score": state.get("breakout_score", 0),
            "breakout_votes_detail": state.get("breakout_votes_detail", {}),
            "opened_by_breakout": state.get("opened_by_breakout", False),
            "tp0_done": state.get("_tp0_done", False),
            "thrust_locked": state.get("thrust_locked", False)
        },
        "strict_close_enabled": STRICT_EXCHANGE_CLOSE,
        "waiting_for_signal": wait_for_next_signal_side,
        "fakeout_protection": {
            "pending": state.get("fakeout_pending", False),
            "need_side": state.get("fakeout_need_side"),
            "confirm_bars": state.get("fakeout_confirm_bars", 0),
            "started_at": state.get("fakeout_started_at")
        },
        "breakout_engine": {
            "active": state.get("breakout_active", False),
            "direction": state.get("breakout_direction"),
            "entry_price": state.get("breakout_entry_price")
        },
        "emergency_protection": {
            "enabled": EMERGENCY_PROTECTION_ENABLED,
            "policy": EMERGENCY_POLICY,
            "harvest_frac": EMERGENCY_HARVEST_FRAC
        },
        "profit_taking": {
            "scalp_targets": SCALP_TARGETS,
            "trend_targets": TREND_TARGETS,
            "scale_in_disabled": True
        },
        "smart_alpha_pack": {
            "tp0_profit_pct": TP0_PROFIT_PCT,
            "tp0_close_frac": TP0_CLOSE_FRAC,
            "thrust_atr_bars": THRUST_ATR_BARS,
            "adaptive_trail_enabled": True
        }
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "loop_stall_s": time.time() - last_loop_ts,
        "mode": "live" if MODE_LIVE else "paper",
        "open": state["open"],
        "side": state["side"],
        "qty": state["qty"],
        "compound_pnl": compound_pnl,
        "consecutive_errors": _consec_err,
        "timestamp": datetime.utcnow().isoformat(),
        "strict_close_enabled": STRICT_EXCHANGE_CLOSE,
        "trade_mode": state.get("trade_mode"),
        "profit_targets_achieved": state.get("profit_targets_achieved", 0),
        "waiting_for_signal": wait_for_next_signal_side,
        "fakeout_protection_active": state.get("fakeout_pending", False),
        "breakout_active": state.get("breakout_active", False),
        "emergency_protection_enabled": EMERGENCY_PROTECTION_ENABLED,
        "breakout_score": state.get("breakout_score", 0),
        "tp0_done": state.get("_tp0_done", False),
        "thrust_locked": state.get("thrust_locked", False),
        "smart_alpha_features": True
    }), 200

@app.route("/ping")
def ping(): return "pong", 200

# --- Optional toggles from env (لا تغيّر المنطق لو مش مستخدمة) ---
ENABLE_SMART_HARVEST = bool(int(os.getenv("ENABLE_SMART_HARVEST", "1")))
ENABLE_PATIENCE_GUARD = bool(int(os.getenv("ENABLE_PATIENCE_GUARD", "1")))

def _validate_market_specs():
    # حارس بسيط قبل الإقلاع (اختياري)
    assert AMT_PREC is None or (isinstance(AMT_PREC, int) and AMT_PREC >= 0), "AMT_PREC invalid"
    assert LOT_MIN is None or (isinstance(LOT_MIN, (int, float)) and LOT_MIN >= 0), "LOT_MIN invalid"
    assert LOT_STEP is None or (isinstance(LOT_STEP, (int, float)) and LOT_STEP >= 0), "LOT_STEP invalid"

# -------- Boot --------
if __name__ == "__main__":
    setup_file_logging()
    _validate_market_specs()

    print(colored("✅ Starting HARDENED Flask server with ALL PROTECTION LAYERS & SMART ALPHA PACK...", "green"))

    load_state()

    print(colored("🛡️ Watchdog started", "cyan"))
    threading.Thread(target=watchdog_check, daemon=True).start()

    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()

    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
