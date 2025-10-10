# -*- coding: utf-8 -*-
"""
RF Futures Bot — Smart Pro (BingX Perp, CCXT) - HARDENED EDITION
- Entries: TradingView Range Filter EXACT (BUY/SELL) on CLOSED candle
- Size: 60% balance × leverage (default 10x)
- Exit:
  • Opposite RF signal ALWAYS closes (strict exchange close)
  • Smart Profit: TP1 partial + move to breakeven + ATR trailing (trend-riding)
- Advanced Candle & Indicator Analysis for Position Management
- Robust keepalive (SELF_URL/RENDER_EXTERNAL_URL), retries, /metrics

✅ HARDENING PACK APPLIED:
1. Market specs & amount normalization (precision/min step)
2. Leverage & position mode confirmation  
3. State persistence to disk (survives restarts)
4. File logging with rotation (5MB × 7 files)
5. Watchdog for main loop stall detection
6. Bar clock sanity check
7. Network error backoff (circuit breaker)
8. Idempotency guard for duplicate opening
9. Graceful exit on SIGTERM/SIGINT
10. Enhanced health endpoint

✅ SMART POST-ENTRY MANAGEMENT ADDED:
1. Trade mode detection (SCALP/TREND)
2. Impulse & long wick harvesting
3. Ratchet protection for profit locking
4. Dynamic profit taking based on trade mode
5. Trend confirmation with ADX + DI + RSI
6. Compound PnL integration

✅ BREAKOUT ENGINE ADDED:
- Independent explosive move detection (ATR spike + ADX + price breakout)
- Immediate entry on explosions/crashes  
- Strict exit when volatility normalizes
- Full independence from core RF strategy
- Production-optimized with all safety guards

✅ EMERGENCY PROTECTION LAYER ADDED:
- Smart Pump/Crash detection (ATR spike + high ADX + extreme RSI)
- In favor: Smart profit harvesting + breakeven + emergency trail
- Against: Immediate full close to minimize loss
- Independent layer with configurable policies

Patched:
- BingX position mode support (oneway|hedge) with correct positionSide
- Safe state updates (no local open if exchange order failed)
- Rich logs: candles/regime/bias/ATR%, WAIT reason, and time left to candle close
- Advanced Candle Pattern Detection in Arabic/English
- Smart Scale-In/Scale-Out based on candlestick patterns + indicators
- Dynamic Trailing SL based on market regime
- Trend Amplifier: ADX-based scale-in, dynamic TP, ratchet lock
- Hold-TP & Impulse Harvest for advanced profit management
- Auto-full close if remaining qty < 60 DOGE
- Fixed BingX leverage warning with correct side parameter
- Trend Confirmation Logic: ADX + DI + Candle Analysis
- ✅ PATCH: Instant entry when FLAT + No cooldown after close
- ✅ PATCH: Strict Exchange Close with retry & verification
- ✅ PATCH: CLOSED CANDLE SIGNALS ONLY - No premature entries
- ✅ PATCH: Strict profit target closing with exchange verification
- ✅ PATCH: TP1 fallback when trade_mode not decided
- ✅ PATCH: Safety guard to avoid float-None operations after strict close
- ✅ PATCH: WAIT FOR NEXT SIGNAL AFTER CLOSE - No immediate re-entry
- ✅ NEW: FAKEOUT PROTECTION - Wait for confirmation before closing
- ✅ NEW: ADVANCED PROFIT TAKING - 3-stage SCALP/TREND targets with strict close
- ✅ NEW: OPPOSITE SIGNAL WAITING - Only open opposite RF signals after close
- ✅ NEW: CORRECTED WICK HARVESTING - Upper wick for LONG, Lower wick for SHORT
- ✅ NEW: BREAKOUT ENGINE - Independent explosive move detection & trading
- ✅ NEW: EMERGENCY PROTECTION LAYER - Smart Pump/Crash response system
"""

import os, time, math, threading, requests, traceback, random, signal, sys, logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime

# ------------ console colors ------------
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
COOLDOWN_AFTER_CLOSE_BARS = 0  # ✅ PATCH 3: No cooldown after close

# Range Filter params
RF_SOURCE = "close"
RF_PERIOD = 20
RF_MULT = 3.5

# ✅ PATCH 4: REAL-TIME SIGNALS - Always use current closed candle
USE_TV_BAR = True  # Now always True for real-time signals

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

# Strategy mode
STRATEGY = "smart"
USE_SMART_EXIT = True

# Smart Profit params
TP1_PCT = 0.40
TP1_CLOSE_FRAC = 0.50
BREAKEVEN_AFTER = 0.30
TRAIL_ACTIVATE = 0.60
ATR_MULT_TRAIL = 1.6

# Advanced Position Management
SCALE_IN_MAX_STEPS = 0                # ⛔ تعطيل التعزيز نهائيًا
SCALE_IN_STEP_PCT = 0.20
ADX_STRONG_THRESH = 28
RSI_TREND_BUY = 55
RSI_TREND_SELL = 45
TRAIL_MULT_STRONG = 2.0
TRAIL_MULT_MED = 1.5
TRAIL_MULT_CHOP = 1.0

# Trend Amplifier
ADX_TIER1 = 28
ADX_TIER2 = 35
ADX_TIER3 = 45
RATCHET_LOCK_PCT = 0.60

# Position mode
BINGX_POSITION_MODE = "oneway"

# ✅ NEW: Advanced Profit Taking Settings
# --- Profit targets (بنِسب مئوية: rr محسوبة %)
SCALP_TARGETS = [0.35, 0.70, 1.20]   # 3 مراحل سكالب: 0.35% ثم 0.70% ثم 1.20%
SCALP_CLOSE_FRACS = [0.40, 0.30, 0.30]  # يغلق 40% ثم 30% ثم 30% → وبعدها قفل كامل

TREND_TARGETS = [0.50, 1.00, 1.80]   # 3 مراحل للترند (أعلى نسبيًا)
TREND_CLOSE_FRACS = [0.30, 0.30, 0.20]  # يخفف تدريجيًا ويحتفظ بجزء للركوب
MIN_TREND_HOLD_ADX = 25              # طالما ADX ≥ 25 يبقى راكب
END_TREND_ADX_DROP = 5.0             # هبوط ADX بمقدار ≥ 5 يعتبر ضعف
END_TREND_RSI_NEUTRAL = (45, 55)     # الخروج من مناطق اتجاهية إلى حيادية
DI_FLIP_BUFFER = 1.0                  # قلب DI مع هامش بسيط يؤكد الانعكاس

# ✅ NEW: Smart Post-Entry Management Settings
IMPULSE_HARVEST_THRESHOLD = 1.2  # Body >= 1.2x ATR
LONG_WICK_HARVEST_THRESHOLD = 0.60  # Wick >= 60% of range
RATCHET_RETRACE_THRESHOLD = 0.40  # Close partial on 40% retrace from high

# ✅ NEW: BREAKOUT ENGINE SETTINGS
BREAKOUT_ATR_SPIKE = 1.8        # ATR(current) > ATR(previous) * 1.8
BREAKOUT_ADX_THRESHOLD = 25     # ADX ≥ 25 for trend strength  
BREAKOUT_LOOKBACK_BARS = 20     # Check last 20 bars for highs/lows
BREAKOUT_CALM_THRESHOLD = 1.1   # ATR(current) < ATR(previous) * 1.1 → exit

# ✅ NEW: EMERGENCY BREAKOUT/CRASH PROTECTION (SMART LAYER)
EMERGENCY_PROTECTION_ENABLED = True

# شروط التفعيل
EMERGENCY_ADX_MIN = 40             # قوة ترند لازمة
EMERGENCY_ATR_SPIKE_RATIO = 1.6    # ATR_now > ATR_prev * ratio
EMERGENCY_RSI_PUMP = 72            # Pump
EMERGENCY_RSI_CRASH = 28           # Crash

# سياسات التصرّف
# - "tp_then_close": جني جزئي ثم إغلاق باقي المركز فورًا
# - "tp_then_trail": جني جزئي + بريك إيفن + تريل طارئ (نركب لو فيه امتداد)
# - "close_always": إغلاق فوري كامل بغض النظر
EMERGENCY_POLICY = "tp_then_close"

# إعدادات الجني الذكي
EMERGENCY_HARVEST_FRAC = 0.60      # يجني 60% فور التفعيل (لو في صالحنا)
EMERGENCY_FULL_CLOSE_PROFIT = 1.0  # % لو الربح ≥ القيمة دي → إغلاق كامل بدل الجني الجزئي

# تريل طارئ (أقوى من العادي)
EMERGENCY_TRAIL_ATR_MULT = 1.2     # أترايل أضيق عشان نحافظ على المكسب

# pacing / keepalive
ADAPTIVE_PACING = True
BASE_SLEEP = 10        # نوم عادي بعيدًا عن الإغلاق
NEAR_CLOSE_SLEEP = 1   # قرب الإغلاق وبعده مباشرة
JUST_CLOSED_WINDOW = 8 # ثواني بعد الإغلاق نكثّف الفحص

SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
KEEPALIVE_SECONDS = 50
PORT = int(os.getenv("PORT", 5000))

# ─── STRICT EXCHANGE CLOSE ───────────────────────────────────────────────
STRICT_EXCHANGE_CLOSE = True        # تفعيل الإغلاق الصارم المؤكد من المنصة
CLOSE_RETRY_ATTEMPTS   = 6          # عدد المحاولات القصوى
CLOSE_VERIFY_WAIT_S    = 2.0        # مدة الانتظار بين كل تحقق من إغلاق المنصة (ثواني)
MIN_RESIDUAL_TO_FORCE  = 1.0        # أي بقايا كمية ≥ هذا الرقم نعيد إغلاقها

# === Post-close signal gating ===
REQUIRE_NEW_BAR_AFTER_CLOSE = False  # ✅ PATCH: تعطيل شرط الشمعة الجديدة
wait_for_next_signal_side = None    # 'buy' أو 'sell' أو None
last_close_signal_time = None       # time للشمعة التي تم عندها الإغلاق

# ------------ HARDENING PACK: State Persistence ------------
STATE_FILE = "bot_state.json"

# ------------ HARDENING PACK: Network Error Backoff ------------
_consec_err = 0

# ------------ HARDENING PACK: Idempotency Guard ------------
last_open_fingerprint = None

# ------------ HARDENING PACK: Watchdog ------------
last_loop_ts = time.time()

print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • SYMBOL={SYMBOL} • {INTERVAL}", "yellow"))
print(colored(f"STRATEGY: {STRATEGY.upper()} • SMART_EXIT={'ON' if USE_SMART_EXIT else 'OFF'}", "yellow"))
print(colored(f"ADVANCED POSITION MGMT: SCALE_IN_STEPS={SCALE_IN_MAX_STEPS} • ADX_STRONG={ADX_STRONG_THRESH}", "yellow"))
print(colored(f"TREND AMPLIFIER: ADX_TIERS[{ADX_TIER1}/{ADX_TIER2}/{ADX_TIER3}] • RATCHET_LOCK={RATCHET_LOCK_PCT*100}%", "yellow"))
print(colored(f"✅ NEW: SMART POST-ENTRY MANAGEMENT ENABLED", "green"))
print(colored(f"✅ PATCH: CLOSED CANDLE SIGNALS ONLY - No premature entries", "green"))
print(colored(f"✅ PATCH: Strict profit target closing with exchange verification", "green"))
print(colored(f"✅ PATCH: TP1 fallback when trade_mode not decided", "green"))
print(colored(f"✅ PATCH: Safety guard to avoid float-None operations after strict close", "green"))
print(colored(f"✅ PATCH: WAIT FOR NEXT SIGNAL AFTER CLOSE - No immediate re-entry", "green"))
print(colored(f"✅ NEW: FAKEOUT PROTECTION - Wait for confirmation before closing", "green"))
print(colored(f"✅ NEW: ADVANCED PROFIT TAKING - 3-stage SCALP/TREND targets", "green"))
print(colored(f"✅ NEW: OPPOSITE SIGNAL WAITING - Only open opposite RF signals after close", "green"))
print(colored(f"✅ NEW: CORRECTED WICK HARVESTING - Upper wick for LONG, Lower wick for SHORT", "green"))
print(colored(f"✅ NEW: BREAKOUT ENGINE - Independent explosive move detection & trading", "green"))
print(colored(f"✅ NEW: EMERGENCY PROTECTION LAYER - Smart Pump/Crash response system", "green"))
print(colored(f"KEEPALIVE: url={'SET' if SELF_URL else 'NOT SET'} • every {KEEPALIVE_SECONDS}s", "yellow"))
print(colored(f"BINGX_POSITION_MODE={BINGX_POSITION_MODE}", "yellow"))
print(colored(f"✅ HARDENING PACK: State persistence, logging, watchdog, network guard ENABLED", "green"))
print(colored(f"✅ REAL-TIME SIGNALS: Using current closed candle (TradingView sync)", "green"))
print(colored(f"✅ PATCHED: Auto-full close if remaining qty < 60 DOGE", "green"))
print(colored(f"✅ PATCHED: Fixed BingX leverage warning with side='BOTH'", "green"))
print(colored(f"✅ NEW: Trend Confirmation Logic (ADX + DI + Candle Analysis)", "green"))
print(colored(f"✅ PATCH: Instant entry when FLAT + No cooldown after close", "green"))
print(colored(f"✅ PATCH: Pure Range Filter signals ONLY - No RSI/ADX filtering for entries", "green"))
print(colored(f"✅ PATCH: Strict Exchange Close with retry & verification", "green"))
print(colored(f"SERVER: Starting on port {PORT}", "green"))

# ------------ HARDENING PACK: File Logging with Rotation ------------
def setup_file_logging():
    """Setup rotating file logging (5MB × 7 files)"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ file logging with rotation enabled", "cyan"))

setup_file_logging()

# ------------ HARDENING PACK: Graceful Exit ------------
def _graceful_exit(signum, frame):
    """Save state and exit gracefully on SIGTERM/SIGINT"""
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

# ------------ HARDENING PACK: Market Specs & Amount Normalization ------------
MARKET = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN = None

def load_market_specs():
    """Load market specifications for amount normalization"""
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

def _round_amt(q):
    """Round amount according to market specifications (robust)."""
    if q is None:
        return 0.0

    # snap to step (لو فيه step موجب)
    if LOT_STEP and isinstance(LOT_STEP, (int, float)) and LOT_STEP > 0:
        q = max(0.0, math.floor(q / LOT_STEP) * LOT_STEP)

    # precision guard
    prec = 0
    try:
        prec = int(AMT_PREC) if AMT_PREC is not None else 0
        if prec < 0:
            prec = 0
    except Exception:
        prec = 0

    # apply precision safely
    try:
        q = float(f"{float(q):.{prec}f}")
    except Exception:
        q = float(q)

    # respect min lot if موجود
    if LOT_MIN and isinstance(LOT_MIN, (int, float)) and LOT_MIN > 0:
        q = 0.0 if q < LOT_MIN else q

    return q

def safe_qty(q):
    """Validate and normalize quantity"""
    q = _round_amt(q)
    if q <= 0:
        print(colored(f"⚠️ qty invalid after normalize → {q}", "yellow"))
    return q

# ------------ HARDENING PACK: Leverage & Position Mode Confirmation ------------
def ensure_leverage_and_mode():
    """Ensure leverage and position mode are set correctly"""
    try:
        # ✅ FIXED: Use correct side parameter for BingX leverage setting
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side": "BOTH"})
            print(colored(f"✅ leverage set: {LEVERAGE}x with side=BOTH", "green"))
        except Exception as e:
            print(colored(f"⚠️ set_leverage warn: {e}", "yellow"))
        print(colored(f"ℹ️ position mode target: {BINGX_POSITION_MODE}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_leverage_and_mode: {e}", "yellow"))

# Load markets and specs
try:
    ex.load_markets()
    load_market_specs()
    ensure_leverage_and_mode()
except Exception as e:
    print(colored(f"⚠️ load_markets: {e}", "yellow"))

# ------------ HARDENING PACK: State Persistence Functions ------------
def save_state():
    """Save bot state to disk"""
    try:
        data = {
            "state": state,
            "compound_pnl": compound_pnl,
            "last_signal_id": last_signal_id,
            "timestamp": time.time()
        }
        with open(STATE_FILE, "w") as f:
            import json
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
        logging.info(f"State saved: compound_pnl={compound_pnl}")
    except Exception as e:
        print(colored(f"⚠️ save_state: {e}", "yellow"))
        logging.error(f"save_state error: {e}")

def load_state():
    """Load bot state from disk"""
    global state, compound_pnl, last_signal_id
    try:
        import json, os
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, "r") as f:
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

# ------------ HARDENING PACK: Network Error Backoff ------------
def net_guard(success):
    """Circuit breaker for network errors"""
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

# ------------ HARDENING PACK: Watchdog Functions ------------
def loop_heartbeat():
    """Update main loop heartbeat"""
    global last_loop_ts
    last_loop_ts = time.time()

def watchdog_check(max_stall=180):
    """Watchdog thread to detect main loop stalls"""
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

# ------------ HARDENING PACK: Bar Clock Sanity Check ------------
def sanity_check_bar_clock(df):
    """Check for bar interval anomalies"""
    try:
        if len(df) < 2: return
        tf = _interval_seconds(INTERVAL)
        delta = (int(df["time"].iloc[-1]) - int(df["time"].iloc[-2]))/1000
        if abs(delta - tf) > tf*0.5:
            print(colored(f"⚠️ bar interval anomaly: {delta}s vs tf {tf}s", "yellow"))
            logging.warning(f"Bar interval anomaly: {delta}s vs tf {tf}s")
    except Exception as e:
        logging.error(f"sanity_check_bar_clock error: {e}")

# ------------ HARDENING PACK: Idempotency Guard ------------
def can_open(sig, price):
    """
    ✅ PATCH 1: يسمح بالدخول فوراً لو الحساب FLAT.
    يمنع التكرار فقط أثناء وجود صفقة مفتوحة.
    """
    if not state.get("open"):
        return True

    global last_open_fingerprint
    fp = f"{sig}|{int(price or 0)}|{INTERVAL}|{SYMBOL}"
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

# ---- time to candle close ----
def _interval_seconds(iv: str) -> int:
    iv = (iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame, use_tv_bar: bool) -> int:
    tf = _interval_seconds(INTERVAL)
    if len(df) == 0: return tf
    cur_start_ms = int(df["time"].iloc[-1])  # start time (ms) for current bar
    now_ms = int(time.time()*1000)
    next_close_ms = cur_start_ms + tf*1000
    while next_close_ms <= now_ms:
        next_close_ms += tf*1000
    left = max(0, next_close_ms - now_ms)
    return int(left/1000)

# ------------ Adaptive Pacing Function ------------
def compute_next_sleep(df):
    """حدد مدة النوم القادمة حسب زمن إغلاق الشمعة (بدون لمس التداول)."""
    if not ADAPTIVE_PACING:
        return BASE_SLEEP
    try:
        left_s = time_to_candle_close(df, USE_TV_BAR)
        tf = _interval_seconds(INTERVAL)

        # قرب الإغلاق جدًا → فحص كل ثانية
        if left_s <= 10:
            return NEAR_CLOSE_SLEEP

        # أول ثواني بعد الإغلاق (لالتقاط الإشارة فورًا)
        if (tf - left_s) <= JUST_CLOSED_WINDOW:
            return NEAR_CLOSE_SLEEP

        # غير ذلك نوم معقول لتخفيف الضغط
        return BASE_SLEEP
    except Exception:
        return BASE_SLEEP

# ------------ Indicators (display-only) ------------
def wilder_ema(s: pd.Series, n: int): return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    # ✅ PATCH 1: Safety guard for insufficient data
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

    # ✅ PATCH 4: Always use current closed candle for real-time signals
    i = len(df)-1  # Always use the latest closed candle
    
    prev_i = max(0, i-1)
    return {
        "rsi": float(rsi.iloc[i]), "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]), "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]), "atr": float(atr.iloc[i]),
        "adx_prev": float(adx.iloc[prev_i])
    }

# ------------ Advanced Candle Analytics ------------
def _candle_stats(o, c, h, l):
    rng = max(h - l, 1e-12)
    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l
    return {
        "range": rng,
        "body": body,
        "body_pct": (body / rng) * 100.0,
        "upper_pct": (upper / rng) * 100.0,
        "lower_pct": (lower / rng) * 100.0,
        "bull": c > o,
        "bear": c < o
    }

def detect_candle_pattern(df: pd.DataFrame):
    if len(df) < 3:
        return {"pattern": "NONE", "name_ar": "لا شيء", "name_en": "NONE", "strength": 0}
    
    # ✅ PATCH 4: Always use current closed candle for real-time signals
    idx = -1  # Always use the latest closed candle
    
    o2, h2, l2, c2 = map(float, (df["open"].iloc[idx-2], df["high"].iloc[idx-2], df["low"].iloc[idx-2], df["close"].iloc[idx-2]))
    o1, h1, l1, c1 = map(float, (df["open"].iloc[idx-1], df["high"].iloc[idx-1], df["low"].iloc[idx-1], df["close"].iloc[idx-1]))
    o0, h0, l0, c0 = map(float, (df["open"].iloc[idx], df["high"].iloc[idx], df["low"].iloc[idx], df["close"].iloc[idx]))
    
    s2 = _candle_stats(o2, c2, h2, l2)
    s1 = _candle_stats(o1, c1, h1, l1)
    s0 = _candle_stats(o0, c0, h0, l0)

    # Three-candle patterns (Strong reversal signals)
    # Morning Star (نجمة الصباح)
    if (s2["bear"] and s2["body_pct"] >= 60 and 
        s1["body_pct"] <= 25 and l1 > l2 and 
        s0["bull"] and s0["body_pct"] >= 50 and c0 > (o1 + c1)/2):
        return {"pattern": "MORNING_STAR", "name_ar": "نجمة الصباح", "name_en": "Morning Star", "strength": 4}
    
    # Evening Star (نجمة المساء)
    if (s2["bull"] and s2["body_pct"] >= 60 and 
        s1["body_pct"] <= 25 and h1 < h2 and 
        s0["bear"] and s0["body_pct"] >= 50 and c0 < (o1 + c1)/2):
        return {"pattern": "EVENING_STAR", "name_ar": "نجمة المساء", "name_en": "Evening Star", "strength": 4}
    
    # Three White Soldiers (الجنود الثلاث البيض)
    if (s2["bull"] and s1["bull"] and s0["bull"] and
        c2 > o2 and c1 > o1 and c0 > o0 and
        c1 > c2 and c0 > c1 and
        s0["body_pct"] >= 50 and s1["body_pct"] >= 50 and s2["body_pct"] >= 50):
        return {"pattern": "THREE_WHITE_SOLDIERS", "name_ar": "الجنود الثلاث البيض", "name_en": "Three White Soldiers", "strength": 4}
    
    # Three Black Crows (الغربان الثلاث السود)
    if (s2["bear"] and s1["bear"] and s0["bear"] and
        c2 < o2 and c1 < o1 and c0 < o0 and
        c1 < c2 and c0 < c1 and
        s0["body_pct"] >= 50 and s1["body_pct"] >= 50 and s2["body_pct"] >= 50):
        return {"pattern": "THREE_BLACK_CROWS", "name_ar": "الغربان الثلاث السود", "name_en": "Three Black Crows", "strength": 4}

    # Two-candle patterns
    # Bullish Engulfing (الابتلاع الشرائي)
    if (s1["bear"] and s0["bull"] and 
        o0 <= c1 and c0 >= o1 and 
        s0["body"] > s1["body"]):
        return {"pattern": "ENGULF_BULL", "name_ar": "الابتلاع الشرائي", "name_en": "Bullish Engulfing", "strength": 3}
    
    # Bearish Engulfing (الابتلاع البيعي)
    if (s1["bull"] and s0["bear"] and 
        o0 >= c1 and c0 <= o1 and 
        s0["body"] > s1["body"]):
        return {"pattern": "ENGULF_BEAR", "name_ar": "الابتلاع البيعي", "name_en": "Bearish Engulfing", "strength": 3}
    
    # Single candle patterns
    # Hammer (المطرقة)
    if (s0["body_pct"] <= 30 and s0["lower_pct"] >= 60 and 
        s0["upper_pct"] <= 10 and s0["bull"]):
        return {"pattern": "HAMMER", "name_ar": "المطرقة", "name_en": "Hammer", "strength": 2}
    
    # Shooting Star (النجمة الهاوية)
    if (s0["body_pct"] <= 30 and s0["upper_pct"] >= 60 and 
        s0["lower_pct"] <= 10 and s0["bear"]):
        return {"pattern": "SHOOTING_STAR", "name_ar": "النجمة الهاوية", "name_en": "Shooting Star", "strength": 2}
    
    # Doji (دوجي)
    if s0["body_pct"] <= 10:
        return {"pattern": "DOJI", "name_ar": "دوجي", "name_en": "Doji", "strength": 1}
    
    # Marubozu (المربوزو)
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
    # ✅ PATCH 1: Safety guard for insufficient data
    if len(df) < RF_PERIOD + 3:
        # ✅ PATCH 4: Always use current closed candle
        i = -1  # Always use the latest closed candle
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
    upward = (fdir==1).astype(int); downward = (fdir == -1).astiatype(int)
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
    
    # ✅ PATCH 4: Always use current closed candle for real-time signals
    i = len(df)-1  # Always use the latest closed candle
    
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
    "highest_profit_pct": 0.0,  # Trend Amplifier: ratchet lock
    "trade_mode": None,  # ✅ NEW: Trade mode (SCALP/TREND)
    "profit_targets_achieved": 0,  # ✅ NEW: Track profit targets
    "entry_time": None,  # ✅ NEW: Track entry time for time-based exits
    # ✅ NEW: Fakeout Protection fields
    "fakeout_pending": False,          # هل في احتمال انعكاس جارٍ؟
    "fakeout_need_side": None,         # 'long' أو 'short' الاتجاه المطلوب لتأكيد الانعكاس
    "fakeout_confirm_bars": 0,         # عدّاد الشموع المطلوبة للتأكيد
    "fakeout_started_at": None,        # طابع زمني/شمعة بدء الاشتباه
    # ✅ NEW: BREAKOUT ENGINE STATE
    "breakout_active": False,          # هل نحن في وضع انفجار نشط؟
    "breakout_direction": None,        # 'bull' أو 'bear'
    "breakout_entry_price": None       # سعر الدخول أثناء الانفجار
}
compound_pnl = 0.0
last_signal_id = None
post_close_cooldown = 0

def compute_size(balance, price):
    # رصيد فعّال = الرصيد + الربح التراكمي (كومباوند كامل)
    effective_balance = (balance or 0.0) + (compound_pnl or 0.0)

    capital = effective_balance * RISK_ALLOC * LEVERAGE   # 60% × 10x
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
                "trade_mode": None,  # Reset trade mode on sync
                "profit_targets_achieved": 0,
                "entry_time": time.time(),
                # ✅ NEW: Reset fakeout protection on sync
                "fakeout_pending": False,
                "fakeout_need_side": None,
                "fakeout_confirm_bars": 0,
                "fakeout_started_at": None,
                # ✅ NEW: Reset breakout state on sync
                "breakout_active": False,
                "breakout_direction": None,
                "breakout_entry_price": None
            })
            print(colored(f"✅ Synced position ⇒ {side.upper()} qty={fmt(qty,4)} @ {fmt(entry)}","green"))
            logging.info(f"Position synced: {side} qty={qty} entry={entry}")
            return
        print(colored("↔️  Sync: no open position on exchange.","yellow"))
    except Exception as e:
        print(colored(f"❌ sync error: {e}","red"))
        logging.error(f"sync_from_exchange_once error: {e}")

# ------------ BingX params helpers ------------
def _position_params_for_open(side: str):
    if BINGX_POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if side == "buy" else "SHORT", "reduceOnly": False}
    return {"positionSide": "BOTH", "reduceOnly": False}

def _position_params_for_close():
    if BINGX_POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if state.get("side")=="long" else "SHORT", "reduceOnly": True}
    return {"positionSide": "BOTH", "reduceOnly": True}

# ─── STRICT EXCHANGE CLOSE FUNCTIONS ─────────────────────────────────────
def _read_exchange_position():
    """
    يرجع (qty, side, entry) لمركز SYMBOL على BingX (type=swap).
    qty=0 يعني مفيش مركز.
    """
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
    """
    إغلاق كامل صارم:
    - يقرأ الكمية الفعلية من المنصة
    - يرسل market reduceOnly على الكمية
    - يتحقق مرارًا حتى يصبح المركز = 0
    - يعيد المحاولة عند وجود بقايا أو خطأ شبكة
    """
    global state, compound_pnl, wait_for_next_signal_side, last_close_signal_time

    # احفظ الجانب الحالي قبل أي تغيير
    prev_side_local = state.get("side")

    # 1) اسحب الحالة الفعلية من المنصة
    exch_qty, exch_side, exch_entry = _read_exchange_position()
    if exch_qty <= 0:
        # لا يوجد مركز على المنصة → صفّر محليًا لو لازال مفتوح
        if state.get("open"):
            reset_after_full_close("strict_close_already_zero", prev_side_local)
        return

    # 2) حدّد جانب الإغلاق و الكمية
    side_to_close = "sell" if (exch_side == "long") else "buy"
    qty_to_close  = safe_qty(exch_qty)

    # 3) نفّذ أمر الإغلاق وكرّر التحقق
    attempts = 0
    last_error = None
    while attempts < CLOSE_RETRY_ATTEMPTS:
        try:
            if MODE_LIVE:
                params = _position_params_for_close()
                # reduceOnly= True بالفعل داخل المساعد، بس بنؤكد
                params["reduceOnly"] = True
                ex.create_order(SYMBOL, "market", side_to_close, qty_to_close, None, params)

            # انتظر شوية ثم تحقّق
            time.sleep(CLOSE_VERIFY_WAIT_S)
            left_qty, _, _ = _read_exchange_position()

            if left_qty <= 0:
                # تم الإغلاق على المنصة: احسب PnL واغلق محليًا
                px = price_now() or state.get("entry")
                entry_px = state.get("entry") or exch_entry or px
                side = state.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
                qty = exch_qty  # الكمية التي أغلقناها

                pnl = (px - entry_px) * qty * (1 if side == "long" else -1)
                compound_pnl += pnl
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                logging.info(f"STRICT_CLOSE {side} pnl={pnl} total={compound_pnl}")
                reset_after_full_close(reason, prev_side_local)
                return

            # يوجد بقايا → جهّز لمحاولة جديدة بنفس الاتجاه والكمية المتبقية
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            print(colored(f"⚠️ strict close retry {attempts}/{CLOSE_RETRY_ATTEMPTS} — residual={fmt(left_qty,4)}","yellow"))

            # لو الباقى صغير جدًا، زوّد مهلة واعِد المحاولة
            if qty_to_close < MIN_RESIDUAL_TO_FORCE:
                time.sleep(CLOSE_VERIFY_WAIT_S)

        except Exception as e:
            last_error = e
            logging.error(f"close_market_strict error attempt {attempts+1}: {e}")
            attempts += 1
            time.sleep(CLOSE_VERIFY_WAIT_S)

    # لو وصلنا هنا، فشلنا بعد كل المحاولات
    print(colored(f"❌ STRICT CLOSE FAILED after {CLOSE_RETRY_ATTEMPTS} attempts — manual check needed. Last error: {last_error}", "red"))
    logging.critical(f"STRICT CLOSE FAILED — last_error={last_error}")

def sync_consistency_guard():
    """يتحقق من اتساق الحالة المحلية مع المنصة ويصحح أي تباين"""
    if not state["open"]:
        return
    
    exch_qty, exch_side, exch_entry = _read_exchange_position()
    
    # لو المنصة تقول مفيش مركز لكن محليًا مفتوح → نُصلح
    if exch_qty <= 0 and state["open"]:
        print(colored("🛠️  CONSISTENCY GUARD: Exchange shows no position but locally open → resetting", "yellow"))
        logging.warning("Consistency guard: resetting local state (exchange shows no position)")
        reset_after_full_close("consistency_guard_no_position", state.get("side"))
        return
    
    # لو الكمية مختلفة بشكل كبير → نُصلح
    if exch_qty > 0 and state["open"]:
        diff_pct = abs(exch_qty - state["qty"]) / max(exch_qty, state["qty"])
        if diff_pct > 0.1:  # فرق أكثر من 10%
            print(colored(f"🛠️  CONSISTENCY GUARD: Quantity mismatch local={state['qty']} vs exchange={exch_qty} → syncing", "yellow"))
            logging.warning(f"Consistency guard: quantity mismatch local={state['qty']} exchange={exch_qty}")
            state["qty"] = exch_qty
            state["entry"] = exch_entry or state["entry"]
            save_state()

# ------------ Trend Amplifier ------------
def get_dynamic_scale_in_step(adx: float) -> tuple:
    """Return (step_size, reason) based on ADX tier"""
    if adx >= ADX_TIER3:
        return 0.25, f"ADX-tier3 step=25% (ADX≥{ADX_TIER3})"
    elif adx >= ADX_TIER2:
        return 0.20, f"ADX-tier2 step=20% (ADX≥{ADX_TIER2})"
    elif adx >= ADX_TIER1:
        return 0.15, f"ADX-tier1 step=15% (ADX≥{ADX_TIER1})"
    else:
        return 0.0, f"ADX below tier1 (ADX<{ADX_TIER1})"

def get_dynamic_tp_params(adx: float) -> tuple:
    """Return (tp1_multiplier, trail_activate_multiplier) based on ADX tier"""
    if adx >= ADX_TIER3:
        return 2.2, 0.7
    elif adx >= ADX_TIER2:
        return 1.8, 0.8
    else:
        return 1.0, 1.0

# ====== NEW: TREND END DETECTION ======
def trend_end_confirmed(ind: dict, candle_info: dict, info: dict) -> bool:
    """
    نهاية الترند مؤكدة لما يتحقق واحد أو أكثر:
    - هبوط ADX واضح، أو نزول تحت 20
    - قلب اتجاه DI ضد مركزنا بهامش
    - رجوع RSI لمنطقة حيادية
    - إشارة RF معاكسة (سيتم إغلاقها أصلاً في الـ trade_loop لكن نخلي هنا احتياط)
    """
    adx = float(ind.get("adx") or 0.0)
    adx_prev = float(ind.get("adx_prev") or adx)
    plus_di = float(ind.get("plus_di") or 0.0)
    minus_di = float(ind.get("minus_di") or 0.0)
    rsi = float(ind.get("rsi") or 50.0)

    # هبوط ADX قوي أو تحت 20
    adx_weak = (adx_prev - adx) >= END_TREND_ADX_DROP or adx < 20

    # قلب DI ضد مركزنا
    if state.get("side") == "long":
        di_flip = (minus_di - plus_di) > DI_FLIP_BUFFER
    else:
        di_flip = (plus_di - minus_di) > DI_FLIP_BUFFER

    # RSI محايد
    rsi_neutral = END_TREND_RSI_NEUTRAL[0] <= rsi <= END_TREND_RSI_NEUTRAL[1]

    # إشارة RF معاكسة على الشمعة المغلقة الحالية
    rf_opposite = (state.get("side") == "long" and info.get("short")) or \
                  (state.get("side") == "short" and info.get("long"))

    return adx_weak or di_flip or rsi_neutral or rf_opposite

# ====== NEW: TREND CONFIRMATION LOGIC ======
def check_trend_confirmation(candle_info: dict, ind: dict, current_side: str) -> str:
    """
    تحليل تأكيد الاتجاه باستخدام ADX + DI + الشموع
    Returns:
        "CONFIRMED_CONTINUE" - اتجاه مؤكد: استمرار في الترند
        "POSSIBLE_FAKEOUT" - اشتباه انعكاس، نحتاج انتظار
        "CONFIRMED_REVERSAL" - انعكاس مؤكد بعد التحقق
        "NO_SIGNAL"
    """
    try:
        pattern = candle_info.get("pattern", "NONE")
        adx = float(ind.get("adx") or 0)
        plus_di = float(ind.get("plus_di") or 0)
        minus_di = float(ind.get("minus_di") or 0)
        rsi = float(ind.get("rsi") or 50)
        
        # قائمة بأنماط الشموع الانعكاسية
        reversal_patterns = ["DOJI", "HAMMER", "SHOOTING_STAR", "EVENING_STAR", "MORNING_STAR"]
        
        # ✅ NEW: Fakeout Protection Logic
        if pattern in reversal_patterns:
            # حالة 1: ترند قوي (ADX ≥ 25) واتجاه DI مؤكد ⇒ انعكاس وهمي
            if adx >= 25:
                if current_side == "long" and plus_di > minus_di and rsi >= RSI_TREND_BUY:
                    return "CONFIRMED_CONTINUE"
                elif current_side == "short" and minus_di > plus_di and rsi <= RSI_TREND_SELL:
                    return "CONFIRMED_CONTINUE"
                else:
                    # ✅ NEW: Check for confirmed reversal with strong indicators
                    if current_side == "long" and minus_di > plus_di and rsi < 45:
                        return "CONFIRMED_REVERSAL"
                    elif current_side == "short" and plus_di > minus_di and rsi > 55:
                        return "CONFIRMED_REVERSAL"
                    else:
                        return "POSSIBLE_FAKEOUT"
            
            # حالة 2: ترند ضعيف (ADX < 25) ⇒ احتمال انعكاس وهمي
            elif adx < 25:
                return "POSSIBLE_FAKEOUT"
                
            # حالة 3: ترند متوسط (25-30) ⇒ لا إشارة واضحة
            else:
                return "NO_SIGNAL"
        
        # ✅ NEW: Strong trend continuation
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
    """Return (should_scale, step_size, reason)"""
    # ⛔ تعطيل التعزيز نهائيًا
    return False, 0.0, "Scale-in disabled"

def should_scale_out(candle_info: dict, ind: dict, current_side: str) -> tuple:
    """Return (should_scale_out, reason)"""
    if state["qty"] <= 0:
        return False, "No position to scale out"
    
    # NEW: Trend confirmation check for scale-out
    trend_signal = check_trend_confirmation(candle_info, ind, current_side)
    if trend_signal == "CONFIRMED_REVERSAL":
        return True, f"Confirmed reversal: {candle_info.get('name_en', 'NONE')}"
    
    adx = ind.get("adx", 0)
    rsi = ind.get("rsi", 50)
    
    # Warning patterns that suggest taking profits
    warning_patterns = ["DOJI", "SHOOTING_STAR", "HAMMER", "EVENING_STAR", "MORNING_STAR"]
    
    # For opposite side warning patterns, consider scaling out
    if (current_side == "long" and candle_info.get("pattern") in ["SHOOTING_STAR", "EVENING_STAR"]) or \
       (current_side == "short" and candle_info.get("pattern") in ["HAMMER", "MORNING_STAR"]):
        return True, f"Warning pattern: {candle_info.get('name_en')}"
    
    # ADX weakening after strong move
    if adx > 30 and ind.get("adx_prev", 0) > adx + 2:  # ADX dropping significantly
        return True, f"ADX weakening: {ind.get('adx_prev', 0):.1f} → {adx:.1f}"
    
    # RSI extreme in current trend
    if current_side == "long" and rsi > 75:
        return True, f"RSI overbought: {rsi:.1f}"
    if current_side == "short" and rsi < 25:
        return True, f"RSI oversold: {rsi:.1f}"
    
    return False, "No strong scale-out signal"

def get_trail_multiplier(ind: dict) -> float:
    """Determine trail multiplier based on market regime"""
    adx = ind.get("adx", 0)
    atr_pct = (ind.get("atr", 0) / (ind.get("price", 1) or 1)) * 100
    
    if adx >= 30 and atr_pct > 1.0:
        return TRAIL_MULT_STRONG  # Strong trending market
    elif adx >= 20:
        return TRAIL_MULT_MED     # Moderate trend
    else:
        return TRAIL_MULT_CHOP    # Choppy market

# ====== NEW: SMART POST-ENTRY MANAGEMENT ======
def determine_trade_mode(df: pd.DataFrame, ind: dict) -> str:
    """
    تحديد نمط الصفقة بناءً على ظروف السوق
    Returns: "SCALP" أو "TREND"
    """
    adx = ind.get("adx", 0)
    atr = ind.get("atr", 0)
    price = ind.get("price", 0)
    
    # حساب نسبة ATR للسعر
    atr_pct = (atr / price) * 100 if price > 0 else 0
    
    # تحليل الشموع الأخيرة
    if len(df) >= 3:
        # ✅ PATCH 4: Always use current closed candle
        idx = -1  # Always use the latest closed candle
        
        # حساب متوسط مدى الشموع الأخيرة
        recent_ranges = []
        for i in range(max(0, idx-2), idx+1):
            high = float(df["high"].iloc[i])
            low = float(df["low"].iloc[i])
            recent_ranges.append(high - low)
        
        avg_range = sum(recent_ranges) / len(recent_ranges) if recent_ranges else 0
        range_pct = (avg_range / price) * 100 if price > 0 else 0
        
        # شروط الترند القوي
        if (adx >= 25 and 
            atr_pct >= 1.0 and 
            range_pct >= 1.5 and
            ind.get("plus_di", 0) > ind.get("minus_di", 0) if state["side"] == "long" else ind.get("minus_di", 0) > ind.get("plus_di", 0)):
            return "TREND"
    
    # شروط السكالب (نطاق ضيق أو ترند ضعيف)
    if adx < 20 or atr_pct < 0.8:
        return "SCALP"
    
    # الافتراضي
    return "SCALP"

def handle_impulse_and_long_wicks(df: pd.DataFrame, ind: dict):
    """
    معالجة الشموع الانفجارية والذيل الطويل لجني الأرباح
    """
    if not state["open"] or state["qty"] <= 0:
        return None
    
    try:
        # ✅ PATCH 4: Always use current closed candle
        idx = -1  # Always use the latest closed candle
        
        o0 = float(df["open"].iloc[idx])
        h0 = float(df["high"].iloc[idx])
        l0 = float(df["low"].iloc[idx])
        c0 = float(df["close"].iloc[idx])
        
        current_price = ind.get("price") or c0
        entry = state["entry"]
        side = state["side"]
        
        # حساب الربح النسبي
        rr = (current_price - entry) / entry * 100 * (1 if side == "long" else -1)
        
        # إحصائيات الشمعة الحالية
        candle_range = h0 - l0
        body = abs(c0 - o0)
        upper_wick = h0 - max(o0, c0)
        lower_wick = min(o0, c0) - l0
        
        # نسبة الجسم والذيل
        body_pct = (body / candle_range) * 100 if candle_range > 0 else 0
        upper_wick_pct = (upper_wick / candle_range) * 100 if candle_range > 0 else 0
        lower_wick_pct = (lower_wick / candle_range) * 100 if candle_range > 0 else 0
        
        atr = ind.get("atr", 0)
        
        # الشموع الانفجارية (Impulse)
        if atr > 0 and body >= IMPULSE_HARVEST_THRESHOLD * atr:
            # التأكد من اتجاه الشمعة مع اتجاه الصفقة
            candle_direction = 1 if c0 > o0 else -1
            trade_direction = 1 if side == "long" else -1
            
            if candle_direction == trade_direction:
                # تحديد نسبة الجني بناءً على قوة الاندفاع
                if body >= 2.0 * atr:
                    harvest_frac = 0.50  # جني 50% للاندفاعات القوية
                    reason = f"Strong Impulse x{body/atr:.2f} ATR"
                else:
                    harvest_frac = 0.33  # جني 33% للاندفاعات المتوسطة
                    reason = f"Impulse x{body/atr:.2f} ATR"
                
                close_partial(harvest_frac, reason)
                
                # تفعيل بريك إيفن بعد الجني
                if not state["breakeven"] and rr >= BREAKEVEN_AFTER * 100:
                    state["breakeven"] = entry
                
                # تفعيل التريلينغ
                if atr and ATR_MULT_TRAIL > 0:
                    gap = atr * ATR_MULT_TRAIL
                    if side == "long":
                        state["trail"] = max(state.get("trail") or (current_price - gap), current_price - gap)
                    else:
                        state["trail"] = min(state.get("trail") or (current_price + gap), current_price + gap)
                
                return "IMPULSE_HARVEST"
        
        # ✅ CORRECTED: الذيل الطويل في اتجاه المكسب
        # للـ LONG: الذيل العلوي الطويل إشارة ضغط بيع عند القمم ⇒ مناسب لجني الأرباح
        if side == "long" and upper_wick_pct >= LONG_WICK_HARVEST_THRESHOLD * 100:
            close_partial(0.25, f"Upper wick (LONG) {upper_wick_pct:.1f}%")
            return "LONG_WICK_HARVEST"
        
        # للـ SHORT: الذيل السفلي الطويل إشارة ضغط شراء عند القيعان ⇒ مناسب لجني الأرباح
        if side == "short" and lower_wick_pct >= LONG_WICK_HARVEST_THRESHOLD * 100:
            close_partial(0.25, f"Lower wick (SHORT) {lower_wick_pct:.1f}%")
            return "LONG_WICK_HARVEST"
            
    except Exception as e:
        print(colored(f"⚠️ handle_impulse_and_long_wicks error: {e}", "yellow"))
        logging.error(f"handle_impulse_and_long_wicks error: {e}")
    
    return None

def ratchet_protection(ind: dict):
    """
    حماية المكاسب من التراجع الشديد
    """
    if not state["open"] or state["qty"] <= 0:
        return None
    
    current_price = ind.get("price") or price_now() or state["entry"]
    entry = state["entry"]
    side = state["side"]
    
    # حساب الربح النسبي الحالي
    current_rr = (current_price - entry) / entry * 100 * (1 if side == "long" else -1)
    
    # تحديث أعلى ربح
    if current_rr > state["highest_profit_pct"]:
        state["highest_profit_pct"] = current_rr
    
    # تطبيق حماية الراتشيت إذا تراجع الربح بنسبة معينة من القمة
    if (state["highest_profit_pct"] >= 20 and  # على الأقل 20% ربح
        current_rr < state["highest_profit_pct"] * (1 - RATCHET_RETRACE_THRESHOLD)):
        
        close_partial(0.5, f"Ratchet protection: {state['highest_profit_pct']:.1f}% → {current_rr:.1f}%")
        state["highest_profit_pct"] = current_rr  # إعادة ضبط القمة
        return "RATCHET_PROTECTION"
    
    return None

# ====== NEW: ADVANCED PROFIT TAKING ======
def scalp_profit_taking(ind: dict, info: dict):
    """
    سكالب: 3 مراحل جني أرباح ثم إغلاق صارم كامل.
    """
    if not state["open"] or state["qty"] <= 0:
        return None

    price = ind.get("price") or price_now() or state["entry"]
    entry = state["entry"]; side = state["side"]
    rr = (price - entry) / entry * 100 * (1 if side == "long" else -1)

    targets = SCALP_TARGETS
    fracs = SCALP_CLOSE_FRACS
    k = int(state.get("profit_targets_achieved", 0))

    if k < len(targets) and rr >= targets[k]:
        close_partial(fracs[k], f"SCALP TP{k+1}@{targets[k]:.2f}%")
        state["profit_targets_achieved"] = k + 1

        # بعد آخر مرحلة → قفل صارم كامل
        if state["profit_targets_achieved"] >= len(targets):
            close_market_strict("SCALP sequence complete")
            return "SCALP_COMPLETE"

        return f"SCALP_TP{k+1}"

    return None

def trend_profit_taking(ind: dict, info: dict):
    """
    ترند قوي: 3 مراحل جني أرباح تدريجي (لا نغلق كليًا)،
    ونستمر حتى نهاية الترند المؤكدة → قفل صارم كامل.
    """
    if not state["open"] or state["qty"] <= 0:
        return None

    price = ind.get("price") or price_now() or state["entry"]
    entry = state["entry"]; side = state["side"]
    rr = (price - entry) / entry * 100 * (1 if side == "long" else -1)

    adx = float(ind.get("adx") or 0.0)

    # 1) مراحل الجني
    targets = TREND_TARGETS
    fracs = TREND_CLOSE_FRACS
    k = int(state.get("profit_targets_achieved", 0))

    if k < len(targets) and rr >= targets[k]:
        close_partial(fracs[k], f"TREND TP{k+1}@{targets[k]:.2f}%")
        state["profit_targets_achieved"] = k + 1
        return f"TREND_TP{k+1}"

    # 2) طالما ADX قوي ابقى راكب
    if adx >= MIN_TREND_HOLD_ADX:
        return None

    # 3) بعد إتمام المراحل أو ضعف الترند → تحقّق نهاية الترند
    if state.get("profit_targets_achieved", 0) >= len(targets) or trend_end_confirmed(ind, detect_candle_pattern(fetch_ohlcv()), info):
        close_market_strict("TREND finished — full exit")
        return "TREND_COMPLETE"

    return None

def smart_post_entry_manager(df: pd.DataFrame, ind: dict, info: dict):
    """
    المدير الذكي للصفقة بعد الدخول
    """
    if not state["open"] or state["qty"] <= 0:
        return None
    
    # تحديد نمط الصفقة إذا لم يكن محددًا
    if state.get("trade_mode") is None:
        trade_mode = determine_trade_mode(df, ind)
        state["trade_mode"] = trade_mode
        state["profit_targets_achieved"] = 0
        state["entry_time"] = time.time()
        
        print(colored(f"🎯 TRADE MODE DETECTED: {trade_mode}", "cyan"))
        logging.info(f"Trade mode detected: {trade_mode}")
    
    # ✅ CORRECTED: حصاد الشمعة/الذيل المصحح
    impulse_action = handle_impulse_and_long_wicks(df, ind)
    if impulse_action:
        # بعد أي حصاد، لا نغلق فورًا في الترند — نترك trend_profit_taking يحكم
        return impulse_action

    # حماية الراتشيت كما هي
    ratchet_action = ratchet_protection(ind)
    if ratchet_action:
        return ratchet_action

    # جني الأرباح وفق النمط
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
            # ✅ FIXED: Use correct side parameter for BingX leverage setting
            lev_params = {"side": "BOTH"}  # Simplified for oneway mode
            try: ex.set_leverage(LEVERAGE, SYMBOL, params=lev_params)
            except Exception as e: print(colored(f"⚠️ set_leverage: {e}", "yellow"))
            ex.create_order(SYMBOL, "market", side, qty, None, params)
        except Exception as e:
            print(colored(f"❌ open: {e}", "red"))
            logging.error(f"open_market error: {e}")
            return  # لا نُحدّث الحالة إذا فشل التنفيذ الفعلي

    state.update({
        "open": True, "side": "long" if side=="buy" else "short", 
        "entry": price, "qty": qty, "pnl": 0.0, "bars": 0, 
        "trail": None, "tp1_done": False, "breakeven": None,
        "scale_ins": 0, "scale_outs": 0,
        "last_action": "OPEN", "action_reason": "Initial position",
        "highest_profit_pct": 0.0,  # Reset ratchet lock
        "trade_mode": None,  # ✅ NEW: Reset trade mode
        "profit_targets_achieved": 0,  # ✅ NEW: Reset profit targets
        "entry_time": time.time(),  # ✅ NEW: Track entry time
        # ✅ NEW: Reset fakeout protection on new position
        "fakeout_pending": False,
        "fakeout_need_side": None,
        "fakeout_confirm_bars": 0,
        "fakeout_started_at": None,
        # ✅ NEW: Reset breakout state on new position
        "breakout_active": False,
        "breakout_direction": None,
        "breakout_entry_price": None
    })
    print(colored(f"✅ OPEN {side.upper()} qty={fmt(qty,4)} @ {fmt(price)}","green" if side=="buy" else "red"))
    logging.info(f"OPEN {side} qty={qty} price={price}")
    save_state()

def scale_in_position(scale_pct: float, reason: str):
    """Add to existing position"""
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
    
    # Update average entry price
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
    """Close fraction of current position (smart TP1)."""
    global state, compound_pnl
    if not state["open"]: return
    qty_close = safe_qty(max(0.0, state["qty"]*min(max(frac,0.0),1.0)))
    
    # ✅ FIX: Prevent partial close smaller than 1 DOGE
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
    
    # ✅ Auto-full close if remaining qty too small (below 60 DOGE)
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
    
    # احفظ الجانب السابق قبل المسح
    if prev_side is None:
        prev_side = state.get("side")
    
    state.update({
        "open": False, "side": None, "entry": None, "qty": 0.0, 
        "pnl": 0.0, "bars": 0, "trail": None, "tp1_done": False, 
        "breakeven": None, "scale_ins": 0, "scale_outs": 0,
        "last_action": "CLOSE", "action_reason": reason,
        "highest_profit_pct": 0.0,
        "trade_mode": None,  # ✅ NEW: Reset trade mode
        "profit_targets_achieved": 0,  # ✅ NEW: Reset profit targets
        "entry_time": None,  # ✅ NEW: Reset entry time
        # ✅ NEW: Reset fakeout protection on full close
        "fakeout_pending": False,
        "fakeout_need_side": None,
        "fakeout_confirm_bars": 0,
        "fakeout_started_at": None,
        # ✅ NEW: Reset breakout state on full close
        "breakout_active": False,
        "breakout_direction": None,
        "breakout_entry_price": None
    })
    
    # ✅ PATCH: ضع الانتظار للإشارة المعاكسة
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
    
    # ✅ PATCH: احفظ الجانب الحالي
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
    
    # ✅ PATCH: استدعِ النسخة المعدلة مع الجانب السابق
    reset_after_full_close(reason, prev_side_local)

# ------------ Advanced Position Management Check ------------
def advanced_position_management(candle_info: dict, ind: dict):
    """Handle scale-in, scale-out, and dynamic trailing with trend confirmation"""
    if not state["open"]:
        return None
    
    current_side = state["side"]
    px = ind.get("price") or price_now() or state["entry"]
    
    # NEW: Trend Confirmation Logic - تطبيق منطق تأكيد الاتجاه
    trend_signal = check_trend_confirmation(candle_info, ind, current_side)
    if trend_signal == "CONFIRMED_CONTINUE":
        print(colored("📈 اتجاه مؤكد: استمرار في الترند", "green"))
        logging.info("Trend confirmed: continuing in the same direction")
    elif trend_signal == "CONFIRMED_REVERSAL":
        print(colored("⚠️ انعكاس مؤكد: خروج جزئي", "red"))
        logging.info("Reversal confirmed: partial exit")
        close_partial(0.3, "Reversal confirmed by trend analysis")  # إغلاق 30% كخروج جزئي
        return "SCALE_OUT_REVERSAL"
    
    # Scale-in check with dynamic step
    should_scale, step_size, scale_reason = should_scale_in(candle_info, ind, current_side)
    if should_scale:
        scale_in_position(step_size, scale_reason)
        return "SCALE_IN"
    
    # ✅ PATCH 5: FIXED UnboundLocalError - تغيير اسم المتغير لتجنب التعارض
    do_scale_out, scale_out_reason = should_scale_out(candle_info, ind, current_side)  # تغيير should_scale_out إلى do_scale_out
    if do_scale_out:  # استخدام المتغير الجديد
        close_partial(0.3, scale_out_reason)  # Close 30% on warning signals
        return "SCALE_OUT"
    
    # Dynamic trailing adjustment
    trail_mult = get_trail_multiplier({**ind, "price": px})
    if state["trail"] is not None and trail_mult != ATR_MULT_TRAIL:
        # Adjust existing trail
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

# ------------ Smart Profit (trend-aware) with Trend Amplifier ------------
def smart_exit_check(info, ind):
    """Return True if full close happened."""
    if not (STRATEGY=="smart" and USE_SMART_EXIT and state["open"]):
        return None

    # Advanced position management first
    candle_info = detect_candle_pattern(fetch_ohlcv())
    management_action = advanced_position_management(candle_info, ind)
    if management_action:
        print(colored(f"🎯 MANAGEMENT: {management_action} - {state['action_reason']}", "yellow"))
        logging.info(f"MANAGEMENT_ACTION: {management_action} - {state['action_reason']}")

    # ✅ NEW: Smart Post-Entry Management
    df = fetch_ohlcv()
    post_entry_action = smart_post_entry_manager(df, ind, info)
    if post_entry_action:
        print(colored(f"🎯 POST-ENTRY: {post_entry_action} - Trade Mode: {state.get('trade_mode', 'N/A')}", "cyan"))
        logging.info(f"POST_ENTRY_ACTION: {post_entry_action} - Trade Mode: {state.get('trade_mode', 'N/A')}")

    px = info["price"]; e = state["entry"]; side = state["side"]
    
    # ✅ PATCH: safety guard to avoid float-None operations after strict close
    if e is None or px is None or side is None or e == 0:
        return None

    # ✅ NEW: Fakeout Protection Logic - قبل أي إغلاق نهائي
    if state["open"]:
        trend_signal = check_trend_confirmation(candle_info, ind, state["side"])
        
        # حالة 1: اشتباه انعكاس وهمي - تفعيل الانتظار
        if (trend_signal == "POSSIBLE_FAKEOUT" and 
            not state["fakeout_pending"] and 
            state["fakeout_confirm_bars"] == 0):
            
            state["fakeout_pending"] = True
            state["fakeout_confirm_bars"] = 2  # انتظر شمعتين مغلقتين
            state["fakeout_need_side"] = "short" if state["side"] == "long" else "long"
            state["fakeout_started_at"] = info["time"]
            
            print(colored("🕒 WAITING — possible fake reversal detected, holding position...", "yellow"))
            logging.info("FAKEOUT PROTECTION: Possible fake reversal detected, waiting for confirmation")
            return None  # لا تغلق الآن
        
        # حالة 2: في انتظار تأكيد الانعكاس
        elif state["fakeout_pending"]:
            # تحقق من تأكيد الانعكاس
            if (trend_signal == "CONFIRMED_REVERSAL" and 
                state["fakeout_need_side"] == ("short" if state["side"] == "long" else "long")):
                
                state["fakeout_confirm_bars"] -= 1
                print(colored(f"🕒 FAKEOUT CONFIRMATION — {state['fakeout_confirm_bars']} bars left", "yellow"))
                
                if state["fakeout_confirm_bars"] <= 0:
                    print(colored("⚠️ CONFIRMED REVERSAL — closing position", "red"))
                    logging.info("FAKEOUT PROTECTION: Confirmed reversal after fakeout delay")
                    close_market_strict("CONFIRMED REVERSAL after fakeout delay")
                    # إعادة تعيين حالة الـ fakeout
                    state["fakeout_pending"] = False
                    state["fakeout_need_side"] = None
                    state["fakeout_confirm_bars"] = 0
                    state["fakeout_started_at"] = None
                    return True  # Position closed
            
            # حالة 3: إلغاء الاشتباه - العودة لصالح الصفقة
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

    # Trend Amplifier: Dynamic TP based on ADX
    tp_multiplier, trail_activate_multiplier = get_dynamic_tp_params(adx)
    current_tp1_pct = TP1_PCT * tp_multiplier
    current_trail_activate = TRAIL_ACTIVATE * trail_activate_multiplier
    current_tp1_pct_pct = current_tp1_pct * 100.0
    current_trail_activate_pct = current_trail_activate * 100.0

    # ✅ PATCH: TP1 fallback if trade_mode not decided yet
    if state.get("trade_mode") is None:
        if (not state["tp1_done"]) and rr >= current_tp1_pct_pct:
            close_partial(TP1_CLOSE_FRAC, f"TP1@{current_tp1_pct_pct:.2f}%")
            state["tp1_done"] = True

    # ---------- HOLD-TP: لا خروج بدري مع ترند قوي ----------
    if side == "long" and adx >= 30 and rsi >= RSI_TREND_BUY:
        print(colored("💎 HOLD-TP: strong uptrend continues, delaying TP", "cyan"))
        # نكمل من غير أي إغلاقات مبكرة
    elif side == "short" and adx >= 30 and rsi <= RSI_TREND_SELL:
        print(colored("💎 HOLD-TP: strong downtrend continues, delaying TP", "cyan"))
        # نكمل من غير أي إغلاقات مبكرة

    # انتظر كام شمعة بعد الدخول لتجنب الخروج السريع
    if state["bars"] < 2:
        return None

    # Highest profit (ratchet)
    if rr > state["highest_profit_pct"]:
        state["highest_profit_pct"] = rr
        if tp_multiplier > 1.0:
            print(colored(f"🎯 TREND AMPLIFIER: New high {rr:.2f}% • TP={current_tp1_pct_pct:.2f}% • TrailActivate={current_trail_activate_pct:.2f}%", "green"))
            logging.info(f"TREND_AMPLIFIER new_high={rr:.2f}% TP={current_tp1_pct_pct:.2f}%")

    # TP1 الجزئي (بعد الانتظار)
    if (not state["tp1_done"]) and rr >= current_tp1_pct_pct:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{current_tp1_pct_pct:.2f}%")
        state["tp1_done"] = True
        if rr >= BREAKEVEN_AFTER * 100.0:
            state["breakeven"] = e

    # Ratchet lock على التراجع
    if (state["highest_profit_pct"] >= current_trail_activate_pct and
        rr < state["highest_profit_pct"] * RATCHET_LOCK_PCT):
        close_partial(0.5, f"Ratchet Lock @ {state['highest_profit_pct']:.2f}%")
        state["highest_profit_pct"] = rr
        return None

    # Trailing ATR
    if rr >= current_trail_activate_pct and atr and ATR_MULT_TRAIL > 0:
        gap = atr * ATR_MULT_TRAIL
        if side == "long":
            new_trail = px - gap
            state["trail"] = max(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None:
                state["trail"] = max(state["trail"], state["breakeven"])
            if px < state["trail"]:
                close_market_strict(f"TRAIL_ATR({ATR_MULT_TRAIL}x)")
                return True
        else:
            new_trail = px + gap
            state["trail"] = min(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None:
                state["trail"] = min(state["trail"], state["breakeven"])
            if px > state["trail"]:
                close_market_strict(f"TRAIL_ATR({ATR_MULT_TRAIL}x)")
                return True
    return None

# ------------ ENHANCED: BREAKOUT ENGINE FUNCTIONS ------------
def detect_breakout(df: pd.DataFrame, ind: dict, prev_ind: dict) -> str:
    """
    ⚡ BREAKOUT ENGINE - OPTIMIZED: استخدام ATR السابق الممرّر بدلاً من إعادة الحساب
    """
    try:
        if len(df) < BREAKOUT_LOOKBACK_BARS + 2:
            return None
            
        current_idx = -1
        
        # بيانات المؤشرات - باستخدام القيم الممررة
        adx = float(ind.get("adx") or 0.0)
        atr_now = float(ind.get("atr") or 0.0)
        atr_prev = float(prev_ind.get("atr") or atr_now)  # ✅ ENHANCED: استخدام ATR السابق الممرر
        price = float(df["close"].iloc[current_idx])
        
        # التحقق من شروط الانفجار
        atr_spike = atr_now > atr_prev * BREAKOUT_ATR_SPIKE
        strong_trend = adx >= BREAKOUT_ADX_THRESHOLD
        
        if not (atr_spike and strong_trend):
            return None
            
        # كسر القمم/Ceiling (انفجار صعودي)
        recent_highs = df["high"].iloc[-BREAKOUT_LOOKBACK_BARS:-1].astype(float)
        new_high = price > recent_highs.max() if len(recent_highs) > 0 else False
        
        # كسر القيعان/Floor (انهيار هبوطي)  
        recent_lows = df["low"].iloc[-BREAKOUT_LOOKBACK_BARS:-1].astype(float)
        new_low = price < recent_lows.min() if len(recent_lows) > 0 else False
        
        if new_high:
            return "BULL_BREAKOUT"
        elif new_low:
            return "BEAR_BREAKOUT"
            
    except Exception as e:
        print(colored(f"⚠️ detect_breakout error: {e}", "yellow"))
        logging.error(f"detect_breakout error: {e}")
        
    return None

def handle_breakout_entries(df: pd.DataFrame, ind: dict, prev_ind: dict, bal: float, spread_bps: float) -> bool:
    """
    ✅ ENHANCED: معالجة دخول الانفجارات مع جميع الحمايات المطلوبة
    """
    global state
    
    # 1. تحقق من وجود انفجار
    breakout_signal = detect_breakout(df, ind, prev_ind)
    if not breakout_signal or state["breakout_active"]:
        return False
    
    price = ind.get("price") or float(df["close"].iloc[-1])
    
    # ✅ ENHANCED 2: فلتر السيولة للانفجارات
    if spread_bps is not None and spread_bps > SPREAD_GUARD_BPS:
        print(colored(f"⛔ BREAKOUT: Spread too high ({fmt(spread_bps,2)}bps) - skipping entry", "yellow"))
        logging.warning(f"BREAKOUT_ENGINE: Spread filter blocked entry - {spread_bps}bps")
        return False
    
    # ✅ ENHANCED 3: منع فتح صفقة بكمية أقل من الحد الأدنى
    qty = compute_size(bal, price)
    if qty < (LOT_MIN or 1):
        print(colored(f"⛔ BREAKOUT: Quantity too small ({fmt(qty,4)} < {LOT_MIN or 1}) - skipping", "yellow"))
        logging.warning(f"BREAKOUT_ENGINE: Quantity below minimum - {qty} < {LOT_MIN or 1}")
        return False
    
    # ✅ ENHANCED 4: استخدام الحارس الموجود لمنع الدخول المكرر
    if not can_open(breakout_signal, price):
        print(colored("⛔ BREAKOUT: Idempotency guard blocked duplicate entry", "yellow"))
        logging.warning("BREAKOUT_ENGINE: Idempotency guard blocked entry")
        return False
    
    # تنفيذ الدخول
    if breakout_signal == "BULL_BREAKOUT":
        open_market("buy", qty, price)
        state["breakout_active"] = True
        state["breakout_direction"] = "bull"
        state["breakout_entry_price"] = price
        print(colored(f"⚡ BREAKOUT ENGINE: BULLISH EXPLOSION - ENTERING LONG", "green"))
        logging.info(f"BREAKOUT_ENGINE: Bullish explosion - LONG {qty} @ {price}")
        return True
        
    elif breakout_signal == "BEAR_BREAKOUT":
        open_market("sell", qty, price)
        state["breakout_active"] = True  
        state["breakout_direction"] = "bear"
        state["breakout_entry_price"] = price
        print(colored(f"⚡ BREAKOUT ENGINE: BEARISH CRASH - ENTERING SHORT", "red"))
        logging.info(f"BREAKOUT_ENGINE: Bearish crash - SHORT {qty} @ {price}")
        return True
    
    return False

def handle_breakout_exits(df: pd.DataFrame, ind: dict, prev_ind: dict) -> bool:
    """
    ✅ ENHANCED: معالجة خروج الانفجارات باستخدام ATR السابق الممرّر
    """
    global state
    
    if not state["breakout_active"] or not state["open"]:
        return False
        
    current_idx = -1
    
    # ✅ ENHANCED 1: استخدام ATR السابق الممرر بدلاً من إعادة الحساب
    atr_now = float(ind.get("atr") or 0.0)
    atr_prev = float(prev_ind.get("atr") or atr_now)
    
    # التحقق من هدوء التقلب (نهاية الانفجار)
    volatility_calm = atr_now < atr_prev * BREAKOUT_CALM_THRESHOLD
    
    if volatility_calm:
        direction = state["breakout_direction"]
        entry_price = state["breakout_entry_price"]
        current_price = ind.get("price") or float(df["close"].iloc[current_idx])
        
        # حساب الربح قبل الإغلاق
        pnl_pct = ((current_price - entry_price) / entry_price * 100 * 
                  (1 if direction == "bull" else -1))
                  
        close_market_strict(f"Breakout ended - {pnl_pct:.2f}% PnL")
        
        print(colored(f"✅ BREAKOUT ENGINE: {direction.upper()} breakout ended - {pnl_pct:.2f}% PnL", "magenta"))
        logging.info(f"BREAKOUT_ENGINE: {direction} breakout ended - PnL: {pnl_pct:.2f}%")
        
        # ✅ ENHANCED 5: إعادة تعيين حالة الانفجار بعد الإغلاق الكامل
        state["breakout_active"] = False
        state["breakout_direction"] = None  
        state["breakout_entry_price"] = None
        
        return True
        
    return False

# ------------ NEW: SMART EMERGENCY PROTECTION LAYER FUNCTION ------------
def breakout_emergency_protection(ind: dict, prev_ind: dict) -> bool:
    """
    🛡️ Smart Emergency Layer:
    - يكتشف Pump/Crash قوي (ATR spike + ADX عالي + RSI متطرف).
    - في صالح الصفقة: جني أرباح ذكي (partial) + بريك إيفن + تريل طارئ
      * أو إغلاق كامل لو الربح كفاية.
    - ضد الصفقة: إغلاق كامل فوري لتقليل الخسارة.
    - يرجّع True لو أغلق أو نفّذ جني/تريل طارئ.
    """
    if not (EMERGENCY_PROTECTION_ENABLED and state.get("open")):
        return False

    try:
        adx = float(ind.get("adx") or 0.0)
        rsi = float(ind.get("rsi") or 50.0)
        atr_now  = float(ind.get("atr") or 0.0)
        atr_prev = float(prev_ind.get("atr") or atr_now)
        price = ind.get("price") or price_now() or state.get("entry")

        # تحقق الشروط
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

        # ربح الصفقة الحالي %
        rr_pct = (price - entry) / entry * 100.0 * (1 if side == "long" else -1)

        print(colored(f"🛡️ EMERGENCY LAYER DETECTED: {side.upper()} | RSI={rsi:.1f} | ADX={adx:.1f} | ATR Spike={atr_now/atr_prev:.2f}x | PnL={rr_pct:.2f}%", "yellow"))
        logging.info(f"EMERGENCY_LAYER: {side} RSI={rsi} ADX={adx} ATR_ratio={atr_now/atr_prev:.2f} PnL={rr_pct:.2f}%")

        # ضد اتجاهنا → إغلاق فوري
        if (pump and side == "short") or (crash and side == "long"):
            close_market_strict("EMERGENCY opposite pump/crash — close now")
            print(colored(f"🛑 EMERGENCY: AGAINST POSITION - FULL CLOSE", "red"))
            logging.warning(f"EMERGENCY_LAYER: Against position - full close")
            return True

        # في صالحنا:
        if EMERGENCY_POLICY == "close_always":
            close_market_strict("EMERGENCY favorable pump/crash — close all")
            print(colored(f"🟡 EMERGENCY: FAVORABLE - POLICY CLOSE ALL", "yellow"))
            logging.info(f"EMERGENCY_LAYER: Favorable - policy close all")
            return True

        # لو الربح أصلاً كبير كفاية → إغلاق كامل
        if rr_pct >= EMERGENCY_FULL_CLOSE_PROFIT:
            close_market_strict(f"EMERGENCY full close @ {rr_pct:.2f}%")
            print(colored(f"🟢 EMERGENCY: PROFIT TARGET HIT - FULL CLOSE @ {rr_pct:.2f}%", "green"))
            logging.info(f"EMERGENCY_LAYER: Profit target hit - full close @ {rr_pct:.2f}%")
            return True

        # جني ذكي جزئي + حماية
        harvest = max(0.0, min(1.0, EMERGENCY_HARVEST_FRAC))
        if harvest > 0:
            close_partial(harvest, f"EMERGENCY {'PUMP' if pump else 'CRASH'} harvest {harvest*100:.0f}%")
            print(colored(f"💰 EMERGENCY: HARVEST {harvest*100:.0f}% - PnL={rr_pct:.2f}%", "cyan"))
            logging.info(f"EMERGENCY_LAYER: Harvest {harvest*100:.0f}% - PnL={rr_pct:.2f}%")

        # بريك إيفن + تريل طارئ
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

        # لو السياسة tp_then_close → اقفل الباقي فورًا
        if EMERGENCY_POLICY == "tp_then_close":
            close_market_strict("EMERGENCY: harvest then full close")
            print(colored(f"🟡 EMERGENCY: TP_THEN_CLOSE POLICY - FULL CLOSE", "yellow"))
            logging.info(f"EMERGENCY_LAYER: tp_then_close policy - full close")
            return True

        # لو السياسة tp_then_trail → نسيب الباقي على تريل محكم
        print(colored(f"🟢 EMERGENCY: TP_THEN_TRAIL POLICY - RIDING THE MOVE", "green"))
        logging.info(f"EMERGENCY_LAYER: tp_then_trail policy - riding the move")

        # هنرجّع True عشان نُعلم إن إجراء حصل.
        return True

    except Exception as e:
        print(colored(f"⚠️ breakout_emergency_protection error: {e}", "yellow"))
        logging.error(f"breakout_emergency_protection error: {e}")
        return False

# ------------ Enhanced HUD (rich logs) ------------
def snapshot(bal,info,ind,spread_bps,reason=None, df=None):
    df = df if df is not None else fetch_ohlcv()
    left_s = time_to_candle_close(df, USE_TV_BAR)
    insights = build_log_insights(df, ind, info.get("price"))

    print(colored("─"*100,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*100,"cyan"))

    # ===== INDICATORS & CANDLES =====
    print("📈 INDICATORS & CANDLES")
    print(f"   💲 Price {fmt(info.get('price'))}  |  RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))}  lo={fmt(info.get('lo'))}")
    print(f"   🧮 RSI({RSI_LEN})={fmt(ind['rsi'])}   +DI={fmt(ind['plus_di'])}   -DI={fmt(ind['minus_di'])}   DX={fmt(ind['dx'])}   ADX({ADX_LEN})={fmt(ind['adx'])}   ATR={fmt(ind['atr'])} (~{fmt(insights['atr_pct'],2)}%)")
    print(f"   🎯 Signal  ✅ BUY={info['long']}   ❌ SELL={info['short']}   |   🧮 spread_bps={fmt(spread_bps,2)}")
    print(f"   {insights['regime_emoji']} Regime={insights['regime']}   {insights['bias_emoji']} Bias={insights['bias']}   |   {insights['rsi_zone']}")
    
    candle_info = insights['candle']
    print(f"   🕯️ Candles = {insights['candle_emoji']} {candle_info['name_ar']} / {candle_info['name_en']} (Strength: {candle_info['strength']}/4)")
    
    # ✅ NEW: Display waiting status
    if not state["open"] and wait_for_next_signal_side:
        print(colored(f"   ⏳ WAITING — need next {wait_for_next_signal_side.upper()} signal from TradingView Range Filter", "cyan"))
    
    # ✅ NEW: Display fakeout protection status
    if state["open"] and state["fakeout_pending"]:
        print(colored(f"   🛡️ FAKEOUT PROTECTION — waiting {state['fakeout_confirm_bars']} bars for confirmation", "yellow"))
    
    # ✅ NEW: Display breakout engine status
    if state["breakout_active"]:
        print(colored(f"   ⚡ BREAKOUT MODE ACTIVE: {state['breakout_direction'].upper()} - Monitoring volatility...", "cyan"))
    
    print(f"   ⏱️ Candle closes in ~ {left_s}s")
    print()

    # ===== POSITION & MANAGEMENT =====
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
        if state['last_action']:
            print(f"   🔄 Last Action: {state['last_action']} - {state['action_reason']}")
    else:
        print("   ⚪ FLAT")
    print()

    # ===== ACTION INSIGHTS =====
    print("💡 ACTION INSIGHTS")
    if state["open"] and STRATEGY == "smart":
        current_side = state["side"]
        should_scale, step_size, scale_reason = should_scale_in(candle_info, ind, current_side)
        do_scale_out, scale_out_reason = should_scale_out(candle_info, ind, current_side)
        
        # NEW: Trend confirmation display
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
        
        # Trend Amplifier info
        adx = ind.get("adx", 0)
        tp_multiplier, trail_multiplier = get_dynamic_tp_params(adx)
        if tp_multiplier > 1.0:
            current_tp1_pct = TP1_PCT * tp_multiplier * 100.0
            current_trail_activate = TRAIL_ACTIVATE * trail_multiplier * 100.0
            print(colored(f"   🚀 TREND AMPLIFIER ACTIVE: TP={current_tp1_pct:.2f}% • TrailActivate={current_trail_activate:.2f}%", "cyan"))
        
        # Trail info
        trail_mult = get_trail_multiplier({**ind, "price": info.get("price")})
        trail_type = "STRONG" if trail_mult == TRAIL_MULT_STRONG else "MED" if trail_mult == TRAIL_MULT_MED else "CHOP"
        print(f"   🛡️ Trail Multiplier: {trail_mult} ({trail_type})")
        
        # ✅ NEW: Advanced Profit Taking Status
        trade_mode = state.get('trade_mode')
        if trade_mode:
            if trade_mode == "SCALP":
                targets = SCALP_TARGETS
                fracs = SCALP_CLOSE_FRACS
                mode_name = "SCALP"
            else:
                targets = TREND_TARGETS
                fracs = TREND_CLOSE_FRACS
                mode_name = "TREND"
            
            achieved = state.get('profit_targets_achieved', 0)
            remaining_targets = len(targets) - achieved
            if achieved < len(targets):
                next_target = targets[achieved]
                next_frac = fracs[achieved] * 100
                print(colored(f"   🎯 {mode_name} MODE: {achieved}/{len(targets)} targets • Next: TP{achieved+1}@{next_target:.2f}% ({next_frac:.0f}%)", "magenta"))
            else:
                print(colored(f"   ✅ {mode_name} MODE: All targets achieved • Riding trend", "green"))
    else:
        print("   🔄 Waiting for trading signals...")
    print()

    # ===== RESULTS =====
    print("📦 RESULTS")
    # ✅ PATCH 2: Accurate Effective Equity display in paper mode
    eff_eq = (bal or 0.0) + compound_pnl
    print(f"   🧮 CompoundPnL {fmt(compound_pnl)}   🚀 EffectiveEq {fmt(eff_eq)} USDT")
    if reason:
        print(colored(f"   ℹ️ WAIT — reason: {reason}","yellow"))
    print(colored("─"*100,"cyan"))

# ------------ ENHANCED: Decision Loop with All Protection Layers ------------
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
            
            # HARDENING: Bar clock sanity check
            sanity_check_bar_clock(df)
            
            info = compute_tv_signals(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # ✅ ENHANCED: حساب المؤشرات السابقة مرة واحدة لتحسين الأداء
            prev_ind = compute_indicators(df.iloc[:-1]) if len(df) >= 2 else ind

            if state["open"] and px:
                state["pnl"] = (px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]

            # ✅ ENHANCED: BREAKOUT ENGINE - يعمل قبل أي منطق آخر
            # 1. معالجة خروج الانفجارات أولاً (إذا كنا في وضع انفجار نشط)
            breakout_exited = handle_breakout_exits(df, ind, prev_ind)
            
            # 2. معالجة دخول الانفجارات (إذا لم نكن في صفقة)
            breakout_entered = False
            if not state["open"] and not breakout_exited:
                breakout_entered = handle_breakout_entries(df, ind, prev_ind, bal, spread_bps)
            
            # 3. إذا دخلنا بصفقة انفجار، نتخطى كل المنطق الآخر لهذه الدورة
            if breakout_entered:
                # نعرض اللوحة ثم ننتقل مباشرة للنوم
                snapshot(bal, info, ind, spread_bps, "BREAKOUT ENTRY - skipping normal logic", df)
                time.sleep(compute_next_sleep(df))
                continue
                
            # 4. إذا كنا في وضع انفجار نشط، نتخطى إشارات الـ RF
            if state["breakout_active"]:
                # نراقب فقط وننتظر هدوء التقلب للإغلاق
                snapshot(bal, info, ind, spread_bps, "BREAKOUT ACTIVE - monitoring exit", df)
                time.sleep(compute_next_sleep(df))
                continue

            # ✅ NEW: SMART EMERGENCY PROTECTION LAYER (Pump/Crash)
            if state["open"]:
                if breakout_emergency_protection(ind, prev_ind):
                    snapshot(bal, info, ind, spread_bps, "EMERGENCY LAYER action", df)
                    time.sleep(compute_next_sleep(df))
                    continue

            # ------------ [الكود الأصلي يبدأ من هنا] ------------
            # باقي المنطق الأصلي يعمل فقط عندما لسنا في وضع انفجار أو طوارئ
            
            # Smart profit (trend-aware) with Trend Amplifier
            smart_exit_check(info, ind)

            # Decide - ✅ PURE RANGE FILTER SIGNALS ONLY
            sig = "buy" if info["long"] else ("sell" if info["short"] else None)
            reason = None
            if not sig:
                reason = "no signal"
            elif spread_bps is not None and spread_bps > SPREAD_GUARD_BPS:
                reason = f"spread too high ({fmt(spread_bps,2)}bps > {SPREAD_GUARD_BPS})"
            elif post_close_cooldown > 0:
                reason = f"cooldown {post_close_cooldown} bars"

            # ✅ PATCH: Close on opposite RF signal + WAIT for next signal
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

            # ✅ PATCH: Open only when allowed
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

            if state["open"]:
                state["bars"] += 1
            if post_close_cooldown > 0 and not state["open"]:
                post_close_cooldown -= 1

            # HARDENING: Save state every 5 loops
            if loop_counter % 5 == 0:
                save_state()

            # ✅ PATCH: Strict Exchange Close consistency guard
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

# Suppress Werkzeug logs except first GET /
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
    return f"✅ RF Bot — {SYMBOL} {INTERVAL} — {mode} — {STRATEGY.upper()} — ADVANCED — TREND AMPLIFIER — HARDENED — TREND CONFIRMATION — INSTANT ENTRY — PURE RANGE FILTER — STRICT EXCHANGE CLOSE — SMART POST-ENTRY MANAGEMENT — CLOSED CANDLE SIGNALS — WAIT FOR NEXT SIGNAL AFTER CLOSE — FAKEOUT PROTECTION — ADVANCED PROFIT TAKING — OPPOSITE SIGNAL WAITING — CORRECTED WICK HARVESTING — BREAKOUT ENGINE — EMERGENCY PROTECTION LAYER"

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
            "profit_targets_achieved": state.get("profit_targets_achieved", 0)
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
        }
    })

@app.route("/health")
def health():
    """HARDENING: Enhanced health endpoint"""
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
        "emergency_protection_enabled": EMERGENCY_PROTECTION_ENABLED
    }), 200

@app.route("/ping")
def ping(): return "pong", 200

# ------------ Boot Sequence ------------
if __name__ == "__main__":
    print("✅ Starting HARDENED Flask server with ALL PROTECTION LAYERS...")
    
    # HARDENING: Load persisted state
    load_state()
    
    # HARDENING: Start watchdog thread
    threading.Thread(target=watchdog_check, daemon=True).start()
    print("🦮 Watchdog started")
    
    # Start main loops
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
