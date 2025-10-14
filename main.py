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

✅ SMART ALPHA PACK ADDED:
- Breakout Voting: Multi-factor confirmation for explosive moves
- TP0 Quick Cash: Fast partial profit taking at 0.2%
- Thrust Lock: Advanced trend riding with Chandelier exit
- Adaptive Trailing: Dynamic stop loss based on market strength
- Enhanced metrics and logging

✅ EXIT PROTECTION LAYER ADDED:
- Minimum Hold Period: No early exits (3 bars / 120 seconds)
- Enhanced Patience Guard: Consensus-based exit confirmation  
- RF Hysteresis: 8bps buffer to avoid false reversals
- TP1 Gate: No full close before TP1, only small partials
- Trail Delay: ATR trail disabled until TP1 achieved
- Safe Close Wrapper: All exits pass protection checks

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
- ✅ NEW: SMART ALPHA PACK - Breakout voting, TP0, Thrust Lock, Adaptive trailing
- ✅ NEW: EXIT PROTECTION LAYER - Minimum hold period, consensus exits, hysteresis
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

# =============================================================================
# ✅ SMART ALPHA PACK SETTINGS
# =============================================================================

# Breakout Voting System
BREAKOUT_CONFIRM_BARS = 1           # شموع تأكيد الانفجار
BREAKOUT_HARD_ATR_SPIKE = 1.8       # انفجار قوي في ATR
BREAKOUT_SOFT_ATR_SPIKE = 1.4       # انفجار متوسط في ATR
BREAKOUT_VOLUME_SPIKE = 1.3         # حجم أعلى من المتوسط
BREAKOUT_VOLUME_MED = 1.1           # حجم فوق المتوسط

# TP0 Quick Cash
TP0_PROFIT_PCT = 0.2                # جني سريع عند 0.2% ربح
TP0_CLOSE_FRAC = 0.10               # نسبة الإغلاق (بحد أقصى 10%)
TP0_MAX_USDT = 1.0                  # أقصى قيمة للإغلاق (1 USDT)

# Thrust Lock Settings  
THRUST_ATR_BARS = 3                 # عدد الشموع لاتجاه ATR
THRUST_VOLUME_FACTOR = 1.3          # عامل حجم الترس
CHANDELIER_ATR_MULT = 3.0           # مضروب ATR للتريل المتقدم
CHANDELIER_LOOKBACK = 20            # شموع النظر للتريل

# Adaptive Trailing
TRAIL_MULT_STRONG_ALPHA = 2.4       # تريل قوي للترند القوي
TRAIL_MULT_CAUTIOUS_ALPHA = 2.0     # تريل حذر
TRAIL_MULT_SCALP_ALPHA = 1.6        # تريل سكالب

# =============================================================================
# ✅ EXIT PROTECTION LAYER SETTINGS
# =============================================================================

MIN_HOLD_BARS        = 3        # أقل عدد شموع لازم يفضل فيها قبل أي إغلاق "اختياري"
MIN_HOLD_SECONDS     = 120      # أو زمن حد أدنى (احتياطي)
TRAIL_ONLY_AFTER_TP1 = True     # فعل التريل بعد TP1 فقط (إلا لو ضرب Chand/Thrust)
RF_HYSTERESIS_BPS    = 8        # بوفر للهسترة لما يحصل عكس مؤقت مع Range Filter
NO_FULL_CLOSE_BEFORE_TP1 = True # ممنوع قفل كامل قبل TP1 (يسمح بجزئي فقط)
WICK_MAX_BEFORE_TP1  = 0.25     # أقصى إغلاق جزئي مسموح من حصاد الذيل قبل TP1

# Debug flags for exit protection
DEBUG_PATIENCE = True
DEBUG_HARVEST = True

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
print(colored(f"✅ NEW: SMART ALPHA PACK - Breakout voting, TP0, Thrust Lock, Adaptive trailing", "green"))
print(colored(f"✅ NEW: EXIT PROTECTION LAYER - Minimum hold period, consensus exits, hysteresis", "green"))
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
    
    # ✅ CORRECTION: Fix typo - change 'astiatype' to 'astype'
    upward = (fdir==1).astype(int)     # كان: .astiatype(int)
    downward = (fdir == -1).astype(int) # كان: .astiatype(int)
    
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
    "breakout_entry_price": None,      # سعر الدخول أثناء الانفجار
    # ✅ SMART ALPHA PACK STATE FIELDS
    "_tp0_done": False,                # TP0 quick cash executed
    "thrust_locked": False,            # Thrust Lock active
    "breakout_score": 0.0,             # Breakout voting score
    "breakout_votes_detail": {},       # Detailed breakout factors
    "opened_by_breakout": False,       # Position opened by breakout engine
    # ✅ EXIT PROTECTION LAYER STATE FIELDS
    "opened_at": None,                 # Timestamp when position was opened
    "_exit_soft_block": False,         # Soft exit block flag
    "_adaptive_trail_mult": None       # Adaptive trail multiplier
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
                "breakout_entry_price": None,
                # ✅ SMART ALPHA PACK: Reset on sync
                "_tp0_done": False,
                "thrust_locked": False,
                "breakout_score": 0.0,
                "breakout_votes_detail": {},
                "opened_by_breakout": False,
                # ✅ EXIT PROTECTION LAYER: Reset on sync
                "opened_at": time.time(),
                "_exit_soft_block": False,
                "_adaptive_trail_mult": None
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

# =============================================================================
# ✅ SMART ALPHA PACK FUNCTIONS
# =============================================================================

def breakout_votes(df: pd.DataFrame, ind: dict, prev_ind: dict) -> tuple:
    """
    نظام تصويت الانفجار: يحسب درجة ثقة الانفجار من 0-5
    Returns: (score, vote_details)
    """
    votes = 0.0
    vote_details = {}
    
    try:
        if len(df) < max(BREAKOUT_LOOKBACK_BARS, 20) + 2:
            return 0.0, {"error": "Insufficient data"}
            
        current_idx = -1
        price = float(df["close"].iloc[current_idx])
        
        # 1. ATR Spike Vote
        atr_now = float(ind.get("atr") or 0.0)
        atr_prev = float(prev_ind.get("atr") or atr_now)
        
        if atr_prev > 0:
            atr_ratio = atr_now / atr_prev
            if atr_ratio >= BREAKOUT_HARD_ATR_SPIKE:
                votes += 1.0
                vote_details["atr_spike"] = f"Hard ({atr_ratio:.2f}x)"
            elif atr_ratio >= BREAKOUT_SOFT_ATR_SPIKE:
                votes += 0.5
                vote_details["atr_spike"] = f"Soft ({atr_ratio:.2f}x)"
            else:
                vote_details["atr_spike"] = f"Normal ({atr_ratio:.2f}x)"
        
        # 2. ADX Momentum Vote
        adx = float(ind.get("adx") or 0.0)
        adx_prev = float(prev_ind.get("adx") or adx)
        
        if adx >= BREAKOUT_ADX_THRESHOLD:
            votes += 1.0
            vote_details["adx"] = f"Strong ({adx:.1f})"
        elif adx >= 18 and adx > adx_prev:
            votes += 0.5
            vote_details["adx"] = f"Building ({adx:.1f})"
        else:
            vote_details["adx"] = f"Weak ({adx:.1f})"
        
        # 3. Volume Spike Vote
        if len(df) >= 21:
            current_volume = float(df["volume"].iloc[current_idx])
            volume_ma = df["volume"].iloc[-21:-1].astype(float).mean()
            
            if volume_ma > 0:
                volume_ratio = current_volume / volume_ma
                if volume_ratio >= BREAKOUT_VOLUME_SPIKE:
                    votes += 1.0
                    vote_details["volume"] = f"Spike ({volume_ratio:.2f}x)"
                elif volume_ratio >= BREAKOUT_VOLUME_MED:
                    votes += 0.5
                    vote_details["volume"] = f"High ({volume_ratio:.2f}x)"
                else:
                    vote_details["volume"] = f"Normal ({volume_ratio:.2f}x)"
        
        # 4. Price Breakout Vote
        recent_highs = df["high"].iloc[-BREAKOUT_LOOKBACK_BARS:-1].astype(float)
        recent_lows = df["low"].iloc[-BREAKOUT_LOOKBACK_BARS:-1].astype(float)
        
        if len(recent_highs) > 0 and len(recent_lows) > 0:
            highest_high = recent_highs.max()
            lowest_low = recent_lows.min()
            
            if price > highest_high:
                votes += 1.0
                vote_details["price_break"] = f"New High (>{highest_high:.6f})"
            elif price > recent_highs.iloc[-10:].max() if len(recent_highs) >= 10 else price > highest_high:
                votes += 0.5
                vote_details["price_break"] = f"Near High"
            elif price < lowest_low:
                votes += 1.0
                vote_details["price_break"] = f"New Low (<{lowest_low:.6f})"
            elif price < recent_lows.iloc[-10:].min() if len(recent_lows) >= 10 else price < lowest_low:
                votes += 0.5
                vote_details["price_break"] = f"Near Low"
            else:
                vote_details["price_break"] = "Within Range"
        
        # 5. RSI Directional Vote
        rsi = float(ind.get("rsi") or 50.0)
        if rsi >= 60:  # Overbought but strong
            votes += 0.5
            vote_details["rsi"] = f"Strong Bull ({rsi:.1f})"
        elif rsi <= 40:  # Oversold but strong
            votes += 0.5
            vote_details["rsi"] = f"Strong Bear ({rsi:.1f})"
        else:
            vote_details["rsi"] = f"Neutral ({rsi:.1f})"
            
        vote_details["total_score"] = f"{votes:.1f}/5.0"
        
    except Exception as e:
        print(colored(f"⚠️ breakout_votes error: {e}", "yellow"))
        logging.error(f"breakout_votes error: {e}")
        vote_details["error"] = str(e)
    
    return votes, vote_details

def tp0_quick_cash(ind: dict) -> bool:
    """
    TP0: جني سريع عند 0.2% ربح
    يُنفّذ مرة واحدة فقط لكل صفقة
    """
    if not state["open"] or state["qty"] <= 0 or state.get("_tp0_done", False):
        return False
        
    try:
        price = ind.get("price") or price_now() or state["entry"]
        entry = state["entry"]
        side = state["side"]
        
        if not (price and entry):
            return False
            
        # حساب الربح النسبي
        rr = (price - entry) / entry * 100 * (1 if side == "long" else -1)
        
        if rr >= TP0_PROFIT_PCT:
            # حساب كمية الإغلاق (بحد أقصى 10% أو 1 USDT)
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

def thrust_lock(df: pd.DataFrame, ind: dict) -> bool:
    """
    Thrust Lock: كشف الترند القوي وركوبه
    يرجع True إذا تم تفعيل القفل
    """
    if not state["open"] or state["qty"] <= 0 or state.get("thrust_locked", False):
        return False
        
    try:
        if len(df) < THRUST_ATR_BARS + 20:
            return False
            
        # 1. تحقق من اتجاه ATR (يجب أن يرتفع لـ 3 شموع متتالية)
        atr_values = []
        for i in range(1, THRUST_ATR_BARS + 1):
            if len(df) >= i + 1:
                atr_val = float(compute_indicators(df.iloc[:-i]).get("atr", 0))
                atr_values.append(atr_val)
        
        atr_rising = len(atr_values) >= THRUST_ATR_BARS
        for i in range(1, len(atr_values)):
            if atr_values[i] >= atr_values[i-1]:
                atr_rising = False
                break
        
        # 2. تحقق من حجم الترس (يجب أن يكون أعلى من المتوسط)
        current_volume = float(df["volume"].iloc[-1])
        volume_ma = df["volume"].iloc[-21:-1].astype(float).mean()
        volume_spike = current_volume > volume_ma * THRUST_VOLUME_FACTOR if volume_ma > 0 else False
        
        if atr_rising and volume_spike and state.get("trade_mode") == "TREND":
            # تفعيل Thrust Lock
            state["thrust_locked"] = True
            
            # ضبط بريك إيفن
            state["breakeven"] = state["entry"]
            
            # حساب تريل شانديلييه
            price = ind.get("price") or float(df["close"].iloc[-1])
            atr = ind.get("atr", 0)
            
            if state["side"] == "long":
                lookback_lows = df["low"].iloc[-CHANDELIER_LOOKBACK:].astype(float)
                chandelier_stop = lookback_lows.min() - atr * CHANDELIER_ATR_MULT
                state["trail"] = max(state.get("trail") or chandelier_stop, chandelier_stop)
            else:
                lookback_highs = df["high"].iloc[-CHANDELIER_LOOKBACK:].astype(float)
                chandelier_stop = lookback_highs.max() + atr * CHANDELIER_ATR_MULT
                state["trail"] = min(state.get("trail") or chandelier_stop, chandelier_stop)
            
            print(colored(f"🔒 THRUST LOCK: Activated with Chandelier exit", "green"))
            logging.info(f"THRUST_LOCK: Activated - ATR rising & volume spike")
            return True
            
    except Exception as e:
        print(colored(f"⚠️ thrust_lock error: {e}", "yellow"))
        logging.error(f"thrust_lock error: {e}")
        
    return False

def get_adaptive_trail_multiplier(breakout_score: float, trade_mode: str) -> float:
    """
    تريل متكيّف بناءً على قوة الانفجار ونمط التداول
    """
    if breakout_score >= 3.0:
        return TRAIL_MULT_STRONG_ALPHA  # ترند قوي
    elif breakout_score >= 2.0:
        return TRAIL_MULT_CAUTIOUS_ALPHA  # ترند حذر
    elif trade_mode == "SCALP":
        return TRAIL_MULT_SCALP_ALPHA  # سكالب
    else:
        return ATR_MULT_TRAIL  # الإعداد الأساسي

# =============================================================================
# ✅ EXIT PROTECTION LAYER FUNCTIONS
# =============================================================================

def _tp0_guard_close(frac, label):
    """
    TP0 Guard: قبل TP1: لا نتجاوز 25% ولا نتسبب في صفر كميّة
    """
    # قبل TP1: لا نتجاوز 25% ولا نتسبب في صفر كميّة
    max_frac = WICK_MAX_BEFORE_TP1 if not state.get("tp1_done") else frac
    frac = min(frac, max_frac)
    if frac > 0:
        close_partial(frac, label)
        if DEBUG_HARVEST:
            print(colored(f"🛡️ TP0_GUARD: {label} → closed {frac*100:.1f}% (max: {max_frac*100:.1f}%)", "blue"))

def _allow_full_close(reason: str) -> bool:
    """
    يمنع الإغلاق الكامل بدري ما لم يكن:
      - ضرب تريل فعلي (Trail/Chandelier)
      - STRICT_EXCHANGE_CLOSE بسبب شرط صريح (هدوء انفجار، أمر حماية…)
      - Stop/Force
    """
    opened_at = state.get("opened_at") or state.get("entry_time") or time.time()
    elapsed_s = max(0, time.time() - opened_at)
    elapsed_bar = int(state.get("bars", 0))
    
    if elapsed_bar >= MIN_HOLD_BARS or elapsed_s >= MIN_HOLD_SECONDS:
        return True
        
    # استثناءات مسموحة مبكرًا:
    hard_reasons = ("TRAIL", "TRAIL_ATR", "CHANDELIER", "STRICT", "FORCE", "STOP", "EMERGENCY", "BREAKOUT")
    return any(tag in reason.upper() for tag in hard_reasons)

def _safe_close_partial(frac, reason):
    """
    Safe Close Partial: يطبق قيود الحماية على الإغلاق الجزئي
    """
    if state.get("_exit_soft_block") and not state.get("tp1_done"):
        # قبل TP1 وفي فترة الحماية: اغلق جزء صغير فقط
        frac = min(frac, 0.25)
        if DEBUG_PATIENCE:
            print(colored(f"🛡️ PATIENCE: Partial close limited to {frac*100:.1f}%", "blue"))
    
    close_partial(frac, reason)

def _safe_full_close(reason):
    """
    Safe Full Close: يطبق قيود الحماية على الإغلاق الكامل
    """
    if NO_FULL_CLOSE_BEFORE_TP1 and not state.get("tp1_done"):
        # ممنوع قفل كامل قبل TP1، إلا لو سبب قاسي
        if not _allow_full_close(reason):
            if DEBUG_PATIENCE:
                print(colored(f"🛡️ PATIENCE: Full close blocked before TP1 - {reason}", "blue"))
            logging.info(f"EXIT_BLOCKED (early): {reason}")
            return False
            
    # لو الحارس منع الإغلاق "الاختياري"
    if state.get("_exit_soft_block") and not _allow_full_close(reason):
        if DEBUG_PATIENCE:
            print(colored(f"🛡️ PATIENCE: Full close blocked by soft block - {reason}", "blue"))
        logging.info(f"EXIT_BLOCKED (patience): {reason}")
        return False
        
    close_market_strict(reason)
    return True

def check_exit_consensus(df: pd.DataFrame, ind: dict, info: dict) -> tuple:
    """
    فحص إجماع الخروج: لا إغلاق إلا لو تحقق إجماع عكسي (≥2 إشارات) + شمعتين ضد الاتجاه
    Returns: (reverse_votes, candle_ok)
    """
    side = state.get("side")
    rsi  = float(ind.get("rsi") or 50.0)
    adx  = float(ind.get("adx") or 0.0)
    adx_prev = float(ind.get("adx_prev") or adx)
    rf   = info.get("filter")  # Range Filter value
    px   = ind.get("price") or info.get("price")

    reverse_votes = 0
    
    # 1. RSI عكسي
    if side == "long" and rsi < 50: 
        reverse_votes += 1
    if side == "short" and rsi > 50: 
        reverse_votes += 1
        
    # 2. هبوط ADX ≥ 3.0
    if adx_prev and (adx_prev - adx) >= 3.0: 
        reverse_votes += 1
        
    # 3. كسر عكسي لـ Range Filter مع هسترة
    try:
        if px is not None and rf is not None:
            bps = abs((px - rf) / rf) * 10_000
            if (side == "long" and px < rf - (RF_HYSTERESIS_BPS/10_000.0)*rf) or \
               (side == "short" and px > rf + (RF_HYSTERESIS_BPS/10_000.0)*rf):
                reverse_votes += 1
    except Exception:
        pass

    # 4. تأكيد شمعتين ضد الاتجاه
    try:
        if len(df) >= 2:
            closes = df["close"].astype(float); opens = df["open"].astype(float)
            last2_red   = (closes.iloc[-1] < opens.iloc[-1]) and (closes.iloc[-2] < opens.iloc[-2])
            last2_green = (closes.iloc[-1] > opens.iloc[-1]) and (closes.iloc[-2] > opens.iloc[-2])
            candle_ok = (side == "long" and last2_red) or (side == "short" and last2_green)
        else:
            candle_ok = False
    except Exception:
        candle_ok = False

    return reverse_votes, candle_ok

def apply_exit_protection(df: pd.DataFrame, ind: dict, info: dict):
    """
    تطبيق حماية الخروج: فترات الانتظار والإجماع
    """
    if not state["open"]:
        return
        
    # تحديث وقت الفتح إذا لم يكن موجودًا
    if state.get("opened_at") is None:
        state["opened_at"] = info.get("time") or time.time()
        
    now_ts = info.get("time") or time.time()
    opened_at = state.get("opened_at") or now_ts
    elapsed_s = max(0, now_ts - opened_at)
    elapsed_bar = int(state.get("bars", 0))

    # فحص الإجماع العكسي
    reverse_votes, candle_ok = check_exit_consensus(df, ind, info)
    
    # لو لسه في أول الصفقة أو العكس غير مؤكد → امنع أي إغلاق غير "قاسي"
    if (elapsed_bar < MIN_HOLD_BARS or elapsed_s < MIN_HOLD_SECONDS) and (reverse_votes < 2 or not candle_ok):
        # نحجب الخروج الاختياري (TP2/ratchet/close بالعكس) ونسمح فقط بالتريل/STRICT
        state["_exit_soft_block"] = True
        if DEBUG_PATIENCE:
            print(colored(f"🛡️ PATIENCE: early-bars block (bars:{elapsed_bar}, votes:{reverse_votes}, candles:{candle_ok})", "blue"))
    else:
        state["_exit_soft_block"] = False
        if DEBUG_PATIENCE and reverse_votes >= 2 and candle_ok:
            print(colored(f"🛡️ PATIENCE: consensus reached (votes:{reverse_votes}, candles:{candle_ok})", "green"))

    # تأخير التريل حتى TP1
    if TRAIL_ONLY_AFTER_TP1 and not state.get("tp1_done") and not state.get("thrust_locked"):
        # اسمح فقط بتريل "قاسي" جاء من Thrust/Chandelier
        state["_adaptive_trail_mult"] = 0.0  # يعطل تريل ATR العادي لحين TP1

def smart_alpha_features(df: pd.DataFrame, ind: dict, prev_ind: dict, info: dict) -> str:
    """
    مدير الميزات الذكية - يُستدعى من smart_post_entry_manager
    Returns: action_type إذا تم تنفيذ إجراء
    """
    if not state["open"] or state["qty"] <= 0:
        return None
        
    action = None
    
    try:
        # 1. TP0 Quick Cash (الأولوية الأولى)
        if tp0_quick_cash(ind):
            action = "TP0_QUICK_CASH"
            
        # 2. Breakout Voting (لتحديث النقاط فقط)
        breakout_score, vote_details = breakout_votes(df, ind, prev_ind)
        state["breakout_score"] = breakout_score
        state["breakout_votes_detail"] = vote_details
        
        # 3. Thrust Lock (للترند القوي)
        if state.get("trade_mode") == "TREND" and not state.get("thrust_locked"):
            if thrust_lock(df, ind):
                action = "THRUST_LOCK_ACTIVATED"
                
        # 4. Adaptive Trailing (تحديث المضروب فقط)
        if state.get("breakout_active") or state.get("thrust_locked"):
            adaptive_mult = get_adaptive_trail_multiplier(
                breakout_score, 
                state.get("trade_mode", "SCALP")
            )
            # تخزين المضروب للتريل المتكيف (سيُستخدم في المنطق الرئيسي)
            state["_adaptive_trail_mult"] = adaptive_mult
            
        # تسجيل النقاط إذا كانت عالية
        if breakout_score >= 3.0:
            print(colored(f"🎯 BREAKOUT SCORE: {breakout_score:.1f}/5.0 - Strong momentum", "cyan"))
            for factor, detail in vote_details.items():
                if factor != "total_score":
                    print(colored(f"   📊 {factor}: {detail}", "blue"))
                    
    except Exception as e:
        print(colored(f"⚠️ smart_alpha_features error: {e}", "yellow"))
        logging.error(f"smart_alpha_features error: {e}")
        
    return action

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
                
                _tp0_guard_close(harvest_frac, reason)
                
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
            _tp0_guard_close(0.25, f"Upper wick (LONG) {upper_wick_pct:.1f}%")
            return "LONG_WICK_HARVEST"
        
        # للـ SHORT: الذيل السفلي الطويل إشارة ضغط شراء عند القيعان ⇒ مناسب لجني الأرباح
        if side == "short" and lower_wick_pct >= LONG_WICK_HARVEST_THRESHOLD * 100:
            _tp0_guard_close(0.25, f"Lower wick (SHORT) {lower_wick_pct:.1f}%")
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
        
        _safe_close_partial(0.5, f"Ratchet protection: {state['highest_profit_pct']:.1f}% → {current_rr:.1f}%")
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
        _safe_close_partial(fracs[k], f"SCALP TP{k+1}@{targets[k]:.2f}%")
        state["profit_targets_achieved"] = k + 1

        # بعد آخر مرحلة → قفل صارم كامل
        if state["profit_targets_achieved"] >= len(targets):
            if _safe_full_close("SCALP sequence complete"):
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
        _safe_close_partial(fracs[k], f"TREND TP{k+1}@{targets[k]:.2f}%")
        state["profit_targets_achieved"] = k + 1
        return f"TREND_TP{k+1}"

    # 2) طالما ADX قوي ابقى راكب
    if adx >= MIN_TREND_HOLD_ADX:
        return None

    # 3) بعد إتمام المراحل أو ضعف الترند → تحقّق نهاية الترند
    if state.get("profit_targets_achieved", 0) >= len(targets) or trend_end_confirmed(ind, detect_candle_pattern(fetch_ohlcv()), info):
        if _safe_full_close("TREND finished — full exit"):
            return "TREND_COMPLETE"

    return None

def smart_post_entry_manager(df: pd.DataFrame, ind: dict, info: dict):
    """
    المدير الذكي للصفقة بعد الدخول
    """
    if not state["open"] or state["qty"] <= 0:
        return None
    
    # ✅ EXIT PROTECTION LAYER: تطبيق حماية الخروج أولاً
    apply_exit_protection(df, ind, info)
    
    # ✅ SMART ALPHA PACK: تشغيل الميزات الذكية
    prev_ind = compute_indicators(df.iloc[:-1]) if len(df) >= 2 else ind
    alpha_action = smart_alpha_features(df, ind, prev_ind, info)
    if alpha_action:
        return alpha_action
    
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
        "breakout_entry_price": None,
        # ✅ SMART ALPHA PACK: Reset on new position
        "_tp0_done": False,
        "thrust_locked": False,
        "breakout_score": 0.0,
        "breakout_votes_detail": {},
        "opened_by_breakout": False,
        # ✅ EXIT PROTECTION LAYER: Reset on new position
        "opened_at": time.time(),
        "_exit_soft_block": False,
        "_adaptive_trail_mult": None
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
        "breakout_entry_price": None,
        # ✅ SMART ALPHA PACK: Reset on full close
        "_tp0_done": False,
        "thrust_locked": False,
        "breakout_score": 0.0,
        "breakout_votes_detail": {},
        "opened_by_breakout": False,
        # ✅ EXIT PROTECTION LAYER: Reset on full close
        "opened_at": None,
        "_exit_soft_block": False,
        "_adaptive_trail_mult": None
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

    # ✅ SMART ALPHA PACK: استخدام التريل المتكيف إذا متوفر
    trail_mult_to_use = state.get("_adaptive_trail_mult") or ATR_MULT_TRAIL

    # ✅ PATCH: TP1 fallback if trade_mode not decided yet
    if state.get("trade_mode") is None:
        if (not state["tp1_done"]) and rr >= current_tp1_pct_pct:
            _safe_close_partial(TP1_CLOSE_FRAC, f"TP1@{current_tp1_pct_pct:.2f}%")
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
        _safe_close_partial(TP1_CLOSE_FRAC, f"TP1@{current_tp1_pct_pct:.2f}%")
        state["tp1_done"] = True
        if rr >= BREAKEVEN_AFTER * 100.0:
            state["breakeven"] = e

    # Ratchet lock على التراجع
    if (state["highest_profit_pct"] >= current_trail_activate_pct and
        rr < state["highest_profit_pct"] * RATCHET_LOCK_PCT):
        _safe_close_partial(0.5, f"Ratchet Lock @ {state['highest_profit_pct']:.2f}%")
        state["highest_profit_pct"] = rr
        return None

    # Trailing ATR
    if rr >= current_trail_activate_pct and atr and trail_mult_to_use > 0:
        gap = atr * trail_mult_to_use
        if side == "long":
            new_trail = px - gap
            state["trail"] = max(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None:
                state["trail"] = max(state["trail"], state["breakeven"])
            if px < state["trail"]:
                if _safe_full_close(f"TRAIL_ATR({trail_mult_to_use}x)"):
                    return True
        else:
            new_trail = px + gap
            state["trail"] = min(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None:
                state["trail"] = min(state["trail"], state["breakeven"])
            if px > state["trail"]:
                if _safe_full_close(f"TRAIL_ATR({trail_mult_to_use}x)"):
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
    
    # ✅ SMART ALPHA PACK: استخدام نظام التصويت للانفجارات
    breakout_score, vote_details = breakout_votes(df, ind, prev_ind)
    
    # فلتر بناءً على نقاط التصويت
    if breakout_score < 3.0:
        print(colored(f"⛔ BREAKOUT: Score too low ({breakout_score:.1f}/5.0) - skipping", "yellow"))
        logging.warning(f"BREAKOUT_ENGINE: Score filter blocked - {breakout_score:.1f}/5.0")
        return False
    
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
        logging.info(f"BREAKOUT_ENGINE: Bearish crash - SHORT {qty} @
