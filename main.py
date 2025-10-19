# -*- coding: utf-8 -*-
"""
RF Futures Bot — Trend-Only Pro (BingX Perp, CCXT) — FINAL
• Entry: Closed-candle Range Filter signals only
• Post-entry: Supertrend + ADX/RSI/ATR dynamic harvesting
• Dynamic TP ladder (ATR% + consensus) + Ratchet Lock
• Impulse & Wick harvesting + Thrust Lock (Chandelier exit)
• Emergency protection layer + Breakout Engine (with exits)
• Strict exchange close + Patience Guard (no early full exits)
• TVR (Time-Volume-Reaction) Enhanced Scout System
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
TRAIL_ACTIVATE = 1.20   # تفعيل التريل عند تجاوز هذا الربح
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
LONG_WICK_HARVEST_THRESHOLD = 0.45  # 45% من مدى الشمعة
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
MIN_HOLD_BARS        = 8
MIN_HOLD_SECONDS     = 3600
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
HOLD_RATCHET_LOCK_PCT_ON_HOLD = 0.85

# Keepalive
KEEPALIVE_SECONDS = 50

# ——— وضع المتداول الصبور ———
PATIENT_TRADER_MODE = True  # يفعل الصبر
PATIENT_HOLD_BARS = 8       # يمسك على الأقل 8 شموع (≈ ساعتين على فريم 15م)
PATIENT_HOLD_SECONDS = 3600 # أو ساعة زمنياً أيهما أسبق
EXIT_ONLY_ON_OPPOSITE_RF = True  # لا إغلاق كامل إلا بإشارة RF عكسية/طوارئ

# =================== TVR (Time-Volume-Reaction) SETTINGS ===================
TVR_ENABLED          = True       # تفعيل النظام
TVR_BUCKETS          = 96         # 96 باكت = يوم كامل على فريم 15m
TVR_VOL_SPIKE        = 1.8        # نسبة سبايك الحجم مقابل المتوسط الزمني
TVR_REACTION_ATR     = 0.9        # جسم الشمعة/ATR كحد أدنى لاعتبارها اندفاعية
TVR_MIN_ADX          = 14         # ADX الأدنى لقبول السبايك
TVR_SCOUT_FRAC       = 0.50       # حجم دخول Scout (نصف الحجم العادي)
TVR_TIMEOUT_BARS     = 6          # أقصى عدد شموع يظل فيها وضع Scout "خاص"
TVR_MAX_SPREAD_BPS   = 6.0        # لا ندخل إن كان السبريد كبير

# ===== TVR live-scout (اختياري) ====
TVR_USE_LIVE_BAR        = True     # فعّل شمّة حيّة
TVR_LIVE_VOL_SPIKE      = 2.2      # حجم لحظي ≥ 2.2× المتوقّع
TVR_LIVE_REACTION_ATR   = 1.4      # جسم الشمعة ≥ 1.4×ATR
TVR_LIVE_MIN_ELAPSED    = 0.25     # لازم يمُر ≥25% من زمن الشمعة
TVR_LIVE_MAX_SPREAD_BPS = 6.0
TVR_SCOUT_TRAIL_MULT    = 1.2      # معامل التريل للدخول السكاوت

# ===== Trap / Stop-Hunt Guard =====
TRAP_ENABLED = True
TRAP_WICK_PCT = 60.0   # % من مدى الشمعة لازم يكون ذيل
TRAP_BODY_MAX_PCT = 25.0   # الجسم صغير → رفض
TRAP_ATR_MIN = 0.6    # مدى الشمعة ≥ 0.6×ATR
TRAP_VOL_SPIKE = 1.30   # حجم أكبر من متوسط 20 شمعة ×1.3
TRAP_PROX_BPS = 12.0   # قرب من EQH/EQL/OB بالـ bps
TRAP_HOLD_BARS = 4      # نمنع القفل الكامل N شموع بعد الفخ

# --- Residual / Dust guard ---
RESIDUAL_MIN_QTY   = 10.0                 # أقل كمية نسمح نكمل بها الصفقة
RESIDUAL_MIN_USDT  = 0.0                  # أو حد بالدولار (اختياري)
RESPECT_PATIENT_MODE_FOR_DUST = True      # لا تقفل كامل قبل TP1 في وضع الصبر

# ===== SMC Pro (Structure / Liquidity) =====
SMC_ENABLED = True
SMC_EQHL_LOOKBACK    = 30   # البحث عن Equal Highs/Lows
SMC_OB_LOOKBACK      = 40   # نطاق البحث عن الـ Order Blocks الأخيرة
SMC_DISPLACEMENT_ATR = 1.20 # اندفاع (جسم ≥ 1.2×ATR) لتأكيد OB
SMC_WICK_MAX         = 0.35 # أقصى نسبة ذيل من مدى الشمعة لاعتماد شمعة OB
SMC_VOL_SPIKE        = 1.30 # سبايك حجم لتأكيد الـ OB/الاندفاع
SMC_FVG_MAX_GAP_ATR  = 2.00 # أقصى فجوة FVG (كـ ATR) نعتمدها
SMC_PREM_WIN         = 40   # نافذة Premium/Discount (عدد شموع)
SMC_FIB_EXTS         = [1.00, 1.272, 1.618]  # أهداف امتداد
SMC_SCORE_STRONG     = 0.65
SMC_SCORE_HOLD       = 0.55
SMC_SCORE_WARN       = 0.40
SMC_WARN_PARTIAL     = 0.20
SMC_TIGHT_TRAIL_MULT = 1.6
SMC_WIDE_TRAIL_MULT  = 2.4

# ============== SMC / MSS SETTINGS ==============
SMC_MSS_ENABLED = True
SMC_STRONG_ADX = 22             # أدنى ADX لاعتبار MSS مؤثرًا
SMC_BUFFER_BPS = 6.0            # هامش كسر القمة/القاع (بِبْس)
SMC_PARTIAL_ON_OPPOSITE = 0.25  # نسبة إغلاق جزئي عند MSS عكسي قوي
SMC_MAX_EXTRA_PARTIAL = 0.30    # سقف كل الإغلاقات الجزئية الإضافية التي يسمح بها MSS
SMC_BOOST_TRAIL_MULT = 0.4      # زيادة على معامل ATR للتريل عند MSS موافق للاتجاه
SMC_DEFER_TP_ON_ALIGNED = True  # تأجيل TP مرة واحدة عندما MSS مع الاتجاه

# زوّد وزن الـ Structure في الأوركسترا
ORCH_WEIGHTS = {"momentum": 0.25, "volatility": 0.20, "trend": 0.20, "structure": 0.35}
ORCH_STRONG = 0.65

# === Patient Trading – لا قفل كامل بدري ===
NEVER_FULL_CLOSE_BEFORE_TP1   = True   # يمنع أي قفل كامل قبل TP1 إلا طوارئ/Trail
OPP_RF_NEED_BARS              = 2      # عدد إشارات RF عكسية مغلقة متتالية قبل التفكير في قفل كامل
OPP_RF_MIN_ADX                = 22     # لازم ADX ≥ هذا عند العكسية
OPP_RF_MIN_HYST_BPS           = 8.0    # كسر واضح عن الـ RF بالـbps
OPP_RF_DEFEND_PARTIAL         = 0.25   # نسبة الجني عند أول عكسية (يتعدل تلقائيًا حسب TP1)
OPP_RF_TIGHT_TRAIL_MULT       = 1.6    # تشديد التريل عند العكسية

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
    "opened_at": None,
    # TVR
    "_tvr_profile": None,
    "tvr_active": False,
    "tvr_bars_alive": 0,
    "tvr_vol_ratio": None,
    "tvr_reaction": None,
    "tvr_bucket": None,
    "tvr_direction": None,  # 1=up, -1=down
    # SMC
    "smc_mss": None,
    "smc_extra_partials": 0.0,
    # Trap Guard
    "_trap_active": False,
    "_trap_dir": None,
    "_trap_left": 0,
    "_last_trap_ts": None,
    # Opposite RF votes
    "_opp_rf_votes": 0,
    "_trend_exit_votes": 0,
}
compound_pnl = 0.0
last_signal_id = None
post_close_cooldown = 0
wait_for_next_signal_side = None
last_close_signal_time = None

_state_lock = threading.Lock()

# --- SAFE STATE ACCESS HELPERS ---
def get_state():
    """يرجع نسخة آمنة من state للقراءة فقط"""
    with _state_lock:
        return dict(state)

def set_state(upd: dict):
    """تحديث state بأمان مع القفل"""
    if not isinstance(upd, dict):
        return
    with _state_lock:
        state.update(upd)

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

# =================== PATCH HELPERS (NEW) ===================
def update_state(mutator):
    """
    يطبق تعديلات على state تحت القفل. استخدمه لتعديلات متعددة الحقول.
    """
    if not callable(mutator):
        return
    with _state_lock:
        mutator(state)

def _safe_price(default=None):
    """قراءة سعر آمنة لتفادي None/NaN."""
    try:
        p = price_now()
        if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
            return default
        return p
    except Exception:
        return default

def _min_tradable_qty(price: float) -> float:
    """
    أقل كمية قابلة للتداول وفق LOT_MIN/LOT_STEP. رجوع 1.0 كحل أخير.
    """
    try:
        if LOT_MIN and LOT_MIN > 0:
            return float(LOT_MIN)
        if LOT_STEP and LOT_STEP > 0:
            return float(LOT_STEP)
        return 1.0
    except Exception:
        return 1.0

def _close_partial_min_check(qty_close: float) -> bool:
    """رفض الجزئي إن كان دون الحد الأدنى الفعلي."""
    min_qty = _min_tradable_qty(_safe_price(state.get("entry")) or state.get("entry") or 0.0)
    if qty_close < (min_qty or 0.0):
        print(colored(f"⚠️ skip partial close (amount={fmt(qty_close,4)} < min lot {fmt(min_qty,4)})", "yellow"))
        return False
    return True

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

# =================== TRAP / FAKEOUT GUARD ===================
def _near_level(px, lvl, bps):
    try: 
        return abs((px-lvl)/lvl)*10000.0 <= bps
    except Exception: 
        return False

def detect_stop_hunt(df: pd.DataFrame, ind: dict, levels: dict):
    """يرصد سحب سيولة بفتيلة طويلة عند قمم/قيعان/OB مع حجم مرتفع."""
    if not TRAP_ENABLED or len(df) < 3: 
        return None
    try:
        o = float(df["open"].iloc[-1]); h = float(df["high"].iloc[-1])
        l = float(df["low"].iloc[-1]);  c = float(df["close"].iloc[-1])
        rng = max(h-l, 1e-12); body = abs(c-o)
        upper = h - max(o,c); lower = min(o,c) - l
        upper_pct = upper/rng*100.0; lower_pct = lower/rng*100.0; body_pct = body/rng*100.0
        atr = float(ind.get("atr") or 0.0)
        v   = float(df["volume"].iloc[-1])
        vma = df["volume"].iloc[-21:-1].astype(float).mean() if len(df)>=21 else 0.0
        vol_ok = (vma>0 and v/vma >= TRAP_VOL_SPIKE)

        eqh = (levels or {}).get("eqh")
        eql = (levels or {}).get("eql")
        ob  = (levels or {}).get("ob")  # dict(side/bot/top)

        near_eqh = (eqh and _near_level(h, eqh, TRAP_PROX_BPS))
        near_eql = (eql and _near_level(l, eql, TRAP_PROX_BPS))
        near_ob_res = (ob and ob.get("side")=="bear" and _near_level(h, ob["bot"], TRAP_PROX_BPS))
        near_ob_sup = (ob and ob.get("side")=="bull" and _near_level(l, ob["top"], TRAP_PROX_BPS))

        bull_trap = (lower_pct>=TRAP_WICK_PCT and body_pct<=TRAP_BODY_MAX_PCT and atr>0 and (rng/atr)>=TRAP_ATR_MIN and (near_eql or near_ob_sup))
        bear_trap = (upper_pct>=TRAP_WICK_PCT and body_pct<=TRAP_BODY_MAX_PCT and atr>0 and (rng/atr)>=TRAP_ATR_MIN and (near_eqh or near_ob_res))

        if (bull_trap or bear_trap) and vol_ok:
            return {"trap": "bull" if bull_trap else "bear", "ts": int(df["time"].iloc[-1])}
    except Exception as e:
        logging.error(f"detect_stop_hunt error: {e}")
    return None

# (REPLACED) apply_trap_guard with atomic updates
def apply_trap_guard(trap: dict, ind: dict):
    """تفعيل حماية الفخ: جزئي دفاعي + تعادل + تريل مشدود، بكتابة ذرّية للـ state."""
    px  = ind.get("price") or _safe_price(state.get("entry")) or state.get("entry")
    atr = float(ind.get("atr") or 0.0)

    update_state(lambda s: (
        s.__setitem__("_trap_active", True),
        s.__setitem__("_trap_dir", trap.get("trap")),
        s.__setitem__("_trap_left", int(TRAP_HOLD_BARS)),
        s.__setitem__("_last_trap_ts", trap.get("ts")),
        s.__setitem__("breakeven", s.get("breakeven") or s.get("entry"))
    ))

    if state.get("open") and state["qty"] > 0:
        close_partial(0.15, f"Trap guard partial - {trap.get('trap')} trap detected")

    if atr > 0 and px and state.get("open"):
        gap = atr * max(state.get("_adaptive_trail_mult") or ATR_MULT_TRAIL, 1.8)
        def _tighten_trail(s):
            side = s.get("side")
            cur  = s.get("trail")
            if side == "long":
                nt = max(cur or (px - gap), px - gap)
            else:
                nt = min(cur or (px + gap), px + gap)
            s["trail"] = nt
        update_state(_tighten_trail)

    logging.info(f"TRAP GUARD ACTIVATED: {trap.get('trap')} trap - blocking full close for {TRAP_HOLD_BARS} bars")

# =================== TVR (Time-Volume-Reaction) FUNCTIONS ===================
def _tvr_bucket_len_min():
    # عدد دقائق الباكت الواحد (يوم = 1440 دقيقة)
    return max(1, 1440 // int(TVR_BUCKETS or 96))

def _tvr_bucket_index(ts_ms: int) -> int:
    """يعطي باكت الوقت (0..TVR_BUCKETS-1) اعتمادًا على وقت الشمعة UTC."""
    bucket_min = _tvr_bucket_len_min()
    mins_in_day = int((ts_ms // 60000) % 1440)
    return int(mins_in_day // bucket_min) % int(TVR_BUCKETS or 96)

def build_tvr_profile(df: pd.DataFrame):
    """
    نبني بروفايل حجم زمني: متوسط/ميديان حجم كل باكت وقتي عبر آخر عدة أيام.
    لا نستخدم أي إنترنت – فقط من بيانات OHLCV الموجودة.
    """
    if not TVR_ENABLED or len(df) < TVR_BUCKETS*2:
        return None
    try:
        vols = df["volume"].astype(float)
        times = df["time"].astype(int)
        bucket_min = _tvr_bucket_len_min()
        bucket_series = ((times // 60000) % 1440 // bucket_min).astype(int)
        grp = vols.groupby(bucket_series)
        med = grp.median()  # ميديان أكثر ثباتًا
        prof = [float(med.get(i, vols.median())) for i in range(int(TVR_BUCKETS or 96))]
        return prof
    except Exception:
        return None

def _bar_elapsed_frac(df):
    tf = _interval_seconds(INTERVAL)
    left = time_to_candle_close(df, USE_TV_BAR)
    return max(0.0, min(1.0, (tf - left) / max(tf, 1)))

def compute_tvr_features_live(df: pd.DataFrame, ind: dict):
    """يحسب TVR على الشمعة الحية (لم تغلق بعد)."""
    if not TVR_ENABLED or len(df) < 3:
        return None
    try:
        # آخر سطر = الشمعة الحيّة
        o = float(df["open"].iloc[-1]); c = float(df["close"].iloc[-1])
        v = float(df["volume"].iloc[-1]); ts = int(df["time"].iloc[-1])
        atr = float(ind.get("atr") or 0.0)
        if atr <= 0: return None

        # تأكد إن البروفايل مبني
        if state.get("_tvr_profile") is None or len(state.get("_tvr_profile") or []) != int(TVR_BUCKETS or 96):
            p = build_tvr_profile(df.iloc[:-1])  # استخدم المغلق لبناء البروفايل
            if p: state["_tvr_profile"] = p
        prof = state.get("_tvr_profile")
        if not prof: return None

        bucket = _tvr_bucket_index(ts)
        base_vol_bucket = max(float(prof[bucket]), 1e-12)

        # عدّى قد إيه من زمن الشمعة؟
        frac = _bar_elapsed_frac(df)
        if frac < TVR_LIVE_MIN_ELAPSED:   # لسه بدري – تجنب false spikes
            return None

        # نقيس الحجم مقابل المتوقع حتى الآن
        exp_vol_so_far = max(base_vol_bucket * frac, 1e-12)
        vol_ratio = v / exp_vol_so_far

        body = abs(c - o)
        reaction = body / max(atr, 1e-12)
        direction = 1 if c > o else -1

        strong = (vol_ratio >= TVR_LIVE_VOL_SPIKE) and (reaction >= TVR_LIVE_REACTION_ATR)
        return {
            "bucket": bucket,
            "vol_ratio": float(vol_ratio),
            "reaction": float(reaction),
            "direction": int(direction),
            "strong": bool(strong),
            "elapsed_frac": float(frac)
        }
    except Exception:
        return None

def compute_tvr_features(df_closed: pd.DataFrame, ind: dict):
    """
    نحسب سبايك الحجم + ردّة الفعل السعري (جسم/ATR) على آخر شمعة مُغلقة فقط.
    """
    if not TVR_ENABLED or len(df_closed) < 3:
        return None
    try:
        # آخر شمعة مُغلقة:
        o = float(df_closed["open"].iloc[-1])
        c = float(df_closed["close"].iloc[-1])
        v = float(df_closed["volume"].iloc[-1])
        ts = int(df_closed["time"].iloc[-1])
        atr = float(ind.get("atr") or 0.0)
        if atr <= 0:
            return None

        # بروفايل الحجم الزمني (نبنيه مرة ونحتفظ به في state)
        if state.get("_tvr_profile") is None or len(state.get("_tvr_profile") or []) != int(TVR_BUCKETS or 96):
            p = build_tvr_profile(df_closed)
            if p: state["_tvr_profile"] = p

        prof = state.get("_tvr_profile")
        if not prof:  # لم نستطع البناء
            return None

        bucket = _tvr_bucket_index(ts)
        base_vol = max(float(prof[bucket]), 1e-12)
        vol_ratio = v / base_vol

        body = abs(c - o)
        reaction = body / max(atr, 1e-12)
        direction = 1 if c > o else -1

        # إشارة قوية؟ (حجم مرتفع + ردّة فعل سعرية + ADX مقبول)
        adx_ok = float(ind.get("adx") or 0.0) >= TVR_MIN_ADX
        strong = (vol_ratio >= TVR_VOL_SPIKE) and (reaction >= TVR_REACTION_ATR) and adx_ok

        return {
            "bucket": bucket,
            "vol_ratio": float(vol_ratio),
            "reaction": float(reaction),
            "direction": int(direction),
            "strong": bool(strong)
        }
    except Exception:
        return None

def tvr_spike_entry(df_full: pd.DataFrame, df_closed: pd.DataFrame, ind: dict, bal: float, px: float, spread_bps: float) -> bool:
    """دخول Scout/Explosion مستقل إذا لا توجد صفقة والمشهد قوي حسب TVR."""
    if not TVR_ENABLED or state["open"]:
        return False

    # فلتر السبريد
    max_spread = TVR_LIVE_MAX_SPREAD_BPS if TVR_USE_LIVE_BAR else TVR_MAX_SPREAD_BPS
    if spread_bps is not None and spread_bps > max_spread:
        return False

    # حاول live أولاً لو مُفعّل
    if TVR_USE_LIVE_BAR:
        feats_live = compute_tvr_features_live(df_full, ind)  # استخدام df_full للشمعة الحية
        if feats_live and feats_live["strong"]:
            side = "buy" if feats_live["direction"] > 0 else "sell"
            qty_full = compute_size(bal, px)
            qty = safe_qty(qty_full * TVR_SCOUT_FRAC)
            if qty > 0:
                open_market(side, qty, px)
                # Initialize SMC snapshot on TVR entries
                if SMC_ENABLED:
                    try:
                        entry_px = px
                        current_atr = float(ind.get("atr") or 0.0)
                        smc_data = _smc_liquidity_targets(df_full, side, entry_px, current_atr)
                        set_state({"_smc": smc_data})
                        logging.info(f"SMC initialized for TVR {side} entry")
                    except Exception as e:
                        logging.error(f"SMC init in TVR error: {e}")
                # حماية فورية
                atr = float(ind.get("atr") or 0.0)
                if atr > 0:
                    gap = atr * TVR_SCOUT_TRAIL_MULT
                    if side == "buy":
                        state["trail"] = (px - gap)
                    else:
                        state["trail"] = (px + gap)
                    state["breakeven"] = state.get("breakeven") or state["entry"]
                # علّم وضع TVR
                state["tvr_active"] = True
                state["tvr_bars_alive"] = 0
                state["tvr_bucket"] = feats_live["bucket"]
                state["tvr_vol_ratio"] = feats_live["vol_ratio"]
                state["tvr_reaction"] = feats_live["reaction"]
                state["tvr_direction"] = feats_live["direction"]
                logging.info(f"TVR LIVE SCOUT ENTRY: {side} at {px}, vol_ratio={feats_live['vol_ratio']:.2f}, reaction={feats_live['reaction']:.2f}")
                return True

    # لو الحيّة مش قوية/غير مفعّلة، جرّب النسخة المُغلقة
    feats = compute_tvr_features(df_closed, ind)
    if not feats or not feats["strong"]:
        return False

    side = "buy" if feats["direction"] > 0 else "sell"
    qty_full = compute_size(bal, px)
    qty = safe_qty(qty_full * TVR_SCOUT_FRAC)
    if qty <= 0:
        return False

    open_market(side, qty, px)
    # Initialize SMC snapshot on TVR entries
    if SMC_ENABLED:
        try:
            entry_px = px
            current_atr = float(ind.get("atr") or 0.0)
            smc_data = _smc_liquidity_targets(df_full, side, entry_px, current_atr)
            set_state({"_smc": smc_data})
            logging.info(f"SMC initialized for TVR {side} entry")
        except Exception as e:
            logging.error(f"SMC init in TVR error: {e}")
    state["tvr_active"] = True
    state["tvr_bars_alive"] = 0
    state["tvr_bucket"] = feats["bucket"]
    state["tvr_vol_ratio"] = feats["vol_ratio"]
    state["tvr_reaction"] = feats["reaction"]
    state["tvr_direction"] = feats["direction"]
    return True

def tvr_post_entry_relax(df: pd.DataFrame, ind: dict):
    """
    أثناء وضع TVR: لا نُضيّق التريل مباشرة.
    ننتظر هدوء موجة الاندفاع: جسم شمعتين صغيرتين أو انتهاء مهلة.
    بعدها نُفعّل إدارة البوت العادية.
    """
    if not TVR_ENABLED or not state.get("tvr_active"):
        return
    try:
        if len(df) < 3:
            return
        # آخر شمعة حيّة (قد تكون لم تغلق تمامًا)
        o = float(df["open"].iloc[-1]); c = float(df["close"].iloc[-1])
        body = abs(c - o)
        atr = float(ind.get("atr") or 0.0)

        state["tvr_bars_alive"] = int(state.get("tvr_bars_alive", 0)) + 1

        small = (atr > 0 and (body / atr) <= 0.35)
        calm_two = False
        # فحص الشمعة المُغلقة قبلها أيضًا:
        if len(df) >= 2 and atr > 0:
            o1 = float(df["open"].iloc[-2]); c1 = float(df["close"].iloc[-2])
            calm_two = small and (abs(c1 - o1) / atr <= 0.35)

        if calm_two or state["tvr_bars_alive"] >= TVR_TIMEOUT_BARS:
            state["tvr_active"] = False  # نعود للإدارة المعتادة
    except Exception:
        pass

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

# =================== SMC / MSS (Market Structure Shift) ===================
def _find_swings(df: pd.DataFrame, left:int=2, right:int=2):
    """يرصد قمم/قيعان محلية بدون إعادة رسم."""
    if len(df) < left+right+3:
        return None, None
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    ph = [None]*len(df)
    pl = [None]*len(df)
    for i in range(left, len(df)-right):
        if all(h[i] >= h[j] for j in range(i-left, i+right+1)):
            ph[i] = h[i]
        if all(l[i] <= l[j] for j in range(i-left, i+right+1)):
            pl[i] = l[i]
    return ph, pl

def detect_mss(df: pd.DataFrame, ind: dict, left:int=2, right:int=2, buffer_bps:float=5.0):
    """يكتشف تغيير هيكل السوق على الشموع المغلقة فقط."""
    try:
        d = df.iloc[:-1] if len(df) >= 2 else df.copy()   # مغلقة فقط
        if len(d) < left+right+3:
            return {"mss": False, "dir": None, "level": None, "swing_high": None, "swing_low": None, "strength": 0.0}
        
        ph, pl = _find_swings(d, left, right)
        last_ph_i = max([i for i, v in enumerate(ph) if v is not None], default=None)
        last_pl_i = max([i for i, v in enumerate(pl) if v is not None], default=None)
        if last_ph_i is None or last_pl_i is None:
            return {"mss": False, "dir": None, "level": None, "swing_high": None, "swing_low": None, "strength": 0.0}
        
        c = float(d["close"].iloc[-1])
        sh = float(ph[last_ph_i])
        sl = float(pl[last_pl_i])
        buf_hi = sh * (1.0 + buffer_bps/10000.0)
        buf_lo = sl * (1.0 - buffer_bps/10000.0)
        strength = float(ind.get("adx") or 0.0)
        
        if c > buf_hi:
            return {"mss": True, "dir": "bull", "level": sh, "swing_high": sh, "swing_low": sl, "strength": strength}
        if c < buf_lo:
            return {"mss": True, "dir": "bear", "level": sl, "swing_high": sh, "swing_low": sl, "strength": strength}
        
        return {"mss": False, "dir": None, "level": None, "swing_high": sh, "swing_low": sl, "strength": strength}
    except Exception:
        return {"mss": False, "dir": None, "level": None, "swing_high": None, "swing_low": None, "strength": 0.0}

# =================== SMC PRO — LIQUIDITY TARGETS ===================
def _smc_liquidity_targets(df: pd.DataFrame, side_or_sig: str, entry_px: float, atr_now: float):
    """
    توليد أهداف سيولة بناءً على الهيكل السعري + امتدادات فيبوناتشي + ATR
    تُرجع:
      - score: 0..1 (قوة الهيكل/السيولة)
      - tp_candidates: قائمة أهداف % من سعر الدخول
      - trail_mult: معامل التريل المقترح
      - levels: معلومات تعليمية (OB/Equal Highs/Lows/BOS)
    """
    try:
        if len(df) < 60 or entry_px is None or atr_now <= 0:
            return {"score": 0.45, "tp_candidates": [], "trail_mult": SMC_TIGHT_TRAIL_MULT, "levels": {}}

        d = df.iloc[:-1] if len(df) >= 2 else df.copy()  # شموع مغلقة فقط
        h = d["high"].astype(float).values
        l = d["low"].astype(float).values
        c = d["close"].astype(float).values
        current_price = float(c[-1])

        # 1) البحث عن قمم وقيعان محلية (Swings)
        ph, pl = _find_swings(d, left=2, right=2)
        
        # 2) Equal Highs/Lows
        eqh_candidates = []; eql_candidates = []
        tolerance_pct = 0.05  # 0.05%
        
        for i, price in enumerate(ph):
            if price is None: continue
            tolerance = price * tolerance_pct / 100.0
            similar_highs = [ph[j] for j in range(max(0, i-10), min(len(ph), i+10)) 
                           if ph[j] is not None and abs(ph[j] - price) <= tolerance]
            if len(similar_highs) >= 2:  # على الأقل قمتين متشابهتين
                eqh_candidates.append(max(similar_highs))
                
        for i, price in enumerate(pl):
            if price is None: continue
            tolerance = price * tolerance_pct / 100.0
            similar_lows = [pl[j] for j in range(max(0, i-10), min(len(pl), i+10)) 
                          if pl[j] is not None and abs(pl[j] - price) <= tolerance]
            if len(similar_lows) >= 2:  # على الأقل قاعين متشابهين
                eql_candidates.append(min(similar_lows))
        
        eqh = max(eqh_candidates) if eqh_candidates else None
        eql = min(eql_candidates) if eql_candidates else None

        # 3) Order Blocks (OB)
        ob_candidates = []
        try:
            for i in range(len(d)-2, max(len(d)-SMC_OB_LOOKBACK, 1), -1):
                high_val = float(d["high"].iloc[i])
                low_val = float(d["low"].iloc[i])
                open_val = float(d["open"].iloc[i])
                close_val = float(d["close"].iloc[i])
                
                candle_range = high_val - low_val
                if candle_range <= 0: continue
                    
                body_size = abs(close_val - open_val)
                upper_wick = high_val - max(open_val, close_val)
                lower_wick = min(open_val, close_val) - low_val
                
                upper_wick_pct = (upper_wick / candle_range) * 100
                lower_wick_pct = (lower_wick / candle_range) * 100
                
                # شمعة اندفاع: جسم كبير + ذيول صغيرة
                if (body_size >= SMC_DISPLACEMENT_ATR * atr_now and 
                    upper_wick_pct <= SMC_WICK_MAX * 100 and 
                    lower_wick_pct <= SMC_WICK_MAX * 100):
                    
                    side = "bull" if close_val > open_val else "bear"
                    ob = {
                        "side": side,
                        "bot": min(open_val, close_val),
                        "top": max(open_val, close_val),
                        "time": int(d["time"].iloc[i])
                    }
                    ob_candidates.append(ob)
                    break  # نأخذ أقرب OB
        except Exception as e:
            logging.error(f"SMC OB detection error: {e}")

        # 4) Break of Structure (BOS) مبسط
        bos_type = "NONE"
        if eqh and current_price > eqh:
            bos_type = "BULL_BOS"
        elif eql and current_price < eql:
            bos_type = "BEAR_BOS"

        # 5) FVG (Fair Value Gap) مبسط
        fvg_candidates = []
        try:
            for i in range(len(d)-3, max(len(d)-20, 2), -1):
                prev_high = float(d["high"].iloc[i-1])
                prev_low = float(d["low"].iloc[i-1])
                current_low = float(d["low"].iloc[i])
                current_high = float(d["high"].iloc[i])
                
                # FVG صاعد: قاع الشمعة الحالية > قمة الشمعة السابقة
                if current_low > prev_high and (current_low - prev_high) <= SMC_FVG_MAX_GAP_ATR * atr_now:
                    fvg_candidates.append({
                        "type": "BULL_FVG",
                        "bottom": prev_high,
                        "top": current_low
                    })
                
                # FVG هابط: قمة الشمعة الحالية < قاع الشمعة السابقة  
                elif current_high < prev_low and (prev_low - current_high) <= SMC_FVG_MAX_GAP_ATR * atr_now:
                    fvg_candidates.append({
                        "type": "BEAR_FVG", 
                        "bottom": current_high,
                        "top": prev_low
                    })
        except Exception as e:
            logging.error(f"SMC FVG detection error: {e}")

        # 6) تحديد الجانب والاتجاه
        side = "long" if side_or_sig in ("long", "buy") else "short"
        
        # 7) توليد أهداف الربح بناء على الهيكل
        tp_candidates = []
        base_price = entry_px
        
        if side == "long":
            # أهداف صعودية: استخدام القمم، OB، فيبوناتشي
            structure_targets = []
            
            if eqh: structure_targets.append(eqh)
            if ob_candidates:
                for ob in ob_candidates:
                    if ob["side"] == "bear":  # OB هابط قد يكون مقاومة
                        structure_targets.append(ob["bot"])
            
            # إضافة FVG كأهداف
            for fvg in fvg_candidates:
                if fvg["type"] == "BULL_FVG":
                    structure_targets.append(fvg["top"])
            
            # أهداف فيبوناتشي من القاع إلى القمة
            if eql and structure_targets:
                for target in structure_targets:
                    for fib_ext in SMC_FIB_EXTS:
                        fib_target = eql + (target - eql) * fib_ext
                        profit_pct = ((fib_target - entry_px) / entry_px) * 100
                        if 0.1 <= profit_pct <= 10.0:  # فلترة واقعية
                            tp_candidates.append(round(profit_pct, 2))
            
            # أهداف ATR-based كبديل
            for atr_mult in [1.5, 2.5, 4.0]:
                atr_target = entry_px + (atr_now * atr_mult)
                profit_pct = ((atr_target - entry_px) / entry_px) * 100
                if 0.1 <= profit_pct <= 8.0:
                    tp_candidates.append(round(profit_pct, 2))
                    
        else:  # short
            # أهداف هبوطية: استخدام القيعان، OB، فيبوناتشي
            structure_targets = []
            
            if eql: structure_targets.append(eql)
            if ob_candidates:
                for ob in ob_candidates:
                    if ob["side"] == "bull":  # OB صاعد قد يكون دعماً
                        structure_targets.append(ob["top"])
            
            # إضافة FVG كأهداف
            for fvg in fvg_candidates:
                if fvg["type"] == "BEAR_FVG":
                    structure_targets.append(fvg["bottom"])
            
            # أهداف فيبوناتشي من القمة إلى القاع
            if eqh and structure_targets:
                for target in structure_targets:
                    for fib_ext in SMC_FIB_EXTS:
                        fib_target = eqh - (eqh - target) * fib_ext
                        profit_pct = ((entry_px - fib_target) / entry_px) * 100
                        if 0.1 <= profit_pct <= 10.0:
                            tp_candidates.append(round(profit_pct, 2))
            
            # أهداف ATR-based كبديل
            for atr_mult in [1.5, 2.5, 4.0]:
                atr_target = entry_px - (atr_now * atr_mult)
                profit_pct = ((entry_px - atr_target) / entry_px) * 100
                if 0.1 <= profit_pct <= 8.0:
                    tp_candidates.append(round(profit_pct, 2))

        # إزالة التكرارات والفرز
        tp_candidates = sorted(set([t for t in tp_candidates if 0.15 <= t <= 8.0]))
        tp_candidates = tp_candidates[:4]  # أقصى 4 أهداف

        # 8) حساب قوة الهيكل (Score)
        score = 0.35  # درجة أساسية
        
        # نقاط للهيكل المتوافق مع الاتجاه
        if side == "long":
            if bos_type == "BULL_BOS": score += 0.25
            if eqh and current_price > eqh: score += 0.15
            if any(ob["side"] == "bull" for ob in ob_candidates): score += 0.15
        else:  # short
            if bos_type == "BEAR_BOS": score += 0.25  
            if eql and current_price < eql: score += 0.15
            if any(ob["side"] == "bear" for ob in ob_candidates): score += 0.15
        
        # نقاط للحجم والتقلب
        if len(d) >= 21:
            current_volume = float(d["volume"].iloc[-1])
            volume_ma = d["volume"].iloc[-21:-1].astype(float).mean()
            if volume_ma > 0 and current_volume > volume_ma * SMC_VOL_SPIKE:
                score += 0.10
        
        score = min(1.0, max(0.0, score))

        # 9) تحديد معامل التريل
        if score >= SMC_SCORE_STRONG:
            trail_mult = SMC_WIDE_TRAIL_MULT
        elif score >= SMC_SCORE_HOLD:
            trail_mult = (SMC_WIDE_TRAIL_MULT + SMC_TIGHT_TRAIL_MULT) / 2
        else:
            trail_mult = SMC_TIGHT_TRAIL_MULT

        # 10) تجميع معلومات المستويات
        levels = {
            "ob": ob_candidates[0] if ob_candidates else None,
            "eqh": eqh,
            "eql": eql, 
            "bos": {"type": bos_type},
            "fvg": fvg_candidates[0] if fvg_candidates else None
        }

        return {
            "score": float(score),
            "tp_candidates": tp_candidates,
            "trail_mult": float(trail_mult),
            "levels": levels
        }

    except Exception as e:
        logging.error(f"SMC liquidity targets error: {e}")
        return {"score": 0.45, "tp_candidates": [], "trail_mult": SMC_TIGHT_TRAIL_MULT, "levels": {}}

# =================== SMC MSS MANAGER ===================
def smc_mss_manager(ind: dict, mss: dict):
    """يساند الإدارة بعد الدخول بناءً على MSS بدون أي تغيير في الدخول."""
    if not (SMC_MSS_ENABLED and state["open"] and mss):
        return None
    if not mss.get("mss"):
        return None

    side = state["side"]
    aligned = (mss["dir"]=="bull" and side=="long") or (mss["dir"]=="bear" and side=="short")
    adx = float(ind.get("adx") or 0.0)
    px = ind.get("price") or price_now() or state["entry"]
    atr = float(ind.get("atr") or 0.0)
    actions = []

    # MSS مع الاتجاه ⇒ اتركها تجري (نُرخي التريل ونؤجل TP مرة واحدة)
    if aligned and adx >= SMC_STRONG_ADX:
        cur = state.get("_adaptive_trail_mult") or ATR_MULT_TRAIL
        state["_adaptive_trail_mult"] = max(cur, cur + SMC_BOOST_TRAIL_MULT)
        if SMC_DEFER_TP_ON_ALIGNED:
            state["_smc_defer_tp_flag"] = True   # تُستهلك مرة واحدة
        actions.append("MSS_ALIGNED_BOOST")

    # MSS عكسي قوي ⇒ حماية: جزئي + تعادل + تريل مشدود
    if (not aligned) and adx >= SMC_STRONG_ADX:
        already = float(state.get("smc_extra_partials", 0.0))
        can = max(0.0, SMC_MAX_EXTRA_PARTIAL - already)
        frac = min(SMC_PARTIAL_ON_OPPOSITE, can)
        if frac > 0 and state["qty"] > 0:
            close_partial(frac, "MSS opposite (SMC guard)")
            state["smc_extra_partials"] = already + frac
        state["breakeven"] = state.get("breakeven") or state["entry"]
        if atr > 0 and px is not None:
            gap = atr * max(1.2, (state.get("_adaptive_trail_mult") or ATR_MULT_TRAIL) * 0.8)
            if side == "long":
                state["trail"] = max(state.get("trail") or (px - gap), px - gap)
            else:
                state["trail"] = min(state.get("trail") or (px + gap), px + gap)
        actions.append("MSS_OPPOSITE_DEFEND")

    return "+".join(actions) if actions else None

def defend_on_opposite_rf(ind: dict, info: dict):
    """عند إشارة RF عكسية: لا قفل كامل. جزئي + تعادل + تريل مشدود + تصويت للخروج."""
    if not state["open"] or state["qty"] <= 0:
        return False

    side = state["side"]
    px   = info.get("price") or price_now() or state["entry"]
    atr  = float(ind.get("atr") or 0.0)
    adx  = float(ind.get("adx") or 0.0)

    # 1) جني جزئي صغير (أكبر لو TP1 لسه محصلش)
    base_frac = OPP_RF_DEFEND_PARTIAL
    if not state.get("tp1_done", False):
        base_frac = min(0.30, max(0.15, base_frac))  # حماية أكبر قبل TP1
    close_partial(base_frac, "Opposite RF — defensive partial")

    # 2) تثبيت التعادل
    state["breakeven"] = state.get("breakeven") or state["entry"]

    # 3) تريل مشدود بقدر الإمكان
    if atr > 0 and px is not None:
        gap = atr * max(state.get("_adaptive_trail_mult") or ATR_MULT_TRAIL, OPP_RF_TIGHT_TRAIL_MULT)
        if side == "long":
            state["trail"] = max(state.get("trail") or (px - gap), px - gap)
        else:
            state["trail"] = min(state.get("trail") or (px + gap), px + gap)

    # 4) عدّ تصويتات العكسية (لا قفل كامل إلا بعد تأكيد × مرتين)
    state["_opp_rf_votes"] = int(state.get("_opp_rf_votes", 0)) + 1
    return True

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
    set_state({
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

# (REPLACED) close_partial with exchange-aware min qty and atomic updates
def close_partial(frac, reason):
    """
    إغلاق جزئي آمن، مع احترام حد الكمية الدنيا وفق مواصفات السوق،
    ومنع كسر حارس البقايا، وتحديث state تحت القفل عند الحاجة.
    """
    global state, compound_pnl
    if not state["open"]:
        return

    qty_close = safe_qty(max(0.0, state["qty"] * min(max(frac, 0.0), 1.0)))

    # 🔒 حارس البقايا: لا ندع الكمية المتبقية تهبط تحت الحد الأدنى
    px = _safe_price(state["entry"]) or state["entry"]
    min_qty_guard = max(RESIDUAL_MIN_QTY, (RESIDUAL_MIN_USDT/(px or 1e-9)))

    if state["qty"] - qty_close < min_qty_guard:
        qty_close = safe_qty(max(0.0, state["qty"] - min_qty_guard))
        if qty_close <= 0:
            print(colored("⏸️ skip partial (residual guard would be broken)", "yellow"))
            return

    # حد الكمية الدنيا القابلة للتداول (بدل شرط ثابت)
    if not _close_partial_min_check(qty_close):
        return

    side = "sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_position_params_for_close())
        except Exception as e: print(colored(f"❌ partial close: {e}","red")); logging.error(f"close_partial error: {e}"); return
    px_now = _safe_price(state["entry"]) or state["entry"]
    pnl=(px_now-state["entry"])*qty_close*(1 if state["side"]=="long" else -1)
    compound_pnl += pnl

    # تحديثات state الذرّية
    def _after_partial(s):
        s["qty"] = safe_qty((s.get("qty") or 0.0) - qty_close)
        s["scale_outs"] = int(s.get("scale_outs") or 0) + 1
        s["last_action"] = "SCALE_OUT"
        s["action_reason"] = reason
    update_state(_after_partial)

    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem_qty={fmt(state['qty'],4)}","magenta"))
    logging.info(f"PARTIAL_CLOSE {reason} qty={qty_close} pnl={pnl} remaining={state['qty']}")

    # إن أصبحت الكمية المتبقية دون الحد الأدنى، أغلِق بالكامل إذا سُمح بذلك
    if state["qty"] < min_qty_guard:
        if RESPECT_PATIENT_MODE_FOR_DUST and PATIENT_TRADER_MODE and (not state.get("tp1_done", False)):
            print(colored("⏸️ residual < guard but patient mode blocks full close before TP1", "yellow"))
        else:
            close_market_strict("auto_full_close_small_qty_guard")
        return
    save_state()

def reset_after_full_close(reason, prev_side=None):
    global state, post_close_cooldown, wait_for_next_signal_side, last_close_signal_time
    print(colored(f"🔚 CLOSE {reason} totalCompounded now={fmt(compound_pnl)}","magenta"))
    logging.info(f"FULL_CLOSE {reason} total_compounded={compound_pnl}")
    if prev_side is None: prev_side = state.get("side")
    set_state({
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
        "opened_at": None,
        # TVR
        "tvr_active": False,
        "tvr_bars_alive": 0,
        "tvr_vol_ratio": None,
        "tvr_reaction": None,
        "tvr_bucket": None,
        "tvr_direction": None,
        # SMC
        "smc_mss": None,
        "smc_extra_partials": 0.0,
        # Trap Guard
        "_trap_active": False,
        "_trap_dir": None,
        "_trap_left": 0,
        "_last_trap_ts": None,
        # Opposite RF votes
        "_opp_rf_votes": 0,
        "_trend_exit_votes": 0,
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
        # ⚠️ إزالة الإغلاق الكامل واستبداله بتثبيت التعادل فقط
        if ((side=="long" and (ind.get("ema9_slope",0.0) <= 0)) or
            (side=="short" and (ind.get("ema9_slope",0.0) >= 0))) and (ind.get("adx",0.0) < 20):
            # في وضع الصبر: لا إغلاق، اكتفِ بتثبيت التعادل
            state["breakeven"] = state.get("breakeven") or state["entry"]
            logging.info("EMA20 break + slope flip (weak ADX) → Breakeven set (no full close in patient mode)")
    
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
    
    # إذا MSS aligned طلب تأجيل TP مرة واحدة
    if state.pop("_smc_defer_tp_flag", False) and k < len(targets)-1:
        return None
        
    if k < len(targets) and rr >= targets[k]:
        if cscore >= STRONG_HOLD_SCORE and k < max(DEFER_TP_UNTIL_IDX, len(targets)-1):
            return None  # تأجيل TP للترند القوي
        close_partial(fracs[k], f"TP{k+1}@{targets[k]:.2f}% (dyn)")
        state["profit_targets_achieved"] = k + 1
        return f"TREND_TP{k+1}"
    if float(ind.get("adx") or 0.0) >= MIN_TREND_HOLD_ADX:
        return None
    if state.get("profit_targets_achieved", 0) >= len(targets) or trend_end_confirmed(ind, detect_candle_pattern(df_cached), info):
        # شدّ التريل وخد تصويت للخروج بدل القفل الفوري
        state["_trend_exit_votes"] = int(state.get("_trend_exit_votes", 0)) + 1

        atr_now = float(ind.get("atr") or 0.0)
        px_now  = ind.get("price") or price_now() or state["entry"]
        tmult   = max(state.get("_adaptive_trail_mult") or ATR_MULT_TRAIL, 1.6)

        if atr_now > 0 and px_now is not None:
            gap = atr_now * tmult
            if state["side"] == "long":
                state["trail"] = max(state.get("trail") or (px_now - gap), px_now - gap)
            else:
                state["trail"] = min(state.get("trail") or (px_now + gap), px_now + gap)

        if state["_trend_exit_votes"] >= 2 and state.get("bars", 0) >= PATIENT_HOLD_BARS:
            close_market_strict("TREND_COMPLETE_CONFIRMED")
            return "TREND_COMPLETE_CONFIRMED"
        return "TREND_EXIT_VOTE"
    return None

def smart_post_entry_manager(df: pd.DataFrame, ind: dict, info: dict):
    if not state["open"] or state["qty"]<=0: return None
    now_ts = info.get("time") or time.time()
    state.setdefault("opened_at", now_ts)
    hold = compute_tci_and_chop(df, ind, state.get("side"))
    state["_hold_trend"]=bool(hold["hold_mode"])
    state["tci"]=hold["tci"]; state["chop01"]=hold["chop01"]
    if hold["strong_hold"]:
        set_state({"_adaptive_trail_mult": max(state.get("_adaptive_trail_mult") or ATR_MULT_TRAIL, TRAIL_MULT_STRONG_ALPHA)})
    if state.get("trade_mode") is None: state["trade_mode"]="TREND"; state["profit_targets_achieved"]=0
    # سلّم ديناميكي كل دورة
    tp_list, tp_fracs, atr_pct, cscore = _build_tp_ladder(info, ind, state.get("side"))
    state["_tp_ladder"]=tp_list; state["_tp_fracs"]=tp_fracs
    state["_consensus_score"]=cscore; state["_atr_pct"]=atr_pct

    # --- SMC Pro Integration ---
    if SMC_ENABLED and state["open"] and state["qty"] > 0:
        try:
            entry_px = state.get("entry") or info.get("price")
            current_atr = float(ind.get("atr") or 0.0)
            smc_data = _smc_liquidity_targets(df, state["side"], entry_px, current_atr)
            
            if smc_data:
                state["_smc"] = smc_data
                
                # SMC قوي: توسيع الأهداف وتهدئة التريل
                if smc_data["score"] >= SMC_SCORE_HOLD:
                    if smc_data.get("tp_candidates"):
                        # دمج أهداف SMC مع الأهداف الديناميكية
                        dynamic_tps = state.get("_tp_ladder") or []
                        all_tps = sorted(set(dynamic_tps + smc_data["tp_candidates"]))
                        state["_tp_ladder"] = all_tps[:4]  # احتفظ بأقرب 4 أهداف
                    
                    # تحسين التريل
                    state["_adaptive_trail_mult"] = max(
                        state.get("_adaptive_trail_mult") or 0.0, 
                        smc_data["trail_mult"]
                    )
                    logging.info(f"SMC STRONG: score={smc_data['score']}, expanded targets, wider trail")
                
                # SMC تحذير: جني جزئي وشد التريل
                elif smc_data["score"] < SMC_SCORE_WARN and not state.get("tp1_done"):
                    close_partial(SMC_WARN_PARTIAL, "SMC weak signal — protective harvest")
                    state["_adaptive_trail_mult"] = max(
                        state.get("_adaptive_trail_mult") or 0.0, 
                        SMC_TIGHT_TRAIL_MULT
                    )
                    logging.info(f"SMC WARN: score={smc_data['score']}, took protective partial")
                    
        except Exception as e:
            logging.error(f"SMC integration error: {e}")

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
    
    # ✅ أثناء trap: امنع أي full close (إلا طوارئ/Trail)
    if state.get("_trap_active"):
        state["_exit_soft_block"] = True
        logging.info("TRAP GUARD: blocking full close in smart_exit_check")
    
    now_ts = info.get("time") or time.time()
    opened_at = state.get("opened_at") or now_ts
    elapsed_s = max(0, now_ts - opened_at)
    elapsed_bar = int(state.get("bars", 0))
    
    # ✅ تطبيق حدود الصبر
    if PATIENT_TRADER_MODE:
        MIN_HOLD_BARS = PATIENT_HOLD_BARS
        MIN_HOLD_SECONDS = PATIENT_HOLD_SECONDS

    def _allow_full_close(reason: str) -> bool:
        min_hold_reached = (elapsed_bar >= PATIENT_HOLD_BARS) or (elapsed_s >= PATIENT_HOLD_SECONDS)
        HARD_ANYTIME = ("EMERGENCY", "TRAIL", "TRAIL_ATR", "CHANDELIER", "STRICT")
        if any(tag in reason for tag in HARD_ANYTIME):
            return True                     # طوارئ/تريل/شانديلير/RF عكسي مسموح في أي وقت
        return min_hold_reached             # غير ذلك: لازم نكون عدّينا الحد الأدنى

    def _safe_full_close(reason):
        # ✅ منع القفل الكامل قبل TP1 في وضع الصبر
        if NEVER_FULL_CLOSE_BEFORE_TP1 and not state.get("tp1_done"):
            # قبل TP1: ممنوع Full Close إلا طوارئ/Trail/Strict
            if not any(tag in reason for tag in ("EMERGENCY", "TRAIL", "TRAIL_ATR", "CHANDELIER", "STRICT")):
                logging.info(f"PATIENT: block full close ({reason}) before TP1")
                return False
        
        # --- SMC Structural Patience Guard ---
        if (SMC_ENABLED and state["open"] and not state.get("tp1_done")):
            smc_data = state.get("_smc")
            if smc_data and smc_data.get("tp_candidates"):
                # منع الإغلاق الكامل إذا لا توجد طوارئ ولم نلمس أهداف السيولة بعد
                hard_exit_reasons = ("EMERGENCY", "OPPOSITE_RF", "TRAIL", "TRAIL_ATR", "CHANDELIER", "STRICT")
                if not any(exit_reason in str(reason) for exit_reason in hard_exit_reasons):
                    logging.info("SMC PATIENCE: Blocking full close - waiting for liquidity targets")
                    return False  # منع الإغلاق الكامل
        
        close_market_strict(reason)
        return True

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
                else: vote_details["volume"]=f"Normal"
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
        set_state({
            "breakout_active": True,
            "breakout_direction": "bull",
            "breakout_entry_price": price,
            "opened_by_breakout": True,
            "breakout_score": breakout_score,
            "breakout_votes_detail": vote_details
        })
    elif breakout_signal == "BEAR_BREAKOUT":
        open_market("sell", qty, price)
        set_state({
            "breakout_active": True, 
            "breakout_direction": "bear",
            "breakout_entry_price": price,
            "opened_by_breakout": True,
            "breakout_score": breakout_score,
            "breakout_votes_detail": vote_details
        })
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
    
    # إضافة بيانات TVR للعرض
    if TVR_ENABLED:
        tvr_line = f"TVR: bucket={state.get('tvr_bucket')}  vol×={fmt(state.get('tvr_vol_ratio'),2)}  react(ATR)={fmt(state.get('tvr_reaction'),2)}  active={bool(state.get('tvr_active'))}"
        print(colored(f"   🕒 {tvr_line}", "yellow"))

    # إضافة بيانات MSS للعرض
    if state.get("smc_mss"):
        m = state["smc_mss"]
        mss_status = "🟢 مع الاتجاه" if ((m.get("dir")=="bull" and state.get("side")=="long") or (m.get("dir")=="bear" and state.get("side")=="short")) else "🔴 عكسي" if m.get("mss") else "⚪ لا شيء"
        print(colored(f"   🧠 MSS: {mss_status} • اتجاه={m.get('dir')} • مستوى={fmt(m.get('level'))} • قوة={fmt(m.get('strength'),1)}", "yellow"))

    # عرض SMC Pro data
    if state.get("_smc"):
        smc = state["_smc"]
        score_color = "green" if smc["score"] >= SMC_SCORE_HOLD else "yellow" if smc["score"] >= SMC_SCORE_WARN else "red"
        score_text = colored(f"{smc['score']:.2f}", score_color)
        print(colored(f"   🧠 SMC Pro: score={score_text} • targets={smc.get('tp_candidates', [])} • trail×{smc.get('trail_mult', 0)}", "cyan"))
        
        # عرض تفاصيل إضافية في وضع verbose
        if smc.get("levels", {}).get("ob"):
            ob = smc["levels"]["ob"]
            print(colored(f"   📦 OB: {ob['side']} zone[{fmt(ob['bot'])}-{fmt(ob['top'])}]", "white"))
        if smc.get("levels", {}).get("bos"):
            print(colored(f"   🏗️ BOS: {smc['levels']['bos']['type']}", "white"))
    
    if state.get("tci") is not None:
        hold_msg = "الترند قوي — امسك الصفقة" if state.get("_hold_trend") else "إدارة عادية"
        print(colored(f"   🧭 TCI={state['tci']:.0f}/100 • Chop01={state.get('chop01',0):.2f} → {hold_msg}", "cyan" if state.get("_hold_trend") else "blue"))
    ach = state.get("profit_targets_achieved", 0); lad = state.get("_tp_ladder") or []
    if ach < len(lad):
        nxt = lad[ach]
        print(colored(f"   🎯 Dynamic TP: next={nxt:.2f}% • ATR%≈{state.get('_atr_pct',0):.2f} • Consensus={state.get('_consensus_score',0):.1f}/5", "magenta"))
    
    # 🔥 NEW: عرض بيانات Trap Guard للعرض
    if state.get("_trap_active"):
        trap_line = f"TRAP: {state.get('_trap_dir')} • bars_left={state.get('_trap_left', 0)}"
        print(colored(f"   🚨 {trap_line}", "red"))
        
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
    global state, post_close_cooldown, wait_for_next_signal_side, last_close_signal_time, last_open_fingerprint, last_signal_id
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

            # تحديث قيم EMA في state للميتريكس
            set_state({
                "ema9": ind.get("ema9"),
                "ema20": ind.get("ema20"), 
                "ema9_slope": ind.get("ema9_slope")
            })

            # SMC / MSS snapshot
            mss_info = detect_mss(df, ind, buffer_bps=SMC_BUFFER_BPS) if SMC_MSS_ENABLED else {"mss": False}
            state["smc_mss"] = mss_info

            # 🔥 NEW: Trap detection من مستويات SMC
            levels = (state.get("_smc") or {}).get("levels", {}) if SMC_ENABLED else {}
            trap = detect_stop_hunt(df, ind, levels)
            if trap and state.get("open"):
                apply_trap_guard(trap, {**ind, "price": px or info_closed["price"]})

            # 🔥 NEW: إطفاء الفخ بعد انتهاء مدته
            if state.get("_trap_active") and new_bar:
                left = int(state.get("_trap_left", 0)) - 1
                state["_trap_left"] = left
                if left <= 0:
                    state["_trap_active"] = False
                    state["_trap_dir"] = None
                    logging.info("TRAP GUARD DEACTIVATED - time expired")

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

            # TVR Scout/Explosion entry (مستقل عند عدم وجود صفقة)
            if not state["open"]:
                entered_tvr = tvr_spike_entry(df, df_closed, ind, bal, px or info_closed["price"], spread_bps)
                if entered_tvr:
                    snapshot(bal, {**info_closed, "price": px or info_closed["price"]}, ind, spread_bps, "TVR SCOUT ENTRY", df)
                    time.sleep(compute_next_sleep(df))
                    continue

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

            # SMC MSS Management
            mss_action = smc_mss_manager({**ind, "price": px or info_closed["price"]}, mss_info)
            if mss_action:
                logging.info(f"SMC_MSS_ACTION: {mss_action}")

            # TVR post-entry relax
            tvr_post_entry_relax(df, ind)

            # ENTRY on closed-candle RF signal only
            sig = "buy" if info_closed["long"] else ("sell" if info_closed["short"] else None)
            
            # Reset opposite-RF votes عندما لا توجد عكسية أو ظهرت إشارة مع اتجاه الصفقة
            if state.get("open"):
                if sig is None or \
                   (state["side"] == "long" and info_closed["long"]) or \
                   (state["side"] == "short" and info_closed["short"]):
                    set_state({"_opp_rf_votes": 0})
            
            reason=None
            if not sig:
                reason="no signal"
            elif spread_bps is not None and spread_bps > 6:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > 6)"
            elif post_close_cooldown > 0:
                reason=f"cooldown {post_close_cooldown} bars"

            # === لا قفل فوري على RF عكسي؛ نفّذ دفاع فقط ===
            if state["open"] and sig and (reason is None):
                desired = "long" if sig == "buy" else "short"
                
                # عرّف المتغيرات قبل أي استخدام
                px_now = px or info_closed.get("price")
                rf_now = info_closed.get("filter")
                
                if state["side"] != desired:
                    # ✅ FIRST: تحقق من Trap Guard - لو شغّال: لا قفل كامل إطلاقًا
                    if state.get("_trap_active"):
                        defend_on_opposite_rf(ind, {**info_closed, "price": px_now})
                        reason = "trap guard active — ignore full close"
                        snapshot(bal, {**info_closed, "price": px_now}, ind, spread_bps, reason, df)
                        time.sleep(compute_next_sleep(df)); 
                        continue
                    
                    # تأكيد العكسية: ADX + كسر واضح عن الفلتر + تكرار
                    adx_now = float(ind.get("adx") or 0.0)

                    bps = 0.0
                    try:
                        if px_now and rf_now:
                            bps = abs((px_now - rf_now) / rf_now) * 10000.0
                    except Exception:
                        pass

                    # نفّذ دفاع دائمًا أول مرة
                    defend_on_opposite_rf(ind, {**info_closed, "price": px_now})

                    # لو تراكمت الأصوات وكمان الشروط اتحققت ومع ذلك عديّنا حد الصبر → قفل كامل
                    min_hold_ok = (state.get("bars", 0) >= PATIENT_HOLD_BARS) or ((time.time() - (state.get("opened_at") or time.time())) >= PATIENT_HOLD_SECONDS)
                    votes_ok    = int(state.get("_opp_rf_votes", 0)) >= OPP_RF_NEED_BARS
                    confirmed   = (adx_now >= OPP_RF_MIN_ADX) and (bps >= OPP_RF_MIN_HYST_BPS)

                    if votes_ok and confirmed and min_hold_ok and (not NEVER_FULL_CLOSE_BEFORE_TP1 or state.get("tp1_done")):
                        close_market_strict("OPPOSITE_RF_CONFIRMED")
                        wait_for_next_signal_side = "sell" if state.get("side") == "long" else "buy"
                        last_close_signal_time = info_closed["time"]
                        set_state({"_opp_rf_votes": 0})  # Reset votes after confirmed close
                        snapshot(bal, {**info_closed, "price": px_now}, ind, spread_bps, "opposite RF confirmed (full close)", df)
                        time.sleep(compute_next_sleep(df)); continue
                    else:
                        reason = f"opp RF defend (votes={int(state.get('_opp_rf_votes',0))}/{OPP_RF_NEED_BARS}, ADX={adx_now:.1f}, Δ={bps:.1f}bps)"

            # Open when flat
            if not state["open"] and (reason is None) and sig:
                if wait_for_next_signal_side:
                    if sig != wait_for_next_signal_side:
                        reason=f"waiting opposite RF signal: need {wait_for_next_signal_side}"
                    else:
                        qty = compute_size(bal, px or info_closed["price"])
                        if qty>0:
                            open_market(sig, qty, px or info_closed["price"])
                            
                            # 🔥 تهيئة SMC Pro فور فتح الصفقة
                            if SMC_ENABLED:
                                try:
                                    entry_px = px or info_closed["price"]
                                    current_atr = float(ind.get("atr") or 0.0)
                                    smc_data = _smc_liquidity_targets(df, sig, entry_px, current_atr)
                                    state["_smc"] = smc_data
                                    logging.info(f"SMC initialized for new {sig} position")
                                except Exception as e:
                                    logging.error(f"SMC init error: {e}")
                            
                            wait_for_next_signal_side=None; last_close_signal_time=None; last_open_fingerprint=None
                            last_signal_id=f"{info_closed['time']}:{sig}"
                        else:
                            reason="qty<=0"
                else:
                    qty = compute_size(bal, px or info_closed["price"])
                    if qty>0:
                        open_market(sig, qty, px or info_closed["price"])
                        
                        # 🔥 تهيئة SMC Pro فور فتح الصفقة
                        if SMC_ENABLED:
                            try:
                                entry_px = px or info_closed["price"]
                                current_atr = float(ind.get("atr") or 0.0)
                                smc_data = _smc_liquidity_targets(df, sig, entry_px, current_atr)
                                state["_smc"] = smc_data
                                logging.info(f"SMC initialized for new {sig} position")
                            except Exception as e:
                                logging.error(f"SMC init error: {e}")
                        
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
    return f"✅ RF Bot — {SYMBOL} {INTERVAL} — {mode} — TREND-ONLY — CLOSED-CANDLE RF — SMART HARVESTING — BREAKOUT ENGINE — EMERGENCY LAYER — STRICT CLOSE — TVR ENHANCED — SMC PRO"

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
        },
        "tvr_enhanced": {
            "enabled": TVR_ENABLED,
            "active": state.get("tvr_active", False),
            "vol_ratio": state.get("tvr_vol_ratio"),
            "reaction": state.get("tvr_reaction"),
            "bucket": state.get("tvr_bucket")
        },
        "smc_pro": {
            "enabled": SMC_ENABLED,
            "score": state.get("_smc", {}).get("score"),
            "tp_candidates": state.get("_smc", {}).get("tp_candidates"),
            "trail_mult": state.get("_smc", {}).get("trail_mult")
        },
        "smc_mss": state.get("smc_mss"),
        "smc_extra_partials": state.get("smc_extra_partials", 0.0),
        "trap_guard": {
            "enabled": TRAP_ENABLED,
            "active": state.get("_trap_active", False),
            "direction": state.get("_trap_dir"),
            "bars_left": state.get("_trap_left", 0)
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
        "ema_indicators": {"ema9": state.get("ema9"), "ema20": state.get("ema20"), "ema9_slope": state.get("ema9_slope")},
        "tvr_enabled": TVR_ENABLED,
        "tvr_active": state.get("tvr_active", False),
        "smc_enabled": SMC_ENABLED,
        "smc_score": state.get("_smc", {}).get("score"),
        "smc_mss": state.get("smc_mss"),
        "smc_extra_partials": state.get("smc_extra_partials", 0.0),
        "trap_guard_active": state.get("_trap_active", False)
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
    print(colored(f"TVR ENHANCED: {'ON' if TVR_ENABLED else 'OFF'} • Buckets={TVR_BUCKETS} • Vol_Spike={TVR_VOL_SPIKE}x • Reaction_ATR={TVR_REACTION_ATR}", "yellow"))
    print(colored(f"SMC PRO: {'ON' if SMC_ENABLED else 'OFF'} • EQHL_LB={SMC_EQHL_LOOKBACK} • OB_LB={SMC_OB_LOOKBACK} • FVG_Gap={SMC_FVG_MAX_GAP_ATR}ATR", "yellow"))
    print(colored(f"SMC MSS: {'ON' if SMC_MSS_ENABLED else 'OFF'} • Strong_ADX={SMC_STRONG_ADX} • Buffer={SMC_BUFFER_BPS}bps • Partial_Opposite={SMC_PARTIAL_ON_OPPOSITE*100}%", "yellow"))
    print(colored(f"TRAP GUARD: {'ON' if TRAP_ENABLED else 'OFF'} • Wick_Pct={TRAP_WICK_PCT}% • Vol_Spike={TRAP_VOL_SPIKE}x • Hold_Bars={TRAP_HOLD_BARS}", "yellow"))
    load_state()
    print(colored("🛡️ Watchdog started", "cyan"))
    threading.Thread(target=watchdog_check, daemon=True).start()
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
