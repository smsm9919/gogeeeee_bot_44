# -*- coding: utf-8 -*-
"""
RF Futures Bot — Smart Pro (BingX Perp, CCXT) - HARDENED EDITION
[الكود الأصلي بالكامل يبقى كما هو...]

✅ NEW: BREAKOUT ENGINE ADDED - Independent explosive move detection
- Monitors ATR spikes (1.8x) + ADX strength (≥25) + price breakout (20-bar highs/lows)
- Enters immediately on detected explosions/crashes
- Closes strictly when volatility normalizes (ATR < 1.1x previous)
- Returns control to normal RF strategy after breakout ends
- Fully independent - doesn't modify existing functions
"""

import os, time, math, threading, requests, traceback, random, signal, sys, logging
from logging.handlers import RotatingFileHandler
import pandas as pd
import ccxt
from flask import Flask, jsonify
from datetime import datetime

# ------------ [جميع الإعدادات الأصلية تبقى كما هي] ------------
API_KEY = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE = bool(API_KEY and API_SECRET)

SYMBOL = "DOGE/USDT:USDT"
INTERVAL = "15m"
LEVERAGE = 10
RISK_ALLOC = 0.60
# ... [كل الإعدادات الأصلية]

# ✅ NEW: BREAKOUT ENGINE SETTINGS
BREAKOUT_ATR_SPIKE = 1.8        # ATR(current) > ATR(previous) * 1.8
BREAKOUT_ADX_THRESHOLD = 25     # ADX ≥ 25 for trend strength  
BREAKOUT_LOOKBACK_BARS = 20     # Check last 20 bars for highs/lows
BREAKOUT_CALM_THRESHOLD = 1.1   # ATR(current) < ATR(previous) * 1.1 → exit

# ------------ [جميع الدوال المساعدة الأصلية تبقى كما هي] ------------
def setup_file_logging():
    """Setup rotating file logging (5MB × 7 files)"""
    # ... [الكود الأصلي]

def _graceful_exit(signum, frame):
    """Save state and exit gracefully on SIGTERM/SIGINT"""
    # ... [الكود الأصلي]

# ... [جميع الدوال الأصلية تبقى كما هي]

# ------------ STATE & SYNC (معدّل بإضافة breakout_active) ------------
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
    # ✅ NEW: Breakout Engine state
    "breakout_active": False,      # هل نحن في وضع انفجار نشط؟
    "breakout_direction": None,    # 'bull' أو 'bear'
    "breakout_entry_price": None   # سعر الدخول أثناء الانفجار
}
compound_pnl = 0.0
last_signal_id = None
post_close_cooldown = 0

# ------------ NEW: BREAKOUT ENGINE FUNCTIONS ------------
def detect_breakout(df: pd.DataFrame, ind: dict) -> str:
    """
    ⚡ BREAKOUT ENGINE: كشف الانفجارات والانهيارات السريعة
    Returns:
        "BULL_BREAKOUT" - انفجار صعودي
        "BEAR_BREAKOUT" - انهيار هبوطي  
        None - لا انفجار
    """
    try:
        if len(df) < BREAKOUT_LOOKBACK_BARS + 2:
            return None
            
        # ✅ PATCH: استخدام الشمعة المغلقة الحالية دائماً
        current_idx = -1  # أحدث شمعة مغلقة
        
        # بيانات المؤشرات
        adx = float(ind.get("adx") or 0.0)
        atr_now = float(ind.get("atr") or 0.0)
        price = float(df["close"].iloc[current_idx])  # السعر الحالي
        
        # ATR السابق (شمعة قبل الحالية)
        if len(df) >= 2:
            prev_atr = compute_indicators(df.iloc[:-1]).get("atr", atr_now)
        else:
            prev_atr = atr_now
            
        # التحقق من شروط الانفجار
        atr_spike = atr_now > prev_atr * BREAKOUT_ATR_SPIKE
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

def handle_breakout_entries(df: pd.DataFrame, ind: dict, bal: float):
    """
    معالجة دخول الانفجارات - تفتح صفقات فورية عند كشف الانفجار
    """
    global state
    
    # تحقق من وجود انفجار
    breakout_signal = detect_breakout(df, ind)
    
    # إذا لم نكن في وضع انفجار نشط وتم كشف انفجار جديد
    if breakout_signal and not state["breakout_active"]:
        price = ind.get("price") or float(df["close"].iloc[-1])
        
        if breakout_signal == "BULL_BREAKOUT":
            # انفجار صعودي - دخول شرائي فوري
            qty = compute_size(bal, price)
            if qty > 0:
                open_market("buy", qty, price)
                state["breakout_active"] = True
                state["breakout_direction"] = "bull"
                state["breakout_entry_price"] = price
                print(colored(f"⚡ BREAKOUT ENGINE: BULLISH EXPLOSION DETECTED - ENTERING LONG", "green"))
                logging.info(f"BREAKOUT_ENGINE: Bullish explosion - LONG entry at {price}")
                return True
                
        elif breakout_signal == "BEAR_BREAKOUT":
            # انهيار هبوطي - دخول بيعي فوري  
            qty = compute_size(bal, price)
            if qty > 0:
                open_market("sell", qty, price)
                state["breakout_active"] = True  
                state["breakout_direction"] = "bear"
                state["breakout_entry_price"] = price
                print(colored(f"⚡ BREAKOUT ENGINE: BEARISH CRASH DETECTED - ENTERING SHORT", "red"))
                logging.info(f"BREAKOUT_ENGINE: Bearish crash - SHORT entry at {price}")
                return True
                
    return False

def handle_breakout_exits(df: pd.DataFrame, ind: dict):
    """
    معالجة خروج الانفجارات - تغلق عندما يهدأ التقلب
    """
    global state
    
    if not state["breakout_active"] or not state["open"]:
        return False
        
    # ✅ PATCH: استخدام الشمعة المغلقة الحالية دائماً
    current_idx = -1
    
    # ATR الحالي والسابق
    atr_now = float(ind.get("atr") or 0.0)
    
    if len(df) >= 2:
        prev_atr = compute_indicators(df.iloc[:-1]).get("atr", atr_now)
    else:
        prev_atr = atr_now
        
    # التحقق من هدوء التقلب (نهاية الانفجار)
    volatility_calm = atr_now < prev_atr * BREAKOUT_CALM_THRESHOLD
    
    if volatility_calm:
        direction = state["breakout_direction"]
        entry_price = state["breakout_entry_price"]
        current_price = ind.get("price") or float(df["close"].iloc[current_idx])
        
        # حساب الربح قبل الإغلاق
        pnl_pct = ((current_price - entry_price) / entry_price * 100 * 
                  (1 if direction == "bull" else -1))
                  
        close_market_strict(f"Breakout ended - {pnl_pct:.2f}% PnL")
        
        print(colored(f"✅ BREAKOUT ENGINE: {direction.upper()} breakout ended - position closed", "magenta"))
        logging.info(f"BREAKOUT_ENGINE: {direction} breakout ended - PnL: {pnl_pct:.2f}%")
        
        # إعادة تعيين حالة الانفجار
        state["breakout_active"] = False
        state["breakout_direction"] = None  
        state["breakout_entry_price"] = None
        
        return True
        
    return False

# ------------ [جميع الدوال الأصلية تبقى كما هي] ------------
def compute_size(balance, price):
    # ... [الكود الأصلي]

def sync_from_exchange_once():
    # ... [الكود الأصلي]

def open_market(side, qty, price):
    # ... [الكود الأصلي]

def close_market_strict(reason):
    # ... [الكود الأصلي]

def reset_after_full_close(reason, prev_side=None):
    global state
    # ... [الكود الأصلي]
    
    # ✅ NEW: Reset breakout state on full close
    state["breakout_active"] = False
    state["breakout_direction"] = None
    state["breakout_entry_price"] = None

def smart_exit_check(info, ind):
    # ... [الكود الأصلي]

def snapshot(bal, info, ind, spread_bps, reason=None, df=None):
    # ... [الكود الأصلي]
    
    # ✅ NEW: Add breakout status to HUD
    if state["breakout_active"]:
        print(colored(f"   ⚡ BREAKOUT MODE ACTIVE: {state['breakout_direction'].upper()} - Monitoring volatility...", "cyan"))

# ------------ DECISION LOOP (معدّل بإضافة Breakout Engine) ------------
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

            # ✅ NEW: BREAKOUT ENGINE - يعمل قبل أي منطق آخر
            # 1. معالجة خروج الانفجارات أولاً (إذا كنا في وضع انفجار نشط)
            breakout_exited = handle_breakout_exits(df, ind)
            
            # 2. معالجة دخول الانفجارات (إذا لم نكن في صفقة)
            breakout_entered = False
            if not state["open"] and not breakout_exited:
                breakout_entered = handle_breakout_entries(df, ind, bal)
            
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

            # ------------ [الكود الأصلي يبدأ من هنا] ------------
            # باقي المنطق الأصلي يعمل فقط عندما لسنا في وضع انفجار
            
            if state["open"] and px:
                state["pnl"] = (px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]

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

# ------------ [باقي الكود الأصلي يبقى كما هو] ------------
def keepalive_loop():
    # ... [الكود الأصلي]

app = Flask(__name__)

# ... [جميع routes الأصلية تبقى كما هي]

@app.route("/metrics")
def metrics():
    return jsonify({
        # ... [الكود الأصلي],
        "breakout_engine": {  # ✅ NEW: Breakout engine status
            "active": state.get("breakout_active", False),
            "direction": state.get("breakout_direction"),
            "entry_price": state.get("breakout_entry_price")
        }
    })

@app.route("/health")
def health():
    return jsonify({
        # ... [الكود الأصلي],
        "breakout_active": state.get("breakout_active", False)  # ✅ NEW
    }), 200

# ------------ Boot Sequence ------------
if __name__ == "__main__":
    print("✅ Starting HARDENED Flask server with BREAKOUT ENGINE...")
    
    # HARDENING: Load persisted state
    load_state()
    
    # HARDENING: Start watchdog thread
    threading.Thread(target=watchdog_check, daemon=True).start()
    print("🦮 Watchdog started")
    
    # Start main loops
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
