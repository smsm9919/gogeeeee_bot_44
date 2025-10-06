# -*- coding: utf-8 -*-
"""
RF Futures Bot — Smart Pro (BingX Perp, CCXT)
- Entries: TradingView Range Filter EXACT (BUY/SELL)
- Size: 60% balance × leverage (default 10x)
- Exit:
  • Opposite RF signal ALWAYS closes
  • Smart Profit: TP1 partial + move to breakeven + ATR trailing (trend-riding)
- Advanced Candle & Indicator Analysis for Position Management
- Robust keepalive (SELF_URL/RENDER_EXTERNAL_URL), retries, /metrics

Patched:
- BingX position mode support (oneway|hedge) with correct positionSide
- Safe state updates (no local open if exchange order failed)
- Rich logs: candles/regime/bias/ATR%, WAIT reason, and time left to candle close
- Advanced Candle Pattern Detection in Arabic/English
- Smart Scale-In/Scale-Out based on candlestick patterns + indicators
- Dynamic Trailing SL based on market regime
- Trend Amplifier: ADX-based scale-in, dynamic TP, ratchet lock
- Hold-TP & Impulse Harvest for advanced profit management
"""

import os, time, math, threading, requests, traceback, random
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
COOLDOWN_AFTER_CLOSE_BARS = 2

# Range Filter params
RF_SOURCE = "close"
RF_PERIOD = 20
RF_MULT = 3.5
USE_TV_BAR = False

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
SCALE_IN_MAX_STEPS = 3
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

# pacing / keepalive
SLEEP_S = 30
SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
KEEPALIVE_SECONDS = 50
PORT = int(os.getenv("PORT", 5000))

print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • SYMBOL={SYMBOL} • {INTERVAL}", "yellow"))
print(colored(f"STRATEGY: {STRATEGY.upper()} • SMART_EXIT={'ON' if USE_SMART_EXIT else 'OFF'}", "yellow"))
print(colored(f"ADVANCED POSITION MGMT: SCALE_IN_STEPS={SCALE_IN_MAX_STEPS} • ADX_STRONG={ADX_STRONG_THRESH}", "yellow"))
print(colored(f"TREND AMPLIFIER: ADX_TIERS[{ADX_TIER1}/{ADX_TIER2}/{ADX_TIER3}] • RATCHET_LOCK={RATCHET_LOCK_PCT*100}%", "yellow"))
print(colored(f"KEEPALIVE: url={'SET' if SELF_URL else 'NOT SET'} • every {KEEPALIVE_SECONDS}s", "yellow"))
print(colored(f"BINGX_POSITION_MODE={BINGX_POSITION_MODE}", "yellow"))
print(colored(f"SERVER: Starting on port {PORT}", "green"))

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
try:
    ex.load_markets()
except Exception as e:
    print(colored(f"⚠️ load_markets: {e}", "yellow"))

# ------------ Helpers ------------
def fmt(v, d=6, na="N/A"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception: return na

def with_retry(fn, attempts=3, base_wait=0.4):
    for i in range(attempts):
        try: return fn()
        except Exception:
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

# ------------ Indicators (display-only) ------------
def wilder_ema(s: pd.Series, n: int): return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
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

    i = len(df)-1 if USE_TV_BAR else len(df)-2
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
    
    idx = -1 if USE_TV_BAR else -2
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
    src = df[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    dfilt = filt - filt.shift(1)
    fdir = pd.Series(0.0, index=filt.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0.0)
    upward = (fdir==1).astype(int); downward=(fdir==-1).astype(int)
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
    i=len(df)-1 if USE_TV_BAR else len(df)-2
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
    "highest_profit_pct": 0.0  # Trend Amplifier: ratchet lock
}
compound_pnl = 0.0
last_signal_id = None
post_close_cooldown = 0

def compute_size(balance, price):
    capital = (balance or 0.0) * RISK_ALLOC * LEVERAGE
    return max(0.0, capital / max(float(price or 0.0), 1e-9))

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
                "highest_profit_pct": 0.0
            })
            print(colored(f"✅ Synced position ⇒ {side.upper()} qty={fmt(qty,4)} @ {fmt(entry)}","green"))
            return
        print(colored("↔️  Sync: no open position on exchange.","yellow"))
    except Exception as e:
        print(colored(f"❌ sync error: {e}","red"))

# ------------ BingX params helpers ------------
def _position_params_for_open(side: str):
    if BINGX_POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if side == "buy" else "SHORT", "reduceOnly": False}
    return {"positionSide": "BOTH", "reduceOnly": False}

def _position_params_for_close():
    if BINGX_POSITION_MODE == "hedge":
        return {"positionSide": "LONG" if state.get("side")=="long" else "SHORT", "reduceOnly": True}
    return {"positionSide": "BOTH", "reduceOnly": True}

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

def should_scale_in(candle_info: dict, ind: dict, current_side: str) -> tuple:
    """Return (should_scale, step_size, reason)"""
    if state["scale_ins"] >= SCALE_IN_MAX_STEPS:
        return False, 0.0, "Max scale-in steps reached"
    
    adx = ind.get("adx", 0)
    rsi = ind.get("rsi", 50)
    plus_di = ind.get("plus_di", 0)
    minus_di = ind.get("minus_di", 0)
    
    # Get dynamic step based on ADX
    step_size, step_reason = get_dynamic_scale_in_step(adx)
    if step_size <= 0:
        return False, 0.0, step_reason
    
    # RSI direction confirmation
    if current_side == "long" and rsi < RSI_TREND_BUY:
        return False, 0.0, f"RSI {rsi:.1f} < {RSI_TREND_BUY}"
    if current_side == "short" and rsi > RSI_TREND_SELL:
        return False, 0.0, f"RSI {rsi:.1f} > {RSI_TREND_SELL}"
    
    # DI direction confirmation
    if current_side == "long" and plus_di <= minus_di:
        return False, 0.0, "+DI <= -DI"
    if current_side == "short" and minus_di <= plus_di:
        return False, 0.0, "-DI <= +DI"
    
    # Candle pattern strength
    candle_strength = candle_info.get("strength", 0)
    if candle_strength < 2:
        return False, 0.0, f"Weak candle pattern: {candle_info.get('name_en', 'NONE')}"
    
    # Specific strong patterns for scale-in
    strong_patterns = ["THREE_WHITE_SOLDIERS", "THREE_BLACK_CROWS", "ENGULF_BULL", "ENGULF_BEAR", "MARUBOZU_BULL", "MARUBOZU_BEAR"]
    if candle_info.get("pattern") in strong_patterns:
        return True, step_size, f"Strong {candle_info.get('name_en')} pattern + {step_reason}"
    
    return False, 0.0, f"Moderate pattern: {candle_info.get('name_en', 'NONE')}"

def should_scale_out(candle_info: dict, ind: dict, current_side: str) -> tuple:
    """Return (should_scale_out, reason)"""
    if state["qty"] <= 0:
        return False, "No position to scale out"
    
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

# ------------ Orders ------------
def open_market(side, qty, price):
    global state
    if qty<=0:
        print(colored("❌ qty<=0 skip open","red")); return

    params = _position_params_for_open(side)

    if MODE_LIVE:
        try:
            lev_params = {"positionSide": params["positionSide"]} if BINGX_POSITION_MODE=="hedge" else {"positionSide":"BOTH"}
            try: ex.set_leverage(LEVERAGE, SYMBOL, params=lev_params)
            except Exception as e: print(colored(f"⚠️ set_leverage: {e}", "yellow"))
            ex.create_order(SYMBOL, "market", side, qty, None, params)
        except Exception as e:
            print(colored(f"❌ open: {e}", "red"))
            return  # لا نُحدّث الحالة إذا فشل التنفيذ الفعلي

    state.update({
        "open": True, "side": "long" if side=="buy" else "short", 
        "entry": price, "qty": qty, "pnl": 0.0, "bars": 0, 
        "trail": None, "tp1_done": False, "breakeven": None,
        "scale_ins": 0, "scale_outs": 0,
        "last_action": "OPEN", "action_reason": "Initial position",
        "highest_profit_pct": 0.0  # Reset ratchet lock
    })
    print(colored(f"✅ OPEN {side.upper()} qty={fmt(qty,4)} @ {fmt(price)}","green" if side=="buy" else "red"))

def scale_in_position(scale_pct: float, reason: str):
    """Add to existing position"""
    global state
    if not state["open"]: return
    
    current_price = price_now() or state["entry"]
    additional_qty = state["qty"] * scale_pct
    side = "buy" if state["side"] == "long" else "sell"
    
    if MODE_LIVE:
        try: 
            ex.create_order(SYMBOL, "market", side, additional_qty, None, _position_params_for_open(side))
        except Exception as e: 
            print(colored(f"❌ scale_in: {e}","red")); return
    
    # Update average entry price
    total_qty = state["qty"] + additional_qty
    state["entry"] = (state["entry"] * state["qty"] + current_price * additional_qty) / total_qty
    state["qty"] = total_qty
    state["scale_ins"] += 1
    state["last_action"] = "SCALE_IN"
    state["action_reason"] = reason
    
    print(colored(f"📈 SCALE IN +{scale_pct*100:.0f}% | Total qty={fmt(state['qty'],4)} | Avg entry={fmt(state['entry'])} | Reason: {reason}", "cyan"))

def close_partial(frac, reason):
    """Close fraction of current position (smart TP1)."""
    global state, compound_pnl
    if not state["open"]: return
    qty_close = max(0.0, state["qty"]*min(max(frac,0.0),1.0))
    if qty_close<=0: return
    px = price_now() or state["entry"]
    side = "sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_position_params_for_close())
        except Exception as e: print(colored(f"❌ partial close: {e}","red")); return
    pnl=(px-state["entry"])*qty_close*(1 if state["side"]=="long" else -1)
    compound_pnl+=pnl
    state["qty"]-=qty_close
    state["scale_outs"] += 1
    state["last_action"] = "SCALE_OUT"
    state["action_reason"] = reason
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem_qty={fmt(state['qty'],4)}","magenta"))
    if state["qty"]<=0:
        reset_after_full_close("fully_exited")

def reset_after_full_close(reason):
    global state, post_close_cooldown
    print(colored(f"🔚 CLOSE {reason} totalCompounded now={fmt(compound_pnl)}","magenta"))
    state.update({
        "open": False, "side": None, "entry": None, "qty": 0.0, 
        "pnl": 0.0, "bars": 0, "trail": None, "tp1_done": False, 
        "breakeven": None, "scale_ins": 0, "scale_outs": 0,
        "last_action": "CLOSE", "action_reason": reason,
        "highest_profit_pct": 0.0
    })
    post_close_cooldown = COOLDOWN_AFTER_CLOSE_BARS

def close_market(reason):
    global state, compound_pnl
    if not state["open"]: return
    px=price_now() or state["entry"]; qty=state["qty"]
    side="sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty,None,_position_params_for_close())
        except Exception as e: print(colored(f"❌ close: {e}","red")); return
    pnl=(px-state["entry"])*qty*(1 if state["side"]=="long" else -1)
    compound_pnl+=pnl
    print(colored(f"🔚 CLOSE {state['side']} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
    reset_after_full_close(reason)

# ------------ Advanced Position Management Check ------------
def advanced_position_management(candle_info: dict, ind: dict):
    """Handle scale-in, scale-out, and dynamic trailing"""
    if not state["open"]:
        return None
    
    current_side = state["side"]
    px = ind.get("price") or price_now() or state["entry"]
    
    # Scale-in check with dynamic step
    should_scale, step_size, scale_reason = should_scale_in(candle_info, ind, current_side)
    if should_scale:
        scale_in_position(step_size, scale_reason)
        return "SCALE_IN"
    
    # Scale-out check
    should_scale_out, scale_out_reason = should_scale_out(candle_info, ind, current_side)
    if should_scale_out:
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

    px = info["price"]; e = state["entry"]; side = state["side"]
    rr = (px - e)/e * 100.0 * (1 if side=="long" else -1)
    atr = ind.get("atr") or 0.0
    adx = ind.get("adx") or 0.0
    rsi = ind.get("rsi") or 50.0

    # ---------- HOLD-TP: لا خروج بدري مع ترند قوي ----------
    if side == "long" and adx >= 30 and rsi >= RSI_TREND_BUY:
        print(colored("💎 HOLD-TP: strong uptrend continues, delaying TP", "cyan"))
        # نكمل من غير أي إغلاقات مبكرة
    elif side == "short" and adx >= 30 and rsi <= RSI_TREND_SELL:
        print(colored("💎 HOLD-TP: strong downtrend continues, delaying TP", "cyan"))
        # نكمل من غير أي إغلاقات مبكرة

    # ---------- Impulse Harvest: استغلال الشموع الطويلة ----------
    try:
        df_last = fetch_ohlcv()
        idx = -1 if USE_TV_BAR else -2
        o0 = float(df_last["open"].iloc[idx]); c0 = float(df_last["close"].iloc[idx])
        body = abs(c0 - o0)
        dir_candle = 1 if c0 > o0 else -1
        impulse = (body / atr) if atr > 0 else 0.0

        # شمعة قوية في اتجاه الصفقة ⇒ جني أرباح جزئي سريع
        if impulse >= 1.2 and ((side == "long" and dir_candle > 0) or (side == "short" and dir_candle < 0)):
            harvest_frac = 0.33 if impulse < 2.0 else 0.50
            close_partial(harvest_frac, f"Impulse x{impulse:.2f} ATR")
            # قفّل على الأقل بريك-إيفن وفعّل تريل لو مش مفعّل
            state["breakeven"] = state.get("breakeven") or e
            if atr and ATR_MULT_TRAIL > 0:
                gap = atr * ATR_MULT_TRAIL
                if side == "long":
                    state["trail"] = max(state.get("trail") or (px - gap), px - gap)
                else:
                    state["trail"] = min(state.get("trail") or (px + gap), px + gap)
    except Exception as _:
        pass

    # انتظر كام شمعة بعد الدخول لتجنب الخروج السريع
    if state["bars"] < 2:
        return None

    # Trend Amplifier: Dynamic TP based on ADX
    tp_multiplier, trail_activate_multiplier = get_dynamic_tp_params(adx)
    current_tp1_pct = TP1_PCT * tp_multiplier
    current_trail_activate = TRAIL_ACTIVATE * trail_activate_multiplier
    current_tp1_pct_pct = current_tp1_pct * 100.0
    current_trail_activate_pct = current_trail_activate * 100.0

    # Highest profit (ratchet)
    if rr > state["highest_profit_pct"]:
        state["highest_profit_pct"] = rr
        if tp_multiplier > 1.0:
            print(colored(f"🎯 TREND AMPLIFIER: New high {rr:.2f}% • TP={current_tp1_pct_pct:.2f}% • TrailActivate={current_trail_activate_pct:.2f}%", "green"))

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
                close_market(f"TRAIL_ATR({ATR_MULT_TRAIL}x)"); return True
        else:
            new_trail = px + gap
            state["trail"] = min(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None:
                state["trail"] = min(state["trail"], state["breakeven"])
            if px > state["trail"]:
                close_market(f"TRAIL_ATR({ATR_MULT_TRAIL}x)"); return True
    return None

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
    print(f"   ⏱️ Candle closes in ~ {left_s}s")
    print()

    # ===== POSITION & MANAGEMENT =====
    print("🧭 POSITION & MANAGEMENT")
    print(f"   💰 Balance {fmt(bal,2)} USDT   Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x   PostCloseCooldown={post_close_cooldown}")
    if state["open"]:
        lamp = '🟩 LONG' if state['side']=='long' else '🟥 SHORT'
        print(f"   📌 {lamp}  Entry={fmt(state['entry'])}  Qty={fmt(state['qty'],4)}  Bars={state['bars']}  PnL={fmt(state['pnl'])}")
        print(f"   🎯 Management: Scale-ins={state['scale_ins']}/{SCALE_IN_MAX_STEPS}  Scale-outs={state['scale_outs']}  Trail={fmt(state['trail'])}")
        print(f"   📊 TP1_done={state['tp1_done']}  Breakeven={fmt(state['breakeven'])}  HighestProfit={fmt(state['highest_profit_pct'],2)}%")
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
        should_scale_out, scale_out_reason = should_scale_out(candle_info, ind, current_side)
        
        if should_scale:
            print(colored(f"   ✅ SCALE-IN READY: {scale_reason}", "green"))
        elif should_scale_out:
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
    else:
        print("   🔄 Waiting for trading signals...")
    print()

    # ===== RESULTS =====
    print("📦 RESULTS")
    eff_eq = (bal or 0.0) + compound_pnl if MODE_LIVE else compound_pnl
    print(f"   🧮 CompoundPnL {fmt(compound_pnl)}   🚀 EffectiveEq {fmt(eff_eq)} USDT")
    if reason:
        print(colored(f"   ℹ️ WAIT — reason: {reason}","yellow"))
    print(colored("─"*100,"cyan"))

# ------------ Decision Loop ------------
def trade_loop():
    global last_signal_id, state, post_close_cooldown
    sync_from_exchange_once()
    while True:
        try:
            bal=balance_usdt()
            px=price_now()
            df=fetch_ohlcv()
            info=compute_tv_signals(df)
            ind=compute_indicators(df)
            spread_bps = orderbook_spread_bps()

            if state["open"] and px:
                state["pnl"]=(px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]

            # Smart profit (trend-aware) with Trend Amplifier
            smart_exit_check(info, ind)

            # Decide
            sig="buy" if info["long"] else ("sell" if info["short"] else None)
            reason=None
            if not sig:
                reason="no signal"
            elif spread_bps is not None and spread_bps>SPREAD_GUARD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {SPREAD_GUARD_BPS})"
            elif post_close_cooldown>0:
                reason=f"cooldown {post_close_cooldown} bars"

            # Close on opposite RF signal ALWAYS
            if state["open"] and sig and (reason is None):
                desired="long" if sig=="buy" else "short"
                if state["side"]!=desired:
                    close_market("opposite_signal")
                    qty=compute_size(bal, px or info["price"])
                    if qty>0:
                        open_market(sig, qty, px or info["price"])
                        last_signal_id=f"{info['time']}:{sig}"
                        snapshot(bal,info,ind,spread_bps,None, df)
                        time.sleep(SLEEP_S); continue

            # Open new position when flat
            if not state["open"] and (reason is None) and sig:
                qty=compute_size(bal, px or info["price"])
                if qty>0:
                    open_market(sig, qty, px or info["price"])
                    last_signal_id=f"{info['time']}:{sig}"
                else:
                    reason="qty<=0"

            snapshot(bal,info,ind,spread_bps,reason, df)

            if state["open"]:
                state["bars"] += 1
            if post_close_cooldown>0 and not state["open"]:
                post_close_cooldown -= 1

        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}","red"))
        time.sleep(SLEEP_S)

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
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

root_logged = False

@app.route("/")
def home():
    global root_logged
    if not root_logged:
        print("GET / HTTP/1.1 200")
        root_logged = True
    mode = 'LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ RF Bot — {SYMBOL} {INTERVAL} — {mode} — {STRATEGY.upper()} — ADVANCED — TREND AMPLIFIER"

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
            "highest_profit_pct": state.get("highest_profit_pct", 0)
        }
    })

@app.route("/ping")
def ping(): return "pong", 200

# Boot
threading.Thread(target=trade_loop, daemon=True).start()
threading.Thread(target=keepalive_loop, daemon=True).start()
if __name__ == "__main__":
    print("✅ Starting Flask server...")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
