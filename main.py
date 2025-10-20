# -*- coding: utf-8 -*-
"""
RF Futures Bot — Trend-Only Pro (BingX Perp, CCXT) — FIXED
- Entry: Range Filter with LIVE-bar option (no scalping, but faster entry)
- Post-entry: Supertrend + ADX/RSI/ATR harvesting
- Dynamic TP ladder (safe zip) + Ratchet + Wick/Impulse harvesting
- Smart peak/valley manager: "wait next candle" on doubtful breaks
- Emergency protection + Strict exchange close (robust retries)
- Final-chunk hard close (e.g., last 50 DOGE)
- SMC-lite: EQH/EQL + FVG sniff + Stop-hunt guard (wick+vol)
- TVR profile (optional) for volume-time context
- Flask /metrics & /health, rotated logging

NOTE: RISK = 60% × 10x by default. Paper mode auto if no keys.
"""

import os, time, math, threading, requests, random, traceback, signal, sys, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pandas as pd
import ccxt

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV / MODE ===================
API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)

SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT     = int(os.getenv("PORT", 5000))

# =================== SETTINGS ===================
SYMBOL   = "DOGE/USDT:USDT"
INTERVAL = "15m"
LEVERAGE = 10
RISK_ALLOC = 0.60
BINGX_POSITION_MODE = "oneway"  # or "hedge"

# Range Filter core
RF_SOURCE = "close"
RF_PERIOD = 20
RF_MULT   = 3.5

# LIVE RF entry
RF_LIVE = True
RF_LIVE_MIN_ELAPSED = 0.28   # لازم يمر ~28% من الشمعة
RF_LIVE_HYST_BPS    = 6.0    # فرق عن الفلتر بالـ bps للتأكيد

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

# Dynamic TP defaults (ستُبنى ديناميكيًا لكن هذه حدود آمنة)
TP_LADDER_MAX = 4
TP1_PCT = 0.40
TP1_CLOSE_FRAC = 0.50
BREAKEVEN_AFTER = 0.30
TRAIL_ACTIVATE = 1.20
ATR_MULT_TRAIL = 1.6

# End/hold heuristics
MIN_TREND_HOLD_ADX = 25
DI_FLIP_BUFFER = 1.0
END_TREND_RSI_NEUTRAL = (45,55)

# Impulse/Wick/Ratchet
IMPULSE_ATR_X = 1.8
WICK_PCT_FOR_HARVEST = 45.0
RATCHET_RETRACE = 0.40

# Strict close
STRICT_EXCHANGE_CLOSE = True
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0
MIN_RESIDUAL_TO_FORCE = 1.0    # عقود قليلة جدًا → تجاهل
FINAL_FULL_CLOSE_QTY  = 50.0   # آخر 50 DOGE → اغلاق كامل

# Spread guard
MAX_SPREAD_BPS = 6.0

# Patient mode
PATIENT_TRADER_MODE = True
PATIENT_HOLD_BARS = 8
PATIENT_HOLD_SECONDS = 3600
NEVER_FULL_CLOSE_BEFORE_TP1 = True

# Peak/Valley logic
SWING_LOOK_LR = (2,2)           # يسار/يمين للفراكتال
SWING_BREAK_BPS = 8.0           # مقدار الاختراق المقبول
SWING_FAIL_ADX_DROP = 3.0       # هبوط ADX للإقرار بالفشل
SWING_WAIT_NEXT_CANDLE = True   # نكتب في اللوج "انتظار الشمعة القادمة"

# TVR (اختياري)
TVR_ENABLED = True
TVR_BUCKETS = 96

# Logging =====================
def setup_file_logging():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) for h in root.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        root.addHandler(fh)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready", "cyan"))
setup_file_logging()

# Exchange ====================
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY, "secret": API_SECRET,
        "enableRateLimit": True, "timeout": 20000,
        "options": {"defaultType":"swap"}
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
        AMT_PREC = int((MARKET.get("precision",{}) or {}).get("amount", 0) or 0)
        lot = (MARKET.get("limits",{}) or {}).get("amount",{}) or {}
        LOT_STEP = lot.get("step")
        LOT_MIN  = lot.get("min")
        print(colored(f"🔧 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}", "yellow"))

def ensure_leverage_and_mode():
    try:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
        except Exception as e:
            print(colored(f"⚠️ set_leverage: {e}", "yellow"))
        print(colored(f"📌 position mode: {BINGX_POSITION_MODE}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_leverage: {e}", "yellow"))

try:
    load_market_specs()
    ensure_leverage_and_mode()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}", "yellow"))

# =============== Helpers / State =================
from decimal import Decimal, ROUND_DOWN, InvalidOperation

def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = int(AMT_PREC or 0)
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except Exception:
        return max(0.0, float(q))

def safe_qty(q):
    q = _round_amt(q)
    if q <= 0: print(colored(f"⚠️ qty invalid after normalize → {q}", "yellow"))
    return q

def fmt(v, d=6):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return "N/A"
        return f"{float(v):.{d}f}"
    except Exception:
        return "N/A"

def _interval_seconds(iv: str) -> int:
    iv = (iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame) -> int:
    tf = _interval_seconds(INTERVAL)
    if len(df)==0: return tf
    cur_start = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    next_close = cur_start + tf*1000
    while next_close <= now_ms:
        next_close += tf*1000
    return int(max(0, next_close-now_ms)/1000)

# State
state = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "scale_outs": 0, "highest_profit_pct": 0.0,
    "tp1_done": False, "profit_idx": 0,
    "_tp_targets": None, "_tp_fracs": None,
    "await_next_candle": None,  # {"type":"swing_fail","dir":"up/down","set_at_bar":...}
    # swings
    "last_swing_high": None, "last_swing_low": None,
}
compound_pnl = 0.0

# Safe state lock
_state_lock = threading.Lock()
def get_state(): 
    with _state_lock: return dict(state)
def set_state(upd: dict):
    with _state_lock: state.update(upd)

# ================= Indicators ====================
def wilder_ema(s: pd.Series, n: int): return s.ewm(alpha=1/n, adjust=False).mean()

def compute_inds(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 3:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"adx":0.0,"atr":0.0,"st":None,"st_dir":0,"ema9":None,"ema20":None}
    c = df["close"].astype(float); h=df["high"].astype(float); l=df["low"].astype(float)

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

    # Supertrend (quick)
    st_val, st_dir = None, 0
    try:
        st_p, st_m = 10, 3.0
        hl2=(h+l)/2.0
        atr_st = wilder_ema(tr, st_p)
        upper=hl2+st_m*atr_st; lower=hl2-st_m*atr_st
        st=[float('nan')]; dirv=[0]
        for i in range(1, len(df)):
            prev_st=st[-1]; prev_dir=dirv[-1]
            cur=float(c.iloc[i]); ub=float(upper.iloc[i]); lb=float(lower.iloc[i])
            if math.isnan(prev_st):
                st.append(lb if cur>lb else ub); dirv.append(1 if cur>lb else -1); continue
            if prev_dir==1:
                ub=min(ub, prev_st); st_now = lb if cur>ub else ub; dir_now = 1 if cur>ub else -1
            else:
                lb=max(lb, prev_st); st_now = ub if cur<lb else lb; dir_now = -1 if cur<lb else 1
            st.append(st_now); dirv.append(dir_now)
        st_series = pd.Series(st[1:], index=df.index[1:]).reindex(df.index, method='pad')
        dir_series= pd.Series(dirv[1:], index=df.index[1:]).reindex(df.index, method='pad').fillna(0)
        st_val=float(st_series.iloc[-1]); st_dir=int(dir_series.iloc[-1])
    except Exception:
        pass

    ema9  = c.ewm(span=9,  adjust=False).mean().iloc[-1]
    ema20 = c.ewm(span=20, adjust=False).mean().iloc[-1]

    i=len(df)-1; j=max(0,i-1)
    return {
        "rsi": float(rsi.iloc[i]),
        "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]),
        "adx": float(adx.iloc[i]),
        "adx_prev": float(adx.iloc[j]),
        "atr": float(atr.iloc[i]),
        "st": st_val, "st_dir": st_dir,
        "ema9": float(ema9), "ema20": float(ema20)
    }

# =============== Range Filter =====================
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

def rf_signal_closed(df_closed: pd.DataFrame):
    if len(df_closed) < RF_PERIOD+3:
        i=-1
        return {"time": int(df_closed["time"].iloc[i]) if len(df_closed) else int(time.time()*1000),
                "price": float(df_closed["close"].iloc[i]) if len(df_closed) else None,
                "long": False, "short": False, "filter": None}
    src=df_closed[RF_SOURCE].astype(float)
    hi,lo,filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    dfilt = filt - filt.shift(1)
    fdir = pd.Series(0.0, index=filt.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0.0)
    upward=(fdir==1).astype(int); downward=(fdir==-1).astype(int)
    src_gt_f=(src>filt); src_lt_f=(src<filt)
    longCond  = src_gt_f & (upward>0)
    shortCond = src_lt_f & (downward>0)
    i=len(src)-1
    longSignal  = bool(longCond.iloc[i]  and (fdir.iloc[i]==1)  and (src.iloc[i]>src.iloc[i-1]))
    shortSignal = bool(shortCond.iloc[i] and (fdir.iloc[i]==-1) and (src.iloc[i]<src.iloc[i-1]))
    return {"time": int(df_closed["time"].iloc[-1]), "price": float(df_closed["close"].iloc[-1]),
            "long": longSignal, "short": shortSignal, "filter": float(filt.iloc[-1]),
            "hi": float(hi.iloc[-1]), "lo": float(lo.iloc[-1])}

def rf_signal_live(df_full: pd.DataFrame, filt_last: float):
    """دخول على الشمعة الحيّة مع فلتر زمن/هيستريسيز."""
    if not RF_LIVE or len(df_full)<2: return {"long":False,"short":False}
    tf = _interval_seconds(INTERVAL)
    cur_start_ms = int(df_full["time"].iloc[-1])
    elapsed = (int(time.time()*1000) - cur_start_ms)/1000.0
    if elapsed < tf*RF_LIVE_MIN_ELAPSED:  # بدري قوي
        return {"long":False,"short":False}

    price = float(df_full["close"].iloc[-1])
    try:
        bps = abs((price - filt_last)/filt_last)*10000.0 if filt_last else 0.0
    except Exception:
        bps = 0.0

    go_long  = price>filt_last and bps>=RF_LIVE_HYST_BPS
    go_short = price<filt_last and bps>=RF_LIVE_HYST_BPS
    return {"long": bool(go_long), "short": bool(go_short)}

# =============== Swings / SMC-lite ==================
def _find_swings(df: pd.DataFrame, L=2, R=2):
    if len(df) < L+R+3: return None,None
    h=df["high"].astype(float).values; l=df["low"].astype(float).values
    ph=[None]*len(df); pl=[None]*len(df)
    for i in range(L, len(df)-R):
        if all(h[i]>=h[j] for j in range(i-L, i+R+1)): ph[i]=h[i]
        if all(l[i]<=l[j] for j in range(i-L, i+R+1)): pl[i]=l[i]
    return ph,pl

def update_swings(df: pd.DataFrame):
    ph,pl = _find_swings(df.iloc[:-1], *SWING_LOOK_LR) if len(df)>=3 else (None,None)
    if not ph or not pl: return
    try:
        last_ph = max(v for v in ph if v is not None)
        last_pl = min(v for v in pl if v is not None)
        state["last_swing_high"] = float(last_ph)
        state["last_swing_low"]  = float(last_pl)
    except Exception:
        pass

def swing_fail_manager(df: pd.DataFrame, ind: dict, price: float):
    """
    لو فشل اختراق القمة/القاع بشكل منطقي:
      - نعلن "انتظار الشمعة القادمة" مرة واحدة.
      - عند الشمعة التالية، لو تأكّد الفشل ⇒ جني جزئي قوي + تشديد تريل أو إغلاق صارم حسب الوضع.
    """
    if not state["open"] or price is None: return
    side = state["side"]; adx=ind.get("adx",0.0); adx_prev=ind.get("adx_prev",adx)
    swing_hi = state.get("last_swing_high"); swing_lo=state.get("last_swing_low")
    if not swing_hi and not swing_lo: return

    def _bps(a,b): 
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0

    # تحقق الفشل
    fail_up   = (side=="long"  and swing_hi and price < swing_hi and _bps(price, swing_hi) <= SWING_BREAK_BPS and (adx_prev-adx)>=SWING_FAIL_ADX_DROP)
    fail_down = (side=="short" and swing_lo and price > swing_lo and _bps(price, swing_lo) <= SWING_BREAK_BPS and (adx_prev-adx)>=SWING_FAIL_ADX_DROP)

    if (fail_up or fail_down) and state.get("await_next_candle") is None and SWING_WAIT_NEXT_CANDLE:
        set_state({"await_next_candle": {"type":"swing_fail", "dir": "up" if fail_up else "down", "set_at_bar": state.get("bars",0)}})
        print(colored("⏳ Swing break doubtful — سننتظر الشمعة القادمة للتأكيد", "yellow"))
        return

    # لو كنا ننتظر الشمعة القادمة
    await_info = state.get("await_next_candle")
    if await_info and state.get("bars",0) > await_info.get("set_at_bar", 0):
        # تأكيد الفشل: لم يحصل الاختراق + ADX ما زال يضعف
        if (side=="long" and swing_hi and price < swing_hi) or (side=="short" and swing_lo and price > swing_lo):
            # جَنِ جزئي قوي ثم شد التريل؛ إن كانت الكمية صغيرة → إغلاق صارم
            print(colored("⚠️ swing فشل مؤكد — جني/تشديد/إغلاق", "yellow"))
            close_partial(0.50, "Swing-fail confirm")
            # لو ما تبقى <= FINAL_FULL_CLOSE_QTY → اقفل
            if state["qty"] <= FINAL_FULL_CLOSE_QTY:
                close_market_strict("FINAL_CHUNK_AFTER_SWING_FAIL")
        set_state({"await_next_candle": None})

# =============== Exchange utils =====================
def with_retry(fn, attempts=3, wait=0.4):
    for i in range(attempts):
        try:
            return fn()
        except Exception:
            if i==attempts-1: raise
            time.sleep(wait*(2**i) + random.random()*0.2)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception:
        return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception:
        return None

def orderbook_spread_bps():
    try:
        ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid=(bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

def compute_size(balance, price):
    effective_balance = (balance or 0.0) + (compound_pnl or 0.0)
    capital = effective_balance * RISK_ALLOC * LEVERAGE
    raw = max(0.0, capital / max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

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

# =============== Orders =========================
def open_market(side, qty, price):
    if qty<=0: 
        print(colored("❌ qty<=0 skip open","red")); 
        return
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception as e: print(colored(f"⚠️ set_leverage: {e}", "yellow"))
            ex.create_order(SYMBOL,"market",side,qty,None,_position_params_for_open(side))
        except Exception as e:
            print(colored(f"❌ open: {e}","red")); logging.error(f"open_market error: {e}"); return
    set_state({
        "open": True, "side": "long" if side=="buy" else "short",
        "entry": price, "qty": qty, "pnl": 0.0, "bars": 0,
        "trail": None, "breakeven": None,
        "scale_outs": 0, "highest_profit_pct": 0.0,
        "tp1_done": False, "profit_idx": 0,
        "_tp_targets": None, "_tp_fracs": None,
        "await_next_candle": None
    })
    print(colored(f"✅ OPEN {side.upper()} qty={fmt(qty,4)} @ {fmt(price)}","green" if side=="buy" else "red"))
    logging.info(f"OPEN {side} qty={qty} price={price}")

def close_market_strict(reason):
    exch_qty, exch_side, exch_entry = _read_exchange_position()
    if exch_qty <= 0:
        if state.get("open"): reset_after_full_close(f"strict-close-zero ({reason})")
        return

    side_to_close = "sell" if (exch_side=="long") else "buy"
    qty_to_close  = safe_qty(exch_qty)
    attempts=0; last_error=None
    while attempts < CLOSE_RETRY_ATTEMPTS:
        try:
            if MODE_LIVE:
                params = _position_params_for_close(); params["reduceOnly"] = True
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            time.sleep(CLOSE_VERIFY_WAIT_S)
            left_qty, _, _ = _read_exchange_position()
            if left_qty <= 0:
                px = price_now() or state.get("entry")
                entry_px = state.get("entry") or exch_entry or px
                side = state.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
                qty  = exch_qty
                pnl  = (px - entry_px) * qty * (1 if side=="long" else -1)
                global compound_pnl
                compound_pnl += pnl
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                reset_after_full_close(reason)
                return
            qty_to_close = safe_qty(left_qty)
            attempts += 1
            print(colored(f"⚠️ strict close retry {attempts}/{CLOSE_RETRY_ATTEMPTS} — residual={fmt(left_qty,4)}","yellow"))
            if qty_to_close < MIN_RESIDUAL_TO_FORCE: time.sleep(CLOSE_VERIFY_WAIT_S)
        except Exception as e:
            last_error = e; logging.error(f"close_market_strict attempt {attempts+1}: {e}"); attempts += 1; time.sleep(CLOSE_VERIFY_WAIT_S)
    print(colored(f"❌ STRICT CLOSE FAILED — last_error={last_error}", "red"))
    logging.critical(f"STRICT CLOSE FAILED — last_error={last_error}")

def close_partial(frac, reason):
    if not state["open"]: return
    qty_close = safe_qty(max(0.0, state["qty"] * min(max(frac,0.0),1.0)))
    # min lot guard from exchange
    min_unit = float(LOT_MIN or 0.0)
    if qty_close <= 0 or (min_unit>0 and qty_close < min_unit):
        print(colored(f"⏸️ skip partial (amount={fmt(qty_close,4)} < min_unit={fmt(min_unit,4)})","yellow"))
        return

    side = "sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_position_params_for_close())
        except Exception as e: print(colored(f"❌ partial close: {e}","red")); logging.error(f"close_partial error: {e}"); return

    px = price_now() or state["entry"]
    pnl=(px-state["entry"])*qty_close*(1 if state["side"]=="long" else -1)
    global compound_pnl
    compound_pnl += pnl

    set_state({"qty": safe_qty(state["qty"]-qty_close), "scale_outs": state["scale_outs"]+1})
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(state['qty'],4)}","magenta"))

    # Final chunk rule
    if state["qty"] <= FINAL_FULL_CLOSE_QTY and state["qty"]>0:
        print(colored(f"🔒 Final chunk ≤ {FINAL_FULL_CLOSE_QTY} — strict close", "cyan"))
        close_market_strict("FINAL_CHUNK_RULE")

def reset_after_full_close(reason):
    print(colored(f"🔚 CLOSE {reason} • CompoundPnL={fmt(compound_pnl)}", "magenta"))
    set_state({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "scale_outs": 0, "highest_profit_pct": 0.0,
        "tp1_done": False, "profit_idx": 0,
        "_tp_targets": None, "_tp_fracs": None,
        "await_next_candle": None
    })

# =============== Harvest / Management ===============
def build_tp_ladder(info: dict, ind: dict, side: str):
    """يُرجع (targets%, fracs) بنفس الطول — آمنة دائماً."""
    px = info.get("price"); atr = float(ind.get("atr") or 0.0)
    adx = float(ind.get("adx") or 0.0); rsi=float(ind.get("rsi") or 50.0)
    if not px or atr<=0: 
        targets = [0.6, 1.2, 2.0]
    else:
        atr_pct = (atr/max(px,1e-9))*100.0
        strong = adx>=30 and ((side=="long" and rsi>=55) or (side=="short" and rsi<=45))
        if strong:
            mults=[1.8, 3.0, 4.8, 6.0]
        else:
            mults=[1.4, 2.4, 4.0]
        targets = [round(m*atr_pct,2) for m in mults]
    targets = targets[:TP_LADDER_MAX]
    # fracs بنفس الطول
    base_fracs = [0.25,0.25,0.20,0.30]
    fracs = base_fracs[:len(targets)]
    return targets, fracs

def impulse_wick_harvest(df: pd.DataFrame, ind: dict):
    if not state["open"] or state["qty"]<=0: return
    o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
    h=float(df["high"].iloc[-1]);  l=float(df["low"].iloc[-1])
    body=abs(c-o); rng=max(h-l,1e-12)
    atr=float(ind.get("atr") or 0.0)
    if atr>0 and body >= IMPULSE_ATR_X*atr:
        close_partial(0.50, f"Impulse x{body/atr:.2f}")
    upper = h - max(o,c); lower = min(o,c) - l
    upper_pct=upper/rng*100; lower_pct=lower/rng*100
    if state["side"]=="long" and upper_pct>=WICK_PCT_FOR_HARVEST: close_partial(0.25, f"Upper wick {upper_pct:.0f}%")
    if state["side"]=="short" and lower_pct>=WICK_PCT_FOR_HARVEST: close_partial(0.25, f"Lower wick {lower_pct:.0f}%")

def ratchet_protect(ind: dict):
    if not state["open"] or state["qty"]<=0: return
    px = ind.get("price") or price_now() or state["entry"]
    e  = state["entry"]
    rr = (px - e)/e*100*(1 if state["side"]=="long" else -1)
    if rr > state["highest_profit_pct"]: state["highest_profit_pct"]=rr
    if (state["highest_profit_pct"]>=20 and rr < state["highest_profit_pct"]*(1-RATCHET_RETRACE)):
        close_partial(0.5, f"Ratchet {state['highest_profit_pct']:.1f}%→{rr:.1f}%")
        state["highest_profit_pct"]=rr

def dynamic_profit_taking(info: dict, ind: dict):
    if not state["open"] or state["qty"]<=0: return
    price = info.get("price") or price_now() or state["entry"]
    e = state["entry"]; side=state["side"]
    rr = (price - e)/e*100*(1 if side=="long" else -1)

    # توليد سُلّم آمن مرة كل دورة (يبقى في state)
    tps, fracs = state.get("_tp_targets"), state.get("_tp_fracs")
    if not tps or not fracs:
        tps, fracs = build_tp_ladder(info, ind, side)
        set_state({"_tp_targets": tps, "_tp_fracs": fracs})

    # TP1 + breakeven
    tp1 = max(TP1_PCT, tps[0] if tps else TP1_PCT)
    if (not state["tp1_done"]) and rr >= tp1:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{tp1:.2f}%")
        state["tp1_done"]=True
        if rr >= BREAKEVEN_AFTER: state["breakeven"]=e

    # سُلّم آمن (لا فهرسة خاطئة)
    k = int(state.get("profit_idx", 0))
    for i, (tgt, frac) in enumerate(list(zip(tps, fracs))[k:], start=k):
        if rr >= tgt and state["qty"]>0:
            close_partial(frac, f"TP_dyn@{tgt:.2f}%")
            state["profit_idx"] = i+1
        else:
            break

def atr_trail_enforce(ind: dict, info: dict):
    if not state["open"] or state["qty"]<=0: return
    price = info.get("price") or price_now() or state["entry"]
    e  = state["entry"]; side=state["side"]
    rr = (price - e)/e*100*(1 if side=="long" else -1)
    atr=float(ind.get("atr") or 0.0)
    if atr<=0: return
    if rr >= TRAIL_ACTIVATE:
        gap = atr * ATR_MULT_TRAIL
        if side=="long":
            new_trail = price - gap
            state["trail"] = max(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None: state["trail"] = max(state["trail"], state["breakeven"])
            if price < state["trail"]: close_market_strict("TRAIL_ATR")
        else:
            new_trail = price + gap
            state["trail"] = min(state["trail"] or new_trail, new_trail)
            if state["breakeven"] is not None: state["trail"] = min(state["trail"], state["breakeven"])
            if price > state["trail"]: close_market_strict("TRAIL_ATR")

# =============== Main Loop ===============
last_loop_ts = time.time()
def loop_heartbeat(): 
    global last_loop_ts; last_loop_ts=time.time()

def snapshot(df, info, ind, spread_bps, reason=None):
    left = time_to_candle_close(df)
    print(colored("─"*90, "cyan"))
    print(colored(f"{SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", "cyan"))
    print(f"💲 Price {fmt(info.get('price'))} | RF={fmt(info.get('filter'))} | spread_bps={fmt(spread_bps,2)} | close_in~{left}s")
    print(f"RSI={fmt(ind.get('rsi'))}  +DI={fmt(ind.get('plus_di'))}  -DI={fmt(ind.get('minus_di'))}  ADX={fmt(ind.get('adx'))}  ATR={fmt(ind.get('atr'))}")
    if state["open"]:
        print(f"📌 {('🟩 LONG' if state['side']=='long' else '🟥 SHORT')} Entry={fmt(state['entry'])} Qty={fmt(state['qty'],4)} Bars={state['bars']} PnL={fmt(state['pnl'])}")
        print(f"TPs={state.get('_tp_targets')} • idx={state.get('profit_idx')} • trail={fmt(state['trail'])} • tp1_done={state['tp1_done']}")
    else:
        print("⚪ FLAT")
    if state.get("await_next_candle"): print(colored("⏳ waiting next candle on swing-fail", "yellow"))
    if reason: print(colored(f"ℹ️ {reason}", "yellow"))
    eff = (balance_usdt() or 0.0) + compound_pnl
    print(f"CompoundPnL={fmt(compound_pnl)} • EffectiveEq≈{fmt(eff)} USDT")
    print(colored("─"*90, "cyan"))

def trade_loop():
    loop_counter=0
    while True:
        try:
            loop_heartbeat(); loop_counter+=1
            bal = balance_usdt()
            px  = price_now()
            df  = fetch_ohlcv()
            if len(df)==0:
                time.sleep(5); continue

            # update bars counter
            if "last_ts" not in state: state["last_ts"]=int(df["time"].iloc[-1])
            if int(df["time"].iloc[-1]) != state["last_ts"]:
                state["last_ts"]=int(df["time"].iloc[-1])
                if state["open"]: state["bars"]+=1
                # check swing-await once per new candle
                if state.get("await_next_candle"): pass

            # indicators & RF
            ind = compute_inds(df)
            df_closed = df.iloc[:-1] if len(df)>=2 else df.copy()
            sig_closed = rf_signal_closed(df_closed)
            filt_last = sig_closed.get("filter")
            sig_live = rf_signal_live(df, filt_last)
            spread_bps = orderbook_spread_bps()

            # PnL snapshot
            if state["open"] and px:
                state["pnl"] = (px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]

            # Update swings
            update_swings(df)

            # Entry
            reason=None
            if not state["open"]:
                if spread_bps is not None and spread_bps>MAX_SPREAD_BPS:
                    reason=f"spread too high ({fmt(spread_bps,2)}bps)"
                else:
                    want = None
                    if RF_LIVE and (sig_live["long"] or sig_live["short"]):
                        want = "buy" if sig_live["long"] else "sell"
                    elif sig_closed["long"] or sig_closed["short"]:
                        want = "buy" if sig_closed["long"] else "sell"

                    if want:
                        qty = compute_size(bal, px or sig_closed["price"])
                        if qty>0:
                            open_market(want, qty, px or sig_closed["price"])
                        else:
                            reason="qty<=0"
                    else:
                        reason="no signal"

            # Post-entry management
            if state["open"]:
                info_now = {"price": px or sig_closed["price"], "filter": filt_last}
                # swing fail logic (may announce waiting-next-candle)
                swing_fail_manager(df, ind, info_now["price"])
                # impulse/wick harvesting
                impulse_wick_harvest(df, ind)
                # dynamic TPs
                dynamic_profit_taking(info_now, ind)
                # ratchet & trail
                ratchet_protect(ind)
                atr_trail_enforce(ind, info_now)

            snapshot(df, {"price": px or sig_closed["price"], "filter": filt_last}, ind, spread_bps, reason)

            time.sleep(1 if time_to_candle_close(df)<=10 else 8)
        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}", "red"))
            logging.error(f"loop error: {e}\n{traceback.format_exc()}")
            time.sleep(5)

# =============== Flask / Keepalive ===============
from flask import Flask, jsonify
app = Flask(__name__)
import logging as flask_logging
flask_logging.getLogger("werkzeug").setLevel(flask_logging.ERROR)

@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ RF Bot — {SYMBOL} {INTERVAL} — {mode} — LIVE RF={RF_LIVE} — StrictClose — DynamicTP SafeZip — SwingWaitNextCandle"

@app.route("/metrics")
def metrics():
    s=get_state()
    return jsonify({
        "symbol":SYMBOL, "interval":INTERVAL, "mode":"live" if MODE_LIVE else "paper",
        "leverage":LEVERAGE, "risk_alloc":RISK_ALLOC, "lot_min":LOT_MIN, "lot_step":LOT_STEP,
        "open":s["open"], "side":s["side"], "qty":s["qty"], "entry":s["entry"],
        "pnl":s["pnl"], "bars":s["bars"], "tp_targets":s.get("_tp_targets"), "tp_fracs":s.get("_tp_fracs"),
        "await_next_candle":s.get("await_next_candle"),
        "swing_high":s.get("last_swing_high"), "swing_low":s.get("last_swing_low"),
        "compound_pnl":compound_pnl
    })

@app.route("/health")
def health():
    return jsonify({"ok":True, "loop_stall_s": time.time()-last_loop_ts, "mode":"live" if MODE_LIVE else "paper",
                    "strict_close":STRICT_EXCHANGE_CLOSE, "final_chunk_qty":FINAL_FULL_CLOSE_QTY}), 200

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive: SELF_URL not set — skip", "yellow")); return
    sess=requests.Session(); sess.headers.update({"User-Agent":"rf-pro-bot/keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}", "cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# =============== Boot ============================
if __name__=="__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x • RF_LIVE={RF_LIVE}", "yellow"))
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
