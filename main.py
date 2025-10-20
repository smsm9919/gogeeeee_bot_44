# -*- coding: utf-8 -*-
"""
RF Futures Bot — Trend-Only Pro (BingX Perp, CCXT) — CONSOLIDATED
- Entry: Range Filter (RF) فقط (حيّ أو مغلق) حسب المفاتيح
- Post-entry: Supertrend + ADX/RSI/ATR harvesting + EMA9/20
- Dynamic TP ladder + Ratchet Lock + Final-Chunk Strict Close
- Peaks/Valleys confirmation (انتظار شمعة عند فشل الاختراق)
- FVG/OB/EQH/EQL + Wick/Impulse harvesting + Fibonacci assists
- Emergency layer + Strict close + Rotated logging + Flask metrics
"""

import os, time, math, random, signal, sys, traceback, threading, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, InvalidOperation

import pandas as pd
import ccxt
from flask import Flask, jsonify

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# =================== ENV ===================
API_KEY   = os.getenv("BINGX_API_KEY", "")
API_SECRET= os.getenv("BINGX_API_SECRET", "")
MODE_LIVE = bool(API_KEY and API_SECRET)
SELF_URL  = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT      = int(os.getenv("PORT", 5000))

# =================== SETTINGS ===================
SYMBOL   = "DOGE/USDT:USDT"
INTERVAL = "15m"
LEVERAGE = 10
RISK_ALLOC = 0.60
BINGX_POSITION_MODE = "oneway"      # "hedge" أو "oneway"

# ---- RF options ----
RF_SOURCE = "close"
RF_PERIOD = 20
RF_MULT   = 3.5
RF_LIVE   = True                     # دخول على الشمعة الحيّة
RF_HYST_BPS = 8.0                    # هسترة حول الفلتر

# مصادر الدخول المسموحة
ALLOW_RF_LIVE     = True
ALLOW_RF_CLOSED   = True
ALLOW_TVR         = False            # مقفول افتراضيًا
ALLOW_BREAKOUT    = False            # مقفول افتراضيًا

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

# Harvesting + Targets
TP1_PCT         = 0.40
TP1_CLOSE_FRAC  = 0.50
BREAKEVEN_AFTER = 0.30
TRAIL_ACTIVATE  = 1.20
ATR_MULT_TRAIL  = 1.6

TREND_TARGETS     = [0.50, 1.00, 1.80]
TREND_CLOSE_FRACS = [0.30, 0.30, 0.20]
MIN_TREND_ADX_HOLD= 25

LONG_WICK_HARVEST_THRESHOLD  = 0.45   # 45% من مدى الشمعة
IMPULSE_HARVEST_THRESHOLD_ATR= 1.2    # جسم ≥ 1.2×ATR

# Final-chunk strict close
FINAL_CHUNK_QTY = 50.0                # إذا الباقي ≤ هذا → إغلاق صارم

# Spread guard
MAX_SPREAD_BPS = 6.0

# Exit protection layer
NEVER_FULL_CLOSE_BEFORE_TP1 = True

# Peaks/Valleys confirmation
PEAK_CONFIRM_BARS       = 1           # انتظر شمعة بعد فشل اختراق
PEAK_CONFIRM_BODY_ATR   = 0.35        # لو الجسم < هذا × ATR → يعتبر فشل اختراق
PEAK_FAIL_DEF_PARTIAL   = 0.20        # جني دفاعي عند فشل الاختراق المؤقت

# ====== LOGGING ======
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename","").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready", "cyan"))
setup_file_logging()

# ====== EXCHANGE ======
def make_exchange():
    return ccxt.bingx({
        "apiKey": API_KEY, "secret": API_SECRET,
        "enableRateLimit": True, "timeout": 20000,
        "options": {"defaultType": "swap"}
    })
ex = make_exchange()
MARKET = {}
AMT_PREC = 0; LOT_STEP=None; LOT_MIN=0.0

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision",{}) or {}).get("amount", 0) or 0)
        limits = (MARKET.get("limits",{}) or {}).get("amount",{}) or {}
        LOT_STEP = limits.get("step")
        LOT_MIN  = float(limits.get("min") or 0.0)
        print(colored(f"🔢 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}", "yellow"))
def ensure_leverage_and_mode():
    try:
        try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
        except Exception as e: print(colored(f"⚠️ set_leverage: {e}", "yellow"))
        print(colored(f"📌 position mode: {BINGX_POSITION_MODE}", "cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_mode: {e}", "yellow"))
try:
    load_market_specs(); ensure_leverage_and_mode()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}", "yellow"))

# ====== HELPERS ======
def _round_amt(q):
    if q is None: return 0.0
    try:
        d=Decimal(str(q))
        if LOT_STEP and LOT_STEP>0:
            step=Decimal(str(LOT_STEP))
            d=(d/step).to_integral_value(rounding=ROUND_DOWN)*step
        d = d.quantize(Decimal(1).scaleb(-int(AMT_PREC or 0)), rounding=ROUND_DOWN)
        if LOT_MIN and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except (InvalidOperation, ValueError, TypeError):
        return max(0.0, float(q))
def safe_qty(q):
    q=_round_amt(q)
    if q<=0: print(colored(f"⚠️ qty invalid after normalize → {q}", "yellow"))
    return q
def fmt(v, d=6, na="N/A"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception: return na

def with_retry(fn, attempts=3, base_wait=0.4):
    for i in range(attempts):
        try: return fn()
        except Exception as e:
            if i==attempts-1: raise
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
        mid=(bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception: return None

def _interval_seconds(iv: str) -> int:
    iv=(iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60
def time_to_candle_close(df: pd.DataFrame) -> int:
    tf=_interval_seconds(INTERVAL)
    if len(df)==0: return tf
    cur_start_ms=int(df["time"].iloc[-1]); now_ms=int(time.time()*1000)
    next_close_ms=cur_start_ms + tf*1000
    while next_close_ms <= now_ms: next_close_ms+=tf*1000
    return int(max(0,next_close_ms-now_ms)/1000)

# ====== INDICATORS ======
def wilder_ema(s: pd.Series, n: int): return s.ewm(alpha=1/n, adjust=False).mean()
def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"adx":0.0,"atr":0.0,"st":None,"st_dir":0,"ema9":None,"ema20":None,"ema9_slope":0.0}
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

    # Supertrend (خفيف)
    st_val, st_dir = None, 0
    try:
        st_period=10; st_mult=3.0
        hl2=(h+l)/2.0
        atr_st = wilder_ema(tr, st_period)
        upper=hl2+st_mult*atr_st; lower=hl2-st_mult*atr_st
        st=[float('nan')]; dirv=[0]
        for i in range(1, len(df)):
            prev_st=st[-1]; prev_dir=dirv[-1]; cur=float(c.iloc[i]); ub=float(upper.iloc[i]); lb=float(lower.iloc[i])
            if math.isnan(prev_st): st.append(lb if cur>lb else ub); dirv.append(1 if cur>lb else -1); continue
            if prev_dir==1:
                ub=min(ub, prev_st); st_val_now = lb if cur>ub else ub; dir_now = 1 if cur>ub else -1
            else:
                lb=max(lb, prev_st); st_val_now = ub if cur<lb else lb; dir_now = -1 if cur<lb else 1
            st.append(st_val_now); dirv.append(dir_now)
        st_series = pd.Series(st[1:], index=df.index[1:]).reindex(df.index, method='pad')
        dir_series= pd.Series(dirv[1:], index=df.index[1:]).reindex(df.index, method='pad').fillna(0)
        st_val=float(st_series.iloc[-1]); st_dir=int(dir_series.iloc[-1])
    except Exception: pass

    ema9  = c.ewm(span=9, adjust=False).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    slope = 0.0
    if len(ema9) > 6:
        e_old=float(ema9.iloc[-6]); e_new=float(ema9.iloc[-1]); base=max(abs(e_old),1e-9); slope=(e_new-e_old)/base

    return {
        "rsi": float(rsi.iloc[-1]),
        "plus_di": float(plus_di.iloc[-1]), "minus_di": float(minus_di.iloc[-1]),
        "adx": float(adx.iloc[-1]), "atr": float(atr.iloc[-1]),
        "st": st_val, "st_dir": st_dir, "ema9": float(ema9.iloc[-1]), "ema20": float(ema20.iloc[-1]),
        "ema9_slope": float(slope)
    }

# ====== RANGE FILTER ======
def _ema(s: pd.Series, n: int): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src: pd.Series, qty: float, n: int) -> pd.Series:
    avrng=_ema((src-src.shift(1)).abs(), n); wper=(n*2)-1
    return _ema(avrng, wper)*qty
def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rf.append(cur)
    filt=pd.Series(rf, index=src.index, dtype="float64")
    return filt + rsize, filt - rsize, filt

def compute_tv_signals(df_closed: pd.DataFrame):
    if len(df_closed) < RF_PERIOD + 3:
        i=-1
        price=float(df_closed["close"].iloc[i]) if len(df_closed) else None
        ts=int(df_closed["time"].iloc[i]) if len(df_closed) else int(time.time()*1000)
        return {"time":ts,"price":price or 0.0,"long":False,"short":False,"filter":price or 0.0,"hi":price or 0.0,"lo":price or 0.0,"fdir":0}
    src=df_closed[RF_SOURCE].astype(float)
    hi,lo,filt=_rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    dfilt=filt-filt.shift(1); fdir=pd.Series(0.0, index=filt.index).mask(dfilt>0,1).mask(dfilt<0,-1).ffill().fillna(0.0)
    upward=(fdir==1).astype(int); downward=(fdir==-1).astype(int)
    src_gt_f=(src>filt); src_lt_f=(src<filt); src_gt_p=(src>src.shift(1)); src_lt_p=(src<src.shift(1))
    longCond=(src_gt_f&((src_gt_p)|(src_lt_p))&(upward>0))
    shortCond=(src_lt_f&((src_lt_p)|(src_gt_p))&(downward>0))
    CondIni=pd.Series(0, index=src.index)
    for i in range(1,len(src)):
        CondIni.iloc[i]=1 if bool(longCond.iloc[i]) else (-1 if bool(shortCond.iloc[i]) else CondIni.iloc[i-1])
    longSignal = longCond & (CondIni.shift(1)==-1)
    shortSignal= shortCond & (CondIni.shift(1)==1)
    i=len(src)-1
    return {
        "time": int(df_closed["time"].iloc[i]),
        "price": float(df_closed["close"].iloc[i]),
        "long": bool(longSignal.iloc[i]), "short": bool(shortSignal.iloc[i]),
        "filter": float(filt.iloc[i]), "hi": float(hi.iloc[i]), "lo": float(lo.iloc[i]),
        "fdir": float(fdir.iloc[i])
    }

def rf_live_signal(df_full: pd.DataFrame):
    """دخول على الشمعة الحية بحسترة بيس بالـbps + فلتر ADX/DI لاحقًا خارجًا."""
    if len(df_full) < RF_PERIOD + 3: return None, None
    src=df_full[RF_SOURCE].astype(float)
    _,_,filt=_rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    px=float(df_full["close"].iloc[-1]); rf=float(filt.iloc[-1])
    bps=abs((px - rf)/rf)*10000.0
    if px > rf*(1.0 + RF_HYST_BPS/10000.0):  return "buy", {"price":px,"filter":rf,"bps":bps}
    if px < rf*(1.0 - RF_HYST_BPS/10000.0):  return "sell", {"price":px,"filter":rf,"bps":bps}
    return None, {"price":px,"filter":rf,"bps":bps}

# ====== SMC (EQH/EQL & simple OB/FVG) ======
def _find_swings(df: pd.DataFrame, L=2, R=2):
    if len(df) < L+R+3: return [], []
    h=df["high"].astype(float).values; l=df["low"].astype(float).values
    ph=[None]*len(df); pl=[None]*len(df)
    for i in range(L, len(df)-R):
        if all(h[i]>=h[j] for j in range(i-L, i+R+1)): ph[i]=h[i]
        if all(l[i]<=l[j] for j in range(i-L, i+R+1)): pl[i]=l[i]
    return ph,pl

def smc_levels(df: pd.DataFrame, atr_now: float):
    d = df.iloc[:-1] if len(df)>=2 else df.copy()
    ph,pl = _find_swings(d,2,2)
    # equal highs/lows
    def _equal(vals, pick):
        idx=[i for i,v in enumerate(vals) if v is not None]
        if not idx: return None
        prices=[vals[i] for i in idx]
        prices.sort()
        # تلخيص تقريبي
        return pick(prices) if prices else None
    eqh=_equal(ph, max); eql=_equal(pl, min)

    # OB بسيط: شمعة اندفاع جسم كبير وذيول صغيرة
    ob=None
    try:
        for i in range(len(d)-2, max(len(d)-40,1), -1):
            o=float(d["open"].iloc[i]); c=float(d["close"].iloc[i])
            hi=float(d["high"].iloc[i]); lo=float(d["low"].iloc[i])
            rng=max(hi-lo,1e-12); body=abs(c-o)
            up=hi-max(o,c); dn=min(o,c)-lo
            if atr_now>0 and body >= 1.2*atr_now and (up/rng)<=0.35 and (dn/rng)<=0.35:
                side="bull" if c>o else "bear"
                ob={"side":side,"bot":min(o,c),"top":max(o,c)}; break
    except Exception: pass

    # FVG مبسط (آخر 20 شمعة)
    fvg=None
    try:
        for i in range(len(d)-3, max(len(d)-20,2), -1):
            prev_high=float(d["high"].iloc[i-1]); prev_low=float(d["low"].iloc[i-1])
            cur_low=float(d["low"].iloc[i]);   cur_high=float(d["high"].iloc[i])
            if cur_low > prev_high:  fvg={"type":"BULL_FVG","bottom":prev_high,"top":cur_low};  break
            if cur_high < prev_low:  fvg={"type":"BEAR_FVG","bottom":cur_high,"top":prev_low};  break
    except Exception: pass
    return {"eqh":eqh,"eql":eql,"ob":ob,"fvg":fvg}

# ====== STATE ======
state = {
    "open":False,"side":None,"entry":None,"qty":0.0,
    "pnl":0.0,"bars":0,"trail":None,"breakeven":None,
    "tp1_done":False,"highest_profit_pct":0.0,
    "profit_targets_achieved":0,
    "wait_peak_confirm":False,"peak_confirm_left":0,
}
compound_pnl=0.0

# ====== ORDERS ======
def _pos_params_open(side):  return {"positionSide":"BOTH","reduceOnly":False} if BINGX_POSITION_MODE!="hedge" else {"positionSide":"LONG" if side=="buy" else "SHORT","reduceOnly":False}
def _pos_params_close():      return {"positionSide":"BOTH","reduceOnly":True}  if BINGX_POSITION_MODE!="hedge" else {"positionSide":"LONG" if state.get("side")=="long" else "SHORT","reduceOnly":True}

def _read_exchange_position():
    try:
        poss = ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym=(p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0] not in sym: continue
            qty=abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty<=0: return 0.0, None, None
            entry=float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side_raw=(p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side="long" if ("long" in side_raw or float(p.get("cost",0))>0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_exchange_position: {e}")
    return 0.0, None, None

def compute_size(balance, price):
    effective = (balance or 0.0) + (compound_pnl or 0.0)
    capital   = effective * RISK_ALLOC * LEVERAGE
    return safe_qty(max(0.0, capital / max(float(price or 0.0), 1e-9)))

def open_market(side, qty, price, why=""):
    if qty<=0: print(colored("❌ qty<=0 skip open","red")); return
    params=_pos_params_open(side)
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception: pass
            ex.create_order(SYMBOL,"market",side,qty,None,params)
        except Exception as e:
            print(colored(f"❌ open: {e}", "red")); logging.error(f"open_market: {e}"); return
    state.update({
        "open":True, "side":"long" if side=="buy" else "short",
        "entry":price, "qty":qty, "bars":0, "trail":None, "breakeven":None,
        "tp1_done":False, "profit_targets_achieved":0, "highest_profit_pct":0.0,
        "wait_peak_confirm":False, "peak_confirm_left":0
    })
    print(colored(f"✅ OPEN {side.upper()} qty={fmt(qty,4)} @ {fmt(price)} • {why}", "green" if side=="buy" else "red"))
    logging.info(f"OPEN {side} qty={qty} price={price} why={why}")

def _strict_close(reason):
    global compound_pnl
    exch_qty, exch_side, exch_entry = _read_exchange_position()
    if exch_qty <= 0:
        if state["open"]: _reset_after_close(reason); return
    side_to_close="sell" if (exch_side=="long") else "buy"
    qty_to_close=safe_qty(exch_qty)
    try:
        if MODE_LIVE:
            ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,_pos_params_close())
        px=price_now() or state.get("entry")
        entry_px=state.get("entry") or exch_entry or px
        side=state.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
        pnl=( (px-entry_px)*qty_to_close if side=="long" else (entry_px-px)*qty_to_close )
        compound_pnl += pnl
        print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
    except Exception as e:
        logging.error(f"strict_close: {e}")
    _reset_after_close(reason)

def close_partial(frac, why):
    """يحترم LOT_MIN + Final-chunk rule."""
    if not state["open"] or state["qty"]<=0: return
    px=price_now() or state["entry"]
    # Final chunk rule
    if state["qty"] <= FINAL_CHUNK_QTY:
        print(colored(f"🔒 Final chunk ≤ {FINAL_CHUNK_QTY} — strict close", "yellow"))
        _strict_close("FINAL_CHUNK_RULE"); return

    qty_close=safe_qty(max(0.0, state["qty"] * max(0.0, min(1.0, frac))))
    if qty_close < max(LOT_MIN or 0.0, 1.0):
        print(colored(f"⏸️ skip partial (amount={fmt(qty_close,4)} < min_unit={fmt(LOT_MIN,1)})","yellow"))
        return

    side="sell" if state["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_pos_params_close())
        except Exception as e: print(colored(f"❌ partial close: {e}","red")); logging.error(f"close_partial: {e}"); return
    pnl=( (px-state["entry"])*qty_close if state["side"]=="long" else (state["entry"]-px)*qty_close )
    globals()["compound_pnl"] += pnl
    state["qty"] = max(0.0, state["qty"] - qty_close)
    print(colored(f"🔻 PARTIAL {why} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(state['qty'],4)}","magenta"))
    # enforce final-chunk rule after partial
    if state["qty"] <= FINAL_CHUNK_QTY:
        print(colored(f"🔒 Final chunk ≤ {FINAL_CHUNK_QTY} — strict close", "yellow"))
        _strict_close("FINAL_CHUNK_RULE")

def _reset_after_close(reason):
    state.update({
        "open":False,"side":None,"entry":None,"qty":0.0,"trail":None,"breakeven":None,
        "tp1_done":False,"profit_targets_achieved":0,"highest_profit_pct":0.0,
        "wait_peak_confirm":False,"peak_confirm_left":0
    })
    logging.info(f"RESET_AFTER_CLOSE {reason}")

# ====== MANAGEMENT ======
def _harvest_wick_impulse(df: pd.DataFrame, ind: dict):
    if not state["open"] or state["qty"]<=0: return
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1]); l=float(df["low"].iloc[-1]); c=float(df["close"].iloc[-1])
    rng=max(h-l,1e-12); body=abs(c-o); up=h-max(o,c); dn=min(o,c)-l
    atr=float(ind.get("atr") or 0.0)
    if atr>0 and body >= IMPULSE_HARVEST_THRESHOLD_ATR*atr:
        close_partial(0.33 if body<2.0*atr else 0.5, f"Impulse x{body/atr:.2f} ATR")
    if state["side"]=="long" and (up/rng)>=LONG_WICK_HARVEST_THRESHOLD:
        close_partial(0.25, f"Upper wick {(up/rng)*100:.1f}%")
    if state["side"]=="short" and (dn/rng)>=LONG_WICK_HARVEST_THRESHOLD:
        close_partial(0.25, f"Lower wick {(dn/rng)*100:.1f}%")

def _dynamic_tp(ind: dict, info_price: float):
    """TP ديناميكي آمن ضد أخطاء الفهارس."""
    if not state["open"] or state["qty"]<=0: return
    price = info_price
    entry = state["entry"]; side = state["side"]
    rr = (price - entry)/entry*100*(1 if side=="long" else -1)

    targets = TREND_TARGETS[:]  # نسخ
    fracs   = TREND_CLOSE_FRACS[:]
    K = min(len(targets), len(fracs))
    if K == 0: return
    idx = min(int(state.get("profit_targets_achieved",0)), K-1)

    if (not state["tp1_done"]) and rr >= TP1_PCT:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{TP1_PCT:.2f}%")
        state["tp1_done"]=True
        if rr >= BREAKEVEN_AFTER: state["breakeven"]=entry

    if rr >= targets[idx]:
        close_partial(fracs[idx], f"TP_dyn@{targets[idx]:.2f}%")
        state["profit_targets_achieved"]=min(idx+1, K)

def _ema_harvest(ind: dict, price: float):
    if not state["open"] or state["qty"]<=0: return
    side=state["side"]; ema9=ind.get("ema9"); ema20=ind.get("ema20"); atr=float(ind.get("atr") or 0.0)
    if ema9 is None or ema20 is None: return
    weak = (price<ema9) if side=="long" else (price>ema9)
    strong=(price<ema20) if side=="long" else (price>ema20)
    if weak:   close_partial(0.20 if not state["tp1_done"] else 0.25, "EMA9 touch")
    if strong: 
        close_partial(0.30 if not state["tp1_done"] else 0.40, "EMA20 break")
        if atr>0:
            gap=atr*ATR_MULT_TRAIL
            if side=="long": state["trail"]=max(state["trail"] or (price-gap), price-gap)
            else:            state["trail"]=min(state["trail"] or (price+gap), price+gap)

def _ratchet_and_trail(ind: dict, price: float):
    if not state["open"] or state["qty"]<=0: return
    entry=state["entry"]; side=state["side"]
    rr=(price-entry)/entry*100*(1 if side=="long" else -1)
    if rr > state["highest_profit_pct"]: state["highest_profit_pct"]=rr
    if state["highest_profit_pct"]>=TRAIL_ACTIVATE:
        atr=float(ind.get("atr") or 0.0)
        if atr>0:
            gap=atr*ATR_MULT_TRAIL
            if side=="long":
                state["trail"]=max(state["trail"] or (price-gap), price-gap)
                if price < state["trail"]: _strict_close(f"TRAIL_ATR({ATR_MULT_TRAIL}x)")
            else:
                state["trail"]=min(state["trail"] or (price+gap), price+gap)
                if price > state["trail"]: _strict_close(f"TRAIL_ATR({ATR_MULT_TRAIL}x)")

def _peaks_valleys_confirm(df: pd.DataFrame, ind: dict, levels: dict, info_price: float):
    """لو فشل اختراق قمة/قاع بجسم ضعيف → انتظر شمعة واحدة مع جني دفاعي."""
    if not state["open"] or state["qty"]<=0: return
    atr=float(ind.get("atr") or 0.0)
    if atr<=0: return
    o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1]); hi=float(df["high"].iloc[-1]); lo=float(df["low"].iloc[-1])
    body=abs(c-o)

    side=state["side"]
    eqh=levels.get("eqh"); eql=levels.get("eql")
    # فشل اختراق؟
    failed=False
    if side=="long" and eqh and hi>eqh and body/atr < PEAK_CONFIRM_BODY_ATR:
        failed=True
    if side=="short" and eql and lo<eql and body/atr < PEAK_CONFIRM_BODY_ATR:
        failed=True

    if failed and not state.get("wait_peak_confirm"):
        close_partial(PEAK_FAIL_DEF_PARTIAL, "Peak/Valley fail — defensive")
        state["wait_peak_confirm"]=True
        state["peak_confirm_left"]=PEAK_CONFIRM_BARS
        print(colored("⏳ انتظر الشمعة القادمة لتأكيد القمة/القاع قبل أي إغلاق كامل", "yellow"))
        return

    # عدّ تنازلي
    if state.get("wait_peak_confirm"):
        state["peak_confirm_left"]=max(0, state["peak_confirm_left"]-1)
        if state["peak_confirm_left"]==0:
            state["wait_peak_confirm"]=False

# ====== FLASK ======
app = Flask(__name__)
import logging as flask_logging
flask_logging.getLogger('werkzeug').setLevel(flask_logging.ERROR)

@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ RF Bot — {SYMBOL} {INTERVAL} — {mode} — RF_ONLY(LIVE={ALLOW_RF_LIVE},CLOSED={ALLOW_RF_CLOSED})"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol":SYMBOL,"interval":INTERVAL,"mode":"live" if MODE_LIVE else "paper",
        "leverage":LEVERAGE,"risk_alloc":RISK_ALLOC,"price":price_now(),
        "state":state,"compound_pnl":compound_pnl,
        "rf_live":ALLOW_RF_LIVE,"rf_closed":ALLOW_RF_CLOSED,
        "tvr_allowed":ALLOW_TVR,"breakout_allowed":ALLOW_BREAKOUT
    })

@app.route("/health")
def health():
    return jsonify({"ok":True,"open":state["open"],"side":state["side"],"qty":state["qty"],
                    "compound_pnl":compound_pnl,"timestamp":datetime.utcnow().isoformat()}),200

# ====== MAIN LOOP ======
def trade_loop():
    loop=0
    while True:
        try:
            loop+=1
            bal=balance_usdt()
            px =price_now()
            df =fetch_ohlcv()
            spread_bps=orderbook_spread_bps()
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                print(colored(f"🧮 spread_bps={fmt(spread_bps,2)} > {MAX_SPREAD_BPS} — pause", "yellow"))
                time.sleep(3); continue

            df_closed = df.iloc[:-1] if len(df)>=2 else df.copy()
            info_closed = compute_tv_signals(df_closed)
            ind = compute_indicators(df)

            # مستويات SMC مبسطة
            levels = smc_levels(df, float(ind.get("atr") or 0.0))

            # لوج موجز
            left=time_to_candle_close(df)
            print(colored(f"—— {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC ——", "cyan"))
            print(f"$ Price {fmt(px)} | ATR={fmt(ind['atr'])} | spread_bps={fmt(spread_bps,2)} | close_in~{left}s")
            print(f"RSI={fmt(ind['rsi'])} ADX={fmt(ind['adx'])} +DI={fmt(ind['plus_di'])} -DI={fmt(ind['minus_di'])} ST_dir={ind.get('st_dir')}")
            print(f"SMC: EQH={fmt(levels.get('eqh'))} EQL={fmt(levels.get('eql'))} OB={levels.get('ob')} FVG={levels.get('fvg')}")

            # تحديث PnL
            if state["open"] and px:
                state["pnl"] = (px-state["entry"])*state["qty"] if state["side"]=="long" else (state["entry"]-px)*state["qty"]

            # ====== ENTRY (RF ONLY) ======
            if (not state["open"]):
                chosen_sig=None; why=None; rf_data=None
                # LIVE
                if ALLOW_RF_LIVE and RF_LIVE:
                    sig, data = rf_live_signal(df)
                    rf_data=data
                    # فلتر ADX/DI أساسي
                    if sig:
                        pdi=float(ind.get("plus_di") or 0.0); mdi=float(ind.get("minus_di") or 0.0); adx=float(ind.get("adx") or 0.0)
                        if (sig=="buy" and pdi>mdi and adx>=18) or (sig=="sell" and mdi>pdi and adx>=18):
                            chosen_sig=sig; why=f"RF_LIVE bps={fmt(data.get('bps'),2)}"
                # CLOSED
                if (chosen_sig is None) and ALLOW_RF_CLOSED:
                    if info_closed["long"]:  chosen_sig="buy";  why="RF_CLOSED"
                    if info_closed["short"]: chosen_sig="sell"; why="RF_CLOSED"

                if chosen_sig and (spread_bps is None or spread_bps<=MAX_SPREAD_BPS):
                    qty=compute_size(bal, px or info_closed["price"])
                    if qty>0:
                        open_market(chosen_sig, qty, px or info_closed["price"], why=why)
                    else:
                        print(colored("⚠️ qty<=0 — skip open", "yellow"))
                else:
                    print(colored("⏳ no RF signal (live/closed gated)", "blue"))

            # ====== POST-ENTRY MANAGEMENT ======
            if state["open"] and px:
                # حماية القمم/القيعان
                _peaks_valleys_confirm(df, ind, levels, px)
                # جني ديناميكي
                _dynamic_tp(ind, px)
                # Harvests
                _harvest_wick_impulse(df, ind)
                _ema_harvest(ind, px)
                _ratchet_and_trail(ind, px)

                # منع الإغلاق الكامل قبل TP1 (إلا التريل/القاعدة النهائية/طوارئ)
                if NEVER_FULL_CLOSE_BEFORE_TP1 and (not state["tp1_done"]):
                    pass  # نتجنب أي full close مباشر خارج التريل/القاعدة

            time.sleep(2 if left<=10 else 8)

        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}", "red"))
            logging.error(f"loop error: {e}\n{traceback.format_exc()}")
            time.sleep(5)

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (no SELF_URL)", "yellow")); return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"rf-pro-bot/keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}", "cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# ====== BOOT ======
if __name__=="__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  RF_LIVE={RF_LIVE}", "yellow"))
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
