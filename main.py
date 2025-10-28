# -*- coding: utf-8 -*-
"""
DOGE/USDT — BingX Perp via CCXT
Pro SMC Council + Peak/Bottom Hunter + Bookmap-lite + FVG + Accum + Trend + Auto-Flip + Chop
Low-Fee Execution (one-shot exits) + RF(Closed) fallback + State Persistence

HTTP: / , /metrics , /health
"""

import os, time, math, random, signal, sys, traceback, logging, threading, json
from logging.handlers import RotatingFileHandler
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from collections import deque

import pandas as pd
import ccxt
from flask import Flask, jsonify

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# ================== KEYS / MODE ==================
API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)
SELF_URL   = (os.getenv("SELF_URL","") or os.getenv("RENDER_EXTERNAL_URL","")).strip()
PORT       = 5000

# ================== STRATEGY SETTINGS ============
SYMBOL        = "DOGE/USDT:USDT"
INTERVAL      = "15m"
LEVERAGE      = 10
RISK_ALLOC    = 0.60
POSITION_MODE = "oneway"   # أو "hedge"

# Range Filter (Closed) — fallback فقط
RF_SOURCE   = "close"
RF_PERIOD   = 20
RF_MULT     = 3.5
RF_HYST_BPS = 6.0
ENTRY_FROM_RF = True

# مؤشرات
RSI_LEN = 14; ADX_LEN = 14; ATR_LEN = 14

# حُرّاس
MAX_SPREAD_BPS      = 8.0
PAUSE_ADX_THRESHOLD = 17.0  # طلبك
WAIT_NEXT_CLOSED    = True

# Trend
TREND_ADX_MIN   = 30.0
STRUCT_BARS     = 48

# SCM / Liquidity
EQ_BPS             = 10.0
SWEEP_LOOKBACK     = 60
DSP_ATR_MIN        = 1.2
RETEST_BPS         = 15.0

# === Trend-aware riding guards ===
TREND_LOCK_ADX            = 30.0
TREND_LOCK_MIN_BARS       = 6
TREND_LOCK_REQUIRE_DI     = True
STRICT_CLOSE_NEEDS_BOS    = True
EXTREME_CLOSE_REQUIRE_TREND_MISMATCH = True

# === Low-fee execution policy ===
LOW_FEE_MODE          = True
MAX_EXITS_PER_TRADE   = 1
ALLOW_PARTIALS_IN_LOW = False

# إدارة (لو عطّلت Low-Fee ترجع السلالم)
TP1_PCT_BASE       = 0.40
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.6
RATCHET_LOCK_FALLBACK = 0.60
FINAL_CHUNK_QTY    = 50.0
RESIDUAL_MIN_QTY   = 9.0
STRICT_CLOSE_DROP_FROM_HP = 0.50
STRICT_COOL_ADX           = 20.0

# Auto-Flip With Trend
FLIP_RF_BARS_CONFIRMED = 2
FLIP_MIN_ADX           = 25.0
FLIP_REQUIRE_DI        = True
FLIP_REQUIRE_BOS       = True

# === Chop / Range detection ===
CHOP_ADX_MAX          = 18.0
CHOP_BB_WIDTH_MAX_PCT = 1.2
CHOP_MIN_BARS         = 20
CHOP_EQ_TOL_BPS       = 12.0
CHOP_EXIT_PROFIT_PCT  = 0.35
CHOP_HOLD_OFF_BARS    = 12
CHOP_MUST_BREAK_BAND  = True

# Council strength
COUNCIL_MIN_STRONG_SCORE   = 3.5
WEAK_DECISION_STRICT_EXIT  = True

# Wick/Impulse grading
WICK_MIN_RATIO            = 0.55
WICK_EXTREME_RATIO        = 0.70
IMPULSE_ATR_MULT          = 2.2
IMPULSE_EXTREME_ATR_MULT  = 3.0
RUNTIME_LOG = True

# Pacing
BASE_SLEEP=5; NEAR_CLOSE_S=1

# ===== Peak/Bottom Hunter =====
PEAK_MIN_SWING_BARS       = 5
PEAK_WICK_RATIO_MIN       = 0.55
PEAK_BODY_ATR_MIN         = 1.0
PEAK_VOL_MULT             = 1.6
PEAK_RSI_EXTREME_HIGH     = 70.0
PEAK_RSI_EXTREME_LOW      = 30.0
PEAK_DSP_ATR_MIN          = 1.3
PEAK_NEAR_WALL_BPS        = 20.0
PEAK_CONFIRM_MIN_SCORE    = 2.5
PEAK_RUNTIME_LOCK_PCT     = 0.60
PEAK_STRICT_IF_BOS        = True

# === Bookmap-lite (orderbook walls) ===
BM_DEPTH_LEVELS       = 50
BM_SNAPSHOTS          = 60
BM_WALL_QTL           = 0.85
BM_NEAR_BPS           = 20.0
BM_MIN_NOTIONAL       = 50.0

# ================== LOGGING =======================
def setup_logs():
    logger=logging.getLogger(); logger.setLevel(logging.INFO)
    if not any(isinstance(h,RotatingFileHandler) and getattr(h,"baseFilename","").endswith("bot.log") for h in logger.handlers):
        fh=RotatingFileHandler("bot.log",maxBytes=5_000_000,backupCount=7,encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready","cyan"))
setup_logs()

# ================== EXCHANGE ======================
def make_ex():
    return ccxt.bingx({
        "apiKey": API_KEY, "secret": API_SECRET,
        "enableRateLimit": True, "timeout": 20000,
        "options": {"defaultType":"swap"}
    })
ex = make_ex()
EX_LOCK = threading.Lock()
def _with_ex(fn):
    with EX_LOCK: return fn()

STATE_FILE = "state.json"

MARKET={}; AMT_PREC=0; LOT_STEP=None; LOT_MIN=None
def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        _with_ex(lambda: ex.load_markets())
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision",{}) or {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits",{}) or {}).get("amount",{}).get("step", None)
        LOT_MIN  = (MARKET.get("limits",{}) or {}).get("amount",{}).get("min",  None)
        print(colored(f"🔧 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}","cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}","yellow"))
def ensure_leverage_mode():
    try:
        try:
            _with_ex(lambda: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"}))
            print(colored(f"✅ leverage set: {LEVERAGE}x","green"))
        except Exception as e:
            print(colored(f"⚠️ set_leverage warn: {e}","yellow"))
        print(colored(f"📌 position mode: {POSITION_MODE}","cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_leverage_mode: {e}","yellow"))
try:
    load_market_specs(); ensure_leverage_mode()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}","yellow"))

# ================== HELPERS =======================
def with_retry(fn, tries=3, base=0.4):
    for i in range(tries):
        try: return fn()
        except Exception:
            if i==tries-1: raise
            time.sleep(base*(2**i)+random.random()*0.25)

def _round_amt(q):
    try:
        d=Decimal(str(q))
        if LOT_STEP and LOT_STEP>0:
            step=Decimal(str(LOT_STEP))
            d=(d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec=int(AMT_PREC) if AMT_PREC>=0 else 0
        d=d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except Exception: return max(0.0,float(q or 0.0))
def safe_qty(q):
    q=_round_amt(q)
    if q<=0: logging.warning(f"qty invalid after normalize: {q}")
    return q
def fmt(x,d=6): 
    try: return f"{float(x):.{d}f}"
    except Exception: return "—"

def fetch_ohlcv(limit=600):
    rows=with_retry(lambda: _with_ex(lambda: ex.fetch_ohlcv(SYMBOL,timeframe=INTERVAL,limit=limit,params={"type":"swap"})))
    return pd.DataFrame(rows,columns=["time","open","high","low","close","volume"])
def price_now():
    try:
        t=with_retry(lambda: _with_ex(lambda: ex.fetch_ticker(SYMBOL)))
        return t.get("last") or t.get("close")
    except Exception: return None
def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b=with_retry(lambda: _with_ex(lambda: ex.fetch_balance(params={"type":"swap"})))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None
def orderbook_spread_bps():
    try:
        ob=with_retry(lambda: _with_ex(lambda: ex.fetch_order_book(SYMBOL,limit=5)))
        bid=ob["bids"][0][0] if ob["bids"] else None
        ask=ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid=(bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception: return None

def _iv_secs(iv):
    iv=iv.lower()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 900
def time_to_candle_close(df):
    tf=_iv_secs(INTERVAL)
    if len(df)==0: return tf
    cur=int(df["time"].iloc[-1]); now=int(time.time()*1000); nxt=cur+tf*1000
    while nxt<=now: nxt+=tf*1000
    return int(max(0,nxt-now)/1000)

def _safe_float(x, d=0.0):
    try: return float(x)
    except Exception: return d

# ================== INDICATORS ====================
def wema(s,n): return s.ewm(alpha=1/n,adjust=False).mean()
def compute_indicators(df):
    if len(df)<max(RSI_LEN,ADX_LEN,ATR_LEN)+2:
        return {"rsi":50,"plus_di":0,"minus_di":0,"adx":0,"atr":0}
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wema(tr,ATR_LEN)
    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = wema(up,RSI_LEN)/wema(dn,RSI_LEN).replace(0,1e-12)
    rsi=100-(100/(1+rs))
    upm=h.diff(); dnm=l.shift(1)-l
    plus_dm=upm.where((upm>dnm)&(upm>0),0.0)
    minus_dm=dnm.where((dnm>upm)&(dnm>0),0.0)
    plus_di=100*(wema(plus_dm,ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(wema(minus_dm,ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0)
    adx=wema(dx,ADX_LEN)
    i=len(df)-1
    return {"rsi":float(rsi.iloc[i]),"plus_di":float(plus_di.iloc[i]),
            "minus_di":float(minus_di.iloc[i]),"adx":float(adx.iloc[i]),
            "atr":float(atr.iloc[i])}

# ================== RF (Closed) ===================
def _ema(s, n):
    return s.ewm(span=n,adjust=False).mean()
def _rng_size(src, qty, n):
    avr=_ema((src-src.shift(1)).abs(),n)
    return _ema(avr,(n*2)-1)*qty
def _rng_filter(src,rsize):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x-r>prev: cur=x-r
        if x+r<prev: cur=x+r
        rf.append(cur)
    filt=pd.Series(rf,index=src.index,dtype="float64")
    return filt+rsize, filt-rsize, filt
def rf_signal_closed(df: pd.DataFrame):
    if len(df)<RF_PERIOD+3:
        return {"time":int(time.time()*1000),"price":None,"long":False,"short":False,"filter":None,"hi":None,"lo":None}
    d=df.iloc[:-1]  # آخر مغلقة فقط
    src=d[RF_SOURCE].astype(float)
    hi,lo,f=_rng_filter(src,_rng_size(src,RF_MULT,RF_PERIOD))
    p_now=float(src.iloc[-1]); p_prev=float(src.iloc[-2]); f_now=float(f.iloc[-1]); f_prev=float(f.iloc[-2])
    def bps(a,b): 
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    return {
        "time":int(d["time"].iloc[-1]), "price":p_now, "filter":f_now,
        "hi":float(hi.iloc[-1]), "lo":float(lo.iloc[-1]),
        "long": (p_prev<=f_prev and p_now>f_now and bps(p_now,f_now)>=RF_HYST_BPS),
        "short":(p_prev>=f_prev and p_now<f_now and bps(p_now,f_now)>=RF_HYST_BPS)
    }

# ================== BOOKMAP-LITE ==================
class LiquidityMap:
    """تجميع دفتر الأوامر وتحديد أقوى جدران عرض/طلب + قربها من السعر."""
    def __init__(self, depth=BM_DEPTH_LEVELS, keep=BM_SNAPSHOTS):
        self.depth = depth
        self.snaps = deque(maxlen=keep)
    def _bucket(self, levels):
        book = {}
        for p, s in levels or []:
            if p is None or s is None: continue
            key = round(float(p), 4)
            book[key] = book.get(key, 0.0) + float(s)
        return book
    def update_from_orderbook(self, ob):
        if not ob: return
        bids = self._bucket(ob.get("bids", [])[: self.depth])
        asks = self._bucket(ob.get("asks", [])[: self.depth])
        self.snaps.append({"bids": bids, "asks": asks})
    def _aggregate(self):
        agg_b, agg_a = {}, {}
        for snap in self.snaps:
            for p, s in (snap["bids"] or {}).items(): agg_b[p] = agg_b.get(p, 0.0) + s
            for p, s in (snap["asks"] or {}).items(): agg_a[p] = agg_a.get(p, 0.0) + s
        return agg_b, agg_a
    def wall_near_price(self, mid_price: float):
        if mid_price is None or not self.snaps: return None, None
        agg_b, agg_a = self._aggregate()
        if not agg_b and not agg_a: return None, None
        def walls(agg: dict):
            if not agg: return []
            sizes = sorted(agg.values())
            if not sizes: return []
            import math
            q_idx = max(0, int(math.floor(len(sizes) * BM_WALL_QTL)) - 1)
            thr = sizes[q_idx] if q_idx < len(sizes) else sizes[-1]
            return [(p, s) for p, s in agg.items() if s >= thr]
        bid_walls = walls(agg_b); ask_walls = walls(agg_a)
        def nearest(walls, side):
            best = None; best_bps = 1e9
            for p, s in walls:
                if side == "bid" and p > mid_price:  continue
                if side == "ask" and p < mid_price:  continue
                bps = abs((p - mid_price) / max(mid_price, 1e-9)) * 10000.0
                notional = s * mid_price
                if notional < BM_MIN_NOTIONAL: continue
                if bps < best_bps: best_bps = bps; best = (p, s, bps, notional)
            return best
        near_bid = nearest(bid_walls, "bid")
        near_ask = nearest(ask_walls, "ask")
        return near_bid, near_ask

LIQMAP = LiquidityMap()
def fetch_orderbook():
    try:
        return with_retry(lambda: _with_ex(lambda: ex.fetch_order_book(SYMBOL, limit=BM_DEPTH_LEVELS)))
    except Exception:
        return None

# ================== FVG (3-candle) ==================
def detect_fvg(df: pd.DataFrame):
    if len(df) < 3: return None
    d = df.iloc[:-1].iloc[-20:]
    res=None
    for i in range(2, len(d)):
        h2 = float(d["high"].iloc[i-2]); l2 = float(d["low"].iloc[i-2])
        h1 = float(d["high"].iloc[i-1]); l1 = float(d["low"].iloc[i-1])
        h0 = float(d["high"].iloc[i]);   l0 = float(d["low"].iloc[i])
        if l1 > h2:  # Bullish FVG
            gap_top, gap_bot = l1, h2
            res={"type":"bull","top":gap_top,"bot":gap_bot,"mid":(gap_top+gap_bot)/2.0,"i":i}
        elif h1 < l2:  # Bearish FVG
            gap_top, gap_bot = l2, h1
            res={"type":"bear","top":gap_top,"bot":gap_bot,"mid":(gap_top+gap_bot)/2.0,"i":i}
    return res
def fvg_retest_signal(fvg, price: float, tol_bps: float = 10.0):
    if not fvg or price is None: return None
    top, bot = float(fvg["top"]), float(fvg["bot"])
    inside = bot <= price <= top
    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 1e9
    near_mid = _bps(price, fvg["mid"]) <= tol_bps
    if inside: return {"type":fvg["type"],"inside":True,"near_mid":near_mid}
    return None

# ================== Accumulation ==================
def detect_accumulation(df: pd.DataFrame):
    if len(df) < 40: 
        return {"type":"none","zone":None,"score":0.0}
    d = df.iloc[-30:]
    rng = float(d["high"].max() - d["low"].min())
    atr = compute_indicators(df).get("atr") or 0.0
    if rng <= 0 or atr <= 0:
        return {"type":"none","zone":None,"score":0.0}
    width_pct = (rng / max(float(d["close"].iloc[-1]), 1e-9)) * 100.0
    vol = d["volume"].astype(float)
    vol_trend = (vol.iloc[-10:].mean() > vol.iloc[:10].mean())
    narrow_ok = width_pct <= 1.2
    score = (1.0 if narrow_ok else 0.0) + (1.0 if vol_trend else 0.0)
    typ = "accum" if score >= 1.5 else "none"
    zone = {"low": float(d["low"].min()), "high": float(d["high"].max())} if typ != "none" else None
    return {"type": typ, "zone": zone, "score": score}

# ================== Wick/Impulse ==================
def analyze_wick_impulse(df, atr_val: float):
    res={"upper_ratio":0.0,"lower_ratio":0.0,"body_atr":0.0,"dir":"flat","wick_up":False,"wick_down":False,"impulse":False,"grade":"none"}
    if len(df) < 2 or atr_val <= 0: return res
    o,h,l,c = map(float, df[["open","high","low","close"]].iloc[-2])
    rng = max(h-l, 1e-12); up = h - max(o,c); dn = min(o,c) - l; body = abs(c-o)
    res["upper_ratio"]=up/rng; res["lower_ratio"]=dn/rng; res["body_atr"]=body/max(atr_val,1e-12)
    res["dir"]="up" if c>o else "down" if c<o else "flat"
    res["wick_up"] = res["upper_ratio"]>=WICK_MIN_RATIO
    res["wick_down"]= res["lower_ratio"]>=WICK_MIN_RATIO
    res["impulse"]  = res["body_atr"]  >= IMPULSE_ATR_MULT
    extreme_wick=(res["upper_ratio"]>=WICK_EXTREME_RATIO) or (res["lower_ratio"]>=WICK_EXTREME_RATIO)
    extreme_imp =(res["body_atr"]>=IMPULSE_EXTREME_ATR_MULT)
    res["grade"]="extreme" if (extreme_wick or extreme_imp) else ("strong" if (res["impulse"] or res["wick_up"] or res["wick_down"]) else "none")
    return res

# ============ Candle Patterns ============
def last_closed_candle(df):
    if len(df) < 2: return None
    return df.iloc[-2]
def candle_features(df, atr: float):
    c = last_closed_candle(df)
    if c is None or atr <= 0: return None
    o,h,l,cl = map(float, [c["open"], c["high"], c["low"], c["close"]])
    rng=max(h-l,1e-12); up=h-max(o,cl); dn=min(o,cl)-l; body=abs(cl-o)
    return {"dir":"up" if cl>o else "down" if cl<o else "flat","upper_ratio":up/rng,"lower_ratio":dn/rng,"body_atr":body/max(atr,1e-12),"range_atr":(h-l)/max(atr,1e-12),"o":o,"h":h,"l":l,"c":cl}
def is_bearish_engulf(df):
    if len(df) < 3: return False
    a=df.iloc[-3]; b=df.iloc[-2]
    return (float(a["close"])>float(a["open"]) and float(b["close"])<float(b["open"]) and float(b["open"])>=float(a["close"]) and float(b["close"])<=float(a["open"]))
def is_bullish_engulf(df):
    if len(df) < 3: return False
    a=df.iloc[-3]; b=df.iloc[-2]
    return (float(a["close"])<float(a["open"]) and float(b["close"])>float(b["open"]) and float(b["open"])<=float(a["close"]) and float(b["close"])>=float(a["open"]))

# ================== Trend & SMC Utils =============
def _near_bps(a,b,bps): 
    try: return abs((a-b)/b)*10000.0 <= bps
    except Exception: return False
def pivots(df, lb=3):
    if len(df)<lb*2+1: return [],[]
    H=[]; L=[]; hi=df["high"].astype(float).values; lo=df["low"].astype(float).values
    for i in range(lb,len(df)-lb):
        if hi[i]==max(hi[i-lb:i+lb+1]): H.append(i)
        if lo[i]==min(lo[i-lb:i+lb+1]): L.append(i)
    return H,L
def trend_mode(df, ind):
    if ind["adx"]<TREND_ADX_MIN or len(df)<STRUCT_BARS+5: return "NEUTRAL"
    d=df.iloc[-(STRUCT_BARS+1):-1]
    hh, ll = d["high"].astype(float).values, d["low"].astype(float).values
    k=max(3,STRUCT_BARS//3)
    up = hh[-k:].mean()>hh[k:2*k].mean()>hh[:k].mean() and ll[-k:].mean()>ll[k:2*k].mean()>ll[:k].mean()
    dn = hh[-k:].mean()<hh[k:2*k].mean()<hh[:k].mean() and ll[-k:].mean()<ll[k:2*k].mean()<ll[:k].mean()
    return "UP" if up else "DOWN" if dn else "NEUTRAL"
def make_zones(df):
    d=df.iloc[:-1] if len(df)>=2 else df.copy()
    H,L = pivots(d, lb=3)
    zones=[]
    for idx in H[-6:]:
        ph=float(d["high"].iloc[idx])
        zones.append({"side":"supply","top":ph,"bot":ph*(1-0.003),"i":idx})
    for idx in L[-6:]:
        pl=float(d["low"].iloc[idx])
        zones.append({"side":"demand","bot":pl,"top":pl*(1+0.003),"i":idx})
    now=len(d)-1
    scored=[]
    for z in zones:
        age = now - z["i"]
        age_pen = max(0.0, 1.0 - (age/150.0))
        scored.append({**z,"score":age_pen})
    return scored
def touch_box_now(df, z):
    if not z or len(df)<1: return False
    low=float(df["low"].iloc[-1]); high=float(df["high"].iloc[-1])
    return (low<=z.get("top",0)) if z["side"]=="demand" else (high>=z.get("bot",0))
def eq_levels(df, bps=EQ_BPS):
    d=df.iloc[-SWEEP_LOOKBACK:] if len(df)>=SWEEP_LOOKBACK else df
    highs=d["high"].astype(float).values; lows =d["low"].astype(float).values
    eqh=None; eql=None
    for i in range(5, len(d)-5):
        for j in range(i+2, min(i+15, len(d)-2)):
            if _near_bps(highs[i], highs[j], bps): eqh=max(eqh or 0.0, highs[i])
            if _near_bps(lows[i],  lows[j],  bps): eql=min(eql or 1e9, lows[i])
    return eqh, eql
def sweep_signal(df, side, atr):
    d=df.iloc[-(STRUCT_BARS+3):-1]
    if len(d)<10 or atr<=0: return None
    H,L = pivots(d, lb=3)
    if side=="long" and L:
        lvl=float(d["low"].iloc[L[-1]]); last=d.iloc[-1]; prev=d.iloc[-2]
        if float(last["low"])<lvl and float(last["close"])>float(prev["close"]):
            body=abs(float(last["close"])-float(last["open"]))
            if body/atr>=DSP_ATR_MIN: return {"type":"sweep_down","lvl":lvl,"dsp":body/atr}
    if side=="short" and H:
        lvl=float(d["high"].iloc[H[-1]]); last=d.iloc[-1]; prev=d.iloc[-2]
        if float(last["high"])>lvl and float(last["close"])<float(prev["close"]):
            body=abs(float(last["close"])-float(last["open"]))
            if body/atr>=DSP_ATR_MIN: return {"type":"sweep_up","lvl":lvl,"dsp":body/atr}
    return None
def liquidity_flow(df, ind):
    o,h,l,c = map(float, df[["open","high","low","close"]].iloc[-1])
    rng=max(h-l,1e-12); body=abs(c-o)
    v=float(df["volume"].iloc[-1]); vol_ema=float(df["volume"].ewm(alpha=1/34, adjust=False).mean().iloc[-1])
    vol_spike = v >= vol_ema*1.8; spread_ok = (body/rng) >= 0.15
    if vol_spike and spread_ok: return "inflow_up" if c>o else "inflow_down"
    return "neutral"

# ================== Trend Conviction ===============
def trend_conviction(df, ind, side):
    tmode = trend_mode(df, ind)
    adx   = float(ind.get("adx") or 0.0)
    pdi, mdi = float(ind.get("plus_di") or 0.0), float(ind.get("minus_di") or 0.0)
    aligned = (tmode == "UP" and side == "long") or (tmode == "DOWN" and side == "short")
    if not aligned or adx < TREND_LOCK_ADX: return False, 0.0
    if TREND_LOCK_REQUIRE_DI:
        if side=="long" and not (pdi>mdi): return False, 0.0
        if side=="short" and not (mdi>pdi): return False, 0.0
    d = df.iloc[-(TREND_LOCK_MIN_BARS+1):-1]
    if len(d) < TREND_LOCK_MIN_BARS: return False, 0.0
    ups  = (d["close"]>d["open"]).sum(); dns  = (d["close"]<d["open"]).sum()
    dir_ok = (ups>=dns) if side=="long" else (dns>=ups)
    score = (adx- TREND_LOCK_ADX)/10.0 + (0.5 if dir_ok else 0.0) + (0.5 if ((pdi-mdi)>0) == (side=="long") else 0.0)
    strong = dir_ok and (score>=0.6)
    return strong, max(0.0, score)

# ================== Council Runtime Assess =========
def council_runtime_assess(ind, wick, rr, trend_mode_val, side, df):
    adx = float(ind.get("adx") or 0.0)
    strong_trend, tscore = trend_conviction(df, ind, side)
    if wick["grade"] == "extreme" and rr >= TRAIL_ACTIVATE_PCT:
        if strong_trend and EXTREME_CLOSE_REQUIRE_TREND_MISMATCH:
            return "tighten_trail", f"Extreme but trend-locked (score={tscore:.2f})", 0.0
        if adx <= STRICT_COOL_ADX:
            if STRICT_CLOSE_NEEDS_BOS and not _bos_micro(df, side):
                return "tighten_trail", "Extreme but no BOS against", 0.0
            return "strict_close", "Extreme + ADX cool + BOS ok", 0.0
    if rr >= TRAIL_ACTIVATE_PCT:
        return "tighten_trail", f"Trail tighten rr={rr:.2f}%", 0.0
    return None, None, 0.0

# ================== Council (SMC) ==================
def council_strength_score(votes_reasons):
    score = 0.0
    for r in votes_reasons or []:
        if "sweep" in r: score += 1.2
        if "retest_" in r: score += 1.0
        if "RF" in r:     score += 0.6
        if "DI" in r:     score += 0.6
        if "liq_inflow" in r: score += 0.6
        if "bid_wall" in r or "ask_wall" in r: score += 0.8
        if "FVG_retest" in r: score += 0.9
        if "accum" in r: score += 0.7
        if "peak_top" in r or "peak_bottom" in r: score += 1.1
        if "RSI neutral" in r: score += 0.3
    return score

class Council:
    def __init__(self):
        self.min_entry=4; self.min_exit=3
        self.log=""
    def vote(self, df, ind, rf):
        atr=float(ind.get("atr") or 0.0)
        zones=make_zones(df)
        z_best = max(zones, key=lambda z: z["score"]) if zones else None
        tmode = trend_mode(df, ind)
        eqh,eql = eq_levels(df)
        sweepB=sweep_signal(df,"long", atr)
        sweepS=sweep_signal(df,"short",atr)
        flow = liquidity_flow(df, ind)
        # Bookmap & FVG & Accum & Peak/Bottom
        px = price_now() or float(df["close"].iloc[-1])
        near_bid, near_ask = LIQMAP.wall_near_price(px)
        fvg = detect_fvg(df)
        fvg_rt = fvg_retest_signal(fvg, px) if fvg else None
        accum = detect_accumulation(df)
        peak = detect_strong_swing(df, ind, LIQMAP, rsi_extremes=True)
        b=s=0; rb=[]; rs=[]
        # Zones + retest
        if z_best and touch_box_now(df, z_best):
            if z_best["side"]=="demand": b+=1; rb.append("retest_demand")
            else: s+=1; rs.append("retest_supply")
        # Liquidity
        if sweepB: b+=2; rb.append(f"sweep↓+dsp{(sweepB['dsp']):.1f}x")
        if sweepS: s+=2; rs.append(f"sweep↑+dsp{(sweepS['dsp']):.1f}x")
        if flow=="inflow_up": b+=1; rb.append("liq_inflow↑")
        if flow=="inflow_down": s+=1; rs.append("liq_inflow↓")
        # Indicators
        pdi,mdi,adx = ind["plus_di"], ind["minus_di"], ind["adx"]; rsi=ind["rsi"]
        o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
        if adx>=20 and pdi>mdi: b+=1; rb.append("+DI>−DI & ADX")
        if adx>=20 and mdi>pdi: s+=1; rs.append("−DI>+DI & ADX")
        if 45<=rsi<=55:
            if c>o: b+=1; rb.append("RSI neutral ↗")
            else:   s+=1; rs.append("RSI neutral ↘")
        # RF closed confirmation
        if rf.get("long"):  b+=1; rb.append("RF↑")
        if rf.get("short"): s+=1; rs.append("RF↓")
        # Bookmap walls
        if near_bid and near_bid[2] <= BM_NEAR_BPS: b += 1; rb.append(f"bid_wall@{near_bid[2]:.1f}bps")
        if near_ask and near_ask[2] <= BM_NEAR_BPS: s += 1; rs.append(f"ask_wall@{near_ask[2]:.1f}bps")
        # FVG retest
        if fvg_rt and fvg_rt["inside"]:
            if fvg_rt["type"]=="bull": b+=1; rb.append("FVG_retest↑")
            else: s+=1; rs.append("FVG_retest↓")
        # Accumulation
        if accum["type"] == "accum" and accum["zone"]:
            z = accum["zone"]
            if px <= z["low"]*(1+CHOP_EQ_TOL_BPS/10000.0):
                b += 1; rb.append("accum@floor")
            else:
                b += 0.5; rb.append("accum")
        # Peak/Bottom hunter
        if peak and peak["type"]=="top":
            s += 2; rs.append("peak_top")
        elif peak and peak["type"]=="bottom":
            b += 2; rb.append("peak_bottom")
        self.log=(f"SCM | trend={tmode} | zone={z_best and z_best['side']} "
                  f"| liq(EQH={fmt(eqh)} EQL={fmt(eql)}) | flow={flow} "
                  f"|| BUY={b}[{', '.join(rb) or '—'}] | SELL={s}[{', '.join(rs) or '—'}]")
        print(colored(self.log,"green" if b>s else "red" if s>b else "cyan"))
        return b,rb,s,rs,tmode,z_best
    def entry_exit(self, df, ind, rf):
        b,rb,s,rs,tmode,zone = self.vote(df, ind, rf)
        entry=None; exit_=None
        buy_strength  = council_strength_score(rb)
        sell_strength = council_strength_score(rs)
        # ENTRY
        if not STATE["open"]:
            if b>=self.min_entry and tmode!="DOWN" and buy_strength>=COUNCIL_MIN_STRONG_SCORE:
                entry={"side":"buy","reason":rb,"tmode":tmode,"zone":zone,"strength":buy_strength}
            elif s>=self.min_entry and tmode!="UP" and sell_strength>=COUNCIL_MIN_STRONG_SCORE:
                entry={"side":"sell","reason":rs,"tmode":tmode,"zone":zone,"strength":sell_strength}
        # EXIT
        if STATE["open"]:
            adx=float(ind.get("adx") or 0.0); side=STATE["side"]
            votes=0; reasons=[]
            atr=float(ind.get("atr") or 0.0)
            swB=sweep_signal(df,"long", atr); swS=sweep_signal(df,"short",atr)
            if side=="long":
                if swS: votes+=1; reasons.append("sweep_up→down")
                if zone and zone.get("side")=="supply" and touch_box_now(df,zone): votes+=1; reasons.append("retest_supply")
                if adx<20: votes+=1; reasons.append("ADX cool-off")
            else:
                if swB: votes+=1; reasons.append("sweep_down→up")
                if zone and zone.get("side")=="demand" and touch_box_now(df,zone): votes+=1; reasons.append("retest_demand")
                if adx<20: votes+=1; reasons.append("ADX cool-off")
            min_exit = self.min_exit + (1 if ((tmode=="UP" and side=="long") or (tmode=="DOWN" and side=="short")) else 0)
            if votes>=min_exit:
                exit_={"action":"close","reason":" / ".join(reasons)}
        return entry, exit_

council = Council()

# ================== STATE / ORDERS ================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0, "_opp_rf_bars": 0,
    "trend_mode":"NEUTRAL", "_flip_rf_bars":0,
    "exits_used": 0,
    "chop_mode": False, "chop_zone": None, "chop_hold_until": 0,
    "_trend_harvests": 0
}
compound_pnl = 0.0
LAST_CLOSE_T = 0

def save_state():
    try:
        data = {"STATE": STATE,"compound_pnl": compound_pnl,"LAST_CLOSE_T": LAST_CLOSE_T,"SYMBOL": SYMBOL,"INTERVAL": INTERVAL,"ts": int(time.time()*1000)}
        with open("state.json","w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
    except Exception as e:
        logging.warning(f"save_state: {e}")
def load_state():
    global compound_pnl, LAST_CLOSE_T
    try:
        if not os.path.exists("state.json"): return False
        with open("state.json","r",encoding="utf-8") as f: data=json.load(f)
        st=data.get("STATE",{})
        for k in STATE.keys():
            if k in st: STATE[k]=st[k]
        compound_pnl = _safe_float(data.get("compound_pnl"),0.0)
        LAST_CLOSE_T  = int(_safe_float(data.get("LAST_CLOSE_T"),0))
        print(colored("🧠 state restored from disk","cyan")); return True
    except Exception as e:
        logging.warning(f"load_state: {e}"); return False
def _norm_sym(s: str) -> str:
    if not s: return ""
    return "".join(ch for ch in str(s) if ch.isalnum()).upper()
def _read_position():
    try:
        poss=_with_ex(lambda: ex.fetch_positions(params={"type":"swap"}))
        base=SYMBOL.split(":")[0]; base_norm=_norm_sym(base)
        for p in poss:
            sym=(p.get("symbol") or p.get("info",{}).get("symbol") or "")
            sym_norm=_norm_sym(sym)
            if base_norm not in sym_norm: continue
            qty=abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty<=0: return 0.0,None,None
            entry=float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            ps=(p.get("info",{}).get("positionSide") or p.get("side") or "").upper()
            side="long" if "LONG" in ps else "short" if "SHORT" in ps else None
            return qty, side, entry
    except Exception as e: logging.error(f"_read_position: {e}")
    return 0.0,None,None
def reconcile_with_exchange():
    try:
        exch_qty, exch_side, exch_entry = _read_position()
        if exch_qty > 0 and exch_side in ("long","short"):
            STATE.update({"open": True,"side": exch_side,"qty": safe_qty(exch_qty),"entry": _safe_float(exch_entry) or STATE.get("entry"),"pnl": 0.0})
            print(colored(f"🧷 reconciled: {exch_side} qty={fmt(STATE['qty'],4)} @ {fmt(STATE['entry'])}","cyan"))
        else:
            if STATE.get("open"):
                print(colored("🧹 reconcile: exchange flat → clearing local open","yellow"))
                STATE.update({"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,"trail":None,"breakeven":None,"tp1_done":False,"highest_profit_pct":0.0,"profit_targets_achieved":0,"_opp_rf_bars":0,"_flip_rf_bars":0,"exits_used":0,"chop_mode":False,"chop_zone":None})
    except Exception as e:
        logging.warning(f"reconcile_with_exchange: {e}")

def _params_open(side):
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if side=="buy" else "SHORT","reduceOnly":False}
    return {"positionSide":"BOTH","reduceOnly":False}
def _params_close():
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if STATE.get("side")=="long" else "SHORT","reduceOnly":True}
    return {"positionSide":"BOTH","reduceOnly":True}

def compute_size(balance, price):
    cap=(balance or 0.0)*RISK_ALLOC*LEVERAGE
    raw=max(0.0, cap/max(float(price or 0.0),1e-9))
    return safe_qty(raw)

def open_market(side, qty, price, tag=""):
    if STATE.get("open"):
        logging.warning("skip open: position already open")
        return False
    if qty<=0:
        logging.warning("skip open qty<=0")
        return False
    if MODE_LIVE:
        try:
            try: _with_ex(lambda: ex.set_leverage(LEVERAGE,SYMBOL,params={"side":"BOTH"}))
            except Exception: pass
            _with_ex(lambda: ex.create_order(SYMBOL,"market",side,qty,None,_params_open(side)))
        except Exception as e:
            logging.error(f"open_market: {e}"); return False
    STATE.update({"open":True,"side":"long" if side=="buy" else "short","entry":price,"qty":qty,"pnl":0.0,"bars":0,"trail":None,"breakeven":None,"tp1_done":False,"highest_profit_pct":0.0,"profit_targets_achieved":0,"_opp_rf_bars":0,"exits_used":0,"_trend_harvests":0})
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)} {tag}","green" if side=='buy' else "red"))
    save_state()
    return True

def _reset_after_close(reason):
    STATE.update({"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,"trail":None,"breakeven":None,"tp1_done":False,"highest_profit_pct":0.0,"profit_targets_achieved":0,"_opp_rf_bars":0,"_flip_rf_bars":0,"exits_used":0,"_trend_harvests":0,"chop_mode":False,"chop_zone":None})
    logging.info(f"AFTER_CLOSE: {reason}")
    save_state()

def close_market_strict(reason="STRICT"):
    global compound_pnl, LAST_CLOSE_T
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty<=0 and not STATE.get("open"): return
    if exch_qty<=0 and STATE.get("open"):
        px=price_now() or STATE["entry"]; side=STATE["side"]; qty=STATE["qty"]; entry=STATE["entry"]
        pnl=(px-entry)*qty*(1 if side=="long" else -1); compound_pnl+=pnl
        print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
        _reset_after_close(reason); LAST_CLOSE_T=int(time.time()*1000); save_state(); return
    side_to_close="sell" if exch_side=="long" else "buy"
    qty_to_close=safe_qty(exch_qty); attempts=0; last=None
    while attempts<6:
        try:
            if MODE_LIVE:
                params=_params_close(); params["reduceOnly"]=True
                _with_ex(lambda: ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params))
            time.sleep(2.0)
            left,_,_=_read_position()
            if left<=0:
                px=price_now() or STATE.get("entry") or exch_entry
                entry_px=STATE.get("entry") or exch_entry or px
                side=STATE.get("side") or exch_side
                pnl=(px-entry_px)*exch_qty*(1 if side=="long" else -1)
                compound_pnl+=pnl
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                _reset_after_close(reason); LAST_CLOSE_T=int(time.time()*1000); save_state(); return
            qty_to_close=safe_qty(left); attempts+=1
            print(colored(f"⚠️ strict close retry {attempts} residual={fmt(left,4)}","yellow"))
        except Exception as e:
            last=e; attempts+=1; time.sleep(2.0)
    print(colored(f"❌ STRICT CLOSE FAILED — last_error={last}","red"))

def close_partial(frac, reason):
    if LOW_FEE_MODE and not ALLOW_PARTIALS_IN_LOW: return
    if not STATE["open"] or STATE["qty"]<=0: return
    qty_close=safe_qty(max(0.0, STATE["qty"]*min(max(frac,0.0),1.0)))
    px=price_now() or STATE["entry"]
    min_unit=max(RESIDUAL_MIN_QTY, LOT_MIN or RESIDUAL_MIN_QTY)
    if qty_close<min_unit: return
    side="sell" if STATE["side"]=="long" else "buy"
    if MODE_LIVE:
        try: _with_ex(lambda: ex.create_order(SYMBOL,"market",side,qty_close,None,_params_close()))
        except Exception as e: logging.error(f"partial: {e}"); return
    STATE["qty"]=safe_qty(STATE["qty"]-qty_close)
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} rem={fmt(STATE['qty'],4)}","magenta"))
    if 0<STATE["qty"]<=FINAL_CHUNK_QTY: close_market_strict("FINAL_CHUNK_RULE")
    save_state()

# ================== Chop Detection =================
def _bb_width_pct(closes: pd.Series, n=20):
    if len(closes) < n+2: return None
    ma = closes.rolling(n).mean(); std = closes.rolling(n).std(ddof=0)
    up  = ma + 2*std; lo  = ma - 2*std
    last = len(closes)-1; mid = float(ma.iloc[last]); hi=float(up.iloc[last]); lw=float(lo.iloc[last])
    if mid<=0: return None
    return (hi - lw) / mid * 100.0
def _eq_inside_range(df, low, high, bps):
    d = df.iloc[-max(CHOP_MIN_BARS, 20):]
    highs = d["high"].astype(float).values; lows  = d["low"].astype(float).values
    eqh = any(_near_bps(h, high, bps) for h in highs)
    eql = any(_near_bps(l, low,  bps) for l in lows)
    return eqh and eql
def detect_chop(df, ind):
    if len(df) < CHOP_MIN_BARS+5: 
        return {"is_chop": False, "zone": None, "score": 0.0}
    adx = float(ind.get("adx") or 0.0)
    if adx > CHOP_ADX_MAX:
        return {"is_chop": False, "zone": None, "score": 0.0}
    d   = df.iloc[-(CHOP_MIN_BARS+2):]
    hi  = float(d["high"].max()); lw = float(d["low"].min())
    mid = float(d["close"].iloc[-1])
    if lw<=0: 
        return {"is_chop": False, "zone": None, "score": 0.0}
    bw_pct = _bb_width_pct(df["close"].astype(float), n=20) or 999.0
    band_ok = (bw_pct <= CHOP_BB_WIDTH_MAX_PCT)
    eq_ok = _eq_inside_range(df, lw, hi, CHOP_EQ_TOL_BPS)
    try:
        rf = rf_signal_closed(df); filt = rf.get("filter")
        around = abs((mid - float(filt)) / float(filt)) * 10000.0 <= 12.0 if filt else False
    except Exception:
        around = False
    flags = [band_ok, eq_ok, around]
    score = sum(flags) / 3.0
    is_chop = (band_ok and (eq_ok or around))
    zone = {"low": lw, "high": hi, "bw_pct": bw_pct} if is_chop else None
    return {"is_chop": is_chop, "zone": zone, "score": score}
def chop_exit_if_any(px, ind):
    if not STATE["open"] or STATE["qty"]<=0: return False
    if STATE.get("exits_used",0) >= MAX_EXITS_PER_TRADE: return False
    entry = STATE["entry"]; side = STATE["side"]
    rr = (px - entry)/entry*100.0*(1 if side=="long" else -1)
    if rr >= CHOP_EXIT_PROFIT_PCT:
        close_market_strict(f"CHOP_EXIT@{rr:.2f}%")
        STATE["exits_used"] = STATE.get("exits_used",0) + 1
        STATE["chop_hold_until"] = int(time.time()*1000) + _iv_secs(INTERVAL)*1000*CHOP_HOLD_OFF_BARS
        return True
    return False

# ================== Opposite RF Defense ============
def defensive_on_opposite_rf(ind, info):
    if not STATE["open"] or STATE["qty"]<=0: return
    opp = (STATE["side"]=="long" and info.get("short")) or (STATE["side"]=="short" and info.get("long"))
    if not opp: return
    atr=float(ind.get("atr") or 0.0); px = info.get("price") or price_now() or STATE["entry"]
    if atr>0 and px:
        gap=atr*max(ATR_TRAIL_MULT,1.3)
        if STATE["side"]=="long": STATE["trail"]=max(STATE["trail"] or (px-gap), px-gap)
        else: STATE["trail"]=min(STATE["trail"] or (px+gap), px+gap)

# ================== Auto-Flip + Reverse (oneway) ===
def _bos_micro(df, side):
    d = df.iloc[-10:-1]
    if len(d) < 5: return False
    if side == "short":
        return float(d["close"].iloc[-1]) > float(d["high"].max())
    else:
        return float(d["close"].iloc[-1]) < float(d["low"].min())
def should_auto_flip(df, ind, rf_info):
    if not STATE["open"]: return None
    cur_side = STATE["side"]
    trend_ok = (STATE["trend_mode"]=="UP" and cur_side=="short") or (STATE["trend_mode"]=="DOWN" and cur_side=="long")
    if not trend_ok: 
        STATE["_flip_rf_bars"]=0; return None
    rf_opp = (cur_side=="long" and rf_info.get("short")) or (cur_side=="short" and rf_info.get("long"))
    if not rf_opp:
        STATE["_flip_rf_bars"]=0; return None
    STATE["_flip_rf_bars"]=int(STATE.get("_flip_rf_bars",0))+1
    if STATE["_flip_rf_bars"] < FLIP_RF_BARS_CONFIRMED: return None
    atr = float(ind.get("atr") or 0.0)
    if atr <= 0: return None
    o,c = map(float, df[["open","close"]].iloc[-2])
    dsp = abs(c-o)/atr
    if dsp < DSP_ATR_MIN: return None
    if float(ind.get("adx") or 0.0) < FLIP_MIN_ADX: return None
    if FLIP_REQUIRE_DI:
        pdi, mdi = float(ind.get("plus_di") or 0.0), float(ind.get("minus_di") or 0.0)
        if cur_side=="short" and not (pdi > mdi): return None
        if cur_side=="long"  and not (mdi > pdi): return None
    if FLIP_REQUIRE_BOS and not _bos_micro(df, cur_side): return None
    new_side = "buy" if cur_side=="short" else "sell"
    return {"flip_to": new_side, "reason": f"AutoFlip trend={STATE['trend_mode']} rfK={STATE['_flip_rf_bars']} dsp≥{DSP_ATR_MIN}xATR DI_flip ADX≥{FLIP_MIN_ADX}"}
def reverse_order_oneway(flip_to, desired_qty, price):
    if POSITION_MODE != "oneway":
        if STATE["exits_used"] < MAX_EXITS_PER_TRADE:
            close_market_strict("AUTO_FLIP_CLOSE"); STATE["exits_used"] += 1
        open_market(flip_to, desired_qty, price, tag="[AutoFlip]")
        return True
    cur_qty = safe_qty(STATE.get("qty") or 0.0)
    if cur_qty<=0:
        return open_market(flip_to, desired_qty, price, tag="[AutoFlip]")
    if flip_to=="sell":  # كنا long ونريد short
        net_sell = safe_qty(cur_qty + desired_qty)
        if MODE_LIVE:
            try: _with_ex(lambda: ex.create_order(SYMBOL,"market","sell",net_sell,None,{"positionSide":"BOTH"}))
            except Exception as e: logging.error(f"reverse_order: {e}"); return False
        STATE.update({"side":"short","qty":desired_qty,"entry":price,"trail":None,"breakeven":None,"tp1_done":False,"highest_profit_pct":0.0,"profit_targets_achieved":0,"exits_used":0})
        print(colored(f"🔄 REV ONE-WAY → SHORT qty={fmt(desired_qty,4)} @ {fmt(price)}","magenta")); save_state(); return True
    else:
        net_buy = safe_qty(cur_qty + desired_qty)
        if MODE_LIVE:
            try: _with_ex(lambda: ex.create_order(SYMBOL,"market","buy",net_buy,None,{"positionSide":"BOTH"}))
            except Exception as e: logging.error(f"reverse_order: {e}"); return False
        STATE.update({"side":"long","qty":desired_qty,"entry":price,"trail":None,"breakeven":None,"tp1_done":False,"highest_profit_pct":0.0,"profit_targets_achieved":0,"exits_used":0})
        print(colored(f"🔄 REV ONE-WAY → LONG qty={fmt(desired_qty,4)} @ {fmt(price)}","magenta")); save_state(); return True

# ================== MANAGEMENT ====================
def _consensus(ind, info, side):
    score=0.0; adx=ind["adx"]; rsi=ind["rsi"]
    if (side=="long" and rsi>=55) or (side=="short" and rsi<=45): score+=1.0
    if adx>=28: score+=1.0
    elif adx>=20: score+=0.5
    try:
        if info.get("filter") and info.get("price"):
            if abs(info["price"]-info["filter"])/max(info["filter"],1e-9) >= (RF_HYST_BPS/10000.0): score+=0.5
    except Exception: pass
    return score
def strict_hp_close_with_df(df, ind, rr):
    if STATE["highest_profit_pct"] < TRAIL_ACTIVATE_PCT: return
    strong_trend, _ = trend_conviction(df, ind, STATE["side"])
    if strong_trend: return
    if rr < STATE["highest_profit_pct"]*STRICT_CLOSE_DROP_FROM_HP and float(ind.get("adx") or 0.0) <= STRICT_COOL_ADX:
        if STRICT_CLOSE_NEEDS_BOS and not _bos_micro(df, STATE["side"]): return
        if STATE["exits_used"] < MAX_EXITS_PER_TRADE:
            close_market_strict(f"STRICT_HP_CLOSE {STATE['highest_profit_pct']:.2f}%→{rr:.2f}%"); STATE["exits_used"] += 1
def manage_after_entry(df, ind, info):
    if not STATE["open"] or STATE["qty"]<=0: return
    px=info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr=(px-entry)/entry*100*(1 if side=="long" else -1)
    STATE["trend_mode"]=trend_mode(df, ind)
    align=(STATE["trend_mode"]=="UP" and side=="long") or (STATE["trend_mode"]=="DOWN" and side=="short")
    if not LOW_FEE_MODE:
        pass
    else:
        if (STATE["breakeven"] is None) and rr>=BREAKEVEN_AFTER:
            STATE["breakeven"]=entry
    wick = analyze_wick_impulse(df, float(ind.get("atr") or 0.0))
    act, why, frac = council_runtime_assess(ind, wick, rr, STATE["trend_mode"], side, df)
    if RUNTIME_LOG and act: print(colored(f"🏛 Runtime Council → {act.upper()} :: {why}","white"))
    if act=="strict_close":
        if STATE["exits_used"] < MAX_EXITS_PER_TRADE:
            close_market_strict(f"HARD_EXIT[{why}]"); STATE["exits_used"] += 1
        return
    if rr>STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=rr
    atr=ind["atr"]; mult=ATR_TRAIL_MULT*(1.25 if align else 1.0)
    if rr>=TRAIL_ACTIVATE_PCT and atr>0:
        gap=atr*mult
        if side=="long":
            new=px-gap; STATE["trail"]=max(STATE["trail"] or new, new)
            if STATE["breakeven"] is not None: STATE["trail"]=max(STATE["trail"], STATE["breakeven"])
            if px<STATE["trail"] and STATE["exits_used"]<MAX_EXITS_PER_TRADE:
                close_market_strict(f"TRAIL_ATR({mult:.2f}x)"); STATE["exits_used"] += 1; return
        else:
            new=px+gap; STATE["trail"]=min(STATE["trail"] or new, new)
            if STATE["breakeven"] is not None: STATE["trail"]=min(STATE["trail"], STATE["breakeven"])
            if px>STATE["trail"] and STATE["exits_used"]<MAX_EXITS_PER_TRADE:
                close_market_strict(f"TRAIL_ATR({mult:.2f}x)"); STATE["exits_used"] += 1; return
    strict_hp_close_with_df(df, ind, rr)
    peak_live = detect_strong_swing(df, ind, LIQMAP, rsi_extremes=True)
    if peak_live and peak_live["type"]!="none" and rr >= PEAK_RUNTIME_LOCK_PCT:
        opp_top = (side=="long" and peak_live["type"]=="top")
        opp_bot = (side=="short" and peak_live["type"]=="bottom")
        if opp_top or opp_bot:
            if PEAK_STRICT_IF_BOS and _bos_micro(df, side):
                if STATE["exits_used"] < MAX_EXITS_PER_TRADE:
                    close_market_strict(f"PEAK_LOCK[{peak_live['type']}] rr={rr:.2f}%"); STATE["exits_used"] += 1
                    return
            atr=float(ind.get("atr") or 0.0)
            if atr>0:
                gap = atr * (ATR_TRAIL_MULT*1.15)
                if side=="long": STATE["trail"] = max(STATE["trail"] or (px-gap), px-gap)
                else:            STATE["trail"] = min(STATE["trail"] or (px+gap), px+gap)

# ================== UI ============================
def snapshot(bal, info, ind, spread, rf, council_log=None, reason=None, df=None, chop=None):
    left=time_to_candle_close(df) if df is not None else 0
    print("─"*120)
    print(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"💲 Price {fmt(info.get('price'))} | RF filt={fmt(rf.get('filter'))} hi={fmt(rf.get('hi'))} lo={fmt(rf.get('lo'))} | spread={fmt(spread,2)}bps | closes_in~{left}s")
    print(f"🧮 RSI={fmt(ind['rsi'])} +DI={fmt(ind['plus_di'])} -DI={fmt(ind['minus_di'])} ADX={fmt(ind['adx'])} ATR={fmt(ind['atr'])} Trend={STATE['trend_mode']}")
    if chop and chop.get("is_chop"):
        z=chop.get("zone") or {}
        print(f"   ⚠️ CHOP: low={fmt(z.get('low'))} high={fmt(z.get('high'))} bw%={fmt(z.get('bw_pct'),2)} hold_until={STATE.get('chop_hold_until',0)}")
    if council_log: print(f"🏛 {council_log}")
    if reason: print(f"ℹ️ {reason}")
    print(f"🧭 Balance={fmt(bal,2)} Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x PnL={fmt(STATE['pnl'])} Eq~{fmt((bal or 0)+compound_pnl,2)} exits_used={STATE['exits_used']}")
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp} Entry={fmt(STATE['entry'])} Qty={fmt(STATE['qty'],4)} Bars={STATE['bars']} Trail={fmt(STATE['trail'])} BE={fmt(STATE['breakeven'])} HP={fmt(STATE['highest_profit_pct'],2)}%")
    else:
        print("   ⚪ FLAT")
    print("─"*120)

# ================== MAIN LOOP =====================
app=Flask(__name__)
def trade_loop():
    global LAST_CLOSE_T
    last_persist = 0
    while True:
        try:
            bal=balance_usdt()
            df =fetch_ohlcv()
            # Bookmap snapshot
            ob = fetch_orderbook()
            if ob: LIQMAP.update_from_orderbook(ob)
            ind=compute_indicators(df)
            rf =rf_signal_closed(df)
            px =price_now() or rf["price"]
            spread=orderbook_spread_bps()
            # Trend
            STATE["trend_mode"]=trend_mode(df, ind)
            # Chop
            ch = detect_chop(df, ind)
            STATE["chop_mode"] = ch["is_chop"]; STATE["chop_zone"] = ch["zone"]
            # Council decisions
            entry, exit_ = council.entry_exit(df, ind, rf)
            # Auto-Flip
            flip = should_auto_flip(df, ind, rf)
            if flip and STATE["open"]:
                print(colored(f"🔄 AUTO FLIP → {flip['flip_to'].upper()} :: {flip['reason']}", "magenta"))
                px_flip = price_now() or rf.get("price"); bal_now = balance_usdt()
                if px_flip and bal_now:
                    qty = compute_size(bal_now, px_flip)
                    reverse_order_oneway(flip["flip_to"], qty, px_flip)
                STATE["_flip_rf_bars"]=0
            # Update PnL
            if STATE["open"] and px:
                STATE["pnl"]=(px-STATE["entry"])*STATE["qty"]*(1 if STATE["side"]=="long" else -1)
            # Chop exit
            if STATE["open"] and ch["is_chop"] and px:
                if chop_exit_if_any(px, ind): pass
            # Manage open trade
            if STATE["open"] and px:
                manage_after_entry(df, ind, {"price":px,"filter":rf["filter"]})
                # Defensive opposite RF
                opp = (STATE["side"]=="long" and rf["short"]) or (STATE["side"]=="short" and rf["long"])
                if opp: defensive_on_opposite_rf(ind, {"price": px, **rf})
            # Guards
            reason=None
            if spread is not None and spread>MAX_SPREAD_BPS: reason=f"spread too high {fmt(spread,2)}bps"
            if (reason is None) and ind["adx"]<PAUSE_ADX_THRESHOLD: reason=f"ADX<{int(PAUSE_ADX_THRESHOLD)} pause"
            # Council EXIT (يحترم سقف الخروج)
            if STATE["open"] and exit_ and STATE["exits_used"] < MAX_EXITS_PER_TRADE:
                print(colored(f"🏁 COUNCIL EXIT: {exit_['reason']}","yellow"))
                close_market_strict(f"COUNCIL_EXIT: {exit_['reason']}"); STATE["exits_used"] += 1
                LAST_CLOSE_T=rf.get("time") or int(time.time()*1000)
            # ENTRY: Council ⇒ RF fallback + قيود التذبذب
            sig=None; tag=""
            now_ms = int(time.time()*1000)
            if (not STATE["open"]) and reason is None and STATE.get("chop_hold_until",0) <= now_ms:
                if entry:
                    sig=entry["side"]; tag=f"[Council] {entry['reason']}"
                elif ENTRY_FROM_RF and (rf["long"] or rf["short"]):
                    sig="buy" if rf["long"] else "sell"; tag=f"[RF-closed {'LONG' if rf['long'] else 'SHORT'}]"
            # انتظار الشمعة المغلقة التالية بعد الإغلاق
            if not STATE["open"] and sig and reason is None and WAIT_NEXT_CLOSED:
                if int(rf.get("time") or 0) <= int(LAST_CLOSE_T or 0):
                    reason="wait_for_next_closed_signal"
            # منع الدخول داخل نطاق التذبذب إلا بكسر واضح
            if not STATE["open"] and sig and reason is None and ch["is_chop"] and CHOP_MUST_BREAK_BAND:
                z = ch["zone"]; pr = px or rf["price"]
                if z and pr is not None:
                    over = (pr > z["high"]*(1+CHOP_EQ_TOL_BPS/10000.0))
                    under= (pr < z["low"] *(1-CHOP_EQ_TOL_BPS/10000.0))
                    if not (over or under): reason = "inside chop band — wait for break"
            # دخول
            if not STATE["open"] and sig and reason is None:
                qty=compute_size(bal, px or rf["price"])
                if qty>0 and (px or rf["price"]):
                    opened = open_market(sig, qty, px or rf["price"], tag)
                    if opened:
                        LAST_CLOSE_T=0
                        weak = False
                        if entry and isinstance(entry, dict): weak = (entry.get("strength", 0.0) < COUNCIL_MIN_STRONG_SCORE)
                        elif tag.startswith("[RF-closed"):   weak = True
                        if WEAK_DECISION_STRICT_EXIT and weak:
                            STATE["breakeven"] = STATE["entry"]
                else:
                    reason="qty<=0 or price=None"
            snapshot(bal, {"price": px or rf["price"]}, ind, spread, rf, council.log, reason, df, ch)
            # bar count
            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"]+=1
            # persistence كل ~10 ثوانٍ
            now = time.time()
            if now - last_persist >= 10:
                save_state(); last_persist = now
            time.sleep(NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP)
        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}","red"))
            logging.error(f"loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# ================== API / KEEPALIVE ==============
app=Flask(__name__)
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ DOGE Pro SMC + Peak/Bottom + Trend + AutoFlip + Chop + LowFee + RF(Closed) — {SYMBOL} {INTERVAL} — {mode}"
@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol":SYMBOL,"interval":INTERVAL,"mode":"live" if MODE_LIVE else "paper",
        "leverage":LEVERAGE,"risk_alloc":RISK_ALLOC,"price":price_now(),
        "state":STATE,"compound_pnl":compound_pnl,
        "guards":{"max_spread_bps":MAX_SPREAD_BPS,"pause_adx":PAUSE_ADX_THRESHOLD},
        "council_log": council.log
    })
@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "trend_mode": STATE["trend_mode"], "timestamp": datetime.utcnow().isoformat()
    }), 200

def keepalive_loop():
    url=SELF_URL.rstrip("/")
    if not url: 
        print(colored("⛔ keepalive disabled (SELF_URL not set)","yellow")); return
    import requests
    s=requests.Session(); s.headers.update({"User-Agent":"pro-smc-autoflip-keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}","cyan"))
    while True:
        try: s.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# ================== BOOT =========================
if __name__=="__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}","yellow"))
    print(colored(f"Council⇒RF fallback | Peak/Bottom | Trend & AutoFlip | Chop | LowFee | ADX pause<{PAUSE_ADX_THRESHOLD}","yellow"))
    logging.info("service starting…")
    load_state()
    reconcile_with_exchange()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
