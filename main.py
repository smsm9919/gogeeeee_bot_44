# -*- coding: utf-8 -*-
"""
DOGE/USDT — Council + RF (Closed) Pro — Smart Trend Rider (+Bookmap)
Exchange: BingX USDT Perp via CCXT
HTTP: /, /metrics, /health, /bookmap

Objectives:
- Entry only on Council (strong SMC+Bookmap) or RF closed-candle.
- Ride trend with ATR-trail; fast exit on reversal/chop.
- Slippage/Spread protection + state persistence across restarts.
- Position sizing: 60% of current balance × 10x × 0.97 buffer.

ENV ONLY: BINGX_API_KEY, BINGX_API_SECRET, PORT, SELF_URL or RENDER_EXTERNAL_URL
Everything else is hard-coded below.
"""

import os, time, math, random, signal, sys, traceback, logging, json, tempfile
from logging.handlers import RotatingFileHandler
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

import pandas as pd
import ccxt
from flask import Flask, jsonify, request

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# ===== ENV (keys/server only) =====
API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)
PORT       = int(os.getenv("PORT", 5000))
SELF_URL   = (os.getenv("SELF_URL") or os.getenv("RENDER_EXTERNAL_URL") or "").strip().rstrip("/")

# ===== Strategy (hard-coded) =====
SYMBOL        = "DOGE/USDT:USDT"
INTERVAL      = "15m"
LEVERAGE      = 10
RISK_ALLOC    = 0.60
POSITION_MODE = "oneway"  # why: منع التعقيد، صفقة واحدة فقط

# RF (Closed)
RF_SOURCE     = "close"
RF_PERIOD     = 20
RF_MULT       = 3.5
RF_HYST_BPS   = 6.0

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

# Gates
MAX_SPREAD_BPS      = 8.0
HARD_SPREAD_BPS     = 15.0
PAUSE_ADX_THRESHOLD = 17.0  # RF gate

# Council thresholds
ENTRY_VOTES_MIN = 6
ENTRY_SCORE_MIN = 4.0
ENTRY_ADX_MIN   = 22.0
EXIT_VOTES_MIN  = 3

# Voting weights
VOTE_SUPPLY_REJECT = 2;  VOTE_DEMAND_REJECT = 2
VOTE_SWEEP         = 2
VOTE_FVG           = 1
VOTE_EQ_LEVELS     = 1
VOTE_RF_CONFIRM    = 1
VOTE_DI_ADX        = 1
VOTE_RSI_NEUT_TURN = 1
VOTE_BOOKMAP_ACC   = 1   # NEW
VOTE_BOOKMAP_SWEEP = 1   # NEW

# X-Protect (Volatility Explosion Index)
VEI_LOOKBACK = 20
VEI_K        = 2.2

# Execution (slippage)
MAX_SLIP_OPEN_BPS  = 20.0
MAX_SLIP_CLOSE_BPS = 30.0
USE_LIMIT_IOC      = True

# Sizing buffer
SIZE_BUFFER = 0.97

# Management
TP1_PCT_BASE       = 0.40
TP1_CLOSE_FRAC     = 0.40
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.6
RATCHET_LOCK_FALLBACK = 0.60

# Exhaustion exit
EXH_MIN_PROFIT   = 0.35
OPP_RF_HYST_BPS  = 8.0
OPP_STRONG_DEBOUNCE = 2

# Chop
CHOP_LOOKBACK      = 12
CHOP_ATR_FRAC_MAX  = 0.45
CHOP_ALT_BODY_RATE = 0.55
CHOP_EXIT_PROFIT   = 0.25

# Restart
STATE_FILE                  = "state_doge.json"
AUTOSAVE_EVERY_LOOP         = True
AUTOSAVE_ON_ORDER           = True
RESTART_SAFE_BARS_HOLD      = 2
RESTART_STRICT_EXCHANGE_SRC = True

# Rate limits & misc
MAX_TRADES_PER_HOUR = 6
CLOSE_COOLDOWN_S    = 90
FINAL_CHUNK_QTY     = 50.0
RESIDUAL_MIN_QTY    = 9.0
BASE_SLEEP          = 5
NEAR_CLOSE_S        = 1

# ===== Logging =====
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h,"baseFilename","").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready","cyan"))
setup_file_logging()

# ===== Bookmap Adapter (optional) =====
class BookmapAdapter:
    """supply(levels=[(price,liquidity,imbalance,absorption_flag),...]) -> evaluate() zones."""
    def __init__(self):
        self.snapshot = []

    def supply(self, levels):
        self.snapshot = levels or []

    def evaluate(self, pip: float = 0.0005):
        if not self.snapshot:
            return {"accumulation": [], "sweep": [], "walls": []}
        # bucketize
        by_bucket = {}
        for p, liq, imb, ab in self.snapshot:
            key = round(p / pip)
            by_bucket.setdefault(key, []).append((p, liq, imb, ab))
        liqs = [r[1] for r in self.snapshot if r[1] is not None]
        imbs = [r[2] for r in self.snapshot if r[2] is not None]
        liq_avg = max(1e-9, sum(liqs)/max(len(liqs),1))
        imb_avg = (sum(imbs)/max(len(imbs),1)) if imbs else 0.0
        zones_acc, zones_walls, zones_sweep = [], [], []
        for rows in by_bucket.values():
            prices = [r[0] for r in rows]
            lo, hi = min(prices), max(prices)
            liq_sum = sum(r[1] for r in rows)
            imb_mean = sum(r[2] for r in rows)/max(len(rows),1)
            ab_hits = sum(1 for r in rows if r[3])
            if liq_sum > 5 * liq_avg: zones_acc.append((lo, hi))
            if abs(imb_mean) > 2 * abs(imb_avg): zones_walls.append((lo, hi))
            if ab_hits >= 3: zones_sweep.append((lo, hi))
        return {"accumulation": zones_acc, "sweep": zones_sweep, "walls": zones_walls}

bookmap = BookmapAdapter()

# ===== Exchange =====
def make_ex():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap"}
    })
ex = make_ex()
MARKET={}; AMT_PREC=0; LOT_STEP=None; LOT_MIN=None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision",{}) or {}).get("amount", 0) or 0)
        lims = (MARKET.get("limits",{}) or {}).get("amount",{}) or {}
        LOT_STEP = lims.get("step"); LOT_MIN = lims.get("min")
        print(colored(f"🔧 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}","cyan"))
    except Exception as e:
        print(colored(f"⚠️ load_market_specs: {e}","yellow"))

def ensure_leverage_mode():
    try:
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            print(colored(f"✅ leverage set {LEVERAGE}x","green"))
        except Exception as e:
            print(colored(f"⚠️ set_leverage warn: {e}","yellow"))
        print(colored(f"📌 position mode: {POSITION_MODE}","cyan"))
    except Exception as e:
        print(colored(f"⚠️ ensure_leverage_mode: {e}","yellow"))

try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}","yellow"))

# ===== Helpers =====
def _round_amt(q):
    if q is None: return 0.0
    try:
        d=Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step=Decimal(str(LOT_STEP))
            d=(d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec=int(AMT_PREC) if AMT_PREC>=0 else 0
        d=d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d<Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except Exception: return max(0.0, float(q))

def safe_qty(q):
    q=_round_amt(q)
    if q<=0: print(colored(f"⚠️ qty invalid after normalize → {q}","yellow"))
    return q

def fmt(v,d=6,na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isinf(v) or math.isnan(v))): return na
        return f"{float(v):.{d}f}"
    except Exception: return na

def with_retry(fn, tries=3, base=0.4):
    for i in range(tries):
        try: return fn()
        except Exception as e:
            if i==tries-1: raise
            time.sleep(base*(2**i)+random.random()*0.25)

def fetch_ohlcv(limit=600):
    rows=with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t=with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b=with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None

def orderbook_spread_bps():
    try:
        ob=with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid=ob["bids"][0][0] if ob["bids"] else None
        ask=ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid=(bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception: return None

def _interval_seconds(iv:str)->int:
    iv=(iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame)->int:
    tf=_interval_seconds(INTERVAL)
    if len(df)==0: return tf
    cur=int(df["time"].iloc[-1]); now=int(time.time()*1000)
    nxt=cur+tf*1000
    while nxt<=now: nxt+=tf*1000
    return int(max(0,nxt-now)/1000)

def cancel_all_orders():
    if not MODE_LIVE: return
    try: ex.cancel_all_orders(SYMBOL)
    except Exception as e: logging.warning(f"cancel_all_orders: {e}")

# ===== State persistence =====
def _atomic_write_json(path: str, payload: dict):
    try:
        d=os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", delete=False, dir=d, encoding="utf-8") as tmp:
            json.dump(payload, tmp, ensure_ascii=False, separators=(",",":"))
            tmp.flush(); os.fsync(tmp.fileno())
            tmp_path=tmp.name
        os.replace(tmp_path, path)
    except Exception as e: logging.error(f"atomic_write_json: {e}")

def save_state(tag=""):
    snap={
        "STATE": STATE, "compound_pnl": compound_pnl,
        "symbol":SYMBOL, "interval":INTERVAL,
        "ts": int(time.time()*1000), "tag": tag
    }
    _atomic_write_json(STATE_FILE, snap)

def load_state():
    try:
        if not os.path.exists(STATE_FILE): return None
        with open(STATE_FILE,"r",encoding="utf-8") as f: return json.load(f)
    except Exception as e:
        logging.error(f"load_state: {e}"); return None

# ===== Indicators =====
def wilder_ema(s: pd.Series, n:int):
    return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 3:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0,"vei":1.0}
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wilder_ema(tr, ATR_LEN)

    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = wilder_ema(up, RSI_LEN) / wilder_ema(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))

    upm=h.diff(); dnm=l.shift(1)-l
    plus_dm = upm.where((upm>dnm)&(upm>0),0.0)
    minus_dm= dnm.where((dnm>upm)&(dnm>0),0.0)
    plus_di = 100*(wilder_ema(plus_dm,ADX_LEN)/atr.replace(0,1e-12))
    minus_di= 100*(wilder_ema(minus_dm,ADX_LEN)/atr.replace(0,1e-12))
    dx = (100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx = wilder_ema(dx, ADX_LEN)

    # VEI
    rng = (h-l).astype(float)
    try:
        lb = rng.rolling(VEI_LOOKBACK).mean()
        vei = float((rng / lb.replace(0,1e-9)).iloc[-1])
        if math.isinf(vei) or math.isnan(vei): vei = 1.0
    except Exception:
        vei = 1.0

    i=len(df)-1
    return {
        "rsi": float(rsi.iloc[i]),
        "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]),
        "dx": float(dx.iloc[i]),
        "adx": float(adx.iloc[i]),
        "atr": float(atr.iloc[i]),
        "vei": vei
    }

# ===== RF (closed) =====
def _ema(s: pd.Series, n:int): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src: pd.Series, qty: float, n:int)->pd.Series:
    avrng = _ema((src-src.shift(1)).abs(), n); wper=(n*2)-1
    return _ema(avrng, wper)*qty
def _rng_filter(src: pd.Series, rsize: pd.Series):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x-r>prev: cur=x-r
        if x+r<prev: cur=x+r
        rf.append(cur)
    filt=pd.Series(rf, index=src.index, dtype="float64")
    return filt+rsize, filt-rsize, filt
def rf_signal_closed(df: pd.DataFrame):
    if len(df) < RF_PERIOD + 4:
        return {"time": int(time.time()*1000), "price": None, "long": False, "short": False,
                "filter": None, "hi": None, "lo": None}
    d = df.iloc[:-1]
    src = d[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    p_now = float(src.iloc[-1]); p_prev = float(src.iloc[-2])
    f_now = float(filt.iloc[-1]); f_prev = float(filt.iloc[-2])
    def _bps(a,b): 
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    long_flip  = (p_prev <= f_prev and p_now > f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    short_flip = (p_prev >= f_prev and p_now < f_now and _bps(p_now, f_now) >= RF_HYST_BPS)
    return {"time": int(d["time"].iloc[-1]), "price": p_now, "long": bool(long_flip),
            "short": bool(short_flip), "filter": f_now, "hi": float(hi.iloc[-1]), "lo": float(lo.iloc[-1])}

# ===== SMC helpers =====
def _find_swings(df: pd.DataFrame, left:int=2, right:int=2):
    if len(df) < left+right+3: return None, None
    h=df["high"].astype(float).values; l=df["low"].astype(float).values
    ph=[None]*len(df); pl=[None]*len(df)
    for i in range(left, len(df)-right):
        if all(h[i] >= h[j] for j in range(i-left, i+right+1)): ph[i]=h[i]
        if all(l[i] <= l[j] for j in range(i-left, i+right+1)): pl[i]=l[i]
    return ph, pl

def detect_fvg(df: pd.DataFrame, lookback=40):
    if len(df) < 5: return None
    d = df.iloc[-lookback-2:-1] if len(df)>lookback+2 else df.iloc[:-1]
    res=[]
    for i in range(2, len(d)):
        h1,l1 = float(d["high"].iloc[i-2]), float(d["low"].iloc[i-2])
        h3,l3 = float(d["high"].iloc[i]),   float(d["low"].iloc[i])
        if l3 > h1:  res.append({"type":"bull","gap_top":l3,"gap_bot":h1})
        if h3 < l1:  res.append({"type":"bear","gap_top":l1,"gap_bot":h3})
    return res[-1] if res else None

def detect_sweep(df: pd.DataFrame, lookback=30, bps=8.0):
    if len(df)<5: return None
    d=df.iloc[-lookback:]
    o,h,l,c = map(float, d[["open","high","low","close"]].iloc[-1])
    prev_h = float(d["high"].iloc[-2]); prev_l = float(d["low"].iloc[-2])
    def near(a,b):
        try: return abs((a-b)/b)*10000.0 <= bps
        except: return False
    if h>prev_h and c<prev_h and near(h, prev_h): return {"type":"sweep_high"}
    if l<prev_l and c>prev_l and near(l, prev_l): return {"type":"sweep_low"}
    return None

def detect_eq_levels(df: pd.DataFrame, lookback=60, tol_bps=10.0):
    if len(df)<5: return None
    d=df.iloc[-lookback:]
    highs=d["high"].astype(float).values; lows=d["low"].astype(float).values
    mx=max(highs); mn=min(lows)
    nearh=sum(abs((h-mx)/mx)*10000.0<=tol_bps for h in highs)>=3
    nearl=sum(abs((l-mn)/mn)*10000.0<=tol_bps for l in lows)>=3
    if nearh or nearl: return {"eqh":nearh,"eql":nearl}
    return None

def detect_retest_displacement(df: pd.DataFrame, atr: float, mult=1.2, lookback=10):
    if len(df)<lookback+3 or atr<=0: return None
    d=df.iloc[-lookback:]
    body=(d["close"]-d["open"]).abs().astype(float)
    rng =(d["high"]-d["low"]).astype(float)
    if float(body.iloc[-1]) >= mult*atr and float(rng.iloc[-1]) >= mult*atr:
        prev_low=float(d["low"].iloc[-2]); prev_high=float(d["high"].iloc[-2])
        last_low=float(d["low"].iloc[-1]); last_high=float(d["high"].iloc[-1])
        if last_low<=prev_high or last_high>=prev_low:
            return {"type":"displacement_retest"}
    return None

def detect_trap_wick(df: pd.DataFrame, ratio=0.6):
    if len(df)<3: return None
    o,h,l,c = map(float, df[["open","high","low","close"]].iloc[-1])
    rng=max(h-l,1e-12); up=h-max(o,c); dn=min(o,c)-l
    if (dn/rng)>=ratio and c>o: return {"type":"bull_trap_reject"}
    if (up/rng)>=ratio and c<o: return {"type":"bear_trap_reject"}
    return None

def detect_boxes(df: pd.DataFrame):
    d=df.iloc[:-1] if len(df)>=2 else df
    ph,pl=_find_swings(d,2,2)
    highs=[p for p in ph if p is not None][-20:]
    lows =[p for p in pl if p is not None][-20:]
    sup=dem=None
    if highs:
        top=max(highs); bot=top - (top-min(highs))*0.25
        sup={"side":"supply","top":top,"bot":bot}
    if lows:
        bot=min(lows); top=bot + (max(lows)-bot)*0.25 if len(lows)>1 else bot*1.002
        dem={"side":"demand","top":top,"bot":bot}
    return {"supply":sup,"demand":dem}

def touch_reject(df: pd.DataFrame, box):
    if not box or len(df)<2: return False
    o,h,l,c = map(float, df[["open","high","low","close"]].iloc[-1])
    rng=max(h-l,1e-12); up=h-max(o,c); dn=min(o,c)-l
    mid=(box["top"]+box["bot"])/2.0
    if box["side"]=="supply" and (h>=box["bot"]) and c<mid and (up/rng)>=0.5: return True
    if box["side"]=="demand" and (l<=box["top"]) and c>mid and (dn/rng)>=0.5: return True
    return False

# ===== Council =====
class Council:
    def __init__(self):
        self.state={"open":False,"side":None,"entry":None}
        self._last_log=None

    def votes(self, df: pd.DataFrame, ind: dict, rf: dict):
        b=s=0; score=0.0; rb=[]; rs=[]
        boxes=detect_boxes(df); sup=boxes.get("supply"); dem=boxes.get("demand")

        # Bookmap influence
        bm = bookmap.evaluate()
        if bm["accumulation"]:
            b += VOTE_BOOKMAP_ACC; score += 0.5; rb.append("BM-acc")
        if bm["sweep"]:
            s += VOTE_BOOKMAP_SWEEP; score += 0.5; rs.append("BM-sweep")

        # Box rejects
        if touch_reject(df, dem): b += VOTE_DEMAND_REJECT; score+=1.6; rb.append("reject@demand")
        if touch_reject(df, sup): s += VOTE_SUPPLY_REJECT; score+=1.6; rs.append("reject@supply")

        # Sweeps
        sw=detect_sweep(df)
        if sw:
            if sw["type"]=="sweep_low":  b+=VOTE_SWEEP; score+=0.6; rb.append("sweep_low")
            else:                        s+=VOTE_SWEEP; score+=0.6; rs.append("sweep_high")

        # FVG
        fvg=detect_fvg(df)
        if fvg:
            if fvg["type"]=="bull": b+=VOTE_FVG; score+=0.5; rb.append("FVG(bull)")
            else:                   s+=VOTE_FVG; score+=0.5; rs.append("FVG(bear)")

        # Displacement/Retest
        atr=float(ind.get("atr") or 0.0)
        disp=detect_retest_displacement(df, atr, mult=1.2, lookback=10)
        if disp:
            if float(df["close"].iloc[-1])>float(df["open"].iloc[-1]): b+=1; score+=0.7; rb.append("displacement")
            else: s+=1; score+=0.7; rs.append("displacement")

        # Trap wick
        trap=detect_trap_wick(df, 0.6)
        if trap:
            if trap["type"]=="bull_trap_reject": b+=1; score+=0.6; rb.append("trap_reject")
            else: s+=1; score+=0.6; rs.append("trap_reject")

        # DI/ADX + Trend strength
        pdi,mdi,adx = ind.get("plus_di",0), ind.get("minus_di",0), ind.get("adx",0)
        if adx>=18 and pdi>mdi: b+=VOTE_DI_ADX; score+=0.5; rb.append("DI+>DI- & ADX")
        if adx>=18 and mdi>pdi: s+=VOTE_DI_ADX; score+=0.5; rs.append("DI->DI+ & ADX")

        # RSI neutral turn
        rsi=ind.get("rsi",50.0); o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
        if 45<=rsi<=55:
            if c>o: b+=VOTE_RSI_NEUT_TURN; score+=0.5; rb.append("RSI_neutral_up")
            else:   s+=VOTE_RSI_NEUT_TURN; score+=0.5; rs.append("RSI_neutral_down")

        # RF confirm-only
        if rf.get("long"):  b+=VOTE_RF_CONFIRM; score+=0.5; rb.append("RF-long")
        if rf.get("short"): s+=VOTE_RF_CONFIRM; score+=0.5; rs.append("RF-short")

        # X-Protect
        vei=float(ind.get("vei") or 1.0)
        if vei>=VEI_K:
            if c>o: b+=1; score+=0.5; rb.append("VEI+")
            else:   s+=1; score+=0.5; rs.append("VEI-")

        self._last_log = f"🏛 SCM | BUY={b} [{', '.join(rb) or '—'}] | SELL={s} [{', '.join(rs) or '—'}] | score={score:.2f} | ADX={ind.get('adx'):.1f}"
        print(colored(self._last_log, "green" if b>s else "red" if s>b else "cyan"))
        return b, s, score

    def decide(self, df, ind, rf):
        b,s,score = self.votes(df, ind, rf)
        adx=float(ind.get("adx") or 0.0)
        entry=None; exit_=None
        if not self.state["open"]:
            if b>=ENTRY_VOTES_MIN and score>=ENTRY_SCORE_MIN and adx>=ENTRY_ADX_MIN:
                self.state.update({"open":True,"side":"long","entry":float(df['close'].iloc[-1])})
                entry={"side":"buy","reason":self._last_log}
            elif s>=ENTRY_VOTES_MIN and score>=ENTRY_SCORE_MIN and adx>=ENTRY_ADX_MIN:
                self.state.update({"open":True,"side":"short","entry":float(df['close'].iloc[-1])})
                entry={"side":"sell","reason":self._last_log}
        return {"entry":entry,"exit":exit_,"log":self._last_log}

council = Council()

# ===== State & Orders =====
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0, "opp_votes": 0,
    "_last_entry_ts": 0, "_last_close_ts": 0, "_rf_debounce": 0,
    "_reversal_guard_bars": 0
}
compound_pnl=0.0
wait_for_next_signal_side=None
RESTART_HOLD_UNTIL_BAR=0
_trades_timestamps=[]

def _within_hour_rate_limit()->bool:
    now=time.time()
    while _trades_timestamps and now-_trades_timestamps[0]>3600: _trades_timestamps.pop(0)
    return len(_trades_timestamps) < MAX_TRADES_PER_HOUR

def _mark_trade_timestamp(): _trades_timestamps.append(time.time())

def _params_open(side):
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if side=="buy" else "SHORT", "reduceOnly":False}
    return {"positionSide":"BOTH","reduceOnly":False}

def _params_close():
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if STATE.get("side")=="long" else "SHORT","reduceOnly":True}
    return {"positionSide":"BOTH","reduceOnly":True}

def _best_quotes():
    ob=with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
    bid=ob["bids"][0][0] if ob["bids"] else None
    ask=ob["asks"][0][0] if ob["asks"] else None
    mid=(bid+ask)/2.0 if (bid and ask) else price_now()
    return bid, ask, mid

def _ioc_price(side: str, mid: float, max_bps: float) -> float:
    if mid is None or mid<=0: return None
    slip = max_bps/10000.0
    return (mid*(1+slip)) if side=="buy" else (mid*(1-slip))

def _create_order_ioc(symbol, side, qty, limit_price, reduce_only=False):
    # why: IOC للحد من الانزلاق
    params={"timeInForce":"IOC","reduceOnly":reduce_only}
    if POSITION_MODE=="hedge":
        params["positionSide"] = "LONG" if (side=="buy") else "SHORT"
    return ex.create_order(symbol, "limit", side, qty, limit_price, params)

def _read_position():
    try:
        poss=with_retry(lambda: ex.fetch_positions(params={"type":"swap"}))
        for p in poss:
            sym=(p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0] not in sym: continue
            qty=abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty<=0: return 0.0,None,None
            entry=float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            # why: بعض البورصات لا تُرجع 'side' ثابتًا
            side_raw=(p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side="long" if "long" in side_raw or float(p.get("cost",0))>0 else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position: {e}")
    return 0.0,None,None

def compute_size(balance, price):
    cap=(balance or 0.0)*RISK_ALLOC*LEVERAGE*SIZE_BUFFER
    raw=max(0.0, cap/max(float(price or 0.0),1e-9))
    return safe_qty(raw)

def open_market(side, qty, price, tag=""):
    if qty<=0: print(colored("❌ skip open (qty<=0)","red")); return False
    if STATE["_reversal_guard_bars"]>0 and side in ("buy","sell"):
        print(colored("⛔ Reversal-Guard active — council-only entries","yellow")); return False
    spr=orderbook_spread_bps()
    if spr is not None and (spr>HARD_SPREAD_BPS or spr>MAX_SPREAD_BPS):
        print(colored(f"⛔ spread {fmt(spr,2)}bps — guard","yellow")); return False
    if not _within_hour_rate_limit():
        print(colored("⛔ rate-limit: too many trades/hour","yellow")); return False
    _,_,mid=_best_quotes()
    if MODE_LIVE and USE_LIMIT_IOC:
        limit_price=_ioc_price(side, mid, MAX_SLIP_OPEN_BPS)
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception: pass
            _create_order_ioc(SYMBOL, side, qty, limit_price, reduce_only=False)
        except Exception as e:
            print(colored(f"❌ IOC open fail: {e}","red")); logging.error(e); return False
    elif MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty,None,_params_open(side))
        except Exception as e: print(colored(f"❌ market open fail: {e}","red")); logging.error(e); return False
    STATE.update({
        "open":True, "side":"long" if side=="buy" else "short", "entry":price,
        "qty":qty, "pnl":0.0, "bars":0, "trail":None, "breakeven":None,
        "tp1_done":False, "highest_profit_pct":0.0, "profit_targets_achieved":0,
        "opp_votes":0, "_last_entry_ts": int(time.time())
    })
    _mark_trade_timestamp()
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)} {tag}","green" if side=="buy" else "red"))
    if AUTOSAVE_ON_ORDER: save_state(tag="open")
    return True

def _reset_after_close(reason, prev_side=None):
    global wait_for_next_signal_side
    prev_side = prev_side or STATE.get("side")
    STATE.update({
        "open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,
        "trail":None,"breakeven":None,"tp1_done":False,
        "highest_profit_pct":0.0,"profit_targets_achieved":0,
        "opp_votes":0,"_last_close_ts": int(time.time())
    })
    wait_for_next_signal_side = "sell" if prev_side=="long" else "buy" if prev_side=="short" else None

def close_market_strict(reason="STRICT"):
    global compound_pnl
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty<=0 and not STATE.get("open"): return
    if exch_qty<=0 and STATE.get("open"):
        px = price_now() or STATE["entry"]; entry=STATE["entry"]; side=STATE["side"]
        pnl=(px-entry)*STATE["qty"]*(1 if side=="long" else -1); compound_pnl+=pnl
        print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
        _reset_after_close(reason, prev_side=side)
        if AUTOSAVE_ON_ORDER: save_state(tag="strict_close")
        return
    side_to_close="sell" if exch_side=="long" else "buy"
    qty_to_close=safe_qty(exch_qty)
    attempts=0; last=None
    while attempts<6:
        try:
            if MODE_LIVE and USE_LIMIT_IOC:
                _,_,mid=_best_quotes()
                limit_price=_ioc_price(side_to_close, mid, MAX_SLIP_CLOSE_BPS)
                _create_order_ioc(SYMBOL, side_to_close, qty_to_close, limit_price, reduce_only=True)
            elif MODE_LIVE:
                params=_params_close(); params["reduceOnly"]=True
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            time.sleep(2.0)
            left,_,_= _read_position()
            if left<=0:
                px=price_now() or STATE.get("entry") or exch_entry
                entry_px=STATE.get("entry") or exch_entry or px
                side=STATE.get("side") or exch_side
                pnl=(px-entry_px)*exch_qty*(1 if side=="long" else -1); compound_pnl+=pnl
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                _reset_after_close(reason, prev_side=side)
                if AUTOSAVE_ON_ORDER: save_state(tag="strict_close")
                return
            qty_to_close=safe_qty(left); attempts+=1
            print(colored(f"⚠️ strict close retry {attempts} residual={fmt(left,4)}","yellow"))
        except Exception as e:
            last=e; attempts+=1; time.sleep(2.0)
    print(colored(f"❌ STRICT CLOSE FAILED last_error={last}","red"))

def close_partial(frac, reason):
    if not STATE["open"] or STATE["qty"]<=0: return
    qty_close=safe_qty(max(0.0, STATE["qty"]*min(max(frac,0.0),1.0)))
    px=price_now() or STATE["entry"]
    min_unit=max(RESIDUAL_MIN_QTY, LOT_MIN or RESIDUAL_MIN_QTY)
    if qty_close<min_unit: 
        print(colored(f"⏸️ skip partial (amount={fmt(qty_close,4)} < min_unit={fmt(min_unit,4)})","yellow")); return
    side="sell" if STATE["side"]=="long" else "buy"
    if MODE_LIVE:
        try: ex.create_order(SYMBOL,"market",side,qty_close,None,_params_close())
        except Exception as e: print(colored(f"❌ partial: {e}","red")); return
    pnl=(px-STATE["entry"])*qty_close*(1 if STATE["side"]=="long" else -1)
    STATE["qty"]=safe_qty(STATE["qty"]-qty_close)
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(STATE['qty'],4)}","magenta"))
    if STATE["qty"]<=FINAL_CHUNK_QTY and STATE["qty"]>0:
        close_market_strict("FINAL_CHUNK_RULE")

# ===== Restart reconcile =====
def reconcile_state_with_exchange():
    global compound_pnl, RESTART_HOLD_UNTIL_BAR
    loaded=load_state()
    exch_qty, exch_side, exch_entry=_read_position()
    exch_open = exch_qty>0 and exch_side in ("long","short") and exch_entry and exch_entry>0
    if loaded and (loaded.get("symbol")==SYMBOL and loaded.get("interval")==INTERVAL):
        st=loaded.get("STATE") or {}
        STATE["highest_profit_pct"]=float(st.get("highest_profit_pct") or 0.0)
        STATE["trail"]=st.get("trail"); STATE["breakeven"]=st.get("breakeven")
        STATE["profit_targets_achieved"]=int(st.get("profit_targets_achieved") or 0)
        try: globals()["compound_pnl"]=float(loaded.get("compound_pnl") or 0.0)
        except Exception: pass
        print(colored("💾 loaded local state (non-pos)","cyan"))
    if RESTART_STRICT_EXCHANGE_SRC:
        if exch_open:
            STATE.update({"open":True,"side":exch_side,"entry":float(exch_entry),
                          "qty":float(exch_qty),"pnl":0.0,"bars":0})
            px=price_now() or exch_entry
            STATE["pnl"]=(px-exch_entry)*exch_qty*(1 if exch_side=="long" else -1)
            RESTART_HOLD_UNTIL_BAR=RESTART_SAFE_BARS_HOLD
            print(colored(f"♻️ resumed live position {exch_side} qty={fmt(exch_qty,4)} entry={fmt(exch_entry)}","yellow"))
        else:
            STATE.update({"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0})
            print(colored("♻️ no live position — flat","yellow"))
    save_state(tag="reconcile_boot")

# ===== Management =====
def _consensus(ind, info, side)->float:
    score=0.0
    adx=float(ind.get("adx") or 0.0); rsi=float(ind.get("rsi") or 50.0)
    if (side=="long" and rsi>=55) or (side=="short" and rsi<=45): score+=1.0
    if adx>=28: score+=1.0
    elif adx>=20: score+=0.5
    if info.get("filter") and info.get("price"):
        try:
            if abs(info["price"]-info["filter"])/max(info["filter"],1e-9) >= (RF_HYST_BPS/10000.0): score+=0.5
        except Exception: pass
    return score

def _tp_ladder(info, ind, side):
    px=info["price"]; atr=float(ind.get("atr") or 0.0)
    atr_pct=(atr/max(px,1e-9))*100.0 if px else 0.5
    score=_consensus(ind, info, side)
    mults = [1.8,3.2,5.0] if score>=2.5 else [1.6,2.8,4.5] if score>=1.5 else [1.2,2.4,4.0]
    return [round(m*atr_pct,2) for m in mults],[0.25,0.30,0.45]

def is_chop(df: pd.DataFrame, atr: float) -> bool:
    if len(df) < CHOP_LOOKBACK+2 or atr<=0: return False
    d = df.iloc[-CHOP_LOOKBACK-1:-1]
    ranges = (d["high"]-d["low"]).astype(float)
    avg_range = float(ranges.mean())
    if avg_range <= CHOP_ATR_FRAC_MAX * atr:
        bodies = (d["close"]>d["open"]).astype(int).values
        altern = sum(bodies[i]!=bodies[i-1] for i in range(1,len(bodies))) / max(len(bodies)-1,1)
        return altern >= CHOP_ALT_BODY_RATE
    return False

def should_take_reversal_profit(df: pd.DataFrame, ind: dict, side: str, rr_pct: float) -> bool:
    adx = float(ind.get("adx") or 0.0)
    if rr_pct <= 0.0: return False
    o,h,l,c = map(float, df[["open","high","low","close"]].iloc[-1])
    rng=max(h-l,1e-12); up=h-max(o,c); dn=min(o,c)-l
    wick_against = (side=="long" and up/rng>=0.6) or (side=="short" and dn/rng>=0.6)
    sw = detect_sweep(df)
    return wick_against and (adx<18 or sw is not None)

def defensive_on_opposite_rf(ind: dict, info: dict):
    if not STATE["open"] or STATE["qty"]<=0: return
    STATE["opp_votes"]=int(STATE.get("opp_votes",0))+1
    adx=float(ind.get("adx") or 0.0)
    px=info.get("price"); rf=info.get("filter")
    hyst=0.0
    try:
        if px and rf: hyst = abs((px-rf)/rf)*10000.0
    except Exception: pass
    if STATE["opp_votes"]>=OPP_STRONG_DEBOUNCE and adx>=28 and hyst>=OPP_RF_HYST_BPS:
        close_market_strict("OPPOSITE_RF_CONFIRMED")

def manage_after_entry(df, ind, info):
    if not STATE["open"] or STATE["qty"]<=0: return
    px=info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr=(px-entry)/entry*100*(1 if side=="long" else -1)

    # Early harvest / TP1
    tp1_now=TP1_PCT_BASE*(2.2 if ind.get("adx",0)>=35 else 1.8 if ind.get("adx",0)>=28 else 1.0)
    if (not STATE["tp1_done"]) and rr>=tp1_now:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{tp1_now:.2f}%"); STATE["tp1_done"]=True
        if rr>=BREAKEVEN_AFTER: STATE["breakeven"]=entry

    # Dynamic ladder
    dyn_tps,dyn_fracs=_tp_ladder(info, ind, side)
    k=int(STATE.get("profit_targets_achieved",0))
    if k<len(dyn_tps) and rr>=dyn_tps[k]:
        close_partial(dyn_fracs[k], f"TP_dyn@{dyn_tps[k]:.2f}%")
        STATE["profit_targets_achieved"]=k+1

    # Ratchet lock
    if rr>STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=rr
    if STATE["highest_profit_pct"]>=TRAIL_ACTIVATE_PCT and rr<STATE["highest_profit_pct"]*RATCHET_LOCK_FALLBACK:
        close_partial(0.50, f"Ratchet {STATE['highest_profit_pct']:.2f}%→{rr:.2f}%")

    # ATR trail
    atr=float(ind.get("atr") or 0.0)
    if rr>=TRAIL_ACTIVATE_PCT and atr>0:
        gap=atr*ATR_TRAIL_MULT
        if side=="long":
            new=px-gap; STATE["trail"]=max(STATE["trail"] or new, new)
            if STATE["breakeven"] is not None: STATE["trail"]=max(STATE["trail"], STATE["breakeven"])
            if px<STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")
        else:
            new=px+gap; STATE["trail"]=min(STATE["trail"] or new, new)
            if STATE["breakeven"] is not None: STATE["trail"]=min(STATE["trail"], STATE["breakeven"])
            if px>STATE["trail"]: close_market_strict(f"TRAIL_ATR({ATR_TRAIL_MULT}x)")

# ===== UI =====
def pretty_snapshot(bal, info, ind, spread_bps, reason=None, df=None, council_log=None):
    left_s=time_to_candle_close(df) if df is not None else 0
    print(colored("─"*110,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*110,"cyan"))
    print("📈 RF CLOSED")
    print(f"   💲 Price {fmt(info.get('price'))} | filt={fmt(info.get('filter'))} hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))} | spread={fmt(spread_bps,2)}bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))} +DI={fmt(ind.get('plus_di'))} -DI={fmt(ind.get('minus_di'))} ADX={fmt(ind.get('adx'))} ATR={fmt(ind.get('atr'))} VEI~{fmt(ind.get('vei'),2)}")
    if council_log: print(colored(council_log,"white"))
    print(f"   ⏱️ closes_in ≈ {left_s}s")
    print("\n🧭 POSITION")
    bal_line=f"Balance={fmt(bal,2)} Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x CompoundPnL={fmt(compound_pnl)} Eq~{fmt((bal or 0)+compound_pnl,2)}"
    print(colored(f"   {bal_line}","yellow"))
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp} Entry={fmt(STATE['entry'])} Qty={fmt(STATE['qty'],4)} Bars={STATE['bars']} Trail={fmt(STATE['trail'])} BE={fmt(STATE['breakeven'])}")
        print(f"   🎯 TP_done={STATE['profit_targets_achieved']} HP={fmt(STATE['highest_profit_pct'],2)}% OppVotes={STATE.get('opp_votes',0)} GuardBars={STATE.get('_reversal_guard_bars',0)}")
    else:
        print("   ⚪ FLAT")
        if wait_for_next_signal_side: print(colored(f"   ⏳ Waiting opposite RF: {wait_for_next_signal_side.upper()}","cyan"))
    if reason: print(colored(f"   ℹ️ reason: {reason}","white"))
    print(colored("─"*110,"cyan"))

# ===== Main loop =====
app=Flask(__name__)

def trade_loop():
    global wait_for_next_signal_side, RESTART_HOLD_UNTIL_BAR
    reconcile_state_with_exchange()
    last_decision_bar_time=0
    while True:
        try:
            bal=balance_usdt()
            df = fetch_ohlcv()
            ind=compute_indicators(df)
            rf = rf_signal_closed(df)
            spread=orderbook_spread_bps()
            px=price_now() or rf["price"] or STATE.get("entry") or 0.0

            # Update pnl
            if STATE["open"] and px:
                STATE["pnl"]=(px-STATE["entry"])*STATE["qty"]*(1 if STATE["side"]=="long" else -1)

            # Council
            council_decision = council.decide(df, ind, rf)
            council_log = council_decision.get("log")

            # Exit logic priority
            reason=None
            if STATE["open"]:
                # Opposite RF defense
                opp = (STATE["side"]=="long" and rf["short"]) or (STATE["side"]=="short" and rf["long"])
                if opp: defensive_on_opposite_rf(ind, {"price":px, **rf})

                # Chop exit
                if is_chop(df, float(ind.get("atr") or 0.0)):
                    rr = (px-STATE["entry"])/STATE["entry"]*100*(1 if STATE["side"]=="long" else -1)
                    if rr>=CHOP_EXIT_PROFIT:
                        print(colored(f"⚪ Chop → light profit {rr:.2f}%","yellow"))
                        close_market_strict("CHOP_EXIT")
                        STATE["_reversal_guard_bars"]=4

                # Reversal smart take
                rr = (px-STATE["entry"])/STATE["entry"]*100*(1 if STATE["side"]=="long" else -1)
                if should_take_reversal_profit(df, ind, STATE["side"], rr) and rr>=EXH_MIN_PROFIT:
                    print(colored(f"🔄 Reversal risk → lock profit {rr:.2f}%","yellow"))
                    close_market_strict("REVERSAL_LOCK")
                    STATE["_reversal_guard_bars"]=4

            # Manage position (trail/ratchet/TP)
            manage_after_entry(df, ind, {"price":px, **rf})

            # Gates
            if spread is not None and spread>HARD_SPREAD_BPS:
                reason=f"hard spread guard {fmt(spread,2)}bps>{HARD_SPREAD_BPS}"
            elif spread is not None and spread>MAX_SPREAD_BPS:
                reason=f"spread guard {fmt(spread,2)}bps>{MAX_SPREAD_BPS}"
            if reason is None and (float(ind.get("adx") or 0.0)<PAUSE_ADX_THRESHOLD):
                reason=f"ADX<{PAUSE_ADX_THRESHOLD:.0f} — RF paused"

            # Entry arbitration (one decision per closed bar)
            decision_time = rf["time"]
            new_bar = decision_time != last_decision_bar_time
            if new_bar:
                last_decision_bar_time = decision_time
                # decay guards
                if STATE["_reversal_guard_bars"]>0: STATE["_reversal_guard_bars"]-=1
                if RESTART_HOLD_UNTIL_BAR>0: RESTART_HOLD_UNTIL_BAR-=1

                if not STATE["open"] and reason is None and RESTART_HOLD_UNTIL_BAR<=0:
                    sig=None; tag=""
                    # Council first (must reach quorum)
                    if council_decision["entry"]:
                        sig=council_decision["entry"]["side"]; tag=f"[COUNCIL] {council_decision['entry']['reason']}"
                    else:
                        # RF fallback if ADX gate passes and reversal guard inactive
                        if STATE["_reversal_guard_bars"]==0 and ((rf["long"] or rf["short"]) and float(ind.get("adx") or 0.0)>=PAUSE_ADX_THRESHOLD):
                            sig="buy" if rf["long"] else "sell"; tag=f"[RF-closed]"
                    # wait-for-opposite logic after close
                    if sig:
                        if wait_for_next_signal_side and sig != wait_for_next_signal_side:
                            reason=f"waiting opposite RF: need {wait_for_next_signal_side.upper()}"
                        elif (time.time()-STATE.get("_last_close_ts",0)) < CLOSE_COOLDOWN_S:
                            reason=f"cooldown {int(CLOSE_COOLDOWN_S - (time.time()-STATE.get('_last_close_ts',0)))}s"
                        elif not _within_hour_rate_limit():
                            reason="rate-limit trades/hour"
                        else:
                            qty=compute_size(bal, px or rf["price"])
                            if qty>0 and (px or rf["price"]):
                                if open_market(sig, qty, px or rf["price"], tag):
                                    wait_for_next_signal_side=None
                            else:
                                reason="qty<=0 or price=None"

            # Snapshot
            pretty_snapshot(bal, {"price":px, **rf}, ind, spread, reason, df, council_log)

            # Bars counter
            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"]+=1

            # Autosave
            if AUTOSAVE_EVERY_LOOP: save_state(tag="loop")

            # Sleep pacing
            time.sleep(NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP)
        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}","red"))
            logging.error(f"loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# ===== HTTP / keepalive =====
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ DOGE Council+RF — {SYMBOL} {INTERVAL} — {mode} — IOC/Slippage — Restart-safe"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol":SYMBOL,"interval":INTERVAL,"mode":"live" if MODE_LIVE else "paper",
        "leverage":LEVERAGE,"risk_alloc":RISK_ALLOC,"price":price_now(),
        "state":STATE,"compound_pnl":compound_pnl,
        "council_log": council._last_log,
        "guards":{"max_spread_bps":MAX_SPREAD_BPS,"hard_spread_bps":HARD_SPREAD_BPS,"pause_adx":PAUSE_ADX_THRESHOLD}
    })

@app.route("/health")
def health():
    return jsonify({"ok":True,"ts":datetime.utcnow().isoformat(),"open":STATE["open"],"side":STATE["side"],"qty":STATE["qty"]}),200

@app.route("/bookmap", methods=["POST"])
def bookmap_feed():
    """body: {"levels":[[price, liquidity, imbalance, absorption_flag], ...]}"""
    try:
        payload = request.get_json(silent=True) or {}
        levels = payload.get("levels", [])
        parsed=[]
        for row in levels:
            p=float(row[0]); liq=float(row[1]); imb=float(row[2]); ab=int(row[3])
            parsed.append((p,liq,imb,ab))
        bookmap.supply(parsed)
        return jsonify({"ok":True,"count":len(parsed)})
    except Exception as e:
        return jsonify({"ok":False,"error":str(e)}),400

def keepalive_loop():
    if not SELF_URL:
        print(colored("⛔ keepalive disabled (no SELF_URL/RENDER_EXTERNAL_URL)","yellow")); return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"doge-council-pro/keepalive"})
    print(colored(f"KEEPALIVE every 50s → {SELF_URL}","cyan"))
    while True:
        try: sess.get(SELF_URL, timeout=8)
        except Exception: pass
        time.sleep(50)

# ===== Boot =====
if __name__=="__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'} • {SYMBOL} • {INTERVAL}","yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}%×{LEVERAGE}x • ENTRY: Council ⇒ RF (closed)","yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
