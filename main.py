# -*- coding: utf-8 -*-
"""
DOGE/USDT — BingX Perp via CCXT
SMC Council + Trend + Auto-Flip + RF(Closed) Fallback
- Council: Supply/Demand boxes, EQH/EQL, Sweep/Grab, Displacement, Retest, Trap
- Trend Engine: HH/HL أو LH/LL + ADX
- Auto-Flip With Trend: يقفل ويقلب بعد تأكيد متعدد
- Runtime Council: Wick/Impulse Harvest + Tighten Trail + Strict Close عند الذروة
- Post-entry: TP ladder + BE + ATR Trail + Ratchet + Strict HP Close
- RF on CLOSED candle فقط كاحتياطي
HTTP: / , /metrics , /health
"""

import os, time, math, random, signal, sys, traceback, logging, threading
from logging.handlers import RotatingFileHandler
from datetime import datetime
from decimal import Decimal, ROUND_DOWN

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

# ================== FIXED SETTINGS ===============
SYMBOL        = "DOGE/USDT:USDT"
INTERVAL      = "15m"
LEVERAGE      = 10
RISK_ALLOC    = 0.60
POSITION_MODE = "oneway"   # or "hedge"

# -------- Range Filter (Closed-candle fallback) -----
RF_SOURCE   = "close"
RF_PERIOD   = 20
RF_MULT     = 3.5
RF_HYST_BPS = 6.0
ENTRY_FROM_RF = True   # fallback فقط

# -------- Indicators lengths ------------------------
RSI_LEN = 14; ADX_LEN = 14; ATR_LEN = 14

# -------- Guards ------------------------------------
MAX_SPREAD_BPS      = 8.0
PAUSE_ADX_THRESHOLD = 15.0
WAIT_NEXT_CLOSED    = True

# -------- Trend engine ------------------------------
TREND_ADX_MIN   = 30.0
STRUCT_BARS     = 48
OPP_RF_DEBOUNCE = 2

# -------- SCM / Liquidity ---------------------------
EQ_BPS             = 10.0
SWEEP_LOOKBACK     = 60
DSP_ATR_MIN        = 1.2
RETEST_BPS         = 15.0
TRAP_CLOSE_IN_BARS = 3

# -------- Runtime Wick/Impulse Harvest --------------
WICK_MIN_RATIO            = 0.55
WICK_EXTREME_RATIO        = 0.70
IMPULSE_ATR_MULT          = 2.2
IMPULSE_EXTREME_ATR_MULT  = 3.0
HARVEST_STEP_FRAC         = 0.33
HARVEST_MAX_ROUNDS        = 2
CLOSE_ON_EXTREME_PROFIT_PCT = 2.8
ADX_COOL_FOR_EXTREME_CLOSE  = 22.0
RUNTIME_LOG = True

# -------- Management --------------------------------
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

# -------- Auto-Flip With Trend ----------------------
FLIP_RF_BARS_CONFIRMED = 2
FLIP_MIN_ADX           = 25.0
FLIP_REQUIRE_DI        = True
FLIP_REQUIRE_BOS       = True

# Pacing
BASE_SLEEP=5; NEAR_CLOSE_S=1

# ================== LOGGING =========================
def setup_logs():
    logger=logging.getLogger(); logger.setLevel(logging.INFO)
    if not any(isinstance(h,RotatingFileHandler) and getattr(h,"baseFilename","").endswith("bot.log") for h in logger.handlers):
        fh=RotatingFileHandler("bot.log",maxBytes=5_000_000,backupCount=7,encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready","cyan"))
setup_logs()

# ================== EXCHANGE ========================
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

# ================== HELPERS =========================
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

# ================== INDICATORS ======================
def wema(s,n): return s.ewm(alpha=1/n,adjust=False).mean()
def compute_indicators(df):
    if len(df)<max(RSI_LEN,ADX_LEN,ATR_LEN)+2:
        return {"rsi":50,"plus_di":0,"minus_di":0,"adx":0,"atr":0}
    c,h,l,v = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float), df["volume"].astype(float)
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

# ================== RF (Closed) =====================
def _ema(s,n): return s.ewm(span=n,adjust=False).mean()
def _rng_size(src,qty,n):
    avr=_ema((src-src.shift(1)).abs(),n); return _ema(avr,(n*2)-1)*qty
def _rng_filter(src,rsize):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x-r>prev: cur=x-r
        if x+r<prev: cur=x+r
        rf.append(cur)
    filt=pd.Series(rf,index=src.index,dtype="float64")
    return filt+rsize, filt-rsize, filt
def rf_signal_closed(df):
    if len(df)<RF_PERIOD+3:
        return {"time":int(time.time()*1000),"price":None,"long":False,"short":False,"filter":None,"hi":None,"lo":None}
    d=df.iloc[:-1]; src=d[RF_SOURCE].astype(float)
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

# ================== SMC Utilities ===================
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
    highs=d["high"].astype(float).values
    lows =d["low"].astype(float).values
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
        lvl=float(d["low"].iloc[L[-1]])
        last=d.iloc[-1]; prev=d.iloc[-2]
        if float(last["low"])<lvl and float(last["close"])>float(prev["close"]):
            body=abs(float(last["close"])-float(last["open"]))
            if body/atr>=DSP_ATR_MIN:
                return {"type":"sweep_down","lvl":lvl,"dsp":body/atr}
    if side=="short" and H:
        lvl=float(d["high"].iloc[H[-1]])
        last=d.iloc[-1]; prev=d.iloc[-2]
        if float(last["high"])>lvl and float(last["close"])<float(prev["close"]):
            body=abs(float(last["close"])-float(last["open"]))
            if body/atr>=DSP_ATR_MIN:
                return {"type":"sweep_up","lvl":lvl,"dsp":body/atr}
    return None

def liquidity_flow(df, ind):
    o,h,l,c = map(float, df[["open","high","low","close"]].iloc[-1])
    rng=max(h-l,1e-12); body=abs(c-o)
    v=float(df["volume"].iloc[-1])
    vol_ema = float(df["volume"].ewm(alpha=1/34, adjust=False).mean().iloc[-1])
    vol_spike = v >= vol_ema*1.8
    spread_ok = (body/rng) >= 0.15
    if vol_spike and spread_ok:
        return "inflow_up" if c>o else "inflow_down"
    return "neutral"

# ================== Wick/Impulse Analysis ===========
def analyze_wick_impulse(df, atr_val: float):
    res = {"upper_ratio":0.0,"lower_ratio":0.0,"body_atr":0.0,
           "dir":"flat","wick_up":False,"wick_down":False,
           "impulse":False,"grade":"none"}
    if len(df) < 2 or atr_val <= 0: return res
    o,h,l,c = map(float, df[["open","high","low","close"]].iloc[-2])  # آخر مغلقة
    rng = max(h-l, 1e-12)
    up  = h - max(o, c)
    dn  = min(o, c) - l
    body = abs(c - o)
    res["upper_ratio"] = up/rng
    res["lower_ratio"] = dn/rng
    res["body_atr"]    = body/max(atr_val,1e-12)
    res["dir"] = "up" if c>o else "down" if c<o else "flat"
    res["wick_up"]   = res["upper_ratio"] >= WICK_MIN_RATIO
    res["wick_down"] = res["lower_ratio"] >= WICK_MIN_RATIO
    res["impulse"]   = res["body_atr"]   >= IMPULSE_ATR_MULT
    extreme_wick = (res["upper_ratio"]>=WICK_EXTREME_RATIO) or (res["lower_ratio"]>=WICK_EXTREME_RATIO)
    extreme_imp  = (res["body_atr"]>=IMPULSE_EXTREME_ATR_MULT)
    if extreme_wick or extreme_imp: res["grade"]="extreme"
    elif res["impulse"] or res["wick_up"] or res["wick_down"]: res["grade"]="strong"
    else: res["grade"]="none"
    return res

def council_runtime_assess(ind, wick, rr, trend_mode, side):
    adx = float(ind.get("adx") or 0.0)
    action = None; reason=None; frac=0.0
    aligned = (trend_mode=="UP" and side=="long") or (trend_mode=="DOWN" and side=="short")

    # إغلاق صارم على الحالة القصوى + ربح كبير + تبريد ADX
    if wick["grade"]=="extreme" and rr>=CLOSE_ON_EXTREME_PROFIT_PCT and adx<=ADX_COOL_FOR_EXTREME_CLOSE:
        return "strict_close", f"Extreme wick/impulse + rr>={CLOSE_ON_EXTREME_PROFIT_PCT:.2f}% + ADX cool", 0.0

    # جني جزئي “أدبّي”
    if wick["grade"] in ("extreme","strong") and rr>=max(0.6, TP1_PCT_BASE):
        opposite = (side=="long" and wick["dir"]=="down" and wick["wick_up"]) or \
                   (side=="short" and wick["dir"]=="up"  and wick["wick_down"])
        frac = HARVEST_STEP_FRAC if not opposite else min(0.5, HARVEST_STEP_FRAC*1.5)
        return "harvest", f"Wick/Impulse {wick['grade']} rr={rr:.2f}% {'opp' if opposite else 'with'}", frac

    if rr>=TRAIL_ACTIVATE_PCT:
        return "tighten_trail", f"Trail tighten rr={rr:.2f}%", 0.0

    return None, None, 0.0

# ================== COUNCIL =========================
class Council:
    def __init__(self):
        self.state={"open":False,"side":None,"entry":None}
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
        if adx>=20 and pdi>mdi: b+=1; rb.append("+DI>−DI & ADX")
        if adx>=20 and mdi>pdi: s+=1; rs.append("−DI>+DI & ADX")
        o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
        if 45<=rsi<=55:
            if c>o: b+=1; rb.append("RSI neutral ↗")
            else:   s+=1; rs.append("RSI neutral ↘")

        # RF closed confirmation
        if rf.get("long"):  b+=1; rb.append("RF↑")
        if rf.get("short"): s+=1; rs.append("RF↓")

        self.log=(f"SCM | trend={tmode} | zone={z_best and z_best['side']} "
                  f"| liq(EQH={fmt(eqh)} EQL={fmt(eql)}) | flow={flow} "
                  f"|| BUY={b}[{', '.join(rb) or '—'}] | SELL={s}[{', '.join(rs) or '—'}]")
        print(colored(self.log,"green" if b>s else "red" if s>b else "cyan"))
        return b,rb,s,rs,tmode,z_best

    def entry_exit(self, df, ind, rf):
        b,rb,s,rs,tmode,zone = self.vote(df, ind, rf)
        entry=None; exit_=None

        if not STATE["open"]:
            if b>=self.min_entry and tmode!="DOWN":
                entry={"side":"buy","reason":rb,"tmode":tmode,"zone":zone}
                self.state.update({"open":True,"side":"long","entry":float(df['close'].iloc[-1])})
            elif s>=self.min_entry and tmode!="UP":
                entry={"side":"sell","reason":rs,"tmode":tmode,"zone":zone}
                self.state.update({"open":True,"side":"short","entry":float(df['close'].iloc[-1])})

        # Exit: إشارات عكسية مؤكدة + تبريد ADX
        if self.state["open"]:
            adx=float(ind.get("adx") or 0.0); side=self.state["side"]
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
                self.state={"open":False,"side":None,"entry":None}

        return entry, exit_

council = Council()

# ================== STATE / ORDERS ==================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0, "_opp_rf_bars": 0,
    "trend_mode":"NEUTRAL", "_flip_rf_bars":0
}
compound_pnl = 0.0
LAST_CLOSE_T = 0

def _params_open(side):
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if side=="buy" else "SHORT","reduceOnly":False}
    return {"positionSide":"BOTH","reduceOnly":False}
def _params_close():
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if STATE.get("side")=="long" else "SHORT","reduceOnly":True}
    return {"positionSide":"BOTH","reduceOnly":True}

def _read_position():
    try:
        poss=_with_ex(lambda: ex.fetch_positions(params={"type":"swap"}))
        base=SYMBOL.split(":")[0]
        for p in poss:
            sym=(p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if base not in sym: continue
            qty=abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty<=0: return 0.0,None,None
            entry=float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            ps=(p.get("info",{}).get("positionSide") or p.get("side") or "").upper()
            side="long" if "LONG" in ps else "short" if "SHORT" in ps else None
            return qty, side, entry
    except Exception as e: logging.error(f"_read_position: {e}")
    return 0.0,None,None

def compute_size(balance, price):
    cap=(balance or 0.0)*RISK_ALLOC*LEVERAGE
    raw=max(0.0, cap/max(float(price or 0.0),1e-9))
    return safe_qty(raw)

def open_market(side, qty, price, tag=""):
    if qty<=0: logging.warning("skip open qty<=0"); return False
    if MODE_LIVE:
        try:
            try: _with_ex(lambda: ex.set_leverage(LEVERAGE,SYMBOL,params={"side":"BOTH"}))
            except Exception: pass
            _with_ex(lambda: ex.create_order(SYMBOL,"market",side,qty,None,_params_open(side)))
        except Exception as e:
            logging.error(f"open_market: {e}"); return False
    STATE.update({"open":True,"side":"long" if side=="buy" else "short","entry":price,
                  "qty":qty,"pnl":0.0,"bars":0,"trail":None,"breakeven":None,
                  "tp1_done":False,"highest_profit_pct":0.0,"profit_targets_achieved":0,
                  "_opp_rf_bars":0})
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)} {tag}","green" if side=='buy' else "red"))
    return True

def _reset_after_close(reason):
    STATE.update({"open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,
                  "trail":None,"breakeven":None,"tp1_done":False,
                  "highest_profit_pct":0.0,"profit_targets_achieved":0,
                  "_opp_rf_bars":0, "_flip_rf_bars":0})
    logging.info(f"AFTER_CLOSE: {reason}")

def close_market_strict(reason="STRICT"):
    global compound_pnl, LAST_CLOSE_T
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty<=0 and not STATE.get("open"): return
    if exch_qty<=0 and STATE.get("open"):
        px=price_now() or STATE["entry"]; side=STATE["side"]; qty=STATE["qty"]; entry=STATE["entry"]
        pnl=(px-entry)*qty*(1 if side=="long" else -1); compound_pnl+=pnl
        print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
        _reset_after_close(reason); LAST_CLOSE_T=int(time.time()*1000); return
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
                _reset_after_close(reason); LAST_CLOSE_T=int(time.time()*1000); return
            qty_to_close=safe_qty(left); attempts+=1
            print(colored(f"⚠️ strict close retry {attempts} residual={fmt(left,4)}","yellow"))
        except Exception as e:
            last=e; attempts+=1; time.sleep(2.0)
    print(colored(f"❌ STRICT CLOSE FAILED — last_error={last}","red"))

def close_partial(frac, reason):
    if not STATE["open"] or STATE["qty"]<=0: return
    qty_close=safe_qty(max(0.0, STATE["qty"]*min(max(frac,0.0),1.0)))
    px=price_now() or STATE["entry"]
    min_unit=max(RESIDUAL_MIN_QTY, LOT_MIN or RESIDUAL_MIN_QTY)
    if qty_close<min_unit: return
    side="sell" if STATE["side"]=="long" else "buy"
    if MODE_LIVE:
        try: _with_ex(lambda: ex.create_order(SYMBOL,"market",side,qty_close,None,_params_close()))
        except Exception as e: logging.error(f"partial: {e}"); return
    pnl=(px-STATE["entry"])*qty_close*(1 if STATE["side"]=="long" else -1)
    STATE["qty"]=safe_qty(STATE["qty"]-qty_close)
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(STATE['qty'],4)}","magenta"))
    if 0<STATE["qty"]<=FINAL_CHUNK_QTY: close_market_strict("FINAL_CHUNK_RULE")

# ================== MANAGEMENT ======================
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

def _tp_ladder(info, ind, side, trend_align):
    px=info["price"]; atr=ind["atr"]; atr_pct=(atr/max(px,1e-9))*100.0 if px else 0.6
    base=_consensus(ind, info, side) + (0.5 if trend_align else 0.0)
    mults = [1.9,3.3,5.2] if base>=2.5 else [1.6,2.8,4.6] if base>=1.5 else [1.2,2.4,4.0]
    return [round(m*atr_pct,2) for m in mults],[0.22,0.28,0.50]

def strict_hp_close(ind, rr):
    if STATE["highest_profit_pct"]>=TRAIL_ACTIVATE_PCT:
        if rr < STATE["highest_profit_pct"]*STRICT_CLOSE_DROP_FROM_HP and float(ind.get("adx") or 0.0)<=STRICT_COOL_ADX:
            close_market_strict(f"STRICT_HP_CLOSE {STATE['highest_profit_pct']:.2f}%→{rr:.2f}%")

def manage_after_entry(df, ind, info):
    if not STATE["open"] or STATE["qty"]<=0: return
    px=info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr=(px-entry)/entry*100*(1 if side=="long" else -1)

    STATE["trend_mode"]=trend_mode(df, ind)
    align=(STATE["trend_mode"]=="UP" and side=="long") or (STATE["trend_mode"]=="DOWN" and side=="short")

    # TP1 + BE
    tp1_now = (TP1_PCT_BASE*1.4) if align else (TP1_PCT_BASE*(2.2 if ind["adx"]>=35 else 1.8 if ind["adx"]>=28 else 1.0))
    if (not STATE["tp1_done"]) and rr>=tp1_now:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{tp1_now:.2f}%"); STATE["tp1_done"]=True
        if rr>=BREAKEVEN_AFTER: STATE["breakeven"]=entry

    # Ladder
    tps,frs=_tp_ladder(info, ind, side, align)
    k=int(STATE.get("profit_targets_achieved",0))
    if k<len(tps) and rr>=tps[k]:
        close_partial(frs[k], f"TP_dyn@{tps[k]:.2f}%"); STATE["profit_targets_achieved"]=k+1

    # Runtime Council (حصاد/إغلاق/تريل)
    wick = analyze_wick_impulse(df, float(ind.get("atr") or 0.0))
    act, why, frac = council_runtime_assess(ind, wick, rr, STATE["trend_mode"], side)
    if RUNTIME_LOG and act: print(colored(f"🏛 Runtime Council → {act.upper()} :: {why}","white"))
    if act=="harvest": 
        if STATE["profit_targets_achieved"] < HARVEST_MAX_ROUNDS + 3:
            close_partial(frac, f"Harvest[{wick['grade']}]")
    elif act=="strict_close":
        close_market_strict(f"HARD_EXIT[{why}]"); return

    # Ratchet
    if rr>STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=rr
    if STATE["highest_profit_pct"]>=TRAIL_ACTIVATE_PCT and rr<STATE["highest_profit_pct"]*RATCHET_LOCK_FALLBACK:
        close_partial(0.50, f"Ratchet {STATE['highest_profit_pct']:.2f}%→{rr:.2f}%")

    # ATR Trail (أوسع مع الترند)
    atr=ind["atr"]; mult=ATR_TRAIL_MULT*(1.25 if align else 1.0)
    if rr>=TRAIL_ACTIVATE_PCT and atr>0:
        gap=atr*mult
        if side=="long":
            new=px-gap; STATE["trail"]=max(STATE["trail"] or new, new)
            if STATE["breakeven"] is not None: STATE["trail"]=max(STATE["trail"], STATE["breakeven"])
            if px<STATE["trail"]: close_market_strict(f"TRAIL_ATR({mult:.2f}x)")
        else:
            new=px+gap; STATE["trail"]=min(STATE["trail"] or new, new)
            if STATE["breakeven"] is not None: STATE["trail"]=min(STATE["trail"], STATE["breakeven"])
            if px>STATE["trail"]: close_market_strict(f"TRAIL_ATR({mult:.2f}x)")

    strict_hp_close(ind, rr)

# ================== Opposite RF Defense =============
def defensive_on_opposite_rf(ind, info):
    if not STATE["open"] or STATE["qty"]<=0: return
    opp = (STATE["side"]=="long" and info.get("short")) or (STATE["side"]=="short" and info.get("long"))
    if (STATE["trend_mode"] in ("UP","DOWN")) and opp:
        STATE["_opp_rf_bars"] += 1
        if STATE["_opp_rf_bars"] < OPP_RF_DEBOUNCE: return
    else:
        STATE["_opp_rf_bars"]=0
        if not opp: return
    px = info.get("price") or price_now() or STATE["entry"]
    base_frac = 0.20 if STATE.get("tp1_done") else 0.25
    close_partial(base_frac, "Opposite RF — defensive")
    if STATE.get("breakeven") is None: STATE["breakeven"]=STATE["entry"]
    atr=float(ind.get("atr") or 0.0)
    if atr>0 and px:
        gap=atr*max(ATR_TRAIL_MULT,1.2)
        if STATE["side"]=="long": STATE["trail"]=max(STATE["trail"] or (px-gap), px-gap)
        else: STATE["trail"]=min(STATE["trail"] or (px+gap), px+gap)

# ================== Auto-Flip With Trend =============
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

# ================== UI ==============================
def snapshot(bal, info, ind, spread, rf, council_log=None, reason=None, df=None):
    left=time_to_candle_close(df) if df is not None else 0
    print("─"*120)
    print(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"💲 Price {fmt(info.get('price'))} | RF filt={fmt(rf.get('filter'))} hi={fmt(rf.get('hi'))} lo={fmt(rf.get('lo'))} | spread={fmt(spread,2)}bps | closes_in~{left}s")
    print(f"🧮 RSI={fmt(ind['rsi'])} +DI={fmt(ind['plus_di'])} -DI={fmt(ind['minus_di'])} ADX={fmt(ind['adx'])} ATR={fmt(ind['atr'])}")
    try:
        wk = analyze_wick_impulse(df, float(ind.get('atr') or 0.0))
        if wk['grade'] != 'none':
            print(f"   🔎 wick/impulse: grade={wk['grade']} up={wk['upper_ratio']:.2f} dn={wk['lower_ratio']:.2f} bodyATR={wk['body_atr']:.2f} dir={wk['dir']}")
    except Exception: pass
    if council_log: print(f"🏛 {council_log}")
    if reason: print(f"ℹ️ {reason}")
    print(f"🧭 Balance={fmt(bal,2)} Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x PnL={fmt(STATE['pnl'])} Trend={STATE['trend_mode']} Eq~{fmt((bal or 0)+compound_pnl,2)}")
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp} Entry={fmt(STATE['entry'])} Qty={fmt(STATE['qty'],4)} Bars={STATE['bars']} Trail={fmt(STATE['trail'])} BE={fmt(STATE['breakeven'])} HP={fmt(STATE['highest_profit_pct'],2)}% TPs={STATE['profit_targets_achieved']}")
    else:
        print("   ⚪ FLAT")
    print("─"*120)

# ================== MAIN LOOP =======================
app=Flask(__name__)

def trade_loop():
    global LAST_CLOSE_T
    while True:
        try:
            bal=balance_usdt()
            df =fetch_ohlcv()
            ind=compute_indicators(df)
            rf =rf_signal_closed(df)
            px =price_now() or rf["price"]
            spread=orderbook_spread_bps()

            # Council (entry/exit)
            entry, exit_ = council.entry_exit(df, ind, rf)
            STATE["trend_mode"]=trend_mode(df, ind)

            # Auto-Flip With Trend (قبل الإدارة)
            flip = should_auto_flip(df, ind, rf)
            if flip and STATE["open"]:
                print(colored(f"🔄 AUTO FLIP → {flip['flip_to'].upper()} :: {flip['reason']}", "magenta"))
                close_market_strict("AUTO_FLIP")
                px_flip = price_now() or rf.get("price")
                bal_now = balance_usdt()
                if px_flip and bal_now:
                    qty = compute_size(bal_now, px_flip)
                    open_market(flip["flip_to"], qty, px_flip, tag="[AutoFlip]")
                STATE["_flip_rf_bars"]=0

            # Update PnL
            if STATE["open"] and px:
                STATE["pnl"]=(px-STATE["entry"])*STATE["qty"]*(1 if STATE["side"]=="long" else -1)

            # Manage open trade
            if STATE["open"] and px:
                manage_after_entry(df, ind, {"price":px,"filter":rf["filter"]})
                # Defensive opposite RF inside trend
                opp = (STATE["side"]=="long" and rf["short"]) or (STATE["side"]=="short" and rf["long"])
                if opp: defensive_on_opposite_rf(ind, {"price": px, **rf})

            # Guards
            reason=None
            if spread is not None and spread>MAX_SPREAD_BPS: reason=f"spread too high {fmt(spread,2)}bps"
            if (reason is None) and ind["adx"]<PAUSE_ADX_THRESHOLD: reason=f"ADX<{int(PAUSE_ADX_THRESHOLD)} pause"

            # Council EXIT
            if STATE["open"] and exit_:
                print(colored(f"🏁 COUNCIL EXIT: {exit_['reason']}","yellow"))
                close_market_strict(f"COUNCIL_EXIT: {exit_['reason']}")
                LAST_CLOSE_T=rf.get("time") or int(time.time()*1000)

            # ENTRY: Council ⇒ RF fallback
            sig=None; tag=""
            if (not STATE["open"]) and reason is None:
                if entry:
                    sig=entry["side"]; tag=f"[Council] {entry['reason']}"
                elif ENTRY_FROM_RF and (rf["long"] or rf["short"]):
                    sig="buy" if rf["long"] else "sell"; tag=f"[RF-closed {'LONG' if rf['long'] else 'SHORT'}]"

            # Wait next closed after close
            if not STATE["open"] and sig and reason is None and WAIT_NEXT_CLOSED:
                if int(rf.get("time") or 0) <= int(LAST_CLOSE_T or 0):
                    reason="wait_for_next_closed_signal"

            if not STATE["open"] and sig and reason is None:
                qty=compute_size(bal, px or rf["price"])
                if qty>0 and (px or rf["price"]):
                    if open_market(sig, qty, px or rf["price"], tag): LAST_CLOSE_T=0
                else:
                    reason="qty<=0 or price=None"

            snapshot(bal, {"price": px or rf["price"]}, ind, spread, rf, council.log, reason, df)

            # bar count
            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"]+=1

            time.sleep(NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP)
        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}","red"))
            logging.error(f"loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# ================== API / KEEPALIVE ================
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ DOGE SMC Council + Trend + AutoFlip + RF(Closed) — {SYMBOL} {INTERVAL} — {mode}"

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
    s=requests.Session(); s.headers.update({"User-Agent":"smc-autoflip-keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}","cyan"))
    while True:
        try: s.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# ================== BOOT ===========================
if __name__=="__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}","yellow"))
    print(colored(f"Council⇒RF fallback | Trend & AutoFlip | ADX≥{PAUSE_ADX_THRESHOLD}","yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
