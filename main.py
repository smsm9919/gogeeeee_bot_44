# -*- coding: utf-8 -*-
"""
DOGE/USDT — BingX Perp via CCXT
SCM/SMC Council + RF(CLOSED) Smart Trend Bot

Core ideas:
- Council = Supply/Demand boxes + Liquidity (EQH/EQL & Sweeps) + Displacement + Retest + Trap detection
- Trend engine (UP/DOWN/NEUTRAL) يقود التحيّز وإدارة الربح
- RF-on-closed fallback فقط لو المجلس مش شايف فرصة قوية
- أثناء أي صفقة (سواء دخلت RF أو Council) المجلس يقيّم الصفقة ويغلق صارم عند انتهاء الموجة

Only API keys from ENV. All strategy params are hardcoded below.
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

SELF_URL = (os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")).strip()
PORT     = 5000

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
ENTRY_FROM_RF = True   # يعمل فقط لو المجلس غير حاسم

# -------- Indicators lengths ------------------------
RSI_LEN = 14; ADX_LEN = 14; ATR_LEN = 14

# -------- Guards ------------------------------------
MAX_SPREAD_BPS      = 8.0
PAUSE_ADX_THRESHOLD = 15.0   # تبدأ التداول من 15 زي ما طلبت
WAIT_NEXT_CLOSED    = True   # لا دخول على نفس الشمعة

# -------- Trend engine ------------------------------
TREND_ADX_MIN    = 30.0
STRUCT_BARS      = 48         # فحص الهيكل (HH/HL أو LH/LL) على آخر 48 شمعة
OPP_RF_DEBOUNCE  = 2          # عدد الشموع المطلوبة لتأكيد RF عكسي داخل الترند

# -------- SCM / Liquidity ---------------------------
EQ_BPS             = 10.0     # مساواة قمم/قيعان (bps)
SWEEP_LOOKBACK     = 60
DSP_ATR_MIN        = 1.2      # Displacement ≥ 1.2×ATR
RETEST_BPS         = 15.0     # لمس منطقة الصندوق
TRAP_CLOSE_IN_BARS = 3        # فخ: كسر ثم إغلاق رجوعي خلال N شموع

# -------- Management --------------------------------
TP1_PCT_BASE       = 0.40
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.6
RATCHET_LOCK_FALLBACK = 0.60
FINAL_CHUNK_QTY    = 50.0
RESIDUAL_MIN_QTY   = 9.0

# Strict close when losing peak gains (and ADX cools)
STRICT_CLOSE_DROP_FROM_HP = 0.50
STRICT_COOL_ADX           = 20.0

# Wick harvest (disabled in strong trend)
WICK_HARVEST_MIN_PCT   = 0.60
WICK_LONG_FRAC         = 0.30
WICK_RATIO_THRESHOLD   = 0.60

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# ================== LOGGING =========================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h,"baseFilename","").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready","cyan"))
setup_file_logging()

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
    with EX_LOCK:
        return fn()

MARKET = {}; AMT_PREC=0; LOT_STEP=None; LOT_MIN=None

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
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    print(colored(f"⚠️ exchange init: {e}","yellow"))

# ================== HELPERS =========================
def with_retry(fn, tries=3, base_wait=0.4):
    for i in range(tries):
        try: return fn()
        except Exception:
            if i==tries-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.25)

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
    except Exception:
        return max(0.0, float(q))

def safe_qty(q):
    q=_round_amt(q)
    if q<=0: print(colored(f"⚠️ qty invalid after normalize → {q}","yellow"))
    return q

def fmt(v,d=6,na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception: return na

def fetch_ohlcv(limit=600):
    rows=with_retry(lambda: _with_ex(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"})))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t=with_retry(lambda: _with_ex(lambda: ex.fetch_ticker(SYMBOL)))
        return t.get("last") or t.get("close")
    except Exception:
        return None

def balance_usdt():
    if not MODE_LIVE: return 100.0
    try:
        b=with_retry(lambda: _with_ex(lambda: ex.fetch_balance(params={"type":"swap"})))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception:
        return None

def orderbook_spread_bps():
    try:
        ob=with_retry(lambda: _with_ex(lambda: ex.fetch_order_book(SYMBOL, limit=5)))
        bid=ob["bids"][0][0] if ob["bids"] else None
        ask=ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid=(bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

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

# ================== INDICATORS ======================
def wilder_ema(s: pd.Series, n:int):
    return s.ewm(alpha=1/n, adjust=False).mean()

def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN)+2:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"dx":0.0,"adx":0.0,"atr":0.0}
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr= wilder_ema(tr, ATR_LEN)

    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = wilder_ema(up,RSI_LEN)/wilder_ema(dn,RSI_LEN).replace(0,1e-12)
    rsi = 100-(100/(1+rs))

    upm=h.diff(); dnm=l.shift(1)-l
    plus_dm = upm.where((upm>dnm)&(upm>0),0.0)
    minus_dm= dnm.where((dnm>upm)&(dnm>0),0.0)
    plus_di = 100*(wilder_ema(plus_dm,ADX_LEN)/atr.replace(0,1e-12))
    minus_di= 100*(wilder_ema(minus_dm,ADX_LEN)/atr.replace(0,1e-12))
    dx = (100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx = wilder_ema(dx, ADX_LEN)

    i=len(df)-1
    return {
        "rsi": float(rsi.iloc[i]),
        "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]),
        "adx": float(adx.iloc[i]),
        "atr": float(atr.iloc[i])
    }

# ================== RF (CLOSED) =====================
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
    if len(df) < RF_PERIOD+3:
        return {"time": int(time.time()*1000), "price": None, "long": False, "short": False,
                "filter": None, "hi": None, "lo": None}
    d=df.iloc[:-1]
    src=d[RF_SOURCE].astype(float)
    hi,lo,filt=_rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    p_now=float(src.iloc[-1]); p_prev=float(src.iloc[-2])
    f_now=float(filt.iloc[-1]); f_prev=float(filt.iloc[-2])
    def _bps(a,b):
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    long_flip  = (p_prev<=f_prev and p_now>f_now and _bps(p_now,f_now)>=RF_HYST_BPS)
    short_flip = (p_prev>=f_prev and p_now<f_now and _bps(p_now,f_now)>=RF_HYST_BPS)
    return {"time": int(d["time"].iloc[-1]), "price": p_now,
            "long": bool(long_flip), "short": bool(short_flip),
            "filter": f_now, "hi": float(hi.iloc[-1]), "lo": float(lo.iloc[-1])}

# ================== SCM/SMC DETECTORS ===============
def _near_bps(a,b,bps): 
    try: return abs((a-b)/b)*10000.0 <= bps
    except Exception: return False

def _pivots(df, lb=3):
    if len(df)<lb*2+1: return [],[]
    hi=df["high"].astype(float).values
    lo=df["low"].astype(float).values
    H=[]; L=[]
    for i in range(lb, len(df)-lb):
        if hi[i]==max(hi[i-lb:i+lb+1]): H.append(i)
        if lo[i]==min(lo[i-lb:i+lb+1]): L.append(i)
    return H,L

def _boxes(df):
    # صناديق تقريبية من آخر شمعة اندفاعية مخالفة للاتجاه
    d=df.iloc[:-1] if len(df)>=2 else df.copy()
    H,L=_pivots(d, lb=3)
    sup=dem=None
    if H:
        top=max(float(d["high"].iloc[i]) for i in H[-6:])
        bot=top - (top - min(float(d["high"].iloc[i]) for i in H[-6:]))*0.25
        sup={"side":"supply","top":top,"bot":bot}
    if L:
        bot=min(float(d["low"].iloc[i]) for i in L[-6:])
        top=bot + (max(float(d["low"].iloc[i]) for i in L[-6:]) - bot)*0.25
        dem={"side":"demand","top":top,"bot":bot}
    return sup,dem

def _eq_levels(df, bps=EQ_BPS):
    # Equal Highs/Lows (مناطق سيولة)
    d=df.iloc[-SWEEP_LOOKBACK:] if len(df)>=SWEEP_LOOKBACK else df
    highs=d["high"].astype(float).values
    lows =d["low"].astype(float).values
    eqh=None; eql=None
    # قمة/قاع متقاربين
    for i in range(5, len(d)-5):
        for j in range(i+2, min(i+15, len(d)-2)):
            if _near_bps(highs[i], highs[j], bps): eqh=max(eqh or 0.0, highs[i])
            if _near_bps(lows[i],  lows[j],  bps): eql=min(eql or 1e9, lows[i])
    return eqh, eql

def _sweep_now(df, eqh, eql):
    # سحب سيولة: كسر Eq ثم إغلاق رجوعي واضح (Displacement لاحقًا)
    if len(df)<3: return {"buy":False,"sell":False}
    o,h,l,c = map(float, df[["open","high","low","close"]].iloc[-2])  # آخر شمعة مغلقة
    buy = (eql is not None) and (l < eql) and (c > o)   # كسر قاع سيولة ثم رجوع
    sell= (eqh is not None) and (h > eqh) and (c < o)   # كسر قمة سيولة ثم رجوع
    return {"buy":buy,"sell":sell}

def _displacement(df, atr):
    # قوة الشمعة الأخيرة المغلقة مقابل ATR
    if len(df)<2 or atr<=0: return 0.0
    o,c = map(float, df[["open","close"]].iloc[-2])
    body=abs(c-o)
    return body/atr

def _retest_now(df, box):
    if not box or len(df)<1: return False
    low=float(df["low"].iloc[-1]); high=float(df["high"].iloc[-1])
    if box["side"]=="demand":
        return (low<=box["top"]) or _near_bps(low, box["top"], RETEST_BPS)
    else:
        return (high>=box["bot"]) or _near_bps(high, box["bot"], RETEST_BPS)

def _trap_recent(df):
    # فخ: كسر ثم إغلاق رجوعي داخل النطاق خلال N شموع
    if len(df)<TRAP_CLOSE_IN_BARS+3: return False
    d=df.iloc[-(TRAP_CLOSE_IN_BARS+3):-1]
    rng_high=d["high"].max(); rng_low=d["low"].min()
    last_close=float(d["close"].iloc[-1])
    broke_up = any(float(d["high"].iloc[i])>rng_high for i in range(len(d)-3,len(d)))
    broke_dn = any(float(d["low"].iloc[i]) <rng_low  for i in range(len(d)-3,len(d)))
    return (broke_up and last_close<rng_high) or (broke_dn and last_close>rng_low)

def _trend(df, ind):
    adx=float(ind.get("adx") or 0.0)
    if adx<TREND_ADX_MIN or len(df)<STRUCT_BARS+5: return "NEUTRAL"
    d=df.iloc[-(STRUCT_BARS+1):-1]
    H=d["high"].astype(float).values
    L=d["low"].astype(float).values
    k=STRUCT_BARS//3
    up  = H[-k:].mean()>H[k:2*k].mean()>H[:k].mean() and L[-k:].mean()>L[k:2*k].mean()>L[:k].mean()
    down= H[-k:].mean()<H[k:2*k].mean()<H[:k].mean() and L[-k:].mean()<L[k:2*k].mean()<L[:k].mean()
    return "UP" if up else "DOWN" if down else "NEUTRAL"

# ================== COUNCIL =========================
class Council:
    def __init__(self):
        self.state={"open":False,"side":None,"entry":None}
        self.min_entry=4
        self.min_exit =3
        self.last_log=""

    def vote(self, df, ind, rf_info):
        # Boxes
        sup, dem = _boxes(df)
        # Liquidity
        eqh, eql  = _eq_levels(df)
        sweep     = _sweep_now(df, eqh, eql)
        atr=float(ind.get("atr") or 0.0)
        dsp=_displacement(df, atr)
        # Retest/Trap
        ret_dem=_retest_now(df, dem); ret_sup=_retest_now(df, sup)
        trap=_trap_recent(df)
        # Trend
        trend=_trend(df, ind)

        b=s=0; rb=[]; rs=[]

        # Strong BUY set
        if sweep["buy"]:            b+=2; rb.append("sweep_liquidity_down→up")
        if dem:                     b+=1; rb.append("demand_box")
        if ret_dem:                 b+=1; rb.append("retest_demand")
        if dsp>=DSP_ATR_MIN and float(df["close"].iloc[-2])>float(df["open"].iloc[-2]):
            b+=1; rb.append(f"displacement {dsp:.2f}xATR")
        if rf_info.get("long"):     b+=1; rb.append("rf_closed_long")

        # Strong SELL set
        if sweep["sell"]:           s+=2; rs.append("sweep_liquidity_up→down")
        if sup:                     s+=1; rs.append("supply_box")
        if ret_sup:                 s+=1; rs.append("retest_supply")
        if dsp>=DSP_ATR_MIN and float(df["close"].iloc[-2])<float(df["open"].iloc[-2]):
            s+=1; rs.append(f"displacement {dsp:.2f}xATR(down)")
        if rf_info.get("short"):    s+=1; rs.append("rf_closed_short")

        # DI/ADX + RSI neutral tilt
        pdi,mdi,adx = ind.get("plus_di",0), ind.get("minus_di",0), ind.get("adx",0)
        if adx>=20 and pdi>mdi: b+=1; rb.append("+DI>−DI & ADX")
        if adx>=20 and mdi>pdi: s+=1; rs.append("−DI>+DI & ADX")
        rsi=ind.get("rsi",50.0); o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
        if 45<=rsi<=55:
            if c>o: b+=1; rb.append("RSI neutral ↗")
            else:   s+=1; rs.append("RSI neutral ↘")

        # Trap penalizes exits/entries opposite to trend
        if trap:
            if trend=="UP": s=max(0, s-1); rs.append("trap_filter(-1)")
            if trend=="DOWN": b=max(0, b-1); rb.append("trap_filter(-1)")

        # Compose log
        self.last_log = (
            f"SCM | trend={trend} | boxes[sup={('%.5f-%.5f'% (sup['bot'],sup['top'])) if sup else '—'}, "
            f"dem={('%.5f-%.5f'% (dem['bot'],dem['top'])) if dem else '—'}] | "
            f"liq[EQH={fmt(eqh)} EQL={fmt(eql)} sweep={'B' if sweep['buy'] else 'S' if sweep['sell'] else '—'} dsp={dsp:.2f}x] | "
            f"retest[dem={ret_dem} sup={ret_sup}] trap={trap} || "
            f"votes buy={b}({', '.join(rb) or '—'}) | sell={s}({', '.join(rs) or '—'})"
        )
        print(colored(self.last_log, "green" if b>s else "red" if s>b else "cyan"))

        return {"buy":b, "sell":s, "reasons_buy":rb, "reasons_sell":rs, "trend":trend,
                "boxes":{"supply":sup,"demand":dem}, "sweep":sweep, "dsp":dsp, "trap":trap}

    def entry_exit(self, df, ind, rf_info):
        v=self.vote(df, ind, rf_info)
        entry=None; exit_=None

        # ENTRY (المجلس أولاً – مع الترند لو قوي)
        if not STATE["open"]:
            if v["buy"]>=self.min_entry and (v["trend"]!="DOWN"):
                entry={"side":"buy","reason":f"council {v['buy']}✓ :: {v['reasons_buy']}"}
                self.state.update({"open":True,"side":"long","entry":float(df['close'].iloc[-1])})
            elif v["sell"]>=self.min_entry and (v["trend"]!="UP"):
                entry={"side":"sell","reason":f"council {v['sell']}✓ :: {v['reasons_sell']}"}
                self.state.update({"open":True,"side":"short","entry":float(df['close'].iloc[-1])})

        # EXIT (مع الترند نشدد الخروج)
        if self.state["open"]:
            side=self.state["side"]
            votes=0; reasons=[]
            adx=float(ind.get("adx",0.0))
            # نهاية موجة: فقدان كبير من HP سيعالجه الـmanager (strict close) — هنا نضيف إشارات بنيوية
            # خروج عند sweep عكسي + retest لصندوق معاكس + dsp عكسي
            if side=="long":
                if v["sweep"]["sell"]: votes+=1; reasons.append("sweep_up→down")
                if v["boxes"]["supply"] and _retest_now(df, v["boxes"]["supply"]): votes+=1; reasons.append("retest_supply")
                if v["dsp"]>=DSP_ATR_MIN and float(df["close"].iloc[-2])<float(df["open"].iloc[-2]): votes+=1; reasons.append("dsp_down")
                if adx<20: votes+=1; reasons.append("ADX cool-off")
            else:
                if v["sweep"]["buy"]: votes+=1; reasons.append("sweep_down→up")
                if v["boxes"]["demand"] and _retest_now(df, v["boxes"]["demand"]): votes+=1; reasons.append("retest_demand")
                if v["dsp"]>=DSP_ATR_MIN and float(df["close"].iloc[-2])>float(df["open"].iloc[-2]): votes+=1; reasons.append("dsp_up")
                if adx<20: votes+=1; reasons.append("ADX cool-off")

            min_exit=self.min_exit + (1 if ((v["trend"]=="UP" and side=="long") or (v["trend"]=="DOWN" and side=="short")) else 0)
            if votes>=min_exit:
                exit_={"action":"close","reason":" / ".join(reasons)}
                self.state={"open":False,"side":None,"entry":None}

        return entry, exit_, v

council = Council()

# ================== STATE / ORDERS ==================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0, "opp_votes": 0,
    "trend_mode": "NEUTRAL", "_opp_rf_bars": 0
}
compound_pnl = 0.0
LAST_CLOSE_TIME = 0

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
            if qty<=0: return 0.0, None, None
            entry=float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            ps=(p.get("info",{}).get("positionSide") or p.get("side") or "").upper()
            if "LONG" in ps: side="long"
            elif "SHORT" in ps: side="short"
            else:
                last=price_now() or entry
                side="long" if last>=entry else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position: {e}")
    return 0.0, None, None

def compute_size(balance, price):
    cap=(balance or 0.0)*RISK_ALLOC*LEVERAGE
    return safe_qty(max(0.0, cap/max(float(price or 0.0), 1e-9)))

def open_market(side, qty, price, tag=""):
    if qty<=0: print(colored("❌ skip open (qty<=0)","red")); return False
    if MODE_LIVE:
        try:
            try: _with_ex(lambda: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"}))
            except Exception: pass
            _with_ex(lambda: ex.create_order(SYMBOL,"market",side,qty,None,_params_open(side)))
        except Exception as e:
            print(colored(f"❌ open: {e}","red")); logging.error(f"open_market: {e}"); return False
    STATE.update({
        "open":True,"side":"long" if side=="buy" else "short","entry":price,"qty":qty,
        "pnl":0.0,"bars":0,"trail":None,"breakeven":None,
        "tp1_done":False,"highest_profit_pct":0.0,"profit_targets_achieved":0,
        "opp_votes":0,"_opp_rf_bars":0
    })
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)} {tag}","green" if side=='buy' else "red"))
    return True

def _reset_after_close(reason):
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0,
        "profit_targets_achieved": 0, "opp_votes": 0, "_opp_rf_bars": 0
    })
    logging.info(f"AFTER_CLOSE: {reason}")

def close_market_strict(reason="STRICT"):
    global compound_pnl, LAST_CLOSE_TIME
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty<=0 and not STATE.get("open"): return
    if exch_qty<=0 and STATE.get("open"):
        px = price_now() or STATE["entry"]; side=STATE["side"]; qty=STATE["qty"]; entry=STATE["entry"]
        pnl=(px-entry)*qty*(1 if side=="long" else -1)
        compound_pnl+=pnl
        print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
        _reset_after_close(reason); LAST_CLOSE_TIME=int(time.time()*1000); return
    side_to_close="sell" if exch_side=="long" else "buy"
    qty_to_close=safe_qty(exch_qty)
    attempts=0; last=None
    while attempts<6:
        try:
            if MODE_LIVE:
                params=_params_close(); params["reduceOnly"]=True
                _with_ex(lambda: ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params))
            time.sleep(2.0)
            left,_,_= _read_position()
            if left<=0:
                px=price_now() or STATE.get("entry") or exch_entry
                entry_px=STATE.get("entry") or exch_entry or px
                side=STATE.get("side") or exch_side
                pnl=(px-entry_px)*exch_qty*(1 if side=="long" else -1)
                compound_pnl+=pnl
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                _reset_after_close(reason); LAST_CLOSE_TIME=int(time.time()*1000); return
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
    if qty_close<min_unit:
        print(colored(f"⏸️ skip partial (amount={fmt(qty_close,4)} < min_unit={fmt(min_unit,4)})","yellow")); return
    side="sell" if STATE["side"]=="long" else "buy"
    if MODE_LIVE:
        try: _with_ex(lambda: ex.create_order(SYMBOL,"market",side,qty_close,None,_params_close()))
        except Exception as e: print(colored(f"❌ partial: {e}","red")); return
    pnl=(px-STATE["entry"])*qty_close*(1 if STATE["side"]=="long" else -1)
    STATE["qty"]=safe_qty(STATE["qty"]-qty_close)
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(STATE['qty'],4)}","magenta"))
    if STATE["qty"]<=FINAL_CHUNK_QTY and STATE["qty"]>0:
        close_market_strict("FINAL_CHUNK_RULE")

# ================== MANAGEMENT ======================
def _consensus(ind, info, side):
    score=0.0
    adx=float(ind.get("adx") or 0.0); rsi=float(ind.get("rsi") or 50.0)
    if (side=="long" and rsi>=55) or (side=="short" and rsi<=45): score+=1.0
    if adx>=28: score+=1.0
    elif adx>=20: score+=0.5
    try:
        if info.get("filter") and info.get("price"):
            if abs(info["price"]-info["filter"])/max(info["filter"],1e-9) >= (RF_HYST_BPS/10000.0):
                score += 0.5
    except Exception: pass
    return score

def _tp_ladder(info, ind, side, trend_align):
    px=info["price"]; atr=float(ind.get("atr") or 0.0)
    atr_pct=(atr/max(px,1e-9))*100.0 if px else 0.5
    score=_consensus(ind, info, side) + (0.5 if trend_align else 0.0)
    mults = [1.8,3.2,5.0] if score>=2.5 else [1.6,2.8,4.5] if score>=1.5 else [1.2,2.4,4.0]
    return [round(m*atr_pct,2) for m in mults],[0.25,0.30,0.45]

def wick_harvest(df, rr):
    if rr < WICK_HARVEST_MIN_PCT or not STATE["open"]: return
    # لا نجني الفتائل في ترند قوي موافق للصفقة
    if (STATE["trend_mode"]=="UP" and STATE["side"]=="long") or (STATE["trend_mode"]=="DOWN" and STATE["side"]=="short"):
        return
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1]); l=float(df["low"].iloc[-1]); c=float(df["close"].iloc[-1])
    rng=max(h-l,1e-12); up=h-max(o,c); dn=min(o,c)-l
    if STATE["side"]=="long" and (up/rng)>=WICK_RATIO_THRESHOLD:
        close_partial(WICK_LONG_FRAC, f"WickHarvest(up {up/rng:.2f})")
    if STATE["side"]=="short" and (dn/rng)>=WICK_RATIO_THRESHOLD:
        close_partial(WICK_LONG_FRAC, f"WickHarvest(down {dn/rng:.2f})")

def strict_hp_close(ind, rr):
    if STATE["highest_profit_pct"]>=TRAIL_ACTIVATE_PCT:
        if rr < STATE["highest_profit_pct"]*STRICT_CLOSE_DROP_FROM_HP and float(ind.get("adx",0.0))<=STRICT_COOL_ADX:
            close_market_strict(f"STRICT_HP_CLOSE {STATE['highest_profit_pct']:.2f}%→{rr:.2f}%")

def manage_after_entry(df, ind, info):
    if not STATE["open"] or STATE["qty"]<=0: return
    px=info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr=(px-entry)/entry*100*(1 if side=="long" else -1)

    STATE["trend_mode"]=_trend(df, ind)
    trend_align = (STATE["trend_mode"]=="UP" and side=="long") or (STATE["trend_mode"]=="DOWN" and side=="short")

    # TP1 + BE
    tp1_now = (TP1_PCT_BASE*1.4) if trend_align else (TP1_PCT_BASE*(2.2 if ind.get("adx",0)>=35 else 1.8 if ind.get("adx",0)>=28 else 1.0))
    if (not STATE["tp1_done"]) and rr>=tp1_now:
        close_partial(TP1_CLOSE_FRAC, f"TP1@{tp1_now:.2f}%"); STATE["tp1_done"]=True
        if rr>=BREAKEVEN_AFTER: STATE["breakeven"]=entry

    # Ladder
    dyn_tps,dyn_fracs=_tp_ladder(info, ind, side, trend_align)
    k=int(STATE.get("profit_targets_achieved",0))
    if k<len(dyn_tps) and rr>=dyn_tps[k]:
        close_partial(dyn_fracs[k], f"TP_dyn@{dyn_tps[k]:.2f}%"); STATE["profit_targets_achieved"]=k+1

    # Highest profit tracking + ratchet
    if rr>STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=rr
    if STATE["highest_profit_pct"]>=TRAIL_ACTIVATE_PCT and rr<STATE["highest_profit_pct"]*RATCHET_LOCK_FALLBACK:
        close_partial(0.50, f"Ratchet {STATE['highest_profit_pct']:.2f}%→{rr:.2f}%")

    # Wick harvest (disabled in strong aligned trend)
    wick_harvest(df, rr)

    # ATR trail
    atr=float(ind.get("atr") or 0.0)
    trail_mult = ATR_TRAIL_MULT*(1.25 if trend_align else 1.0)
    if rr>=TRAIL_ACTIVATE_PCT and atr>0:
        gap=atr*trail_mult
        if side=="long":
            new=px-gap; STATE["trail"]=max(STATE["trail"] or new, new)
            if STATE["breakeven"] is not None: STATE["trail"]=max(STATE["trail"], STATE["breakeven"])
            if px<STATE["trail"]: close_market_strict(f"TRAIL_ATR({trail_mult:.2f}x)")
        else:
            new=px+gap; STATE["trail"]=min(STATE["trail"] or new, new)
            if STATE["breakeven"] is not None: STATE["trail"]=min(STATE["trail"], STATE["breakeven"])
            if px>STATE["trail"]: close_market_strict(f"TRAIL_ATR({trail_mult:.2f}x)")

    # Strict close at wave end
    strict_hp_close(ind, rr)

# ================== OPPOSITE RF DEFENSE =============
def defensive_on_opposite_rf(ind, info):
    if not STATE["open"] or STATE["qty"]<=0: return
    opp = (STATE["side"]=="long" and info.get("short")) or (STATE["side"]=="short" and info.get("long"))
    if (STATE["trend_mode"] in ("UP","DOWN")) and opp:
        STATE["_opp_rf_bars"] += 1
        if STATE["_opp_rf_bars"] < OPP_RF_DEBOUNCE:
            return  # تجاهل أول N شموع عكسية داخل ترند
    else:
        STATE["_opp_rf_bars"]=0
        if not opp: return

    px = info.get("price") or price_now() or STATE["entry"]
    base_frac = 0.20 if STATE.get("tp1_done") else 0.25
    close_partial(base_frac, "Opposite RF — defensive")
    if STATE.get("breakeven") is None: STATE["breakeven"]=STATE["entry"]
    atr=float(ind.get("atr") or 0.0)
    if atr>0 and px is not None:
        gap=atr*max(ATR_TRAIL_MULT,1.2)
        if STATE["side"]=="long": STATE["trail"]=max(STATE["trail"] or (px-gap), px-gap)
        else: STATE["trail"]=min(STATE["trail"] or (px+gap), px+gap)

# ================== UI ==============================
def pretty_snapshot(bal, info, ind, spread_bps, council_log=None, reason=None, df=None):
    left_s=time_to_candle_close(df) if df is not None else 0
    print(colored("─"*120,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*120,"cyan"))
    print("📈 RF CLOSED")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))} hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))} | spread={fmt(spread_bps,2)} bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))} +DI={fmt(ind.get('plus_di'))} -DI={fmt(ind.get('minus_di'))} ADX={fmt(ind.get('adx'))} ATR={fmt(ind.get('atr'))}")
    if council_log: print(colored(f"   {council_log}","white"))
    if reason:      print(colored(f"   ℹ️ reason: {reason}","yellow"))
    print(f"   ⏱️ closes_in ≈ {left_s}s")
    print("\n🧭 POSITION")
    bal_line=f"Balance={fmt(bal,2)} Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x CompoundPnL={fmt(compound_pnl)} Eq~{fmt((bal or 0)+compound_pnl,2)}"
    print(colored(f"   {bal_line}","yellow"))
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp} Entry={fmt(STATE['entry'])} Qty={fmt(STATE['qty'],4)} Bars={STATE['bars']} Trail={fmt(STATE['trail'])} BE={fmt(STATE['breakeven'])} Trend={STATE['trend_mode']}")
        print(f"   🎯 TP_done={STATE['profit_targets_achieved']} HP={fmt(STATE['highest_profit_pct'],2)}% OppVotes={STATE.get('_opp_rf_bars',0)}")
    else:
        print("   ⚪ FLAT")
    print(colored("─"*120,"cyan"))

# ================== MAIN LOOP =======================
app=Flask(__name__)

def trade_loop():
    global LAST_CLOSE_TIME
    while True:
        try:
            bal=balance_usdt()
            px =price_now()
            df =fetch_ohlcv()
            ind  = compute_indicators(df)
            rf_c = rf_signal_closed(df)
            spread_bps=orderbook_spread_bps()

            # Council
            entry, exit_, v = council.entry_exit(df, ind, rf_c)
            STATE["trend_mode"]=v["trend"]

            # Update PnL
            if STATE["open"] and px:
                STATE["pnl"]=(px-STATE["entry"])*STATE["qty"]*(1 if STATE["side"]=="long" else -1)

            # Manage open trade (Council يحلل الصفقة طول الوقت)
            manage_after_entry(df, ind, {"price": px or rf_c["price"], **rf_c})

            # Defensive opposite RF inside trend
            if STATE["open"]:
                opp = (STATE["side"]=="long" and rf_c["short"]) or (STATE["side"]=="short" and rf_c["long"])
                if opp: defensive_on_opposite_rf(ind, {"price": px or rf_c["price"], **rf_c})

            # Guards
            reason=None
            if spread_bps is not None and spread_bps>MAX_SPREAD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            if (reason is None) and (float(ind.get("adx") or 0.0) < PAUSE_ADX_THRESHOLD):
                reason=f"ADX<{PAUSE_ADX_THRESHOLD:.0f} — trading paused"

            # Council EXIT (أولوية)
            if STATE["open"] and exit_:
                print(colored(f"🏛️ COUNCIL EXIT: {exit_['reason']}","yellow"))
                close_market_strict(f"COUNCIL_EXIT: {exit_['reason']}")
                LAST_CLOSE_TIME=rf_c.get("time") or int(time.time()*1000)

            # ENTRY priority: Council strong; else RF fallback
            sig=None; tag=""
            if entry:
                sig=entry["side"]; tag=f"[Council] {entry['reason']}"
            elif (not STATE["open"]) and ENTRY_FROM_RF and (rf_c["long"] or rf_c["short"]):
                # فقط لو المجلس غير حاسم
                sig="buy" if rf_c["long"] else "sell"; tag=f"[RF-closed {'LONG' if rf_c['long'] else 'SHORT'}]"

            # Wait-for-next-closed-signal after any close
            if not STATE["open"] and sig and reason is None and WAIT_NEXT_CLOSED:
                if int(rf_c.get("time") or 0) <= int(LAST_CLOSE_TIME or 0):
                    reason="wait_for_next_closed_signal"

            if not STATE["open"] and sig and reason is None:
                qty=compute_size(bal, px or rf_c["price"])
                if qty>0 and (px or rf_c["price"]):
                    if open_market(sig, qty, px or rf_c["price"], tag):
                        LAST_CLOSE_TIME=0
                else:
                    reason="qty<=0 or price=None"

            pretty_snapshot(bal, {"price": px or rf_c["price"], **rf_c}, ind, spread_bps, council.last_log, reason, df)

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
    return f"✅ DOGE SCM Council + RF(Closed) — {SYMBOL} {INTERVAL} — {mode} — Trend-aware strict management"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol":SYMBOL,"interval":INTERVAL,"mode":"live" if MODE_LIVE else "paper",
        "leverage":LEVERAGE,"risk_alloc":RISK_ALLOC,"price":price_now(),
        "state":STATE,"compound_pnl":compound_pnl,
        "guards":{"max_spread_bps":MAX_SPREAD_BPS,"pause_adx":PAUSE_ADX_THRESHOLD},
        "trend_mode": STATE["trend_mode"], "council_log": council.last_log
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
    }), 200

def keepalive_loop():
    url=SELF_URL.rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (SELF_URL not set)","yellow")); return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"scm-rf-keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}","cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# ================== BOOT ===========================
if __name__=="__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}","yellow"))
    print(colored(f"ENTRY: Council strong ⇒ RF(Closed) fallback | ADX≥{PAUSE_ADX_THRESHOLD}","yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
