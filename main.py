# -*- coding: utf-8 -*-
"""
DOGE/USDT – BingX  • RF+Council OB Bot (Closed-candle fallback)
- Council hunts bottoms/tops via 6 OB patterns (Normal, PinBar, Doji, Engulfing, Tweezer, Manip/Swipe)
- If no clean council entry: fallback to Range Filter on the last CLOSED candle
- Fusion management: TP ladder, BE, ATR-trail, ratchet lock, strict close
- Clear council logs
"""

import os, time, math, random, signal, sys, traceback, logging
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

# =============== ENV ===============
API_KEY   = os.getenv("BINGX_API_KEY", "")
API_SECRET= os.getenv("BINGX_API_SECRET", "")
MODE_LIVE = bool(API_KEY and API_SECRET)

SELF_URL  = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT      = int(os.getenv("PORT", 5000))

SYMBOL    = os.getenv("SYMBOL", "DOGE/USDT:USDT")
INTERVAL  = os.getenv("INTERVAL", "15m")
LEVERAGE  = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC= float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("BINGX_POSITION_MODE", "oneway")

# Range Filter
RF_SOURCE = os.getenv("RF_SOURCE","close")
RF_PERIOD = int(os.getenv("RF_PERIOD", 20))
RF_MULT   = float(os.getenv("RF_MULT", 3.5))
RF_HYST_BPS = float(os.getenv("RF_HYST_BPS", 6))  # فقط للهسترة، إشارة الدخول على الشمعة المغلقة
USE_RF_CLOSED = True   # ← الدخول بالشمعة المغلقة fallback

# Indicators
RSI_LEN = int(os.getenv("RSI_LEN",14))
ADX_LEN = int(os.getenv("ADX_LEN",14))
ATR_LEN = int(os.getenv("ATR_LEN",14))

MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 8.0))

# Management
TP1_PCT_BASE       = float(os.getenv("TP1_PCT_BASE", 0.40))
TP1_CLOSE_FRAC     = float(os.getenv("TP1_CLOSE_FRAC", 0.50))
BREAKEVEN_AFTER    = float(os.getenv("BREAKEVEN_AFTER", 0.30))
TRAIL_ACTIVATE_PCT = float(os.getenv("TRAIL_ACTIVATE_PCT", 1.20))
ATR_TRAIL_MULT     = float(os.getenv("ATR_TRAIL_MULT", 1.6))

FINAL_CHUNK_QTY    = float(os.getenv("FINAL_CHUNK_QTY", 50.0))   # لوت الدوجي الافتراضي
RESIDUAL_MIN_QTY   = float(os.getenv("RESIDUAL_MIN_QTY", 9.0))

CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0

BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# =============== LOGGING ===============
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

# =============== EXCHANGE ===============
def make_ex():
    return ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType":"swap"}
    })
ex = make_ex()
MARKET = {}
AMT_PREC=0; LOT_STEP=None; LOT_MIN=None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
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

# =============== HELPERS ===============
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
        except Exception:
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

# =============== INDICATORS ===============
def wilder_ema(s: pd.Series, n:int):
    return s.ewm(alpha=1/n, adjust=False).mean()
def compute_indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN)+2:
        return {"rsi":50,"plus_di":0,"minus_di":0,"adx":0,"atr":0}
    c,h,l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    tr = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr= wilder_ema(tr, ATR_LEN)

    delta=c.diff(); up=delta.clip(lower=0); dn=(-delta).clip(lower=0)
    rs = wilder_ema(up,RSI_LEN)/wilder_ema(dn,RSI_LEN).replace(0,1e-12)
    rsi = 100-(100/(1+rs))

    upm=h.diff(); dnm=l.shift(1)-l
    plus_dm = upm.where((upm>dnm)&(upm>0),0.0)
    minus_dm= dnm.where((dnm>upm)&(dnm>0),0.0)
    plus_di = 100*(wilder_ema(plus_dm,ADX_LEN)/atr.replace(0,1e-12))
    minus_di= 100*(wilder_ema(minus_dm,ADX_LEN)/atr.replace(0,1e-12))
    dx = (100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0)
    adx = wilder_ema(dx, ADX_LEN)

    i=len(df)-1
    return {
        "rsi": float(rsi.iloc[i]),
        "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]),
        "adx": float(adx.iloc[i]),
        "atr": float(atr.iloc[i])
    }

# =============== RANGE FILTER (TV-like, CLOSED) ===============
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
    """إشارة على الشمعة المُغلقة السابقة فقط."""
    if len(df) < RF_PERIOD+3:
        return {"time": int(df["time"].iloc[-1]) if len(df) else int(time.time()*1000),
                "long": False, "short": False, "filter": float(df["close"].iloc[-1]) if len(df) else 0.0,
                "hi":0.0,"lo":0.0,"price": float(df["close"].iloc[-1]) if len(df) else 0.0}
    src=df[RF_SOURCE].astype(float)
    hi,lo,filt=_rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    # استخدم الشمعة المغلقة: -2 بالنسبة للسعر والفِلتر
    p_prev=float(src.iloc[-3]); p_close=float(src.iloc[-2])
    f_prev=float(filt.iloc[-3]); f_close=float(filt.iloc[-2])
    long_flip  = (p_prev<=f_prev and p_close>f_close and abs((p_close-f_close)/f_close)*10000.0>=RF_HYST_BPS)
    short_flip = (p_prev>=f_prev and p_close<f_close and abs((p_close-f_close)/f_close)*10000.0>=RF_HYST_BPS)
    return {
        "time": int(df["time"].iloc[-2]),
        "price": float(df["close"].iloc[-1]),   # السعر الحالي للتنفيذ
        "long": bool(long_flip),
        "short": bool(short_flip),
        "filter": f_close, "hi": float(hi.iloc[-2]), "lo": float(lo.iloc[-2])
    }

# =============== OB / COUNCIL ==================
def _body_wicks(o,h,l,c):
    rng=max(h-l,1e-12); body=abs(c-o)
    upper=h-max(o,c); lower=min(o,c)-l
    return rng, body, upper, lower

def detect_ob_type(prev_o, prev_h, prev_l, prev_c, o, h, l, c):
    """يصنف نوع الـ OB الصاعد من شمعتين (تقريبي – للصيد فقط)."""
    # normal: شمعة حمراء صغيرة ثم خضراء اندفاعية
    rng1, body1, *_ = _body_wicks(prev_o, prev_h, prev_l, prev_c)
    rng2, body2, up2, lo2 = _body_wicks(o, h, l, c)

    res=None; score=0
    if prev_c<prev_o and body1<=0.5*rng1 and c>o and body2>=0.6*rng2:
        res=("NORMAL", 1)

    # pin bar: ذيل سفلي طويل في الشمعة الثانية
    if c>o and (lo2/rng2)>=0.55:
        res=("PINBAR", max(score,2))

    # doji: جسم صغير جداً في الشمعة الثانية
    if rng2>0 and (body2/rng2)<=0.12 and c>o:
        res=("DOJI", max(score,1))

    # engulfing: الشمعة الثانية تغطي الأولى
    if c>o and (o<=prev_c) and (c>=prev_o):
        res=("ENGULFING", 3)

    # tweezer bottom: قاعين متقاربين
    if abs(l-prev_l)/max(l,1e-9) <= 0.0015 and c>o:
        res=("TWEEZER", 3)

    # manip/swipe: نزول يكسر لوّ سابق ثم ارتداد قوي وإغلاق فوقه
    if l<prev_l and c>o and c>prev_o:
        res=("MANIP", 3)

    return res  # (name, score) or None

def build_demand_zone(prev_o, prev_h, prev_l, prev_c):
    # نعمل box تحت آخر شمعة حمراء: [low, high] = demand
    bot=min(prev_o, prev_c, prev_l); top=max(prev_o, prev_c, prev_l)
    return {"side":"demand","bot":bot,"top":top}

def find_recent_bull_ob(df: pd.DataFrame):
    """يرجع أقوى OB صاعد قريب + نوعه + درجته"""
    if len(df)<5: return None
    d=df.iloc[:-1]  # استخدم شموع مغلقة
    for i in range(len(d)-2, 1, -1):
        prev_o,prev_h,prev_l,prev_c = map(float, d[["open","high","low","close"]].iloc[i-1])
        o,h,l,c = map(float, d[["open","high","low","close"]].iloc[i])
        clas = detect_ob_type(prev_o,prev_h,prev_l,prev_c,o,h,l,c)
        if clas:
            name,score=clas
            zone=build_demand_zone(prev_o,prev_h,prev_l,prev_c)
            return {"name":name,"score":score,"zone":zone,"time":int(d["time"].iloc[i])}
    return None

def near(px, lvl, bps=12.0):
    try: return abs((px-lvl)/lvl)*10000.0 <= bps
    except Exception: return False

class Council:
    def __init__(self):
        self.open=False; self.side=None; self.entry=None
        self.min_votes_entry=4; self.min_votes_exit=3
        self._last_log=None  # يخزن آخر لوج منسق

    def vote(self, df: pd.DataFrame, ind: dict):
        """يجمع الأصوات وأسبابها + log واضح."""
        votes_buy=votes_sell=0; reasons_b=[]; reasons_s=[]
        price=float(df["close"].iloc[-1])

        # 1) OB detection
        ob=find_recent_bull_ob(df)
        if ob:
            z=ob["zone"]
            touched = (float(df["low"].iloc[-1])<=z["top"]) or near(float(df["low"].iloc[-1]), z["top"])
            if touched:
                votes_buy += min(3, ob["score"])
                reasons_b.append(f"OB:{ob['name']}@{fmt(z['bot'])}~{fmt(z['top'])} (+{min(3,ob['score'])})")

        # 2) RSI محايد وانعكاس لأعلى/لأسفل
        rsi=ind.get("rsi",50.0); o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
        if 45<=rsi<=55:
            if c>o: votes_buy+=1; reasons_b.append("RSI neutral ↗")
            if c<o: votes_sell+=1; reasons_s.append("RSI neutral ↘")

        # 3) +DI/-DI مع ADX
        pdi,mdi,adx = ind.get("plus_di",0), ind.get("minus_di",0), ind.get("adx",0)
        if adx>=18 and pdi>mdi: votes_buy+=1; reasons_b.append("+DI>−DI & ADX")
        if adx>=18 and mdi>pdi: votes_sell+=1; reasons_s.append("−DI>+DI & ADX")

        # 4) شموع (هامر/شوتنج ستار)
        rng, body, up, lo = _body_wicks(o,float(df["high"].iloc[-1]),float(df["low"].iloc[-1]),c)
        if (lo/rng)>=0.6 and c>o: votes_buy+=1; reasons_b.append("Hammer-like")
        if (up/rng)>=0.6 and c<o: votes_sell+=1; reasons_s.append("Shooting-like")

        # 5) ATR انفجار حجم/مدى (خفيف)
        atr=float(ind.get("atr") or 0.0)
        if atr>0 and abs(c-o)/atr>=1.2: 
            if c>o: votes_buy+=1; reasons_b.append("Body≥1.2×ATR")
            else:   votes_sell+=1; reasons_s.append("Body≥1.2×ATR (down)")

        # صياغة اللوج المنسق
        council_log = (
            f"🏛 Council | buy={votes_buy} [{', '.join(reasons_b) or '—'}] "
            f"| sell={votes_sell} [{', '.join(reasons_s) or '—'}]"
        )
        self._last_log=council_log
        print(colored(council_log, "green" if votes_buy>votes_sell else "red" if votes_sell>votes_buy else "cyan"))

        return votes_buy, reasons_b, votes_sell, reasons_s, ob

    def entry_decision(self, df, ind):
        b, rb, s, rs, ob = self.vote(df, ind)
        if not self.open:
            if b >= self.min_votes_entry:
                self.open=True; self.side="long"; self.entry=float(df["close"].iloc[-1])
                return {"side":"buy", "reason": f"council {b}✓ :: {rb}", "ob": ob}
            if s >= self.min_votes_entry:
                self.open=True; self.side="short"; self.entry=float(df["close"].iloc[-1])
                return {"side":"sell","reason": f"council {s}✓ :: {rs}", "ob": None}
        return None

    def exit_decision(self, df, ind):
        if not self.open: return None
        # خروج بسيط: ADX يبرد + RSI يعود للحياد + شمعة معاكسة قوية
        rsi=ind.get("rsi",50.0); adx=ind.get("adx",0.0)
        o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
        counter=(c<o and self.side=="long") or (c>o and self.side=="short")
        votes=0; reasons=[]
        if 45<=rsi<=55: votes+=1; reasons.append("RSI neutral")
        if adx<18:      votes+=1; reasons.append("ADX cool-off")
        if counter:     votes+=1; reasons.append("Counter candle")
        if votes>=self.min_votes_exit:
            self.open=False; self.side=None; self.entry=None
            return {"action":"close","reason":" / ".join(reasons)}
        return None

council = Council()

# =============== STATE/ORDERS ===============
STATE = {
    "open":False,"side":None,"entry":None,"qty":0.0,"pnl":0.0,"bars":0,
    "trail":None,"breakeven":None,"tp1_done":False,
    "highest_profit_pct":0.0,"profit_targets_achieved":0
}
compound_pnl=0.0

def _params_open(side):
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if side=="buy" else "SHORT", "reduceOnly":False}
    return {"positionSide":"BOTH","reduceOnly":False}
def _params_close():
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if STATE.get("side")=="long" else "SHORT","reduceOnly":True}
    return {"positionSide":"BOTH","reduceOnly":True}

def _read_position():
    try:
        poss=ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym=(p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if SYMBOL.split(":")[0] not in sym: continue
            qty=abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty<=0: return 0.0,None,None
            entry=float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side_raw=(p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side="long" if "long" in side_raw or float(p.get("cost",0))>0 else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position: {e}")
    return 0.0,None,None

def compute_size(balance, price):
    cap=(balance or 0.0)*RISK_ALLOC*LEVERAGE
    raw=max(0.0, cap/max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def open_market(side, qty, price, tag=""):
    if qty<=0: print(colored("❌ skip open (qty<=0)","red")); return False
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception: pass
            ex.create_order(SYMBOL,"market",side,qty,None,_params_open(side))
        except Exception as e:
            print(colored(f"❌ open: {e}","red")); logging.error(f"open_market: {e}"); return False
    STATE.update({
        "open":True, "side":"long" if side=="buy" else "short", "entry":price,
        "qty":qty, "pnl":0.0, "bars":0, "trail":None, "breakeven":None,
        "tp1_done":False, "highest_profit_pct":0.0, "profit_targets_achieved":0
    })
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)} {tag}","green" if side=='buy' else "red"))
    return True

def close_market_strict(reason="STRICT"):
    global compound_pnl
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty<=0 and not STATE.get("open"): return
    if exch_qty<=0 and STATE.get("open"):
        # ورقي
        px = price_now() or STATE["entry"]; entry=STATE["entry"]; side=STATE["side"]
        pnl=(px-entry)*STATE["qty"]*(1 if side=="long" else -1)
        compound_pnl+=pnl
        print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
        STATE.update({"open":False,"side":None,"entry":None,"qty":0.0})
        return
    side_to_close="sell" if exch_side=="long" else "buy"
    qty_to_close=safe_qty(exch_qty)
    attempts=0; last=None
    while attempts<CLOSE_RETRY_ATTEMPTS:
        try:
            if MODE_LIVE:
                params=_params_close(); params["reduceOnly"]=True
                ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
            time.sleep(CLOSE_VERIFY_WAIT_S)
            left,_,_= _read_position()
            if left<=0:
                px=price_now() or STATE.get("entry") or exch_entry
                entry_px=STATE.get("entry") or exch_entry or px
                side=STATE.get("side") or exch_side
                qty=exch_qty
                pnl=(px-entry_px)*qty*(1 if side=="long" else -1)
                compound_pnl+=pnl
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                STATE.update({"open":False,"side":None,"entry":None,"qty":0.0})
                return
            qty_to_close=safe_qty(left); attempts+=1
            print(colored(f"⚠️ strict close retry {attempts} residual={fmt(left,4)}","yellow"))
        except Exception as e:
            last=e; attempts+=1; time.sleep(CLOSE_VERIFY_WAIT_S)
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

# =============== MANAGEMENT ===============
def _consensus(ind, price, filt, side):
    score=0.0
    adx=float(ind.get("adx") or 0.0); rsi=float(ind.get("rsi") or 50.0)
    if (side=="long" and rsi>=55) or (side=="short" and rsi<=45): score+=1.0
    if adx>=28: score+=1.0
    elif adx>=20: score+=0.5
    try:
        if abs(price-filt)/max(filt,1e-9) >= (RF_HYST_BPS/10000.0): score+=0.5
    except Exception: pass
    return score
def _tp_ladder(info, ind, side):
    px=info["price"]; atr=float(ind.get("atr") or 0.0)
    atr_pct=(atr/max(px,1e-9))*100.0 if px else 0.5
    score=_consensus(ind, px, info.get("filter",px), side)
    mults = [1.8,3.2,5.0] if score>=2.5 else [1.6,2.8,4.5] if score>=1.5 else [1.2,2.4,4.0]
    return [round(m*atr_pct,2) for m in mults],[0.25,0.30,0.45]

def manage_after_entry(df, ind, info):
    if not STATE["open"] or STATE["qty"]<=0: return
    px=info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr=(px-entry)/entry*100*(1 if side=="long" else -1)

    # TP1 + Breakeven
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

    # Ratchet
    if rr>STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=rr
    if STATE["highest_profit_pct"]>=TRAIL_ACTIVATE_PCT and rr<STATE["highest_profit_pct"]*0.60:
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

# =============== UI LOG ===============
def pretty_snapshot(bal, info, ind, spread_bps, extra=None, df=None):
    left_s=time_to_candle_close(df) if df is not None else 0
    print(colored("─"*110,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*110,"cyan"))
    print("📈 INDICATORS")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))} hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))} | spread={fmt(spread_bps,2)}bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))} +DI={fmt(ind.get('plus_di'))} -DI={fmt(ind.get('minus_di'))} ADX={fmt(ind.get('adx'))} ATR={fmt(ind.get('atr'))}")
    if extra and extra.get("council_log"):
        print(colored(extra["council_log"], "green"))
    print(f"   ⏱️ closes_in ≈ {left_s}s")

    print("\n🧭 POSITION")
    bal_line=f"Balance={fmt(bal,2)} Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x CompoundPnL={fmt(compound_pnl)} Eq~{fmt((bal or 0)+compound_pnl,2)}"
    print(colored(f"   {bal_line}","yellow"))
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp} Entry={fmt(STATE['entry'])} Qty={fmt(STATE['qty'],4)} Bars={STATE['bars']} Trail={fmt(STATE['trail'])} BE={fmt(STATE['breakeven'])}")
        print(f"   🎯 TP_done={STATE['profit_targets_achieved']} HP={fmt(STATE['highest_profit_pct'],2)}%")
    else:
        print("   ⚪ FLAT")
    if extra and extra.get("reason"): print(colored(f"   ℹ️ reason: {extra['reason']}","white"))
    print(colored("─"*110,"cyan"))

# =============== MAIN LOOP ===============
app=Flask(__name__)

def trade_loop():
    while True:
        try:
            bal=balance_usdt()
            df = fetch_ohlcv()
            ind=compute_indicators(df)
            spread=orderbook_spread_bps()

            # Council decisions (logs happen inside)
            entry_council = council.entry_decision(df, ind)
            exit_council  = council.exit_decision(df, ind)

            info_rf = rf_signal_closed(df) if USE_RF_CLOSED else {"price": price_now()}

            # Exit first
            if STATE["open"] and exit_council:
                print(colored(f"🏛 Council EXIT → {exit_council['reason']}","yellow"))
                close_market_strict(f"COUNCIL_EXIT: {exit_council['reason']}")

            # Manage open trade
            px=price_now() or info_rf["price"]
            if STATE["open"] and px:
                STATE["pnl"]=(px-STATE["entry"])*STATE["qty"]*(1 if STATE["side"]=="long" else -1)
                manage_after_entry(df, ind, {"price":px, **info_rf})

            # Entry logic
            reason=None
            if spread is not None and spread>MAX_SPREAD_BPS:
                reason=f"spread too high ({fmt(spread,2)}bps > {MAX_SPREAD_BPS})"

            if (reason is None) and (float(ind.get("adx") or 0.0) < 15.0):
                reason=f"ADX<{15} — pause entries"

            if not STATE["open"] and reason is None:
                sig=None; tag=""
                if entry_council:
                    sig=entry_council["side"]; tag=f"[Council] {entry_council['reason']}"
                elif info_rf["long"]:
                    sig="buy"; tag="[RF-closed LONG]"
                elif info_rf["short"]:
                    sig="sell"; tag="[RF-closed SHORT]"

                if sig:
                    qty=compute_size(bal, px or info_rf["price"])
                    if qty>0: open_market(sig, qty, px or info_rf["price"], tag)
                    else: reason="qty<=0"

            # snapshot
            pretty_snapshot(bal, {"price":px or info_rf["price"], **info_rf}, ind, spread,
                            extra={"reason":reason,"council_log":council._last_log}, df=df)

            # bar counter
            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"]+=1

            sleep = NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP
            time.sleep(sleep)
        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}","red"))
            logging.error(f"loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

@app.route("/")
def home():
    mode="LIVE" if MODE_LIVE else "PAPER"
    return f"✅ DOGE RF+Council OB — {SYMBOL} {INTERVAL} — {mode} — Closed-candle RF fallback — FinalChunk={FINAL_CHUNK_QTY}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol":SYMBOL,"interval":INTERVAL,"mode":"live" if MODE_LIVE else "paper",
        "leverage":LEVERAGE,"risk_alloc":RISK_ALLOC,"price":price_now(),
        "state":STATE,"compound_pnl":compound_pnl,"council_log":council._last_log
    })

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (SELF_URL not set)","yellow")); return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"rf-council-keepalive"})
    print(colored(f"KEEPALIVE → {url} every 50s","cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

if __name__=="__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL} • {INTERVAL}","yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}%×{LEVERAGE}x • RF closed fallback={'ON' if USE_RF_CLOSED else 'OFF'}","yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    import threading
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
