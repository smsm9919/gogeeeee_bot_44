# -*- coding: utf-8 -*-
"""
DOGE RF + Council Futures Bot — BingX Perp via CCXT (SMART EDITION)
Order:
  1) Council (tops/bottoms + zones): primary entries & exits
  2) Fallback: Range Filter (TradingView-like) on CLOSED candle only
Post-entry:
  - Dynamic TP ladder + Breakeven + ATR trailing (ratchet)
  - Strict highest-profit close + wick-harvest
  - Defensive opposite-RF while in trade
HTTP:
  - / , /metrics , /health
"""

import os, time, math, random, signal, sys, traceback, logging, threading
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

# ============== KEYS / MODE ================
API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)

SELF_URL = (os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")).strip()
PORT     = 5000

# ============== FIXED SETTINGS =============
SYMBOL        = "DOGE/USDT:USDT"
INTERVAL      = "15m"
LEVERAGE      = 10
RISK_ALLOC    = 0.60
POSITION_MODE = "oneway"      # or "hedge"

# Range Filter (CLOSED candle)
RF_SOURCE   = "close"
RF_PERIOD   = 20
RF_MULT     = 3.5
RF_HYST_BPS = 6.0
ENTRY_FROM_RF = True

# Indicators
RSI_LEN = 14; ADX_LEN = 14; ATR_LEN = 14

# Guards
MAX_SPREAD_BPS        = 8.0
PAUSE_ADX_THRESHOLD   = 15.0   # يبدأ التداول من 15 (حسب طلبك)
WAIT_NEXT_SIGNAL_SIDE = True   # انتظار إشارة مغلقة جديدة بعد الإغلاق
OPP_RF_VOTES_NEEDED   = 2
OPP_RF_MIN_ADX        = 22.0
OPP_RF_MIN_HYST_BPS   = 8.0

# Management
TP1_PCT_BASE       = 0.40
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.6
RATCHET_LOCK_FALLBACK = 0.60
FINAL_CHUNK_QTY    = 50.0
RESIDUAL_MIN_QTY   = 9.0

# Strict close on losing peak gains
STRICT_CLOSE_DROP_FROM_HP = 0.50   # فقدان ≥50% من أعلى ربح
STRICT_COOL_ADX           = 20.0   # مع ADX بارد

# Wick harvest
WICK_HARVEST_MIN_PCT   = 0.60
WICK_LONG_FRAC         = 0.30
WICK_RATIO_THRESHOLD   = 0.60

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# ============== LOGGING ====================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h,"baseFilename","").endswith("bot.log") for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    print(colored("🗂️ log rotation ready","cyan"))
setup_file_logging()

# ============== EXCHANGE ===================
def make_ex():
    return ccxt.bingx({
        "apiKey": API_KEY, "secret": API_SECRET,
        "enableRateLimit": True, "timeout": 20000,
        "options": {"defaultType":"swap"}
    })
ex = make_ex()
EX_LOCK = threading.Lock()  # thread-safe ccxt access

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

# ============== HELPERS ====================
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

# ============== INDICATORS ==================
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

# ============== RF (CLOSED) =================
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

# ============== COUNCIL (zones) =============
def _find_swings(df: pd.DataFrame, left:int=2, right:int=2):
    if len(df) < left+right+3:
        return None, None
    h = df["high"].astype(float).values
    l = df["low"].astype(float).values
    ph = [None]*len(df); pl = [None]*len(df)
    for i in range(left, len(df)-right):
        if all(h[i] >= h[j] for j in range(i-left, i+right+1)): ph[i] = h[i]
        if all(l[i] <= l[j] for j in range(i-left, i+right+1)): pl[i] = l[i]
    return ph, pl

def _near_level(px, lvl, bps):
    try: return abs((px-lvl)/lvl)*10000.0 <= bps
    except Exception: return False

class Council:
    def __init__(self):
        self.state={"open":False,"side":None,"entry":None}
        self.min_votes_entry=4; self.min_votes_exit=3
        self.level_near_bps=12.0
        self.rsi_neutral_min=45.0; self.rsi_neutral_max=55.0
        self._last_log=None

    def detect_zones(self, df: pd.DataFrame):
        d=df.iloc[:-1] if len(df)>=2 else df.copy()
        ph,pl=_find_swings(d,2,2)
        highs=[p for p in ph if p is not None][-20:]
        lows =[p for p in pl if p is not None][-20:]
        sup=dem=None
        if highs:
            top=max(highs); bot=top-(top-min(highs))*0.25
            sup={"side":"supply","top":top,"bot":bot}
        if lows:
            bot=min(lows); top=bot+(max(lows)-bot)*0.25 if len(lows)>1 else bot*1.002
            dem={"side":"demand","top":top,"bot":bot}
        return {"supply":sup,"demand":dem}

    def touch_reject(self, df: pd.DataFrame, zones):
        if len(df)<2: return {"rej_sup":False,"rej_dem":False}
        o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1]); l=float(df["low"].iloc[-1]); c=float(df["close"].iloc[-1])
        rng=max(h-l,1e-12); up=h-max(o,c); dn=min(o,c)-l
        sup=zones.get("supply"); dem=zones.get("demand")
        rej_sup=rej_dem=False
        if sup:
            mid=(sup["top"]+sup["bot"])/2.0
            if (h>=sup["bot"] or _near_level(h,sup["bot"],self.level_near_bps)) and c<mid and (up/rng)>=0.5:
                rej_sup=True
        if dem:
            mid=(dem["top"]+dem["bot"])/2.0
            if (l<=dem["top"] or _near_level(l,dem["top"],self.level_near_bps)) and c>mid and (dn/rng)>=0.5:
                rej_dem=True
        return {"rej_sup":rej_sup,"rej_dem":rej_dem}

    def votes(self, df: pd.DataFrame, ind: dict, info: dict, zones: dict):
        b=s=0; rb=[]; rs=[]
        rej=self.touch_reject(df, zones)
        if rej["rej_dem"]: b+=1; rb.append("reject@demand")
        if rej["rej_sup"]: s+=1; rs.append("reject@supply")

        pdi,mdi,adx = ind.get("plus_di",0), ind.get("minus_di",0), ind.get("adx",0)
        if pdi>mdi and adx>=18: b+=1; rb.append("DI+>DI- & ADX")
        if mdi>pdi and adx>=18: s+=1; rs.append("DI->DI+ & ADX")

        rsi=ind.get("rsi",50.0); o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
        if self.rsi_neutral_min<=rsi<=self.rsi_neutral_max:
            if c>o: b+=1; rb.append("RSI_neutral_up")
            else:  s+=1; rs.append("RSI_neutral_down")

        rng=max(float(df["high"].iloc[-1])-float(df["low"].iloc[-1]),1e-12)
        up=float(df["high"].iloc[-1])-max(o,c); dn=min(o,c)-float(df["low"].iloc[-1])
        if (dn/rng)>=0.6 and c>o: b+=1; rb.append("hammer_like")
        if (up/rng)>=0.6 and c<o: s+=1; rs.append("shooting_like")

        if info.get("long"):  b+=1; rb.append("rf_closed_long")
        if info.get("short"): s+=1; rs.append("rf_closed_short")
        self._last_log = f"🏛 Council | buy={b} [{', '.join(rb) or '—'}] | sell={s} [{', '.join(rs) or '—'}]"
        print(colored(self._last_log, "green" if b>s else "red" if s>b else "cyan"))
        return b, rb, s, rs

    def update(self, df: pd.DataFrame, ind: dict, rf_closed_info: dict):
        if df is None or len(df)<max(RF_PERIOD,ATR_LEN,RSI_LEN,ADX_LEN)+3:
            return {"entry":None, "exit":None, "debug":{"reason":"warmup"}}
        zones=self.detect_zones(df)
        bv,br,sv,sr = self.votes(df, ind, rf_closed_info, zones)
        entry=None; exit_=None

        if not self.state["open"]:
            if bv>=self.min_votes_entry:
                entry={"side":"buy","reason":f"council {bv}✓ :: {br}"}
                self.state.update({"open":True,"side":"long","entry":float(df['close'].iloc[-1])})
            elif sv>=self.min_votes_entry:
                entry={"side":"sell","reason":f"council {sv}✓ :: {sr}"}
                self.state.update({"open":True,"side":"short","entry":float(df['close'].iloc[-1])})

        if self.state["open"]:
            rej=self.touch_reject(df, zones)
            if self.state["side"]=="long" and rej["rej_sup"]:
                exit_={"action":"close","reason":"reject@supply"}
            if self.state["side"]=="short" and rej["rej_dem"]:
                exit_={"action":"close","reason":"reject@demand"}

        return {"entry":entry,"exit":exit_,"debug":{"zones":zones,"votes":{"buy":bv,"buy_reasons":br,"sell":sv,"sell_reasons":sr},"state":self.state.copy()}}

council = Council()

# ============== STATE / ORDERS =============
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0, "opp_votes": 0
}
compound_pnl = 0.0
LAST_CLOSE_CANDLE_TIME = 0  # لانتظار شمعة مغلقة جديدة بعد الإغلاق

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
                upnl=float(p.get("unrealizedPnl") or p.get("info",{}).get("unrealizedPnl") or 0.0)
                last=price_now() or entry
                side="long" if (last>=entry and upnl>=0) or (last<entry and upnl<0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position error: {e}")
    return 0.0, None, None

def compute_size(balance, price):
    cap=(balance or 0.0)*RISK_ALLOC*LEVERAGE
    raw=max(0.0, cap/max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def open_market(side, qty, price, tag=""):
    if qty<=0: print(colored("❌ skip open (qty<=0)","red")); return False
    if MODE_LIVE:
        try:
            try: _with_ex(lambda: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"}))
            except Exception: pass
            _with_ex(lambda: ex.create_order(SYMBOL,"market",side,qty,None,_params_open(side)))
        except Exception as e:
            print(colored(f"❌ open: {e}","red")); logging.error(f"open_market error: {e}"); return False
    STATE.update({
        "open":True,"side":"long" if side=="buy" else "short","entry":price,"qty":qty,
        "pnl":0.0,"bars":0,"trail":None,"breakeven":None,
        "tp1_done":False,"highest_profit_pct":0.0,"profit_targets_achieved":0,"opp_votes":0
    })
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)} {tag}","green" if side=='buy' else "red"))
    logging.info(f"OPEN {side} qty={qty} price={price}")
    return True

def _mark_closed_bar_time(rf_info):
    global LAST_CLOSE_CANDLE_TIME
    try:
        LAST_CLOSE_CANDLE_TIME = int(rf_info.get("time") or int(time.time()*1000))
    except Exception:
        LAST_CLOSE_CANDLE_TIME = int(time.time()*1000)

def _reset_after_close(reason):
    STATE.update({
        "open": False, "side": None, "entry": None, "qty": 0.0,
        "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
        "tp1_done": False, "highest_profit_pct": 0.0,
        "profit_targets_achieved": 0, "opp_votes": 0
    })
    logging.info(f"AFTER_CLOSE: {reason}")

def close_market_strict(reason="STRICT"):
    global compound_pnl
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty <= 0 and not STATE.get("open"):
        return
    if exch_qty <= 0 and STATE.get("open"):
        px = price_now() or STATE["entry"]; side=STATE["side"]; qty=STATE["qty"]; entry=STATE["entry"]
        pnl = (px-entry)*qty*(1 if side=="long" else -1)
        compound_pnl += pnl
        print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
        _reset_after_close(reason); return
    side_to_close="sell" if exch_side=="long" else "buy"
    qty_to_close=safe_qty(exch_qty)
    attempts=0; last_error=None
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
                qty=exch_qty
                pnl=(px-entry_px)*qty*(1 if side=="long" else -1)
                compound_pnl+=pnl
                print(colored(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}","magenta"))
                _reset_after_close(reason); return
            qty_to_close=safe_qty(left); attempts+=1
            print(colored(f"⚠️ strict close retry {attempts} residual={fmt(left,4)}","yellow"))
        except Exception as e:
            last_error=e; attempts+=1; time.sleep(2.0)
    print(colored(f"❌ STRICT CLOSE FAILED — last_error={last_error}","red"))

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
        except Exception as e: print(colored(f"❌ partial close: {e}","red")); return
    pnl=(px-STATE["entry"])*qty_close*(1 if STATE["side"]=="long" else -1)
    STATE["qty"]=safe_qty(STATE["qty"]-qty_close)
    logging.info(f"PARTIAL_CLOSE {reason} qty={qty_close} pnl={pnl} rem={STATE['qty']}")
    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(STATE['qty'],4)}","magenta"))
    if STATE["qty"]<=FINAL_CHUNK_QTY and STATE["qty"]>0:
        print(colored(f"🧹 Final chunk ≤ {FINAL_CHUNK_QTY} DOGE → strict close","yellow"))
        close_market_strict("FINAL_CHUNK_RULE")

# ============== DEFENSE: OPPOSITE RF ========
def defensive_on_opposite_rf(ind: dict, info: dict):
    if not STATE["open"] or STATE["qty"]<=0: return
    px = info.get("price") or price_now() or STATE["entry"]
    rf = info.get("filter")
    # partial + BE + tighten trail
    base_frac = 0.25 if not STATE.get("tp1_done") else 0.20
    close_partial(base_frac, "Opposite RF — defensive")
    if STATE.get("breakeven") is None: STATE["breakeven"]=STATE["entry"]
    atr=float(ind.get("atr") or 0.0)
    if atr>0 and px is not None:
        gap=atr*max(ATR_TRAIL_MULT,1.2)
        if STATE["side"]=="long": STATE["trail"]=max(STATE["trail"] or (px-gap), px-gap)
        else: STATE["trail"]=min(STATE["trail"] or (px+gap), px+gap)
    # voting
    # reset opp_votes إذا اختفى العكس
    opp = (STATE["side"]=="long" and info.get("short")) or (STATE["side"]=="short" and info.get("long"))
    if not opp: 
        STATE["opp_votes"]=0
        return
    STATE["opp_votes"]=int(STATE.get("opp_votes",0))+1
    hyst=0.0
    try:
        if px and rf: hyst=abs((px-rf)/rf)*10000.0
    except Exception: pass
    adx=float(ind.get("adx") or 0.0)
    if STATE["opp_votes"]>=OPP_RF_VOTES_NEEDED and (adx>=OPP_RF_MIN_ADX) and (hyst>=OPP_RF_MIN_HYST_BPS):
        close_market_strict("OPPOSITE_RF_CONFIRMED")

# ============== MANAGEMENT ==================
def _consensus(ind, info, side) -> float:
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

def _tp_ladder(info, ind, side):
    px=info["price"]; atr=float(ind.get("atr") or 0.0)
    atr_pct=(atr/max(px,1e-9))*100.0 if px else 0.5
    score=_consensus(ind, info, side)
    mults = [1.8,3.2,5.0] if score>=2.5 else [1.6,2.8,4.5] if score>=1.5 else [1.2,2.4,4.0]
    return [round(m*atr_pct,2) for m in mults],[0.25,0.30,0.45]

def wick_harvest(df, rr):
    if rr < WICK_HARVEST_MIN_PCT or not STATE["open"]: return
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1]); l=float(df["low"].iloc[-1]); c=float(df["close"].iloc[-1])
    rng=max(h-l,1e-12); up=h-max(o,c); dn=min(o,c)-l
    if STATE["side"]=="long" and (up/rng)>=WICK_RATIO_THRESHOLD:
        close_partial(WICK_LONG_FRAC, f"WickHarvest(up {up/rng:.2f})")
    if STATE["side"]=="short" and (dn/rng)>=WICK_RATIO_THRESHOLD:
        close_partial(WICK_LONG_FRAC, f"WickHarvest(down {dn/rng:.2f})")

def strict_highest_profit_close(ind, rr):
    if STATE["highest_profit_pct"]>=TRAIL_ACTIVATE_PCT:
        if rr < STATE["highest_profit_pct"]*STRICT_CLOSE_DROP_FROM_HP and float(ind.get("adx",0.0))<=STRICT_COOL_ADX:
            close_market_strict(f"STRICT_HP_CLOSE {STATE['highest_profit_pct']:.2f}%→{rr:.2f}%")

def manage_after_entry(df, ind, info):
    if not STATE["open"] or STATE["qty"]<=0: return
    px=info["price"]; entry=STATE["entry"]; side=STATE["side"]
    rr=(px-entry)/entry*100*(1 if side=="long" else -1)

    # TP1 + BE
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

    # Highest profit tracking + ratchet
    if rr>STATE["highest_profit_pct"]: STATE["highest_profit_pct"]=rr
    if STATE["highest_profit_pct"]>=TRAIL_ACTIVATE_PCT and rr<STATE["highest_profit_pct"]*RATCHET_LOCK_FALLBACK:
        close_partial(0.50, f"Ratchet {STATE['highest_profit_pct']:.2f}%→{rr:.2f}%")

    # Wick harvest
    wick_harvest(df, rr)

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

    # STRICT close on losing peak
    strict_highest_profit_close(ind, rr)

# ============== SNAPSHOT ====================
def pretty_snapshot(bal, info, ind, spread_bps, reason=None, df=None):
    left_s = time_to_candle_close(df) if df is not None else 0
    print(colored("─"*110,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*110,"cyan"))
    print("📈 RF CLOSED")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))} hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))} | spread={fmt(spread_bps,2)} bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))} +DI={fmt(ind.get('plus_di'))} -DI={fmt(ind.get('minus_di'))} ADX={fmt(ind.get('adx'))} ATR={fmt(ind.get('atr'))}")
    print(f"   ⏱️ closes_in ≈ {left_s}s")
    print("\n🧭 POSITION")
    bal_line=f"Balance={fmt(bal,2)} Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x CompoundPnL={fmt(compound_pnl)} Eq~{fmt((bal or 0)+compound_pnl,2)}"
    print(colored(f"   {bal_line}","yellow"))
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp} Entry={fmt(STATE['entry'])} Qty={fmt(STATE['qty'],4)} Bars={STATE['bars']} Trail={fmt(STATE['trail'])} BE={fmt(STATE['breakeven'])}")
        print(f"   🎯 TP_done={STATE['profit_targets_achieved']} HP={fmt(STATE['highest_profit_pct'],2)}% OppVotes={STATE.get('opp_votes',0)}")
    else:
        print("   ⚪ FLAT")
    if reason: print(colored(f"   ℹ️ reason: {reason}","white"))
    print(colored("─"*110,"cyan"))

# ============== MAIN LOOP ===================
app = Flask(__name__)

def trade_loop():
    global LAST_CLOSE_CANDLE_TIME
    while True:
        try:
            bal=balance_usdt()
            px =price_now()
            df =fetch_ohlcv()

            ind  = compute_indicators(df)
            rf_c = rf_signal_closed(df)   # closed-candle only
            spread_bps = orderbook_spread_bps()

            # Council decisions
            council_decision = council.update(df, ind, rf_c)

            # Update PnL & manage
            if STATE["open"] and px:
                STATE["pnl"]=(px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            manage_after_entry(df, ind, {"price": px or rf_c["price"], **rf_c})

            # Defensive opposite RF (inside trade)
            if STATE["open"]:
                opp = (STATE["side"]=="long" and rf_c["short"]) or (STATE["side"]=="short" and rf_c["long"])
                if opp: defensive_on_opposite_rf(ind, {"price": px or rf_c["price"], **rf_c})
                else: STATE["opp_votes"]=0  # reset votes when no opposite anymore

            # Guards
            reason=None
            if spread_bps is not None and spread_bps > MAX_SPREAD_BPS:
                reason=f"spread too high ({fmt(spread_bps,2)}bps > {MAX_SPREAD_BPS})"
            if (reason is None) and (float(ind.get("adx") or 0.0) < PAUSE_ADX_THRESHOLD):
                reason=f"ADX<{PAUSE_ADX_THRESHOLD:.0f} — trading paused"

            # Council exit
            if STATE["open"] and council_decision["exit"]:
                print(colored(f"🏛️ COUNCIL EXIT: {council_decision['exit']['reason']}","yellow"))
                close_market_strict(f"COUNCIL_EXIT: {council_decision['exit']['reason']}")
                _mark_closed_bar_time(rf_c)

            # Entry
            sig=None; tag=""
            if council_decision["entry"]:
                sig=council_decision["entry"]["side"]; tag=f"[Council] {council_decision['entry']['reason']}"
            elif ENTRY_FROM_RF and (rf_c["long"] or rf_c["short"]):
                sig="buy" if rf_c["long"] else "sell"; tag=f"[RF-closed {'LONG' if rf_c['long'] else 'SHORT'}]"

            # Wait-for-next-closed-signal after any close (no same-bar re-entry)
            if not STATE["open"] and sig and reason is None and WAIT_NEXT_SIGNAL_SIDE:
                # لازم تكون إشارة الشمعة المغلقة الحالية أحدث من آخر إغلاق
                if int(rf_c.get("time") or 0) <= int(LAST_CLOSE_CANDLE_TIME or 0):
                    reason="wait_for_next_closed_signal"

            if not STATE["open"] and sig and reason is None:
                qty=compute_size(bal, px or rf_c["price"])
                if qty>0 and (px or rf_c["price"]):
                    if open_market(sig, qty, px or rf_c["price"], tag):
                        LAST_CLOSE_CANDLE_TIME = 0  # تم الدخول بعد انتظار
                else:
                    reason="qty<=0 or price=None"

            pretty_snapshot(bal, {"price": px or rf_c["price"], **rf_c}, ind, spread_bps, reason, df)

            # bar counter
            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"]+=1

            time.sleep(NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP)
        except Exception as e:
            print(colored(f"❌ loop error: {e}\n{traceback.format_exc()}","red"))
            logging.error(f"trade_loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# ============== API / KEEPALIVE =============
@app.route("/")
def home():
    mode='LIVE' if MODE_LIVE else 'PAPER'
    return f"✅ DOGE Council+RF Closed — {SYMBOL} {INTERVAL} — {mode} — TP/BE/Trail — Strict HP Close — FinalChunk={FINAL_CHUNK_QTY}DOGE"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "guards": {"max_spread_bps": MAX_SPREAD_BPS, "pause_adx": PAUSE_ADX_THRESHOLD, "final_chunk_qty": FINAL_CHUNK_QTY},
        "wait_policy": {"wait_next_closed_signal": WAIT_NEXT_SIGNAL_SIDE, "last_close_candle_time": LAST_CLOSE_CANDLE_TIME}
    })

@app.route("/health")
def health():
    return jsonify({
        "ok": True, "mode": "live" if MODE_LIVE else "paper",
        "open": STATE["open"], "side": STATE["side"], "qty": STATE["qty"],
        "compound_pnl": compound_pnl, "timestamp": datetime.utcnow().isoformat(),
        "tp_done": STATE.get("profit_targets_achieved", 0), "opp_votes": STATE.get("opp_votes",0)
    }), 200

def keepalive_loop():
    url=SELF_URL.rstrip("/")
    if not url:
        print(colored("⛔ keepalive disabled (SELF_URL not set)","yellow")); return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"doge-council-rf/keepalive"})
    print(colored(f"KEEPALIVE every 50s → {url}","cyan"))
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# ============== BOOT ========================
if __name__ == "__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}","yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  •  ENTRY: Council ⇒ RF(CLOSED)","yellow"))
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
