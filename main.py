# -*- coding: utf-8 -*-
"""
DOGE/USDT — BingX • RF + Council OB/Swing Smart Bot (Closed-candle fallback)
• هدف: اقتناص قاع/قمة مؤكدة (Sweep + Displacement + Retest من OB) ثم إدارة ذكية لركوب الترند
• Council بعد الدخول: يحلل شموع/SMC/ADX/RSI/ATR ويقرر جزئي/Trail/Close صارم عند فقدان القمة
• Strict Highest-Profit Close: عند تسجيل قمة ربح ثم انعكاس قوي → إغلاق كامل صارم
• Harvest على الفتائل الطويلة داخل الربح
• RF (CLOSED candle) fallback عند غياب مدخل Council "نظيف"
• كل القيم من الكود نفسه (مش من ENV) — باستثناء مفاتيح API
"""

import os, time, math, random, signal, sys, traceback, logging, threading
from logging.handlers import RotatingFileHandler
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, InvalidOperation

import pandas as pd
import ccxt
from flask import Flask, jsonify

# ========= مفاتيح (يمكن من ENV)، لكن بقية الإعدادات من الكود =========
API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)

# ========= إعدادات ثابتة من الكود =========
SYMBOL           = "DOGE/USDT:USDT"
INTERVAL         = "15m"
LEVERAGE         = 10
RISK_ALLOC       = 0.60
POSITION_MODE    = "oneway"        # or "hedge"
PORT             = 5000
SELF_URL         = (os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")).strip()

# Range Filter (CLOSED candle fallback)
RF_SOURCE        = "close"
RF_PERIOD        = 20
RF_MULT          = 3.5
RF_HYST_BPS      = 6.0
USE_RF_CLOSED    = True

# مؤشرات
RSI_LEN          = 14
ADX_LEN          = 14
ATR_LEN          = 14
ADX_MIN_ENTRY    = 15.0             # بناءً على طلبك

# حارس سبريد
MAX_SPREAD_BPS   = 8.0

# إدارة مركز
TP1_PCT_BASE       = 0.40           # %
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30           # %
TRAIL_ACTIVATE_PCT = 1.20           # %
ATR_TRAIL_MULT     = 1.6
FINAL_CHUNK_QTY    = 50.0
RESIDUAL_MIN_QTY   = 9.0

# إغلاق صارم عند خسارة نسبة من أعلى ربح محقق
STRICT_CLOSE_DROP_FROM_HP = 0.50    # إذا rr < HP * 0.50 بعد التفعيل → Close صارم
STRICT_COOL_ADX           = 20.0    # ويُفضّل ADX يبرد

# Harvest عند فتائل طويلة داخل الربح
WICK_HARVEST_MIN_PCT      = 0.60    # لازم نكون في ربح ≥ 0.60%
WICK_LONG_FRAC            = 0.30    # يغلق جزئي 30% عند فتيلة واضحة
WICK_RATIO_THRESHOLD      = 0.60    # نسبة الفتيلة للمدى

# BOS/CHoCH و OB
SWING_LOOKBACK            = 60
EQ_BPS                    = 12.0     # مساواة قمم/قيعان (bps)
DISPLACEMENT_ATR_MIN      = 1.2
OB_TOUCH_BPS              = 15.0
OB_EXPIRY_BARS            = 120

# انتظار الإشارة العكسية بعد الإغلاق (صبر محترف)
WAIT_NEXT_SIGNAL_SIDE     = True

# محاولات الإغلاق
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# =============== LOGGING ===============
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    print("🗂️ log rotation ready")
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
EX_LOCK = threading.Lock()

MARKET = {}
AMT_PREC=0; LOT_STEP=None; LOT_MIN=None

def with_retry(fn, tries=3, base=0.4):
    for i in range(tries):
        try: return fn()
        except Exception:
            if i==tries-1: raise
            time.sleep(base*(2**i) + random.random()*0.25)

def _with_ex(fn):
    with EX_LOCK:
        return fn()

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        _with_ex(lambda: ex.load_markets())
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision",{}) or {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits",{}) or {}).get("amount",{}).get("step", None)
        LOT_MIN  = (MARKET.get("limits",{}) or {}).get("amount",{}).get("min",  None)
        print(f"🔧 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}")
    except Exception as e:
        print(f"⚠️ load_market_specs: {e}")

def ensure_leverage_mode():
    try:
        try:
            _with_ex(lambda: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"}))
            print(f"✅ leverage set {LEVERAGE}x")
        except Exception as e:
            print(f"⚠️ set_leverage warn: {e}")
        print(f"📌 position mode: {POSITION_MODE}")
    except Exception as e:
        print(f"⚠️ ensure_leverage_mode: {e}")

try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    print(f"⚠️ exchange init: {e}")

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
    except Exception:
        return max(0.0, float(q))

def safe_qty(q):
    q=_round_amt(q)
    if q<=0: print(f"⚠️ qty invalid after normalize → {q}")
    return q

def fmt(v,d=6,na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isinf(v) or math.isnan(v))): return na
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

# =============== Range Filter (Closed candle) ===============
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
        return {"time": int(df["time"].iloc[-1]) if len(df) else int(time.time()*1000),
                "long": False, "short": False, "filter": float(df["close"].iloc[-1]) if len(df) else 0.0,
                "hi":0.0,"lo":0.0,"price": float(df["close"].iloc[-1]) if len(df) else 0.0}
    src=df[RF_SOURCE].astype(float)
    hi,lo,filt=_rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    p_prev=float(src.iloc[-3]); p_close=float(src.iloc[-2])
    f_prev=float(filt.iloc[-3]); f_close=float(filt.iloc[-2])
    hyst = abs((p_close-f_close)/max(f_close,1e-12))*10000.0 >= RF_HYST_BPS
    long_flip  = (p_prev<=f_prev and p_close>f_close and hyst)
    short_flip = (p_prev>=f_prev and p_close<f_close and hyst)
    return {
        "time": int(df["time"].iloc[-2]),
        "price": float(df["close"].iloc[-1]),
        "long": bool(long_flip),
        "short": bool(short_flip),
        "filter": f_close, "hi": float(hi.iloc[-2]), "lo": float(lo.iloc[-2])
    }

# =============== SMC Utils: Swings / EQ / Sweep / OB / BOS ===============
def _body_wicks(o,h,l,c):
    rng=max(h-l,1e-12); body=abs(c-o)
    upper=h-max(o,c); lower=min(o,c)-l
    return rng, body, upper, lower

def pivots(df: pd.DataFrame, lb=3):
    """يرصد قمم/قيعان محلية بسيطة."""
    if len(df)<lb*2+1: return [],[]
    H=[]; L=[]
    hi=df["high"].astype(float).values
    lo=df["low"].astype(float).values
    for i in range(lb, len(df)-lb):
        if hi[i]==max(hi[i-lb:i+lb+1]): H.append(i)
        if lo[i]==min(lo[i-lb:i+lb+1]): L.append(i)
    return H,L

def near_bps(px, lvl, bps=EQ_BPS):
    try: return abs((px-lvl)/lvl)*10000.0 <= bps
    except Exception: return False

def last_swings(df: pd.DataFrame):
    H,L = pivots(df, lb=3)
    return H[-3:], L[-3:]

def detect_sweep_displacement_retest(df: pd.DataFrame, ind: dict, side: str):
    """
    منطق تأكيد القاع/القمة:
    • Long: كسر قاع سابق (sweep) ثم إغلاق قوي أعلى مع Displacement ≥ 1.2×ATR ثم لمس OB طلب لاحقًا.
    • Short: العكس.
    """
    if len(df) < max(SWING_LOOKBACK, ATR_LEN)+5:
        return None
    d=df.iloc[-SWING_LOOKBACK:-1]  # شموع مغلقة فقط
    atr=float(ind.get("atr") or 0.0)
    if atr<=0: return None

    # آخر شمعتين للتحقق من الإغلاق/dsp
    prev = d.iloc[-2]
    last = d.iloc[-1]

    if side=="long":
        # ابحث عن قاع محلي سابق
        _, L = last_swings(d)
        if not L: return None
        low_idx = L[-1]
        low_lvl = float(d["low"].iloc[low_idx])
        # Sweep: آخر شمعة كسرت القاع ثم أغلقت أعلى من إغلاق الشمعة السابقة وبجسم ≥ 1.2×ATR
        if float(last["low"]) < low_lvl and float(last["close"]) > float(prev["close"]):
            rng, body, up, lo = _body_wicks(float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"]))
            if (body/atr) >= DISPLACEMENT_ATR_MIN:
                # OB: منطقة طلب من آخر شمعة حمراء قبل اندفاع (تقريب)
                # نحدد شمعة قبل last ذات إغلاق أقل من فتحها
                ob = None
                for i in range(len(d)-2, 1, -1):
                    o,h,l,c = map(float, d[["open","high","low","close"]].iloc[i])
                    if c<o:
                        bot=min(o,c,l); top=max(o,c,l)
                        ob={"side":"demand","bot":bot,"top":top,"time":int(d["time"].iloc[i])}
                        break
                return {"type":"LONG_CONFIRM","swept":low_lvl,"ob":ob,"dsp":body/atr}
        return None

    else: # short
        H, _ = last_swings(d)
        if not H: return None
        high_idx = H[-1]
        high_lvl = float(d["high"].iloc[high_idx])
        if float(last["high"]) > high_lvl and float(last["close"]) < float(prev["close"]):
            rng, body, up, lo = _body_wicks(float(last["open"]), float(last["high"]), float(last["low"]), float(last["close"]))
            if (body/atr) >= DISPLACEMENT_ATR_MIN:
                ob = None
                for i in range(len(d)-2, 1, -1):
                    o,h,l,c = map(float, d[["open","high","low","close"]].iloc[i])
                    if c>o:
                        bot=min(o,c,l); top=max(o,c,l)
                        ob={"side":"supply","bot":bot,"top":top,"time":int(d["time"].iloc[i])}
                        break
                return {"type":"SHORT_CONFIRM","swept":high_lvl,"ob":ob,"dsp":body/atr}
        return None

def ob_touched_now(df: pd.DataFrame, zone: dict):
    if not zone: return False
    try:
        low=float(df["low"].iloc[-1]); high=float(df["high"].iloc[-1])
        if zone["side"]=="demand":
            return (low<=zone["top"]) or near_bps(low, zone["top"], OB_TOUCH_BPS)
        else:
            return (high>=zone["bot"]) or near_bps(high, zone["bot"], OB_TOUCH_BPS)
    except Exception:
        return False

# =============== Council ===============
class Council:
    def __init__(self):
        self.open=False; self.side=None; self.entry=None
        self.min_votes_entry=4; self.min_votes_exit=3
        self._last_log=None
        self._ob_cache=None

    def _score_wicks(self, df):
        o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1]); l=float(df["low"].iloc[-1]); c=float(df["close"].iloc[-1])
        rng, body, up, lo = _body_wicks(o,h,l,c)
        s_buy=s_sell=0; rb=[]; rs=[]
        if rng>0:
            if (lo/rng)>=0.60 and c>o: s_buy+=1; rb.append("Hammer-like")
            if (up/rng)>=0.60 and c<o: s_sell+=1; rs.append("Shooting-like")
        return s_buy, rb, s_sell, rs

    def vote(self, df: pd.DataFrame, ind: dict):
        """سياسة متوازنة للطرفين + تأكيد قاع/قمة عبر Sweep/Displacement/OB."""
        votes_buy=votes_sell=0; reasons_b=[]; reasons_s=[]
        price=float(df["close"].iloc[-1])

        # 0) تأكيد قاع/قمة (الأولوية)
        conf_long = detect_sweep_displacement_retest(df, ind, "long")
        conf_short= detect_sweep_displacement_retest(df, ind, "short")
        if conf_long:
            self._ob_cache = conf_long.get("ob")
            if ob_touched_now(df, self._ob_cache):
                votes_buy += 3; reasons_b.append(f"LONG confirm: sweep@{fmt(conf_long['swept'])} dsp≥{DISPLACEMENT_ATR_MIN}×ATR + OB touch")
        if conf_short:
            self._ob_cache = conf_short.get("ob")
            if ob_touched_now(df, self._ob_cache):
                votes_sell += 3; reasons_s.append(f"SHORT confirm: sweep@{fmt(conf_short['swept'])} dsp≥{DISPLACEMENT_ATR_MIN}×ATR + OB touch")

        # 1) DI/ADX
        pdi,mdi,adx = ind.get("plus_di",0), ind.get("minus_di",0), ind.get("adx",0)
        if adx>=20 and pdi>mdi: votes_buy+=1; reasons_b.append("+DI>−DI & ADX≥20")
        if adx>=20 and mdi>pdi: votes_sell+=1; reasons_s.append("−DI>+DI & ADX≥20")

        # 2) RSI neutral drift
        rsi=ind.get("rsi",50.0); o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
        if 45<=rsi<=55:
            if c>o: votes_buy+=1; reasons_b.append("RSI neutral ↗")
            if c<o: votes_sell+=1; reasons_s.append("RSI neutral ↘")

        # 3) شمعات
        sb,rb,ss,rs = self._score_wicks(df)
        votes_buy += sb; reasons_b.extend(rb)
        votes_sell += ss; reasons_s.extend(rs)

        # 4) جسم ≥1.2×ATR
        atr=float(ind.get("atr") or 0.0)
        if atr>0 and abs(c-o)/atr>=1.2:
            if c>o: votes_buy+=1; reasons_b.append("Body≥1.2×ATR")
            else:   votes_sell+=1; reasons_s.append("Body≥1.2×ATR(down)")

        council_log = f"🏛 Council | buy={votes_buy} [{', '.join(reasons_b) or '—'}] | sell={votes_sell} [{', '.join(reasons_s) or '—'}]"
        self._last_log=council_log
        print(council_log)

        return votes_buy, reasons_b, votes_sell, reasons_s

    def entry_decision(self, df, ind):
        b, rb, s, rs = self.vote(df, ind)
        if not self.open:
            if b >= self.min_votes_entry:
                self.open=True; self.side="long"; self.entry=float(df["close"].iloc[-1])
                return {"side":"buy", "reason": f"council {b}✓ :: {rb}"}
            if s >= self.min_votes_entry:
                self.open=True; self.side="short"; self.entry=float(df["close"].iloc[-1])
                return {"side":"sell","reason": f"council {s}✓ :: {rs}"}
        return None

    def exit_decision(self, df, ind):
        if not self.open: return None
        rsi=ind.get("rsi",50.0); adx=ind.get("adx",0.0)
        o=float(df["open"].iloc[-1]); c=float(df["close"].iloc[-1])
        counter=(c<o and self.side=="long") or (c>o and self.side=="short")
        votes=0; reasons=[]
        if 45<=rsi<=55: votes+=1; reasons.append("RSI neutral")
        if adx<20:      votes+=1; reasons.append("ADX cool-off")
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
LAST_CLOSE_SIDE=None
LAST_CLOSE_TIME=0

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
        base = SYMBOL.split(":")[0]
        for p in poss:
            sym=(p.get("symbol") or p.get("info",{}).get("symbol") or "")
            if base not in sym: continue
            qty=abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty<=0: return 0.0,None,None
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
        logging.error(f"_read_position: {e}")
    return 0.0,None,None

def compute_size(balance, price):
    cap=(balance or 0.0)*RISK_ALLOC*LEVERAGE
    raw=max(0.0, cap/max(float(price or 0.0), 1e-9))
    return safe_qty(raw)

def open_market(side, qty, price, tag=""):
    if qty<=0: print("❌ skip open (qty<=0)"); return False
    if MODE_LIVE:
        try:
            try: _with_ex(lambda: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"}))
            except Exception: pass
            _with_ex(lambda: ex.create_order(SYMBOL,"market",side,qty,None,_params_open(side)))
        except Exception as e:
            print(f"❌ open: {e}"); logging.error(f"open_market: {e}"); return False
    STATE.update({
        "open":True, "side":"long" if side=="buy" else "short", "entry":price,
        "qty":qty, "pnl":0.0, "bars":0, "trail":None, "breakeven":None,
        "tp1_done":False, "highest_profit_pct":0.0, "profit_targets_achieved":0
    })
    print(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)} {tag}")
    return True

def _mark_last_close(side):
    global LAST_CLOSE_SIDE, LAST_CLOSE_TIME
    LAST_CLOSE_SIDE = side
    LAST_CLOSE_TIME = int(time.time())

def close_market_strict(reason="STRICT"):
    global compound_pnl
    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty<=0 and not STATE.get("open"): return
    if exch_qty<=0 and STATE.get("open"):
        px = price_now() or STATE["entry"]; entry=STATE["entry"]; side=STATE["side"]
        pnl=(px-entry)*STATE["qty"]*(1 if side=="long" else -1)
        compound_pnl+=pnl
        print(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}")
        _mark_last_close(side)
        STATE.update({"open":False,"side":None,"entry":None,"qty":0.0})
        return
    side_to_close="sell" if exch_side=="long" else "buy"
    qty_to_close=safe_qty(exch_qty)
    attempts=0; last=None
    while attempts<CLOSE_RETRY_ATTEMPTS:
        try:
            if MODE_LIVE:
                params=_params_close(); params["reduceOnly"]=True
                _with_ex(lambda: ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params))
            time.sleep(CLOSE_VERIFY_WAIT_S)
            left,_,_= _read_position()
            if left<=0:
                px=price_now() or STATE.get("entry") or exch_entry
                entry_px=STATE.get("entry") or exch_entry or px
                side=STATE.get("side") or exch_side
                qty=exch_qty
                pnl=(px-entry_px)*qty*(1 if side=="long" else -1)
                compound_pnl+=pnl
                print(f"🔚 STRICT CLOSE {side} reason={reason} pnl={fmt(pnl)} total={fmt(compound_pnl)}")
                _mark_last_close(side)
                STATE.update({"open":False,"side":None,"entry":None,"qty":0.0})
                return
            qty_to_close=safe_qty(left); attempts+=1
            print(f"⚠️ strict close retry {attempts} residual={fmt(left,4)}")
        except Exception as e:
            last=e; attempts+=1; time.sleep(CLOSE_VERIFY_WAIT_S)
    print(f"❌ STRICT CLOSE FAILED last_error={last}")

def close_partial(frac, reason):
    if not STATE["open"] or STATE["qty"]<=0: return
    qty_close=safe_qty(max(0.0, STATE["qty"]*min(max(frac,0.0),1.0)))
    px=price_now() or STATE["entry"]
    min_unit=max(RESIDUAL_MIN_QTY, LOT_MIN or RESIDUAL_MIN_QTY)
    if qty_close<min_unit:
        print(f"⏸️ skip partial (amount={fmt(qty_close,4)} < min_unit={fmt(min_unit,4)})"); return
    side="sell" if STATE["side"]=="long" else "buy"
    if MODE_LIVE:
        try: _with_ex(lambda: ex.create_order(SYMBOL,"market",side,qty_close,None,_params_close()))
        except Exception as e: print(f"❌ partial: {e}"); return
    pnl=(px-STATE["entry"])*qty_close*(1 if STATE["side"]=="long" else -1)
    STATE["qty"]=safe_qty(STATE["qty"]-qty_close)
    print(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem={fmt(STATE['qty'],4)}")
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

def wick_harvest(df, rr):
    """جني جزئي على فتيلة طويلة داخل ربح محترم."""
    if rr < WICK_HARVEST_MIN_PCT or not STATE["open"]: return
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1]); l=float(df["low"].iloc[-1]); c=float(df["close"].iloc[-1])
    rng, body, up, lo = _body_wicks(o,h,l,c)
    if rng<=0: return
    if STATE["side"]=="long" and (up/rng)>=WICK_RATIO_THRESHOLD:
        close_partial(WICK_LONG_FRAC, f"WickHarvest(up {up/rng:.2f})")
    if STATE["side"]=="short" and (lo/rng)>=WICK_RATIO_THRESHOLD:
        close_partial(WICK_LONG_FRAC, f"WickHarvest(down {lo/rng:.2f})")

def strict_highest_profit_close(ind, rr):
    """إغلاق صارم عند فقدان نسبة كبيرة من أعلى ربح، بعد التفعيل."""
    if STATE["highest_profit_pct"]>=TRAIL_ACTIVATE_PCT:
        drop_ratio = 0 if STATE["highest_profit_pct"]==0 else rr/(STATE["highest_profit_pct"]+1e-9)
        if rr < STATE["highest_profit_pct"]*STRICT_CLOSE_DROP_FROM_HP and float(ind.get("adx",0.0))<=STRICT_COOL_ADX:
            close_market_strict(f"STRICT_HP_CLOSE {STATE['highest_profit_pct']:.2f}%→{rr:.2f}%")

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

    # Wick harvest (داخل ربح فقط)
    wick_harvest(df, rr)

    # ATR trail (ركوب الترند)
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

    # Strict highest-profit close
    strict_highest_profit_close(ind, rr)

# =============== UI LOG ===============
def pretty_snapshot(bal, info, ind, spread_bps, reason, df=None):
    left_s=time_to_candle_close(df) if df is not None else 0
    print("─"*110)
    print(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("─"*110)
    print("📈 INDICATORS")
    print(f"   💲 Price {fmt(info.get('price'))} | RF filt={fmt(info.get('filter'))} hi={fmt(info.get('hi'))} lo={fmt(info.get('lo'))} | spread={fmt(spread_bps,2)}bps")
    print(f"   🧮 RSI={fmt(ind.get('rsi'))} +DI={fmt(ind.get('plus_di'))} -DI={fmt(ind.get('minus_di'))} ADX={fmt(ind.get('adx'))} ATR={fmt(ind.get('atr'))}")
    if council._last_log: print(f"   {council._last_log}")
    if reason: print(f"   ℹ️ reason: {reason}")
    print(f"   ⏱️ closes_in ≈ {left_s}s")

    print("\n🧭 POSITION")
    bal_line=f"Balance={fmt(bal,2)} Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x CompoundPnL={fmt(compound_pnl)} Eq~{fmt((bal or 0)+compound_pnl,2)}"
    print(f"   {bal_line}")
    if STATE["open"]:
        lamp='🟩 LONG' if STATE['side']=='long' else '🟥 SHORT'
        print(f"   {lamp} Entry={fmt(STATE['entry'])} Qty={fmt(STATE['qty'],4)} Bars={STATE['bars']} Trail={fmt(STATE['trail'])} BE={fmt(STATE['breakeven'])}")
        print(f"   🎯 TP_done={STATE['profit_targets_achieved']} HP={fmt(STATE['highest_profit_pct'],2)}%")
    else:
        print("   ⚪ FLAT")
    print("─"*110)

# =============== MAIN LOOP ===============
app=Flask(__name__)

def trade_loop():
    while True:
        try:
            bal=balance_usdt()
            df = fetch_ohlcv()
            ind=compute_indicators(df)
            spread=orderbook_spread_bps()

            # Council decisions
            entry_council = council.entry_decision(df, ind)
            exit_council  = council.exit_decision(df, ind)

            info_rf = rf_signal_closed(df) if USE_RF_CLOSED else {"price": price_now()}

            # Exit first (Council)
            if STATE["open"] and exit_council:
                print(f"🏛 Council EXIT → {exit_council['reason']}")
                close_market_strict(f"COUNCIL_EXIT: {exit_council['reason']}")

            # Manage open trade
            px=price_now() or info_rf["price"]
            if STATE["open"] and px:
                STATE["pnl"]=(px-STATE["entry"])*STATE["qty"]*(1 if STATE["side"]=="long" else -1)
                manage_after_entry(df, ind, {"price":px, **info_rf})

            # Entry filters
            reason=None
            if spread is not None and spread>MAX_SPREAD_BPS:
                reason=f"spread too high ({fmt(spread,2)}bps > {MAX_SPREAD_BPS})"

            if (reason is None) and (float(ind.get("adx") or 0.0) < ADX_MIN_ENTRY):
                reason=f"ADX<{int(ADX_MIN_ENTRY)} — pause entries"

            # Wait-for-next-signal after close (صبر)
            if WAIT_NEXT_SIGNAL_SIDE and LAST_CLOSE_SIDE and reason is None and not STATE["open"]:
                # امنع إعادة الدخول في نفس اتجاه آخر إغلاق حتى تتكوّن إشارة مغلقة جديدة
                same_buy  = (LAST_CLOSE_SIDE=="long")  and ((entry_council and entry_council['side']=='buy')  or info_rf.get("long"))
                same_sell = (LAST_CLOSE_SIDE=="short") and ((entry_council and entry_council['side']=='sell') or info_rf.get("short"))
                if same_buy or same_sell:
                    reason="wait_for_next_signal_side"

            # Entry logic
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
            pretty_snapshot(bal, {"price":px or info_rf["price"], **info_rf}, ind, spread, reason, df=df)

            # bar counter
            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"]+=1

            sleep = NEAR_CLOSE_S if time_to_candle_close(df)<=10 else BASE_SLEEP
            time.sleep(sleep)
        except Exception as e:
            print(f"❌ loop error: {e}\n{traceback.format_exc()}")
            logging.error(f"loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

@app.route("/")
def home():
    mode="LIVE" if MODE_LIVE else "PAPER"
    return f"✅ DOGE Smart SMC+RF — {SYMBOL} {INTERVAL} — {mode} — Closed-RF fallback — FinalChunk={FINAL_CHUNK_QTY}"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol":SYMBOL,"interval":INTERVAL,"mode":"live" if MODE_LIVE else "paper",
        "leverage":LEVERAGE,"risk_alloc":RISK_ALLOC,"price":price_now(),
        "state":STATE,"compound_pnl":compound_pnl,"council_log":council._last_log,
        "adx_min_entry": ADX_MIN_ENTRY, "wait_next_signal_side": WAIT_NEXT_SIGNAL_SIDE
    })

def keepalive_loop():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url:
        print("⛔ keepalive disabled (SELF_URL not set)"); return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"rf-council-keepalive"})
    print(f"KEEPALIVE → {url} every 50s")
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

if __name__=="__main__":
    print(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL} • {INTERVAL}")
    print(f"RISK: {int(RISK_ALLOC*100)}%×{LEVERAGE}x • RF closed fallback={'ON' if USE_RF_CLOSED else 'OFF'} • ADX_MIN={ADX_MIN_ENTRY}")
    logging.info("service starting…")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    threading.Thread(target=trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
