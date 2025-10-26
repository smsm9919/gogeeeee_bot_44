# -*- codin
TP1_PCT       = 0.40     # % من السعر
TP1_CLOSE_FR  = 0.50     # يغلق 50% في TP1
TP2_PCT       = 1.20     # يصفّي كل الباقي
FULL_TAKE_MIN = 0.60     # % حد أدنى للربح لإتاحة Full Take بالذيول

# تهدئة ومنع الازدواج
COOLDOWN_AFTER_CLOSE_S = 90
MAX_TRADES_PER_DAY     = 12
DAILY_STOP_PCT         = 6.0            # إيقاف يومي لو خسرنا X%

# سيرفر
PORT     = int(os.getenv("PORT", 5000))
SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")

# مفاتيح (لو حطيتها كـ Secrets هتشتغل لايف؛ لو فاضية البوت Paper)
API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET)

# ========= لوجينج =========
def setup_logs():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=3_000_000, backupCount=4, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
setup_logs()

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# ========= بورصة =========
def make_ex():
    ex = ccxt.bingx({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
        "options": {"defaultType": "swap"}
    })
    return ex

ex = make_ex()
MARKET = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN  = None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET   = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision", {}) or {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits", {}) or {}).get("amount", {}).get("step", None)
        LOT_MIN  = (MARKET.get("limits", {}) or {}).get("amount", {}).get("min",  None)
        print(colored(f"🔧 precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}", "cyan"))
        try:
            ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            print(colored(f"✅ leverage set {LEVERAGE}x", "green"))
        except Exception as e:
            print(colored(f"⚠️ set_leverage warn: {e}", "yellow"))
    except Exception as e:
        print(colored(f"⚠️ load markets: {e}", "yellow"))

load_market_specs()

# ========= أدوات =========
def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = int(AMT_PREC) if AMT_PREC and AMT_PREC>=0 else 0
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d < Decimal(str(LOT_MIN)):
            return 0.0
        return float(d)
    except Exception:
        return float(max(0.0, q or 0.0))

def safe_qty(q): 
    q = _round_amt(q)
    if q<=0: print(colored(f"⚠️ qty invalid after normalize → {q}", "yellow"))
    return q

def fmt(v, d=6, na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def with_retry(fn, tries=3, base=0.35):
    for i in range(tries):
        try: return fn()
        except Exception as e:
            if i==tries-1: raise
            time.sleep(base*(2**i) + random.random()*0.2)

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
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

def _iv_sec(iv):
    iv = (iv or "15m").lower()
    if iv.endswith("m"): return int(iv[:-1])*60
    if iv.endswith("h"): return int(iv[:-1])*3600
    if iv.endswith("d"): return int(iv[:-1])*86400
    return 900

def time_to_close(df):
    tf = _iv_sec(INTERVAL)
    if len(df)==0: return tf
    cur_start = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    nxt = cur_start + tf*1000
    while nxt <= now_ms: nxt += tf*1000
    return int(max(0, nxt-now_ms)/1000)

# ========= مؤشرات و RF-Closed =========
def wema(s, n): return s.ewm(alpha=1/n, adjust=False).mean()

def indicators(df: pd.DataFrame):
    if len(df) < max(ATR_LEN, RSI_LEN, ADX_LEN) + 3:
        return {"rsi":50.0,"plus_di":0.0,"minus_di":0.0,"adx":0.0,"atr":0.0,"ema_fast":0.0,"ema_slow":0.0}
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)

    tr  = pd.concat([(h-l).abs(), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    atr = wema(tr, ATR_LEN)

    delta=c.diff(); up=delta.clip(lower=0.0); dn=(-delta).clip(lower=0.0)
    rs = wema(up, RSI_LEN) / wema(dn, RSI_LEN).replace(0,1e-12)
    rsi = 100 - (100/(1+rs))

    up_move=h.diff(); down_move=l.shift(1)-l
    plus_dm=up_move.where((up_move>down_move)&(up_move>0),0.0)
    minus_dm=down_move.where((down_move>up_move)&(down_move>0),0.0)
    plus_di=100*(wema(plus_dm, ADX_LEN)/atr.replace(0,1e-12))
    minus_di=100*(wema(minus_dm, ADX_LEN)/atr.replace(0,1e-12))
    dx=(100*(plus_di-minus_di).abs()/(plus_di+minus_di).replace(0,1e-12)).fillna(0.0)
    adx=wema(dx, ADX_LEN)

    ema_fast = c.ewm(span=EMA_FAST, adjust=False).mean()
    ema_slow = c.ewm(span=EMA_SLOW, adjust=False).mean()

    i=len(df)-1
    return {
        "rsi": float(rsi.iloc[i]),
        "plus_di": float(plus_di.iloc[i]),
        "minus_di": float(minus_di.iloc[i]),
        "adx": float(adx.iloc[i]),
        "atr": float(atr.iloc[i]),
        "ema_fast": float(ema_fast.iloc[i]),
        "ema_slow": float(ema_slow.iloc[i]),
    }

def _ema(s, n): return s.ewm(span=n, adjust=False).mean()
def _rng_size(src, qty, n):
    avrng = _ema((src - src.shift(1)).abs(), n); wper=(n*2)-1
    return _ema(avrng, wper) * qty
def _rng_filter(src, rsize):
    rf=[float(src.iloc[0])]
    for i in range(1,len(src)):
        prev=rf[-1]; x=float(src.iloc[i]); r=float(rsize.iloc[i]); cur=prev
        if x - r > prev: cur = x - r
        if x + r < prev: cur = x + r
        rf.append(cur)
    filt=pd.Series(rf, index=src.index, dtype="float64")
    return filt + rsize, filt - rsize, filt

def rf_signal_closed(df: pd.DataFrame):
    """إشارة RF على الشمعة المغلقة (نقارن آخر شمعتين مغلقتين)."""
    if len(df) < RF_PERIOD + 3:
        i=-2 if len(df)>=2 else -1
        price = float(df["close"].iloc[i])
        return {"time": int(df["time"].iloc[i]), "price": price,
                "long": False, "short": False, "filter": price, "hi": price, "lo": price}
    d = df.iloc[:-1]  # نستخدم المغلقة فقط
    src = d[RF_SOURCE].astype(float)
    hi, lo, filt = _rng_filter(src, _rng_size(src, RF_MULT, RF_PERIOD))
    p_now = float(src.iloc[-1]); p_prev=float(src.iloc[-2])
    f_now = float(filt.iloc[-1]); f_prev=float(filt.iloc[-2])
    def _bps(a,b): 
        try: return abs((a-b)/b)*10000.0
        except Exception: return 0.0
    long_flip  = (p_prev<=f_prev and p_now>f_now and _bps(p_now,f_now)>=RF_HYST_BPS)
    short_flip = (p_prev>=f_prev and p_now<f_now and _bps(p_now,f_now)>=RF_HYST_BPS)
    return {"time": int(d["time"].iloc[-1]), "price": p_now, "long": bool(long_flip), "short": bool(short_flip),
            "filter": f_now, "hi": float(hi.iloc[-1]), "lo": float(lo.iloc[-1])}

# ========= مجلس الإدارة (اصطياد قمم/قيعان + ذيول) =========
def _near(px, lvl, bps):
    try: return abs((px-lvl)/lvl)*10000.0 <= bps
    except Exception: return False

def swing_points(df, left=2, right=2):
    if len(df) < left+right+3: return [], []
    h=df["high"].astype(float).values; l=df["low"].astype(float).values
    ph=[None]*len(df); pl=[None]*len(df)
    for i in range(left, len(df)-right):
        if all(h[i]>=h[j] for j in range(i-left, i+right+1)): ph[i]=h[i]
        if all(l[i]<=l[j] for j in range(i-left, i+right+1)): pl[i]=l[i]
    return ph, pl

def detect_zones(df):
    d = df.iloc[:-1] if len(df)>=2 else df.copy()
    ph, pl = swing_points(d,2,2)
    highs=[p for p in ph if p is not None][-12:]
    lows =[p for p in pl if p is not None][-12:]
    sup=dem=None
    if highs:
        top=max(highs); bot=top - (top-min(highs))*0.25 if len(highs)>1 else top*0.996
        sup={"top":top,"bot":bot}
    if lows:
        bot=min(lows); top=bot + (max(lows)-bot)*0.25 if len(lows)>1 else bot*1.004
        dem={"top":top,"bot":bot}
    return {"supply":sup,"demand":dem}

def wick_rejection(df):
    if len(df)<2: return {"bull":False,"bear":False,"strength":0}
    o=float(df["open"].iloc[-1]); h=float(df["high"].iloc[-1])
    l=float(df["low"].iloc[-1]);  c=float(df["close"].iloc[-1])
    rng=max(h-l,1e-12); upper=h-max(o,c); lower=min(o,c)-l
    bull = (lower/rng)>=0.55 and c>o
    bear = (upper/rng)>=0.55 and c<o
    strength = max(upper, lower)/rng
    return {"bull":bull,"bear":bear,"strength":float(strength)}

def council_entry(df, ind):
    zones = detect_zones(df)
    w = wick_rejection(df)
    px = float(df["close"].iloc[-1])
    # لمس منطقة + ذيل رفض قوي → دخول معاكس
    if zones["demand"]:
        near_dem = (px<=zones["demand"]["top"]) or _near(px,zones["demand"]["top"],12.0)
        if near_dem and w["bull"]: return {"side":"buy", "reason":"Reject@Demand wick"}
    if zones["supply"]:
        near_sup = (px>=zones["supply"]["bot"]) or _near(px,zones["supply"]["bot"],12.0)
        if near_sup and w["bear"]: return {"side":"sell","reason":"Reject@Supply wick"}
    # لو في اتجاه واضح (EMA/ADX) نركب التريند
    if ind["ema_fast"]>ind["ema_slow"] and ind["adx"]>=ADX_TREND_MIN:
        return {"side":"buy","reason":"Trend EMA↑ + ADX"}
    if ind["ema_fast"]<ind["ema_slow"] and ind["adx"]>=ADX_TREND_MIN:
        return {"side":"sell","reason":"Trend EMA↓ + ADX"}
    return None

def full_take_signal(rr_pct, df, ind):
    """قفل كامل لو ربح محترم + ذيول جماعية مؤيدة."""
    if rr_pct < FULL_TAKE_MIN: 
        return False
    w = wick_rejection(df)
    if rr_pct>TP2_PCT*0.9:  # قريب من TP2
        return True
    return (w["bull"] and rr_pct>FULL_TAKE_MIN) or (w["bear"] and rr_pct>FULL_TAKE_MIN)

# ========= الحالة =========
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "tp1_done": False, "bars": 0, "trail": None,
}
compound_pnl = 0.0

SAFE = {
    "BAR_LOCK": None,                # لا دخول أكثر من مرة في نفس الشمعة
    "LAST_CLOSE_AT": 0.0,
    "JUST_CLOSED_UNTIL": 0.0,
    "TRADES_TODAY": 0,
    "DAY_KEY": datetime.utcnow().date().isoformat(),
    "START_EQUITY": None,
    "LOSS_STOP_ACTIVE": False,
}

def reset_day_if_needed(bal):
    today = datetime.utcnow().date().isoformat()
    if SAFE["DAY_KEY"] != today:
        SAFE["DAY_KEY"] = today
        SAFE["TRADES_TODAY"] = 0
        SAFE["LOSS_STOP_ACTIVE"] = False
        SAFE["START_EQUITY"] = (bal or 0.0) + (compound_pnl or 0.0)

def equity_now(bal): return (bal or 0.0) + (compound_pnl or 0.0)

def daily_stop_hit(bal):
    if SAFE["START_EQUITY"] is None:
        SAFE["START_EQUITY"] = equity_now(bal)
    drop = SAFE["START_EQUITY"] - equity_now(bal)
    if SAFE["START_EQUITY"]>0:
        dd = (drop/SAFE["START_EQUITY"])*100.0
        if dd >= DAILY_STOP_PCT: SAFE["LOSS_STOP_ACTIVE"]=True
    return SAFE["LOSS_STOP_ACTIVE"]

# ========= أوامر =========
def _params_open(side):
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if side=="buy" else "SHORT", "reduceOnly":False}
    return {"positionSide":"BOTH", "reduceOnly":False}

def _params_close():
    if POSITION_MODE=="hedge":
        return {"positionSide":"LONG" if STATE.get("side")=="long" else "SHORT", "reduceOnly":True}
    return {"positionSide":"BOTH", "reduceOnly":True}

def _read_position():
    try:
        poss = ex.fetch_positions(params={"type":"swap"})
        for p in poss:
            sym = p.get("symbol") or p.get("info",{}).get("symbol") or ""
            if SYMBOL.split(":")[0] not in sym: continue
            qty = abs(float(p.get("contracts") or p.get("info",{}).get("positionAmt") or 0))
            if qty<=0: return 0.0, None, None
            entry = float(p.get("entryPrice") or p.get("info",{}).get("avgEntryPrice") or 0)
            side_raw = (p.get("side") or p.get("info",{}).get("positionSide") or "").lower()
            side = "long" if ("long" in side_raw or float(p.get("cost",0))>0) else "short"
            return qty, side, entry
    except Exception as e:
        logging.error(f"_read_position: {e}")
    return 0.0, None, None

def compute_size(balance, price):
    cap = (balance or 0.0) * RISK_ALLOC * LEVERAGE
    raw = max(0.0, cap/max(price or 1e-9, 1e-9))
    return safe_qty(raw)

def open_market(side, qty, price, tag=""):
    now = time.time()
    if STATE["open"]:
        print(colored("⏸️ position already open — skip", "yellow")); return False
    if SAFE["LOSS_STOP_ACTIVE"]:
        print(colored("⛔ daily stop active — skip open", "red")); return False
    if SAFE["TRADES_TODAY"] >= MAX_TRADES_PER_DAY:
        print(colored("⛔ max trades per day reached", "red")); return False
    if now - SAFE["LAST_CLOSE_AT"] < COOLDOWN_AFTER_CLOSE_S:
        print(colored("⏳ cooldown after close — skip", "yellow")); return False
    if qty<=0:
        print(colored("❌ qty<=0", "red")); return False
    if MODE_LIVE:
        try:
            try: ex.set_leverage(LEVERAGE, SYMBOL, params={"side":"BOTH"})
            except Exception: pass
            ex.create_order(SYMBOL,"market",side,qty,None,_params_open(side))
        except Exception as e:
            print(colored(f"❌ open: {e}", "red")); logging.error(f"open_market: {e}"); return False
    STATE.update({"open":True, "side":"long" if side=="buy" else "short",
                  "entry":price, "qty":qty, "tp1_done":False, "bars":0, "trail":None})
    SAFE["TRADES_TODAY"] += 1
    print(colored(f"🚀 OPEN {('🟩 LONG' if side=='buy' else '🟥 SHORT')} qty={fmt(qty,4)} @ {fmt(price)} {tag}",
                  "green" if side=='buy' else "red"))
    return True

def close_market_strict(reason="CLOSE"):
    global compound_pnl
    now = time.time()
    if now < SAFE["JUST_CLOSED_UNTIL"]:
        print(colored("🧯 close already triggered — skip duplicate", "yellow")); return
    SAFE["JUST_CLOSED_UNTIL"] = now + 3.0

    exch_qty, exch_side, exch_entry = _read_position()
    if exch_qty<=0 and not STATE["open"]:
        return
    side_to_close = "sell" if (STATE.get("side")=="long" or exch_side=="long") else "buy"
    qty_to_close  = safe_qty(STATE.get("qty") or exch_qty)
    if qty_to_close<=0:
        STATE.update({"open":False,"side":None,"entry":None,"qty":0.0})
        SAFE["LAST_CLOSE_AT"]=now; SAFE["BAR_LOCK"]=None
        return

    try:
        if MODE_LIVE:
            params=_params_close(); params["reduceOnly"]=True
            ex.create_order(SYMBOL,"market",side_to_close,qty_to_close,None,params)
        time.sleep(1.5)
        px = price_now() or STATE.get("entry") or exch_entry
        entry_px = STATE.get("entry") or exch_entry or px
        side = STATE.get("side") or exch_side or ("long" if side_to_close=="sell" else "short")
        pnl = (px - entry_px) * qty_to_close * (1 if side=="long" else -1)
        compound_pnl += pnl
    except Exception as e:
        logging.error(f"close_market_strict: {e}")
    finally:
        STATE.update({"open":False,"side":None,"entry":None,"qty":0.0})
        SAFE["LAST_CLOSE_AT"]=now; SAFE["BAR_LOCK"]=None
        print(colored(f"🔚 CLOSE reason={reason} totalPnL={fmt(compound_pnl)}", "magenta"))

def close_partial(frac, reason):
    if not STATE["open"] or STATE["qty"]<=0: return
    qty = safe_qty(STATE["qty"] * min(max(frac,0.0),1.0))
    if qty<=0: return
    side = "sell" if STATE["side"]=="long" else "buy"
    if MODE_LIVE:
        try:
            params=_params_close(); params["reduceOnly"]=True
            ex.create_order(SYMBOL,"market",side,qty,None,params)
        except Exception as e:
            print(colored(f"❌ partial close: {e}", "red")); return
    px = price_now() or STATE["entry"]
    pnl = (px - STATE["entry"]) * qty * (1 if STATE["side"]=="long" else -1)
    print(colored(f"🔻 PARTIAL {reason} qty={fmt(qty,4)} pnl={fmt(pnl)}", "magenta"))
    STATE["qty"] = safe_qty(STATE["qty"] - qty)
    if STATE["qty"]<=0: 
        close_market_strict("PARTIAL_ALL_DONE")

# ========= الإدارة بعد الدخول (TP1/TP2 فقط + FullTake) =========
def manage_position(df, ind, px):
    if not STATE["open"] or STATE["qty"]<=0: return
    entry = STATE["entry"]; side = STATE["side"]
    rr = (px - entry)/entry*100.0 * (1 if side=="long" else -1)

    # Full take لو ربح محترم + ذيول
    if full_take_signal(rr, df, ind):
        close_market_strict("FULL_TAKE_WICKS")
        return

    # TP1 → يغلق 50% فقط مرّة واحدة
    if (not STATE["tp1_done"]) and rr >= TP1_PCT:
        close_partial(TP1_CLOSE_FR, f"TP1@{TP1_PCT:.2f}%")
        STATE["tp1_done"] = True

    # TP2 → يغلق كل المتبقي
    if rr >= TP2_PCT:
        close_market_strict(f"TP2@{TP2_PCT:.2f}%")

# ========= الطباعة/اللوج لمجلس الإدارة =========
def print_council_log(df, ind, rf, reason_entry=None, reason_skip=None):
    tz = time_to_close(df)
    s = (
        f"\n━━━━━━━━━━━━━━━━ COUNCIL VIEW ━━━━━━━━━━━━━━━━\n"
        f"Symbol {SYMBOL}  TF {INTERVAL}  Mode {'LIVE' if MODE_LIVE else 'PAPER'}\n"
        f"Price {fmt(rf['price'])}  RF={fmt(rf['filter'])}  spread≈{fmt(orderbook_spread_bps(),2)} bps  closes_in≈{tz}s\n"
        f"ADX {fmt(ind['adx'])}  RSI {fmt(ind['rsi'])}  +DI {fmt(ind['plus_di'])}  -DI {fmt(ind['minus_di'])}\n"
        f"EMA Fast {fmt(ind['ema_fast'])}  Slow {fmt(ind['ema_slow'])}\n"
    )
    if reason_entry: s += f"ENTRY ⟶ {reason_entry}\n"
    if reason_skip:  s += f"SKIP  ⟶ {reason_skip}\n"
    if STATE["open"]:
        s += f"POSITION ⟶ {'LONG' if STATE['side']=='long' else 'SHORT'}  entry {fmt(STATE['entry'])} qty {fmt(STATE['qty'],4)} tp1_done={STATE['tp1_done']}\n"
    else:
        s += "POSITION ⟶ FLAT\n"
    s +=   "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    print(colored(s, "cyan"))

# ========= الحلقة الرئيسية =========
def loop():
    while True:
        try:
            bal = balance_usdt()
            reset_day_if_needed(bal)
            if daily_stop_hit(bal):
                print(colored("🛑 Daily stop active — pause today", "red"))
                time.sleep(30); continue

            df  = fetch_ohlcv()
            ind = indicators(df)
            rf  = rf_signal_closed(df)
            px  = price_now() or rf["price"]

            # تحديث إدارة الصفقة
            if STATE["open"]: 
                manage_position(df, ind, px)

            # حارس سبريد و ADX
            reason_skip = None
            sp = orderbook_spread_bps()
            if sp is not None and sp > MAX_SPREAD_BPS:
                reason_skip = f"spread too high ({fmt(sp,2)}bps)"
            if reason_skip is None and ind["adx"] < 15.0:
                reason_skip = "ADX<15 pause"

            # لا دخول أثناء وجود مركز
            if STATE["open"]:
                print_council_log(df, ind, rf, reason_skip=reason_skip)
                time.sleep(5); continue

            # قفل الشمعة (لا ندخل نفس الشمعة مرتين)
            bar_id = int(df["time"].iloc[-2]) if len(df)>=2 else int(df["time"].iloc[-1])
            if SAFE["BAR_LOCK"] == bar_id:
                time.sleep(3); continue

            # قرار مجلس الإدارة/التريند
            entry = None if reason_skip else council_entry(df, ind)

            # لو مفيش قرار واضح → fallback RF-Closed
            if (entry is None) and (reason_skip is None):
                if rf["long"]:  entry = {"side":"buy",  "reason":"RF-Closed LONG"}
                if rf["short"]: entry = {"side":"sell", "reason":"RF-Closed SHORT"}

            # تنفيذ الدخول (صفقة واحدة فقط)
            if entry and (reason_skip is None):
                qty = compute_size(bal, px)
                ok  = open_market(entry["side"], qty, px, tag=f"[{entry['reason']}]")
                if ok:
                    SAFE["BAR_LOCK"] = bar_id
                print_council_log(df, ind, rf, reason_entry=entry["reason"])
            else:
                print_council_log(df, ind, rf, reason_skip=reason_skip)

            # عداد الشموع للمركز لو اتفتح
            if len(df)>=2 and int(df["time"].iloc[-1])!=int(df["time"].iloc[-2]) and STATE["open"]:
                STATE["bars"] += 1

            time.sleep(5)
        except Exception as e:
            logging.error(f"loop: {e}\n{traceback.format_exc()}")
            print(colored(f"❌ loop error: {e}", "red"))
            time.sleep(5)

# ========= API / Keepalive =========
app = Flask(__name__)

@app.route("/")
def root():
    return f"DOGE BingX Bot — {SYMBOL} {INTERVAL} — {'LIVE' if MODE_LIVE else 'PAPER'} — TP1/TP2 only — Council+Trend & RF-Closed"

@app.route("/metrics")
def metrics():
    return jsonify({
        "symbol": SYMBOL, "interval": INTERVAL, "mode": "live" if MODE_LIVE else "paper",
        "leverage": LEVERAGE, "risk_alloc": RISK_ALLOC, "price": price_now(),
        "state": STATE, "compound_pnl": compound_pnl,
        "limits": {"max_trades_day": MAX_TRADES_PER_DAY, "daily_stop_pct": DAILY_STOP_PCT},
    })

def keepalive():
    url=(SELF_URL or "").strip().rstrip("/")
    if not url: return
    import requests
    sess=requests.Session(); sess.headers.update({"User-Agent":"doge-council/keepalive"})
    while True:
        try: sess.get(url, timeout=8)
        except Exception: pass
        time.sleep(50)

# ========= BOOT =========
if __name__ == "__main__":
    print(colored(f"MODE: {'LIVE' if MODE_LIVE else 'PAPER'}  •  {SYMBOL}  •  {INTERVAL}", "yellow"))
    print(colored(f"RISK: {int(RISK_ALLOC*100)}% × {LEVERAGE}x  — TP1={TP1_PCT}%/{int(TP1_CLOSE_FR*100)}%  TP2={TP2_PCT}% (full)", "yellow"))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT,  lambda *_: sys.exit(0))
    import threading
    threading.Thread(target=loop, daemon=True).start()
    threading.Thread(target=keepalive, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
