# --- أدوات تحسين الأمان والتزامن (لصقها بعد تعريف state و _state_lock) ---
import math

def update_state(mutator):
    """
    يطبق تعديلات على state تحت القفل. استعمله حين تحتاج تعديل عدة حقول معًا.
    """
    if not callable(mutator):
        return
    with _state_lock:
        mutator(state)

def _safe_price(default=None):
    """حارس لقراءة السعر لتفادي None/NaN عند التذبذب أو مهلات."""
    try:
        p = price_now()
        if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
            return default
        return p
    except Exception:
        return default

def _min_tradable_qty(price: float) -> float:
    """
    يحسب أقل كمية قابلة للتداول اعتمادًا على مواصفات السوق.
    مهم لتفادي رفض الإغلاق الجزئي من البورصة.
    """
    try:
        if LOT_MIN and LOT_MIN > 0:
            return float(LOT_MIN)
        if LOT_STEP and LOT_STEP > 0:
            return float(LOT_STEP)
        return 1.0  # آخر حل
    except Exception:
        return 1.0

def _close_partial_min_check(qty_close: float) -> bool:
    """يرفض الإغلاق الجزئي لو الكمية دون الحد الأدنى القابل للتداول."""
    min_qty = _min_tradable_qty(_safe_price(state.get("entry")) or state.get("entry") or 0.0)
    if qty_close < (min_qty or 0.0):
        print(colored(f"⚠️ skip partial close (amount={fmt(qty_close,4)} < min lot {fmt(min_qty,4)})", "yellow"))
        return False
    return True

# --- استبدال دالة apply_trap_guard بالكامل بهذا الإصدار ---
def apply_trap_guard(trap: dict, ind: dict):
    """تفعيل حماية الفخ: جزئي دفاعي + تعادل + تريل مشدود، بكتابة ذرّية للـ state."""
    # why: تقليل سباقات الكتابة وضمان trail/breakeven متسقة
    px  = ind.get("price") or _safe_price(state.get("entry")) or state.get("entry")
    atr = float(ind.get("atr") or 0.0)

    update_state(lambda s: (
        s.__setitem__("_trap_active", True),
        s.__setitem__("_trap_dir", trap.get("trap")),
        s.__setitem__("_trap_left", int(TRAP_HOLD_BARS)),
        s.__setitem__("_last_trap_ts", trap.get("ts")),
        s.__setitem__("breakeven", s.get("breakeven") or s.get("entry"))
    ))

    if state.get("open") and state["qty"] > 0:
        close_partial(0.15, f"Trap guard partial - {trap.get('trap')} trap detected")

    if atr > 0 and px and state.get("open"):
        gap = atr * max(state.get("_adaptive_trail_mult") or ATR_MULT_TRAIL, 1.8)
        def _tighten_trail(s):
            side = s.get("side")
            cur  = s.get("trail")
            if side == "long":
                nt = max(cur or (px - gap), px - gap)
            else:
                nt = min(cur or (px + gap), px + gap)
            s["trail"] = nt
        update_state(_tighten_trail)

    logging.info(f"TRAP GUARD ACTIVATED: {trap.get('trap')} trap - blocking full close for {TRAP_HOLD_BARS} bars")

# --- استبدال دالة close_partial بالكامل بهذا الإصدار ---
def close_partial(frac, reason):
    """
    إغلاق جزئي آمن، مع احترام حد الكمية الدنيا وفق مواصفات السوق،
    ومنع كسر حارس البقايا، وتحديث state تحت القفل عند الحاجة.
    """
    global state, compound_pnl
    if not state["open"]:
        return

    qty_close = safe_qty(max(0.0, state["qty"] * min(max(frac, 0.0), 1.0)))

    # حارس البقايا: لا تدع الكمية المتبقية تهبط تحت الحد الأدنى
    px = _safe_price(state["entry"]) or state["entry"]
    min_qty_guard = max(RESIDUAL_MIN_QTY, (RESIDUAL_MIN_USDT / (px or 1e-9)))

    if state["qty"] - qty_close < min_qty_guard:
        qty_close = safe_qty(max(0.0, state["qty"] - min_qty_guard))
        if qty_close <= 0:
            print(colored("⏸️ skip partial (residual guard would be broken)", "yellow"))
            return

    # حد الكمية الدنيا القابلة للتداول (بدل شرط “< 1 DOGE”)
    if not _close_partial_min_check(qty_close):
        return

    side = "sell" if state["side"] == "long" else "buy"
    if MODE_LIVE:
        try:
            ex.create_order(SYMBOL, "market", side, qty_close, None, _position_params_for_close())
        except Exception as e:
            print(colored(f"❌ partial close: {e}", "red"))
            logging.error(f"close_partial error: {e}")
            return

    px_now = _safe_price(state["entry"]) or state["entry"]
    pnl = (px_now - state["entry"]) * qty_close * (1 if state["side"] == "long" else -1)
    compound_pnl += pnl

    # تحديثات state الذرّية
    def _after_partial(s):
        s["qty"] = safe_qty((s.get("qty") or 0.0) - qty_close)
        s["scale_outs"] = int(s.get("scale_outs") or 0) + 1
        s["last_action"] = "SCALE_OUT"
        s["action_reason"] = reason
    update_state(_after_partial)

    print(colored(f"🔻 PARTIAL {reason} closed={fmt(qty_close,4)} pnl={fmt(pnl)} rem_qty={fmt(state['qty'],4)}", "magenta"))
    logging.info(f"PARTIAL_CLOSE {reason} qty={qty_close} pnl={pnl} remaining={state['qty']}")

    # إن أصبحت الكمية المتبقية دون الحد الأدنى، أغلِق بالكامل إذا سُمح بذلك
    if state["qty"] < min_qty_guard:
        if RESPECT_PATIENT_MODE_FOR_DUST and PATIENT_TRADER_MODE and (not state.get("tp1_done", False)):
            print(colored("⏸️ residual < guard but patient mode blocks full close before TP1", "yellow"))
        else:
            close_market_strict("auto_full_close_small_qty_guard")
        return

    save_state()

# --- تذكير تنفيذي (غير برمجي): ---
# 1) احذف بلوك TRAP المكرر (اترك كتلة واحدة فقط لتعريف TRAP_*).
# 2) داخل trade_loop() أضف في أول السطر داخل الدالة:
#       global last_signal_id
#    لضمان حفظه فعليًا في save_state().
