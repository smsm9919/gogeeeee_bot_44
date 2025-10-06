# ------------ Advanced Position Management Check ------------
def advanced_position_management(candle_info: dict, ind: dict):
    """Handle scale-in, scale-out, and dynamic trailing"""
    if not state["open"]:
        return None
    
    current_side = state["side"]
    px = ind.get("price") or price_now() or state["entry"]
    
    # Scale-in check with dynamic step
    should_scale, step_size, scale_reason = should_scale_in(candle_info, ind, current_side)
    if should_scale:
        scale_in_position(step_size, scale_reason)
        return "SCALE_IN"
    
    # Scale-out check
    should_scale_out, scale_out_reason = should_scale_out(candle_info, ind, current_side)
    if should_scale_out:
        close_partial(0.3, scale_out_reason)  # Close 30% on warning signals
        return "SCALE_OUT"
    
    # Dynamic trailing adjustment
    trail_mult = get_trail_multiplier({**ind, "price": px})
    if state["trail"] is not None and trail_mult != ATR_MULT_TRAIL:
        # Adjust existing trail
        atr = ind.get("atr", 0)
        if current_side == "long":
            new_trail = px - atr * trail_mult
            state["trail"] = max(state["trail"], new_trail)
        else:
            new_trail = px + atr * trail_mult
            state["trail"] = min(state["trail"], new_trail)
        
        if trail_mult != ATR_MULT_TRAIL:
            state["last_action"] = "TRAIL_ADJUST"
            state["action_reason"] = f"Trail mult {ATR_MULT_TRAIL} → {trail_mult}"
            print(colored(f"🔄 TRAIL ADJUST: multiplier {ATR_MULT_TRAIL} → {trail_mult} ({'STRONG' if trail_mult==TRAIL_MULT_STRONG else 'MED' if trail_mult==TRAIL_MULT_MED else 'CHOP'})", "blue"))
    
    return None

# ------------ Enhanced HUD (rich logs) ------------
def snapshot(bal,info,ind,spread_bps,reason=None, df=None):
    df = df if df is not None else fetch_ohlcv()
    left_s = time_to_candle_close(df, USE_TV_BAR)
    insights = build_log_insights(df, ind, info.get("price"))

    print(colored("─"*100,"cyan"))
    print(colored(f"📊 {SYMBOL} {INTERVAL} • {'LIVE' if MODE_LIVE else 'PAPER'} • {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC","cyan"))
    print(colored("─"*100,"cyan"))

    # ===== INDICATORS & CANDLES =====
    print("📈 INDICATORS & CANDLES")
    print(f"   💲 Price {fmt(info.get('price'))}  |  RF filt={fmt(info.get('filter'))}  hi={fmt(info.get('hi'))}  lo={fmt(info.get('lo'))}")
    print(f"   🧮 RSI({RSI_LEN})={fmt(ind['rsi'])}   +DI={fmt(ind['plus_di'])}   -DI={fmt(ind['minus_di'])}   DX={fmt(ind['dx'])}   ADX({ADX_LEN})={fmt(ind['adx'])}   ATR={fmt(ind['atr'])} (~{fmt(insights['atr_pct'],2)}%)")
    print(f"   🎯 Signal  ✅ BUY={info['long']}   ❌ SELL={info['short']}   |   🧮 spread_bps={fmt(spread_bps,2)}")
    print(f"   {insights['regime_emoji']} Regime={insights['regime']}   {insights['bias_emoji']} Bias={insights['bias']}   |   {insights['rsi_zone']}")
    
    candle_info = insights['candle']
    print(f"   🕯️ Candles = {insights['candle_emoji']} {candle_info['name_ar']} / {candle_info['name_en']} (Strength: {candle_info['strength']}/4)")
    print(f"   ⏱️ Candle closes in ~ {left_s}s")
    print()

    # ===== POSITION & MANAGEMENT =====
    print("🧭 POSITION & MANAGEMENT")
    print(f"   💰 Balance {fmt(bal,2)} USDT   Risk={int(RISK_ALLOC*100)}%×{LEVERAGE}x   PostCloseCooldown={post_close_cooldown}")
    if state["open"]:
        lamp = '🟩 LONG' if state['side']=='long' else '🟥 SHORT'
        print(f"   📌 {lamp}  Entry={fmt(state['entry'])}  Qty={fmt(state['qty'],4)}  Bars={state['bars']}  PnL={fmt(state['pnl'])}")
        print(f"   🎯 Management: Scale-ins={state['scale_ins']}/{SCALE_IN_MAX_STEPS}  Scale-outs={state['scale_outs']}  Trail={fmt(state['trail'])}")
        print(f"   📊 TP1_done={state['tp1_done']}  Breakeven={fmt(state['breakeven'])}  HighestProfit={fmt(state['highest_profit_pct'],2)}%")
        if state['last_action']:
            print(f"   🔄 Last Action: {state['last_action']} - {state['action_reason']}")
    else:
        print("   ⚪ FLAT")
    print()

    # ===== ACTION INSIGHTS =====
    print("💡 ACTION INSIGHTS")
    if state["open"] and STRATEGY == "smart":
        current_side = state["side"]
        should_scale, step_size, scale_reason = should_scale_in(candle_info, ind, current_side)
        should_scale_out, scale_out_reason = should_scale_out(candle_info, ind, current_side)
        
        if should_scale:
            print(colored(f"   ✅ SCALE-IN READY: {scale_reason}", "green"))
        elif should_scale_out:
            print(colored(f"   ⚠️ SCALE-OUT ADVISED: {scale_out_reason}", "yellow"))
        else:
            print(colored(f"   ℹ️ HOLD POSITION: {scale_reason}", "blue"))
        
        # Trend Amplifier info
        adx = ind.get("adx", 0)
        tp_multiplier, trail_multiplier = get_dynamic_tp_params(adx)
        if tp_multiplier > 1.0:
            current_tp1_pct = TP1_PCT * tp_multiplier * 100.0
            current_trail_activate = TRAIL_ACTIVATE * trail_multiplier * 100.0
            print(colored(f"   🚀 TREND AMPLIFIER ACTIVE: TP={current_tp1_pct:.2f}% • TrailActivate={current_trail_activate:.2f}%", "cyan"))
        
        # Trail info
        trail_mult = get_trail_multiplier({**ind, "price": info.get("price")})
        trail_type = "STRONG" if trail_mult == TRAIL_MULT_STRONG else "MED" if trail_mult == TRAIL_MULT_MED else "CHOP"
        print(f"   🛡️ Trail Multiplier: {trail_mult} ({trail_type})")
    else:
        print("   🔄 Waiting for trading signals...")
    print()

    # ===== RESULTS =====
    print("📦 RESULTS")
    eff_eq = (bal or 0.0) + compound_pnl if MODE_LIVE else compound_pnl
    print(f"   🧮 CompoundPnL {fmt(compound_pnl)}   🚀 EffectiveEq {fmt(eff_eq)} USDT")
    if reason:
        print(colored(f"   ℹ️ WAIT — reason: {reason}","yellow"))
    print(colored("─"*100,"cyan"))
