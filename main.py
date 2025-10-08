# -*- coding: utf-8 -*-
"""
RF Futures Bot — Enhanced False Reversal & EOC Detection Module
- Advanced false reversal detection with multiple confirmation signals
- Dynamic position management based on candle patterns and market regime
- Integrated with main bot's state management
"""

import math
from termcolor import colored

def _is_false_reversal_enhanced(df, ind, side: str) -> bool:
    """
    Enhanced false reversal detection with multiple factors:
    - Candle pattern analysis
    - ADX trend strength
    - RSI and DI confirmation
    - Volume and volatility context
    """
    try:
        if len(df) < 3:
            return False
            
        patt = detect_candle_pattern(df)
        pattern = patt.get("pattern", "NONE")
        adx = float(ind.get("adx") or 0)
        rsi = float(ind.get("rsi") or 50)
        plus_di = float(ind.get("plus_di") or 0)
        minus_di = float(ind.get("minus_di") or 0)
        atr = float(ind.get("atr") or 0)
        atr_pct = (atr / (ind.get("price") or 1)) * 100

        # Weak reversal patterns that might be false signals
        weak_reversal_patterns = {
            "DOJI", "HAMMER", "SHOOTING_STAR", "INVERTED_HAMMER",
            "SPINNING_TOP", "HIGH_WAVE"
        }

        # Strong continuation patterns that override weak reversals
        strong_continuation_patterns = {
            "MARUBOZU_BULL", "MARUBOZU_BEAR", "THREE_WHITE_SOLDIERS", 
            "THREE_BLACK_CROWS", "ENGULF_BULL", "ENGULF_BEAR"
        }

        current_pattern = detect_candle_pattern(df)
        
        # If we have a strong continuation pattern, definitely false reversal
        if current_pattern.get("pattern") in strong_continuation_patterns:
            print(colored(f"🎯 STRONG CONTINUATION: {current_pattern.get('name_en')} - ignoring reversal", "green"))
            return True

        # Check if current pattern is a weak reversal
        if pattern in weak_reversal_patterns and adx >= 25:
            # Additional confirmation for false reversal
            volume_ok = True  # You could add volume analysis here
            volatility_ok = atr_pct > 0.5  # Minimum volatility threshold
            
            if side == "long" and (rsi >= 55 and plus_di > minus_di) and volume_ok and volatility_ok:
                print(colored(f"⚡ FALSE REVERSAL DETECTED: {pattern} in strong uptrend (ADX:{adx:.1f}, RSI:{rsi:.1f})", "cyan"))
                return True
            if side == "short" and (rsi <= 45 and minus_di > plus_di) and volume_ok and volatility_ok:
                print(colored(f"⚡ FALSE REVERSAL DETECTED: {pattern} in strong downtrend (ADX:{adx:.1f}, RSI:{rsi:.1f})", "cyan"))
                return True
        
        return False
    except Exception as e:
        print(colored(f"⚠️ False reversal detection error: {e}", "yellow"))
        return False

def _is_strong_impulse_enhanced(df, ind, side: str) -> tuple:
    """
    Enhanced impulse detection with strength rating
    Returns: (is_impulse, strength_rating 1-3, reason)
    """
    try:
        atr = float(ind.get("atr") or 0)
        if atr <= 0: 
            return False, 0, "No ATR data"
            
        # ✅ PATCH 4: Always use current closed candle
        idx = -1  # Always use the latest closed candle
        
        o0 = float(df["open"].iloc[idx])
        c0 = float(df["close"].iloc[idx])
        h0 = float(df["high"].iloc[idx])
        l0 = float(df["low"].iloc[idx])
        
        body = abs(c0 - o0)
        total_range = h0 - l0
        body_ratio = body / total_range if total_range > 0 else 0
        
        dir_candle = 1 if c0 > o0 else -1
        impulse_ratio = (body / atr) if atr > 0 else 0
        
        # Strength rating
        if impulse_ratio >= 2.5 and body_ratio >= 0.8:
            strength = 3  # Very strong
        elif impulse_ratio >= 1.8 and body_ratio >= 0.7:
            strength = 2  # Strong
        elif impulse_ratio >= 1.2 and body_ratio >= 0.6:
            strength = 1  # Moderate
        else:
            return False, 0, f"Weak impulse: {impulse_ratio:.2f} ATR"
        
        # Direction confirmation
        if side == "long" and dir_candle > 0:
            return True, strength, f"Bullish impulse x{impulse_ratio:.2f} ATR (strength:{strength})"
        if side == "short" and dir_candle < 0:
            return True, strength, f"Bearish impulse x{impulse_ratio:.2f} ATR (strength:{strength})"
            
        return False, 0, f"Impulse in wrong direction: {impulse_ratio:.2f} ATR"
        
    except Exception as e:
        return False, 0, f"Impulse detection error: {e}"

def _should_early_exit(df, ind, side: str) -> tuple:
    """
    Check if we should exit early due to strong counter-trend signals
    Returns: (should_exit, exit_reason, exit_urgency 1-3)
    """
    try:
        patt = detect_candle_pattern(df)
        pattern = patt.get("pattern", "NONE")
        adx = float(ind.get("adx") or 0)
        rsi = float(ind.get("rsi") or 50)
        plus_di = float(ind.get("plus_di") or 0)
        minus_di = float(ind.get("minus_di") or 0)
        
        # Strong reversal patterns
        strong_reversal_patterns = {
            "EVENING_STAR", "MORNING_STAR", "ENGULF_BEAR", "ENGULF_BULL",
            "THREE_BLACK_CROWS", "THREE_WHITE_SOLDIERS"
        }
        
        # Urgency levels based on multiple confirmations
        urgency = 0
        reasons = []
        
        # Pattern-based urgency
        if pattern in strong_reversal_patterns:
            urgency += 2
            reasons.append(f"Strong reversal pattern: {patt.get('name_en')}")
        
        # RSI extreme urgency
        if side == "long" and rsi >= 80:
            urgency += 2
            reasons.append(f"RSI extreme overbought: {rsi:.1f}")
        elif side == "short" and rsi <= 20:
            urgency += 2
            reasons.append(f"RSI extreme oversold: {rsi:.1f}")
        elif side == "long" and rsi >= 70:
            urgency += 1
            reasons.append(f"RSI overbought: {rsi:.1f}")
        elif side == "short" and rsi <= 30:
            urgency += 1
            reasons.append(f"RSI oversold: {rsi:.1f}")
        
        # DI crossover urgency
        if side == "long" and minus_di > plus_di and adx > 20:
            urgency += 1
            reasons.append("-DI crossed above +DI")
        elif side == "short" and plus_di > minus_di and adx > 20:
            urgency += 1
            reasons.append("+DI crossed above -DI")
        
        # ADX weakening urgency
        adx_prev = float(ind.get("adx_prev") or 0)
        if adx_prev > adx + 5 and adx_prev > 25:  # ADX dropping significantly from high level
            urgency += 1
            reasons.append(f"ADX weakening: {adx_prev:.1f}→{adx:.1f}")
        
        should_exit = urgency >= 3  # Need multiple confirmations
        
        return should_exit, " | ".join(reasons), min(urgency, 3)
        
    except Exception as e:
        return False, f"Early exit check error: {e}", 0

def run_advanced_directive(command: str = "EOC_FALSE_REVERSAL"):
    """
    Enhanced directive with multiple position management strategies:
    
    Supported commands:
    - "EOC_FALSE_REVERSAL": End-of-candle false reversal detection & impulse harvesting
    - "EARLY_EXIT_CHECK": Check for strong counter-trend signals
    - "FULL_POSITION_REVIEW": Comprehensive position health check
    """
    
    if not state.get("open"):
        print(colored("ℹ️ directive: no open position — nothing to do", "cyan"))
        return

    side = state.get("side")
    df = fetch_ohlcv()
    ind = compute_indicators(df)
    
    print(colored("🎯 ADVANCED DIRECTIVE ANALYSIS", "blue"))
    print(colored("─" * 80, "blue"))
    
    if command == "EOC_FALSE_REVERSAL":
        _run_eoc_false_reversal(df, ind, side)
    elif command == "EARLY_EXIT_CHECK":
        _run_early_exit_check(df, ind, side)
    elif command == "FULL_POSITION_REVIEW":
        _run_full_position_review(df, ind, side)
    else:
        print(colored(f"⚠️ Unsupported directive: {command}", "yellow"))
    
    print(colored("─" * 80, "blue"))

def _run_eoc_false_reversal(df, ind, side):
    """Execute EOC false reversal detection and impulse harvesting"""
    
    # 1) False reversal protection
    if _is_false_reversal_enhanced(df, ind, side):
        print(colored("🛡️ FALSE REVERSAL PROTECTION — maintaining position", "green"))
        return

    # 2) Early exit check (override if very strong signals)
    should_exit, exit_reason, urgency = _should_early_exit(df, ind, side)
    if should_exit:
        if urgency >= 3:
            close_fraction = min(0.7, 0.3 + (urgency - 2) * 0.2)  # 30-70% based on urgency
            close_partial(close_fraction, f"URGENT EXIT: {exit_reason}")
            print(colored(f"🚨 URGENT EXIT: Closing {close_fraction*100:.0f}% - {exit_reason}", "red"))
        else:
            close_partial(0.2, f"Early warning: {exit_reason}")
            print(colored(f"⚠️ EARLY WARNING: Partial close 20% - {exit_reason}", "yellow"))
        return

    # 3) Impulse harvesting
    is_impulse, strength, impulse_reason = _is_strong_impulse_enhanced(df, ind, side)
    if is_impulse:
        # Dynamic profit taking based on impulse strength
        if strength == 3:
            close_fraction = 0.5  # 50% for very strong impulse
            action = "MAJOR harvest"
        elif strength == 2:
            close_fraction = 0.35  # 35% for strong impulse
            action = "STRONG harvest"
        else:
            close_fraction = 0.25  # 25% for moderate impulse
            action = "MODERATE harvest"
            
        close_partial(close_fraction, f"{action}: {impulse_reason}")
        print(colored(f"🎯 IMPULSE HARVEST: Closing {close_fraction*100:.0f}% - {impulse_reason}", "green"))
        return

    print(colored("ℹ️ No actionable EOC signals detected", "cyan"))

def _run_early_exit_check(df, ind, side):
    """Check for early exit conditions"""
    should_exit, exit_reason, urgency = _should_early_exit(df, ind, side)
    
    if should_exit:
        print(colored(f"🚨 EARLY EXIT SIGNAL (Urgency: {urgency}/3)", "red"))
        print(colored(f"   Reason: {exit_reason}", "red"))
        
        if urgency >= 3:
            recommendation = "Consider closing 50-70% of position"
        elif urgency >= 2:
            recommendation = "Consider closing 30-50% of position" 
        else:
            recommendation = "Consider closing 20-30% of position"
            
        print(colored(f"   Recommendation: {recommendation}", "yellow"))
    else:
        print(colored("✅ No strong early exit signals", "green"))

def _run_full_position_review(df, ind, side):
    """Comprehensive position health analysis"""
    print(colored("📊 POSITION HEALTH ANALYSIS", "cyan"))
    
    # Current metrics
    px = ind.get("price") or price_now() or state["entry"]
    entry = state["entry"]
    rr = (px - entry) / entry * 100.0 * (1 if side == "long" else -1)
    adx = ind.get("adx", 0)
    rsi = ind.get("rsi", 50)
    
    print(f"   PnL: {rr:.2f}% | ADX: {adx:.1f} | RSI: {rsi:.1f}")
    print(f"   Scale-ins: {state.get('scale_ins', 0)} | Scale-outs: {state.get('scale_outs', 0)}")
    
    # Health assessment
    if adx >= 30 and ((side == "long" and rsi >= 55) or (side == "short" and rsi <= 45)):
        health = "EXCELLENT"
        color = "green"
        recommendation = "Consider scaling in if pattern confirms"
    elif adx >= 20 and ((side == "long" and rsi >= 50) or (side == "short" and rsi <= 50)):
        health = "GOOD" 
        color = "cyan"
        recommendation = "Maintain position, watch for impulses"
    else:
        health = "CAUTION"
        color = "yellow"
        recommendation = "Consider scaling out, trend may be weakening"
    
    print(colored(f"   Health: {health}", color))
    print(colored(f"   Recommendation: {recommendation}", color))
    
    # Pattern analysis
    candle_info = detect_candle_pattern(df)
    print(f"   Current Candle: {candle_info.get('name_en')} (Strength: {candle_info.get('strength', 0)}/4)")

# Integration with main bot - add this to your main loop
def integrate_directives_with_main_loop():
    """
    Call this function from your main trading loop to integrate directives
    """
    try:
        # Run EOC check at appropriate times (e.g., near candle close)
        time_left = time_to_candle_close(fetch_ohlcv(), USE_TV_BAR)
        
        # Run directives in last 30 seconds of candle or every 10th cycle
        if time_left <= 30 or int(time.time()) % 600 == 0:  # Last 30s or every 10min
            run_advanced_directive("EOC_FALSE_REVERSAL")
            
        # Run full review less frequently
        if int(time.time()) % 1800 == 0:  # Every 30 minutes
            run_advanced_directive("FULL_POSITION_REVIEW")
            
    except Exception as e:
        print(colored(f"⚠️ Directive integration error: {e}", "yellow"))
