# -*- coding: utf-8 -*-
"""
RF Futures Bot — Trend Pro++ (BingX Perp, CCXT) — TRAP MASTER EDITION
• نظام متكامل لتحليل الفخاخ والقمم/القيعان الحقيقية
• قراءة ذكية للشموع - أحجام - زخم - فجوات سعرية
• دخول مبكر على RF حي + إدارة محترفة للصفقات
"""

import os, time, math, threading, requests, traceback, random, signal, sys, logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, InvalidOperation
import pandas as pd
import numpy as np

try: import ccxt
except Exception: ccxt=None

def colored(t,*a,**k):
    try:
        from termcolor import colored as _c
        return _c(t,*a,**k)
    except Exception:
        return t

# =================== ENV / MODE ===================
API_KEY    = os.getenv("BINGX_API_KEY", "")
API_SECRET = os.getenv("BINGX_API_SECRET", "")
MODE_LIVE  = bool(API_KEY and API_SECRET and ccxt is not None)
SELF_URL   = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT       = int(os.getenv("PORT", 5000))

# =================== SETTINGS ===================
SYMBOL   = "DOGE/USDT:USDT"
INTERVAL = "15m"

LEVERAGE   = 10
RISK_ALLOC = 0.60
BINGX_POSITION_MODE = "oneway"

# RF live entry
RF_SOURCE   = "close"
RF_PERIOD   = 20
RF_MULT     = 3.5
USE_RF_LIVE = True

# =================== TRAP & MOMENTUM DETECTION SETTINGS ===================
# إعدادات تحليل الفخاخ والزخم
TRAP_CONFIRMATION_BARS = 2                # عدد الشموع للتأكد من الفخ
MIN_VOLUME_SPIKE = 1.8                    # نسبة ارتفاع الحجم للتأكيد
WEAK_MOMENTUM_THRESHOLD = 0.3             # نسبة زخم ضعيف (للفخاخ)
STRONG_MOMENTUM_THRESHOLD = 0.7           # نسبة زخم قوي (للاتجاهات الحقيقية)

# إعدادات مناطق التشبع
OVERBOUGHT_RSI = 72
OVERSOLD_RSI = 28
RSI_DIVERGENCE_LOOKBACK = 10

# إعدادات الانفجارات
EXPLOSION_VOLUME_RATIO = 2.0
EXPLOSION_BODY_RATIO = 0.75
EXPLOSION_ATR_RATIO = 2.2

# إعدادات الفجوات السعرية
FVG_GAP_RATIO = 1.5                       # نسبة الفجوة إلى ATR
FVG_LOOKBACK = 5

# =================== ADVANCED CANDLE ANALYSIS ===================
def advanced_candle_analysis(df, atr):
    """تحليل متقدم للشموع - الأجسام، الذيول، الأحجام، الأنماط"""
    if len(df) < 3: return {}
    
    current = df.iloc[-1]
    prev1 = df.iloc[-2] if len(df) >= 2 else current
    prev2 = df.iloc[-3] if len(df) >= 3 else prev1
    
    o, h, l, c, v = (
        float(current['open']), float(current['high']), 
        float(current['low']), float(current['close']), 
        float(current['volume'])
    )
    
    # الأساسيات
    body = abs(c - o)
    range_val = h - l
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    
    # النسب
    body_ratio = body / range_val if range_val > 0 else 0
    upper_wick_ratio = upper_wick / range_val if range_val > 0 else 0
    lower_wick_ratio = lower_wick / range_val if range_val > 0 else 0
    
    # تحليل الحجم
    avg_volume = df['volume'].astype(float).tail(20).mean()
    volume_ratio = v / avg_volume if avg_volume > 0 else 1
    
    # اكتشاف الأنماط
    patterns = {
        'hammer': lower_wick_ratio > 0.6 and body_ratio < 0.3 and c > o,
        'shooting_star': upper_wick_ratio > 0.6 and body_ratio < 0.3 and c < o,
        'bullish_engulfing': c > o and prev1['close'] < prev1['open'] and o < prev1['close'] and c > prev1['open'],
        'bearish_engulfing': c < o and prev1['close'] > prev1['open'] and o > prev1['close'] and c < prev1['open'],
        'doji': body_ratio < 0.1,
        'marubozu': body_ratio > 0.9
    }
    
    # قوة الشمعة
    candle_strength = 0.0
    if patterns['marubozu']: candle_strength = 1.0
    elif patterns['bullish_engulfing'] or patterns['bearish_engulfing']: candle_strength = 0.8
    elif patterns['hammer'] or patterns['shooting_star']: candle_strength = 0.6
    
    return {
        'body': body, 'range': range_val, 'upper_wick': upper_wick, 'lower_wick': lower_wick,
        'body_ratio': body_ratio, 'upper_wick_ratio': upper_wick_ratio, 'lower_wick_ratio': lower_wick_ratio,
        'volume_ratio': volume_ratio, 'patterns': patterns, 'candle_strength': candle_strength,
        'is_bullish': c > o, 'is_bearish': c < o
    }

def detect_explosion_candle(candle_analysis, atr, volume_ratio):
    """اكتشاف شمعة الانفجار/الانهيار"""
    if candle_analysis['body_ratio'] >= EXPLOSION_BODY_RATIO:
        if candle_analysis['range'] >= atr * EXPLOSION_ATR_RATIO:
            if volume_ratio >= EXPLOSION_VOLUME_RATIO:
                return True, "STRONG_EXPLOSION"
            elif volume_ratio >= EXPLOSION_VOLUME_RATIO * 0.7:
                return True, "MODERATE_EXPLOSION"
    return False, "NO_EXPLOSION"

# =================== TRAP DETECTION SYSTEM ===================
class TrapDetectionSystem:
    """نظام متكامل لاكتشاف الفخاخ والاختراقات الكاذبة"""
    
    @staticmethod
    def detect_liquidity_trap(df, level, side, indicators):
        """اكتشاف فخ السيولة عند مستويات رئيسية"""
        if len(df) < 3: return False, 0.0
        
        current = df.iloc[-1]
        prev1 = df.iloc[-2]
        
        current_high = float(current['high'])
        current_low = float(current['low'])
        current_close = float(current['close'])
        prev_high = float(prev1['high'])
        prev_low = float(prev1['low'])
        
        trap_probability = 0.0
        trap_conditions = []
        
        if side == "bullish_trap":  # فخ صاعد - اختراق كاذب لأعلى
            # الشرط 1: اختراق المستوى ثم عودة
            if current_high > level and current_close < level:
                trap_conditions.append("false_breakout")
                trap_probability += 0.4
                
            # الشرط 2: ذيل علوي طويل
            candle_analysis = advanced_candle_analysis(df, indicators.get('atr', 0))
            if candle_analysis['upper_wick_ratio'] > 0.6:
                trap_conditions.append("long_upper_wick")
                trap_probability += 0.3
                
            # الشرط 3: حجم تداول منخفض عند الاختراق
            volume_avg = df['volume'].astype(float).tail(10).mean()
            if float(current['volume']) < volume_avg * 0.8:
                trap_conditions.append("low_volume")
                trap_probability += 0.2
                
            # الشرط 4: زخم ضعيف (ADX منخفض)
            if indicators.get('adx', 0) < 25:
                trap_conditions.append("weak_momentum")
                trap_probability += 0.1
        
        elif side == "bearish_trap":  # فخ هابط - اختراق كاذب لأسفل
            # الشرط 1: اختراق المستوى ثم عودة
            if current_low < level and current_close > level:
                trap_conditions.append("false_breakout")
                trap_probability += 0.4
                
            # الشرط 2: ذيل سفلي طويل
            candle_analysis = advanced_candle_analysis(df, indicators.get('atr', 0))
            if candle_analysis['lower_wick_ratio'] > 0.6:
                trap_conditions.append("long_lower_wick")
                trap_probability += 0.3
                
            # الشرط 3: حجم تداول منخفض عند الاختراق
            volume_avg = df['volume'].astype(float).tail(10).mean()
            if float(current['volume']) < volume_avg * 0.8:
                trap_conditions.append("low_volume")
                trap_probability += 0.2
                
            # الشرط 4: زخم ضعيف (ADX منخفض)
            if indicators.get('adx', 0) < 25:
                trap_conditions.append("weak_momentum")
                trap_probability += 0.1
        
        is_trap = trap_probability >= 0.6
        return is_trap, trap_probability

    @staticmethod
    def detect_momentum_divergence(df, indicators, lookback=RSI_DIVERGENCE_LOOKBACK):
        """اكتشاف الاختلافات بين السعر والمؤشرات (تصحيح/انعكاس محتمل)"""
        if len(df) < lookback + 1: return None
        
        prices = df['close'].astype(float).tail(lookback)
        rsi_values = []
        
        # حساب RSI يدوي للفترة
        for i in range(len(df) - lookback, len(df)):
            slice_df = df.iloc[:i+1]
            if len(slice_df) >= RSI_DIVERGENCE_LOOKBACK + 1:
                ind = compute_indicators(slice_df)
                rsi_values.append(ind.get('rsi', 50))
        
        if len(rsi_values) < lookback: return None
        
        # اكتشاف الاختلافات
        price_highs = []
        rsi_highs = []
        
        for i in range(2, len(prices)-2):
            if (prices.iloc[i] > prices.iloc[i-1] and 
                prices.iloc[i] > prices.iloc[i-2] and
                prices.iloc[i] > prices.iloc[i+1] and
                prices.iloc[i] > prices.iloc[i+2]):
                price_highs.append((i, prices.iloc[i]))
                
            if (rsi_values[i] > rsi_values[i-1] and 
                rsi_values[i] > rsi_values[i-2] and
                rsi_values[i] > rsi_values[i+1] and
                rsi_values[i] > rsi_values[i+2]):
                rsi_highs.append((i, rsi_values[i]))
        
        # اختلاف هابط (السعر يصنع قمة أعلى لكن RSI يصنع قمة أدنى)
        bearish_divergence = False
        if len(price_highs) >= 2 and len(rsi_highs) >= 2:
            if (price_highs[-1][1] > price_highs[-2][1] and 
                rsi_highs[-1][1] < rsi_highs[-2][1]):
                bearish_divergence = True
        
        # اختلاف صاعد (السعر يصنع قاع أدنى لكن RSI يصنع قاع أعلى)
        bullish_divergence = False
        price_lows = []
        rsi_lows = []
        
        for i in range(2, len(prices)-2):
            if (prices.iloc[i] < prices.iloc[i-1] and 
                prices.iloc[i] < prices.iloc[i-2] and
                prices.iloc[i] < prices.iloc[i+1] and
                prices.iloc[i] < prices.iloc[i+2]):
                price_lows.append((i, prices.iloc[i]))
                
            if (rsi_values[i] < rsi_values[i-1] and 
                rsi_values[i] < rsi_values[i-2] and
                rsi_values[i] < rsi_values[i+1] and
                rsi_values[i] < rsi_values[i+2]):
                rsi_lows.append((i, rsi_values[i]))
        
        if len(price_lows) >= 2 and len(rsi_lows) >= 2:
            if (price_lows[-1][1] < price_lows[-2][1] and 
                rsi_lows[-1][1] > rsi_lows[-2][1]):
                bullish_divergence = True
        
        return {
            'bearish_divergence': bearish_divergence,
            'bullish_divergence': bullish_divergence
        }

# =================== PRICE LEVEL ANALYSIS ===================
def analyze_key_levels(df, atr):
    """تحليل المستويات الرئيسية: قمم، قيعان، مناطق دعم ومقاومة"""
    if len(df) < 20: return {}
    
    highs = df['high'].astype(float)
    lows = df['low'].astype(float)
    closes = df['close'].astype(float)
    
    # القمم والقيعان المحلية
    local_highs = []
    local_lows = []
    
    for i in range(2, len(df)-2):
        if (highs.iloc[i] > highs.iloc[i-1] and 
            highs.iloc[i] > highs.iloc[i-2] and
            highs.iloc[i] > highs.iloc[i+1] and
            highs.iloc[i] > highs.iloc[i+2]):
            local_highs.append(highs.iloc[i])
            
        if (lows.iloc[i] < lows.iloc[i-1] and 
            lows.iloc[i] < lows.iloc[i-2] and
            lows.iloc[i] < lows.iloc[i+1] and
            lows.iloc[i] < lows.iloc[i+2]):
            local_lows.append(lows.iloc[i])
    
    # المتوسطات المتحركة كمستويات ديناميكية
    ma20 = closes.rolling(20).mean().iloc[-1] if len(closes) >= 20 else closes.iloc[-1]
    ma50 = closes.rolling(50).mean().iloc[-1] if len(closes) >= 50 else closes.iloc[-1]
    
    # أقرب مستويات دعم ومقاومة
    current_price = closes.iloc[-1]
    resistance_levels = [h for h in local_highs if h > current_price]
    support_levels = [l for l in local_lows if l < current_price]
    
    nearest_resistance = min(resistance_levels) if resistance_levels else current_price * 1.05
    nearest_support = max(support_levels) if support_levels else current_price * 0.95
    
    return {
        'local_highs': local_highs[-5:],  # آخر 5 قمم
        'local_lows': local_lows[-5:],    # آخر 5 قيعان
        'ma20': ma20,
        'ma50': ma50,
        'nearest_resistance': nearest_resistance,
        'nearest_support': nearest_support,
        'distance_to_resistance': (nearest_resistance - current_price) / current_price * 100,
        'distance_to_support': (current_price - nearest_support) / current_price * 100
    }

# =================== MOMENTUM ANALYSIS ===================
def analyze_momentum(df, indicators):
    """تحليل الزخم والقوة الشرائية/البيعية"""
    if len(df) < 3: return {}
    
    current = df.iloc[-1]
    prev1 = df.iloc[-2]
    
    current_close = float(current['close'])
    prev_close = float(prev1['close'])
    price_change = (current_close - prev_close) / prev_close * 100
    
    # قوة الزخم بناءً على ADX وDI
    adx = indicators.get('adx', 0)
    plus_di = indicators.get('plus_di', 0)
    minus_di = indicators.get('minus_di', 0)
    
    momentum_strength = 0.0
    if adx > 25:
        if plus_di > minus_di:
            momentum_strength = min(1.0, (plus_di - minus_di) / 50)
        else:
            momentum_strength = -min(1.0, (minus_di - plus_di) / 50)
    
    # تحليل الحجم
    volume_trend = "NEUTRAL"
    current_volume = float(current['volume'])
    avg_volume = df['volume'].astype(float).tail(10).mean()
    
    if current_volume > avg_volume * 1.5:
        volume_trend = "STRONG"
    elif current_volume > avg_volume * 1.2:
        volume_trend = "MODERATE"
    elif current_volume < avg_volume * 0.8:
        volume_trend = "WEAK"
    
    # تحديد التشبع الشرائي/البيعي
    rsi = indicators.get('rsi', 50)
    overbought = rsi > OVERBOUGHT_RSI
    oversold = rsi < OVERSOLD_RSI
    
    return {
        'price_change_pct': price_change,
        'momentum_strength': momentum_strength,
        'volume_trend': volume_trend,
        'overbought': overbought,
        'oversold': oversold,
        'trend_direction': 'BULLISH' if plus_di > minus_di else 'BEARISH',
        'trend_strength': 'STRONG' if adx > 40 else 'MODERATE' if adx > 25 else 'WEAK'
    }

# =================== POST-ENTRY INTELLIGENCE ===================
def post_entry_analysis(df, entry_price, side, indicators):
    """تحليل ذكي بعد فتح الصفقة لاكتشاف المخاطر والفرص"""
    if len(df) < 2: return {}
    
    current_price = float(df['close'].iloc[-1])
    levels = analyze_key_levels(df, indicators.get('atr', 0))
    momentum = analyze_momentum(df, indicators)
    candle_analysis = advanced_candle_analysis(df, indicators.get('atr', 0))
    
    # حساب الربح/الخسارة
    pnl_pct = (current_price - entry_price) / entry_price * 100 * (1 if side == 'long' else -1)
    
    # تحليل المخاطر
    risks = []
    opportunities = []
    
    # المخاطر
    if side == 'long':
        if current_price < entry_price and momentum['momentum_strength'] < -0.3:
            risks.append("زخم هابط قوي بعد الدخول")
        if levels['distance_to_support'] < 1.0:
            risks.append("قرب مستوى دعم حاسم")
        if candle_analysis['patterns']['bearish_engulfing']:
            risks.append("نمط engulfing هابط")
            
    else:  # short
        if current_price > entry_price and momentum['momentum_strength'] > 0.3:
            risks.append("زخم صاعد قوي بعد الدخول")
        if levels['distance_to_resistance'] < 1.0:
            risks.append("قرب مستوى مقاومة حاسم")
        if candle_analysis['patterns']['bullish_engulfing']:
            risks.append("نمط engulfing صاعد")
    
    # الفرص
    if side == 'long' and momentum['momentum_strength'] > 0.5:
        opportunities.append("زخم صاعد قوي يدعم الاتجاه")
    if side == 'short' and momentum['momentum_strength'] < -0.5:
        opportunities.append("زخم هابط قوي يدعم الاتجاه")
    if candle_analysis['volume_ratio'] > 2.0:
        opportunities.append("حجم تداول مرتفع يؤكد الحركة")
    
    # اكتشاف الفجوات السعرية القريبة
    nearby_fvg = detect_nearby_fvg(df, current_price, indicators.get('atr', 0))
    if nearby_fvg:
        if side == 'long' and current_price > nearby_fvg['top']:
            opportunities.append(f"فجوة سعرية داعمة عند {nearby_fvg['top']:.6f}")
        elif side == 'short' and current_price < nearby_fvg['bottom']:
            opportunities.append(f"فجوة سعرية داعمة عند {nearby_fvg['bottom']:.6f}")
    
    return {
        'pnl_pct': pnl_pct,
        'risks': risks,
        'opportunities': opportunities,
        'nearby_resistance': levels['nearest_resistance'],
        'nearby_support': levels['nearest_support'],
        'momentum_score': momentum['momentum_strength']
    }

def detect_nearby_fvg(df, current_price, atr):
    """اكتشاف الفجوات السعرية القريبة من السعر الحالي"""
    if len(df) < 3: return None
    
    for i in range(len(df)-2, max(0, len(df)-FVG_LOOKBACK-1), -1):
        candle1 = df.iloc[i]
        candle2 = df.iloc[i+1]
        
        high1 = float(candle1['high'])
        low1 = float(candle1['low'])
        high2 = float(candle2['high'])
        low2 = float(candle2['low'])
        
        # فجوة صاعدة (Bullish FVG)
        if low2 > high1 and (low2 - high1) <= atr * FVG_GAP_RATIO:
            if abs(current_price - high1) / current_price * 100 < 2.0:  # within 2%
                return {'type': 'BULLISH_FVG', 'bottom': high1, 'top': low2}
        
        # فجوة هابطة (Bearish FVG)
        if high2 < low1 and (low1 - high2) <= atr * FVG_GAP_RATIO:
            if abs(current_price - low1) / current_price * 100 < 2.0:  # within 2%
                return {'type': 'BEARISH_FVG', 'bottom': high2, 'top': low1}
    
    return None

# =================== ENHANCED TRADE LOOP ===================
def enhanced_trade_loop():
    """نظام التداول المحسن بذكاء اكتشاف الفخاخ والانفجارات"""
    loop = 0
    trap_system = TrapDetectionSystem()
    
    while True:
        try:
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            
            if len(df) < 50:
                time.sleep(BASE_SLEEP)
                continue
            
            new_bar = update_bar_counters(df)
            ind = compute_indicators(df)
            prev_ind = compute_indicators(df.iloc[:-1]) if len(df) >= 2 else ind
            rf_live = compute_rf_live(df)
            
            # التحليلات المتقدمة
            candle_analysis = advanced_candle_analysis(df, ind.get('atr', 0))
            key_levels = analyze_key_levels(df, ind.get('atr', 0))
            momentum_analysis = analyze_momentum(df, ind)
            divergence = trap_system.detect_momentum_divergence(df, ind)
            
            # اكتشاف الانفجارات
            explosion_detected, explosion_type = detect_explosion_candle(
                candle_analysis, ind.get('atr', 0), candle_analysis['volume_ratio']
            )
            
            # نظام اكتشاف الفخاخ المتقدم
            trap_analysis = {
                'bull_trap': 0.0,
                'bear_trap': 0.0,
                'divergence': divergence
            }
            
            # فحص فخاخ عند المستويات الرئيسية
            if key_levels['nearest_resistance']:
                is_trap, prob = trap_system.detect_liquidity_trap(
                    df, key_levels['nearest_resistance'], "bullish_trap", ind
                )
                if is_trap:
                    trap_analysis['bull_trap'] = prob
            
            if key_levels['nearest_support']:
                is_trap, prob = trap_system.detect_liquidity_trap(
                    df, key_levels['nearest_support'], "bearish_trap", ind
                )
                if is_trap:
                    trap_analysis['bear_trap'] = prob
            
            # دخول مبكر على إشارات RF الحي
            if not state["open"] and USE_RF_LIVE:
                sig = "buy" if rf_live["buy"] else ("sell" if rf_live["sell"] else None)
                
                if sig:
                    # فحص الفخاخ قبل الدخول
                    trap_prob = trap_analysis['bull_trap'] if sig == 'buy' else trap_analysis['bear_trap']
                    if trap_prob > 0.7:
                        print(colored(f"🚫 تجنب الدخول {sig.upper()} - فخ محتمل (احتمال: {trap_prob:.1%})", "red"))
                    else:
                        qty = compute_size(bal, px or rf_live["price"])
                        if qty > 0:
                            reason = f"RF_LIVE"
                            if explosion_detected:
                                reason += f"+{explosion_type}"
                            open_market(sig, qty, px or rf_live["price"], reason=reason)
            
            # إدارة ذكية بعد فتح الصفقة
            if state["open"]:
                post_analysis = post_entry_analysis(
                    df, state["entry"], state["side"], ind
                )
                
                # تنبيهات المخاطر
                if post_analysis['risks']:
                    for risk in post_analysis['risks']:
                        print(colored(f"⚠️ تنبيه: {risk}", "yellow"))
                
                # استغلال الفرص
                if post_analysis['opportunities']:
                    for opp in post_analysis['opportunities']:
                        print(colored(f"💡 فرصة: {opp}", "green"))
                
                # جني الأرباح من الشموع القوية
                if (candle_analysis['candle_strength'] > 0.7 and 
                    candle_analysis['volume_ratio'] > 1.5):
                    profit_pct = trend_rr_pct(px or rf_live["price"])
                    if profit_pct > 1.0:
                        close_partial(0.3, f"جني أرباح من شمعة قوية (قوة: {candle_analysis['candle_strength']:.1f})")
            
            # اللوج المحسن
            enhanced_snapshot(bal, ind, df, key_levels, momentum_analysis, 
                            trap_analysis, explosion_detected, explosion_type)
            
            if loop % 5 == 0:
                save_state()
            loop += 1
            time.sleep(compute_next_sleep(df))
            
        except Exception as e:
            print(colored(f"❌ خطأ في النظام المحسن: {e}\n{traceback.format_exc()}", "red"))
            time.sleep(BASE_SLEEP)

def enhanced_snapshot(bal, ind, df, levels, momentum, traps, explosion, explosion_type):
    """لوج محسن يعرض كل التحليلات"""
    left = time_to_candle_close(df)
    current_price = float(df['close'].iloc[-1])
    
    print(colored("═" * 120, "cyan"))
    print(colored(f"🎯 نظام التداول الذكي - {SYMBOL} {INTERVAL} - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", "cyan", attrs=['bold']))
    print(colored("─" * 120, "cyan"))
    
    # قسم السعر والمؤشرات
    print(f"💰 السعر: {fmt(current_price)} | ATR: {fmt(ind['atr'])} | الوقت المتبقي: {left}s")
    print(f"📊 المؤشرات: RSI={fmt(ind['rsi'])} ADX={fmt(ind['adx'])} +DI={fmt(ind['plus_di'])} -DI={fmt(ind['minus_di'])}")
    
    # قسم الزخم والاتجاه
    trend_icon = "🟢" if momentum['trend_direction'] == 'BULLISH' else "🔴"
    print(f"🎭 الزخم: {trend_icon} {momentum['trend_direction']} - قوة: {momentum['trend_strength']} - حجم: {momentum['volume_trend']}")
    
    # قسم المستويات
    print(f"📈 المستويات: مقاومة={fmt(levels['nearest_resistance'])} | دعم={fmt(levels['nearest_support'])}")
    print(f"    المسافة: +{fmt(levels['distance_to_resistance'], 2)}% | -{fmt(levels['distance_to_support'], 2)}%")
    
    # قسم الفخاخ والانفجارات
    trap_status = ""
    if traps['bull_trap'] > 0.5 or traps['bear_trap'] > 0.5:
        trap_status = f" | فخاخ: 🐂{traps['bull_trap']:.1%} 🐻{traps['bear_trap']:.1%}"
    
    explosion_status = ""
    if explosion:
        explosion_status = f" | 💥 {explosion_type}"
    
    print(f"🎯 التنبيهات:{trap_status}{explosion_status}")
    
    # قسم الاختلافات
    if traps['divergence']:
        div = traps['divergence']
        if div['bearish_divergence']:
            print("    ⚠️ اختلاف هابط (انعكاس محتمل)")
        if div['bullish_divergence']:
            print("    ⚠️ اختلاف صاعد (انعكاس محتمل)")
    
    # قسم المحفظة
    eff = (bal or 0.0) + compound_pnl
    print(f"💼 الرصيد: {fmt(bal, 2)} | PnL تراكمي: {fmt(compound_pnl)} | حقوق: {fmt(eff)}")
    
    # قسم الصفقة المفتوحة
    if state["open"]:
        side_icon = "🟢 LONG" if state['side'] == 'long' else "🔴 SHORT"
        pnl_pct = trend_rr_pct(current_price)
        print(f"{side_icon} دخول: {fmt(state['entry'])} | كمية: {fmt(state['qty'], 4)} | PnL: {fmt(pnl_pct, 2)}%")
        print(f"   الشموع: {state['bars']} | أعلى ربح: {fmt(state['highest_profit_pct'], 2)}% | أصوات فشل: {state['_fail_votes']}")
    
    print(colored("═" * 120, "cyan"))

# استبدال حلقة التداول الرئيسية
def main():
    print(colored("🚀 بدء نظام التداول الذكي - إصدار اكتشاف الفخاخ والانفجارات", "green", attrs=['bold']))
    print(colored(f"📊 الرمز: {SYMBOL} | الإطار: {INTERVAL} | النمط: {'LIVE' if MODE_LIVE else 'PAPER'}", "yellow"))
    print(colored(f"🎯 المخاطرة: {int(RISK_ALLOC * 100)}% × {LEVERAGE}x | RF_LIVE: {USE_RF_LIVE}", "yellow"))
    
    load_state()
    
    # تشغيل الأنظمة
    threading.Thread(target=enhanced_trade_loop, daemon=True).start()
    threading.Thread(target=keepalive_loop, daemon=True).start()
    
    # تشغيل الخادم
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
