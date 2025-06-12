#!/usr/bin/env python3
"""
Quick test script to debug signal generation
Save this as: test_signals.py
Run with: python test_signals.py
"""

import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_signal_generation():
    """Test signal generation system"""
    try:
        from strategy_ai import strategy_ai
        from config import CONFIDENCE_THRESHOLDS, MARKET_CONDITIONS
        
        print("🧪 ProTradeAI Pro+ Signal Generation Test")
        print("=" * 50)
        
        # Test 1: Model status
        print("🤖 Testing AI Model...")
        model_info = strategy_ai.get_model_info()
        print(f"  ✅ Model type: {model_info['model_type']}")
        print(f"  ✅ Features: {model_info['feature_count']}")
        
        # Test 2: Market conditions
        print("\n📊 Testing Market Conditions...")
        volatility = strategy_ai.get_market_volatility()
        print(f"  📈 Market volatility: {volatility:.4f}")
        print(f"  🎯 Threshold: {MARKET_CONDITIONS['low_volatility_threshold']}")
        
        if volatility < MARKET_CONDITIONS['low_volatility_threshold']:
            print("  ⚠️  LOW VOLATILITY DETECTED - This explains fewer signals!")
        else:
            print("  ✅ Normal volatility")
        
        # Test 3: Configuration
        print("\n⚙️ Testing Configuration...")
        print(f"  🎯 Min confidence: {CONFIDENCE_THRESHOLDS['MIN_SIGNAL']}%")
        print(f"  📊 Confidence adjustment: {MARKET_CONDITIONS['confidence_adjustment']}")
        
        # Test 4: Data fetching
        print("\n📡 Testing Data Fetching...")
        test_symbols = ['BTCUSDT', 'ETHUSDT']
        
        for symbol in test_symbols:
            try:
                df = strategy_ai.get_binance_data(symbol, '4h', limit=50)
                if df.empty:
                    print(f"  ❌ {symbol}: No data fetched")
                else:
                    print(f"  ✅ {symbol}: {len(df)} candles fetched")
            except Exception as e:
                print(f"  ❌ {symbol}: API Error - {e}")
        
        # Test 5: Signal generation
        print("\n🎯 Testing Signal Generation...")
        signals_generated = 0
        
        for symbol in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']:
            for timeframe in ['4h', '1h']:
                try:
                    signal = strategy_ai.predict_signal(symbol, timeframe)
                    if signal:
                        print(f"  ✅ {symbol} {timeframe}: {signal['signal_type']} ({signal['confidence']:.1f}%)")
                        signals_generated += 1
                    else:
                        print(f"  📭 {symbol} {timeframe}: No signal")
                except Exception as e:
                    print(f"  ❌ {symbol} {timeframe}: Error - {e}")
        
        # Summary
        print(f"\n📋 Test Summary:")
        print(f"  🎯 Signals generated: {signals_generated}")
        print(f"  📊 Market volatility: {volatility:.4f} ({'LOW' if volatility < 0.03 else 'NORMAL'})")
        
        if signals_generated == 0:
            print("\n🔍 Possible reasons for no signals:")
            print("  1. Low market volatility (most likely)")
            print("  2. All predictions are HOLD")
            print("  3. Confidence below threshold")
            print("  4. Recent signals still in cooldown")
            print(f"  5. Model needs retraining")
            
            print("\n💡 Recommended actions:")
            print("  1. Lower confidence threshold temporarily")
            print("  2. Retrain model with recent data")
            print("  3. Wait for market volatility to increase")
        
        return signals_generated > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_signal_generation()
    sys.exit(0 if success else 1)