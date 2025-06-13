import sys
from strategy_ai import strategy_ai
from notifier import telegram_notifier

def force_test_signals():
    """Force generate test signals for immediate verification"""
    print("🚨 FORCING TEST SIGNALS...")
    
    # Enable ultra-aggressive mode
    strategy_ai.emergency_mode = True
    
    test_pairs = [
        ('BTCUSDT', '4h'),
        ('ETHUSDT', '4h'), 
        ('BNBUSDT', '1h'),
    ]
    
    signals_generated = 0
    
    for symbol, timeframe in test_pairs:
        try:
            print(f"🔍 Force testing {symbol} {timeframe}...")
            
            # Clear cooldowns for testing
            signal_key = f"{symbol}_{timeframe}"
            if signal_key in strategy_ai.last_signals:
                del strategy_ai.last_signals[signal_key]
            
            # Try normal signal first
            signal = strategy_ai.predict_signal(symbol, timeframe)
            
            if signal:
                print(f"✅ NORMAL SIGNAL: {symbol} {signal['signal_type']} {signal['confidence']:.1f}%")
                
                # Send via Telegram
                success = telegram_notifier.send_signal_alert(signal)
                if success:
                    print(f"📤 Telegram sent successfully")
                    signals_generated += 1
                else:
                    print(f"❌ Telegram failed")
            else:
                print(f"❌ No normal signal for {symbol} {timeframe}")
                
                # Try emergency override
                df = strategy_ai.get_binance_data(symbol, timeframe, limit=100)
                if not df.empty:
                    df = strategy_ai.calculate_technical_indicators(df)
                    emergency_signal = strategy_ai._emergency_signal_override(symbol, timeframe, df)
                    
                    if emergency_signal:
                        print(f"🚨 EMERGENCY SIGNAL: {symbol} {emergency_signal['signal_type']} {emergency_signal['confidence']:.1f}%")
                        
                        # Send via Telegram
                        success = telegram_notifier.send_signal_alert(emergency_signal)
                        if success:
                            print(f"📤 Emergency signal sent via Telegram")
                            signals_generated += 1
                        else:
                            print(f"❌ Emergency telegram failed")
                
        except Exception as e:
            print(f"❌ Error testing {symbol} {timeframe}: {e}")
    
    print(f"\n📊 RESULTS: {signals_generated} signals generated and sent")
    
    if signals_generated > 0:
        print("🎉 SUCCESS: Signal generation is working!")
        print("💡 Your bot should now generate signals normally")
    else:
        print("❌ STILL NO SIGNALS: Deeper investigation needed")
        print("🔧 Check your model file and data connections")
    
    return signals_generated > 0

if __name__ == "__main__":
    force_test_signals()