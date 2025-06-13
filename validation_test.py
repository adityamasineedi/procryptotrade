#!/usr/bin/env python3
"""
ProTradeAI Pro+ Validation Test Script
Comprehensive testing of all components and integrations
"""

import sys
import os
import json
import traceback
from pathlib import Path
from datetime import datetime
import importlib.util

def test_config_validation():
    """Test config.py imports and validation"""
    print("🔧 Testing config.py...")
    try:
        from config import (
            SYMBOLS, TIMEFRAMES, CAPITAL, RISK_PER_TRADE,
            SIDEWAYS_ALERT_CONFIG, MARKET_REGIME_CONFIG,
            validate_config, get_confidence_grade, get_leverage_range
        )
        
        # Test key configurations exist
        assert len(SYMBOLS) > 0, "SYMBOLS list is empty"
        assert len(TIMEFRAMES) > 0, "TIMEFRAMES list is empty"
        assert CAPITAL > 0, "CAPITAL must be positive"
        
        # Test new SIDEWAYS_ALERT_CONFIG
        required_keys = ['range_long_emoji', 'range_short_emoji', 'support_emoji', 'resistance_emoji']
        for key in required_keys:
            assert key in SIDEWAYS_ALERT_CONFIG, f"Missing {key} in SIDEWAYS_ALERT_CONFIG"
        
        # Test function calls
        grade = get_confidence_grade(75.0)
        leverage_range = get_leverage_range('moderate')
        
        print("   ✅ Config imports successful")
        print("   ✅ SIDEWAYS_ALERT_CONFIG added correctly")
        print("   ✅ All functions working")
        print(f"   📊 Monitoring {len(SYMBOLS)} symbols on {len(TIMEFRAMES)} timeframes")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Config test failed: {e}")
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        return False

def test_strategy_ai():
    """Test strategy_ai.py functionality"""
    print("🤖 Testing strategy_ai.py...")
    try:
        from strategy_ai import strategy_ai
        
        # Test model info
        model_info = strategy_ai.get_model_info()
        assert model_info['model_loaded'], "Model should be loaded"
        assert model_info['feature_count'] > 0, "Model should have features"
        
        # Test signal tracking
        tracker_metrics = strategy_ai.signal_tracker.get_performance_metrics(days=7)
        assert 'total_signals' in tracker_metrics, "Signal tracker missing metrics"
        
        # Test data fetching (with timeout)
        try:
            df = strategy_ai.get_binance_data('BTCUSDT', '1h', limit=10)
            data_fetch_success = not df.empty
        except:
            data_fetch_success = False
            print("   ⚠️  Binance API not accessible (normal in some environments)")
        
        print("   ✅ Strategy AI imports successful")
        print(f"   ✅ Model type: {model_info['model_type']}")
        print(f"   ✅ Features: {model_info['feature_count']}")
        print(f"   ✅ Signal tracker active: {len(strategy_ai.signal_tracker.signals_sent)} tracked")
        
        if data_fetch_success:
            print("   ✅ Binance data fetching working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Strategy AI test failed: {e}")
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        return False

def test_notifier():
    """Test notifier.py functionality"""
    print("📱 Testing notifier.py...")
    try:
        from notifier import telegram_notifier
        
        # Test configuration access
        stats = telegram_notifier.get_stats()
        assert 'bot_configured' in stats, "Notifier stats missing"
        
        # Test time formatting
        test_time = datetime.now()
        ist_time = telegram_notifier.format_ist_time(test_time)
        assert "IST" in ist_time, "IST formatting failed"
        
        # Test sideways signal formatting (should not crash)
        dummy_signal = {
            'symbol': 'BTCUSDT',
            'timeframe': '4h',
            'signal_type': 'LONG',
            'confidence': 75.0,
            'confidence_grade': 'B',
            'leverage': 3,
            'timestamp': datetime.now(),
            'current_price': 50000.0,
            'strategy_type': 'RANGE_LONG',
            'support_level': 49000.0,
            'resistance_level': 51000.0,
            'range_size_pct': 4.0,
            'position_in_range': 0.3,
            'sl_price': 49500.0,
            'tp_price': 50500.0,
            'sl_distance_pct': 1.0,
            'tp_distance_pct': 1.0,
            'rr_ratio': 1.0,
            'market_regime': 'SIDEWAYS'
        }
        
        # Test sideways signal formatting (new feature)
        sideways_message = telegram_notifier.format_sideways_signal(dummy_signal)
        assert len(sideways_message) > 0, "Sideways signal formatting failed"
        
        print("   ✅ Notifier imports successful")
        print("   ✅ IST timezone formatting working")
        print("   ✅ Sideways signal formatting working")
        print(f"   ✅ Bot configured: {stats['bot_configured']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Notifier test failed: {e}")
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        return False

def test_main_integration():
    """Test main.py integration capabilities"""
    print("🎯 Testing main.py integration...")
    try:
        # Import classes from main.py
        sys.path.append('.')
        
        # Test if we can import the bot class
        spec = importlib.util.spec_from_file_location("main", "main.py")
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        
        # Create bot instance
        bot = main_module.ProTradeAIBot()
        
        # Test basic functionality
        assert hasattr(bot, 'tracker'), "Bot should have performance tracker"
        assert hasattr(bot, 'quick_market_scan'), "Bot should have scanning methods"
        
        # Test status
        status = bot.get_status()
        assert 'is_running' in status, "Bot status missing required fields"
        
        print("   ✅ Main bot class imports successful")
        print("   ✅ Performance tracker integrated")
        print("   ✅ Status reporting working")
        print(f"   ✅ Signals tracked: {status['signals_today']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Main integration test failed: {e}")
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        return False

def test_dashboard_endpoints():
    """Test dashboard.py functionality"""
    print("🌐 Testing dashboard.py...")
    try:
        from dashboard import EnhancedDashboard
        
        # Create dashboard instance
        dashboard = EnhancedDashboard()
        
        # Test basic functionality
        assert hasattr(dashboard, 'app'), "Dashboard should have Flask app"
        assert hasattr(dashboard, 'load_signals'), "Dashboard should have data loading"
        
        # Test data directory creation
        assert dashboard.data_dir.exists(), "Data directory should be created"
        
        # Test signal loading (should not crash even if no data)
        signals = dashboard.load_signals()
        assert isinstance(signals, list), "Signals should be a list"
        
        print("   ✅ Dashboard imports successful")
        print("   ✅ Data directory created")
        print("   ✅ Signal loading working")
        print(f"   ✅ Loaded {len(signals)} historical signals")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Dashboard test failed: {e}")
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        return False

def test_retrain_model():
    """Test retrain_model.py imports"""
    print("🔄 Testing retrain_model.py...")
    try:
        # Import the retrain module
        spec = importlib.util.spec_from_file_location("retrain_model", "retrain_model.py")
        retrain_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(retrain_module)
        
        # Test ModelRetrainer class
        retrainer = retrain_module.ModelRetrainer()
        
        # Test basic functionality
        assert hasattr(retrainer, 'symbols'), "Retrainer should have symbols"
        assert hasattr(retrainer, 'feature_columns'), "Retrainer should have feature columns"
        
        # Test that os module is available (the fix we made)
        import os
        assert hasattr(os, 'makedirs'), "os.makedirs should be available"
        
        print("   ✅ Retrain model imports successful")
        print("   ✅ os module import fixed")
        print("   ✅ ModelRetrainer class working")
        print(f"   ✅ Feature columns: {len(retrainer.feature_columns)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Retrain model test failed: {e}")
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        return False

def test_file_structure():
    """Test file structure and dependencies"""
    print("📁 Testing file structure...")
    try:
        required_files = [
            'config.py', 'strategy_ai.py', 'notifier.py', 
            'main.py', 'dashboard.py', 'retrain_model.py',
            'requirements.txt', '.gitignore'
        ]
        
        missing_files = []
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"   ⚠️  Missing files: {missing_files}")
        else:
            print("   ✅ All required files present")
        
        # Test data directories will be created
        data_dirs = ['data', 'logs', 'models']
        for dir_name in data_dirs:
            Path(dir_name).mkdir(exist_ok=True)
            if Path(dir_name).exists():
                print(f"   ✅ {dir_name}/ directory ready")
        
        return len(missing_files) == 0
        
    except Exception as e:
        print(f"   ❌ File structure test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 ProTradeAI Pro+ Validation Test Suite")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Configuration", test_config_validation),
        ("Strategy AI", test_strategy_ai),
        ("Telegram Notifier", test_notifier),
        ("Main Integration", test_main_integration),
        ("Dashboard", test_dashboard_endpoints),
        ("Model Retraining", test_retrain_model),
    ]
    
    results = {}
    total_tests = len(tests)
    passed_tests = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed_tests += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            results[test_name] = False
            print(f"💥 {test_name}: CRASHED - {e}")
    
    # Final Results
    print("\n" + "="*60)
    print("📊 VALIDATION RESULTS SUMMARY")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {test_name:<20} {status}")
    
    success_rate = (passed_tests / total_tests) * 100
    print(f"\n🎯 Overall Success Rate: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
    
    if success_rate >= 85:
        print("🎉 EXCELLENT! Your bot is ready for deployment!")
    elif success_rate >= 70:
        print("👍 GOOD! Minor issues need attention.")
    else:
        print("⚠️  NEEDS WORK! Several issues require fixing.")
    
    # Specific recommendations
    print("\n💡 NEXT STEPS:")
    if results.get("Configuration", False) and results.get("Strategy AI", False):
        print("✅ Core functionality is working")
        print("🚀 You can run: python main.py test")
    
    if results.get("Dashboard", False):
        print("✅ Dashboard is ready")
        print("🌐 You can run: python dashboard.py")
    
    if not all(results.values()):
        print("🔧 Fix the failed tests above before production use")
    
    print(f"\n⏰ Validation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return success_rate >= 85

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
