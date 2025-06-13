"""
ProTradeAI Pro+ Configuration - COMPLETELY FIXED VERSION
Removed duplicates, conflicts, and optimized for reliable operation

KEY FIXES:
- Removed duplicate/conflicting settings
- Lowered thresholds for more signals
- Simplified configuration structure
- Production-ready values
- Fixed validation logic
"""

import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CORE TRADING SETTINGS (OPTIMIZED FOR SIGNAL GENERATION)
# ============================================================================

# Trading symbols (top cryptocurrencies)
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
    'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'MATICUSDT'
]

# Trading timeframes (priority order)
TIMEFRAMES = ['1h', '4h', '1d']

# Timeframe priority for signal selection
TIMEFRAME_PRIORITY = {
    '1d': 100,  # Highest priority
    '4h': 80,
    '1h': 60,
    '15m': 40,
    '5m': 20    # Lowest priority
}

# 🔧 FIXED: Trading capital and risk settings (PRODUCTION VALUES)
CAPITAL = float(os.getenv('TRADING_CAPITAL', '1000'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))  # 2.0% per trade
MAX_DAILY_TRADES = int(os.getenv('MAX_DAILY_TRADES', '15'))   # 15 trades per day

# ============================================================================
# 🔧 LOWERED CONFIDENCE THRESHOLDS (KEY FIX FOR MORE SIGNALS)
# ============================================================================

CONFIDENCE_THRESHOLDS = {
    'MIN_SIGNAL': 25,               # LOWERED from 45 to 25
    'HIGH_CONFIDENCE': 40,          # LOWERED from 65 to 40
    'MAX_CONFIDENCE': 90,           
    'RANGE_TRADING_MIN': 30,        # LOWERED from 40 to 30
    'TREND_TRADING_MIN': 35,        # LOWERED from 50 to 35
    'EMERGENCY_MODE': 15,           # LOWERED from 25 to 15
}

# ============================================================================
# SCHEDULER CONFIGURATION (OPTIMIZED)
# ============================================================================

SCHEDULER_CONFIG = {
    # Scanning intervals
    'quick_scan_interval': 5,       # Minutes - Quick scan every 5 minutes
    'full_scan_interval': 15,       # Minutes - Full scan every 15 minutes
    'health_check_interval': 60,    # Minutes - Health check every hour (reduced)
    'shutdown_check_interval': 20,  # Minutes - Check shutdown every 20 minutes
    'cleanup_interval': 240,        # Minutes - Cleanup every 4 hours
    
    # Auto shutdown settings (IST timezone)
    'auto_shutdown_enabled': True,
    'shutdown_start_hour': 1,       # 1 AM IST
    'shutdown_end_hour': 5,         # 5 AM IST
    'shutdown_timezone': 'Asia/Kolkata',
    
    # Maintenance settings
    'max_log_size_mb': 50,          # Increased from 10
    'max_signal_history_days': 30,  # Increased from 14
    'backup_frequency_hours': 24,   # Daily backups
}

# ============================================================================
# BINANCE API CONFIGURATION (SIMPLIFIED)
# ============================================================================

BINANCE_CONFIG = {
    'base_url': 'https://api.binance.com',
    'timeout': 10,
    'max_retries': 3,
    'retry_delay': 1,
    'rate_limit_calls': 1000,       # Reduced from 1200
    'rate_limit_window': 60
}

def get_timeframe_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes"""
    timeframe_map = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440, '1w': 10080
    }
    return timeframe_map.get(timeframe, 60)

# ============================================================================
# TELEGRAM CONFIGURATION (SIMPLIFIED)
# ============================================================================

TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID'),
    'alert_format': os.getenv('TELEGRAM_ALERT_FORMAT', 'pro_plus'),
    'retry_attempts': 3,
    'retry_delay': 2,
    'rate_limit_messages': 30,
    'rate_limit_window': 60,
    
    # Notification settings (simplified)
    'send_startup_message': True,
    'send_shutdown_messages': True,
    'send_error_alerts': True,
    'send_daily_summary': True,
}

# ============================================================================
# EMERGENCY MODE (DISABLED FOR PRODUCTION)
# ============================================================================

EMERGENCY_MODE = {
    'enabled': False,               # DISABLED for production
    'min_confidence': 15,           # Ultra low for testing
    'max_daily_signals': 30,
    'scan_interval_minutes': 3,
    'disable_cooldowns': False,
    'relax_validation': True,       # More relaxed validation
    'force_signals': False,
}

# ============================================================================
# 🔧 SIMPLIFIED MARKET CONDITIONS
# ============================================================================

MARKET_CONDITIONS = {
    'low_volatility_threshold': 0.025,
    'high_volatility_threshold': 0.08,
    'confidence_adjustment': 0.9,
    'min_signals_per_day': 3,
    'max_signals_per_day': 15,      # Matches MAX_DAILY_TRADES
    'cooldown_reduction_factor': 0.7,
}

# ============================================================================
# 🔧 RELAXED SIGNAL QUALITY VALIDATION
# ============================================================================

SIGNAL_QUALITY_CONFIG = {
    'min_volume_ratio': 0.3,        # RELAXED from 0.6 to 0.3
    'max_volatility': 0.20,         # RELAXED from 0.12 to 0.20
    'rsi_overbought': 85,           # RELAXED from 78 to 85
    'rsi_oversold': 15,             # RELAXED from 22 to 15
    'min_atr_movement': 0.3,        # RELAXED from 0.5 to 0.3
    'trend_confirmation_periods': 2, # REDUCED from 3 to 2
}

# ============================================================================
# AI MODEL CONFIGURATION (OPTIMIZED)
# ============================================================================

MODEL_CONFIG = {
    'model_path': 'ai_model.pkl',
    'backup_path': 'models/backups/',
    'retrain_frequency_days': 14,   # Increased from 7
    'feature_window': 50,           # REDUCED from 100
    'validation_split': 0.2,
    'random_state': 42,
    
    # Performance thresholds (relaxed)
    'min_accuracy': 0.60,           # LOWERED from 0.65
    'min_precision': 0.65,          # LOWERED from 0.70
    'min_recall': 0.60,             # LOWERED from 0.65
    'retrain_if_below_threshold': True,
}

# ============================================================================
# TECHNICAL INDICATORS PARAMETERS (OPTIMIZED)
# ============================================================================

INDICATOR_PARAMS = {
    'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
    'macd': {'fast': 12, 'slow': 26, 'signal': 9},
    'bollinger': {'period': 20, 'std': 2},
    'atr': {'period': 14},
    'ema_fast': {'period': 9},
    'ema_slow': {'period': 21},
    'volume_sma': {'period': 20},
    'stochastic': {'k_period': 14, 'd_period': 3},
    'williams_r': {'period': 14}
}

# ============================================================================
# LEVERAGE CONFIGURATION (SIMPLIFIED)
# ============================================================================

DEFAULT_LEVERAGE_MODE = 'moderate'

LEVERAGE_CONFIG = {
    'conservative': {'min': 2, 'max': 3},   # Low risk
    'moderate': {'min': 2, 'max': 5},       # Balanced (default) - increased max
    'aggressive': {'min': 3, 'max': 7},     # Higher risk - increased max
    
    # Market condition adjustments
    'high_volatility_max': 3,
    'sideways_market_max': 4,
    'trending_market_max': 6,
}

def get_leverage_range(mode: str = None) -> dict:
    """Get leverage range for specified mode"""
    mode = mode or DEFAULT_LEVERAGE_MODE
    return LEVERAGE_CONFIG.get(mode, LEVERAGE_CONFIG['moderate'])

# ============================================================================
# STOP LOSS / TAKE PROFIT CONFIGURATION
# ============================================================================

SL_TP_CONFIG = {
    '1h': {'sl_atr_mult': 1.5, 'tp_atr_mult': 2.5, 'hold_hours': 4},
    '4h': {'sl_atr_mult': 2.0, 'tp_atr_mult': 3.5, 'hold_hours': 8},
    '1d': {'sl_atr_mult': 2.5, 'tp_atr_mult': 4.0, 'hold_hours': 24}
}

# ============================================================================
# DASHBOARD CONFIGURATION (SIMPLIFIED)
# ============================================================================

DASHBOARD_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.getenv('PORT', '5000')),
    'debug': False,                 # ALWAYS False for production
    'refresh_interval': 30,
    'max_signals_display': 50,
    'enable_charts': True,
    'enable_realtime_updates': True,
}

# Dashboard Authentication (OPTIONAL)
DASHBOARD_AUTH = {
    'enabled': False,               # DISABLED by default for simplicity
    'username': os.getenv('DASHBOARD_USER', 'admin'),
    'password': os.getenv('DASHBOARD_PASS'),
    'session_timeout_hours': 24,
}

# ============================================================================
# LOGGING CONFIGURATION (SIMPLIFIED)
# ============================================================================

LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'logs/protrade_ai.log',
    'max_file_size': '50MB',        # Increased from 10MB
    'backup_count': 3,              # Reduced from 5
    'log_rotation': True,
}

# ============================================================================
# SIDEWAYS MARKET TRADING (SIMPLIFIED)
# ============================================================================

SIDEWAYS_ALERT_CONFIG = {
    'range_long_emoji': '📈',
    'range_short_emoji': '📉', 
    'mean_reversion_emoji': '🔄',
    'support_emoji': '🟢',
    'resistance_emoji': '🔴',
    'range_emoji': '📊',
    'breakout_emoji': '💥',
    'consolidation_emoji': '⏸️'
}

RANGE_TRADING_CONFIG = {
    'support_proximity': 0.3,       # RELAXED from 0.2
    'resistance_proximity': 0.3,    # RELAXED from 0.2
    'entry_buffer': 0.02,
    'exit_buffer': 0.01,
    'max_range_leverage': 4,        # Increased from 3
    'min_range_leverage': 2,
}

SIDEWAYS_RSI_CONFIG = {
    'oversold_level': 45,           # RELAXED from 40
    'overbought_level': 55,         # RELAXED from 60
    'extreme_oversold': 30,         # RELAXED from 35
    'extreme_overbought': 70,       # RELAXED from 65
}

STRATEGY_CONFIDENCE = {
    'trending_min': 35,             # LOWERED from 60
    'range_trading_min': 30,        # LOWERED from 55
    'mean_reversion_min': 25,       # LOWERED from 50
    'sideways_bonus': 5,
}

# ============================================================================
# 🔧 SIMPLIFIED PRODUCTION LIMITS
# ============================================================================

API_RATE_LIMITING = {
    'enabled': True,
    'calls_per_minute': 800,        # REDUCED from 1000
    'calls_per_second': 8,          # REDUCED from 10
    'weight_per_minute': 4000,      # REDUCED from 5000
    'cooldown_on_limit': 60,
    'track_weights': False,         # DISABLED for simplicity
}

PRODUCTION_LIMITS = {
    'max_position_size_pct': 20,    # INCREASED from 15
    'daily_loss_limit_pct': 10,     # INCREASED from 8
    'consecutive_loss_limit': 7,    # INCREASED from 5
    'max_drawdown_alert_pct': 20,   # INCREASED from 15
    'position_timeout_hours': 72,   # INCREASED from 48
    'max_memory_mb': 1000,          # INCREASED from 500
    'max_cpu_percent': 90,          # INCREASED from 80
}

ERROR_HANDLING = {
    'max_api_errors_per_hour': 20,  # INCREASED from 10
    'restart_on_critical_error': False,  # DISABLED to prevent loops
    'error_cooldown_minutes': 5,    # REDUCED from 15
    'telegram_error_threshold': 10, # INCREASED from 5
    'log_api_errors': True,
}

SYSTEM_MONITORING = {
    'enable_health_endpoint': True,
    'alert_on_no_signals_hours': 12,  # INCREASED from 6
    'system_check_interval': 600,   # 10 minutes
    'memory_alert_threshold_mb': 800,  # INCREASED from 400
    'cpu_alert_threshold_pct': 90,   # INCREASED from 85
}

# ============================================================================
# FEATURE FLAGS (SIMPLIFIED)
# ============================================================================

FEATURE_FLAGS = {
    'enable_auto_shutdown': True,
    'enable_quick_scans': True,
    'enable_full_scans': True,
    'enable_telegram_alerts': True,
    'enable_dashboard': True,
    'enable_model_retraining': False,  # DISABLED to prevent issues
    'enable_performance_tracking': True,
    'enable_risk_management': True,
    'enable_backup_system': False,     # DISABLED for simplicity
}

# ============================================================================
# UTILITY FUNCTIONS (SIMPLIFIED)
# ============================================================================

def get_confidence_grade(confidence: float) -> str:
    """Get confidence grade letter"""
    if confidence >= 90: return 'A+'
    elif confidence >= 85: return 'A'
    elif confidence >= 80: return 'B+'
    elif confidence >= 75: return 'B'
    elif confidence >= 70: return 'C+'
    elif confidence >= 60: return 'C'
    elif confidence >= 50: return 'D+'
    else: return 'D'

def validate_config() -> list:
    """🔧 SIMPLIFIED: Validate configuration settings"""
    errors = []
    
    # Check required environment variables
    required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")
    
    # 🔧 RELAXED: Validate numeric ranges
    if RISK_PER_TRADE <= 0 or RISK_PER_TRADE > 0.10:  # Allow up to 10%
        errors.append(f"RISK_PER_TRADE must be between 0 and 0.10 (10%), current: {RISK_PER_TRADE}")
    
    if CAPITAL <= 0:
        errors.append("CAPITAL must be positive")
    
    if MAX_DAILY_TRADES <= 0 or MAX_DAILY_TRADES > 50:  # Allow up to 50
        errors.append(f"MAX_DAILY_TRADES must be between 1 and 50, current: {MAX_DAILY_TRADES}")
    
    # Validate timeframes
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
    for tf in TIMEFRAMES:
        if tf not in valid_timeframes:
            errors.append(f"Invalid timeframe: {tf}")
    
    # Validate symbols
    for symbol in SYMBOLS:
        if not symbol.endswith('USDT'):
            errors.append(f"Symbol {symbol} should end with USDT")
    
    # 🔧 OPTIONAL: Dashboard password check (only if enabled)
    if DASHBOARD_AUTH['enabled'] and not os.getenv('DASHBOARD_PASS'):
        errors.append("DASHBOARD_PASS must be set in .env file when authentication is enabled")
    
    return errors

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

ENVIRONMENT = os.getenv('ENVIRONMENT', 'production')  # Default to production
IS_PRODUCTION = ENVIRONMENT == 'production'

# ============================================================================
# 🔧 FIXED: Configuration Validation on Import
# ============================================================================

# Validate configuration on import but don't crash
config_errors = validate_config()
if config_errors:
    print("⚠️  Configuration Issues Found:")
    for error in config_errors:
        print(f"   - {error}")
    print(f"\n📋 Found {len(config_errors)} issues. Bot will attempt to run with defaults.")
    print("🔧 Please fix these issues for optimal operation.")
else:
    print("✅ Configuration validated successfully!")
    
# Display current configuration summary
if __name__ == "__main__":
    print("\n🤖 ProTradeAI Pro+ Configuration Summary:")
    print("=" * 50)
    print(f"📊 Symbols: {len(SYMBOLS)} crypto pairs")
    print(f"⏰ Timeframes: {TIMEFRAMES}")
    print(f"💰 Capital: ${CAPITAL:,.2f}")
    print(f"🎯 Risk per trade: {RISK_PER_TRADE*100:.1f}%")
    print(f"📈 Max daily trades: {MAX_DAILY_TRADES}")
    print(f"🔧 Min signal confidence: {CONFIDENCE_THRESHOLDS['MIN_SIGNAL']}%")
    print(f"🌍 Environment: {ENVIRONMENT}")
    print(f"📱 Telegram configured: {'✅' if TELEGRAM_CONFIG['bot_token'] else '❌'}")
    print(f"🚨 Emergency mode: {'✅ ON' if EMERGENCY_MODE['enabled'] else '❌ OFF'}")
    print("=" * 50)