"""
ProTradeAI Pro+ Configuration - CLEANED VERSION
Production-ready configuration with no duplicates or conflicts
"""

import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CORE TRADING SETTINGS (FINAL VALUES)
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

# Trading capital and risk settings (OPTIMIZED VALUES)
CAPITAL = float(os.getenv('TRADING_CAPITAL', '10000'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.015'))  # 1.5% per trade (FINAL)
MAX_DAILY_TRADES = int(os.getenv('MAX_DAILY_TRADES', '15'))   # 15 trades (FINAL)

# ============================================================================
# SCHEDULER CONFIGURATION
# ============================================================================

SCHEDULER_CONFIG = {
    # Scanning intervals
    'quick_scan_interval': 5,       # Minutes - Quick scan every 5 minutes
    'full_scan_interval': 15,       # Minutes - Full scan every 15 minutes
    'health_check_interval': 30,    # Minutes - Health check every 30 minutes
    'shutdown_check_interval': 10,   # Minutes - Check shutdown status every 10 minutes
    'cleanup_interval': 120,         # Minutes - Cleanup every 2 hours
    
    # Auto shutdown settings (IST timezone)
    'auto_shutdown_enabled': True,
    'shutdown_start_hour': 2,       # 2 AM IST
    'shutdown_end_hour': 4,         # 4 AM IST
    'shutdown_timezone': 'Asia/Kolkata',
    
    # Maintenance settings
    'max_log_size_mb': 10,
    'max_signal_history_days': 14,
    'backup_frequency_hours': 12,
}

# ============================================================================
# BINANCE API CONFIGURATION
# ============================================================================

BINANCE_CONFIG = {
    'base_url': 'https://api.binance.com',
    'timeout': 10,
    'max_retries': 3,
    'retry_delay': 1,
    'rate_limit_calls': 1200,
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
# TELEGRAM CONFIGURATION
# ============================================================================

TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID'),
    'alert_format': os.getenv('TELEGRAM_ALERT_FORMAT', 'pro_plus'),
    'retry_attempts': 3,
    'retry_delay': 2,
    'rate_limit_messages': 30,
    'rate_limit_window': 60,
    
    # Notification settings
    'send_startup_message': True,
    'send_shutdown_messages': True,
    'send_maintenance_alerts': True,
    'send_error_alerts': True,
    'send_daily_summary': True,
    'send_health_reports': True,
}

# ============================================================================
# EMERGENCY MODE (DISABLED FOR PRODUCTION)
# ============================================================================

EMERGENCY_MODE = {
    'enabled': False,           # DISABLED for production
    'min_confidence': 25,
    'max_daily_signals': 25,
    'scan_interval_minutes': 5,
    'disable_cooldowns': False,
    'relax_validation': False,
    'force_signals': False,
}

# ============================================================================
# CONFIDENCE THRESHOLDS (PRODUCTION QUALITY)
# ============================================================================

CONFIDENCE_THRESHOLDS = {
    'MIN_SIGNAL': 45,               # Quality signals only
    'HIGH_CONFIDENCE': 65,          # Good confidence level
    'MAX_CONFIDENCE': 90,           
    'RANGE_TRADING_MIN': 40,        # Sideways market signals
    'TREND_TRADING_MIN': 50,        # Trending market signals
    'EMERGENCY_MODE': 25,           # Emergency only
}

# ============================================================================
# MARKET REGIME DETECTION (SINGLE DEFINITION)
# ============================================================================

MARKET_REGIME_CONFIG = {
    'sideways_threshold': 0.25,     # Price movement ratio for sideways
    'trending_threshold': 0.65,     # Price movement ratio for trending
    'bb_squeeze_threshold': 0.05,   # Bollinger Band squeeze indicator
    'min_range_size_pct': 2.5,      # Minimum range size to trade
    'regime_lookbook': 20,          # Candles to analyze
    'volatility_periods': 14,       # ATR periods for volatility
    'trend_strength_periods': 10,   # EMA difference periods
}

# ============================================================================
# MARKET CONDITIONS DETECTION
# ============================================================================

MARKET_CONDITIONS = {
    'low_volatility_threshold': 0.025,
    'high_volatility_threshold': 0.08,
    'confidence_adjustment': 0.9,
    'min_signals_per_day': 3,
    'max_signals_per_day': 12,
    'cooldown_reduction_factor': 0.7,
    
    # Market regime specific settings
    'bull_market_threshold': 0.7,
    'bear_market_threshold': -0.7,
    'sideways_market_range': 0.3,
}

# ============================================================================
# SIGNAL QUALITY VALIDATION
# ============================================================================

SIGNAL_QUALITY_CONFIG = {
    'min_volume_ratio': 0.6,
    'max_volatility': 0.12,
    'rsi_overbought': 78,
    'rsi_oversold': 22,
    'min_atr_movement': 0.5,
    'trend_confirmation_periods': 3,
}

# ============================================================================
# AI MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    'model_path': 'ai_model.pkl',
    'backup_path': 'models/backups/',
    'retrain_frequency_days': 7,
    'feature_window': 100,
    'validation_split': 0.2,
    'random_state': 42,
    
    # Performance thresholds
    'min_accuracy': 0.65,
    'min_precision': 0.70,
    'min_recall': 0.65,
    'retrain_if_below_threshold': True,
}

# ============================================================================
# TECHNICAL INDICATORS PARAMETERS
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
# LEVERAGE CONFIGURATION
# ============================================================================

DEFAULT_LEVERAGE_MODE = 'moderate'

LEVERAGE_CONFIG = {
    'conservative': {'min': 2, 'max': 3},   # Low risk
    'moderate': {'min': 2, 'max': 4},       # Balanced (default)
    'aggressive': {'min': 3, 'max': 6},     # Higher risk
    
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
# DASHBOARD CONFIGURATION
# ============================================================================

DASHBOARD_CONFIG = {
    'host': '0.0.0.0',
    'port': int(os.getenv('PORT', '5000')),
    'debug': os.getenv('DASHBOARD_DEBUG', 'False').lower() == 'true',
    'refresh_interval': 30,
    'max_signals_display': 50,
    'enable_charts': True,
    'enable_realtime_updates': True,
}

# Dashboard Authentication (SECURE)
DASHBOARD_AUTH = {
    'enabled': True,
    'username': os.getenv('DASHBOARD_USER', 'admin'),
    'password': os.getenv('DASHBOARD_PASS'),  # NO DEFAULT - Must be set in .env
    'session_timeout_hours': 24,
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'log_file': 'logs/protrade_ai.log',
    'max_file_size': '10MB',
    'backup_count': 5,
    'log_rotation': True,
}

# ============================================================================
# PERFORMANCE MONITORING
# ============================================================================

PERFORMANCE_CONFIG = {
    'track_signals': True,
    'track_accuracy': True,
    'track_returns': True,
    'track_drawdown': True,
    'performance_window_days': 30,
    'benchmark_symbol': 'BTCUSDT',
}

# ============================================================================
# SAFETY AND LIMITS
# ============================================================================

SAFETY_CONFIG = {
    'max_concurrent_positions': 5,
    'max_daily_loss_pct': 10,
    'max_drawdown_pct': 20,
    'cooldown_after_loss_minutes': 30,
    'emergency_stop_enabled': True,
}

# ============================================================================
# SIDEWAYS MARKET TRADING
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
    'support_proximity': 0.2,
    'resistance_proximity': 0.2,
    'entry_buffer': 0.02,
    'exit_buffer': 0.01,
    'max_range_leverage': 3,
    'min_range_leverage': 2,
}

SIDEWAYS_RSI_CONFIG = {
    'oversold_level': 40,
    'overbought_level': 60,
    'extreme_oversold': 35,
    'extreme_overbought': 65,
}

STRATEGY_CONFIDENCE = {
    'trending_min': 60,
    'range_trading_min': 55,
    'mean_reversion_min': 50,
    'sideways_bonus': 5,
}

# ============================================================================
# PRODUCTION SECURITY & MONITORING
# ============================================================================

API_RATE_LIMITING = {
    'enabled': True,
    'calls_per_minute': 1000,
    'calls_per_second': 10,
    'weight_per_minute': 5000,
    'cooldown_on_limit': 60,
    'track_weights': True,
}

PRODUCTION_LIMITS = {
    'max_position_size_pct': 15,
    'daily_loss_limit_pct': 8,
    'consecutive_loss_limit': 5,
    'max_drawdown_alert_pct': 15,
    'position_timeout_hours': 48,
    'max_memory_mb': 500,
    'max_cpu_percent': 80,
}

ERROR_HANDLING = {
    'max_api_errors_per_hour': 10,
    'restart_on_critical_error': True,
    'error_cooldown_minutes': 15,
    'telegram_error_threshold': 5,
    'log_api_errors': True,
}

SYSTEM_MONITORING = {
    'enable_health_endpoint': True,
    'alert_on_no_signals_hours': 6,
    'system_check_interval': 300,
    'memory_alert_threshold_mb': 400,
    'cpu_alert_threshold_pct': 85,
}

# ============================================================================
# FEATURE FLAGS
# ============================================================================

FEATURE_FLAGS = {
    'enable_auto_shutdown': True,
    'enable_quick_scans': True,
    'enable_full_scans': True,
    'enable_telegram_alerts': True,
    'enable_dashboard': True,
    'enable_model_retraining': True,
    'enable_performance_tracking': True,
    'enable_risk_management': True,
    'enable_backup_system': True,
}

# ============================================================================
# UTILITY FUNCTIONS
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
    """Validate configuration settings (UPDATED FOR ACTUAL VALUES)"""
    errors = []
    
    # Check required environment variables
    required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")
    
    # Validate numeric ranges (CORRECTED)
    if RISK_PER_TRADE <= 0 or RISK_PER_TRADE > 0.05:  # Updated for 1.5%
        errors.append("RISK_PER_TRADE must be between 0 and 0.05 (5%)")
    
    if CAPITAL <= 0:
        errors.append("CAPITAL must be positive")
    
    if MAX_DAILY_TRADES <= 0 or MAX_DAILY_TRADES > 25:  # Updated for 15
        errors.append("MAX_DAILY_TRADES must be between 1 and 25")
    
    # Validate timeframes
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
    for tf in TIMEFRAMES:
        if tf not in valid_timeframes:
            errors.append(f"Invalid timeframe: {tf}")
    
    # Validate symbols
    for symbol in SYMBOLS:
        if not symbol.endswith('USDT'):
            errors.append(f"Symbol {symbol} should end with USDT")
    
    # Check dashboard password
    if DASHBOARD_AUTH['enabled'] and not os.getenv('DASHBOARD_PASS'):
        errors.append("DASHBOARD_PASS must be set in .env file for security")
    
    return errors

# ============================================================================
# ENVIRONMENT DETECTION
# ============================================================================

ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
IS_PRODUCTION = ENVIRONMENT == 'production'

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

# Validate configuration on import
config_errors = validate_config()
if config_errors:
    print("⚠️  Configuration Errors Found:")
    for error in config_errors:
        print(f"   - {error}")
    print("\n📋 Please fix these errors before running the bot.")