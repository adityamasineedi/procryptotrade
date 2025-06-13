"""
ProTradeAI Pro+ Configuration
Enhanced configuration with auto shutdown and improved scheduling
"""

import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# CORE TRADING SETTINGS
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

# Trading capital and risk settings
CAPITAL = float(os.getenv('TRADING_CAPITAL', '10000'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.02'))  # 2% per trade
MAX_DAILY_TRADES = int(os.getenv('MAX_DAILY_TRADES', '8'))

# ============================================================================
# SCHEDULER CONFIGURATION (Enhanced)
# ============================================================================

SCHEDULER_CONFIG = {
    # Scanning intervals
    'quick_scan_interval': 5,       # Minutes - Quick scan every 5 minutes
    'full_scan_interval': 15,       # Minutes - Full scan every 15 minutes (cron)
    'health_check_interval': 30,    # Minutes - Health check every 30 minutes
    'shutdown_check_interval': 10,   # Minutes - Check shutdown status every 10 minutes
    'cleanup_interval': 120,         # Minutes - Cleanup every 2 hours
    
    # Auto shutdown settings
    'auto_shutdown_enabled': True,  # ENABLED for production
    'shutdown_start_hour': 2,       # 2 AM IST
    'shutdown_end_hour': 4,         # 4 AM IST (shorter window)
    'shutdown_timezone': 'Asia/Kolkata',  # IST
    
    # Maintenance settings
    'max_log_size_mb': 10,
    'max_signal_history_days': 14,  # Keep 2 weeks
    'backup_frequency_hours': 12,   # Backup every 12 hours
}

# ============================================================================
# BINANCE API CONFIGURATION
# ============================================================================

BINANCE_CONFIG = {
    'base_url': 'https://api.binance.com',
    'timeout': 10,
    'max_retries': 3,
    'retry_delay': 1,
    'rate_limit_calls': 1200,  # Calls per minute
    'rate_limit_window': 60    # Seconds
}

def get_timeframe_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes"""
    timeframe_map = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '4h': 240,
        '1d': 1440,
        '1w': 10080
    }
    return timeframe_map.get(timeframe, 60)

# ============================================================================
# TELEGRAM CONFIGURATION (Enhanced)
# ============================================================================

TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID'),
    'alert_format': os.getenv('TELEGRAM_ALERT_FORMAT', 'pro_plus'),  # 'pro_plus' or 'summary'
    'retry_attempts': 3,
    'retry_delay': 2,
    'rate_limit_messages': 30,  # Messages per minute
    'rate_limit_window': 60,    # Seconds
    
    # Enhanced notification settings
    'send_startup_message': True,
    'send_shutdown_messages': True,
    'send_maintenance_alerts': True,
    'send_error_alerts': True,
    'send_daily_summary': True,
    'send_health_reports': True,
}

# DISABLE emergency mode for production
EMERGENCY_MODE = {
    'enabled': False,           # DISABLED for production
    'min_confidence': 25,
    'max_daily_signals': 25,
    'scan_interval_minutes': 5,
    'disable_cooldowns': False,
    'relax_validation': False,
    'force_signals': False,     # DISABLED for production
}

# PRODUCTION-QUALITY CONFIDENCE THRESHOLDS
CONFIDENCE_THRESHOLDS = {
    'MIN_SIGNAL': 45,               # Higher quality (was 15 in testing)
    'HIGH_CONFIDENCE': 65,          # Good confidence level
    'MAX_CONFIDENCE': 90,           # Keep same
    'RANGE_TRADING_MIN': 40,        # Quality sideways signals
    'TREND_TRADING_MIN': 50,        # Quality trend signals
    'EMERGENCY_MODE': 25,           # Only for emergencies
}

# MARKET REGIME DETECTION (Enhanced for Bull/Bear/Sideways)
MARKET_REGIME_CONFIG = {
    'sideways_threshold': 0.25,     # Price movement ratio for sideways
    'trending_threshold': 0.65,     # Price movement ratio for trending
    'bb_squeeze_threshold': 0.05,   # Bollinger Band squeeze indicator
    'min_range_size_pct': 2.5,      # Minimum range size to trade
    'regime_lookbook': 20,          # Candles to analyze
    'volatility_periods': 14,       # ATR periods for volatility
    'trend_strength_periods': 10,   # EMA difference periods
}

# ENHANCED MARKET CONDITIONS DETECTION
MARKET_CONDITIONS = {
    'low_volatility_threshold': 0.025,   # 2.5% volatility threshold
    'high_volatility_threshold': 0.08,   # 8% volatility threshold
    'confidence_adjustment': 0.9,        # 10% reduction in low vol
    'min_signals_per_day': 3,            # Minimum expected signals
    'max_signals_per_day': 12,           # Maximum signals per day
    'cooldown_reduction_factor': 0.7,    # Moderate cooldown reduction
    
    # Market regime specific settings
    'bull_market_threshold': 0.7,        # Strong uptrend
    'bear_market_threshold': -0.7,       # Strong downtrend
    'sideways_market_range': 0.3,        # Range-bound market
}

# QUALITY SIGNAL VALIDATION
SIGNAL_QUALITY_CONFIG = {
    'min_volume_ratio': 0.6,        # Require decent volume
    'max_volatility': 0.12,         # Avoid extremely volatile periods
    'rsi_overbought': 78,           # RSI overbought level
    'rsi_oversold': 22,             # RSI oversold level
    'min_atr_movement': 0.5,        # Minimum ATR for valid signals
    'trend_confirmation_periods': 3, # Periods for trend confirmation
}

# ============================================================================
# AI MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    'model_path': 'ai_model.pkl',
    'backup_path': 'models/backups/',
    'retrain_frequency_days': 7,
    'feature_window': 100,  # Minimum data points needed
    'validation_split': 0.2,
    'random_state': 42,
    
    # Model performance thresholds
    'min_accuracy': 0.65,
    'min_precision': 0.70,
    'min_recall': 0.65,
    'retrain_if_below_threshold': True,
}

# ============================================================================
# TECHNICAL INDICATORS PARAMETERS
# ============================================================================

INDICATOR_PARAMS = {
    'rsi': {
        'period': 14,
        'overbought': 70,
        'oversold': 30
    },
    'macd': {
        'fast': 12,
        'slow': 26,
        'signal': 9
    },
    'bollinger': {
        'period': 20,
        'std': 2
    },
    'atr': {
        'period': 14
    },
    'ema_fast': {
        'period': 9
    },
    'ema_slow': {
        'period': 21
    },
    'volume_sma': {
        'period': 20
    },
    'stochastic': {
        'k_period': 14,
        'd_period': 3
    },
    'williams_r': {
        'period': 14
    }
}

# ============================================================================
# INCREASE TRADING FREQUENCY
# ============================================================================

MAX_DAILY_TRADES = 15  # Increased from 8
RISK_PER_TRADE = 0.015  # Slightly lower risk per trade (1.5% instead of 2%)

# ============================================================================
# MORE AGGRESSIVE LEVERAGE  
# ============================================================================

DEFAULT_LEVERAGE_MODE = 'moderate'  # or 'aggressive' for higher leverage

LEVERAGE_CONFIG = {
    'conservative': {'min': 2, 'max': 3},   # Low risk
    'moderate': {'min': 2, 'max': 4},       # Balanced (default)
    'aggressive': {'min': 3, 'max': 6},     # Higher risk
    
    # Market condition adjustments
    'high_volatility_max': 3,       # Max leverage in high volatility
    'sideways_market_max': 4,       # Max leverage in sideways markets
    'trending_market_max': 6,       # Max leverage in trending markets
}

def get_leverage_range(mode: str = None) -> dict:
    """Get leverage range for specified mode"""
    mode = mode or DEFAULT_LEVERAGE_MODE
    return LEVERAGE_CONFIG.get(mode, LEVERAGE_CONFIG['moderate'])

# ============================================================================
# STOP LOSS / TAKE PROFIT CONFIGURATION
# ============================================================================

SL_TP_CONFIG = {
    '1h': {
        'sl_atr_mult': 1.5,
        'tp_atr_mult': 2.5,
        'hold_hours': 4
    },
    '4h': {
        'sl_atr_mult': 2.0,
        'tp_atr_mult': 3.5,
        'hold_hours': 8
    },
    '1d': {
        'sl_atr_mult': 2.5,
        'tp_atr_mult': 4.0,
        'hold_hours': 24
    }
}

# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================

DASHBOARD_CONFIG = {
    'host': '0.0.0.0',  # Allow external access for deployment
    'port': int(os.getenv('PORT', '5000')),  # Use PORT env var for deployment
    'debug': os.getenv('DASHBOARD_DEBUG', 'False').lower() == 'true',
    'refresh_interval': 30,  # Seconds
    'max_signals_display': 50,
    'enable_charts': True,
    'enable_realtime_updates': True,
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
    'max_daily_loss_pct': 10,  # Stop trading if daily loss exceeds 10%
    'max_drawdown_pct': 20,    # Alert if drawdown exceeds 20%
    'cooldown_after_loss_minutes': 30,
    'emergency_stop_enabled': True,
}

# ============================================================================
# SIDEWAYS MARKET TRADING CONFIGURATION
# ============================================================================

# Sideways market alert configuration (for Telegram notifications)
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

# Market regime detection settings
MARKET_REGIME_CONFIG = {
    'sideways_threshold': 0.25,     # Price movement ratio for sideways
    'trending_threshold': 0.65,     # Price movement ratio for trending
    'bb_squeeze_threshold': 0.05,   # Bollinger Band squeeze indicator
    'min_range_size_pct': 2.5,      # Minimum range size to trade
    'regime_lookbook': 20,          # Candles to analyze
    'volatility_periods': 14,       # ATR periods for volatility
    'trend_strength_periods': 10,   # EMA difference periods
}

# Range trading specific settings
RANGE_TRADING_CONFIG = {
    'support_proximity': 0.2,       # 20% of range from support = "near support"
    'resistance_proximity': 0.2,    # 20% of range from resistance = "near resistance"
    'entry_buffer': 0.02,           # 2% buffer for entries (within 2% of levels)
    'exit_buffer': 0.01,            # 1% buffer for exits (1% before levels)
    'max_range_leverage': 3,        # Maximum leverage for range trading
    'min_range_leverage': 2,        # Minimum leverage for range trading
}

# RSI levels for sideways trading
SIDEWAYS_RSI_CONFIG = {
    'oversold_level': 40,           # RSI below this = oversold in sideways
    'overbought_level': 60,         # RSI above this = overbought in sideways
    'extreme_oversold': 35,         # Very oversold for mean reversion
    'extreme_overbought': 65,       # Very overbought for mean reversion
}

# Confidence thresholds for different strategies
STRATEGY_CONFIDENCE = {
    'trending_min': 60,             # Minimum confidence for trending signals
    'range_trading_min': 55,        # Lower threshold for range trading
    'mean_reversion_min': 50,       # Even lower for mean reversion
    'sideways_bonus': 5,            # Extra confidence points in sideways markets
}

def get_confidence_grade(confidence: float) -> str:
    """Get confidence grade letter"""
    if confidence >= 90:
        return 'A+'
    elif confidence >= 85:
        return 'A'
    elif confidence >= 80:
        return 'B+'
    elif confidence >= 75:
        return 'B'
    elif confidence >= 70:
        return 'C+'
    elif confidence >= 60:
        return 'C'
    elif confidence >= 50:
        return 'D+'
    else:
        return 'D'

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_config() -> list:
    """Validate configuration settings and return errors"""
    errors = []
    
    # Check required environment variables
    required_vars = [
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]
    
    for var in required_vars:
        if not os.getenv(var):
            errors.append(f"Missing required environment variable: {var}")
    
    # Validate numeric ranges
    if RISK_PER_TRADE <= 0 or RISK_PER_TRADE > 0.1:
        errors.append("RISK_PER_TRADE must be between 0 and 0.1 (10%)")
    
    if CAPITAL <= 0:
        errors.append("CAPITAL must be positive")
    
    if MAX_DAILY_TRADES <= 0 or MAX_DAILY_TRADES > 50:
        errors.append("MAX_DAILY_TRADES must be between 1 and 50")
    
    # Validate timeframes
    valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
    for tf in TIMEFRAMES:
        if tf not in valid_timeframes:
            errors.append(f"Invalid timeframe: {tf}")
    
    # Validate symbols
    for symbol in SYMBOLS:
        if not symbol.endswith('USDT'):
            errors.append(f"Symbol {symbol} should end with USDT")
    
    return errors

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
# EXPORT VALIDATION
# ============================================================================

# Validate configuration on import
config_errors = validate_config()
if config_errors:
    print("⚠️  Configuration Errors Found:")
    for error in config_errors:
        print(f"   - {error}")
    print("\n📋 Please fix these errors before running the bot.")

# ============================================================================
# PRODUCTION SECURITY & MONITORING (ADD TO EXISTING CONFIG.PY)
# ============================================================================

# Dashboard Authentication
DASHBOARD_AUTH = {
    'enabled': True,
    'username': os.getenv('DASHBOARD_USER', 'admin'),
    'password': os.getenv('DASHBOARD_PASS', 'change_me_in_production'),
    'session_timeout_hours': 24,
}

# API Rate Limiting Protection
API_RATE_LIMITING = {
    'enabled': True,
    'calls_per_minute': 1000,
    'calls_per_second': 10,
    'weight_per_minute': 5000,
    'cooldown_on_limit': 60,
    'track_weights': True,
}

# Production Safety Limits
PRODUCTION_LIMITS = {
    'max_position_size_pct': 15,
    'daily_loss_limit_pct': 8,
    'consecutive_loss_limit': 5,
    'max_drawdown_alert_pct': 15,
    'position_timeout_hours': 48,
    'max_memory_mb': 500,
    'max_cpu_percent': 80,
}

# Enhanced Error Handling
ERROR_HANDLING = {
    'max_api_errors_per_hour': 10,
    'restart_on_critical_error': True,
    'error_cooldown_minutes': 15,
    'telegram_error_threshold': 5,
    'log_api_errors': True,
}

# System Monitoring
SYSTEM_MONITORING = {
    'enable_health_endpoint': True,
    'alert_on_no_signals_hours': 6,
    'system_check_interval': 300,
    'memory_alert_threshold_mb': 400,
    'cpu_alert_threshold_pct': 85,
}

# Environment Detection
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')  # production, development, testing
IS_PRODUCTION = ENVIRONMENT == 'production'