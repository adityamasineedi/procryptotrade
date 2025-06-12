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
    'quick_scan_interval': 3,       # Minutes - Quick scan every 3 minutes
    'full_scan_interval': 10,       # Minutes - Full scan every 10 minutes (cron)
    'health_check_interval': 10,    # Minutes - Health check every 10 minutes
    'shutdown_check_interval': 5,   # Minutes - Check shutdown status every 5 minutes
    'cleanup_interval': 60,         # Minutes - Cleanup every hour
    
    # Auto shutdown settings
    'auto_shutdown_enabled': False,
    'shutdown_start_hour': 1,       # 1 AM IST
    'shutdown_end_hour': 5,         # 5 AM IST
    'shutdown_timezone': 'Asia/Kolkata',  # IST
    
    # Maintenance settings
    'max_log_size_mb': 10,
    'max_signal_history_days': 7,
    'backup_frequency_hours': 6,
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
# CONFIDENCE AND SIGNAL THRESHOLDS (EMERGENCY LOW VOLATILITY SETTINGS)
# ============================================================================

CONFIDENCE_THRESHOLDS = {
    'MIN_SIGNAL': 60,      # Reduced from 70 for low volatility market
    'HIGH_CONFIDENCE': 75, # Reduced from 85 
    'MAX_CONFIDENCE': 90,  # Reduced from 95
}

# ============================================================================
# Market condition adaptation settings
# ============================================================================

MARKET_CONDITIONS = {
    'low_volatility_threshold': 0.03,  # 3% volatility threshold
    'confidence_adjustment': 0.85,     # 15% reduction in low volatility
    'min_signals_per_day': 2,          # Minimum expected signals
    'max_signals_per_day': 15,         # Maximum signals
    'cooldown_reduction_factor': 0.5,  # Reduce cooldowns in low volatility
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
        return 'C'
    else:
        return 'D'

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
    'conservative': {'min': 2, 'max': 4},   # Increased
    'moderate': {'min': 3, 'max': 7},       # Increased  
    'aggressive': {'min': 5, 'max': 12}     # Increased
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
    'host': os.getenv('DASHBOARD_HOST', '127.0.0.1'),
    'port': int(os.getenv('DASHBOARD_PORT', '5000')),
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