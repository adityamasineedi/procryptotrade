"""
ProTradeAI Pro+ Configuration
Risk management, leverage settings, and trading parameters
"""

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================

# Risk Management
RISK_PER_TRADE = 0.02  # 2% of capital per trade
MAX_DAILY_TRADES = 10
MAX_OPEN_POSITIONS = 5
CAPITAL = float(os.getenv('TRADING_CAPITAL', 10000))  # Default $10,000

# Leverage Settings
LEVERAGE_CONFIG = {
    'conservative': {'min': 2, 'max': 3, 'atr_multiplier': 1.5},
    'moderate': {'min': 3, 'max': 5, 'atr_multiplier': 2.0},
    'aggressive': {'min': 5, 'max': 10, 'atr_multiplier': 2.5}
}

DEFAULT_LEVERAGE_MODE = 'moderate'

# Signal Confidence Thresholds
CONFIDENCE_THRESHOLDS = {
    'A+': 90,  # Highest confidence
    'A': 85,
    'B+': 80,
    'B': 75,
    'C+': 70,
    'C': 65,
    'MIN_SIGNAL': 70  # Minimum confidence to send signal
}

# =============================================================================
# TIMEFRAMES & SYMBOLS
# =============================================================================

TIMEFRAMES = ['1h', '4h', '1d']
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
    'XRPUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'MATICUSDT'
]

# Timeframe priorities (higher number = higher priority)
TIMEFRAME_PRIORITY = {
    '1h': 1,
    '4h': 2,
    '1d': 3
}

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

INDICATOR_PARAMS = {
    'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
    'macd': {'fast': 12, 'slow': 26, 'signal': 9},
    'bollinger': {'period': 20, 'std': 2},
    'atr': {'period': 14},
    'volume_sma': {'period': 20},
    'ema_fast': {'period': 9},
    'ema_slow': {'period': 21}
}

# =============================================================================
# STOP LOSS & TAKE PROFIT
# =============================================================================

SL_TP_CONFIG = {
    '1h': {'sl_atr_mult': 1.5, 'tp_atr_mult': 3.0, 'hold_hours': 2},
    '4h': {'sl_atr_mult': 2.0, 'tp_atr_mult': 4.0, 'hold_hours': 8},
    '1d': {'sl_atr_mult': 2.5, 'tp_atr_mult': 5.0, 'hold_hours': 24}
}

# =============================================================================
# TELEGRAM CONFIGURATION
# =============================================================================

TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN'),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID'),
    'alert_format': 'pro_plus',  # 'pro_plus' or 'summary'
    'retry_attempts': 3,
    'retry_delay': 5  # seconds
}

# =============================================================================
# SCHEDULER CONFIGURATION
# =============================================================================

SCHEDULER_CONFIG = {
    'signal_interval': 15,  # minutes
    'health_check_interval': 5,  # minutes
    'cleanup_interval': 60,  # minutes
    'max_log_size_mb': 50
}

# =============================================================================
# MODEL & AI CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    'model_path': 'ai_model.pkl',
    'feature_window': 100,  # Number of candles for features
    'retrain_days': 7,  # Retrain every 7 days
    'feature_importance_threshold': 0.01
}

# =============================================================================
# DASHBOARD CONFIGURATION
# =============================================================================

DASHBOARD_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'max_signals_display': 100,
    'refresh_interval': 30  # seconds
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG = {
    'log_file': 'logs/protrade_ai.log',
    'max_log_files': 5,
    'log_level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

# =============================================================================
# BINANCE API CONFIGURATION (Read-only for price data)
# =============================================================================

BINANCE_CONFIG = {
    'base_url': 'https://api.binance.com',
    'timeout': 10,
    'max_retries': 3
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    if not TELEGRAM_CONFIG['bot_token']:
        errors.append("TELEGRAM_BOT_TOKEN not set in environment")
    
    if not TELEGRAM_CONFIG['chat_id']:
        errors.append("TELEGRAM_CHAT_ID not set in environment")
    
    if CAPITAL <= 0:
        errors.append("TRADING_CAPITAL must be positive")
    
    if RISK_PER_TRADE <= 0 or RISK_PER_TRADE > 0.1:
        errors.append("RISK_PER_TRADE should be between 0 and 0.1 (10%)")
    
    return errors

def get_leverage_range(mode=None):
    """Get leverage range for given mode"""
    mode = mode or DEFAULT_LEVERAGE_MODE
    return LEVERAGE_CONFIG.get(mode, LEVERAGE_CONFIG['moderate'])

def get_confidence_grade(confidence):
    """Convert confidence percentage to grade"""
    for grade, threshold in CONFIDENCE_THRESHOLDS.items():
        if grade == 'MIN_SIGNAL':
            continue
        if confidence >= threshold:
            return grade
    return 'D'

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_currency(amount):
    """Format currency for display"""
    return f"${amount:,.2f}"

def format_percentage(value):
    """Format percentage for display"""
    return f"{value:.2f}%"

def get_timeframe_minutes(timeframe):
    """Convert timeframe to minutes"""
    timeframe_map = {
        '1m': 1, '5m': 5, '15m': 15, '30m': 30,
        '1h': 60, '4h': 240, '1d': 1440
    }
    return timeframe_map.get(timeframe, 60)
