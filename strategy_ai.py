"""
ProTradeAI Pro+ Strategy Engine
AI-powered signal generation with leverage-aware risk management
"""

import pandas as pd
import numpy as np
import requests
import pickle
import logging
from datetime import datetime, timedelta
import ta
from typing import Dict, Tuple, Optional, List
import joblib
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings('ignore')

from config import (
    MODEL_CONFIG,
    BINANCE_CONFIG,
    INDICATOR_PARAMS,
    SL_TP_CONFIG,
    SYMBOLS,
    TIMEFRAMES,
    MAX_DAILY_TRADES,
    TIMEFRAME_PRIORITY,
    CONFIDENCE_THRESHOLDS,
    DEFAULT_LEVERAGE_MODE,
    get_leverage_range,
    get_confidence_grade,
    MARKET_CONDITIONS
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrategyAI:
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.last_signals = {}
        self.load_or_create_model()
        
    def load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            with open(MODEL_CONFIG['model_path'], 'rb') as f:
                model_data = pickle.load(f)
                self.model = model_data['model']
                self.feature_columns = model_data['features']
            logger.info("AI model loaded successfully")
        except FileNotFoundError:
            logger.info("No existing model found, creating new one")
            self.create_default_model()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.create_default_model()
    
    def create_default_model(self):
        """Create a default Random Forest model with dummy data"""
        # Create dummy feature columns
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_middle',
            'atr', 'volume_ratio', 'ema_fast', 'ema_slow', 'price_position',
            'volatility', 'momentum', 'trend_strength', 'volume_trend'
        ]
        
        # Create a simple Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Train with dummy data
        X_dummy = np.random.random((1000, len(self.feature_columns)))
        y_dummy = np.random.choice([0, 1, 2], 1000)  # 0=HOLD, 1=LONG, 2=SHORT
        self.model.fit(X_dummy, y_dummy)
        
        # Save model
        self.save_model()
        logger.info("Default AI model created and saved")
    
    def save_model(self):
        """Save model to disk"""
        try:
            model_data = {
                'model': self.model,
                'features': self.feature_columns,
                'created': datetime.now(),
                'version': '1.0'
            }
            with open(MODEL_CONFIG['model_path'], 'wb') as f:
                pickle.dump(model_data, f)
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def get_binance_data(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """Fetch OHLCV data from Binance"""
        try:
            url = f"{BINANCE_CONFIG['base_url']}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=BINANCE_CONFIG['timeout'])
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'count', 'buy_volume', 
                'buy_quote_volume', 'ignore'
            ])
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        if df.empty or len(df) < 50:
            return df
        
        try:
            # RSI
            df['rsi'] = ta.momentum.RSIIndicator(
                df['close'], 
                window=INDICATOR_PARAMS['rsi']['period']
            ).rsi()
            
            # MACD
            macd = ta.trend.MACD(
                df['close'],
                window_fast=INDICATOR_PARAMS['macd']['fast'],
                window_slow=INDICATOR_PARAMS['macd']['slow'],
                window_sign=INDICATOR_PARAMS['macd']['signal']
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(
                df['close'],
                window=INDICATOR_PARAMS['bollinger']['period'],
                window_dev=INDICATOR_PARAMS['bollinger']['std']
            )
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            
            # ATR
            df['atr'] = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'],
                window=INDICATOR_PARAMS['atr']['period']
            ).average_true_range()
            
            # Volume
            df['volume_sma'] = df['volume'].rolling(
                window=INDICATOR_PARAMS['volume_sma']['period']
            ).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # EMAs
            df['ema_fast'] = ta.trend.EMAIndicator(
                df['close'], 
                window=INDICATOR_PARAMS['ema_fast']['period']
            ).ema_indicator()
            
            df['ema_slow'] = ta.trend.EMAIndicator(
                df['close'], 
                window=INDICATOR_PARAMS['ema_slow']['period']
            ).ema_indicator()
            
            # Additional features
            df['price_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['volatility'] = df['atr'] / df['close']
            df['momentum'] = df['close'].pct_change(10)
            df['trend_strength'] = abs(df['ema_fast'] - df['ema_slow']) / df['close']
            df['volume_trend'] = df['volume'].pct_change(5)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return df
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for model prediction"""
        if df.empty:
            return np.array([])
        
        try:
            # Get the latest row with all features
            latest = df.iloc[-1]
            features = []
            
            for col in self.feature_columns:
                if col in df.columns:
                    value = latest[col]
                    # Handle NaN values
                    if pd.isna(value):
                        value = 0.0
                    features.append(value)
                else:
                    features.append(0.0)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return np.array([]).reshape(1, -1)
    
    def calculate_leverage(self, df: pd.DataFrame, confidence: float, mode: str = None) -> int:
        """Calculate appropriate leverage based on volatility and confidence"""
        try:
            mode = mode or DEFAULT_LEVERAGE_MODE
            leverage_config = get_leverage_range(mode)
            
            # Get ATR-based volatility
            atr = df['atr'].iloc[-1]
            price = df['close'].iloc[-1]
            volatility = atr / price
            
            # Base leverage on confidence and volatility
            base_leverage = leverage_config['min']
            max_leverage = leverage_config['max']
            
            # Confidence adjustment (higher confidence = higher leverage)
            confidence_factor = min(confidence / 100, 1.0)
            
            # Volatility adjustment (higher volatility = lower leverage)
            volatility_factor = max(0.5, 1 - (volatility * 10))
            
            # Calculate final leverage
            leverage = base_leverage + (max_leverage - base_leverage) * confidence_factor * volatility_factor
            leverage = max(2, min(max_leverage, round(leverage)))
            
            return int(leverage)
            
        except Exception as e:
            logger.error(f"Error calculating leverage: {e}")
            return 2
    
    def calculate_sl_tp(self, df: pd.DataFrame, signal_type: str, timeframe: str, leverage: int) -> Dict:
        """Calculate Stop Loss and Take Profit levels"""
        try:
            price = df['close'].iloc[-1]
            atr = df['atr'].iloc[-1]
            
            # Get timeframe config
            tf_config = SL_TP_CONFIG.get(timeframe, SL_TP_CONFIG['4h'])
            
            # Adjust for leverage (higher leverage = tighter stops)
            leverage_factor = min(1.0, 3.0 / leverage)
            
            sl_distance = atr * tf_config['sl_atr_mult'] * leverage_factor
            tp_distance = atr * tf_config['tp_atr_mult'] * leverage_factor
            
            if signal_type == 'LONG':
                sl_price = price - sl_distance
                tp_price = price + tp_distance
            else:  # SHORT
                sl_price = price + sl_distance
                tp_price = price - tp_distance
            
            # Calculate risk/reward ratio
            risk = abs(price - sl_price)
            reward = abs(tp_price - price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            return {
                'entry_price': price,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'sl_distance_pct': (abs(price - sl_price) / price) * 100,
                'tp_distance_pct': (abs(tp_price - price) / price) * 100,
                'rr_ratio': rr_ratio,
                'hold_hours': tf_config['hold_hours']
            }
            
        except Exception as e:
            logger.error(f"Error calculating SL/TP: {e}")
            return {}
    
    def predict_signal(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Generate trading signal for symbol and timeframe - Enhanced version"""
        try:
            # Calculate dynamic cooldown based on market conditions
            signal_key = f"{symbol}_{timeframe}"
            
            # Reduced cooldown for low volatility markets
            market_volatility = self.get_market_volatility()
            if market_volatility < MARKET_CONDITIONS['low_volatility_threshold']:
                cooldown_minutes = 30 if timeframe == '1h' else 45  # Reduced cooldown
            else:
                cooldown_minutes = 60 if timeframe == '1h' else 90  # Normal cooldown
            
            if signal_key in self.last_signals:
                last_time = self.last_signals[signal_key]
                if datetime.now() - last_time < timedelta(minutes=cooldown_minutes):
                    logger.debug(f"Signal cooldown active for {symbol} {timeframe} ({cooldown_minutes}m)")
                    return None

            # Get market data
            df = self.get_binance_data(symbol, timeframe)
            if df.empty or len(df) < MODEL_CONFIG['feature_window']:
                logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(df)} candles")
                return None

            # Calculate indicators
            df = self.calculate_technical_indicators(df)
            
            # Check for sufficient indicator data
            required_indicators = ['rsi', 'macd', 'atr', 'bb_upper', 'bb_lower']
            missing_indicators = [ind for ind in required_indicators if df[ind].isna().all()]
            if missing_indicators:
                logger.warning(f"Missing indicators for {symbol} {timeframe}: {missing_indicators}")
                return None

            # Prepare features
            features = self.prepare_features(df)
            if features.size == 0:
                logger.warning(f"No features prepared for {symbol} {timeframe}")
                return None

            # Get model prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]

            # Convert prediction to signal
            signal_map = {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}
            signal_type = signal_map[prediction]

            # Calculate raw confidence
            raw_confidence = max(probabilities) * 100
            
            # Adjust confidence for market conditions
            adjusted_confidence = self.adjust_confidence_for_market(raw_confidence)
            
            logger.info(f"Signal analysis for {symbol} {timeframe}: {signal_type}, raw confidence: {raw_confidence:.1f}%, adjusted: {adjusted_confidence:.1f}%")

            # Apply adjusted confidence threshold
            if signal_type == 'HOLD' or adjusted_confidence < CONFIDENCE_THRESHOLDS['MIN_SIGNAL']:
                logger.info(f"Signal filtered out - Type: {signal_type}, Confidence: {adjusted_confidence:.1f}%")
                return None

            # Calculate leverage
            leverage = self.calculate_leverage(df, adjusted_confidence)

            # Calculate SL/TP
            sl_tp_data = self.calculate_sl_tp(df, signal_type, timeframe, leverage)

            # Create signal with market volatility info
            signal = {
                'symbol': symbol,
                'timeframe': timeframe,
                'signal_type': signal_type,
                'confidence': adjusted_confidence,
                'raw_confidence': raw_confidence,
                'confidence_grade': get_confidence_grade(adjusted_confidence),
                'leverage': leverage,
                'timestamp': datetime.now(),
                'current_price': df['close'].iloc[-1],
                'market_volatility': market_volatility,
                'rsi': df['rsi'].iloc[-1],
                'macd': df['macd'].iloc[-1],
                'atr': df['atr'].iloc[-1],
                'volume_ratio': df['volume_ratio'].iloc[-1],
                **sl_tp_data
            }

            # Store signal timestamp
            self.last_signals[signal_key] = datetime.now()

            logger.info(f"✅ Signal generated: {symbol} {timeframe} {signal_type} {adjusted_confidence:.1f}% (volatility: {market_volatility:.4f})")
            return signal

        except Exception as e:
            logger.error(f"Error generating signal for {symbol} {timeframe}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def scan_all_symbols(self) -> List[Dict]:
        """Scan all symbols and timeframes for signals"""
        signals = []
        
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                try:
                    signal = self.predict_signal(symbol, timeframe)
                    if signal:
                        signals.append(signal)
                        
                        # Prioritize higher timeframes
                        if len(signals) >= MAX_DAILY_TRADES:
                            break
                            
                except Exception as e:
                    logger.error(f"Error scanning {symbol} {timeframe}: {e}")
                    continue
            
            if len(signals) >= MAX_DAILY_TRADES:
                break
        
        # Sort by confidence and timeframe priority
        signals.sort(key=lambda x: (
            x['confidence'], 
            TIMEFRAME_PRIORITY.get(x['timeframe'], 0)
        ), reverse=True)
        
        return signals[:MAX_DAILY_TRADES]
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        return {
            'model_type': type(self.model).__name__,
            'feature_count': len(self.feature_columns),
            'features': self.feature_columns,
            'last_prediction': datetime.now()
        }

    def get_market_volatility(self, symbol: str = 'BTCUSDT', days: int = 7) -> float:
        """Get current market volatility"""
        try:
            df = self.get_binance_data(symbol, '1d', limit=days)
            if df.empty or len(df) < 3:
                return 0.05  # Default moderate volatility
            
            # Calculate ATR-based volatility
            atr = ta.volatility.AverageTrueRange(
                df['high'], df['low'], df['close'], window=min(len(df)-1, 7)
            ).average_true_range()
            
            avg_price = df['close'].mean()
            volatility = atr.iloc[-1] / avg_price if avg_price > 0 else 0.05
            
            logger.info(f"Market volatility ({symbol}): {volatility:.4f}")
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating market volatility: {e}")
            return 0.05

    def adjust_confidence_for_market(self, confidence: float) -> float:
        """Adjust confidence threshold based on current market conditions"""
        try:
            # Get current market volatility
            volatility = self.get_market_volatility()
            
            # Adjust confidence for low volatility markets
            if volatility < MARKET_CONDITIONS['low_volatility_threshold']:
                adjusted_confidence = confidence * MARKET_CONDITIONS['confidence_adjustment']
                logger.info(f"Low volatility detected ({volatility:.4f}), adjusting confidence: {confidence:.1f}% -> {adjusted_confidence:.1f}%")
                return adjusted_confidence
            
            return confidence
            
        except Exception as e:
            logger.error(f"Error adjusting confidence: {e}")
            return confidence

    def debug_signal_generation(self) -> Dict:
        """Debug signal generation process"""
        debug_info = {
            'model_loaded': self.model is not None,
            'feature_columns': len(self.feature_columns) if self.feature_columns else 0,
            'last_signals_count': len(self.last_signals),
            'market_volatility': self.get_market_volatility(),
            'symbol_analysis': {}
        }
        
        # Test top 3 symbols
        test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        
        for symbol in test_symbols:
            symbol_debug = {}
            
            for timeframe in ['1h', '4h']:
                try:
                    # Test data fetching
                    df = self.get_binance_data(symbol, timeframe, limit=100)
                    symbol_debug[timeframe] = {
                        'data_length': len(df),
                        'data_available': not df.empty,
                        'latest_price': df['close'].iloc[-1] if not df.empty else None,
                    }
                    
                    if not df.empty and len(df) >= 50:
                        # Test indicator calculation
                        df_with_indicators = self.calculate_technical_indicators(df)
                        symbol_debug[timeframe]['indicators_calculated'] = True
                        symbol_debug[timeframe]['rsi'] = df_with_indicators['rsi'].iloc[-1] if 'rsi' in df_with_indicators else None
                        
                        # Test feature preparation
                        features = self.prepare_features(df_with_indicators)
                        symbol_debug[timeframe]['features_prepared'] = features.size > 0
                        
                        if features.size > 0 and self.model:
                            # Test model prediction
                            prediction = self.model.predict(features)[0]
                            probabilities = self.model.predict_proba(features)[0]
                            
                            symbol_debug[timeframe]['model_prediction'] = prediction
                            symbol_debug[timeframe]['raw_confidence'] = max(probabilities) * 100
                            symbol_debug[timeframe]['adjusted_confidence'] = self.adjust_confidence_for_market(max(probabilities) * 100)
                    
                except Exception as e:
                    symbol_debug[timeframe] = {'error': str(e)}
            
            debug_info['symbol_analysis'][symbol] = symbol_debug
        
        return debug_info

# Global strategy instance
strategy_ai = StrategyAI()
