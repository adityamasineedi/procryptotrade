"""
ProTradeAI Pro+ Model Retraining Script
Weekly model retraining with fresh market data
"""

import pandas as pd
import numpy as np
import requests
import pickle
import logging
from datetime import datetime, timedelta
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

from config import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelRetrainer:
    def __init__(self):
        self.symbols = SYMBOLS
        self.timeframes = TIMEFRAMES
        self.feature_columns = [
            'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower', 'bb_middle',
            'atr', 'volume_ratio', 'ema_fast', 'ema_slow', 'price_position',
            'volatility', 'momentum', 'trend_strength', 'volume_trend',
            'price_change_1h', 'price_change_4h', 'volume_change_1h',
            'rsi_divergence', 'macd_divergence', 'support_resistance'
        ]
        
    def fetch_training_data(self, symbol: str, timeframe: str, days: int = 90) -> pd.DataFrame:
        """Fetch training data from Binance"""
        try:
            # Calculate the number of candles needed
            timeframe_minutes = get_timeframe_minutes(timeframe)
            total_candles = (days * 24 * 60) // timeframe_minutes
            total_candles = min(total_candles, 1000)  # Binance limit
            
            url = f"{BINANCE_CONFIG['base_url']}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': timeframe,
                'limit': total_candles
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
            logger.error(f"Error fetching training data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df.empty or len(df) < 50:
            return df
        
        try:
            # Basic indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            
            # ATR
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # EMAs
            df['ema_fast'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['ema_slow'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            
            # Advanced features
            df['price_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['volatility'] = df['atr'] / df['close']
            df['momentum'] = df['close'].pct_change(10)
            df['trend_strength'] = abs(df['ema_fast'] - df['ema_slow']) / df['close']
            df['volume_trend'] = df['volume'].pct_change(5)
            
            # Price changes
            df['price_change_1h'] = df['close'].pct_change(1)
            df['price_change_4h'] = df['close'].pct_change(4)
            df['volume_change_1h'] = df['volume'].pct_change(1)
            
            # Divergences (simplified)
            df['rsi_divergence'] = (df['rsi'].diff() * df['close'].diff()) < 0
            df['macd_divergence'] = (df['macd'].diff() * df['close'].diff()) < 0
            
            # Support/Resistance (simplified)
            df['support_resistance'] = (
                (df['close'] <= df['low'].rolling(20).min() * 1.01) |
                (df['close'] >= df['high'].rolling(20).max() * 0.99)
            ).astype(int)
            
            # Convert boolean to numeric
            df['rsi_divergence'] = df['rsi_divergence'].astype(int)
            df['macd_divergence'] = df['macd_divergence'].astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            return df
    
    def create_labels(self, df: pd.DataFrame, timeframe: str) -> pd.Series:
        """Create trading labels based on future price movements"""
        try:
            # Define periods based on timeframe
            periods = {
                '1h': 4,   # 4 hours ahead
                '4h': 6,   # 24 hours ahead
                '1d': 5    # 5 days ahead
            }
            
            look_ahead = periods.get(timeframe, 4)
            
            # Calculate future returns
            future_return = df['close'].shift(-look_ahead) / df['close'] - 1
            
            # Define thresholds based on ATR
            atr_threshold = df['atr'] / df['close']
            
            # Create labels
            labels = pd.Series(index=df.index, dtype=int)
            labels[:] = 0  # Default: HOLD
            
            # LONG signals (positive return above threshold)
            long_condition = future_return > atr_threshold * 1.5
            labels[long_condition] = 1
            
            # SHORT signals (negative return below threshold)
            short_condition = future_return < -atr_threshold * 1.5
            labels[short_condition] = 2
            
            return labels
            
        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return pd.Series(index=df.index, dtype=int)
    
    def prepare_dataset(self, days: int = 90) -> tuple:
        """Prepare complete training dataset"""
        logger.info(f"Preparing dataset with {days} days of data...")
        
        all_features = []
        all_labels = []
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                try:
                    logger.info(f"Processing {symbol} {timeframe}...")
                    
                    # Fetch data
                    df = self.fetch_training_data(symbol, timeframe, days)
                    if df.empty or len(df) < 100:
                        logger.warning(f"Insufficient data for {symbol} {timeframe}")
                        continue
                    
                    # Calculate features
                    df = self.calculate_advanced_features(df)
                    
                    # Create labels
                    labels = self.create_labels(df, timeframe)
                    
                    # Extract features
                    feature_data = []
                    for idx in df.index[50:-10]:  # Skip first 50 and last 10 rows
                        try:
                            row_features = []
                            for col in self.feature_columns:
                                if col in df.columns:
                                    value = df.loc[idx, col]
                                    if pd.isna(value):
                                        value = 0.0
                                    row_features.append(value)
                                else:
                                    row_features.append(0.0)
                            
                            if len(row_features) == len(self.feature_columns):
                                all_features.append(row_features)
                                all_labels.append(labels.loc[idx])
                        except KeyError:
                            continue
                    
                except Exception as e:
                    logger.error(f"Error processing {symbol} {timeframe}: {e}")
                    continue
        
        if not all_features:
            raise ValueError("No training data collected")
        
        X = np.array(all_features)
        y = np.array(all_labels)
        
        # Remove invalid samples
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Label distribution: HOLD={np.sum(y==0)}, LONG={np.sum(y==1)}, SHORT={np.sum(y==2)}")
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Train and evaluate multiple models"""
        logger.info("Training models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = {
            'random_forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        results = {}
        
        for name, model in models.items():
            try:
                logger.info(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=5)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'classification_report': classification_report(y_test, y_pred)
                }
                
                logger.info(f"{name} - Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        return results
    
    def select_best_model(self, results: dict) -> tuple:
        """Select the best performing model"""
        best_name = None
        best_score = 0
        
        for name, result in results.items():
            # Combined score (accuracy + cv_mean)
            score = (result['accuracy'] + result['cv_mean']) / 2
            if score > best_score:
                best_score = score
                best_name = name
        
        if best_name is None:
            raise ValueError("No valid models found")
        
        logger.info(f"Best model: {best_name} (score: {best_score:.3f})")
        return best_name, results[best_name]['model']
    
    def save_model(self, model, name: str):
        """Save model to disk"""
        try:
            model_data = {
                'model': model,
                'features': self.feature_columns,
                'created': datetime.now(),
                'version': '2.0',
                'model_type': name,
                'symbols': self.symbols,
                'timeframes': self.timeframes
            }
            
            # Save main model
            with open(MODEL_CONFIG['model_path'], 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save backup with timestamp
            backup_path = f"models/model_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            os.makedirs('models', exist_ok=True)
            with open(backup_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved: {MODEL_CONFIG['model_path']}")
            logger.info(f"Backup saved: {backup_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def retrain_model(self, days: int = 90):
        """Complete model retraining process"""
        try:
            logger.info("Starting model retraining process...")
            start_time = datetime.now()
            
            # Prepare dataset
            X, y = self.prepare_dataset(days)
            
            if len(X) < 1000:
                logger.warning(f"Low sample count: {len(X)}. Consider increasing days parameter.")
            
            # Train models
            results = self.train_model(X, y)
            
            if not results:
                raise ValueError("No models were successfully trained")
            
            # Select best model
            best_name, best_model = self.select_best_model(results)
            
            # Save model
            self.save_model(best_model, best_name)
            
            # Log results
            duration = datetime.now() - start_time
            logger.info(f"Retraining completed in {duration}")
            logger.info(f"Best model: {best_name}")
            logger.info(f"Training samples: {len(X)}")
            logger.info(f"Features: {len(self.feature_columns)}")
            
            return {
                'success': True,
                'best_model': best_name,
                'training_samples': len(X),
                'duration': str(duration),
                'accuracy': results[best_name]['accuracy']
            }
            
        except Exception as e:
            logger.error(f"Error in model retraining: {e}")
            return {
                'success': False,
                'error': str(e)
            }

def main():
    """Main retraining script"""
    print("🤖 ProTradeAI Pro+ Model Retraining")
    print("=" * 50)
    
    import argparse
    parser = argparse.ArgumentParser(description='Retrain ProTradeAI Pro+ model')
    parser.add_argument('--days', type=int, default=90, help='Days of training data (default: 90)')
    parser.add_argument('--verbose', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize retrainer
    retrainer = ModelRetrainer()
    
    # Start retraining
    result = retrainer.retrain_model(args.days)
    
    if result['success']:
        print("✅ Model retraining successful!")
        print(f"📊 Best model: {result['best_model']}")
        print(f"📈 Accuracy: {result['accuracy']:.3f}")
        print(f"⏱️ Duration: {result['duration']}")
        print(f"📋 Training samples: {result['training_samples']}")
    else:
        print("❌ Model retraining failed!")
        print(f"Error: {result['error']}")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    import os
    sys.exit(main())
