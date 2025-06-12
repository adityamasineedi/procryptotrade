"""
ProTradeAI Pro+ Main Application
Scheduler and orchestrator for the trading bot with auto shutdown
"""

import os
import logging
import time
import signal
import sys
from datetime import datetime, timedelta
from typing import List, Dict
import threading
import json
from pathlib import Path
import pytz

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from config import MARKET_CONDITIONS, SYMBOLS, TIMEFRAMES, RISK_PER_TRADE, CAPITAL, LOGGING_CONFIG, SCHEDULER_CONFIG, validate_config
from strategy_ai import strategy_ai
from notifier import telegram_notifier

# Setup logging
def setup_logging():
    """Setup comprehensive logging"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(LOGGING_CONFIG['log_file']),
            logging.StreamHandler(sys.stdout)
        ]
    )

setup_logging()
logger = logging.getLogger(__name__)

class ProTradeAIBot:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.is_running = False
        self.is_shutdown_period = False
        self.start_time = datetime.now()
        self.signals_today = []
        self.daily_stats = {}
        self.last_health_check = datetime.now()
        
        # Timezone for shutdown schedule (IST)
        self.timezone = pytz.timezone('Asia/Kolkata')
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Initialize data storage
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)
    
    def is_shutdown_time(self) -> bool:
        """Check if current time is in shutdown period (1 AM - 5 AM IST)"""
        try:
            current_time = datetime.now(self.timezone)
            current_hour = current_time.hour
            
            # Shutdown between 1 AM and 5 AM IST
            return 1 <= current_hour < 5
            
        except Exception as e:
            logger.error(f"Error checking shutdown time: {e}")
            return False
    
    def check_shutdown_status(self):
        """Check and handle shutdown/resume based on time"""
        try:
            should_shutdown = self.is_shutdown_time()
            
            if should_shutdown and not self.is_shutdown_period:
                # Enter shutdown period
                logger.info("🌙 Entering maintenance shutdown period (1-5 AM IST)")
                self.is_shutdown_period = True
                
                # Pause trading-related jobs but keep system monitoring
                for job in self.scheduler.get_jobs():
                    if job.id in ['signal_scan', 'quick_scan']:
                        job.pause()
                        logger.info(f"Paused job: {job.id}")
                
                # Send shutdown notification
                telegram_notifier.send_message(
                    "🌙 <b>Maintenance Period Started</b>\n\n"
                    "🔸 Signal scanning paused (1-5 AM IST)\n"
                    "🔸 System monitoring continues\n"
                    "🔸 Bot will resume automatically at 5 AM IST\n\n"
                    "<i>Good night! 😴</i>"
                )
                
            elif not should_shutdown and self.is_shutdown_period:
                # Exit shutdown period
                logger.info("☀️ Exiting maintenance shutdown period")
                self.is_shutdown_period = False
                
                # Resume trading-related jobs
                for job in self.scheduler.get_jobs():
                    if job.id in ['signal_scan', 'quick_scan']:
                        job.resume()
                        logger.info(f"Resumed job: {job.id}")
                
                # Send resume notification
                telegram_notifier.send_message(
                    "☀️ <b>Trading Resumed</b>\n\n"
                    "✅ Signal scanning reactivated\n"
                    "🔸 All systems operational\n"
                    "🔸 Ready for new trading opportunities\n\n"
                    "<i>Good morning! Let's trade! 🚀</i>"
                )
                
        except Exception as e:
            logger.error(f"Error in shutdown status check: {e}")
    
    def validate_configuration(self) -> bool:
        """Validate all configuration settings"""
        logger.info("Validating configuration...")
        
        # Check config
        errors = validate_config()
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False
        
        # Test Telegram connection
        logger.info("Testing Telegram connection...")
        if not telegram_notifier.test_connection():
            logger.error("Telegram connection test failed")
            return False
        
        # Check AI model
        logger.info("Checking AI model...")
        model_info = strategy_ai.get_model_info()
        logger.info(f"Model loaded: {model_info['model_type']} with {model_info['feature_count']} features")
        
        logger.info("Configuration validation successful")
        return True
    
    def quick_market_scan(self):
        """Enhanced market scan with sideways trading support"""
        try:
            if self.is_shutdown_period:
                logger.info("Skipping quick scan - in shutdown period")
                return

            logger.info("🔍 Starting enhanced market scan (trending + sideways)...")
            
            # Check overall market conditions
            market_volatility = strategy_ai.get_market_volatility()
            logger.info(f"📊 Current market volatility: {market_volatility:.4f}")
            
            # Adaptive scanning based on market conditions
            if market_volatility < MARKET_CONDITIONS['low_volatility_threshold']:
                scan_symbols = SYMBOLS[:5]  # Focus on top 5 in low volatility
                scan_timeframes = ['4h', '1h']  # Include 1h for more range opportunities
                scan_strategy = "SIDEWAYS_FOCUSED"
                logger.info("🔄 Low volatility - focusing on range/sideways trading")
            else:
                scan_symbols = SYMBOLS[:3]
                scan_timeframes = ['1h', '4h']
                scan_strategy = "TREND_FOCUSED"
                logger.info("⚡ Normal volatility - using standard trend scanning")

            signals = []
            processed_pairs = 0
            trending_signals = 0
            sideways_signals = 0
            
            for symbol in scan_symbols:
                for timeframe in scan_timeframes:
                    try:
                        processed_pairs += 1
                        logger.info(f"🔎 Scanning {symbol} {timeframe} ({processed_pairs}/{len(scan_symbols)*len(scan_timeframes)})")
                        
                        # Use enhanced signal prediction (combines trending + sideways)
                        signal = strategy_ai.predict_signal_enhanced(symbol, timeframe)
                        
                        if signal:
                            signals.append(signal)
                            
                            # Track signal types
                            market_regime = signal.get('market_regime', 'TRENDING')
                            if market_regime == 'SIDEWAYS' or 'RANGE' in signal.get('strategy_type', ''):
                                sideways_signals += 1
                                logger.info(f"✅ SIDEWAYS Signal: {signal['symbol']} {signal['signal_type']} {signal['confidence']:.1f}% (Range trading)")
                            else:
                                trending_signals += 1
                                logger.info(f"✅ TRENDING Signal: {signal['symbol']} {signal['signal_type']} {signal['confidence']:.1f}% (Momentum)")
                        else:
                            logger.debug(f"❌ No signal: {symbol} {timeframe}")
                            
                    except Exception as e:
                        logger.error(f"Error scanning {symbol} {timeframe}: {e}")

            # Results summary
            total_signals = len(signals)
            if total_signals > 0:
                logger.info(f"🎯 Market scan generated {total_signals} signals:")
                logger.info(f"   📈 Trending signals: {trending_signals}")
                logger.info(f"   🔄 Sideways signals: {sideways_signals}")
                logger.info(f"   📊 Strategy mix: {scan_strategy}")
                
                for signal in signals:
                    self.process_signal(signal)
            else:
                logger.info("📭 Market scan: No signals generated")
                
                # Enhanced debug info for no signals
                if not hasattr(self, '_last_debug_time'):
                    self._last_debug_time = datetime.now()
                
                time_since_debug = datetime.now() - self._last_debug_time
                if time_since_debug > timedelta(hours=1):  # Debug every hour if no signals
                    logger.info("🔧 Running enhanced signal generation debug...")
                    self.debug_no_signals_enhanced()
                    self._last_debug_time = datetime.now()

        except Exception as e:
            logger.error(f"Error in enhanced market scan: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def full_market_scan(self):
        """Full market scan (every 15 minutes during active hours)"""
        try:
            if self.is_shutdown_period:
                logger.info("Skipping full scan - in shutdown period")
                return
            
            logger.info("Starting full market scan...")
            
            # Get signals from AI strategy (all symbols and timeframes)
            signals = strategy_ai.scan_all_symbols()
            
            if not signals:
                logger.info("Full scan: No signals generated")
                return
            
            logger.info(f"Full scan generated {len(signals)} signals")
            
            # Process each signal
            for signal in signals:
                self.process_signal(signal)
                time.sleep(2)  # Small delay between signals
            
            # Update daily stats
            self._update_daily_stats()
            
        except Exception as e:
            logger.error(f"Error in full market scan: {e}")
            telegram_notifier.send_error_alert(str(e), "Full Market Scanner")
    
    def process_signal(self, signal: Dict):
        """Process and send a trading signal"""
        try:
            # Add to daily tracking
            self.signals_today.append(signal)
            
            # Send telegram alert
            success = telegram_notifier.send_signal_alert(signal)
            
            if success:
                logger.info(f"Signal sent: {signal['symbol']} {signal['signal_type']} {signal['confidence']:.1f}%")
                
                # Save signal to file
                self._save_signal(signal)
            else:
                logger.error(f"Failed to send signal: {signal['symbol']}")
                
        except Exception as e:
            logger.error(f"Error processing signal {signal.get('symbol', 'Unknown')}: {e}")
    
    def health_check(self):
        """System health check"""
        try:
            logger.info("Performing health check...")
            
            current_time = datetime.now()
            uptime = current_time - self.start_time
            
            # Check if we're generating signals
            signals_last_hour = [
                s for s in self.signals_today 
                if s['timestamp'] > current_time - timedelta(hours=1)
            ]
            
            # Calculate success rate (dummy for now)
            success_rate = 85.0  # In real implementation, track actual performance
            
            status = {
                'healthy': True,
                'status': 'Shutdown Period' if self.is_shutdown_period else 'Running',
                'uptime': f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m",
                'last_signal_time': max([s['timestamp'] for s in self.signals_today], default=datetime.min).strftime('%H:%M:%S') if self.signals_today else 'Never',
                'signals_today': len(self.signals_today),
                'signals_last_hour': len(signals_last_hour),
                'success_rate': success_rate,
                'model_accuracy': 78.5,  # Dummy value
                'is_shutdown_period': self.is_shutdown_period,
                'next_resume_time': '05:00 IST' if self.is_shutdown_period else 'N/A'
            }
            
            # Send status update if it's been more than 6 hours
            if current_time - self.last_health_check > timedelta(hours=6):
                telegram_notifier.send_system_status(status)
                self.last_health_check = current_time
            
            # Log health status
            logger.info(f"Health check: {len(self.signals_today)} signals today, {len(signals_last_hour)} in last hour")
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
    
    def cleanup_old_data(self):
        """Clean up old data and logs"""
        try:
            logger.info("Cleaning up old data...")
            
            # Clean old signals (keep last 7 days)
            cutoff_date = datetime.now() - timedelta(days=7)
            self.signals_today = [
                s for s in self.signals_today 
                if s['timestamp'] > cutoff_date
            ]
            
            # Clean old log files
            log_file = Path(LOGGING_CONFIG['log_file'])
            if log_file.exists() and log_file.stat().st_size > SCHEDULER_CONFIG['max_log_size_mb'] * 1024 * 1024:
                # Rotate log file
                backup_file = log_file.with_suffix(f'.{datetime.now().strftime("%Y%m%d")}.log')
                log_file.rename(backup_file)
                logger.info(f"Log file rotated to {backup_file}")
            
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
    
    def send_daily_summary(self):
        """Send daily trading summary"""
        try:
            if not self.signals_today:
                logger.info("No signals today, skipping daily summary")
                return
            
            logger.info("Sending daily summary...")
            
            stats = {
                'total_signals': len(self.signals_today),
                'avg_confidence': sum(s['confidence'] for s in self.signals_today) / len(self.signals_today),
                'timeframes': list(set(s['timeframe'] for s in self.signals_today))
            }
            
            telegram_notifier.send_daily_summary(self.signals_today, stats)
            logger.info("Daily summary sent")
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
    
    def _save_signal(self, signal: Dict):
        """Save signal to JSON file"""
        try:
            signals_file = self.data_dir / 'signals.json'
            
            # Load existing signals
            signals = []
            if signals_file.exists():
                with open(signals_file, 'r') as f:
                    signals = json.load(f)
            
            # Add new signal
            signal_data = signal.copy()
            signal_data['timestamp'] = signal['timestamp'].isoformat()
            signals.append(signal_data)
            
            # Keep only last 1000 signals
            signals = signals[-1000:]
            
            # Save back to file
            with open(signals_file, 'w') as f:
                json.dump(signals, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
    
    def _update_daily_stats(self):
        """Update daily statistics"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            self.daily_stats[today] = {
                'signals_count': len(self.signals_today),
                'long_signals': len([s for s in self.signals_today if s['signal_type'] == 'LONG']),
                'short_signals': len([s for s in self.signals_today if s['signal_type'] == 'SHORT']),
                'avg_confidence': sum(s['confidence'] for s in self.signals_today) / len(self.signals_today) if self.signals_today else 0,
                'symbols_traded': list(set(s['symbol'] for s in self.signals_today)),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error updating daily stats: {e}")
    
    def get_status(self) -> Dict:
        """Get current bot status"""
        return {
            'is_running': self.is_running,
            'is_shutdown_period': self.is_shutdown_period,
            'start_time': self.start_time.isoformat(),
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'signals_today': len(self.signals_today),
            'last_health_check': self.last_health_check.isoformat(),
            'daily_stats': self.daily_stats,
            'scheduler_jobs': len(self.scheduler.get_jobs()),
            'next_quick_scan': self.scheduler.get_job('quick_scan').next_run_time.isoformat() if self.scheduler.get_job('quick_scan') else None,
            'next_full_scan': self.scheduler.get_job('full_scan').next_run_time.isoformat() if self.scheduler.get_job('full_scan') else None
        }
    
    def start(self):
        """Start the trading bot"""
        try:
            logger.info("Starting ProTradeAI Pro+ Bot...")
            
            # Validate configuration
            if not self.validate_configuration():
                logger.error("Configuration validation failed")
                return False
            
            # Setup scheduler jobs
            logger.info("Setting up scheduler...")
            
            # Quick market scan (every 5 minutes)
            self.scheduler.add_job(
                func=self.quick_market_scan,
                trigger=IntervalTrigger(minutes=5),
                id='quick_scan',
                name='Quick Market Scanner',
                max_instances=1,
                replace_existing=True
            )
            
            # Full market scan (every 15 minutes with cron)
            self.scheduler.add_job(
                func=self.full_market_scan,
                trigger=CronTrigger(minute='*/15'),  # Cron: every 15 minutes
                id='full_scan',
                name='Full Market Scanner',
                max_instances=1,
                replace_existing=True
            )
            
            # Shutdown status check (every 5 minutes)
            self.scheduler.add_job(
                func=self.check_shutdown_status,
                trigger=IntervalTrigger(minutes=5),
                id='shutdown_check',
                name='Shutdown Status Check',
                max_instances=1,
                replace_existing=True
            )
            
            # Health check (every 10 minutes)
            self.scheduler.add_job(
                func=self.health_check,
                trigger=IntervalTrigger(minutes=10),
                id='health_check',
                name='Health Check',
                max_instances=1,
                replace_existing=True
            )
            
            # Cleanup (every hour)
            self.scheduler.add_job(
                func=self.cleanup_old_data,
                trigger=IntervalTrigger(hours=1),
                id='cleanup',
                name='Data Cleanup',
                max_instances=1,
                replace_existing=True
            )
            
            # Daily summary (at 23:30 IST)
            self.scheduler.add_job(
                func=self.send_daily_summary,
                trigger=CronTrigger(hour=23, minute=30, timezone=self.timezone),
                id='daily_summary',
                name='Daily Summary',
                max_instances=1,
                replace_existing=True
            )
            
            # Start scheduler
            self.scheduler.start()
            self.is_running = True
            
            # Check initial shutdown status
            self.check_shutdown_status()
            
            logger.info("✅ ProTradeAI Pro+ Bot started successfully!")
            logger.info(f"📊 Monitoring {len(SYMBOLS)} symbols on {len(TIMEFRAMES)} timeframes")
            logger.info("⚡ Quick scans every 5 minutes")
            logger.info("🔍 Full scans every 15 minutes (cron)")
            logger.info("🌙 Auto shutdown: 1-5 AM IST")
            logger.info(f"💰 Risk per trade: {RISK_PER_TRADE*100:.1f}% of ${CAPITAL:,.2f}")
            
            # Send startup notification
            current_status = "🌙 Shutdown Period" if self.is_shutdown_period else "🚀 Active Trading"
            telegram_notifier.send_message(
                f"🚀 <b>ProTradeAI Pro+ Started</b>\n\n"
                f"✅ Bot is now running\n"
                f"📊 Monitoring {len(SYMBOLS)} symbols\n"
                f"⚡ Quick scans: Every 5 minutes\n"
                f"🔍 Full scans: Every 15 minutes\n"
                f"🌙 Auto shutdown: 1-5 AM IST\n"
                f"💰 Capital: ${CAPITAL:,.2f}\n"
                f"🎯 Risk per trade: {RISK_PER_TRADE*100:.1f}%\n"
                f"📈 Status: {current_status}\n\n"
                f"<i>Ready to generate trading signals!</i>"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            return False
    
    def stop(self):
        """Stop the trading bot"""
        try:
            logger.info("Stopping ProTradeAI Pro+ Bot...")
            
            if self.scheduler.running:
                self.scheduler.shutdown(wait=True)
            
            self.is_running = False
            
            # Send shutdown notification
            uptime = datetime.now() - self.start_time
            telegram_notifier.send_message(
                f"🛑 <b>ProTradeAI Pro+ Stopped</b>\n\n"
                f"⏰ Uptime: {uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m\n"
                f"📊 Signals today: {len(self.signals_today)}\n"
                f"💼 Sessions completed: {len(self.daily_stats)}\n\n"
                f"<i>Bot has been shut down gracefully.</i>"
            )
            
            logger.info("✅ ProTradeAI Pro+ Bot stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
    
    def run_forever(self):
        """Run the bot indefinitely"""
        if not self.start():
            return
        
        try:
            logger.info("Bot is running... Press Ctrl+C to stop")
            
            # Keep the main thread alive
            while self.is_running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()

    def debug_no_signals_enhanced(self):
        """Enhanced debug for both trending and sideways signals"""
        logger.info("🔍 Starting enhanced signal generation debug (trending + sideways)...")
        
        try:
            # Get debug info from strategy
            debug_info = strategy_ai.debug_signal_generation()
            
            logger.info("🤖 Enhanced Model Status:")
            logger.info(f"  - Model loaded: {debug_info['model_loaded']}")
            logger.info(f"  - Feature columns: {debug_info['feature_columns']}")
            logger.info(f"  - Market volatility: {debug_info['market_volatility']:.4f}")
            
            # Test both trending and sideways approaches
            logger.info("🔧 Testing Dual Strategy Approach...")
            
            test_symbols = ['BTCUSDT', 'ETHUSDT']
            for symbol in test_symbols:
                try:
                    # Test market regime detection
                    df = strategy_ai.get_binance_data(symbol, '4h', limit=50)
                    if not df.empty:
                        df = strategy_ai.calculate_technical_indicators(df)
                        regime = strategy_ai.detect_market_regime(df)
                        
                        logger.info(f"📊 {symbol} Market Analysis:")
                        logger.info(f"   - Market regime: {regime}")
                        
                        # Test trending signal
                        trending_signal = strategy_ai.predict_signal(symbol, '4h')
                        trending_result = f"Generated {trending_signal['signal_type']}" if trending_signal else "No signal"
                        logger.info(f"   - Trending approach: {trending_result}")
                        
                        # Test sideways signal
                        sideways_signal = strategy_ai.generate_sideways_signal(symbol, '4h', df)
                        sideways_result = f"Generated {sideways_signal['signal_type']}" if sideways_signal else "No signal"
                        logger.info(f"   - Sideways approach: {sideways_result}")
                        
                        # Test enhanced (combined) signal
                        enhanced_signal = strategy_ai.predict_signal_enhanced(symbol, '4h')
                        enhanced_result = f"Generated {enhanced_signal['signal_type']}" if enhanced_signal else "No signal"
                        logger.info(f"   - Enhanced approach: {enhanced_result}")
                        
                except Exception as e:
                    logger.error(f"Error in enhanced debug for {symbol}: {e}")
            
            # Market condition summary
            market_volatility = strategy_ai.get_market_volatility()
            if market_volatility < 0.03:
                logger.info("💡 LOW VOLATILITY DETECTED:")
                logger.info("   - Trending signals will be rare (expected)")
                logger.info("   - Sideways signals should be more frequent")
                logger.info("   - This is normal market behavior")
            else:
                logger.info("📈 NORMAL VOLATILITY:")
                logger.info("   - Both trending and sideways signals possible")
                logger.info("   - Mixed strategy approach active")
            
        except Exception as e:
            logger.error(f"Error in enhanced debug process: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    def get_enhanced_status(self) -> Dict:
        """Get enhanced bot status with sideways trading info"""
        base_status = self.get_status()
        
        try:
            # Add sideways trading specific status
            market_volatility = strategy_ai.get_market_volatility()
            
            # Count recent signal types
            today_signals = [s for s in self.signals_today if s['timestamp'] > datetime.now() - timedelta(hours=24)]
            trending_count = len([s for s in today_signals if s.get('market_regime') != 'SIDEWAYS'])
            sideways_count = len([s for s in today_signals if s.get('market_regime') == 'SIDEWAYS'])
            
            enhanced_status = {
                **base_status,
                'market_volatility': market_volatility,
                'volatility_regime': 'LOW' if market_volatility < 0.03 else 'NORMAL',
                'trading_mode': 'SIDEWAYS_FOCUSED' if market_volatility < 0.03 else 'MIXED',
                'signals_today_by_type': {
                    'trending': trending_count,
                    'sideways': sideways_count,
                    'total': len(today_signals)
                },
                'sideways_features_enabled': hasattr(strategy_ai, 'detect_market_regime'),
                'enhanced_prediction_active': hasattr(strategy_ai, 'predict_signal_enhanced')
            }
            
            return enhanced_status
            
        except Exception as e:
            logger.error(f"Error getting enhanced status: {e}")
            return base_status
def main():
    """Main entry point"""
    print("🤖 ProTradeAI Pro+ Trading Bot v2.0")
    print("=" * 50)
    
    # Create and run bot
    bot = ProTradeAIBot()
    
    # Run based on command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            # Test mode - run one scan
            print("🧪 Running in test mode...")
            if bot.validate_configuration():
                bot.quick_market_scan()
                print("✅ Test completed")
            else:
                print("❌ Configuration validation failed")
                
        elif command == 'status':
            # Status check
            print("📊 Bot Status:")
            status = bot.get_status()
            for key, value in status.items():
                print(f"  {key}: {value}")
                
        elif command == 'scan':
            # Manual full scan
            print("🔍 Running manual full scan...")
            if bot.validate_configuration():
                bot.full_market_scan()
                print("✅ Scan completed")
            else:
                print("❌ Configuration validation failed")
                
        else:
            print(f"Unknown command: {command}")
            print("Available commands: test, status, scan")
    else:
        # Normal operation
        bot.run_forever()

if __name__ == "__main__":
    main()