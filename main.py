"""
ProTradeAI Pro+ Main Application - FIXED VERSION
Stops continuous restart notifications while maintaining full compatibility
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

from config import (
    MARKET_CONDITIONS,
    SYMBOLS,
    TIMEFRAMES,
    RISK_PER_TRADE,
    CAPITAL,
    LOGGING_CONFIG,
    SCHEDULER_CONFIG,
    validate_config,
    EMERGENCY_MODE,
)
from strategy_ai import strategy_ai
from notifier import telegram_notifier

# Setup logging
def setup_logging():
    """Setup comprehensive logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format=LOGGING_CONFIG["format"],
        handlers=[
            logging.FileHandler(LOGGING_CONFIG["log_file"]),
            logging.StreamHandler(sys.stdout),
        ],
    )

setup_logging()
logger = logging.getLogger(__name__)

class SimplePerformanceTracker:
    """Simple performance tracking for compatibility with existing code"""

    def __init__(self):
        self.signals_sent = []
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)
        self.performance_file = self.data_dir / "simple_performance.json"
        self.load_data()

    def track_signal(self, signal: Dict):
        """Track a signal that was sent"""
        try:
            signal_record = {
                "symbol": signal["symbol"],
                "timeframe": signal["timeframe"],
                "signal_type": signal["signal_type"],
                "confidence": signal["confidence"],
                "leverage": signal["leverage"],
                "timestamp": signal["timestamp"].isoformat(),
                "entry_price": signal["current_price"],
            }

            self.signals_sent.append(signal_record)
            self.save_data()

            logger.info(f"📊 Tracked signal: {signal['symbol']} {signal['signal_type']}")

        except Exception as e:
            logger.error(f"Error tracking signal: {e}")

    def get_today_stats(self) -> Dict:
        """Get today's statistics"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            today_signals = [
                s for s in self.signals_sent if s["timestamp"].startswith(today)
            ]

            if not today_signals:
                return {
                    "total_signals": 0,
                    "long_signals": 0,
                    "short_signals": 0,
                    "avg_confidence": 0,
                    "symbols_active": 0,
                }

            return {
                "total_signals": len(today_signals),
                "long_signals": len(
                    [s for s in today_signals if s["signal_type"] == "LONG"]
                ),
                "short_signals": len(
                    [s for s in today_signals if s["signal_type"] == "SHORT"]
                ),
                "avg_confidence": sum(s["confidence"] for s in today_signals)
                / len(today_signals),
                "symbols_active": len(set(s["symbol"] for s in today_signals)),
            }

        except Exception as e:
            logger.error(f"Error getting today stats: {e}")
            return {
                "total_signals": 0,
                "long_signals": 0,
                "short_signals": 0,
                "avg_confidence": 0,
                "symbols_active": 0,
            }

    def get_week_stats(self) -> Dict:
        """Get this week's statistics"""
        try:
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            week_signals = [s for s in self.signals_sent if s["timestamp"] >= week_ago]

            if not week_signals:
                return {"total_signals": 0, "avg_confidence": 0}

            return {
                "total_signals": len(week_signals),
                "avg_confidence": sum(s["confidence"] for s in week_signals)
                / len(week_signals),
            }

        except Exception as e:
            logger.error(f"Error getting week stats: {e}")
            return {"total_signals": 0, "avg_confidence": 0}

    def save_data(self):
        """Save tracking data"""
        try:
            data = {
                "signals_sent": self.signals_sent[-500:],
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.performance_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving performance data: {e}")

    def load_data(self):
        """Load tracking data"""
        try:
            if self.performance_file.exists():
                with open(self.performance_file, "r") as f:
                    data = json.load(f)
                    self.signals_sent = data.get("signals_sent", [])

                logger.info(f"📊 Loaded {len(self.signals_sent)} tracked signals")
            else:
                self.signals_sent = []

        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            self.signals_sent = []

class ProTradeAIBot:
    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self.is_running = False
        self.is_shutdown_period = False
        self.start_time = datetime.now()
        self.signals_today = []
        self.daily_stats = {}
        self.last_health_check = datetime.now()

        # 🔧 FIX: Prevent notification spam
        self.notification_state_file = Path("data/notification_state.json")
        self.load_notification_state()

        self.emergency_mode = EMERGENCY_MODE.get('enabled', False)
        self.scan_count = 0
        self.last_signal_time = None

        # System monitoring
        self.system_errors = 0
        self.last_system_check = datetime.now()
        self.memory_alerts_sent = 0

        # Performance tracking
        self.tracker = SimplePerformanceTracker()

        # Timezone for shutdown schedule (IST)
        self.timezone = pytz.timezone("Asia/Kolkata")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Initialize data storage
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

        if self.emergency_mode:
            logger.warning("🚨 EMERGENCY MODE ACTIVE - Aggressive scanning enabled")

    def load_notification_state(self):
        """🔧 FIX: Load notification state to prevent spam"""
        try:
            if self.notification_state_file.exists():
                with open(self.notification_state_file, 'r') as f:
                    state = json.load(f)
                    self.last_startup_notification = state.get('last_startup_notification')
                    self.last_shutdown_notification = state.get('last_shutdown_notification')
                    self.last_resume_notification = state.get('last_resume_notification')
                    self.startup_count_today = state.get('startup_count_today', 0)
                    self.last_startup_date = state.get('last_startup_date')
            else:
                self.last_startup_notification = None
                self.last_shutdown_notification = None
                self.last_resume_notification = None
                self.startup_count_today = 0
                self.last_startup_date = None
        except Exception as e:
            logger.error(f"Error loading notification state: {e}")
            self.last_startup_notification = None
            self.last_shutdown_notification = None
            self.last_resume_notification = None
            self.startup_count_today = 0
            self.last_startup_date = None

    def save_notification_state(self):
        """🔧 FIX: Save notification state"""
        try:
            state = {
                'last_startup_notification': self.last_startup_notification,
                'last_shutdown_notification': self.last_shutdown_notification,
                'last_resume_notification': self.last_resume_notification,
                'startup_count_today': self.startup_count_today,
                'last_startup_date': self.last_startup_date
            }
            with open(self.notification_state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving notification state: {e}")

    def should_send_startup_notification(self) -> bool:
        """🔧 FIX: Check if we should send startup notification"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Reset daily count if new day
            if self.last_startup_date != today:
                self.startup_count_today = 0
                self.last_startup_date = today
            
            # Don't spam - max 3 startup notifications per day
            if self.startup_count_today >= 3:
                return False
            
            # Don't send if we sent one in the last 30 minutes
            if self.last_startup_notification:
                last_time = datetime.fromisoformat(self.last_startup_notification)
                if datetime.now() - last_time < timedelta(minutes=30):
                    return False
            
            return True
        except Exception:
            return True  # Default to allowing if check fails

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
        sys.exit(0)

    def is_shutdown_time(self) -> bool:
        """Check if current time is in shutdown period"""
        if self.emergency_mode:
            return False

        try:
            current_time = datetime.now(self.timezone)
            current_hour = current_time.hour
            return 1 <= current_hour < 5
        except Exception as e:
            logger.error(f"Error checking shutdown time: {e}")
            return False

    def check_shutdown_status(self):
        """🔧 FIX: Check shutdown status with reduced notifications"""
        try:
            should_shutdown = self.is_shutdown_time()

            if should_shutdown and not self.is_shutdown_period:
                logger.info("🌙 Entering maintenance shutdown period (1-5 AM IST)")
                self.is_shutdown_period = True

                # Pause trading jobs
                for job in self.scheduler.get_jobs():
                    if job.id in ["signal_scan", "quick_scan"]:
                        job.pause()
                        logger.info(f"Paused job: {job.id}")

                # 🔧 FIX: Only send shutdown notification once per shutdown period
                if (not self.last_shutdown_notification or 
                    datetime.now() - datetime.fromisoformat(self.last_shutdown_notification) > timedelta(hours=8)):
                    
                    today_stats = self.tracker.get_today_stats()
                    telegram_notifier.send_message(
                        "🌙 <b>Maintenance Period Started</b>\n\n"
                        "🔸 Signal scanning paused (1-5 AM IST)\n"
                        "🔸 System monitoring continues\n"
                        f"📊 Today's signals: {today_stats['total_signals']}\n"
                        f"📈 Avg confidence: {today_stats['avg_confidence']:.1f}%\n"
                        "🔸 Bot will resume automatically at 5 AM IST\n\n"
                        "<i>Good night! 😴</i>"
                    )
                    self.last_shutdown_notification = datetime.now().isoformat()
                    self.save_notification_state()

            elif not should_shutdown and self.is_shutdown_period:
                logger.info("☀️ Exiting maintenance shutdown period")
                self.is_shutdown_period = False

                # Resume trading jobs
                for job in self.scheduler.get_jobs():
                    if job.id in ["signal_scan", "quick_scan"]:
                        job.resume()
                        logger.info(f"Resumed job: {job.id}")

                # 🔧 FIX: Only send resume notification once per resume
                if (not self.last_resume_notification or 
                    datetime.now() - datetime.fromisoformat(self.last_resume_notification) > timedelta(hours=8)):
                    
                    telegram_notifier.send_message(
                        "☀️ <b>Trading Resumed</b>\n\n"
                        "✅ Signal scanning reactivated\n"
                        "🔸 All systems operational\n"
                        "🔸 Ready for new trading opportunities\n\n"
                        "<i>Good morning! Let's trade! 🚀</i>"
                    )
                    self.last_resume_notification = datetime.now().isoformat()
                    self.save_notification_state()

        except Exception as e:
            logger.error(f"Error in shutdown status check: {e}")

    def validate_configuration(self) -> bool:
        """🔧 FIX: Validate config without testing Telegram every time"""
        logger.info("Validating configuration...")

        # Check config
        errors = validate_config()
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False

        # 🔧 FIX: Only test Telegram connection on first validation of the day
        test_telegram = True
        if hasattr(self, 'last_telegram_test'):
            last_test = datetime.fromisoformat(self.last_telegram_test)
            if datetime.now() - last_test < timedelta(hours=6):
                test_telegram = False

        if test_telegram:
            logger.info("Testing Telegram connection...")
            if telegram_notifier.test_connection():
                self.last_telegram_test = datetime.now().isoformat()
            else:
                logger.error("Telegram connection test failed")
                return False

        # Check AI model
        logger.info("Checking AI model...")
        try:
            model_info = strategy_ai.get_model_info()
            logger.info(
                f"Model loaded: {model_info['model_type']} with {model_info['feature_count']} features"
            )
        except Exception as e:
            logger.error(f"Error checking model: {e}")
            return False

        logger.info("Configuration validation successful")
        return True

    def get_simple_market_volatility(self) -> float:
        """Get simple market volatility estimate"""
        try:
            df = strategy_ai.get_binance_data("BTCUSDT", "1d", limit=7)
            if df.empty or len(df) < 3:
                return 0.05

            returns = df["close"].pct_change().dropna()
            volatility = returns.std()
            return volatility

        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.05

    def quick_market_scan(self):
        """🔧 FIX: Enhanced market scan with reduced logging"""
        try:
            if self.is_shutdown_period:
                return

            self.scan_count += 1

            # 🔧 FIX: Much less verbose logging
            if self.scan_count % 20 == 0:  # Log every 20th scan instead of every scan
                logger.info(f"🔍 Quick scan #{self.scan_count}")

            # Check market conditions
            market_volatility = self.get_simple_market_volatility()

            if market_volatility < MARKET_CONDITIONS["low_volatility_threshold"]:
                scan_symbols = SYMBOLS[:5]
                scan_timeframes = ["4h", "1h"]
            else:
                scan_symbols = SYMBOLS[:3]
                scan_timeframes = ["1h", "4h"]

            signals = []

            for symbol in scan_symbols:
                for timeframe in scan_timeframes:
                    try:
                        signal = strategy_ai.predict_signal(symbol, timeframe)
                        if signal:
                            signals.append(signal)
                            logger.info(
                                f"✅ Signal: {signal['symbol']} {signal['signal_type']} {signal['confidence']:.1f}%"
                            )
                    except Exception as e:
                        logger.error(f"Error scanning {symbol} {timeframe}: {e}")

            # Process signals
            if signals:
                logger.info(f"🎯 Market scan generated {len(signals)} signals")
                for signal in signals:
                    self.process_signal(signal)

        except Exception as e:
            logger.error(f"Error in enhanced market scan: {e}")

    def full_market_scan(self):
        """Full market scan using existing scan_all_symbols function"""
        try:
            if self.is_shutdown_period:
                return

            logger.info("Starting full market scan...")

            signals = strategy_ai.scan_all_symbols()

            if not signals:
                logger.info("Full scan: No signals generated")
                return

            logger.info(f"Full scan generated {len(signals)} signals")

            for signal in signals:
                self.process_signal(signal)
                time.sleep(2)

            self._update_daily_stats()

        except Exception as e:
            logger.error(f"Error in full market scan: {e}")
            # 🔧 FIX: Only send error alert once per hour
            if (not hasattr(self, 'last_scan_error_alert') or 
                datetime.now() - self.last_scan_error_alert > timedelta(hours=1)):
                telegram_notifier.send_error_alert(str(e), "Full Market Scanner")
                self.last_scan_error_alert = datetime.now()

    def process_signal(self, signal: Dict):
        """Process and send a trading signal"""
        try:
            self.last_signal_time = datetime.now()
            self.signals_today.append(signal)

            success = telegram_notifier.send_signal_alert(signal)

            if success:
                logger.info(
                    f"Signal sent: {signal['symbol']} {signal['signal_type']} {signal['confidence']:.1f}%"
                )
                self.tracker.track_signal(signal)
                self._save_signal(signal)
            else:
                logger.error(f"Failed to send signal: {signal['symbol']}")

        except Exception as e:
            logger.error(f"Error processing signal {signal.get('symbol', 'Unknown')}: {e}")

    def enhanced_health_check(self):
        """🔧 IMPROVED: Enhanced health check with safe config imports"""
        try:
            # Safe import of psutil
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
            except ImportError:
                logger.warning("psutil not available, skipping system monitoring")
                return
            except Exception as e:
                logger.error(f"Error getting system stats: {e}")
                return

            # Safe import of config values with defaults
            try:
                from config import PRODUCTION_LIMITS, SYSTEM_MONITORING
                max_memory = PRODUCTION_LIMITS.get('max_memory_mb', 500)
                alert_hours = SYSTEM_MONITORING.get('alert_on_no_signals_hours', 6)
            except (ImportError, AttributeError):
                # Use defaults if config values don't exist
                max_memory = 500
                alert_hours = 6
                logger.debug("Using default monitoring thresholds")

            # Memory monitoring with safe thresholds
            if memory_mb > max_memory:
                if (not hasattr(self, 'last_memory_alert') or
                    datetime.now() - self.last_memory_alert > timedelta(hours=24)):
                    telegram_notifier.send_error_alert(
                        f"High memory usage: {memory_mb:.1f}MB (limit: {max_memory}MB)",
                        "System Monitor"
                    )
                    self.last_memory_alert = datetime.now()

            # CPU monitoring (optional alert)
            if cpu_percent > 85:
                logger.warning(f"High CPU usage: {cpu_percent:.1f}%")

            # Check for no signals alert - only once per 6 hours
            if hasattr(self, 'last_signal_time') and self.last_signal_time:
                hours_since_signal = (datetime.now() - self.last_signal_time).total_seconds() / 3600
                if (hours_since_signal > alert_hours and
                    (not hasattr(self, 'last_no_signal_alert') or
                     datetime.now() - self.last_no_signal_alert > timedelta(hours=6))):
                    telegram_notifier.send_error_alert(
                        f"No signals generated for {hours_since_signal:.1f} hours",
                        "Signal Generator"
                    )
                    self.last_no_signal_alert = datetime.now()

            # Log health status occasionally
            if self.scan_count % 120 == 0:  # Every 2 hours (120 scans * 1min)
                logger.info(f"Health check: Memory {memory_mb:.1f}MB, CPU {cpu_percent:.1f}%")

        except Exception as e:
            logger.error(f"Error in enhanced health check: {e}")

    def _save_signal(self, signal: Dict):
        """Save signal to JSON file"""
        try:
            signals_file = self.data_dir / "signals.json"

            signals = []
            if signals_file.exists():
                with open(signals_file, "r") as f:
                    signals = json.load(f)

            signal_data = signal.copy()
            signal_data["timestamp"] = signal["timestamp"].isoformat()
            signals.append(signal_data)

            signals = signals[-1000:]

            with open(signals_file, "w") as f:
                json.dump(signals, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving signal: {e}")

    def _update_daily_stats(self):
        """Update daily statistics"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")

            self.daily_stats[today] = {
                "signals_count": len(self.signals_today),
                "long_signals": len(
                    [s for s in self.signals_today if s["signal_type"] == "LONG"]
                ),
                "short_signals": len(
                    [s for s in self.signals_today if s["signal_type"] == "SHORT"]
                ),
                "avg_confidence": sum(s["confidence"] for s in self.signals_today)
                / len(self.signals_today)
                if self.signals_today
                else 0,
                "symbols_traded": list(set(s["symbol"] for s in self.signals_today)),
                "last_updated": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error updating daily stats: {e}")

    def cleanup_old_data(self):
        """Clean up old data and logs"""
        try:
            logger.info("Cleaning up old data...")

            cutoff_date = datetime.now() - timedelta(days=7)
            self.signals_today = [
                s for s in self.signals_today if s["timestamp"] > cutoff_date
            ]

            log_file = Path(LOGGING_CONFIG["log_file"])
            if (log_file.exists() and 
                log_file.stat().st_size > SCHEDULER_CONFIG["max_log_size_mb"] * 1024 * 1024):
                backup_file = log_file.with_suffix(f'.{datetime.now().strftime("%Y%m%d")}.log')
                log_file.rename(backup_file)
                logger.info(f"Log file rotated to {backup_file}")

        except Exception as e:
            logger.error(f"Error in cleanup: {e}")

    def send_daily_summary(self):
        """Send daily trading summary"""
        try:
            today_stats = self.tracker.get_today_stats()

            if today_stats["total_signals"] == 0:
                logger.info("No signals today, skipping daily summary")
                return

            logger.info("Sending daily summary...")
            telegram_notifier.send_daily_summary(self.signals_today, today_stats)

        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")

    def get_status(self) -> Dict:
        """Get current bot status"""
        today_stats = self.tracker.get_today_stats()
        week_stats = self.tracker.get_week_stats()

        return {
            "is_running": self.is_running,
            "is_shutdown_period": self.is_shutdown_period,
            "start_time": self.start_time.isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "signals_today": today_stats["total_signals"],
            "avg_confidence_today": today_stats["avg_confidence"],
            "signals_this_week": week_stats["total_signals"],
            "last_health_check": self.last_health_check.isoformat(),
            "daily_stats": self.daily_stats,
            "scheduler_jobs": len(self.scheduler.get_jobs()),
        }

    def start(self):
        """🔧 FIX: Start bot with controlled notifications"""
        try:
            logger.info("Starting ProTradeAI Pro+ Bot...")

            # Validate configuration
            if not self.validate_configuration():
                logger.error("Configuration validation failed")
                return False

            # Setup scheduler
            logger.info("Setting up scheduler...")

            # 🔧 FIX: More reasonable intervals for production
            quick_interval = 3 if self.emergency_mode else 5
            full_interval = 8 if self.emergency_mode else 15

            # Quick market scan
            self.scheduler.add_job(
                func=self.quick_market_scan,
                trigger=IntervalTrigger(minutes=quick_interval),
                id="quick_scan",
                name="Quick Market Scanner",
                max_instances=1,
                replace_existing=True,
            )

            # Full market scan
            self.scheduler.add_job(
                func=self.full_market_scan,
                trigger=IntervalTrigger(minutes=full_interval),
                id="full_scan",
                name="Full Market Scanner",
                max_instances=1,
                replace_existing=True,
            )

            # Shutdown check (less frequent)
            self.scheduler.add_job(
                func=self.check_shutdown_status,
                trigger=IntervalTrigger(minutes=15),
                id="shutdown_check",
                name="Shutdown Status Check",
                max_instances=1,
                replace_existing=True,
            )

            # Enhanced health check
            self.scheduler.add_job(
                func=self.enhanced_health_check,
                trigger=IntervalTrigger(minutes=30),  # Less frequent
                id="health_check",
                name="Enhanced Health Check",
                max_instances=1,
                replace_existing=True,
            )

            # Cleanup (every 2 hours)
            self.scheduler.add_job(
                func=self.cleanup_old_data,
                trigger=IntervalTrigger(hours=2),
                id="cleanup",
                name="Data Cleanup",
                max_instances=1,
                replace_existing=True,
            )

            # Daily summary
            self.scheduler.add_job(
                func=self.send_daily_summary,
                trigger=CronTrigger(hour=23, minute=30, timezone=self.timezone),
                id="daily_summary",
                name="Daily Summary",
                max_instances=1,
                replace_existing=True,
            )

            # Start scheduler
            self.scheduler.start()
            self.is_running = True

            # 🆕 Enable Telegram commands
            try:
                from notifier import enable_simple_commands
                enable_simple_commands(strategy_ai)
                logger.info("✅ Manual commands activated")
            except Exception as e:
                logger.warning(f"Commands not enabled: {e}")

            # Check initial shutdown status
            self.check_shutdown_status()

            logger.info("✅ ProTradeAI Pro+ Bot started successfully!")

            # 🔧 FIX: Only send startup notification if appropriate
            if self.should_send_startup_notification():
                today_stats = self.tracker.get_today_stats()
                current_status = "🌙 Shutdown Period" if self.is_shutdown_period else "🚀 Active Trading"

                telegram_notifier.send_message(
                    f"🚀 <b>ProTradeAI Pro+ Started</b>\n\n"
                    f"✅ Bot is now running with enhanced tracking\n"
                    f"📊 Monitoring {len(SYMBOLS)} symbols\n"
                    f"⚡ Quick scans: Every {quick_interval} minutes\n"
                    f"🔍 Full scans: Every {full_interval} minutes\n"
                    f"🌙 Auto shutdown: 1-5 AM IST\n"
                    f"💰 Capital: ${CAPITAL:,.2f}\n"
                    f"🎯 Risk per trade: {RISK_PER_TRADE*100:.1f}%\n"
                    f"📈 Status: {current_status}\n\n"
                    f"📊 <b>Today's Activity:</b>\n"
                    f"🔸 Signals: {today_stats['total_signals']}\n"
                    f"🔸 Avg Confidence: {today_stats['avg_confidence']:.1f}%\n\n"
                    f"<i>Ready to generate profitable signals! 🚀</i>"
                )
                
                self.startup_count_today += 1
                self.last_startup_notification = datetime.now().isoformat()
                self.save_notification_state()

            return True

        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            return False

    def stop(self):
        """🔧 FIX: Stop bot with controlled notifications"""
        try:
            logger.info("Stopping ProTradeAI Pro+ Bot...")

            # 🆕 Disable commands
            try:
                from notifier import disable_simple_commands
                disable_simple_commands()
            except Exception as e:
                logger.warning(f"Error disabling commands: {e}")

            if self.scheduler.running:
                self.scheduler.shutdown(wait=True)

            self.is_running = False

            # 🔧 FIX: Only send stop notification occasionally  
            uptime = datetime.now() - self.start_time
            today_stats = self.tracker.get_today_stats()

            # Don't spam stop notifications - only if uptime > 5 minutes
            if uptime > timedelta(minutes=5):
                telegram_notifier.send_message(
                    f"🛑 <b>ProTradeAI Pro+ Stopped</b>\n\n"
                    f"⏰ Uptime: {uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m\n"
                    f"📊 Signals today: {today_stats['total_signals']}\n"
                    f"📈 Avg confidence: {today_stats['avg_confidence']:.1f}%\n"
                    f"💼 Sessions completed: {len(self.daily_stats)}\n\n"
                    f"<i>Bot has been shut down gracefully with tracking saved.</i>"
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

            while self.is_running:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.stop()

def main():
    """Main entry point"""
    print("🤖 ProTradeAI Pro+ Trading Bot v2.0")
    print("=" * 50)

    bot = ProTradeAIBot()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "test":
            print("🧪 Running in test mode...")
            if bot.validate_configuration():
                bot.quick_market_scan()

                today_stats = bot.tracker.get_today_stats()
                week_stats = bot.tracker.get_week_stats()

                print("\n📊 Performance Stats:")
                print(f"   Today: {today_stats['total_signals']} signals")
                print(f"   Week: {week_stats['total_signals']} signals")
                print(f"   Avg Confidence: {today_stats['avg_confidence']:.1f}%")

                print("✅ Test completed successfully")
            else:
                print("❌ Configuration validation failed")

        elif command == "status":
            print("📊 Bot Status:")
            status = bot.get_status()
            print(f"   Running: {status['is_running']}")
            print(f"   Signals Today: {status['signals_today']}")
            print(f"   Signals This Week: {status['signals_this_week']}")
            print(f"   Avg Confidence: {status['avg_confidence_today']:.1f}%")

        elif command == "scan":
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
