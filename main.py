"""
ProTradeAI Pro+ Main Application - RESTART LOOP FIXED
Prevents continuous restart notifications and stabilizes bot operation
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
import gc
import psutil

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

class RestartLoopPrevention:
    """🔧 NEW: Prevents restart loops and duplicate notifications"""
    
    def __init__(self):
        self.state_file = Path('data/restart_state.json')
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)
        
        self.max_restarts_per_hour = 3
        self.notification_cooldown_minutes = 30
        
        self.load_state()
    
    def load_state(self):
        """Load restart tracking state"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self.restart_history = data.get('restart_history', [])
                    self.last_notification = data.get('last_notification')
                    self.startup_count_today = data.get('startup_count_today', 0)
                    self.last_date = data.get('last_date')
            else:
                self.restart_history = []
                self.last_notification = None
                self.startup_count_today = 0
                self.last_date = None
        except Exception as e:
            logger.error(f"Error loading restart state: {e}")
            self.restart_history = []
            self.last_notification = None
            self.startup_count_today = 0
            self.last_date = None
    
    def save_state(self):
        """Save restart tracking state"""
        try:
            data = {
                'restart_history': self.restart_history[-10:],  # Keep only last 10
                'last_notification': self.last_notification,
                'startup_count_today': self.startup_count_today,
                'last_date': self.last_date
            }
            with open(self.state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving restart state: {e}")
    
    def should_send_startup_notification(self) -> bool:
        """🔧 FIXED: Determine if startup notification should be sent"""
        try:
            now = datetime.now()
            today = now.strftime('%Y-%m-%d')
            current_time = now.isoformat()
            
            # Reset daily counter if new day
            if self.last_date != today:
                self.startup_count_today = 0
                self.last_date = today
                self.save_state()
            
            # Clean old restart history (only keep last hour)
            hour_ago = now - timedelta(hours=1)
            self.restart_history = [
                r for r in self.restart_history 
                if datetime.fromisoformat(r) > hour_ago
            ]
            
            # Add current restart
            self.restart_history.append(current_time)
            
            # Check if too many restarts
            if len(self.restart_history) > self.max_restarts_per_hour:
                logger.warning(f"🚨 TOO MANY RESTARTS: {len(self.restart_history)} in last hour")
                # Only send notification once about restart loop
                if self.startup_count_today == 0:
                    self.startup_count_today = 1
                    self.save_state()
                    return True
                return False
            
            # Check notification cooldown
            if self.last_notification:
                last_notif_time = datetime.fromisoformat(self.last_notification)
                if now - last_notif_time < timedelta(minutes=self.notification_cooldown_minutes):
                    logger.info(f"⏳ Startup notification on cooldown")
                    return False
            
            # Check daily limit
            if self.startup_count_today >= 5:  # Max 5 notifications per day
                logger.info(f"📵 Daily notification limit reached: {self.startup_count_today}")
                return False
            
            # All checks passed
            self.last_notification = current_time
            self.startup_count_today += 1
            self.save_state()
            return True
            
        except Exception as e:
            logger.error(f"Error checking startup notification: {e}")
            return False
    
    def is_restart_loop(self) -> bool:
        """Check if we're in a restart loop"""
        return len(self.restart_history) > self.max_restarts_per_hour

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
            
            # Limit memory usage
            if len(self.signals_sent) > 200:
                self.signals_sent = self.signals_sent[-150:]
            
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
                "signals_sent": self.signals_sent[-200:],  # Limit size
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

        # Initialize data storage FIRST
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

        # 🔧 NEW: Restart loop prevention
        self.restart_prevention = RestartLoopPrevention()

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

        if self.emergency_mode:
            logger.warning("🚨 EMERGENCY MODE ACTIVE - Aggressive scanning enabled")

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
        """🔧 SIMPLIFIED: Check shutdown with reduced notifications"""
        try:
            should_shutdown = self.is_shutdown_time()

            if should_shutdown and not self.is_shutdown_period:
                logger.info("🌙 Entering maintenance shutdown period (1-5 AM IST)")
                self.is_shutdown_period = True

                # Pause trading jobs
                for job in self.scheduler.get_jobs():
                    if job.id in ["signal_scan", "quick_scan"]:
                        job.pause()

                # Send simple shutdown message (once per shutdown)
                telegram_notifier.send_message(
                    "🌙 <b>Maintenance Period</b>\n\n"
                    "Signal scanning paused (1-5 AM IST)\n"
                    "Will resume automatically at 5 AM"
                )

            elif not should_shutdown and self.is_shutdown_period:
                logger.info("☀️ Exiting maintenance shutdown period")
                self.is_shutdown_period = False

                # Resume trading jobs
                for job in self.scheduler.get_jobs():
                    if job.id in ["signal_scan", "quick_scan"]:
                        job.resume()

                # Send simple resume message
                telegram_notifier.send_message(
                    "☀️ <b>Trading Resumed</b>\n\n"
                    "Signal scanning reactivated\n"
                    "Ready for trading opportunities!"
                )

        except Exception as e:
            logger.error(f"Error in shutdown status check: {e}")

    def validate_configuration(self) -> bool:
        """🔧 SIMPLIFIED: Quick validation without testing every time"""
        logger.info("Validating configuration...")

        # Check config
        errors = validate_config()
        if errors:
            for error in errors:
                logger.error(f"Configuration error: {error}")
            return False

        # Check AI model (don't test Telegram every time)
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
        """🔧 ENHANCED: Quick scan with better logging and error handling"""
        try:
            if self.is_shutdown_period:
                return

            self.scan_count += 1

            # Less verbose logging
            if self.scan_count % 30 == 0:  # Log every 30th scan
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
                        signal = strategy_ai.predict_signal(symbol, timeframe, bypass_cooldown=True)
                        if signal:
                            signals.append(signal)
                            logger.info(
                                f"✅ Signal: {signal['symbol']} {signal['signal_type']} {signal['confidence']:.1f}%"
                            )
                    except Exception as e:
                        logger.error(f"Error scanning {symbol} {timeframe}: {e}")
                        continue

            # Process signals
            if signals:
                logger.info(f"🎯 Quick scan generated {len(signals)} signals")
                for signal in signals:
                    self.process_signal(signal)

        except Exception as e:
            logger.error(f"Error in quick market scan: {e}")

    def full_market_scan(self):
        """🔧 ENHANCED: Full scan with better signal processing"""
        try:
            if self.is_shutdown_period:
                return

            logger.info("Starting full market scan...")

            signals = strategy_ai.scan_all_symbols()

            if not signals:
                logger.info("Full scan: No signals generated")
                return

            logger.info(f"Full scan generated {len(signals)} signals")

            # Process each signal to send detailed alerts
            for i, signal in enumerate(signals, 1):
                try:
                    logger.info(f"Processing signal {i}/{len(signals)}: {signal['symbol']} {signal['signal_type']} {signal['confidence']:.1f}%")
                    
                    # Send the detailed alert
                    success = telegram_notifier.send_signal_alert(signal)
                    
                    if success:
                        logger.info(f"✅ Alert sent: {signal['symbol']} {signal['signal_type']}")
                        # Track the signal
                        self.tracker.track_signal(signal)
                        self._save_signal(signal)
                    else:
                        logger.error(f"❌ Failed to send alert: {signal['symbol']}")
                    
                    # Delay between signals to avoid rate limits
                    if i < len(signals):
                        time.sleep(2)  # Reduced from 3 seconds
                        
                except Exception as e:
                    logger.error(f"Error processing signal {i}: {e}")
                    continue

            self._update_daily_stats()
            
            # Send summary after all alerts
            if len(signals) > 0:
                telegram_notifier.send_message(
                    f"📊 <b>Scan Summary</b>\n\n"
                    f"✅ {len(signals)} alerts sent\n"
                    f"🎯 Highest confidence: {max(s['confidence'] for s in signals):.1f}%\n"
                    f"💰 Opportunities identified\n\n"
                    f"<i>Check above for details</i>"
                )

        except Exception as e:
            logger.error(f"Error in full market scan: {e}")

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
        """🔧 SIMPLIFIED: Basic health check with memory monitoring"""
        try:
            # Memory check
            try:
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                if memory_mb > 400:  # 400MB limit for Replit
                    logger.warning(f"High memory usage: {memory_mb:.1f}MB")
                    # Force garbage collection
                    gc.collect()
                    
                    # Clear old data if memory still high
                    if psutil.Process().memory_info().rss / 1024 / 1024 > 400:
                        logger.warning("Clearing old signal data to reduce memory")
                        self.signals_today = self.signals_today[-50:]  # Keep only recent
                        self.tracker.signals_sent = self.tracker.signals_sent[-100:]
                        gc.collect()
                        
            except ImportError:
                pass  # Skip if psutil not available

            # Restart loop check
            if self.restart_prevention.is_restart_loop():
                logger.error("🚨 RESTART LOOP DETECTED - Sending alert")
                telegram_notifier.send_message(
                    "🚨 <b>RESTART LOOP DETECTED</b>\n\n"
                    "Bot is restarting too frequently.\n"
                    "Please check logs for errors.\n\n"
                    "Common causes:\n"
                    "• Memory limit exceeded\n"
                    "• Configuration errors\n"
                    "• API rate limiting\n"
                    "• Telegram connection issues"
                )

            # Periodic health log
            if self.scan_count % 120 == 0:  # Every 2 hours
                logger.info(f"Health: {self.scan_count} scans, memory: {psutil.Process().memory_info().rss / 1024 / 1024:.1f}MB")

        except Exception as e:
            logger.error(f"Error in health check: {e}")

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

            # Limit file size
            signals = signals[-500:]

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

            # Force garbage collection
            gc.collect()

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
            "restart_prevention": {
                "restart_count": len(self.restart_prevention.restart_history),
                "is_restart_loop": self.restart_prevention.is_restart_loop(),
                "startup_count_today": self.restart_prevention.startup_count_today
            }
        }

    def start(self):
        """🔧 COMPLETELY FIXED: Start bot with proper restart loop prevention"""
        try:
            logger.info("Starting ProTradeAI Pro+ Bot...")

            # Check if we're in a restart loop
            if self.restart_prevention.is_restart_loop():
                logger.error("🚨 RESTART LOOP DETECTED - Delaying startup")
                time.sleep(30)  # Wait 30 seconds before continuing

            # Validate configuration
            if not self.validate_configuration():
                logger.error("Configuration validation failed")
                return False

            # Setup scheduler
            logger.info("Setting up scheduler...")

            # Production intervals
            quick_interval = 5
            full_interval = 15

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

            # Shutdown check
            self.scheduler.add_job(
                func=self.check_shutdown_status,
                trigger=IntervalTrigger(minutes=30),  # Less frequent
                id="shutdown_check",
                name="Shutdown Status Check",
                max_instances=1,
                replace_existing=True,
            )

            # Health check
            self.scheduler.add_job(
                func=self.enhanced_health_check,
                trigger=IntervalTrigger(minutes=60),  # Much less frequent
                id="health_check",
                name="Health Check",
                max_instances=1,
                replace_existing=True,
            )

            # Cleanup (every 6 hours)
            self.scheduler.add_job(
                func=self.cleanup_old_data,
                trigger=IntervalTrigger(hours=6),
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

            # Enable Telegram commands (with error handling)
            try:
                from notifier import enable_simple_commands
                enable_simple_commands(strategy_ai)
                logger.info("✅ Commands activated")
            except Exception as e:
                logger.warning(f"Commands not enabled: {e}")

            # Check initial shutdown status
            self.check_shutdown_status()

            logger.info("✅ ProTradeAI Pro+ Bot started successfully!")

            # 🔧 FIXED: Only send notification if needed
            if self.restart_prevention.should_send_startup_notification():
                today_stats = self.tracker.get_today_stats()
                current_status = "🌙 Shutdown Period" if self.is_shutdown_period else "🚀 Active Trading"

                # Check for restart loop warning
                restart_warning = ""
                if self.restart_prevention.is_restart_loop():
                    restart_warning = "\n⚠️ <b>Restart loop detected</b> - monitoring stability"

                telegram_notifier.send_message(
                    f"🚀 <b>ProTradeAI Pro+ Started</b>\n\n"
                    f"✅ Bot running with enhanced tracking\n"
                    f"📊 Monitoring {len(SYMBOLS)} symbols\n"
                    f"⚡ Quick scans: Every {quick_interval} min\n"
                    f"🔍 Full scans: Every {full_interval} min\n"
                    f"💰 Capital: ${CAPITAL:,.2f}\n"
                    f"🎯 Risk: {RISK_PER_TRADE*100:.1f}% per trade\n"
                    f"📈 Status: {current_status}{restart_warning}\n\n"
                    f"<i>Ready to generate signals! 🚀</i>"
                )
            else:
                logger.info("🔕 Startup notification skipped (cooldown/limit)")

            return True

        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            return False

    def stop(self):
        """🔧 SIMPLIFIED: Stop bot without excessive notifications"""
        try:
            logger.info("Stopping ProTradeAI Pro+ Bot...")

            # Disable commands
            try:
                from notifier import disable_simple_commands
                disable_simple_commands()
            except Exception as e:
                logger.warning(f"Error disabling commands: {e}")

            if self.scheduler.running:
                self.scheduler.shutdown(wait=True)

            self.is_running = False

            # Simple stop message (only for significant uptimes)
            uptime = datetime.now() - self.start_time
            if uptime > timedelta(minutes=10):  # Only if running > 10 minutes
                today_stats = self.tracker.get_today_stats()
                telegram_notifier.send_message(
                    f"🛑 <b>Bot Stopped</b>\n\n"
                    f"⏰ Uptime: {uptime.seconds//3600}h {(uptime.seconds//60)%60}m\n"
                    f"📊 Signals today: {today_stats['total_signals']}\n"
                    f"<i>Bot shutdown gracefully</i>"
                )

            logger.info("✅ ProTradeAI Pro+ Bot stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping bot: {e}")

    def run_forever(self):
        """Run the bot indefinitely with restart loop protection"""
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
    print("🤖 ProTradeAI Pro+ Trading Bot v2.0 (RESTART LOOP FIXED)")
    print("=" * 60)

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
            print(f"   Restart Count: {status['restart_prevention']['restart_count']}")
            print(f"   Restart Loop: {status['restart_prevention']['is_restart_loop']}")

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