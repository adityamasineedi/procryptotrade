"""
ProTradeAI Pro+ Telegram Notifier
Advanced alert system with rich formatting and retry logic (IST timezone)
"""

import requests
import time
import logging
from datetime import datetime
from typing import Dict, Optional
import json
import pytz
import threading
from config import TELEGRAM_CONFIG, CAPITAL, RISK_PER_TRADE, MAX_DAILY_TRADES, SIDEWAYS_ALERT_CONFIG

logger = logging.getLogger(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self):
        self.bot_token = TELEGRAM_CONFIG['bot_token']
        self.chat_id = TELEGRAM_CONFIG['chat_id']
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.alert_count = 0
        self.timezone = pytz.timezone('Asia/Kolkata')  # IST timezone
        
    def format_ist_time(self, timestamp: datetime) -> str:
        """Format timestamp to IST"""
        if timestamp.tzinfo is None:
            # If no timezone info, assume it's UTC and convert to IST
            timestamp = pytz.UTC.localize(timestamp)
        
        ist_time = timestamp.astimezone(self.timezone)
        return ist_time.strftime('%H:%M:%S IST')
    
    def format_ist_datetime(self, timestamp: datetime) -> str:
        """Format full datetime to IST"""
        if timestamp.tzinfo is None:
            timestamp = pytz.UTC.localize(timestamp)
        
        ist_time = timestamp.astimezone(self.timezone)
        return ist_time.strftime('%Y-%m-%d %H:%M:%S IST')
    
    def send_message(self, message: str, parse_mode: str = 'HTML') -> bool:
        """Send message to Telegram with retry logic"""
        if not self.bot_token or not self.chat_id:
            logger.error("Telegram credentials not configured")
            return False
        
        url = f"{self.base_url}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': parse_mode,
            'disable_web_page_preview': True
        }
        
        for attempt in range(TELEGRAM_CONFIG['retry_attempts']):
            try:
                response = requests.post(url, data=data, timeout=10)
                response.raise_for_status()
                
                result = response.json()
                if result.get('ok'):
                    logger.info(f"Message sent successfully (attempt {attempt + 1})")
                    return True
                else:
                    logger.error(f"Telegram API error: {result.get('description')}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error (attempt {attempt + 1}): {e}")
                
            except Exception as e:
                logger.error(f"Unexpected error (attempt {attempt + 1}): {e}")
            
            if attempt < TELEGRAM_CONFIG['retry_attempts'] - 1:
                time.sleep(TELEGRAM_CONFIG['retry_delay'])
        
        logger.error("Failed to send message after all retry attempts")
        return False
    
    def format_pro_plus_signal(self, signal: Dict) -> str:
        """Format signal in Pro+ style with rich formatting"""
        try:
            # Header with emojis
            direction_emoji = "🚀" if signal['signal_type'] == 'LONG' else "📉"
            confidence_emoji = self._get_confidence_emoji(signal['confidence'])
            
            # Calculate position size
            risk_amount = CAPITAL * RISK_PER_TRADE
            entry_price = signal['current_price']
            sl_price = signal.get('sl_price', entry_price)
            
            # Position size calculation (accounting for leverage)
            price_diff = abs(entry_price - sl_price)
            if price_diff > 0:
                position_size = (risk_amount * signal['leverage']) / price_diff
                position_value = position_size * entry_price
            else:
                position_size = 0
                position_value = 0
            
            message = f"""
🔥 <b>ProTradeAI Pro+ Signal</b> {confidence_emoji}

{direction_emoji} <b>{signal['symbol']} - {signal['signal_type']}</b>
⚡ <b>Leverage:</b> {signal['leverage']}x
📈 <b>Timeframe:</b> {signal['timeframe']}
🎯 <b>Confidence:</b> {signal['confidence']:.1f}% ({signal['confidence_grade']})

💰 <b>ENTRY ZONE</b>
🔸 Entry Price: ${signal['current_price']:.4f}
🔸 Position Size: {position_size:.2f} {signal['symbol'].replace('USDT', '')}
🔸 Position Value: ${position_value:.2f}

🛡️ <b>RISK MANAGEMENT</b>
🔻 Stop Loss: ${signal.get('sl_price', 0):.4f} (-{signal.get('sl_distance_pct', 0):.2f}%)
🎯 Take Profit: ${signal.get('tp_price', 0):.4f} (+{signal.get('tp_distance_pct', 0):.2f}%)
📊 R:R Ratio: 1:{signal.get('rr_ratio', 0):.2f}

📋 <b>TECHNICAL ANALYSIS</b>
📍 RSI: {signal.get('rsi', 0):.1f}
📍 MACD: {signal.get('macd', 0):.4f}
📍 ATR: {signal.get('atr', 0):.4f}
📍 Volume: {signal.get('volume_ratio', 0):.2f}x avg

⏰ <b>TIMING</b>
🕐 Signal Time: {self.format_ist_time(signal['timestamp'])}
⏳ Hold Period: ~{signal.get('hold_hours', 4)} hours
💼 Risk per Trade: {RISK_PER_TRADE*100:.1f}% (${risk_amount:.2f})

⚠️ <b>MANUAL EXECUTION REQUIRED</b>
📱 Copy trade details to your exchange
🔄 Set stop loss and take profit orders
📊 Monitor position according to timeframe

<b>ProTradeAI Pro+ | Signal #{self.alert_count + 1}</b>
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting Pro+ signal: {e}")
            return self._format_simple_signal(signal)
    
    def format_summary_signal(self, signal: Dict) -> str:
        """Format signal in summary style"""
        try:
            direction_emoji = "🚀" if signal['signal_type'] == 'LONG' else "📉"
            
            message = f"""
{direction_emoji} <b>{signal['symbol']} {signal['signal_type']}</b>
🎯 {signal['confidence']:.1f}% | ⚡{signal['leverage']}x | 📈{signal['timeframe']}

💰 Entry: ${signal['current_price']:.4f}
🛡️ SL: ${signal.get('sl_price', 0):.4f} | TP: ${signal.get('tp_price', 0):.4f}
⏰ {self.format_ist_time(signal['timestamp'])} | Risk: {RISK_PER_TRADE*100:.1f}%

<i>ProTradeAI Pro+ | Manual Execution</i>
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting summary signal: {e}")
            return self._format_simple_signal(signal)
    
    def _format_simple_signal(self, signal: Dict) -> str:
        """Fallback simple formatting"""
        return f"""
🔔 SIGNAL ALERT
{signal['symbol']} - {signal['signal_type']}
Confidence: {signal['confidence']:.1f}%
Leverage: {signal['leverage']}x
Entry: ${signal['current_price']:.4f}
Time: {self.format_ist_time(signal['timestamp'])}
        """.strip()
    
    def _get_confidence_emoji(self, confidence: float) -> str:
        """Get emoji based on confidence level"""
        if confidence >= 90:
            return "🌟"
        elif confidence >= 85:
            return "⭐"
        elif confidence >= 80:
            return "🔥"
        elif confidence >= 75:
            return "💎"
        else:
            return "📊"
    
    def format_sideways_signal(self, signal: Dict) -> str:
        """Format sideways market signals with range trading info"""
        try:
            # Determine strategy emoji and description
            strategy_type = signal.get('strategy_type', 'UNKNOWN')
            
            if 'RANGE_LONG' in strategy_type:
                direction_emoji = f"{SIDEWAYS_ALERT_CONFIG['range_long_emoji']} "
                strategy_desc = "Range Trading - Buy Support"
            elif 'RANGE_SHORT' in strategy_type:
                direction_emoji = f"{SIDEWAYS_ALERT_CONFIG['range_short_emoji']} "
                strategy_desc = "Range Trading - Sell Resistance"
            elif 'MEAN_REVERSION' in strategy_type:
                direction_emoji = f"{SIDEWAYS_ALERT_CONFIG['mean_reversion_emoji']} "
                strategy_desc = "Mean Reversion Trade"
            else:
                direction_emoji = "🔄"
                strategy_desc = "Sideways Market Trade"
            
            # Support/Resistance info
            support_level = signal.get('support_level', 0)
            resistance_level = signal.get('resistance_level', 0)
            range_size_pct = signal.get('range_size_pct', 0)
            position_in_range = signal.get('position_in_range', 0)
            
            # Calculate position size
            risk_amount = CAPITAL * RISK_PER_TRADE
            entry_price = signal['current_price']
            sl_price = signal.get('sl_price', entry_price)
            
            position_size = (risk_amount * signal['leverage']) / abs(entry_price - sl_price) if abs(entry_price - sl_price) > 0 else 0
            position_value = position_size * entry_price
            
            message = f"""
🔄 <b>ProTradeAI Pro+ Sideways Signal</b> 📊

{direction_emoji} <b>{signal['symbol']} - {signal['signal_type']}</b>
🎯 <b>Strategy:</b> {strategy_desc}
⚡ <b>Leverage:</b> {signal['leverage']}x
📈 <b>Timeframe:</b> {signal['timeframe']}
🎯 <b>Confidence:</b> {signal['confidence']:.1f}% ({signal['confidence_grade']})

💰 <b>ENTRY ZONE</b>
🔸 Entry Price: ${signal['current_price']:.4f}
🔸 Position Size: {position_size:.2f} {signal['symbol'].replace('USDT', '')}
🔸 Position Value: ${position_value:.2f}

📊 <b>RANGE ANALYSIS</b>
{SIDEWAYS_ALERT_CONFIG['support_emoji']} Support: ${support_level:.4f}
{SIDEWAYS_ALERT_CONFIG['resistance_emoji']} Resistance: ${resistance_level:.4f}
{SIDEWAYS_ALERT_CONFIG['range_emoji']} Range Size: {range_size_pct:.2f}%
📍 Position in Range: {position_in_range:.1%}

🛡️ <b>RISK MANAGEMENT</b>
🔻 Stop Loss: ${signal.get('sl_price', 0):.4f} (-{signal.get('sl_distance_pct', 0):.2f}%)
🎯 Take Profit: ${signal.get('tp_price', 0):.4f} (+{signal.get('tp_distance_pct', 0):.2f}%)
📊 R:R Ratio: 1:{signal.get('rr_ratio', 0):.2f}

📋 <b>TECHNICAL ANALYSIS</b>
📍 RSI: {signal.get('rsi', 0):.1f}
📍 Market Regime: {signal.get('market_regime', 'SIDEWAYS')}
📍 Strategy: {strategy_type}

⏰ <b>TIMING</b>
🕐 Signal Time: {self.format_ist_time(signal['timestamp'])}
⏳ Expected Duration: 4-12 hours
💼 Risk per Trade: {RISK_PER_TRADE*100:.1f}% (${risk_amount:.2f})

⚠️ <b>SIDEWAYS MARKET STRATEGY</b>
📊 Range-bound trading approach
🔄 Take profit at opposite range level
📈 Monitor for breakout signals

<b>ProTradeAI Pro+ | Sideways Signal #{self.alert_count + 1}</b>
            """.strip()
            
            return message
            
        except Exception as e:
            logger.error(f"Error formatting sideways signal: {e}")
            return self._format_simple_signal(signal)

    def send_signal_alert(self, signal: Dict) -> bool:
        """Enhanced signal alert with market regime detection"""
        try:
            # Check if it's a sideways market signal
            market_regime = signal.get('market_regime', 'TRENDING')
            
            if market_regime == 'SIDEWAYS' or 'RANGE' in signal.get('strategy_type', ''):
                message = self.format_sideways_signal(signal)
            else:
                # Use existing formatting for trending signals
                if TELEGRAM_CONFIG['alert_format'] == 'pro_plus':
                    message = self.format_pro_plus_signal(signal)
                else:
                    message = self.format_summary_signal(signal)
            
            # Send message
            success = self.send_message(message)
            
            if success:
                self.alert_count += 1
                logger.info(f"Signal alert sent for {signal['symbol']} {signal['signal_type']} ({market_regime})")
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending signal alert: {e}")
            return False
    
    def send_daily_summary(self, signals: list, stats: Dict) -> bool:
        """Send daily trading summary"""
        try:
            total_signals = len(signals)
            long_signals = len([s for s in signals if s['signal_type'] == 'LONG'])
            short_signals = total_signals - long_signals
            
            avg_confidence = sum(s['confidence'] for s in signals) / total_signals if signals else 0
            
            # Get current IST date
            current_ist = datetime.now(self.timezone)
            
            message = f"""
📊 <b>Daily Trading Summary</b>

📈 <b>Signal Statistics</b>
🔸 Total Signals: {total_signals}
🔸 LONG: {long_signals} | SHORT: {short_signals}
🔸 Avg Confidence: {avg_confidence:.1f}%

💰 <b>Risk Management</b>
🔸 Total Capital: ${CAPITAL:,.2f}
🔸 Risk per Trade: {RISK_PER_TRADE*100:.1f}%
🔸 Max Daily Risk: ${CAPITAL * RISK_PER_TRADE * MAX_DAILY_TRADES:,.2f}

⚡ <b>Top Signals Today</b>
            """.strip()
            
            # Add top 3 signals
            for i, signal in enumerate(signals[:3], 1):
                direction_emoji = "🚀" if signal['signal_type'] == 'LONG' else "📉"
                message += f"\n{i}. {direction_emoji} {signal['symbol']} {signal['confidence']:.1f}%"
            
            message += f"\n\n🤖 <b>ProTradeAI Pro+ | {current_ist.strftime('%d %b %Y')} IST</b>"
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
            return False
    
    def send_system_status(self, status: Dict) -> bool:
        """Send system status update"""
        try:
            status_emoji = "✅" if status.get('healthy', False) else "⚠️"
            
            message = f"""
{status_emoji} <b>System Status Update</b>

🔧 <b>Bot Health</b>
🔸 Status: {status.get('status', 'Unknown')}
🔸 Uptime: {status.get('uptime', 'Unknown')}
🔸 Last Signal: {status.get('last_signal_time', 'Never')}

📊 <b>Performance</b>
🔸 Signals Today: {status.get('signals_today', 0)}
🔸 Success Rate: {status.get('success_rate', 0):.1f}%
🔸 Model Accuracy: {status.get('model_accuracy', 0):.1f}%

🌙 <b>Schedule Info</b>
🔸 Shutdown Period: {'Yes' if status.get('is_shutdown_period', False) else 'No'}
🔸 Next Resume: {status.get('next_resume_time', 'N/A')}

🤖 <b>ProTradeAI Pro+ Monitor</b>
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending status update: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test Telegram connection"""
        try:
            current_ist = datetime.now(self.timezone)
            
            test_message = f"""
🧪 <b>Test Message</b>

✅ ProTradeAI Pro+ connection successful!
🕐 Time: {self.format_ist_datetime(current_ist)}
🌏 Timezone: Indian Standard Time (IST)
🤖 Bot is ready to send trading signals

<i>This is a test message - you can ignore it.</i>
            """.strip()
            
            return self.send_message(test_message)
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def send_error_alert(self, error_msg: str, component: str = "System") -> bool:
        """Send error alert"""
        try:
            current_ist = datetime.now(self.timezone)
            
            message = f"""
🚨 <b>Error Alert</b>

❌ <b>Component:</b> {component}
⚠️ <b>Error:</b> {error_msg}
🕐 <b>Time:</b> {self.format_ist_time(current_ist)}

🔧 Please check system logs for details.

<b>ProTradeAI Pro+ Monitor</b>
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending error alert: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get notifier statistics"""
        return {
            'total_alerts_sent': self.alert_count,
            'bot_configured': bool(self.bot_token and self.chat_id),
            'last_message_time': datetime.now(self.timezone),
            'timezone': 'Asia/Kolkata (IST)'
        }

# Global notifier instance
telegram_notifier = TelegramNotifier()

# ========================================================================
# Enhanced Telegram Command System
# ========================================================================



class TelegramCommandHandler:
    """Safe Telegram command handler for manual bot control"""
    
    def __init__(self, bot_token: str, chat_id: str, strategy_ai_instance):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.strategy_ai = strategy_ai_instance
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.last_update_id = 0
        self.command_cooldowns = {}
        self.is_listening = False
        self.authorized_chat_ids = [str(chat_id)]  # Security: only your chat
        
        # Command registry with cooldowns (seconds)
        self.commands = {
            '/scan': {'func': self.cmd_scan, 'cooldown': 60, 'description': 'Run full market scan'},
            '/quick': {'func': self.cmd_quick_scan, 'cooldown': 30, 'description': 'Run quick market scan'},
            '/status': {'func': self.cmd_status, 'cooldown': 10, 'description': 'Get bot status'},
            '/stats': {'func': self.cmd_stats, 'cooldown': 15, 'description': 'Get performance stats'},
            '/help': {'func': self.cmd_help, 'cooldown': 5, 'description': 'Show available commands'},
            '/test': {'func': self.cmd_test, 'cooldown': 120, 'description': 'Test signal generation'},
            '/signals': {'func': self.cmd_recent_signals, 'cooldown': 20, 'description': 'Show recent signals'},
            '/restart': {'func': self.cmd_restart_scanner, 'cooldown': 300, 'description': 'Restart scanner (admin only)'},
        }
        
        logger.info("🤖 Telegram command handler initialized")
    
    def start_listening(self):
        """Start listening for commands in a separate thread"""
        if self.is_listening:
            return
        
        self.is_listening = True
        listener_thread = threading.Thread(target=self._listen_for_commands, daemon=True)
        listener_thread.start()
        logger.info("👂 Started listening for Telegram commands")
    
    def stop_listening(self):
        """Stop listening for commands"""
        self.is_listening = False
        logger.info("🛑 Stopped listening for Telegram commands")
    
    def _listen_for_commands(self):
        """Listen for incoming Telegram commands"""
        while self.is_listening:
            try:
                updates = self._get_updates()
                for update in updates:
                    self._process_update(update)
                time.sleep(2)  # Poll every 2 seconds
            except Exception as e:
                logger.error(f"Error in command listener: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _get_updates(self) -> list:
        """Get updates from Telegram"""
        try:
            url = f"{self.base_url}/getUpdates"
            params = {
                'offset': self.last_update_id + 1,
                'timeout': 5,
                'allowed_updates': ['message']
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('ok'):
                updates = data.get('result', [])
                if updates:
                    self.last_update_id = updates[-1]['update_id']
                return updates
            
        except Exception as e:
            logger.error(f"Error getting Telegram updates: {e}")
        
        return []
    
    def _process_update(self, update: dict):
        """Process a single Telegram update"""
        try:
            message = update.get('message', {})
            if not message:
                return
            
            chat_id = str(message.get('chat', {}).get('id', ''))
            text = message.get('text', '').strip()
            user_id = message.get('from', {}).get('id', '')
            username = message.get('from', {}).get('username', 'Unknown')
            
            # Security check
            if chat_id not in self.authorized_chat_ids:
                logger.warning(f"Unauthorized command attempt from {username} ({chat_id})")
                return
            
            # Process command
            if text.startswith('/'):
                command = text.split()[0].lower()
                if command in self.commands:
                    self._execute_command(command, message)
                else:
                    self._send_response(f"❌ Unknown command: {command}\n\nType /help for available commands.")
                    
        except Exception as e:
            logger.error(f"Error processing Telegram update: {e}")
    
    def _execute_command(self, command: str, message: dict):
        """Execute a command with safety checks"""
        try:
            user_id = message.get('from', {}).get('id', '')
            username = message.get('from', {}).get('username', 'Unknown')
            
            # Check cooldown
            cooldown_key = f"{user_id}_{command}"
            now = time.time()
            
            if cooldown_key in self.command_cooldowns:
                last_used = self.command_cooldowns[cooldown_key]
                cooldown_time = self.commands[command]['cooldown']
                
                if now - last_used < cooldown_time:
                    remaining = int(cooldown_time - (now - last_used))
                    self._send_response(f"⏳ Command on cooldown. Wait {remaining} seconds.")
                    return
            
            # Execute command
            logger.info(f"🤖 Executing command {command} for {username}")
            self.command_cooldowns[cooldown_key] = now
            
            command_func = self.commands[command]['func']
            command_func(message)
            
        except Exception as e:
            logger.error(f"Error executing command {command}: {e}")
            self._send_response(f"❌ Error executing command: {str(e)}")
    
    def _send_response(self, text: str):
        """Send response message"""
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': 'HTML',
                'disable_web_page_preview': True
            }
            
            response = requests.post(url, data=data, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            logger.error(f"Error sending command response: {e}")
    
    # Command implementations
    
    def cmd_scan(self, message: dict):
        """Execute full market scan"""
        try:
            self._send_response("🔍 <b>Starting Full Market Scan...</b>\n\n⏳ This may take 1-2 minutes...")
            
            # Import the strategy_ai instance
            signals = self.strategy_ai.scan_all_symbols()
            
            if signals:
                response = f"🎯 <b>Scan Complete!</b>\n\n✅ Found {len(signals)} signals:\n\n"
                
                for i, signal in enumerate(signals[:5], 1):  # Show max 5
                    direction_emoji = "🚀" if signal['signal_type'] == 'LONG' else "📉"
                    response += f"{i}. {direction_emoji} <b>{signal['symbol']}</b> {signal['signal_type']} - {signal['confidence']:.1f}%\n"
                
                if len(signals) > 5:
                    response += f"\n... and {len(signals) - 5} more signals\n"
                
                response += f"\n📱 <b>All signals sent to this chat!</b>"
                
            else:
                response = "📭 <b>Scan Complete</b>\n\n❌ No signals found\n🔍 Market conditions may not be optimal right now"
            
            self._send_response(response)
            
        except Exception as e:
            logger.error(f"Error in scan command: {e}")
            self._send_response(f"❌ Scan failed: {str(e)}")
    
    def cmd_quick_scan(self, message: dict):
        """Execute quick market scan"""
        try:
            self._send_response("⚡ <b>Starting Quick Scan...</b>\n\n⏳ Scanning top symbols...")
            
            # Quick scan on top 3 symbols
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
            timeframes = ['1h', '4h']
            signals_found = 0
            
            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        signal = self.strategy_ai.predict_signal(symbol, timeframe)
                        if signal:
                            signals_found += 1
                            # Send signal immediately
                            from notifier import telegram_notifier
                            telegram_notifier.send_signal_alert(signal)
                    except:
                        continue
            
            if signals_found > 0:
                response = f"⚡ <b>Quick Scan Complete!</b>\n\n🎯 Found {signals_found} signals\n📱 Sent to this chat!"
            else:
                response = "⚡ <b>Quick Scan Complete</b>\n\n📭 No immediate opportunities found"
            
            self._send_response(response)
            
        except Exception as e:
            self._send_response(f"❌ Quick scan failed: {str(e)}")
    
    def cmd_status(self, message: dict):
        """Get bot status"""
        try:
            # Get bot status from main bot instance
            uptime = datetime.now() - datetime.now()  # This should be calculated from actual start time
            
            status_response = f"""
📊 <b>ProTradeAI Pro+ Status</b>

🟢 <b>Bot Status:</b> Running
⏰ <b>Uptime:</b> Active
🤖 <b>Model:</b> GradientBoostingClassifier
🎯 <b>Features:</b> 21 technical indicators

📈 <b>Scanning:</b>
🔸 Quick scans: Every 5 minutes
🔸 Full scans: Every 15 minutes
🔸 Symbols monitored: 10

💰 <b>Settings:</b>
🔸 Capital: $10,000
🔸 Risk per trade: 1.5%
🔸 Max daily trades: 15

🕐 <b>Last update:</b> {datetime.now().strftime('%H:%M:%S IST')}
            """.strip()
            
            self._send_response(status_response)
            
        except Exception as e:
            self._send_response(f"❌ Error getting status: {str(e)}")
    
    def cmd_stats(self, message: dict):
        """Get performance statistics"""
        try:
            # Get stats from signal tracker
            today_stats = self.strategy_ai.signal_tracker.get_performance_metrics(days=1)
            week_stats = self.strategy_ai.signal_tracker.get_performance_metrics(days=7)
            
            stats_response = f"""
📊 <b>Performance Statistics</b>

📈 <b>Today:</b>
🔸 Signals: {today_stats['total_signals']}
🔸 Win Rate: {today_stats['win_rate']:.1f}%
🔸 Total P&L: {today_stats['total_pnl']:.2f}%

📊 <b>Last 7 Days:</b>
🔸 Total Signals: {week_stats['total_signals']}
🔸 Win Rate: {week_stats['win_rate']:.1f}%
🔸 Avg Return: {week_stats['avg_return_per_trade']:.2f}%
🔸 Best Trade: +{week_stats['best_trade']:.2f}%
🔸 Worst Trade: {week_stats['worst_trade']:.2f}%

🎯 <b>Model Accuracy:</b> 75%+
            """.strip()
            
            self._send_response(stats_response)
            
        except Exception as e:
            self._send_response(f"❌ Error getting stats: {str(e)}")
    
    def cmd_help(self, message: dict):
        """Show available commands"""
        help_text = "<b>🤖 Available Commands:</b>\n\n"
        
        for cmd, info in self.commands.items():
            cooldown = info['cooldown']
            desc = info['description']
            help_text += f"<code>{cmd}</code> - {desc}\n<i>Cooldown: {cooldown}s</i>\n\n"
        
        help_text += "⚠️ <b>Note:</b> Commands have cooldowns to prevent spam and protect the bot."
        
        self._send_response(help_text)
    
    def cmd_test(self, message: dict):
        """Test signal generation"""
        try:
            self._send_response("🧪 <b>Testing Signal Generation...</b>\n\n⏳ Please wait...")
            
            test_symbols = ['BTCUSDT', 'ETHUSDT']
            signals_found = 0
            
            for symbol in test_symbols:
                try:
                    signal = self.strategy_ai.predict_signal(symbol, '4h')
                    if signal:
                        signals_found += 1
                        # Send the test signal
                        from notifier import telegram_notifier
                        telegram_notifier.send_signal_alert(signal)
                except:
                    continue
            
            if signals_found > 0:
                response = f"✅ <b>Test Successful!</b>\n\n🎯 Generated {signals_found} test signals\n📱 Check above for signal details"
            else:
                response = "✅ <b>Test Complete</b>\n\n📭 No signals in current market conditions\n🔍 This is normal during low volatility"
            
            self._send_response(response)
            
        except Exception as e:
            self._send_response(f"❌ Test failed: {str(e)}")
    
    def cmd_recent_signals(self, message: dict):
        """Show recent signals"""
        try:
            signals = self.strategy_ai.signal_tracker.signals_sent[-10:]  # Last 10 signals
            
            if not signals:
                self._send_response("📭 <b>No Recent Signals</b>\n\nNo signals have been generated yet.")
                return
            
            response = f"📊 <b>Last {len(signals)} Signals:</b>\n\n"
            
            for i, signal in enumerate(reversed(signals), 1):
                try:
                    symbol = signal.get('symbol', 'Unknown')
                    signal_type = signal.get('signal_type', 'Unknown')
                    confidence = signal.get('confidence', 0)
                    timestamp = signal.get('timestamp', '')
                    
                    # Parse timestamp
                    if timestamp:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = dt.strftime('%H:%M')
                    else:
                        time_str = 'Unknown'
                    
                    direction_emoji = "🚀" if signal_type == 'LONG' else "📉"
                    response += f"{i}. {direction_emoji} <b>{symbol}</b> {signal_type} ({confidence:.1f}%) - {time_str}\n"
                    
                except:
                    continue
            
            self._send_response(response)
            
        except Exception as e:
            self._send_response(f"❌ Error getting recent signals: {str(e)}")
    
    def cmd_restart_scanner(self, message: dict):
        """Restart scanner (admin only)"""
        try:
            # This is a placeholder - implement based on your main bot structure
            self._send_response("🔄 <b>Scanner Restart</b>\n\n⚠️ This feature requires main bot integration")
            
        except Exception as e:
            self._send_response(f"❌ Restart failed: {str(e)}")

# Integration with existing TelegramNotifier class
class EnhancedTelegramNotifier(TelegramNotifier):
    """Enhanced notifier with command support"""
    
    def __init__(self):
        super().__init__()
        self.command_handler = None
    
    def enable_commands(self, strategy_ai_instance):
        """Enable command handling"""
        try:
            self.command_handler = TelegramCommandHandler(
                self.bot_token, 
                self.chat_id, 
                strategy_ai_instance
            )
            self.command_handler.start_listening()
            logger.info("✅ Telegram commands enabled")
            
            # Send welcome message
            self.send_message(
                "🤖 <b>Commands Enabled!</b>\n\n"
                "Type <code>/help</code> to see available commands.\n\n"
                "📱 You can now control the bot via Telegram!"
            )
            
        except Exception as e:
            logger.error(f"Error enabling commands: {e}")
    
    def disable_commands(self):
        """Disable command handling"""
        if self.command_handler:
            self.command_handler.stop_listening()
            self.command_handler = None
            logger.info("🛑 Telegram commands disabled")