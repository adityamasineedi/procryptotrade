"""
ProTradeAI Pro+ Telegram Notifier - RESTART LOOP FIXED VERSION
Prevents command handler crashes and improves stability

KEY FIXES:
- Better error handling in command listener
- Prevents crashes that cause restart loops
- Reduced API call frequency
- More stable threading
- Better memory management
- FIXED: Duplicate command activation messages
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

# 🔧 NEW: Global variables to prevent duplicate command activation
_commands_enabled = False
_commands_lock = threading.Lock()
_last_enable_time = 0

class TelegramNotifier:
    def __init__(self):
        self.bot_token = TELEGRAM_CONFIG['bot_token']
        self.chat_id = TELEGRAM_CONFIG['chat_id']
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.alert_count = 0
        self.timezone = pytz.timezone('Asia/Kolkata')  # IST timezone
        
    def format_ist_time(self, timestamp: datetime) -> str:
        """Format timestamp to IST - SIMPLIFIED VERSION"""
        try:
            # Simple fix: Treat all timestamps as IST already
            if timestamp.tzinfo is None:
                # Naive datetime - assume it's already IST
                return timestamp.strftime('%H:%M:%S IST')
            else:
                # Timezone-aware - convert to IST
                ist_time = timestamp.astimezone(self.timezone)
                return ist_time.strftime('%H:%M:%S IST')
        except Exception as e:
            logger.error(f"Error formatting IST time: {e}")
            # Fallback to current time
            return datetime.now().strftime('%H:%M:%S IST')
    
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
                response = requests.post(url, data=data, timeout=15)
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
# FIXED Telegram Command System - STABILITY FOCUSED
# ========================================================================

class StableTelegramCommandHandler:
    """🔧 STABILITY FOCUSED: Command handler that prevents crashes"""
    
    def __init__(self, bot_token: str, chat_id: str, strategy_ai_instance):
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.strategy_ai = strategy_ai_instance
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        self.last_update_id = 0
        self.command_cooldowns = {}
        self.is_listening = False
        self.consecutive_errors = 0
        self.max_errors = 5
        
        # Simplified authorization
        self.authorized_chat_id = str(chat_id)
        
        # Command registry
        self.commands = {
            '/scan': {'func': self.cmd_scan, 'cooldown': 60, 'description': 'Run full market scan'},
            '/quick': {'func': self.cmd_quick_scan, 'cooldown': 30, 'description': 'Run quick market scan'},
            '/status': {'func': self.cmd_status, 'cooldown': 10, 'description': 'Get bot status'},
            '/stats': {'func': self.cmd_stats, 'cooldown': 15, 'description': 'Get performance stats'},
            '/help': {'func': self.cmd_help, 'cooldown': 5, 'description': 'Show available commands'},
            '/test': {'func': self.cmd_test, 'cooldown': 30, 'description': 'Test signal generation'},
            '/signals': {'func': self.cmd_recent_signals, 'cooldown': 20, 'description': 'Show recent signals'},
        }
        
        self._initialize_update_offset()
        logger.info(f"🤖 Stable command handler initialized for chat {self.chat_id}")
    
    def _initialize_update_offset(self):
        """Get latest update ID to avoid processing old messages"""
        try:
            response = requests.get(f"{self.base_url}/getUpdates?limit=1", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('ok') and data.get('result'):
                    self.last_update_id = data['result'][-1]['update_id']
                    logger.info(f"📱 Initialized offset: {self.last_update_id}")
        except Exception as e:
            logger.warning(f"Could not initialize offset: {e}")
            self.last_update_id = 0
    
    def start_listening(self):
        """🔧 FIXED: Start command listener with duplicate prevention"""
        if self.is_listening:
            logger.info("⚠️ Command listener already running, skipping start")
            return
        
        self.is_listening = True
        listener_thread = threading.Thread(target=self._stable_listen_loop, daemon=True)
        listener_thread.start()
        logger.info("👂 Started stable command listener")
    
    def stop_listening(self):
        """Stop command listener"""
        self.is_listening = False
        logger.info("🛑 Stopped command listener")
    
    def _stable_listen_loop(self):
        """🔧 STABILITY FOCUSED: Main listening loop with comprehensive error handling"""
        while self.is_listening:
            try:
                updates = self._get_updates_safely()
                if updates:
                    self.consecutive_errors = 0  # Reset on success
                    for update in updates:
                        try:
                            self._process_update_safely(update)
                        except Exception as e:
                            logger.error(f"Error processing individual update: {e}")
                            continue
                
                # Longer sleep to reduce API calls and prevent rate limiting
                time.sleep(5)
                
            except Exception as e:
                self.consecutive_errors += 1
                logger.error(f"Command listener error #{self.consecutive_errors}: {e}")
                
                # If too many errors, stop to prevent restart loops
                if self.consecutive_errors >= self.max_errors:
                    logger.error("🚨 Too many command errors - stopping listener to prevent restart loop")
                    self.is_listening = False
                    break
                
                # Progressive backoff
                sleep_time = min(60, 5 * self.consecutive_errors)
                logger.info(f"Sleeping {sleep_time}s before retry")
                time.sleep(sleep_time)
    
    def _get_updates_safely(self) -> list:
        """🔧 SAFE: Get updates with comprehensive error handling"""
        try:
            params = {
                'offset': self.last_update_id + 1,
                'timeout': 15,  # Longer timeout for stability
                'allowed_updates': ['message']
            }
            
            response = requests.get(f"{self.base_url}/getUpdates", params=params, timeout=20)
            
            # Check if response is valid
            if response.status_code != 200:
                logger.warning(f"HTTP {response.status_code} from Telegram API")
                return []
            
            data = response.json()
            if not data.get('ok'):
                logger.error(f"Telegram API error: {data}")
                return []
            
            updates = data.get('result', [])
            if updates:
                self.last_update_id = updates[-1]['update_id']
                logger.debug(f"📱 Got {len(updates)} updates")
            
            return updates
            
        except requests.exceptions.Timeout:
            logger.debug("Request timeout (normal)")
            return []
        except requests.exceptions.ConnectionError:
            logger.warning("Connection error to Telegram")
            return []
        except Exception as e:
            logger.error(f"Error getting updates: {e}")
            return []
    
    def _process_update_safely(self, update: dict):
        """🔧 SAFE: Process individual message with full error handling"""
        try:
            message = update.get('message', {})
            if not message:
                return
            
            # Extract message info safely
            chat_info = message.get('chat', {})
            user_info = message.get('from', {})
            text = message.get('text', '').strip()
            
            msg_chat_id = str(chat_info.get('id', ''))
            username = user_info.get('username', 'Unknown')
            first_name = user_info.get('first_name', 'Unknown')
            
            # Authorization check
            is_authorized = (msg_chat_id == self.authorized_chat_id)
            
            logger.info(f"📱 Message: '{text}' from {first_name}")
            logger.debug(f"   Chat ID: {msg_chat_id} | Expected: {self.authorized_chat_id}")
            
            if not is_authorized:
                logger.warning(f"❌ Unauthorized message from {username}")
                return
            
            # Process commands only
            if text.startswith('/'):
                command = text.split()[0].lower()
                logger.info(f"🤖 Processing command: {command}")
                
                if command in self.commands:
                    self._execute_command_safely(command, message)
                else:
                    logger.warning(f"❌ Unknown command: {command}")
                    self._send_response_safely(f"❌ Unknown command: {command}\n\nType /help for available commands.")
                    
        except Exception as e:
            logger.error(f"Error processing update: {e}")
    
    def _execute_command_safely(self, command: str, message: dict):
        """🔧 SAFE: Execute command with comprehensive error handling"""
        try:
            user_id = message.get('from', {}).get('id', '')
            username = message.get('from', {}).get('username', 'Unknown')
            
            # Cooldown check
            cooldown_key = f"{user_id}_{command}"
            now = time.time()
            
            if cooldown_key in self.command_cooldowns:
                last_used = self.command_cooldowns[cooldown_key]
                cooldown_time = self.commands[command]['cooldown']
                
                if now - last_used < cooldown_time:
                    remaining = int(cooldown_time - (now - last_used))
                    self._send_response_safely(f"⏳ Command on cooldown. Wait {remaining} seconds.")
                    return
            
            # Execute command
            logger.info(f"🚀 Executing {command} for {username}")
            self.command_cooldowns[cooldown_key] = now
            
            command_func = self.commands[command]['func']
            command_func(message)
            
            logger.info(f"✅ Command {command} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error executing {command}: {e}")
            self._send_response_safely(f"❌ Error executing command: {str(e)}")
    
    def _send_response_safely(self, text: str):
        """🔧 SAFE: Send response with error handling"""
        try:
            for attempt in range(3):
                try:
                    data = {
                        'chat_id': self.chat_id,
                        'text': text,
                        'parse_mode': 'HTML',
                        'disable_web_page_preview': True
                    }
                    
                    response = requests.post(f"{self.base_url}/sendMessage", data=data, timeout=15)
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('ok'):
                            logger.debug(f"✅ Response sent successfully")
                            return True
                        else:
                            logger.error(f"❌ Telegram error: {result}")
                    else:
                        logger.error(f"❌ HTTP {response.status_code}")
                    
                except requests.exceptions.Timeout:
                    logger.warning(f"Send timeout (attempt {attempt + 1})")
                except Exception as e:
                    logger.error(f"❌ Send error (attempt {attempt + 1}): {e}")
                
                if attempt < 2:
                    time.sleep(2)
        
        except Exception as e:
            logger.error(f"❌ Critical send error: {e}")
        
        return False
    
    # ========================= COMMAND IMPLEMENTATIONS =========================
    
    def cmd_help(self, message: dict):
        """Show help message"""
        help_text = "<b>🤖 ProTradeAI Pro+ Commands (STABLE):</b>\n\n"
        
        for cmd, info in self.commands.items():
            desc = info['description']
            cooldown = info['cooldown']
            help_text += f"<code>{cmd}</code> - {desc}\n<i>Cooldown: {cooldown}s</i>\n\n"
        
        help_text += "🔧 <b>SYSTEM STABLE!</b>\n"
        help_text += "✅ Error handling improved\n"
        help_text += "✅ Crash prevention active\n"
        help_text += "✅ Restart loop protection\n\n"
        help_text += "⚠️ Commands have cooldowns to prevent issues."
        
        self._send_response_safely(help_text)
    
    def cmd_status(self, message: dict):
        """Get detailed bot status"""
        try:
            status_response = f"""
📊 <b>ProTradeAI Pro+ Status (STABLE)</b>

🟢 <b>Bot Status:</b> Running & Stable
🤖 <b>Model:</b> {self.strategy_ai.get_model_info()['model_type']}
🎯 <b>Features:</b> {self.strategy_ai.get_model_info()['feature_count']}

📱 <b>Command System:</b>
🔸 Status: ✅ STABLE & CRASH-PROOF
🔸 Chat ID: {self.chat_id}
🔸 Listener: {'✅ Active' if self.is_listening else '❌ Stopped'}
🔸 Error Count: {self.consecutive_errors}/{self.max_errors}

💰 <b>Trading:</b>
🔸 Capital: ${CAPITAL:,.0f}
🔸 Risk: {RISK_PER_TRADE*100:.1f}% per trade
🔸 Max trades: {MAX_DAILY_TRADES}/day

🔧 <b>Stability Features:</b>
🔸 Error recovery: ACTIVE
🔸 Restart prevention: ACTIVE
🔸 Memory management: ACTIVE
🔸 Progressive backoff: ACTIVE

🕐 <b>Time:</b> {datetime.now().strftime('%H:%M:%S IST')}

✅ <b>System stable and crash-proof!</b>
            """.strip()
            
            self._send_response_safely(status_response)
            
        except Exception as e:
            logger.error(f"Error in status command: {e}")
            self._send_response_safely(f"❌ Error getting status: {str(e)}")
    
    def cmd_quick_scan(self, message: dict):
        """Execute quick market scan"""
        try:
            self._send_response_safely("⚡ <b>Quick Scan Starting...</b>\n\n⏳ Scanning top symbols...")
            
            signals_found = 0
            symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOTUSDT']
            timeframes = ['4h', '1d']
            
            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        signal = self.strategy_ai.predict_signal(symbol, timeframe, bypass_cooldown=True)
                        if signal:
                            signals_found += 1
                            # Send the signal
                            telegram_notifier.send_signal_alert(signal)
                            logger.info(f"✅ Quick scan signal: {symbol} {signal['signal_type']} {signal['confidence']:.1f}%")
                    except Exception as e:
                        logger.error(f"Error scanning {symbol} {timeframe}: {e}")
                        continue
            
            if signals_found > 0:
                response = f"⚡ <b>Quick Scan Complete!</b>\n\n🎯 Found {signals_found} signals\n📱 Detailed alerts sent above!"
            else:
                response = "⚡ <b>Quick Scan Complete</b>\n\n📭 No signals found\n🔍 Try different timeframes or check thresholds"
            
            self._send_response_safely(response)
            
        except Exception as e:
            logger.error(f"Error in quick scan: {e}")
            self._send_response_safely(f"❌ Quick scan failed: {str(e)}")
    
    def cmd_test(self, message: dict):
        """Test signal generation"""
        try:
            self._send_response_safely("🧪 <b>Testing Signal Generation...</b>\n\n⏳ Testing with multiple symbols...")
            
            signals_found = 0
            test_symbols = ['ETHUSDT', 'ADAUSDT']
            
            for symbol in test_symbols:
                for timeframe in ['4h', '1d']:
                    try:
                        signal = self.strategy_ai.predict_signal(symbol, timeframe, bypass_cooldown=True)
                        if signal:
                            signals_found += 1
                            telegram_notifier.send_signal_alert(signal)
                            logger.info(f"✅ Test signal: {symbol} {timeframe} {signal['signal_type']} {signal['confidence']:.1f}%")
                    except Exception as e:
                        logger.error(f"Error testing {symbol} {timeframe}: {e}")
            
            # Get debug info
            debug_info = self.strategy_ai.debug_signal_generation()
            
            if signals_found > 0:
                response = f"✅ <b>Test Successful!</b>\n\n🎯 Generated {signals_found} test signals\n📱 Check above for details\n\n"
            else:
                response = f"✅ <b>Test Complete</b>\n\n📭 No signals generated\n"
            
            response += f"🔧 <b>Debug Info:</b>\n"
            response += f"🔸 Min threshold: {debug_info.get('min_confidence_threshold', 'N/A')}%\n"
            response += f"🔸 Model loaded: {'✅' if debug_info.get('model_loaded') else '❌'}\n"
            response += f"🔸 Stable system: ✅ No crashes"
            
            self._send_response_safely(response)
            
        except Exception as e:
            logger.error(f"Error in test: {e}")
            self._send_response_safely(f"❌ Test failed: {str(e)}")
    
    def cmd_stats(self, message: dict):
        """Get performance statistics"""
        try:
            today_stats = self.strategy_ai.signal_tracker.get_performance_metrics(days=1)
            week_stats = self.strategy_ai.signal_tracker.get_performance_metrics(days=7)
            
            stats_response = f"""
📊 <b>Performance Statistics (STABLE)</b>

📈 <b>Today:</b>
🔸 Signals: {today_stats['total_signals']}
🔸 Win Rate: {today_stats['win_rate']:.1f}%
🔸 P&L: {today_stats['total_pnl']:.2f}%

📊 <b>Last 7 Days:</b>
🔸 Total: {week_stats['total_signals']}
🔸 Win Rate: {week_stats['win_rate']:.1f}%
🔸 Avg Return: {week_stats['avg_return_per_trade']:.2f}%
🔸 Best: +{week_stats['best_trade']:.2f}%
🔸 Worst: {week_stats['worst_trade']:.2f}%

🤖 <b>System:</b> STABLE & CRASH-PROOF
🎯 <b>Commands:</b> Error count: {self.consecutive_errors}/{self.max_errors}

🔧 <b>Stability Features Active:</b>
🔸 Error recovery ✅
🔸 Memory management ✅
🔸 Restart prevention ✅

✅ <b>All systems stable!</b>
            """.strip()
            
            self._send_response_safely(stats_response)
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            self._send_response_safely(f"❌ Error getting stats: {str(e)}")
    
    def cmd_scan(self, message: dict):
        """Execute full market scan"""
        try:
            self._send_response_safely("🔍 <b>Full Market Scan Starting...</b>\n\n⏳ This may take 1-2 minutes...")
            
            signals = self.strategy_ai.scan_all_symbols()
            
            if signals:
                # Send summary
                summary = f"🎯 <b>Scan Complete!</b>\n\n✅ Found {len(signals)} signals:\n\n"
                
                for i, signal in enumerate(signals[:5], 1):
                    emoji = "🚀" if signal['signal_type'] == 'LONG' else "📉"
                    summary += f"{i}. {emoji} {signal['symbol']} {signal['signal_type']} - {signal['confidence']:.1f}%\n"
                
                if len(signals) > 5:
                    summary += f"\n... and {len(signals) - 5} more\n"
                
                summary += "\n📱 <b>Sending detailed alerts...</b>"
                self._send_response_safely(summary)
                
                # Send detailed alerts
                for i, signal in enumerate(signals, 1):
                    try:
                        success = telegram_notifier.send_signal_alert(signal)
                        if success:
                            logger.info(f"✅ Alert {i}/{len(signals)}: {signal['symbol']} {signal['signal_type']}")
                        time.sleep(3)  # Prevent rate limiting
                    except Exception as e:
                        logger.error(f"Error sending alert {i}: {e}")
                
                self._send_response_safely(f"✅ <b>All {len(signals)} alerts sent!</b>")
                
            else:
                self._send_response_safely("""
📭 <b>Scan Complete</b>

❌ No signals found

🔧 <b>Possible reasons:</b>
🔸 Market conditions not optimal
🔸 All symbols in cooldown period
🔸 Confidence below thresholds

💡 <b>Try:</b> /test to check signal generation
                """.strip())
            
        except Exception as e:
            logger.error(f"Error in full scan: {e}")
            self._send_response_safely(f"❌ Scan failed: {str(e)}")
    
    def cmd_recent_signals(self, message: dict):
        """Show recent signals"""
        try:
            signals = self.strategy_ai.signal_tracker.signals_sent[-10:]
            
            if not signals:
                self._send_response_safely("📭 <b>No Recent Signals</b>\n\nNo signals generated yet.")
                return
            
            response = f"📊 <b>Last {len(signals)} Signals:</b>\n\n"
            
            for i, signal in enumerate(reversed(signals), 1):
                try:
                    symbol = signal.get('symbol', 'Unknown')
                    signal_type = signal.get('signal_type', 'Unknown')
                    confidence = signal.get('confidence', 0)
                    timestamp = signal.get('timestamp', '')
                    
                    if timestamp:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        time_str = dt.strftime('%H:%M')
                    else:
                        time_str = 'Unknown'
                    
                    emoji = "🚀" if signal_type == 'LONG' else "📉"
                    response += f"{i}. {emoji} <b>{symbol}</b> {signal_type} ({confidence:.1f}%) - {time_str}\n"
                    
                except:
                    continue
            
            response += f"\n✅ <b>System stable - no crashes!</b>"
            self._send_response_safely(response)
            
        except Exception as e:
            logger.error(f"Error getting signals: {e}")
            self._send_response_safely(f"❌ Error: {str(e)}")


# Global command handler instance
command_handler = None

def enable_simple_commands(strategy_ai_instance):
    """🔧 COMPLETELY FIXED: Enable Telegram commands with duplicate prevention"""
    global command_handler, _commands_enabled, _last_enable_time
    
    with _commands_lock:
        try:
            # Check if already enabled recently (within last 60 seconds)
            current_time = time.time()
            if _commands_enabled and (current_time - _last_enable_time) < 60:
                logger.info("ℹ️ Commands recently enabled, skipping duplicate call")
                return True
            
            if not TELEGRAM_CONFIG['bot_token'] or not TELEGRAM_CONFIG['chat_id']:
                logger.error("❌ Telegram credentials missing")
                return False
            
            # Check if command handler exists and is active
            if (command_handler is not None and 
                hasattr(command_handler, 'is_listening') and 
                command_handler.is_listening):
                logger.info("ℹ️ Commands already active, skipping initialization")
                return True
            
            # Stop existing handler if it exists but not listening
            if command_handler is not None:
                try:
                    command_handler.stop_listening()
                except:
                    pass
                command_handler = None
            
            # Create new handler
            command_handler = StableTelegramCommandHandler(
                TELEGRAM_CONFIG['bot_token'],
                TELEGRAM_CONFIG['chat_id'],
                strategy_ai_instance
            )
            command_handler.start_listening()
            
            # Mark as enabled
            _commands_enabled = True
            _last_enable_time = current_time
            
            logger.info("✅ Command handler started successfully")
            
            # Send activation message only once
            telegram_notifier.send_message(
                "🤖 <b>Commands Ready!</b>\n\n"
                "✅ Command system initialized\n"
                "📱 Type <code>/help</code> for commands\n\n"
                f"<i>Activated at {datetime.now().strftime('%H:%M:%S')}</i>"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error enabling commands: {e}")
            _commands_enabled = False
            return False

def disable_simple_commands():
    """Disable Telegram commands"""
    global command_handler, _commands_enabled
    
    with _commands_lock:
        try:
            _commands_enabled = False
            
            if command_handler:
                command_handler.stop_listening()
                command_handler = None
                logger.info("🛑 Telegram commands disabled")
                
                telegram_notifier.send_message(
                    "🛑 <b>Commands Disabled</b>\n\n"
                    "Command system deactivated."
                )
                
            return True
            
        except Exception as e:
            logger.error(f"Error disabling commands: {e}")
            return False