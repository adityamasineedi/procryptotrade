"""
ProTradeAI Pro+ Telegram Notifier
Advanced alert system with rich formatting and retry logic
"""

import requests
import time
import logging
from datetime import datetime
from typing import Dict, Optional
import json

from config import *

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self):
        self.bot_token = TELEGRAM_CONFIG['bot_token']
        self.chat_id = TELEGRAM_CONFIG['chat_id']
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.alert_count = 0
        
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
🕐 Signal Time: {signal['timestamp'].strftime('%H:%M:%S UTC')}
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
⏰ {signal['timestamp'].strftime('%H:%M')} | Risk: {RISK_PER_TRADE*100:.1f}%

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
Time: {signal['timestamp'].strftime('%H:%M:%S')}
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
    
    def send_signal_alert(self, signal: Dict) -> bool:
        """Send trading signal alert"""
        try:
            # Choose format based on config
            if TELEGRAM_CONFIG['alert_format'] == 'pro_plus':
                message = self.format_pro_plus_signal(signal)
            else:
                message = self.format_summary_signal(signal)
            
            # Send message
            success = self.send_message(message)
            
            if success:
                self.alert_count += 1
                logger.info(f"Signal alert sent for {signal['symbol']} {signal['signal_type']}")
            
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
            
            message += f"\n\n🤖 <b>ProTradeAI Pro+ | {datetime.now().strftime('%Y-%m-%d')}</b>"
            
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

🤖 <b>ProTradeAI Pro+ Monitor</b>
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            logger.error(f"Error sending status update: {e}")
            return False
    
    def test_connection(self) -> bool:
        """Test Telegram connection"""
        try:
            test_message = f"""
🧪 <b>Test Message</b>

✅ ProTradeAI Pro+ connection successful!
🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
            message = f"""
🚨 <b>Error Alert</b>

❌ <b>Component:</b> {component}
⚠️ <b>Error:</b> {error_msg}
🕐 <b>Time:</b> {datetime.now().strftime('%H:%M:%S')}

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
            'last_message_time': datetime.now()
        }

# Global notifier instance
telegram_notifier = TelegramNotifier()
