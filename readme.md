# 🤖 ProTradeAI Pro+ Trading Bot

**Advanced AI-powered cryptocurrency trading signal generator with Telegram alerts and web dashboard.**

## ⚡ Quick Start

1. **Clone/Download** all files to a folder
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Configure Telegram** (see setup below)
4. **Copy** `.env.template` to `.env` and fill in your values
5. **Run**: `python main.py`

## 📋 Features

- 🧠 **AI-Powered Signal Generation** - XGBoost/Random Forest models
- ⚡ **Leverage-Aware Risk Management** - Dynamic SL/TP calculation
- 📱 **Rich Telegram Alerts** - Professional formatted signals
- 🌐 **Real-time Web Dashboard** - Live monitoring interface
- ⏰ **Automated Scheduling** - Scans every 15 minutes
- 🛡️ **Manual Trading Focus** - No auto-execution (safety first)
- 📊 **Multi-timeframe Analysis** - 1H, 4H, 1D timeframes
- 💎 **10 Major Symbols** - BTC, ETH, BNB, ADA, SOL, XRP, DOT, LINK, LTC, MATIC

## 🚀 Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Setup Telegram Bot

#### Step 1: Create Bot
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot`
3. Choose a name: `ProTradeAI Pro+`
4. Choose a username: `your_unique_bot_name_bot`
5. Copy the **bot token**

#### Step 2: Get Chat ID
1. Start your bot (click the link from BotFather)
2. Send any message to your bot
3. Visit: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
4. Find your **chat ID** in the response

### 3. Configure Environment

```bash
cp .env.template .env
```

Edit `.env` file:
```env
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
TRADING_CAPITAL=10000
```

## 🎯 Usage

### Start the Bot

```bash
# Normal operation (runs forever)
python main.py

# Test mode (single scan)
python main.py test

# Check status
python main.py status
```

### Access Dashboard

1. **Start dashboard**: `python dashboard.py`
2. **Open browser**: http://localhost:5000
3. **View signals**: Real-time signal monitoring

### Manual Trading Workflow

1. **Receive signal** on Telegram
2. **Copy trade details** to your exchange
3. **Set stop loss** and take profit orders
4. **Monitor position** according to timeframe

## 📊 Signal Format

Each signal includes:

```
🚀 ProTradeAI Pro+ Signal ⭐

🚀 ETHUSDT - LONG
⚡ Leverage: 3x
📈 Timeframe: 4H
🎯 Confidence: 87.5% (A)

💰 ENTRY ZONE
🔸 Entry Price: $2,847.50
🔸 Position Size: 1.25 ETH
🔸 Position Value: $3,559.38

🛡️ RISK MANAGEMENT
🔻 Stop Loss: $2,785.20 (-2.19%)
🎯 Take Profit: $2,972.40 (+4.38%)
📊 R:R Ratio: 1:2.00

📋 TECHNICAL ANALYSIS
📍 RSI: 45.2
📍 MACD: 0.0234
📍 ATR: 67.83
📍 Volume: 1.8x avg

⏰ TIMING
🕐 Signal Time: 14:23:45 UTC
⏳ Hold Period: ~8 hours
💼 Risk per Trade: 2.0% ($200.00)

⚠️ MANUAL EXECUTION REQUIRED
```

## ⚙️ Configuration

### Risk Management

Edit `config.py`:

```python
RISK_PER_TRADE = 0.02  # 2% per trade
MAX_DAILY_TRADES = 10
CAPITAL = 10000  # Your trading capital
```

### Leverage Settings

```python
LEVERAGE_CONFIG = {
    'conservative': {'min': 2, 'max': 3},
    'moderate': {'min': 3, 'max': 5},    # Default
    'aggressive': {'min': 5, 'max': 10}
}
```

### Alert Format

```python
TELEGRAM_CONFIG = {
    'alert_format': 'pro_plus',  # or 'summary'
}
```

## 🔄 Model Retraining

### Automatic Retraining

The model automatically retrains weekly using fresh market data.

### Manual Retraining

```bash
# Retrain with 90 days of data
python retrain_model.py

# Retrain with custom period
python retrain_model.py --days 120

# Verbose output
python retrain_model.py --verbose
```

## 📂 File Structure

```
ProTradeAI_Pro+/
├── main.py              # Main bot runner
├── strategy_ai.py       # AI strategy engine
├── notifier.py          # Telegram alerts
├── dashboard.py         # Web dashboard
├── config.py            # Configuration
├── retrain_model.py     # Model retraining
├── requirements.txt     # Dependencies
├── .env.template        # Environment template
├── .env                 # Your config (create this)
├── ai_model.pkl         # AI model (auto-created)
├── data/                # Signal storage
├── logs/                # Log files
└── models/              # Model backups
```

## 🛠️ Advanced Features

### Custom Symbols

Edit `config.py`:
```python
SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'YOUR_SYMBOL_HERE'
]
```

### Custom Timeframes

```python
TIMEFRAMES = ['1h', '4h', '1d', '12h']  # Add 12h
```

### Dashboard Customization

```python
DASHBOARD_CONFIG = {
    'host': '0.0.0.0',      # Allow external access
    'port': 8080,           # Custom port
    'refresh_interval': 15  # Faster refresh
}
```

## 🔧 Troubleshooting

### Common Issues

**1. Telegram not working**
```bash
# Test connection
python -c "from notifier import telegram_notifier; print(telegram_notifier.test_connection())"
```

**2. No signals generated**
- Check internet connection
- Verify Binance API access
- Review logs in `logs/` folder

**3. Model errors**
```bash
# Recreate model
rm ai_model.pkl
python main.py test
```

**4. Dashboard not loading**
```bash
# Check if port is available
python dashboard.py
# Try different port in config
```

### Logs

Check `logs/protrade_ai.log` for detailed error information.

## 📈 Performance Optimization

### High-Performance Setup

1. **Run on VPS/Cloud** for 24/7 operation
2. **Use SSD storage** for faster data access
3. **Increase memory** for better model performance
4. **Monitor system resources** regularly

### Replit Deployment

1. Upload all files to Replit
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables in Replit Secrets
4. Run: `python main.py`

## 🚨 Important Disclaimers

- **Manual Trading Only**: This bot generates signals, not automatic trades
- **Risk Management**: Only trade with money you can afford to lose
- **Past Performance**: No guarantee of future results
- **Education Only**: This is for educational and research purposes
- **No Financial Advice**: Signals are not financial advice

## 🛡️ Security

- **API Keys**: Store safely in `.env` file
- **Telegram Token**: Keep your bot token private
- **Regular Updates**: Update dependencies regularly
- **Backups**: Backup your configuration and models

## 📞 Support

### Getting Help

1. **Check logs** in `logs/` folder
2. **Review configuration** in `config.py`
3. **Test components** individually
4. **Check internet connectivity**

### Development Mode

```bash
# Run with debug logging
python main.py --verbose

# Single scan for testing
python main.py test

# Dashboard in debug mode
python dashboard.py --debug
```

## 🔮 Future Enhancements

- Multi-exchange support
- Portfolio management
- Advanced risk metrics
- Mobile app notifications
- Social trading features

## 📄 License

This project is for educational purposes. Use at your own risk.

---

**⚡ ProTradeAI Pro+ - Where AI Meets Trading Excellence**

*Happy Trading! 🚀*