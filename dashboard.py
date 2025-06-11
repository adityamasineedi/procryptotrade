"""
ProTradeAI Pro+ Web Dashboard
Real-time monitoring interface for trading signals and system status
"""

from flask import Flask, render_template_string, jsonify, request
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from config import DASHBOARD_CONFIG, CAPITAL, RISK_PER_TRADE, MAX_DAILY_TRADES, TELEGRAM_CONFIG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Dashboard:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'protrade-ai-pro-plus-dashboard'
        self.data_dir = Path('data')
        self.data_dir.mkdir(exist_ok=True)  # Ensure data directory exists
        
        # Setup routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return render_template_string(
                self.get_dashboard_html(), 
                refresh_interval=DASHBOARD_CONFIG['refresh_interval']
            )
        
        @self.app.route('/api/signals')
        def api_signals():
            """API endpoint for signals data"""
            try:
                signals = self.load_signals()
                
                # Filter by parameters
                signal_type = request.args.get('type', 'all')
                timeframe = request.args.get('timeframe', 'all')
                limit = int(request.args.get('limit', DASHBOARD_CONFIG['max_signals_display']))
                
                filtered_signals = []
                for signal in signals:
                    if signal_type != 'all' and signal.get('signal_type') != signal_type.upper():
                        continue
                    if timeframe != 'all' and signal.get('timeframe') != timeframe:
                        continue
                    filtered_signals.append(signal)
                
                # Sort by timestamp (newest first)
                filtered_signals.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                
                return jsonify({
                    'success': True,
                    'signals': filtered_signals[:limit],
                    'total': len(filtered_signals)
                })
                
            except Exception as e:
                logger.error(f"Error loading signals: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'signals': [],
                    'total': 0
                })
        
        @self.app.route('/api/stats')
        def api_stats():
            """API endpoint for statistics"""
            try:
                signals = self.load_signals()
                today = datetime.now().strftime('%Y-%m-%d')
                
                # Filter today's signals
                today_signals = [
                    s for s in signals 
                    if s.get('timestamp', '').startswith(today)
                ]
                
                # Calculate statistics
                stats = {
                    'total_signals_today': len(today_signals),
                    'long_signals': len([s for s in today_signals if s.get('signal_type') == 'LONG']),
                    'short_signals': len([s for s in today_signals if s.get('signal_type') == 'SHORT']),
                    'avg_confidence': sum(s.get('confidence', 0) for s in today_signals) / len(today_signals) if today_signals else 0,
                    'highest_confidence': max((s.get('confidence', 0) for s in today_signals), default=0),
                    'symbols_active': len(set(s.get('symbol', '') for s in today_signals)),
                    'timeframes_used': list(set(s.get('timeframe', '') for s in today_signals)),
                    'last_signal_time': max((s.get('timestamp', '') for s in signals), default='Never'),
                    'capital': CAPITAL,
                    'risk_per_trade': RISK_PER_TRADE * 100,
                    'max_daily_trades': MAX_DAILY_TRADES
                }
                
                return jsonify({
                    'success': True,
                    'stats': stats
                })
                
            except Exception as e:
                logger.error(f"Error calculating stats: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'stats': {}
                })
        
        @self.app.route('/api/system-status')
        def api_system_status():
            """API endpoint for system status"""
            try:
                # Check if main bot is running (simplified)
                status = {
                    'bot_running': True,  # In real implementation, check actual bot status
                    'model_loaded': True,
                    'telegram_connected': bool(TELEGRAM_CONFIG['bot_token'] and TELEGRAM_CONFIG['chat_id']),
                    'last_health_check': datetime.now().isoformat(),
                    'uptime': '1d 5h 23m',  # Dummy value
                    'memory_usage': '145 MB',  # Dummy value
                    'cpu_usage': '12%',  # Dummy value
                    'next_scan': (datetime.now() + timedelta(minutes=15)).isoformat()
                }
                
                return jsonify({
                    'success': True,
                    'status': status
                })
                
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'status': {}
                })
        
        @self.app.route('/api/performance')
        def api_performance():
            """API endpoint for performance metrics"""
            try:
                signals = self.load_signals()
                
                # Calculate performance metrics (simplified)
                performance = {
                    'total_signals': len(signals),
                    'win_rate': 72.5,  # Dummy value
                    'avg_return': 3.2,  # Dummy value
                    'max_drawdown': -8.7,  # Dummy value
                    'sharpe_ratio': 1.85,  # Dummy value
                    'profit_factor': 2.1,  # Dummy value
                    'recent_performance': [
                        {'date': '2025-06-11', 'return': 2.3, 'signals': 5},
                        {'date': '2025-06-10', 'return': -1.2, 'signals': 3},
                        {'date': '2025-06-09', 'return': 4.7, 'signals': 7},
                        {'date': '2025-06-08', 'return': 1.8, 'signals': 4},
                        {'date': '2025-06-07', 'return': 3.4, 'signals': 6},
                    ]
                }
                
                return jsonify({
                    'success': True,
                    'performance': performance
                })
                
            except Exception as e:
                logger.error(f"Error calculating performance: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'performance': {}
                })
    
    def load_signals(self):
        """Load signals from JSON file"""
        signals_file = self.data_dir / 'signals.json'
        try:
            if signals_file.exists():
                with open(signals_file, 'r') as f:
                    return json.load(f)
            else:
                # Return empty list if no signals file exists yet
                logger.info(f"No signals file found at {signals_file}, returning empty list")
                return []
        except Exception as e:
            logger.error(f"Error loading signals: {e}")
            return []
    
    def get_dashboard_html(self):
        """Get HTML template for dashboard"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ProTradeAI Pro+ Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            color: white;
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: white;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #4CAF50;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(4px);
            border: 1px solid rgba(255, 255, 255, 0.18);
            text-align: center;
            transition: transform 0.3s ease;
        }
        
        .stat-card:hover {
            transform: translateY(-5px);
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 0.5rem;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9rem;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 2rem;
        }
        
        .signals-section {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        
        .performance-section {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
        }
        
        .section-title {
            font-size: 1.3rem;
            font-weight: bold;
            margin-bottom: 1rem;
            color: #333;
        }
        
        .filters {
            display: flex;
            gap: 1rem;
            margin-bottom: 1rem;
            flex-wrap: wrap;
        }
        
        .filter-select {
            padding: 0.5rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: white;
            color: #333;
        }
        
        .signals-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }
        
        .signals-table th,
        .signals-table td {
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        .signals-table th {
            background: #f8f9fa;
            font-weight: bold;
            color: #333;
        }
        
        .signal-long {
            color: #4CAF50;
            font-weight: bold;
        }
        
        .signal-short {
            color: #f44336;
            font-weight: bold;
        }
        
        .confidence-badge {
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
            font-size: 0.8rem;
            font-weight: bold;
        }
        
        .confidence-high {
            background: #4CAF50;
            color: white;
        }
        
        .confidence-medium {
            background: #FF9800;
            color: white;
        }
        
        .confidence-low {
            background: #f44336;
            color: white;
        }
        
        .chart-container {
            margin-top: 1rem;
            height: 300px;
        }
        
        .refresh-info {
            text-align: center;
            color: #666;
            font-size: 0.9rem;
            margin-top: 1rem;
        }
        
        .system-status {
            display: grid;
            gap: 0.5rem;
            margin-top: 1rem;
        }
        
        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 0.5rem;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .status-online {
            color: #4CAF50;
        }
        
        .status-offline {
            color: #f44336;
        }
        
        .no-signals {
            text-align: center;
            color: #666;
            padding: 2rem;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">🤖 ProTradeAI Pro+ Dashboard</div>
        <div class="status-indicator">
            <div class="status-dot"></div>
            <span>Live</span>
        </div>
    </div>
    
    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-number" id="total-signals">-</div>
                <div class="stat-label">Signals Today</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="avg-confidence">-</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="win-rate">-</div>
                <div class="stat-label">Win Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-number" id="active-symbols">-</div>
                <div class="stat-label">Active Symbols</div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="signals-section">
                <div class="section-title">📊 Recent Signals</div>
                <div class="filters">
                    <select class="filter-select" id="type-filter">
                        <option value="all">All Types</option>
                        <option value="long">LONG</option>
                        <option value="short">SHORT</option>
                    </select>
                    <select class="filter-select" id="timeframe-filter">
                        <option value="all">All Timeframes</option>
                        <option value="1h">1H</option>
                        <option value="4h">4H</option>
                        <option value="1d">1D</option>
                    </select>
                    <button class="filter-select" onclick="refreshData()" style="background: #667eea; color: white; border: none; cursor: pointer;">
                        🔄 Refresh
                    </button>
                </div>
                <div style="overflow-x: auto;">
                    <table class="signals-table">
                        <thead>
                            <tr>
                                <th>Time</th>
                                <th>Symbol</th>
                                <th>Type</th>
                                <th>Confidence</th>
                                <th>Leverage</th>
                                <th>Entry</th>
                                <th>TF</th>
                            </tr>
                        </thead>
                        <tbody id="signals-tbody">
                            <tr>
                                <td colspan="7" class="no-signals">Loading signals...</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div class="performance-section">
                <div class="section-title">⚡ Performance & Status</div>
                
                <div class="chart-container">
                    <canvas id="performance-chart"></canvas>
                </div>
                
                <div class="section-title" style="margin-top: 2rem;">🔧 System Status</div>
                <div class="system-status">
                    <div class="status-item">
                        <span>Bot Status</span>
                        <span id="bot-status" class="status-online">Loading...</span>
                    </div>
                    <div class="status-item">
                        <span>Model Loaded</span>
                        <span id="model-status" class="status-online">Loading...</span>
                    </div>
                    <div class="status-item">
                        <span>Telegram</span>
                        <span id="telegram-status" class="status-online">Loading...</span>
                    </div>
                    <div class="status-item">
                        <span>Uptime</span>
                        <span id="uptime">-</span>
                    </div>
                    <div class="status-item">
                        <span>Next Scan</span>
                        <span id="next-scan">-</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="refresh-info">
            ⏰ Auto-refresh every {{ refresh_interval }} seconds | Last updated: <span id="last-updated">-</span>
        </div>
    </div>

    <script>
        let performanceChart;
        
        function formatTime(timestamp) {
            if (!timestamp) return '-';
            const date = new Date(timestamp);
            return date.toLocaleTimeString();
        }
        
        function formatDate(timestamp) {
            if (!timestamp) return '-';
            const date = new Date(timestamp);
            return date.toLocaleDateString();
        }
        
        function getConfidenceClass(confidence) {
            if (confidence >= 80) return 'confidence-high';
            if (confidence >= 70) return 'confidence-medium';
            return 'confidence-low';
        }
        
        function updateStats(stats) {
            document.getElementById('total-signals').textContent = stats.total_signals_today || 0;
            document.getElementById('avg-confidence').textContent = (stats.avg_confidence || 0).toFixed(1) + '%';
            document.getElementById('active-symbols').textContent = stats.symbols_active || 0;
        }
        
        function updateSignalsTable(signals) {
            const tbody = document.getElementById('signals-tbody');
            tbody.innerHTML = '';
            
            if (signals.length === 0) {
                const row = tbody.insertRow();
                row.innerHTML = '<td colspan="7" class="no-signals">No signals generated yet. Start the bot to see signals here.</td>';
                return;
            }
            
            signals.slice(0, 20).forEach(signal => {
                const row = tbody.insertRow();
                row.innerHTML = `
                    <td>${formatTime(signal.timestamp)}</td>
                    <td><strong>${signal.symbol}</strong></td>
                    <td><span class="signal-${signal.signal_type?.toLowerCase()}">${signal.signal_type}</span></td>
                    <td><span class="confidence-badge ${getConfidenceClass(signal.confidence)}">${(signal.confidence || 0).toFixed(1)}%</span></td>
                    <td>${signal.leverage}x</td>
                    <td>$${(signal.current_price || 0).toFixed(4)}</td>
                    <td>${signal.timeframe}</td>
                `;
            });
        }
        
        function updatePerformanceChart(performance) {
            const ctx = document.getElementById('performance-chart').getContext('2d');
            
            if (performanceChart) {
                performanceChart.destroy();
            }
            
            performanceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: (performance.recent_performance || []).map(p => formatDate(p.date)),
                    datasets: [{
                        label: 'Daily Return %',
                        data: (performance.recent_performance || []).map(p => p.return),
                        borderColor: '#667eea',
                        backgroundColor: 'rgba(102, 126, 234, 0.1)',
                        tension: 0.4,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: false,
                            grid: {
                                color: 'rgba(0,0,0,0.1)'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(0,0,0,0.1)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }
        
        function updateSystemStatus(status) {
            document.getElementById('bot-status').textContent = status.bot_running ? 'Running' : 'Stopped';
            document.getElementById('bot-status').className = status.bot_running ? 'status-online' : 'status-offline';
            
            document.getElementById('model-status').textContent = status.model_loaded ? 'Online' : 'Offline';
            document.getElementById('model-status').className = status.model_loaded ? 'status-online' : 'status-offline';
            
            document.getElementById('telegram-status').textContent = status.telegram_connected ? 'Connected' : 'Disconnected';
            document.getElementById('telegram-status').className = status.telegram_connected ? 'status-online' : 'status-offline';
            
            document.getElementById('uptime').textContent = status.uptime || '-';
            document.getElementById('next-scan').textContent = status.next_scan ? formatTime(status.next_scan) : '-';
        }
        
        async function fetchData() {
            try {
                const typeFilter = document.getElementById('type-filter').value;
                const timeframeFilter = document.getElementById('timeframe-filter').value;
                
                // Fetch signals
                const signalsResponse = await fetch(`/api/signals?type=${typeFilter}&timeframe=${timeframeFilter}`);
                const signalsData = await signalsResponse.json();
                
                // Fetch stats
                const statsResponse = await fetch('/api/stats');
                const statsData = await statsResponse.json();
                
                // Fetch performance
                const performanceResponse = await fetch('/api/performance');
                const performanceData = await performanceResponse.json();
                
                // Fetch system status
                const statusResponse = await fetch('/api/system-status');
                const statusData = await statusResponse.json();
                
                if (signalsData.success) {
                    updateSignalsTable(signalsData.signals);
                }
                
                if (statsData.success) {
                    updateStats(statsData.stats);
                }
                
                if (performanceData.success) {
                    updatePerformanceChart(performanceData.performance);
                    document.getElementById('win-rate').textContent = (performanceData.performance.win_rate || 0).toFixed(1) + '%';
                }
                
                if (statusData.success) {
                    updateSystemStatus(statusData.status);
                }
                
                document.getElementById('last-updated').textContent = new Date().toLocaleTimeString();
                
            } catch (error) {
                console.error('Error fetching data:', error);
                // Show error in table
                const tbody = document.getElementById('signals-tbody');
                tbody.innerHTML = '<tr><td colspan="7" class="no-signals">Error loading data. Check console for details.</td></tr>';
            }
        }
        
        function refreshData() {
            fetchData();
        }
        
        // Event listeners
        document.getElementById('type-filter').addEventListener('change', fetchData);
        document.getElementById('timeframe-filter').addEventListener('change', fetchData);
        
        // Initial load
        fetchData();
        
        // Auto-refresh every {{ refresh_interval }} seconds
        setInterval(fetchData, {{ refresh_interval }} * 1000);
    </script>
</body>
</html>
        """
    
    def run(self, host=None, port=None, debug=None):
        """Run the dashboard server"""
        host = host or DASHBOARD_CONFIG['host']
        port = port or DASHBOARD_CONFIG['port']
        debug = debug or DASHBOARD_CONFIG['debug']
        
        logger.info(f"Starting dashboard on http://{host}:{port}")
        
        try:
            self.app.run(
                host=host,
                port=port,
                debug=debug,
                threaded=True
            )
        except Exception as e:
            logger.error(f"Error running dashboard: {e}")

def run_dashboard():
    """Run dashboard in separate thread"""
    dashboard = Dashboard()
    dashboard.run()

def main():
    """Main entry point for dashboard"""
    print("🌐 ProTradeAI Pro+ Dashboard")
    print("=" * 40)
    
    dashboard = Dashboard()
    dashboard.run()

if __name__ == "__main__":
    main()