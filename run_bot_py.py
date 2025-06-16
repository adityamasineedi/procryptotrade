#!/usr/bin/env python3
"""
ProTradeAI Pro+ Bot Runner - WINDOWS COMPATIBLE VERSION
Handles bot startup with proper error handling and recovery (No Emoji)
"""

import os
import sys
import time
import subprocess
import logging
from pathlib import Path
from datetime import datetime

# Setup basic logging (Windows compatible - no emojis)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/startup.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories():
    """Ensure required directories exist"""
    directories = ['logs', 'data', 'models']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"[OK] Directory ensured: {directory}")

def check_environment():
    """Check if environment is properly configured"""
    env_file = Path('.env')
    if not env_file.exists():
        logger.error("[ERROR] .env file not found!")
        logger.info("[INFO] Please copy .env.template to .env and configure it")
        return False
    
    # Check for critical environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"[ERROR] Missing environment variables: {missing_vars}")
        logger.info("[INFO] Please configure these in your .env file")
        return False
    
    logger.info("[OK] Environment configuration validated")
    return True

def install_dependencies():
    """Install required dependencies"""
    try:
        logger.info("[INFO] Installing dependencies...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        logger.info("[OK] Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"[ERROR] Failed to install dependencies: {e}")
        return False

def run_bot():
    """Run the main bot"""
    try:
        logger.info("[START] Starting ProTradeAI Pro+ Bot...")
        
        # Import and run the main bot
        from main import main
        main()
        
    except KeyboardInterrupt:
        logger.info("[STOP] Bot stopped by user")
    except Exception as e:
        logger.error(f"[ERROR] Bot crashed: {e}")
        logger.info("[RESTART] Bot will restart in 10 seconds...")
        time.sleep(10)

def run_with_restart_protection():
    """Run bot with restart loop protection"""
    max_restarts = 5
    restart_count = 0
    start_time = time.time()
    
    while restart_count < max_restarts:
        try:
            logger.info(f"[BOT] Starting bot (attempt {restart_count + 1}/{max_restarts})")
            run_bot()
            
            # If we get here, bot stopped gracefully
            logger.info("[OK] Bot stopped gracefully")
            break
            
        except Exception as e:
            restart_count += 1
            logger.error(f"[ERROR] Bot crashed (restart {restart_count}/{max_restarts}): {e}")
            
            # Check if restarts are happening too quickly
            if time.time() - start_time < 300:  # 5 minutes
                logger.warning("[WARNING] Too many restarts in short time, increasing delay")
                time.sleep(30)
            else:
                # Reset counter if enough time has passed
                start_time = time.time()
                restart_count = 0
            
            if restart_count < max_restarts:
                logger.info(f"[RESTART] Restarting in 10 seconds...")
                time.sleep(10)
    
    if restart_count >= max_restarts:
        logger.error("[CRITICAL] Maximum restart attempts reached. Bot stopped.")
        logger.info("[INFO] Please check logs and fix any issues before restarting")

def main():
    """Main entry point"""
    print("ProTradeAI Pro+ Bot Runner (Windows Compatible)")
    print("=" * 60)
    
    # Ensure required directories exist
    ensure_directories()
    
    # Check environment configuration
    if not check_environment():
        print("\n[ERROR] Environment check failed!")
        print("[INFO] Please configure your .env file and try again")
        return 1
    
    # Install dependencies if needed
    if len(sys.argv) > 1 and sys.argv[1] == '--install-deps':
        if not install_dependencies():
            print("\n[ERROR] Dependency installation failed!")
            return 1
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'test':
            print("[TEST] Running in test mode...")
            os.system('python main.py test')
            return 0
            
        elif command == 'status':
            print("[STATUS] Getting bot status...")
            os.system('python main.py status')
            return 0
            
        elif command == 'scan':
            print("[SCAN] Running manual scan...")
            os.system('python main.py scan')
            return 0
            
        elif command == 'dashboard':
            print("[DASHBOARD] Starting dashboard...")
            os.system('python dashboard.py')
            return 0
            
        elif command == '--help':
            print("\n[HELP] Available commands:")
            print("  python run_bot_windows.py              - Start bot normally")
            print("  python run_bot_windows.py test         - Run test scan")
            print("  python run_bot_windows.py status       - Show bot status")
            print("  python run_bot_windows.py scan         - Run manual scan")
            print("  python run_bot_windows.py dashboard    - Start web dashboard")
            print("  python run_bot_windows.py --install-deps - Install dependencies")
            return 0
    
    # Normal bot operation
    print("[START] Starting bot with restart protection...")
    run_with_restart_protection()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())