#!/usr/bin/env python3
"""
ProTradeAI Pro+ Setup Verification Script
Verify Python 3.11 compatibility and package installation
"""

import sys
import pkg_resources
import os
from pathlib import Path

def check_python_version():
    """Check if Python 3.11+ is being used"""
    print("🐍 Checking Python Version...")
    version = sys.version_info
    
    if version < (3, 11):
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} detected")
        print("⚠️  Python 3.11+ is required for optimal performance")
        print("📥 Please upgrade to Python 3.11 or higher")
        return False
    else:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Perfect!")
        if version >= (3, 11):
            print("🚀 Python 3.11+ optimizations available!")
        return True

def check_required_packages():
    """Check if all required packages are installed"""
    print("\n📦 Checking Required Packages...")
    
    required_packages = [
        ('pandas', '2.2.0'),
        ('numpy', '1.26.0'),
        ('scikit-learn', '1.4.0'),
        ('xgboost', '2.0.0'),
        ('flask', '3.0.0'),
        ('requests', '2.32.0'),
        ('APScheduler', '3.10.0'),
        ('python-dotenv', '1.0.0'),
        ('ta', '0.11.0'),
        ('joblib', '1.4.0')
    ]
    
    all_installed = True
    
    for package_name, min_version in required_packages:
        try:
            installed_version = pkg_resources.get_distribution(package_name).version
            
            # Simple version comparison (works for most cases)
            if installed_version >= min_version:
                print(f"✅ {package_name}: {installed_version}")
            else:
                print(f"⚠️  {package_name}: {installed_version} (recommend {min_version}+)")
                
        except pkg_resources.DistributionNotFound:
            print(f"❌ {package_name}: Not installed")
            all_installed = False
    
    return all_installed

def check_project_structure():
    """Check if all required files are present"""
    print("\n📁 Checking Project Structure...")
    
    required_files = [
        'main.py',
        'strategy_ai.py', 
        'notifier.py',
        'dashboard.py',
        'config.py',
        'retrain_model.py',
        'requirements.txt',
        '.env.template',
        'README.md'
    ]
    
    missing_files = []
    
    for file_name in required_files:
        if Path(file_name).exists():
            print(f"✅ {file_name}")
        else:
            print(f"❌ {file_name} - Missing!")
            missing_files.append(file_name)
    
    return len(missing_files) == 0

def check_environment_config():
    """Check environment configuration"""
    print("\n⚙️ Checking Environment Configuration...")
    
    env_file = Path('.env')
    env_template = Path('.env.template')
    
    if not env_template.exists():
        print("❌ .env.template not found")
        return False
    else:
        print("✅ .env.template found")
    
    if env_file.exists():
        print("✅ .env file exists")
        
        # Check if basic variables are set
        try:
            with open('.env', 'r') as f:
                content = f.read()
                
            required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID', 'TRADING_CAPITAL']
            configured_vars = []
            
            for var in required_vars:
                if f"{var}=" in content and "your_" not in content.split(f"{var}=")[1].split('\n')[0]:
                    configured_vars.append(var)
                    print(f"✅ {var} configured")
                else:
                    print(f"⚠️  {var} needs configuration")
            
            return len(configured_vars) == len(required_vars)
            
        except Exception as e:
            print(f"❌ Error reading .env: {e}")
            return False
    else:
        print("⚠️  .env file not found - copy from .env.template")
        return False

def test_imports():
    """Test if core modules can be imported"""
    print("\n🔧 Testing Core Module Imports...")
    
    test_modules = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', 'scikit-learn'),
        ('xgboost', 'xgb'),
        ('flask', 'Flask'),
        ('requests', 'requests'),
        ('ta', 'technical analysis')
    ]
    
    all_imports_ok = True
    
    for module, description in test_modules:
        try:
            __import__(module)
            print(f"✅ {description}")
        except ImportError as e:
            print(f"❌ {description}: {e}")
            all_imports_ok = False
    
    return all_imports_ok

def performance_test():
    """Quick performance test for Python 3.11"""
    print("\n⚡ Running Python 3.11 Performance Test...")
    
    try:
        import time
        import numpy as np
        
        # Simple computation test
        start_time = time.time()
        
        # Matrix operations
        a = np.random.random((1000, 1000))
        b = np.random.random((1000, 1000))
        c = np.dot(a, b)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Matrix computation: {duration:.3f} seconds")
        
        if duration < 1.0:
            print("🚀 Excellent performance!")
        elif duration < 2.0:
            print("👍 Good performance")
        else:
            print("⚠️  Consider upgrading hardware or Python version")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🔍 ProTradeAI Pro+ Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Project Structure", check_project_structure),
        ("Environment Config", check_environment_config),
        ("Module Imports", test_imports),
        ("Performance Test", performance_test)
    ]
    
    results = {}
    
    for check_name, check_function in checks:
        try:
            results[check_name] = check_function()
        except Exception as e:
            print(f"❌ {check_name} failed: {e}")
            results[check_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 VERIFICATION SUMMARY")
    print("=" * 50)
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\n📊 Result: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("\n🎉 SUCCESS! ProTradeAI Pro+ is ready to run!")
        print("🚀 Next steps:")
        print("   1. Configure your .env file if not done")
        print("   2. Run: python main.py")
        print("   3. Access dashboard: python dashboard.py")
        return 0
    else:
        print("\n⚠️  ISSUES DETECTED - Please fix the failed checks above")
        print("📚 Check the README.md for detailed setup instructions")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
