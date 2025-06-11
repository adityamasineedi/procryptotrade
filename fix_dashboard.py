#!/usr/bin/env python3
"""
Quick fix for dashboard template issue
Replace the index route in dashboard.py
"""

import os
import shutil
from pathlib import Path

def fix_dashboard_template():
    """Fix the dashboard template issue"""
    dashboard_file = Path('dashboard.py')
    
    if not dashboard_file.exists():
        print("❌ dashboard.py not found!")
        return False
    
    # Read the current file
    with open(dashboard_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create backup
    backup_file = Path('dashboard.py.backup')
    shutil.copy2(dashboard_file, backup_file)
    print(f"✅ Backup created: {backup_file}")
    
    # Fix the template issue
    old_line = "return render_template_string(self.get_dashboard_html())"
    new_line = """return render_template_string(
                self.get_dashboard_html(), 
                refresh_interval=DASHBOARD_CONFIG['refresh_interval']
            )"""
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        print("✅ Fixed render_template_string call")
    else:
        print("⚠️  render_template_string line not found - might already be fixed")
    
    # Also fix the template string replacement issue
    old_template_fix = """.replace('{{ refresh_interval }}', str(DASHBOARD_CONFIG['refresh_interval']))"""
    
    if old_template_fix in content:
        content = content.replace(old_template_fix, "")
        print("✅ Removed unnecessary template replacement")
    
    # Write the fixed content
    with open(dashboard_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Dashboard.py has been fixed!")
    return True

def test_dashboard_fix():
    """Test if the dashboard fix works"""
    try:
        from dashboard import Dashboard
        from config import DASHBOARD_CONFIG
        
        dashboard = Dashboard()
        print("✅ Dashboard import successful")
        print(f"✅ Refresh interval: {DASHBOARD_CONFIG['refresh_interval']} seconds")
        
        # Test template rendering
        html = dashboard.get_dashboard_html()
        if "{{ refresh_interval }}" in html:
            print("⚠️  Template still contains unresolved variables")
            return False
        else:
            print("✅ Template variables look good")
            return True
            
    except Exception as e:
        print(f"❌ Dashboard test failed: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 ProTradeAI Pro+ Dashboard Quick Fix")
    print("=" * 50)
    
    # Step 1: Fix the dashboard file
    print("1. Fixing dashboard.py...")
    fix_success = fix_dashboard_template()
    
    if not fix_success:
        print("❌ Failed to fix dashboard.py")
        return 1
    
    # Step 2: Test the fix
    print("\n2. Testing the fix...")
    test_success = test_dashboard_fix()
    
    # Summary
    print("\n" + "=" * 50)
    print("🔧 QUICK FIX SUMMARY")
    print("=" * 50)
    
    if fix_success and test_success:
        print("✅ SUCCESS! Dashboard should now work!")
        print("\n🚀 Next steps:")
        print("   1. Stop any running dashboard (Ctrl+C)")
        print("   2. Run: python dashboard.py")
        print("   3. Open: http://localhost:5000")
        print("   4. Dashboard should load without errors!")
        return 0
    else:
        print("❌ Some issues remain. Check the errors above.")
        if Path('dashboard.py.backup').exists():
            print("💾 Backup available at: dashboard.py.backup")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())