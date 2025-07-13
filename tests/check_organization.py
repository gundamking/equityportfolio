#!/usr/bin/env python3
"""
Project Organization Checker

Verifies that the project structure is properly organized and all components are in place.
"""

import os
import sys
from pathlib import Path

def check_directory_structure():
    """Check if all required directories exist."""
    required_dirs = [
        'src/portfolio_management',
        'notebooks',
        'examples', 
        'tests',
        'docs',
        'data'
    ]
    
    print("🔍 Checking Directory Structure...")
    all_good = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - MISSING")
            all_good = False
    
    return all_good

def check_core_modules():
    """Check if all core modules exist."""
    required_modules = [
        'src/portfolio_management/__init__.py',
        'src/portfolio_management/data_loader.py',
        'src/portfolio_management/strategies.py',
        'src/portfolio_management/portfolio_engine.py',
        'src/portfolio_management/performance_analytics.py',
        'src/portfolio_management/risk_management.py',
        'src/portfolio_management/backtester.py'
    ]
    
    print("\n🔍 Checking Core Modules...")
    all_good = True
    
    for module_path in required_modules:
        if os.path.exists(module_path):
            print(f"✅ {module_path}")
        else:
            print(f"❌ {module_path} - MISSING")
            all_good = False
    
    return all_good

def check_configuration_files():
    """Check if configuration files exist."""
    config_files = [
        'requirements.txt',
        'setup.py',
        'README.md',
        '.gitignore'
    ]
    
    print("\n🔍 Checking Configuration Files...")
    all_good = True
    
    for file_path in config_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_good = False
    
    return all_good

def check_example_files():
    """Check if example and demo files exist."""
    example_files = [
        'examples/simple_backtest.py',
        'notebooks/portfolio_analysis_demo.ipynb'
    ]
    
    print("\n🔍 Checking Example Files...")
    all_good = True
    
    for file_path in example_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_good = False
    
    return all_good

def check_data_files():
    """Check if sample data files exist."""
    expected_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NFLX', 'FB', 'ORCL', 'IBM', 'SAP']
    
    print("\n🔍 Checking Data Files...")
    found_files = 0
    
    for symbol in expected_symbols:
        file_path = f"data/{symbol}.csv"
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
            found_files += 1
        else:
            print(f"⚠️  {file_path} - Not found (optional)")
    
    print(f"📊 Found {found_files}/{len(expected_symbols)} data files")
    return found_files > 0  # At least some data files should exist

def check_imports():
    """Check if imports work correctly."""
    print("\n🔍 Checking Module Imports...")
    
    try:
        sys.path.insert(0, 'src')
        
        # Test main imports
        from portfolio_management import (
            DataLoader, 
            PortfolioEngine, 
            PerformanceAnalytics,
            MomentumStrategy,
            DividendStrategy,
            RiskManager,
            Backtester
        )
        
        print("✅ All main modules import successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_documentation():
    """Check if documentation files exist."""
    doc_files = [
        'PROJECT_SUMMARY.md',
        'docs/PROJECT_STRUCTURE.md'
    ]
    
    print("\n🔍 Checking Documentation...")
    all_good = True
    
    for file_path in doc_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            all_good = False
    
    return all_good

def main():
    """Run all organization checks."""
    print("🏦 Equity Portfolio Management System - Organization Check")
    print("=" * 60)
    
    checks = [
        ("Directory Structure", check_directory_structure),
        ("Core Modules", check_core_modules),
        ("Configuration Files", check_configuration_files),
        ("Example Files", check_example_files),
        ("Data Files", check_data_files),
        ("Module Imports", check_imports),
        ("Documentation", check_documentation)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error in {check_name}: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 ORGANIZATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
    
    print(f"\n🎯 Overall Score: {passed}/{total} checks passed")
    
    if passed == total:
        print("🏆 EXCELLENT! Project is perfectly organized!")
        return 0
    elif passed >= total * 0.8:
        print("👍 GOOD! Project is well organized with minor issues.")
        return 0
    else:
        print("⚠️  NEEDS WORK! Several organizational issues found.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 