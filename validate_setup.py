#!/usr/bin/env python3
"""
Validation script to check if all dependencies and setup are correct
This script is for GitHub Actions validation only - do not run locally
"""

import sys
import importlib

def check_imports():
    """Check if all required packages can be imported"""
    required_packages = [
        'pandas',
        'numpy',
        'lightgbm',
        'skforecast',
        'gspread',
        'google.oauth2.service_account',
        'sklearn',
        'datetime'
    ]
    
    print("Checking package imports...")
    print("-" * 40)
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if '.' in package:
                # Handle nested imports
                module_parts = package.split('.')
                module = importlib.import_module('.'.join(module_parts[:-1]))
                getattr(module, module_parts[-1])
            else:
                importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package} - {str(e)}")
            missing_packages.append(package)
    
    print("-" * 40)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("All packages imported successfully!")
        return True

def check_environment():
    """Check environment variables and setup"""
    import os
    
    print("\nChecking environment setup...")
    print("-" * 40)
    
    # Check for Google Sheets credentials
    creds = os.environ.get('GOOGLE_SHEETS_CREDENTIALS')
    if creds:
        print("✓ GOOGLE_SHEETS_CREDENTIALS environment variable found")
        
        # Try to parse as JSON
        try:
            import json
            json.loads(creds)
            print("✓ GOOGLE_SHEETS_CREDENTIALS is valid JSON")
        except json.JSONDecodeError:
            print("✗ GOOGLE_SHEETS_CREDENTIALS is not valid JSON")
            return False
    else:
        print("✗ GOOGLE_SHEETS_CREDENTIALS environment variable not found")
        print("  This is expected for local validation, but required for GitHub Actions")
    
    print("-" * 40)
    return True

def validate_forecasting_script():
    """Validate the main forecasting script structure"""
    print("\nValidating forecasting script...")
    print("-" * 40)
    
    try:
        # Import the main script
        import forecast_fieldwork
        
        # Check if main class exists
        if hasattr(forecast_fieldwork, 'FieldworkForecaster'):
            print("✓ FieldworkForecaster class found")
        else:
            print("✗ FieldworkForecaster class not found")
            return False
            
        # Check if main function exists
        if hasattr(forecast_fieldwork, 'main'):
            print("✓ main function found")
        else:
            print("✗ main function not found")
            return False
            
        print("✓ Forecasting script structure is valid")
        return True
        
    except ImportError as e:
        print(f"✗ Could not import forecasting script: {str(e)}")
        return False

def main():
    """Main validation function"""
    print("Fieldwork Forecasting - Setup Validation")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Check package imports
    if not check_imports():
        all_checks_passed = False
    
    # Check environment
    if not check_environment():
        all_checks_passed = False
    
    # Validate forecasting script
    if not validate_forecasting_script():
        all_checks_passed = False
    
    print("\n" + "=" * 60)
    
    if all_checks_passed:
        print("✓ All validation checks passed!")
        print("The forecasting system is ready to run.")
        sys.exit(0)
    else:
        print("✗ Some validation checks failed!")
        print("Please fix the issues before running the forecasting system.")
        sys.exit(1)

if __name__ == "__main__":
    main()
