#!/usr/bin/env python3
"""
Test script to verify the FANUC-ML installation is working correctly.
"""

import sys
import importlib
import traceback

def check_module(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError as e:
        print(f"Cannot import {module_name}: {e}")
        return False

def main():
    """Test the installation of FANUC-ML package and its dependencies."""
    print("Testing FANUC-ML installation...")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Required core modules
    core_modules = [
        "src",
        "src.core",
        "src.envs",
        "src.utils",
        "src.directml"
    ]
    
    # Dependencies
    dependencies = [
        "torch",
        "numpy",
        "pybullet",
        "gymnasium",
        "stable_baselines3",
        "matplotlib"
    ]
    
    # Optional dependencies
    optional_dependencies = [
        "torch_directml"
    ]
    
    # Check core modules
    print("Checking core modules:")
    core_ok = True
    for module in core_modules:
        if check_module(module):
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module}")
            core_ok = False
    
    if not core_ok:
        print("\nWARNING: Some core modules are not installed correctly.")
        print("Try reinstalling the package with: pip install -e .")
    else:
        print("\nAll core modules are installed correctly!")
    
    # Check required dependencies
    print("\nChecking required dependencies:")
    deps_ok = True
    for module in dependencies:
        if check_module(module):
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module}")
            deps_ok = False
    
    if not deps_ok:
        print("\nWARNING: Some dependencies are missing.")
        print("Try reinstalling with: pip install -r requirements.txt")
    else:
        print("\nAll required dependencies are installed correctly!")
    
    # Check optional dependencies
    print("\nChecking optional dependencies:")
    opt_ok = True
    for module in optional_dependencies:
        if check_module(module):
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module}")
            opt_ok = False
    
    if not opt_ok:
        print("\nNOTE: Some optional dependencies are not installed.")
        print("For DirectML support, install: pip install torch-directml")
    else:
        print("\nAll optional dependencies are installed!")
    
    # Try importing a key class
    print("\nTesting imports of key classes:")
    try:
        from src.envs.robot_sim import FANUCRobotEnv
        print("  ✓ Successfully imported FANUCRobotEnv")
    except ImportError:
        print("  ✗ Failed to import FANUCRobotEnv")
        print(traceback.format_exc())
    
    # Print summary
    print("\nInstallation test summary:")
    if core_ok and deps_ok:
        print("  ✓ Basic installation looks good!")
        if opt_ok:
            print("  ✓ DirectML support is available")
        else:
            print("  ✗ DirectML support is not available")
    else:
        print("  ✗ Installation has issues that need to be fixed")
    
    return 0 if core_ok and deps_ok else 1

if __name__ == "__main__":
    sys.exit(main()) 