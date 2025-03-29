"""
Installation testing functionality for FANUC Robot ML Platform.
"""

import os
import sys
import importlib
import traceback
from src.core.utils import print_banner

def check_module(module_name):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError as e:
        print(f"Cannot import {module_name}: {e}")
        return False

def test_install():
    """
    Test the installation of FANUC-ML package and DirectML dependencies.
    
    Returns:
        0 if successful, 1 otherwise
    """
    print_banner("DirectML Installation Test")
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print()
    
    # Required core modules
    core_modules = [
        "src",
        "src.core",
        "src.envs",
        "src.utils",
        "src.dml"
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
    
    # DirectML is required
    directml_dependencies = [
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
    
    # Check DirectML
    directml_ok = True
    print("\nChecking DirectML support:")
    for module in directml_dependencies:
        if check_module(module):
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module}")
            directml_ok = False
    
    if not directml_ok:
        print("\nERROR: DirectML support is not available.")
        print("This implementation requires AMD GPU with DirectML support.")
        print("Install with: pip install torch-directml")
        return 1
    
    # Test DirectML
    print("\nTesting DirectML AMD GPU detection:")
    try:
        import torch
        import torch_directml
        
        # Check for DirectML devices
        device_count = torch_directml.device_count()
        print(f"  ✓ Found {device_count} DirectML device(s)")
        
        # Try to initialize a DirectML device
        device = torch_directml.device()
        print(f"  ✓ Successfully initialized DirectML device: {device}")
        
        # Check for AMD GPU info
        from src.dml import setup_directml
        dml_device = setup_directml()
        
        print("\nDirectML setup successful!")
        
    except Exception as e:
        print(f"\nERROR: DirectML initialization failed: {e}")
        if "trace" in locals():
            print(traceback.format_exc())
        return 1

    if all([core_ok, deps_ok, directml_ok]):
        print("\n✅ All checks passed! DirectML support is ready to use.")
        return 0
    else:
        print("\n⚠️ Some checks failed. Please fix the issues above before proceeding.")
        return 1 