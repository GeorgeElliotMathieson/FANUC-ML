#!/usr/bin/env python3
# install_amd_gpu_support.py
# Helper script to install PyTorch with AMD ROCm support

import os
import sys
import platform
import subprocess
import argparse

def check_os():
    """Check if the operating system is supported for ROCm."""
    system = platform.system().lower()
    if system != "linux" and system != "windows":
        print(f"WARNING: ROCm support is typically best on Linux. Your system ({system}) may have limited support.")
        return False
    return True

def check_python_version():
    """Check if the Python version is compatible with ROCm PyTorch."""
    major, minor, _ = sys.version_info[:3]
    if (major == 3 and minor >= 8) and major < 4:
        return True
    print(f"WARNING: PyTorch with ROCm works best with Python 3.8-3.11. You have Python {major}.{minor}")
    return False

def check_gpu():
    """Check if an AMD GPU is detected."""
    try:
        # Try to detect AMD GPU using different methods
        # Windows: Check for AMD GPU using wmic
        if platform.system().lower() == "windows":
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"], 
                capture_output=True, 
                text=True
            )
            output = result.stdout.lower()
            if "amd" in output or "radeon" in output:
                print("AMD GPU detected in system.")
                for line in output.splitlines():
                    if "amd" in line.lower() or "radeon" in line.lower():
                        print(f"  - {line.strip()}")
                return True
        # Linux: Check for AMD GPU using lspci
        elif platform.system().lower() == "linux":
            result = subprocess.run(
                ["lspci", "-nn", "|", "grep", "-i", "amd[/]ati"], 
                capture_output=True, 
                text=True, 
                shell=True
            )
            if result.stdout:
                print("AMD GPU detected in system:")
                print(result.stdout)
                return True
        
        print("WARNING: Could not detect AMD GPU. Please verify your hardware.")
        return False
    except Exception as e:
        print(f"Error detecting GPU: {e}")
        return False

def check_rocm():
    """Check if ROCm is already installed."""
    try:
        # Check for rocm-smi
        rocm_smi = subprocess.run(
            ["where", "rocm-smi"] if platform.system().lower() == "windows" else ["which", "rocm-smi"],
            capture_output=True,
            text=True
        )
        if rocm_smi.returncode == 0:
            print("ROCm tools detected in system.")
            return True
        return False
    except Exception:
        return False

def check_pytorch_rocm():
    """Check if PyTorch with ROCm support is installed."""
    try:
        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            print(f"PyTorch with ROCm support is already installed (version {torch.__version__})")
            return True
        print("PyTorch is installed but WITHOUT ROCm support.")
        return False
    except ImportError:
        print("PyTorch is not installed.")
        return False

def install_pytorch_rocm():
    """Install PyTorch with ROCm support."""
    print("\nInstalling PyTorch with ROCm support...")
    
    system = platform.system().lower()
    
    if system == "windows":
        print("\nROCm installation on Windows:")
        print("1. For AMD GPUs on Windows, PyTorch +ROCM wheels are not directly available via pip")
        print("2. Install a prebuilt wheel from the community or AMD")
        print("\nVisit these resources to download PyTorch with ROCm support:")
        print("- AMD ROCm documentation: https://rocmdocs.amd.com/en/latest/")
        print("- PyTorch ROCm info: https://pytorch.org/get-started/locally/")
        
        print("\nRecommended steps:")
        print("1. Download the PyTorch wheel compatible with ROCm from PyTorch website")
        print("2. Install the wheel using pip: pip install [downloaded-wheel-file]")
        print("3. Test installation with: python -c \"import torch; print(torch.version.hip)\"")
        
    elif system == "linux":
        # Linux installation
        print("\nROCm installation on Linux:")
        print("1. First install the ROCm platform")
        print("2. Then install PyTorch with ROCm support")
        
        print("\nTo install PyTorch with ROCm support, run:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7")
        
        # Offer to install PyTorch with ROCm support
        response = input("\nWould you like to install PyTorch with ROCm support now? (y/n): ")
        if response.lower() == 'y':
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", 
                     "--index-url", "https://download.pytorch.org/whl/rocm5.7"],
                    check=True
                )
                print("PyTorch with ROCm support has been installed.")
            except subprocess.CalledProcessError as e:
                print(f"Installation failed: {e}")
    else:
        print(f"No specific instructions available for {system}. Please visit the PyTorch website.")

def configure_environment():
    """Configure environment variables for ROCm."""
    if platform.system().lower() == "windows":
        print("\nEnvironment setup for Windows:")
        print("Add these environment variables to your system:")
        print("  HIP_VISIBLE_DEVICES=0")
        print("  PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128")
        
        print("\nYou can set them temporarily in PowerShell:")
        print("  $env:HIP_VISIBLE_DEVICES=0")
        print("  $env:PYTORCH_HIP_ALLOC_CONF=\"max_split_size_mb:128\"")
        
    else:  # Linux
        print("\nEnvironment setup for Linux:")
        print("Add these lines to your .bashrc or .zshrc:")
        print("  export HIP_VISIBLE_DEVICES=0")
        print("  export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128")
        
        print("\nOr set them temporarily:")
        print("  export HIP_VISIBLE_DEVICES=0")
        print("  export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128")

def test_pytorch_rocm():
    """Test PyTorch with ROCm support."""
    print("\nTesting PyTorch with ROCm support...")
    
    test_code = """
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"ROCm version: {torch.version.hip if hasattr(torch.version, 'hip') else 'Not available'}")
print(f"HIP available: {hasattr(torch, 'hip') and torch.hip.is_available() if hasattr(torch, 'hip') else False}")

if hasattr(torch, 'hip') and torch.hip.is_available():
    print(f"Number of AMD GPUs: {torch.hip.device_count()}")
    print(f"Current device: {torch.hip.current_device()}")
    x = torch.rand(5, 3).to('hip')
    print("Successfully created tensor on AMD GPU")
else:
    print("HIP/ROCm support not available in PyTorch")
    
print("\\nTo use AMD GPU acceleration with the robot positioning demo, run:")
print("python train_robot_rl_demo.py --use-amd --parallel 8")
"""
    
    try:
        subprocess.run([sys.executable, "-c", test_code], check=True)
    except subprocess.CalledProcessError:
        print("Test failed. PyTorch with ROCm support may not be properly installed.")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Install and configure PyTorch with AMD ROCm support')
    parser.add_argument('--install', action='store_true', help='Install PyTorch with ROCm support')
    parser.add_argument('--test', action='store_true', help='Test PyTorch with ROCm support')
    parser.add_argument('--all', action='store_true', help='Run all steps (check, install, configure, test)')
    return parser.parse_args()

def main():
    """Main function."""
    print("=== AMD GPU Support Setup Tool ===")
    print("This tool helps set up PyTorch with ROCm support for AMD GPUs like the Radeon RX 6700S\n")
    
    args = parse_args()
    
    # If no specific actions requested, run in interactive mode
    if not (args.install or args.test or args.all):
        print("Running in interactive mode...\n")
        
        # System compatibility check
        print("=== System Compatibility Check ===")
        os_supported = check_os()
        python_ok = check_python_version()
        gpu_detected = check_gpu()
        rocm_installed = check_rocm()
        pytorch_rocm_installed = check_pytorch_rocm()
        
        if args.install or (not pytorch_rocm_installed and input("\nWould you like to install PyTorch with ROCm support? (y/n): ").lower() == 'y'):
            install_pytorch_rocm()
            configure_environment()
        
        if args.test or input("\nWould you like to test PyTorch with ROCm support? (y/n): ").lower() == 'y':
            test_pytorch_rocm()
            
    else:
        # Run requested actions
        if args.all or args.install:
            # System compatibility check
            print("=== System Compatibility Check ===")
            check_os()
            check_python_version()
            check_gpu()
            check_rocm()
            check_pytorch_rocm()
            
            install_pytorch_rocm()
            configure_environment()
        
        if args.all or args.test:
            test_pytorch_rocm()
    
    print("\n=== Summary ===")
    print("To use your AMD Radeon RX 6700S GPU with the robot positioning demo:")
    print("1. Ensure PyTorch is installed with ROCm support")
    print("2. Set the appropriate environment variables")
    print("3. Run the demo with: python train_robot_rl_demo.py --use-amd --parallel 8")
    print("\nFor more information, refer to the README_DEMO.md file.")

if __name__ == "__main__":
    main() 