#!/usr/bin/env python3
# install_amd_support.py - Script to install PyTorch with AMD GPU support

import os
import sys
import subprocess
import platform

def main():
    print("=" * 80)
    print("Installing PyTorch with AMD GPU (ROCm) support".center(80))
    print("=" * 80)
    
    # Check if we're on Windows
    if platform.system() != "Windows":
        print("This script is designed for Windows. For other platforms, please follow the instructions at:")
        print("https://pytorch.org/get-started/locally/")
        return
    
    # Check if we have an AMD GPU
    try:
        result = subprocess.run(['wmic', 'path', 'win32_VideoController', 'get', 'name'], 
                               capture_output=True, text=True, check=True)
        output = result.stdout.lower()
        
        has_amd_gpu = 'amd' in output or 'radeon' in output
        
        if not has_amd_gpu:
            print("No AMD GPU detected. This script is intended for AMD GPUs.")
            print("If you have an AMD GPU, please make sure it's properly installed and recognized by Windows.")
            return
        
        print("AMD GPU detected. Proceeding with installation...")
        
        # Uninstall existing PyTorch
        print("\nStep 1: Uninstalling existing PyTorch installation...")
        subprocess.run([
            sys.executable, "-m", "pip", "uninstall", "-y", 
            "torch", "torchvision", "torchaudio"
        ], check=True)
        
        # Install PyTorch with ROCm support
        print("\nStep 2: Installing PyTorch with ROCm support...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/rocm5.6"
        ], check=True)
        
        # Verify installation
        print("\nStep 3: Verifying installation...")
        try:
            # Create a temporary script to check if PyTorch can access the GPU
            verify_script = """
import torch
print(f"PyTorch version: {torch.__version__}")
if hasattr(torch, '_C') and hasattr(torch._C, '_ROCM_VERSION'):
    print(f"PyTorch built with ROCm support (version: {torch._C._ROCM_VERSION})")
else:
    print("PyTorch NOT built with ROCm support")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # Test GPU with a small tensor operation
    print("Testing GPU with tensor operations...")
    x = torch.ones(1000, 1000, device='cuda')
    y = x + x
    print("GPU tensor operations successful!")
else:
    print("GPU not available to PyTorch")
"""
            with open("verify_pytorch.py", "w") as f:
                f.write(verify_script)
            
            # Run the verification script
            subprocess.run([sys.executable, "verify_pytorch.py"], check=True)
            
            # Clean up
            os.remove("verify_pytorch.py")
            
        except Exception as e:
            print(f"Error during verification: {e}")
        
        print("\nInstallation complete!")
        print("If your GPU is still not detected, you may need to:")
        print("1. Update your AMD drivers from: https://www.amd.com/en/support")
        print("2. Restart your computer")
        print("3. Check if your GPU model is supported by PyTorch+ROCm")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 