"""
Script to install PyTorch with ROCm support for AMD GPUs
"""
import os
import sys
import subprocess
import platform

def main():
    print("Installing PyTorch with ROCm support for AMD GPUs...")
    
    # Check if we're on Windows
    if platform.system() == "Windows":
        print("Warning: ROCm support on Windows is limited. You might need to use WSL2 for full AMD GPU support.")
        print("Installing PyTorch with DirectML support instead, which can work with AMD GPUs on Windows.")
        
        # Install PyTorch with DirectML support
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", "torch-directml"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        print("\nInstallation completed. Now let's verify the installation...")
        
        # Create a verification script
        verify_script = """
import torch
import torch_directml
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"DirectML available: {torch_directml.is_available()}")

if torch_directml.is_available():
    # Create a DirectML device
    device = torch_directml.device()
    print(f"DirectML device: {device}")
    
    # Try a simple operation on the GPU
    x = torch.randn(3, 3, device=device)
    y = torch.randn(3, 3, device=device)
    z = x @ y
    print(f"Matrix multiplication result shape: {z.shape}")
    print(f"Result device: {z.device}")
    print("DirectML is working correctly with your AMD GPU!")
else:
    print("DirectML is not available. Please check your installation.")
"""
        
        with open("verify_directml.py", "w") as f:
            f.write(verify_script)
        
        print("\nTo verify the installation, run: python verify_directml.py")
        
    else:  # Linux
        print("Installing PyTorch with ROCm support for AMD GPUs on Linux...")
        
        # Install PyTorch with ROCm support
        cmd = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/rocm5.6"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        
        print("\nInstallation completed. Now let's verify the installation...")
        
        # Create a verification script
        verify_script = """
import torch
import sys

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

if hasattr(torch.version, 'hip'):
    print(f"ROCm version: {torch.version.hip}")

# Check if ROCm/HIP is available
if torch.cuda.is_available():  # PyTorch uses cuda namespace for both CUDA and ROCm
    print(f"GPU available: Yes")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Try a simple operation on the GPU
    device = torch.device("cuda")
    x = torch.randn(3, 3, device=device)
    y = torch.randn(3, 3, device=device)
    z = x @ y
    print(f"Matrix multiplication result shape: {z.shape}")
    print(f"Result device: {z.device}")
    print("ROCm is working correctly with your AMD GPU!")
else:
    print("GPU is not available. Please check your installation.")
"""
        
        with open("verify_rocm.py", "w") as f:
            f.write(verify_script)
        
        print("\nTo verify the installation, run: python verify_rocm.py")
    
    print("\nAfter verifying the installation, you can run your curriculum_training.py script again.")
    print("It should automatically detect and use your AMD GPU.")

if __name__ == "__main__":
    main() 