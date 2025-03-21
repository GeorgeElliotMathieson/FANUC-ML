# setup_amd_gpu_env.ps1
# PowerShell script to set up environment variables for AMD GPU usage with PyTorch

Write-Host "Setting up environment variables for AMD GPU (Radeon RX 6700S) support..." -ForegroundColor Green

# Set environment variables for the current session
$env:HIP_VISIBLE_DEVICES = "0"
$env:PYTORCH_HIP_ALLOC_CONF = "max_split_size_mb:128"

# Try to detect AMD GPU
try {
    $gpuInfo = Get-WmiObject -Class Win32_VideoController | Where-Object { $_.Name -match "AMD|Radeon" }
    if ($gpuInfo) {
        Write-Host "AMD GPU detected: $($gpuInfo.Name)" -ForegroundColor Green
    } else {
        Write-Host "No AMD GPU detected. Script will still set environment variables." -ForegroundColor Yellow
    }
} catch {
    Write-Host "Error detecting GPU: $_" -ForegroundColor Red
    Write-Host "Continuing with environment setup..." -ForegroundColor Yellow
}

# Check if PyTorch with ROCm is available
Write-Host "`nChecking PyTorch with ROCm support..." -ForegroundColor Cyan
$pythonCode = @"
import importlib.util
import sys

# Check if PyTorch is installed
torch_installed = importlib.util.find_spec("torch") is not None
if not torch_installed:
    print("PyTorch is not installed.")
    sys.exit(1)

import torch
print(f"PyTorch version: {torch.__version__}")

# Check for ROCm/HIP support
if hasattr(torch.version, 'hip') and torch.version.hip is not None:
    print(f"ROCm version: {torch.version.hip}")
    print("PyTorch has ROCm/HIP support: YES")
else:
    print("ROCm version: Not available")
    print("PyTorch has ROCm/HIP support: NO")
    print("To use AMD GPU acceleration, you need PyTorch with ROCm support.")
    print("Run the install_amd_gpu_support.py script for installation instructions.")
"@

python -c $pythonCode

Write-Host "`nEnvironment variables set for this PowerShell session:" -ForegroundColor Green
Write-Host "  HIP_VISIBLE_DEVICES = $env:HIP_VISIBLE_DEVICES" -ForegroundColor Cyan
Write-Host "  PYTORCH_HIP_ALLOC_CONF = $env:PYTORCH_HIP_ALLOC_CONF" -ForegroundColor Cyan

Write-Host "`nTo use AMD GPU for training, run:" -ForegroundColor Green
Write-Host "  python train_robot_rl_demo.py --use-amd --steps 10000 --parallel 8" -ForegroundColor White

Write-Host "`nNote: These environment variables are only set for the current PowerShell session." -ForegroundColor Yellow
Write-Host "To make them permanent, add them to your system environment variables." -ForegroundColor Yellow 