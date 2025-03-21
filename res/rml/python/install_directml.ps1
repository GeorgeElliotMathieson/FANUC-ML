#!/usr/bin/env pwsh
# install_directml.ps1
# Script to install DirectML for AMD GPU support

Write-Host "=========================================================="
Write-Host "DirectML Installation Script for AMD GPU Support" -ForegroundColor Green
Write-Host "=========================================================="

# Check for administrator privileges
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

Write-Host "Checking Python installation..."
try {
    $pythonVersion = python --version
    Write-Host "Found $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or newer and try again."
    exit 1
}

Write-Host "Checking pip installation..."
try {
    $pipVersion = python -m pip --version
    Write-Host "Found pip: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "Installing pip..."
    Invoke-Expression "python -m ensurepip --upgrade"
}

# Install or upgrade pip
Write-Host "Upgrading pip..."
Invoke-Expression "python -m pip install --upgrade pip"

# Check if torch-directml is already installed
Write-Host "Checking for torch-directml..."
$dmlInstalled = $false
try {
    $dmlInfo = python -c "import torch_directml; print(f'torch-directml {torch_directml.__version__} installed')"
    Write-Host $dmlInfo -ForegroundColor Green
    $dmlInstalled = $true
} catch {
    Write-Host "torch-directml not installed, will install it now" -ForegroundColor Yellow
}

if (-not $dmlInstalled) {
    # Install torch-directml
    Write-Host "Installing torch-directml..."
    Invoke-Expression "python -m pip install torch-directml"
    
    # Verify installation
    try {
        $dmlInfo = python -c "import torch_directml; print(f'Successfully installed torch-directml {torch_directml.__version__}')"
        Write-Host $dmlInfo -ForegroundColor Green
        $dmlInstalled = $true
    } catch {
        Write-Host "ERROR: Failed to install torch-directml" -ForegroundColor Red
        Write-Host "Error details: $_" -ForegroundColor Red
        exit 1
    }
}

# Set environment variables for the current session
Write-Host "Setting environment variables for this session..."
$env:HIP_VISIBLE_DEVICES = "0"
$env:PYTORCH_HIP_ALLOC_CONF = "max_split_size_mb:128"

# Create a test script
$testScriptPath = Join-Path $PWD "test_directml.py"
Write-Host "Creating test script at $testScriptPath..."

@"
import torch
import os
import sys
import time

print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    import torch_directml
    print(f"DirectML version: {torch_directml.__version__}")
    
    # Create a DirectML device
    device = torch_directml.device()
    
    # Test with a simple matrix multiplication
    size = 2000
    print(f"Testing with {size}x{size} matrix multiplication...")
    
    # CPU timing
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    
    start_time = time.time()
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"CPU time: {cpu_time:.4f} seconds")
    
    # DirectML timing
    a_dml = a_cpu.to(device)
    b_dml = b_cpu.to(device)
    
    # Warmup
    _ = torch.mm(a_dml, b_dml)
    
    start_time = time.time()
    c_dml = torch.mm(a_dml, b_dml)
    _ = c_dml.to('cpu')  # Synchronize
    dml_time = time.time() - start_time
    print(f"DirectML time: {dml_time:.4f} seconds")
    
    speedup = cpu_time / dml_time
    print(f"Speedup: {speedup:.2f}x")
    
    if speedup > 1:
        print("SUCCESS: DirectML is working and providing acceleration")
    else:
        print("WARNING: DirectML is working but not providing acceleration")
        
except ImportError:
    print("ERROR: torch_directml not installed")
    print("Please run 'pip install torch-directml' to install it")
"@ | Set-Content $testScriptPath

Write-Host "=========================================================="
Write-Host "Installation complete!" -ForegroundColor Green
Write-Host "=========================================================="
Write-Host "To use AMD GPU acceleration with DirectML:"
Write-Host "1. Run your scripts with: python train_robot_rl_demo_directml.py --use-directml"
Write-Host "2. For demo mode: python train_robot_rl_demo_directml.py --demo --use-directml --load [model_path]"
Write-Host "3. For evaluation: python train_robot_rl_demo_directml.py --eval-only --use-directml --load [model_path]"
Write-Host ""
Write-Host "To test if DirectML is working correctly, run:"
Write-Host "python test_directml.py"
Write-Host "=========================================================="

# Ask if the user wants to run the test now
$runTest = Read-Host "Do you want to run the DirectML test now? (y/n)"
if ($runTest -eq "y" -or $runTest -eq "Y") {
    Write-Host "Running DirectML test..."
    Invoke-Expression "python $testScriptPath"
}

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 