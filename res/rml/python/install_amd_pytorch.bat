@echo off
REM install_amd_pytorch.bat
REM Script to install PyTorch with ROCm support for AMD GPUs

echo ========================================================
echo Installing PyTorch with ROCm support for AMD GPUs
echo ========================================================
echo.

REM Check if pip is available
where pip >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Error: pip not found. Please make sure Python is installed properly.
    pause
    exit /b 1
)

echo Current PyTorch installation:
pip list | findstr torch
echo.

echo Uninstalling existing PyTorch...
pip uninstall -y torch torchvision torchaudio
echo.

echo Installing PyTorch with ROCm support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
echo.

echo Setting environment variables for AMD GPU...
set HIP_VISIBLE_DEVICES=0
set PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
echo.

echo Verifying installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'ROCm/HIP available: {hasattr(torch, \"hip\")}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else (torch.hip.device_count() if hasattr(torch, \"hip\") and hasattr(torch.hip, \"device_count\") else 0)}');"
echo.

echo ========================================================
echo Next steps:
echo 1. Run 'set_permanent_amd_vars.ps1' as Administrator to set permanent environment variables
echo 2. Test GPU performance with 'python test_amd_gpu_performance.py'
echo 3. Train your model with 'python train_robot_rl_demo.py --steps 500 --parallel 1'
echo ========================================================

pause 