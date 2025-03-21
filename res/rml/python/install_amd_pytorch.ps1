# install_amd_pytorch.ps1
# Script to install PyTorch with ROCm support for AMD GPUs

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Installing PyTorch with ROCm support for AMD GPUs" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Check if pip is available
$pipCheck = Get-Command pip -ErrorAction SilentlyContinue
if ($null -eq $pipCheck) {
    Write-Host "Error: pip not found. Please make sure Python is installed properly." -ForegroundColor Red
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

Write-Host "Current PyTorch installation:" -ForegroundColor Yellow
pip list | Select-String -Pattern "torch"
Write-Host ""

Write-Host "Uninstalling existing PyTorch..." -ForegroundColor Yellow
pip uninstall -y torch torchvision torchaudio
Write-Host ""

Write-Host "Installing PyTorch with ROCm support..." -ForegroundColor Yellow
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
Write-Host ""

Write-Host "Setting environment variables for AMD GPU..." -ForegroundColor Yellow
$env:HIP_VISIBLE_DEVICES = "0"
$env:PYTORCH_HIP_ALLOC_CONF = "max_split_size_mb:128"
Write-Host ""

Write-Host "Verifying installation..." -ForegroundColor Yellow
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'ROCm/HIP available: {hasattr(torch, \"hip\")}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else (torch.hip.device_count() if hasattr(torch, \"hip\") and hasattr(torch.hip, \"device_count\") else 0)}');"
Write-Host ""

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Green
Write-Host "1. Run 'set_permanent_amd_vars.ps1' as Administrator to set permanent environment variables" -ForegroundColor White
Write-Host "2. Test GPU performance with 'python test_amd_gpu_performance.py'" -ForegroundColor White
Write-Host "3. Train your model with 'python train_robot_rl_demo.py --steps 500 --parallel 1'" -ForegroundColor White
Write-Host "========================================================" -ForegroundColor Cyan

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 