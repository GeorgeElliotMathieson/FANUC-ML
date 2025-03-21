# set_permanent_amd_vars.ps1
# Script to set permanent AMD GPU environment variables in Windows
# Must be run as Administrator

# Check if running as administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "This script needs to be run as Administrator to set system environment variables." -ForegroundColor Red
    Write-Host "Please right-click on PowerShell and select 'Run as Administrator', then run this script again." -ForegroundColor Yellow
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Setting Permanent AMD GPU Environment Variables" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Set permanent environment variables
Write-Host "Setting HIP_VISIBLE_DEVICES=0..." -ForegroundColor Yellow
[System.Environment]::SetEnvironmentVariable("HIP_VISIBLE_DEVICES", "0", [System.EnvironmentVariableTarget]::Machine)

Write-Host "Setting PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128..." -ForegroundColor Yellow
[System.Environment]::SetEnvironmentVariable("PYTORCH_HIP_ALLOC_CONF", "max_split_size_mb:128", [System.EnvironmentVariableTarget]::Machine)

# Verify environment variables were set
$hipDevices = [System.Environment]::GetEnvironmentVariable("HIP_VISIBLE_DEVICES", [System.EnvironmentVariableTarget]::Machine)
$hipAllocConf = [System.Environment]::GetEnvironmentVariable("PYTORCH_HIP_ALLOC_CONF", [System.EnvironmentVariableTarget]::Machine)

if ($hipDevices -eq "0" -and $hipAllocConf -eq "max_split_size_mb:128") {
    Write-Host "Environment variables successfully set!" -ForegroundColor Green
    Write-Host "HIP_VISIBLE_DEVICES = $hipDevices" -ForegroundColor Cyan
    Write-Host "PYTORCH_HIP_ALLOC_CONF = $hipAllocConf" -ForegroundColor Cyan
} else {
    Write-Host "Error: Failed to set environment variables." -ForegroundColor Red
}

Write-Host ""
Write-Host "Note: You may need to restart your computer for these changes to take effect." -ForegroundColor Yellow
Write-Host "After restarting, you can run training with:" -ForegroundColor Yellow
Write-Host "  python train_robot_rl_demo.py --use-amd --parallel 8" -ForegroundColor White
Write-Host ""

Write-Host "Press any key to exit..." -ForegroundColor Green
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown") 