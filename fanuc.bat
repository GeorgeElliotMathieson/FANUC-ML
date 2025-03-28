@echo off
setlocal

REM Check for Python
python --version > nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.7 or newer.
    exit /b 1
)

REM Get the command
set command=%1
if "%command%"=="" (
    echo FANUC Robot ML Platform (DirectML-Only)
    echo Usage: fanuc.bat [install^|train^|eval^|test] [options]
    echo See fanuc_platform.py for detailed options
    exit /b 1
)

REM Set DirectML environment variable to ensure DirectML is used
set FANUC_DIRECTML=1
set USE_DIRECTML=1
set USE_GPU=1

REM Forward to the unified FANUC platform script (DirectML-only version)
python fanuc_platform.py %command% %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL% 