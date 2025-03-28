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
    echo DirectML tools for FANUC Robot ML Platform
    echo Usage: directml.bat [install^|train^|eval^|test] [options]
    echo See fanuc_platform.py for detailed options
    exit /b 1
)

REM Set DirectML environment variable
set FANUC_DIRECTML=1

REM Forward to the unified FANUC platform script with DirectML flag
python fanuc_platform.py %command% --directml %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL% 