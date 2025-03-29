@echo off
setlocal

REM ======================================================================
REM FANUC Robot ML Platform - Single Entry Point (DirectML Edition)
REM
REM This batch file is the single entry point for all platform operations
REM such as training, evaluation, testing, and installation verification.
REM ======================================================================

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
    echo.
    echo FANUC Robot ML Platform (DirectML Edition)
    echo ======================================
    echo.
    echo Usage: fanuc.bat [command] [options]
    echo.
    echo Commands:
    echo   install         - Test installation and verify DirectML availability
    echo   train           - Train a new model or continue training existing one
    echo   eval            - Run thorough evaluation on a model
    echo   test            - Run quick test on a model
    echo.
    echo For help with a specific command:
    echo   fanuc.bat [command] --help
    echo.
    echo Examples:
    echo   fanuc.bat install
    echo   fanuc.bat train --steps 1000000 --no-gui
    echo   fanuc.bat eval models/my_model --episodes 20
    echo   fanuc.bat test models/my_model
    echo.
    exit /b 1
)

REM Set DirectML environment variable to ensure DirectML is used
set FANUC_DIRECTML=1
set USE_DIRECTML=1
set USE_GPU=1

REM Forward to the unified FANUC platform script (DirectML-only version)
python fanuc_platform.py %command% %2 %3 %4 %5 %6 %7 %8 %9
exit /b %ERRORLEVEL% 