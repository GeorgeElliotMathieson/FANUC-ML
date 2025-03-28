@echo off

REM train.bat - Training entry point for FANUC robot
REM Usage: train.bat [options]

setlocal enabledelayedexpansion

echo.
echo ====================================================================
echo FANUC Robot - Training Mode
echo ====================================================================
echo.

REM Run training through the main entry point
python main.py --train %*

endlocal 