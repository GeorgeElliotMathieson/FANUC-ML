@echo off

REM eval.bat - Evaluation entry point for FANUC robot
REM Usage: eval.bat --load MODEL_PATH [options]

setlocal enabledelayedexpansion

echo.
echo ====================================================================
echo FANUC Robot - Evaluation Mode
echo ====================================================================
echo.

REM Run evaluation through the main entry point
python main.py --eval %*

endlocal 