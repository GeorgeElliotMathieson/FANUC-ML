@echo off

REM demo.bat - Demo entry point for FANUC robot
REM Usage: demo.bat --load MODEL_PATH [options]

setlocal enabledelayedexpansion

echo.
echo ====================================================================
echo FANUC Robot - Demonstration Mode
echo ====================================================================
echo.

REM Run demo through the main entry point
python main.py --demo --viz-speed 0.02 %*

endlocal 