@echo off

REM test_directml.bat - Test a DirectML model
REM Usage: test_directml.bat [model_path] [episodes]

setlocal enabledelayedexpansion

REM Default values
set MODEL=../../models/ppo_directml_20250326_202801/final_model
set EPISODES=1

REM Check for parameters
if not "%~1"=="" set MODEL=%~1
if not "%~2"=="" set EPISODES=%~2

echo.
echo ====================================================================
echo FANUC Robot - DirectML Model Test
echo ====================================================================
echo.
echo Model: %MODEL%
echo Episodes: %EPISODES%
echo.

REM Run the test script from its directory
python test_directml_model.py --model %MODEL% --episodes %EPISODES%

endlocal 