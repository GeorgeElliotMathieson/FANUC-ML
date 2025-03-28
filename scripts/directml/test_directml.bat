@echo off

REM test_directml.bat - Test a model (requires AMD GPU with DirectML)
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
echo FANUC Robot - DirectML Model Test (Requires AMD GPU)
echo ====================================================================
echo.
echo Model: %MODEL%
echo Episodes: %EPISODES%
echo.

REM Run the test script from its directory
python test_directml_model.py --model %MODEL% --episodes %EPISODES%
if errorlevel 1 (
    echo.
    echo ERROR: Test failed with error code %ERRORLEVEL%
    echo This script requires DirectML and AMD GPU support.
    echo Make sure torch-directml is installed: pip install torch-directml
    exit /b 1
)

endlocal 