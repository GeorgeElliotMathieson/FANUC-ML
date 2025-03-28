@echo off

REM evaluate_directml.bat - Evaluate a DirectML model with more configuration options
REM Usage: evaluate_directml.bat [model_path] [episodes] [extra_args]

setlocal enabledelayedexpansion

REM Default values
set MODEL=../../models/ppo_directml_20250326_202801/final_model
set EPISODES=5
set EXTRA_ARGS=

REM Check for parameters
if not "%~1"=="" set MODEL=%~1
if not "%~2"=="" set EPISODES=%~2
if not "%~3"=="" set EXTRA_ARGS=%~3

echo.
echo ====================================================================
echo FANUC Robot - DirectML Model Evaluation
echo ====================================================================
echo.
echo Model: %MODEL%
echo Episodes: %EPISODES%
echo Extra arguments: %EXTRA_ARGS%
echo.

REM Run the evaluation script
python test_directml_model.py --model %MODEL% --episodes %EPISODES% %EXTRA_ARGS%

endlocal 