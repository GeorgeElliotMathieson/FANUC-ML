@echo off

REM evaluate_directml.bat - Easy DirectML model evaluation
REM Usage: evaluate_directml.bat [model_path] [episodes] [options]

setlocal enabledelayedexpansion

REM Default values
set MODEL=./models/ppo_directml_20250326_202801/final_model
set EPISODES=5
set OPTIONS=

REM Check for parameters
if not "%~1"=="" set MODEL=%~1
if not "%~2"=="" set EPISODES=%~2
if not "%~3"=="" set OPTIONS=%~3

echo.
echo ====================================================================
echo FANUC Robot - DirectML Model Evaluation
echo ====================================================================
echo.
echo Model: %MODEL%
echo Episodes: %EPISODES%
echo Options: %OPTIONS%
echo.

REM Run the evaluation
python main.py --eval --load %MODEL% --eval-episodes %EPISODES% --directml %OPTIONS%

endlocal 