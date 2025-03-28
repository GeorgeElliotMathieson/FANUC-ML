@echo off
REM directml_summary.bat - Evaluate DirectML model performance with detailed metrics
REM Usage: directml_summary.bat [model_folder] [episodes]

setlocal enabledelayedexpansion

echo.
echo ====================================================================
echo FANUC Robot - DirectML Model Performance Summary
echo ====================================================================
echo.

REM Set default values
set MODEL_FOLDER=ppo_directml_20250326_202801
set EPISODES=10

REM Check for parameters
if not "%~1"=="" set MODEL_FOLDER=%~1
if not "%~2"=="" set EPISODES=%~2

REM Check if model folder exists
if exist "./models/%MODEL_FOLDER%" (
    echo Using model from: models\%MODEL_FOLDER%
    echo Evaluating with %EPISODES% episodes
    echo.
    
    REM Run the demonstration in headless mode with more episodes for statistical significance
    python directml_show.py ./models/%MODEL_FOLDER%/final_model --episodes %EPISODES% --no-gui
) else (
    echo ERROR: Model folder "./models/%MODEL_FOLDER%" not found
    echo.
    echo Available model folders:
    dir /b /ad .\models\
    echo.
    echo Usage: directml_summary.bat [model_folder] [episodes]
    echo    model_folder: Name of model folder (default: ppo_directml_20250326_202801)
    echo    episodes: Number of episodes to run (default: 10)
)

endlocal 