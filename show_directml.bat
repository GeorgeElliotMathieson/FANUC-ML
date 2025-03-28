@echo off
REM show_directml.bat - Easy wrapper for demonstrating DirectML models
REM Usage: show_directml.bat [model_folder] [episodes] [viz_speed]

setlocal enabledelayedexpansion

echo.
echo ====================================================================
echo FANUC Robot - DirectML Model Demonstration (Enhanced Visualization)
echo ====================================================================
echo.

REM Set default values
set MODEL_FOLDER=ppo_directml_20250326_202801
set EPISODES=3
set VIZ_SPEED=0.03

REM Check for parameters
if not "%~1"=="" set MODEL_FOLDER=%~1
if not "%~2"=="" set EPISODES=%~2
if not "%~3"=="" set VIZ_SPEED=%~3

REM Check if model folder exists
if exist "./models/%MODEL_FOLDER%" (
    echo Using model from: models\%MODEL_FOLDER%
    echo Running %EPISODES% episodes with visualization speed %VIZ_SPEED%
    echo.
    echo Visualization includes:
    echo - Red sphere marking target position
    echo - Green line showing direction to target
    echo - Distance display in yellow text
    echo - Enhanced camera position and lighting
    echo.
    
    REM Run the demonstration
    python directml_show.py ./models/%MODEL_FOLDER%/final_model --episodes %EPISODES% --viz-speed %VIZ_SPEED%
) else (
    echo ERROR: Model folder "./models/%MODEL_FOLDER%" not found
    echo.
    echo Available model folders:
    dir /b /ad .\models\
    echo.
    echo Usage: show_directml.bat [model_folder] [episodes] [viz_speed]
    echo    model_folder: Name of model folder (default: ppo_directml_20250326_202801)
    echo    episodes: Number of episodes to run (default: 3)
    echo    viz_speed: Visualization speed (default: 0.03, higher = slower)
)

endlocal 