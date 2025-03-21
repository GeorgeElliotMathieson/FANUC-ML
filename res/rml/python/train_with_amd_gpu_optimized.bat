@echo off
echo Setting up environment for optimized AMD GPU training...

REM Set environment variables for AMD GPU
set HIP_VISIBLE_DEVICES=0
set PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:256
set DML_PREFETCH_BUFFERS=1

REM Set thread count based on system (use half of logical cores)
for /f "tokens=2 delims==" %%i in ('wmic cpu get NumberOfLogicalProcessors /value') do set CORES=%%i
set /a DML_THREAD_COUNT=%CORES% / 2
if %DML_THREAD_COUNT% LEQ 0 set DML_THREAD_COUNT=4
set DML_THREAD_COUNT=%DML_THREAD_COUNT%

echo.
echo Using AMD GPU optimization settings:
echo HIP_VISIBLE_DEVICES=%HIP_VISIBLE_DEVICES%
echo PYTORCH_HIP_ALLOC_CONF=%PYTORCH_HIP_ALLOC_CONF%
echo DML_PREFETCH_BUFFERS=%DML_PREFETCH_BUFFERS%
echo DML_THREAD_COUNT=%DML_THREAD_COUNT%
echo.

REM Set up parameters with defaults that can be overridden by command-line args
set STEPS=10000
set PARALLEL=8
set MODEL_PATH=
set EVAL_EPISODES=5
set MODE=train
set VIZ_SPEED=0.05

REM Parse command-line arguments
:parse_args
if "%~1"=="" goto :end_parse
if /i "%~1"=="--train" set MODE=train & shift & goto :parse_args
if /i "%~1"=="--demo" set MODE=demo & shift & goto :parse_args
if /i "%~1"=="--eval" set MODE=eval & shift & goto :parse_args
if /i "%~1"=="--steps" set STEPS=%~2 & shift & shift & goto :parse_args
if /i "%~1"=="--parallel" set PARALLEL=%~2 & shift & shift & goto :parse_args
if /i "%~1"=="--model" set MODEL_PATH=%~2 & shift & shift & goto :parse_args
if /i "%~1"=="--episodes" set EVAL_EPISODES=%~2 & shift & shift & goto :parse_args
if /i "%~1"=="--viz-speed" set VIZ_SPEED=%~2 & shift & shift & goto :parse_args
shift
goto :parse_args
:end_parse

REM Verify required arguments based on mode
if /i "%MODE%"=="demo" (
    if "%MODEL_PATH%"=="" (
        echo Error: Demo mode requires --model parameter
        goto :usage
    )
) else if /i "%MODE%"=="eval" (
    if "%MODEL_PATH%"=="" (
        echo Error: Evaluation mode requires --model parameter
        goto :usage
    )
)

REM Run the script with the appropriate mode
if /i "%MODE%"=="train" (
    echo Starting optimized training with AMD GPU acceleration...
    echo Training for %STEPS% steps with %PARALLEL% parallel environments
    python train_robot_rl_demo_directml.py --steps %STEPS% --parallel %PARALLEL%
) else if /i "%MODE%"=="demo" (
    echo Starting demo with AMD GPU acceleration...
    echo Using model: %MODEL_PATH%
    python train_robot_rl_demo_directml.py --demo --load %MODEL_PATH% --viz-speed %VIZ_SPEED%
) else if /i "%MODE%"=="eval" (
    echo Starting evaluation with AMD GPU acceleration...
    echo Evaluating model: %MODEL_PATH% for %EVAL_EPISODES% episodes
    python train_robot_rl_demo_directml.py --eval-only --load %MODEL_PATH% --eval-episodes %EVAL_EPISODES%
)

echo.
echo Operation complete.
goto :end

:usage
echo.
echo Usage:
echo train_with_amd_gpu_optimized.bat [mode] [options]
echo.
echo Modes:
echo   --train          Train a new model (default)
echo   --demo           Run demo with existing model
echo   --eval           Evaluate existing model
echo.
echo Options:
echo   --steps N        Number of training steps (default: 10000)
echo   --parallel N     Number of parallel environments (default: 8)
echo   --model PATH     Path to model (required for --demo and --eval)
echo   --episodes N     Number of evaluation episodes (default: 5)
echo   --viz-speed N    Visualization speed for demo (default: 0.05)
echo.
echo Examples:
echo   train_with_amd_gpu_optimized.bat --train --steps 20000 --parallel 8
echo   train_with_amd_gpu_optimized.bat --demo --model ./models/revamped_20250321_190817/final_model
echo   train_with_amd_gpu_optimized.bat --eval --model ./models/revamped_20250321_190817/final_model --episodes 10
echo.

:end
echo Press any key to exit...
pause > nul 