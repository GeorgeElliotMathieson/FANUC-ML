@echo off
REM Unified backward compatibility script for FANUC Robot ML Platform (DirectML-only)
REM This script handles all legacy batch file calls

setlocal EnableDelayedExpansion

REM Set DirectML environment variables
set FANUC_DIRECTML=1
set USE_DIRECTML=1
set USE_GPU=1

REM Detect which script called us based on the first argument
set SCRIPT_NAME=%1

REM Check if we have at least one argument after the script name
set HAS_ARGS=0
if not "%2"=="" set HAS_ARGS=1

if /i "%SCRIPT_NAME%"=="test_model" (
    if "%HAS_ARGS%"=="0" (
        echo Usage: test_model [model_path] [episodes] [options]
        echo DirectML model test utility for FANUC Robot ML Platform
        exit /b 1
    )
    echo Forwarding to DirectML test mode...
    call fanuc.bat test %2 %3 %4 %5 %6 %7 %8 %9
    exit /b %ERRORLEVEL%
)

if /i "%SCRIPT_NAME%"=="evaluate_model" (
    if "%HAS_ARGS%"=="0" (
        echo Usage: evaluate_model [model_path] [episodes] [options]
        echo DirectML model evaluation utility for FANUC Robot ML Platform
        exit /b 1
    )
    echo Forwarding to DirectML eval mode...
    call fanuc.bat eval %2 %3 %4 %5 %6 %7 %8 %9
    exit /b %ERRORLEVEL%
)

if /i "%SCRIPT_NAME%"=="test_directml" (
    if "%HAS_ARGS%"=="0" (
        echo Usage: test_directml [model_path] [episodes] [options]
        echo DirectML model test utility for FANUC Robot ML Platform
        exit /b 1
    )
    echo Forwarding to DirectML test mode...
    call fanuc.bat test %2 %3 %4 %5 %6 %7 %8 %9
    exit /b %ERRORLEVEL%
)

if /i "%SCRIPT_NAME%"=="evaluate_directml" (
    if "%HAS_ARGS%"=="0" (
        echo Usage: evaluate_directml [model_path] [episodes] [options]
        echo DirectML model evaluation utility for FANUC Robot ML Platform
        exit /b 1
    )
    echo Forwarding to DirectML eval mode...
    call fanuc.bat eval %2 %3 %4 %5 %6 %7 %8 %9
    exit /b %ERRORLEVEL%
)

REM Default case - just show help
echo FANUC Robot ML Platform - DirectML Legacy Script Handler
echo.
echo This script provides backward compatibility for older batch files.
echo It requires one of the following script names as first argument:
echo   test_model - Run quick tests on models with DirectML
echo   evaluate_model - Run thorough evaluation on models with DirectML
echo   test_directml - Run quick tests with DirectML
echo   evaluate_directml - Run thorough evaluation with DirectML
echo.
echo Or use the new unified interface directly:
echo   fanuc.bat - For DirectML accelerated operations
exit /b 1 