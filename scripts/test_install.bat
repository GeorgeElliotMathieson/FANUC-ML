@echo off

REM test_install.bat - Test the FANUC-ML installation
REM Usage: test_install.bat

setlocal enabledelayedexpansion

echo.
echo ====================================================================
echo FANUC Robot ML - Installation Test
echo ====================================================================
echo.

REM Run the test script
python tools/test_install.py

REM Check exit code
if %errorlevel% neq 0 (
    echo.
    echo Installation test failed. Please check the error messages above.
    exit /b 1
)

echo.
echo Installation test completed successfully.
echo.

endlocal 