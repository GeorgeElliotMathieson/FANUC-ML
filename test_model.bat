@echo off
REM Backward compatibility script for DirectML model testing
call legacy.bat test_model %*
exit /b %ERRORLEVEL% 