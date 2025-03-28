@echo off
REM Backward compatibility script for model testing
call legacy.bat test_model %*
exit /b %ERRORLEVEL% 