@echo off
REM Backward compatibility script for DirectML model evaluation
call legacy.bat evaluate_model %*
exit /b %ERRORLEVEL% 