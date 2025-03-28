@echo off
REM Backward compatibility script for model evaluation
call legacy.bat evaluate_model %*
exit /b %ERRORLEVEL% 