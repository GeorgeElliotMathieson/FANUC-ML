@echo off
REM This is a backward compatibility script
REM Now that the platform is DirectML-only, this just forwards to fanuc.bat

call fanuc.bat %*
exit /b %ERRORLEVEL% 