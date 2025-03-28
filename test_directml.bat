@echo off

REM test_directml.bat - Root convenience script
REM Forwards calls to scripts/directml/test_directml.bat

echo Forwarding to scripts/directml/test_directml.bat...
cd scripts\directml
call test_directml.bat %*
cd ..\.. 