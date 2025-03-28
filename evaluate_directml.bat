@echo off

REM evaluate_directml.bat - Root convenience script
REM Forwards calls to scripts/directml/evaluate_directml.bat

echo Forwarding to scripts/directml/evaluate_directml.bat...
cd scripts\directml
call evaluate_directml.bat %*
cd ..\.. 