@echo off
set "ROOT=%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\run-app.ps1"
if errorlevel 1 pause
