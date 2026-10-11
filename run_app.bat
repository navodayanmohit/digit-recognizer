
@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "draw_digit.py"
if errorlevel 1 pause