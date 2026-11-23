@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
title Fire Alarm Server - close this window to stop
set "PYEXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYEXE%" set "PYEXE=python"
"%PYEXE%" -m streamlit run "gui\app.py" --server.headless true --server.port 8501
echo.
echo [INFO] Server stopped. If it failed to start, run install_env.bat first.
pause