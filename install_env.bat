@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ============================================================
echo   Fire Alarm System - Environment Setup (run once)
echo ============================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo.
    echo   Install Python 3.10 - 3.12 first:
    echo     https://www.python.org/downloads/
    echo   IMPORTANT: tick "Add python.exe to PATH" during setup.
    echo.
    pause
    exit /b 1
)
echo [1/4] Python found:
python --version
echo.

echo [2/4] Creating virtual environment .venv ...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create .venv
        pause
        exit /b 1
    )
) else (
    echo       .venv already exists, skip.
)
echo.

echo [3/4] Installing dependencies ^(Tsinghua mirror^) ...
echo       This may take 5 - 15 minutes, please wait.
".venv\Scripts\python.exe" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
".venv\Scripts\python.exe" -m pip install -r gui\requirements-gui.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check your network and retry.
    pause
    exit /b 1
)
echo.

echo [4/4] Verifying ...
".venv\Scripts\python.exe" -c "import torch, ultralytics, streamlit, cv2; print('torch      ', torch.__version__); print('ultralytics', ultralytics.__version__); print('streamlit  ', streamlit.__version__); print('opencv     ', cv2.__version__)"
echo.
echo ============================================================
echo   Done. Now double-click the start script to launch the demo.
echo ============================================================
pause
