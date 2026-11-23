@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==========================================
echo   Starting Fire Alarm Demo System ...
echo ==========================================
echo.
start "FireAlarm" /min cmd /c ""%~dp0_server.bat""
echo Waiting for the server (usually 4-8 seconds) ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_wait_and_open.ps1"
echo.
echo A minimized "FireAlarm" window is running the server.
echo Close that window to stop the system.
echo.
timeout /t 3 >nul