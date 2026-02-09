@echo off
echo Starting Trade Relay on port 8080 (manual mode)...
echo The service is the recommended way to run — this is for debugging.
echo Press Ctrl+C to stop.
echo.
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

if exist "%PROJECT_DIR%\venv\Scripts\python.exe" (
    "%PROJECT_DIR%\venv\Scripts\python.exe" relay.py
) else (
    python relay.py
)
pause
