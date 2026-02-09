@echo off
echo ============================================
echo   Trade Relay - Uninstall Service
echo   Run as Administrator!
echo ============================================
echo.

:: ---- Check for admin privileges ----
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] This script must be run as Administrator.
    pause
    exit /b 1
)

set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
set "NSSM_EXE=%PROJECT_DIR%\nssm\nssm.exe"

:: ---- Stop and remove service ----
if exist "%NSSM_EXE%" (
    echo Stopping TradeRelay service...
    "%NSSM_EXE%" stop TradeRelay >nul 2>&1
    timeout /t 3 /nobreak >nul
    echo Removing TradeRelay service...
    "%NSSM_EXE%" remove TradeRelay confirm >nul 2>&1
    echo [OK] Service removed
) else (
    echo [WARN] NSSM not found — attempting removal via sc...
    sc stop TradeRelay >nul 2>&1
    timeout /t 3 /nobreak >nul
    sc delete TradeRelay >nul 2>&1
    echo [OK] Service removed
)
echo.

:: ---- Remove firewall rule ----
echo Removing firewall rule...
netsh advfirewall firewall delete rule name="TradeRelay" >nul 2>&1
echo [OK] Firewall rule removed
echo.

echo ============================================
echo   TradeRelay service has been removed.
echo   Your data (database, logs) is still intact.
echo   To reinstall, run install.bat
echo ============================================
pause
