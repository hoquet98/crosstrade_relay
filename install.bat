@echo off
setlocal enabledelayedexpansion

:: ---- Configuration ----
set "REPO_URL=https://github.com/hoquet98/crosstrade_relay/archive/refs/heads/main.zip"
set "INSTALL_DIR=C:\TradeRelay"
set "PYTHON_URL=https://www.python.org/ftp/python/3.13.2/python-3.13.2-amd64.exe"
set "SERVICE_NAME=TradeRelay"
set "SERVICE_PORT=8080"
set "LOG_FILE=%INSTALL_DIR%\install.log"

:: Create install dir early so we can log
mkdir "%INSTALL_DIR%" 2>nul

echo ============================================>> "%LOG_FILE%"
echo   Trade Relay Install - %date% %time%>> "%LOG_FILE%"
echo ============================================>> "%LOG_FILE%"

echo ============================================
echo   Trade Relay - One-Click Installer
echo   Run as Administrator
echo ============================================
echo.

:: ---- Check for admin privileges ----
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] This script must be run as Administrator.
    echo Right-click install.bat and select "Run as administrator".
    pause
    exit /b 1
)
call :log "[OK] Running as Administrator"
echo.

:: ---- Download and extract project files ----
if exist "%INSTALL_DIR%\relay.py" goto :skip_download

call :log "Downloading Trade Relay from GitHub..."
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%REPO_URL%' -OutFile '%INSTALL_DIR%\repo.zip' -UseBasicParsing"
if not exist "%INSTALL_DIR%\repo.zip" goto :download_failed

call :log "Extracting files..."
powershell -Command "Expand-Archive -Path '%INSTALL_DIR%\repo.zip' -DestinationPath '%INSTALL_DIR%\temp' -Force"
powershell -Command "Get-ChildItem '%INSTALL_DIR%\temp\crosstrade_relay-main\*' | Move-Item -Destination '%INSTALL_DIR%\' -Force"
rmdir /s /q "%INSTALL_DIR%\temp" 2>nul
del "%INSTALL_DIR%\repo.zip" 2>nul

if not exist "%INSTALL_DIR%\relay.py" goto :extract_failed
call :log "[OK] Project files downloaded to %INSTALL_DIR%"
goto :done_download

:download_failed
call :log "[ERROR] Failed to download project files from GitHub."
pause
exit /b 1

:extract_failed
call :log "[ERROR] Failed to extract project files."
pause
exit /b 1

:skip_download
call :log "[OK] Project files already present at %INSTALL_DIR%"

:done_download
echo.

:: ---- Check for Python, install if missing ----
:: Use "python -c" to verify it's real Python, not the Windows Store stub
python -c "import sys; sys.exit(0)" >nul 2>&1
if !errorlevel! equ 0 goto :skip_python

call :log "Python not found - downloading installer..."
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%INSTALL_DIR%\python-installer.exe' -UseBasicParsing"
if not exist "%INSTALL_DIR%\python-installer.exe" goto :python_download_failed

call :log "Installing Python 3.13 (this may take a few minutes)..."
start /wait "" "%INSTALL_DIR%\python-installer.exe" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1
del "%INSTALL_DIR%\python-installer.exe" 2>nul

:: Refresh PATH so python is available in this session
for /f "tokens=2*" %%A in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "PATH=%%B;%PATH%"
for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "PATH=%%B;%PATH%"

python --version >nul 2>&1
if !errorlevel! neq 0 goto :python_path_failed

call :log "[OK] Python installed"
goto :skip_python

:python_download_failed
call :log "[ERROR] Failed to download Python installer."
call :log "Download manually from https://www.python.org/downloads/"
pause
exit /b 1

:python_path_failed
call :log "[ERROR] Python installed but not found in PATH. Please restart this script."
pause
exit /b 1

:skip_python
for /f "tokens=*" %%V in ('python --version 2^>^&1') do call :log "[OK] %%V"
echo.

:: ---- Create virtual environment ----
if exist "%INSTALL_DIR%\venv\Scripts\python.exe" goto :skip_venv

call :log "Creating virtual environment..."
python -m venv "%INSTALL_DIR%\venv"
if not exist "%INSTALL_DIR%\venv\Scripts\python.exe" goto :venv_failed
call :log "[OK] Virtual environment created"
goto :done_venv

:venv_failed
call :log "[ERROR] Failed to create virtual environment."
pause
exit /b 1

:skip_venv
call :log "[OK] Virtual environment already exists"

:done_venv
echo.

:: ---- Install dependencies into venv ----
call :log "Installing dependencies..."
"%INSTALL_DIR%\venv\Scripts\pip.exe" install -r "%INSTALL_DIR%\requirements.txt"
if !errorlevel! neq 0 goto :deps_failed
call :log "[OK] Dependencies installed"
goto :done_deps

:deps_failed
call :log "[ERROR] Failed to install dependencies."
pause
exit /b 1

:done_deps
echo.

:: ---- Initialize database ----
call :log "Initializing database..."
"%INSTALL_DIR%\venv\Scripts\python.exe" -c "import sys; sys.path.insert(0, r'%INSTALL_DIR%'); import database; database.init_db(); print('[OK] Database initialized')"
call :log "[OK] Database initialized"
echo.

:: ---- Download NSSM if not present ----
set "NSSM_DIR=%INSTALL_DIR%\nssm"
set "NSSM_EXE=%NSSM_DIR%\nssm.exe"

if exist "%NSSM_EXE%" goto :skip_nssm

call :log "Downloading NSSM (Windows Service Manager)..."
mkdir "%NSSM_DIR%" 2>nul

powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://nssm.cc/release/nssm-2.24.zip' -OutFile '%NSSM_DIR%\nssm.zip' -UseBasicParsing"
if not exist "%NSSM_DIR%\nssm.zip" goto :nssm_download_failed

call :log "Extracting NSSM..."
powershell -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; $zip = [System.IO.Compression.ZipFile]::OpenRead('%NSSM_DIR%\nssm.zip'); $entry = $zip.Entries | Where-Object { $_.FullName -like '*/win64/nssm.exe' }; [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, '%NSSM_EXE%', $true); $zip.Dispose()"
del "%NSSM_DIR%\nssm.zip" 2>nul
if not exist "%NSSM_EXE%" goto :nssm_extract_failed

call :log "[OK] NSSM downloaded"
goto :done_nssm

:nssm_download_failed
call :log "[ERROR] Failed to download NSSM."
call :log "You can manually download from https://nssm.cc/download"
pause
exit /b 1

:nssm_extract_failed
call :log "[ERROR] Failed to extract NSSM."
pause
exit /b 1

:skip_nssm
call :log "[OK] NSSM already present"

:done_nssm
echo.

:: ---- Stop existing service if running ----
"%NSSM_EXE%" status %SERVICE_NAME% >nul 2>&1
if !errorlevel! neq 0 goto :skip_stop_service

call :log "Stopping existing %SERVICE_NAME% service..."
"%NSSM_EXE%" stop %SERVICE_NAME% >nul 2>&1
timeout /t 3 /nobreak >nul
call :log "Removing existing service..."
"%NSSM_EXE%" remove %SERVICE_NAME% confirm >nul 2>&1
timeout /t 2 /nobreak >nul
call :log "[OK] Old service removed"

:skip_stop_service
echo.

:: ---- Install Windows service via NSSM ----
call :log "Installing %SERVICE_NAME% as a Windows service..."

"%NSSM_EXE%" install %SERVICE_NAME% "%INSTALL_DIR%\venv\Scripts\python.exe"
"%NSSM_EXE%" set %SERVICE_NAME% AppParameters "relay.py"
"%NSSM_EXE%" set %SERVICE_NAME% AppDirectory "%INSTALL_DIR%"
"%NSSM_EXE%" set %SERVICE_NAME% Description "Trade Relay - TradingView to CrossTrade webhook relay"
"%NSSM_EXE%" set %SERVICE_NAME% Start SERVICE_AUTO_START
"%NSSM_EXE%" set %SERVICE_NAME% AppExit Default Restart
"%NSSM_EXE%" set %SERVICE_NAME% AppRestartDelay 5000

:: ---- Configure service logging ----
set "LOG_DIR=%INSTALL_DIR%\logs"
mkdir "%LOG_DIR%" 2>nul
"%NSSM_EXE%" set %SERVICE_NAME% AppStdout "%LOG_DIR%\service_stdout.log"
"%NSSM_EXE%" set %SERVICE_NAME% AppStderr "%LOG_DIR%\service_stderr.log"
"%NSSM_EXE%" set %SERVICE_NAME% AppStdoutCreationDisposition 4
"%NSSM_EXE%" set %SERVICE_NAME% AppStderrCreationDisposition 4

call :log "[OK] Service installed (auto-start on boot, auto-restart on crash)"
echo.

:: ---- Open firewall port ----
call :log "Configuring firewall..."
netsh advfirewall firewall delete rule name="%SERVICE_NAME%" >nul 2>&1
netsh advfirewall firewall add rule name="%SERVICE_NAME%" dir=in action=allow protocol=tcp localport=%SERVICE_PORT% >nul 2>&1
if !errorlevel! equ 0 (
    call :log "[OK] Firewall rule added (port %SERVICE_PORT% inbound)"
) else (
    call :log "[WARN] Could not add firewall rule - you may need to open port %SERVICE_PORT% manually"
)
echo.

:: ---- Start the service ----
call :log "Starting %SERVICE_NAME% service..."
"%NSSM_EXE%" start %SERVICE_NAME%
timeout /t 3 /nobreak >nul

"%NSSM_EXE%" status %SERVICE_NAME% | findstr /i "running" >nul 2>&1
if !errorlevel! equ 0 (
    call :log "[OK] %SERVICE_NAME% service is running"
) else (
    call :log "[WARN] Service may not have started. Check logs at: %LOG_DIR%\"
)
echo.

:: ---- Final instructions ----
echo ============================================
echo   Installation complete!
echo.
echo   Installed to: %INSTALL_DIR%
echo   Service:      %SERVICE_NAME% (auto-starts on boot)
echo   Port:         %SERVICE_PORT%
echo   Logs:         %LOG_DIR%\
echo   Install log:  %LOG_FILE%
echo.
echo   Next step - add your user config:
echo     cd /d %INSTALL_DIR%
echo     venv\Scripts\python.exe manage.py add-user
echo.
echo   Useful commands:
echo     %NSSM_EXE% status %SERVICE_NAME%
echo     %NSSM_EXE% stop %SERVICE_NAME%
echo     %NSSM_EXE% start %SERVICE_NAME%
echo     %NSSM_EXE% restart %SERVICE_NAME%
echo ============================================
call :log "Installation complete"
pause
exit /b 0

:: ---- Logging helper ----
:log
echo %~1
echo [%date% %time%] %~1 >> "%LOG_FILE%"
goto :eof
