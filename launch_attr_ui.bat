@echo off
setlocal ENABLEDELAYEDEXPANSION

REM ------------------------------------------------------------------
REM Launch the multi-attribute search UI (Windows helper).
REM Requirements:
REM   - .env contains LMSTUDIO_* connection settings
REM   - venv_win\Scripts\activate.bat exists and dependencies are installed
REM ------------------------------------------------------------------

if not exist "venv_win" (
    echo [ERROR] Missing venv_win. Run start.bat first.
    exit /b 1
)

call "venv_win\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

set "HOST=%~1"
set "PORT=%~2"
if "%HOST%"=="" set "HOST=0.0.0.0"
if "%PORT%"=="" set "PORT=8000"

echo [INFO] Launching attr-serve on %HOST%:%PORT%
python -m photocat.cli attr-serve --host %HOST% --port %PORT%
if errorlevel 1 (
    echo [ERROR] attr-serve exited with failure.
    exit /b 1
)

echo.
echo [INFO] Press Ctrl+C to stop the server.
endlocal
