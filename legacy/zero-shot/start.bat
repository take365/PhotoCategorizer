@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Create Windows virtual environment if needed
if not exist "venv_win" (
    echo [INFO] Creating virtual environment at venv_win ...
    py -3 -m venv venv_win
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
)

REM Activate virtual environment
call "venv_win\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    exit /b 1
)

REM Install dependencies (upgrade pip first for reliability)
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install required packages.
    exit /b 1
)

REM Run evaluation (parameters are read from .env if not supplied)
python -m photocat.cli eval input

if errorlevel 1 (
    echo [ERROR] Evaluation command failed.
    exit /b 1
)

echo.
echo [INFO] Completed. Check outputs directory for results.
echo.
echo All done! You can close this window.
pause
exit /b 0
