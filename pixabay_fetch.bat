@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Ensure UTF-8 code page for consistent Unicode handling
chcp 65001 >nul

REM Prepare Windows virtual environment if not present
if not exist "venv_win" (
    echo [INFO] Creating virtual environment at venv_win ...
    py -3 -m venv venv_win
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call "venv_win\Scripts\activate.bat"
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

REM Install dependencies
python -m pip install --upgrade pip >nul
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install required packages.
    pause
    exit /b 1
)

echo.
echo === Pixabay Downloader ===

set "PIXABAY_QUERY="
set /p PIXABAY_QUERY=Enter keyword (Japanese allowed): 
if "%PIXABAY_QUERY%"=="" (
    echo [ERROR] Keyword is required.
    goto :cleanup
)

set "PIXABAY_LIMIT="
set /p PIXABAY_LIMIT=How many images should be downloaded?: 
if "%PIXABAY_LIMIT%"=="" set "PIXABAY_LIMIT=10"

set "__INVALID_LIMIT=0"
for /f "delims=0123456789" %%A in ("%PIXABAY_LIMIT%") do set "__INVALID_LIMIT=1"
if "%PIXABAY_LIMIT%"=="0" set "__INVALID_LIMIT=1"
if "!__INVALID_LIMIT!"=="1" (
    echo [WARN] Invalid number entered. Using default value 10.
    set "PIXABAY_LIMIT=10"
)

set "SANITIZED_QUERY=%PIXABAY_QUERY%"
for /f "tokens=* delims=" %%I in ("%SANITIZED_QUERY%") do set "SANITIZED_QUERY=%%I"
call set "SANITIZED_QUERY=%%SANITIZED_QUERY:\=_%%"
call set "SANITIZED_QUERY=%%SANITIZED_QUERY:/=_%%"
call set "SANITIZED_QUERY=%%SANITIZED_QUERY::=_%%"
call set "SANITIZED_QUERY=%%SANITIZED_QUERY:*=_%%"
call set "SANITIZED_QUERY=%%SANITIZED_QUERY:?=_%%"
call set "SANITIZED_QUERY=%%SANITIZED_QUERY:"=_%%"
call set "SANITIZED_QUERY=%%SANITIZED_QUERY:^<=_%%"
call set "SANITIZED_QUERY=%%SANITIZED_QUERY:^>=_%%"
call set "SANITIZED_QUERY=%%SANITIZED_QUERY:^|=_%%"
for /f "tokens=* delims=" %%I in ("%SANITIZED_QUERY%") do set "SANITIZED_QUERY=%%I"
if not defined SANITIZED_QUERY set "SANITIZED_QUERY=keyword"

set "OUTPUT_BASE=%~dp0pixabay"
set "OUTPUT_DIR=%OUTPUT_BASE%\%SANITIZED_QUERY%"

if not exist "%OUTPUT_BASE%" (
    mkdir "%OUTPUT_BASE%"
    if errorlevel 1 (
        echo [ERROR] Failed to create output directory.
        goto :cleanup
    )
)

echo.
echo [INFO] Query: %PIXABAY_QUERY%
echo [INFO] Limit: %PIXABAY_LIMIT%
echo [INFO] Saving into: %OUTPUT_DIR%
echo.

python -m photocat.cli pixabay-download --query "%PIXABAY_QUERY%" --limit %PIXABAY_LIMIT% --out-dir "%OUTPUT_DIR%"
if errorlevel 1 (
    echo [ERROR] Pixabay download command failed.
    goto :cleanup
)

echo.
echo [INFO] Download completed.

:cleanup
echo.
echo Done. Press any key to close.
pause >nul
exit /b 0
