@echo off
chcp 65001 >nul

:: Ensure we run from the project root directory
cd /d "%~dp0"

:: Activate venv
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] venv not found. Run setup_env.bat first.
    pause
    exit /b 1
)
call venv\Scripts\activate.bat

:: Add local bin to PATH (for bundled ffmpeg etc.)
if exist "bin" set PATH=%~dp0bin;%PATH%

:: Start the backend server
echo Starting utai backend server...
echo Frontend: http://localhost:8765
echo Press Ctrl+C to stop
echo.
python -m uvicorn app.backend.main:app --host 127.0.0.1 --port 8765 --reload
