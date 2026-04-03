@echo off
chcp 65001 >nul

:: Ensure we run from the project root directory
cd /d "%~dp0"

echo ========================================
echo   utai - Environment Setup
echo ========================================
echo.

:: Check py command
py -3.10 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10 not found via 'py -3.10'.
    echo         Please install Python 3.10 from:
    echo         https://www.python.org/downloads/release/python-31011/
    echo         Make sure to install the "py" launcher.
    pause
    exit /b 1
)

echo [OK] Python 3.10 found:
py -3.10 --version

:: Create venv
if not exist "venv" (
    echo.
    echo [1/3] Creating virtual environment...
    py -3.10 -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] venv created.
) else (
    echo [OK] venv already exists.
)

:: Activate venv
call venv\Scripts\activate.bat

:: Select GPU mode (allow override via UTAI_GPU_MODE env var)
if defined UTAI_GPU_MODE (
    set GPU_CHOICE=%UTAI_GPU_MODE%
    echo Using GPU mode from UTAI_GPU_MODE: %GPU_CHOICE%
) else (
    echo.
    echo ========================================
    echo   Select compute device:
    echo   1. NVIDIA GPU - CUDA 12.1 (RTX 30/40/50 series) [Recommended]
    echo   2. NVIDIA GPU - CUDA 12.4 (alternative)
    echo   3. CPU only (no GPU acceleration)
    echo ========================================
    set /p GPU_CHOICE="Enter choice (1/2/3) [default: 1]: "
    if "%GPU_CHOICE%"=="" set GPU_CHOICE=1
)

:: Install PyTorch first
echo.
echo [2/3] Installing PyTorch...
if "%GPU_CHOICE%"=="1" (
    echo Installing PyTorch with CUDA 12.1...
    pip install -r requirements-torch-cu121.txt
) else if "%GPU_CHOICE%"=="2" (
    echo Installing PyTorch with CUDA 12.4...
    pip install -r requirements-torch-cu124.txt
) else (
    echo Installing PyTorch CPU...
    pip install -r requirements-torch-cpu.txt
)

if errorlevel 1 (
    echo [ERROR] PyTorch installation failed.
    pause
    exit /b 1
)

:: Install other dependencies
echo.
echo [3/3] Installing application dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)

:: Verify ffmpeg
echo.
where ffmpeg >nul 2>&1
if errorlevel 1 (
    if exist "bin\ffmpeg.exe" (
        echo [OK] ffmpeg found in bin/
    ) else (
        echo [WARN] ffmpeg not found. Video features may not work.
        echo        Place ffmpeg.exe in the bin/ folder, or install it system-wide.
    )
) else (
    echo [OK] ffmpeg found on PATH.
)

echo.
echo ========================================
echo   Setup complete!
echo   Run 'run.bat' to start the application.
echo ========================================
pause
