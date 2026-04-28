@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set "INSTALL_PATH=%~1"
set "GPU=%~2"
set "PYPI_MIRROR=%~3"
set "PYTORCH_MIRROR=%~4"
set "BD=%~5"
if "%BD%"=="" set "BD=%BUNDLE_PATH%"

if "%INSTALL_PATH%"=="" (
    echo [!] No install path specified
    exit /b 1
)

echo [STAGE] Creating directories...
echo [PROGRESS] 5
if not exist "%INSTALL_PATH%" mkdir "%INSTALL_PATH%"

:: Copy application files (installer bundles these next to setup.bat)
echo [STAGE] Copying application files...
echo [PROGRESS] 10
if "%BD%"=="" set "BD=%~dp0bundle"
echo INSTALL_PATH=%INSTALL_PATH%
echo BUNDLE_PATH=%BD%
echo BUNDLE_EXISTS=
if exist "%BD%\app" (echo YES) else (echo NO)
xcopy /E /I /Y /Q "%BD%\app" "%INSTALL_PATH%\app" >nul 2>&1
xcopy /E /I /Y /Q "%BD%\electron" "%INSTALL_PATH%\electron" >nul 2>&1
xcopy /E /I /Y /Q "%BD%\synth" "%INSTALL_PATH%\synth" >nul 2>&1
xcopy /E /I /Y /Q "%BD%\score2hubert_v2" "%INSTALL_PATH%\score2hubert_v2" >nul 2>&1
copy /Y "%BD%\requirements*.txt" "%INSTALL_PATH%\" >nul 2>&1
copy /Y "%BD%\icon.png" "%INSTALL_PATH%\" >nul 2>&1

:: Check if MSST/RVC should be included
set "CONFIG_FILE=%INSTALL_PATH%\_install_config.json"
if exist "%CONFIG_FILE%" (
    findstr /c:"\"audio\": true" "%CONFIG_FILE%" >nul && (
        echo [STAGE] Copying audio processing tools...
        echo [PROGRESS] 15
        xcopy /E /I /Y /Q "%BD%\MSST" "%INSTALL_PATH%\MSST" >nul 2>&1
        xcopy /E /I /Y /Q "%BD%\RVC" "%INSTALL_PATH%\RVC" >nul 2>&1
    )
)

:: Setup Python venv
echo [STAGE] Setting up Python environment...
echo [PROGRESS] 20

set "PYTHON_DIR=%INSTALL_PATH%\python"
set "VENV_DIR=%INSTALL_PATH%\venv"

:: Copy embedded Python if not already present
if not exist "%PYTHON_DIR%\python.exe" (
    xcopy /E /I /Y /Q "%BD%\python" "%PYTHON_DIR%" >nul 2>&1
)

:: Create venv
echo [STAGE] Creating virtual environment...
echo [PROGRESS] 25
"%PYTHON_DIR%\python.exe" -m venv "%VENV_DIR%" 2>&1
if errorlevel 1 (
    echo [!] Failed to create venv
    echo [STAGE] Trying alternative: pip install directly...
    :: Embedded Python may not have venv — use pip directly into python dir
    set "PIP_CMD=%PYTHON_DIR%\python.exe -m pip"
    goto :pip_install
)

set "PIP_CMD=%VENV_DIR%\Scripts\python.exe -m pip"

:pip_install
:: Ensure pip is available
echo [STAGE] Ensuring pip is available...
echo [PROGRESS] 30
if exist "%BD%\get-pip.py" (
    "%PYTHON_DIR%\python.exe" "%BD%\get-pip.py" --no-warn-script-location 2>&1
)

:: Set pip mirror if configured
set "PIP_ARGS="
if not "%PYPI_MIRROR%"=="" (
    set "PIP_ARGS=--index-url %PYPI_MIRROR% --trusted-host %PYPI_MIRROR:~8,50%"
    echo Using PyPI mirror: %PYPI_MIRROR%
)

:: Install requirements
echo [STAGE] Installing Python dependencies (this may take a few minutes)...
echo [PROGRESS] 35
%PIP_CMD% install --no-warn-script-location %PIP_ARGS% -r "%INSTALL_PATH%\requirements.txt" 2>&1
if errorlevel 1 (
    echo [!] WARNING: Some dependencies may have failed to install
)

:: Install PyTorch
echo [STAGE] Installing PyTorch for %GPU%...
echo [PROGRESS] 55

set "TORCH_INDEX="
if "%GPU%"=="cu121" (
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
    if not "%PYTORCH_MIRROR%"=="" set "TORCH_INDEX=%PYTORCH_MIRROR%/cu121"
)
if "%GPU%"=="cu128" (
    set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
    if not "%PYTORCH_MIRROR%"=="" set "TORCH_INDEX=%PYTORCH_MIRROR%/cu128"
)
if "%GPU%"=="cpu" (
    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    if not "%PYTORCH_MIRROR%"=="" set "TORCH_INDEX=%PYTORCH_MIRROR%/cpu"
)

if "%GPU%"=="cu128" (
    %PIP_CMD% install --no-warn-script-location -r "%INSTALL_PATH%\requirements-torch-cu128.txt" --extra-index-url %TORCH_INDEX% 2>&1
) else if "%GPU%"=="cpu" (
    %PIP_CMD% install --no-warn-script-location -r "%INSTALL_PATH%\requirements-torch-cpu.txt" --extra-index-url %TORCH_INDEX% 2>&1
) else (
    %PIP_CMD% install --no-warn-script-location -r "%INSTALL_PATH%\requirements-torch-cu121.txt" --extra-index-url %TORCH_INDEX% 2>&1
)

echo [PROGRESS] 80

:: Save mirror config
echo [STAGE] Saving configuration...
echo [PROGRESS] 85
if not exist "%INSTALL_PATH%\config" mkdir "%INSTALL_PATH%\config"
echo {"pypi":"%PYPI_MIRROR%","github":"","huggingface":"","pytorch":"%PYTORCH_MIRROR%"} > "%INSTALL_PATH%\config\mirrors.json"

:: Verify installation
echo [STAGE] Verifying installation...
echo [PROGRESS] 90
if exist "%VENV_DIR%\Scripts\python.exe" (
    "%VENV_DIR%\Scripts\python.exe" -c "import torch; print('PyTorch', torch.__version__, '- CUDA:', torch.cuda.is_available())" 2>&1
) else (
    "%PYTHON_DIR%\python.exe" -c "import torch; print('PyTorch', torch.__version__, '- CUDA:', torch.cuda.is_available())" 2>&1
)
if errorlevel 1 (
    echo [!] PyTorch verification failed - installation may be incomplete
)

:: Clean up
echo [STAGE] Cleaning up...
echo [PROGRESS] 95
del /f /q "%INSTALL_PATH%\_install_config.json" 2>nul

echo [STAGE] Installation complete!
echo [PROGRESS] 100
exit /b 0
