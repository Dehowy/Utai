@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo === Building Utai Installer ===
echo.

:: ── 1. Build frontend ──
echo [1/6] Building frontend...
cd app\frontend
call npm run build
if errorlevel 1 (
    echo [ERROR] Frontend build failed
    pause
    exit /b 1
)
cd ..\..

:: ── 2. Copy built frontend to backend/static ──
echo [2/6] Copying frontend to backend/static...
rd /s /q "app\backend\static" 2>nul
mkdir "app\backend\static" 2>nul
xcopy /e /q /y "app\frontend\dist\*" "app\backend\static\"
if errorlevel 1 (
    echo [ERROR] Failed to copy frontend to backend/static
    pause
    exit /b 1
)

:: ── 3. Sync backend + requirements to installer bundle ──
echo [3/6] Syncing backend to installer bundle...
rd /s /q "installer\bundle\app\backend" 2>nul
mkdir "installer\bundle\app\backend" 2>nul
xcopy /e /q /y "app\backend\*" "installer\bundle\app\backend\" /exclude:installer_exclude.txt
copy /y "app\__init__.py" "installer\bundle\app\" >nul 2>nul
:: Sync requirements files (they may have changed since last build)
for %%f in (requirements.txt requirements-torch-cu121.txt requirements-torch-cu128.txt requirements-torch-cpu.txt) do (
    if exist "%%f" copy /y "%%f" "installer\bundle\" >nul
)

:: Sync MSST data files (models_info.json — not in upstream repo)
if exist "MSST\data\models_info.json" (
    mkdir "installer\bundle\msst-data" 2>nul
    copy /y "MSST\data\models_info.json" "installer\bundle\msst-data\" >nul
    if exist "MSST\data\webui_config.json" copy /y "MSST\data\webui_config.json" "installer\bundle\msst-data\" >nul
    echo   msst-data synced
)
:: Sync yt-dlp source (has local patches for YouTube 412 fix)
if exist "yt-dlp\yt_dlp\__init__.py" (
    rd /s /q "installer\bundle\yt-dlp" 2>nul
    mkdir "installer\bundle\yt-dlp" 2>nul
    xcopy /e /q /y "yt-dlp\yt_dlp\*" "installer\bundle\yt-dlp\yt_dlp\" >nul
    echo   yt-dlp source synced
)
:: Sync MSST patches — upstream repo has different structure from what our wrappers expect
:: See README.md "Packaging Pitfalls" section for full explanation
if exist "MSST\inference\__init__.py" (
    rd /s /q "installer\bundle\msst-patches" 2>nul
    mkdir "installer\bundle\msst-patches" 2>nul
    :: inference/ package (upstream has inference.py single file)
    xcopy /e /q /y "MSST\inference\*" "installer\bundle\msst-patches\inference\" >nul
    :: modules/ (model architectures — upstream has models/ with different layout)
    xcopy /e /q /y "MSST\modules\*" "installer\bundle\msst-patches\modules\" >nul
    :: utils/ (upstream has completely different files)
    mkdir "installer\bundle\msst-patches\utils" 2>nul
    for %%f in (__init__.py constant.py logger.py utils.py) do (
        if exist "MSST\utils\%%f" copy /y "MSST\utils\%%f" "installer\bundle\msst-patches\utils\%%f" >nul
    )
    :: Per-model configs (upstream doesn't have these)
    xcopy /e /q /y "MSST\configs\vocal_models\*" "installer\bundle\msst-patches\configs\vocal_models\" >nul
    xcopy /e /q /y "MSST\configs\single_stem_models\*" "installer\bundle\msst-patches\configs\single_stem_models\" >nul
    xcopy /e /q /y "MSST\configs\multi_stem_models\*" "installer\bundle\msst-patches\configs\multi_stem_models\" >nul
    :: VR model params (upstream doesn't have these — required for VR model inference)
    xcopy /e /q /y "MSST\configs\vr_modelparams\*" "installer\bundle\msst-patches\configs\vr_modelparams\" >nul
    echo   msst-patches synced (inference, modules, utils, configs, vr_modelparams)
)

:: Sync RVC patches — upstream needs fixes for portable env:
::   audio.py: ffmpeg subprocess → PyAV (no ffmpeg binary needed)
::   process_ckpt.py: remove i18n import (i18n dir deleted in cleanup)
::   train.py: argparse guard for mp.Process, USE_LIBUV, logger rebuild
::   utils.py: matplotlib tostring_rgb → buffer_rgba
if exist "RVC\RVC20240604Nvidia\infer\lib\audio.py" (
    rd /s /q "installer\bundle\rvc-patches" 2>nul
    mkdir "installer\bundle\rvc-patches\infer\lib\train" 2>nul
    mkdir "installer\bundle\rvc-patches\infer\modules\train" 2>nul
    mkdir "installer\bundle\rvc-patches\configs\v1" 2>nul
    mkdir "installer\bundle\rvc-patches\configs\v2" 2>nul
    copy /y "RVC\RVC20240604Nvidia\infer\lib\audio.py" "installer\bundle\rvc-patches\infer\lib\audio.py" >nul
    copy /y "RVC\RVC20240604Nvidia\infer\lib\train\process_ckpt.py" "installer\bundle\rvc-patches\infer\lib\train\process_ckpt.py" >nul
    copy /y "RVC\RVC20240604Nvidia\infer\lib\train\utils.py" "installer\bundle\rvc-patches\infer\lib\train\utils.py" >nul
    copy /y "RVC\RVC20240604Nvidia\infer\modules\train\train.py" "installer\bundle\rvc-patches\infer\modules\train\train.py" >nul
    xcopy /e /q /y "RVC\RVC20240604Nvidia\configs\v1\*" "installer\bundle\rvc-patches\configs\v1\" >nul
    xcopy /e /q /y "RVC\RVC20240604Nvidia\configs\v2\*" "installer\bundle\rvc-patches\configs\v2\" >nul
    echo   rvc-patches synced (audio, process_ckpt, train, utils, configs)
)

:: Sync VST host (needed for VST3 plugin editor — e.g. Dexed)
if exist "VST\host\UtaiVSTHost.exe" (
    mkdir "installer\bundle\VST\host" 2>nul
    copy /y "VST\host\UtaiVSTHost.exe" "installer\bundle\VST\host\" >nul
    echo   VST host synced
)

:: ── 4. Rebuild Electron app (embeds custom icon into .exe) ──
echo [4/6] Rebuilding Electron app...
:: Copy frontend build to electron/dist/ (electron-packager --overwrite deletes it)
mkdir "electron\dist" 2>nul
xcopy /e /q /y "app\frontend\dist\*" "electron\dist\"
cd electron
call npx electron-packager . Utai --platform=win32 --arch=x64 --overwrite --icon=build/icon.ico
if errorlevel 1 (
    echo [ERROR] Electron packaging failed
    cd ..
    pause
    exit /b 1
)
cd ..

:: ── 5. Sync Electron app to installer bundle ──
echo [5/6] Syncing Electron to installer bundle...
rd /s /q "installer\bundle\utai-app\Utai-win32-x64" 2>nul
xcopy /e /q /y "electron\Utai-win32-x64\*" "installer\bundle\utai-app\Utai-win32-x64\"

:: ── 6. Apply NSIS patches + Build installer executable ──
echo [6/6] Building installer .exe...
cd installer
:: Apply portable.nsi splash fix (detach from Explorer, topmost)
if exist "patches\portable.nsi" (
    copy /y "patches\portable.nsi" "node_modules\app-builder-lib\templates\nsis\portable.nsi" >nul
    echo   portable.nsi patched
)
call npm run build
if errorlevel 1 (
    echo [ERROR] Installer build failed
    cd ..
    pause
    exit /b 1
)
cd ..

echo.
echo === Installer build complete! ===
echo Output: installer\dist\Utai_Installer.exe
pause
