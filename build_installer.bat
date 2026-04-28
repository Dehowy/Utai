@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo === Building Utai Installer ===
echo.

:: ── 1. Build frontend ──
echo [1/5] Building frontend...
cd app\frontend
call npm run build
if errorlevel 1 (
    echo [ERROR] Frontend build failed
    pause
    exit /b 1
)
cd ..\..

:: ── 2. Copy built frontend to backend/static ──
echo [2/5] Copying frontend to backend/static...
rd /s /q "app\backend\static" 2>nul
mkdir "app\backend\static" 2>nul
xcopy /e /q /y "app\frontend\dist\*" "app\backend\static\"
if errorlevel 1 (
    echo [ERROR] Failed to copy frontend to backend/static
    pause
    exit /b 1
)

:: ── 3. Sync backend + requirements to installer bundle ──
echo [3/5] Syncing backend to installer bundle...
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

:: Sync RVC patches — upstream audio.py needs subprocess fix for hidden terminal
if exist "RVC\RVC20240604Nvidia\infer\lib\audio.py" (
    rd /s /q "installer\bundle\rvc-patches" 2>nul
    mkdir "installer\bundle\rvc-patches\infer\lib" 2>nul
    copy /y "RVC\RVC20240604Nvidia\infer\lib\audio.py" "installer\bundle\rvc-patches\infer\lib\audio.py" >nul
    echo   rvc-patches synced (audio.py terminal fix)
)

:: Sync VST host (needed for VST3 plugin editor — e.g. Dexed)
if exist "VST\host\UtaiVSTHost.exe" (
    mkdir "installer\bundle\VST\host" 2>nul
    copy /y "VST\host\UtaiVSTHost.exe" "installer\bundle\VST\host\" >nul
    echo   VST host synced
)

:: ── 4. Sync Electron app files to bundle ──
echo [4/5] Syncing Electron files to installer bundle...
set ELECTRON_DEST=installer\bundle\utai-app\Utai-win32-x64\resources\app
copy /y "electron\main.js"      "%ELECTRON_DEST%\" >nul
copy /y "electron\preload.js"   "%ELECTRON_DEST%\" >nul
copy /y "electron\loading.html" "%ELECTRON_DEST%\" >nul

:: ── 5. Apply NSIS patches + Build installer executable ──
echo [5/5] Building installer .exe...
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
