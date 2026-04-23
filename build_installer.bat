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
