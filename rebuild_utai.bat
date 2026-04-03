@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo === Rebuilding Utai ===

echo [1/3] Building frontend...
cd app\frontend
call npm run build
if errorlevel 1 (
    echo [ERROR] Frontend build failed
    pause
    exit /b 1
)

echo [2/3] Copying frontend to backend/static...
rd /s /q "..\backend\static" 2>nul
mkdir "..\backend\static" 2>nul
xcopy /e /q /y "dist\*" "..\backend\static\"
cd ..\..

echo [3/3] Syncing to Utai package...
:: Copy backend code (not the large junctioned dirs)
xcopy /e /q /y "app\backend\*" "Utai\resources\app\backend\" /exclude:rebuild_exclude.txt 2>nul
copy /y "app\__init__.py" "Utai\resources\app\" >nul
:: Copy electron files
copy /y "electron\main.js" "Utai\resources\app\" >nul
copy /y "electron\preload.js" "Utai\resources\app\" >nul
copy /y "electron\loading.html" "Utai\resources\app\" >nul

echo === Build complete! Launch Utai\Utai.exe ===
pause
