@echo off
REM Open Mocap installer launcher (Windows). Double-click or run from Blender.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
echo.
echo [Open Mocap] Installer finished. You can close this window.
pause
