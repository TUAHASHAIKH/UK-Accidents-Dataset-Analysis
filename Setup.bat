@echo off
title Setup UK Road Accidents Dashboard
echo ========================================
echo   UK Road Accidents Dashboard Setup
echo ========================================
echo.
echo This script will install all required packages.
echo Please wait...
echo.

cd /d "%~dp0"

echo Installing Python packages from requirements.txt...
echo.

pip install -r requirements.txt

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo You can now run the dashboard by double-clicking "Run_Dashboard.bat"
echo.

pause
