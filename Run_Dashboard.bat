@echo off
title UK Road Accidents Dashboard
echo ========================================
echo Starting UK Road Accidents Dashboard...
echo ========================================
echo.

cd /d "%~dp0"

streamlit run streamlit_dashboard.py

pause
