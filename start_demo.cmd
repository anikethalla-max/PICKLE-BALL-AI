@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment missing. See DASHBOARD_GUIDE.md for setup.
  pause
  exit /b 1
)
echo Smart Doubles AI - local demonstration
echo Your browser will open at http://127.0.0.1:8501 when the app is ready.
echo Keep this window open. Press Ctrl+C here to stop the application.
".venv\Scripts\python.exe" launch_demo.py
if errorlevel 1 pause
