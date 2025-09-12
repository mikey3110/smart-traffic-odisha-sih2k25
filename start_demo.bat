@echo off
echo Starting Smart Traffic Management Demo...

REM Kill any existing Python processes
taskkill /f /im python.exe 2>nul

REM Start the server
cd "C:\Users\dasar\OneDrive\Desktop\Smart India Hackathon\smart-traffic-odisha-sih2k25\src\backend"
python simple_main.py

pause
