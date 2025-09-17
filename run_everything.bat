@echo off
echo ========================================
echo Smart Traffic Management System
echo Complete Startup Script
echo ========================================
echo.

echo [1/6] Starting PostgreSQL Database...
cd src\backend
start "PostgreSQL" cmd /k "python -c \"import subprocess; subprocess.run(['python', 'main.py'])\""
timeout /t 5 /nobreak > nul

echo [2/6] Starting ML API...
cd ..\ml_engine
start "ML API" cmd /k "python api\ml_api.py"
timeout /t 5 /nobreak > nul

echo [3/6] Starting Computer Vision Service...
cd ..\computer_vision
start "CV Service" cmd /k "python demo_integration.py"
timeout /t 5 /nobreak > nul

echo [4/6] Starting Frontend...
cd ..\frontend\smart-traffic-ui
start "Frontend" cmd /k "npm run dev"
timeout /t 5 /nobreak > nul

echo [5/6] Starting SUMO Simulation...
cd ..\..\..\sumo
start "SUMO Normal Traffic" cmd /k "sumo-gui -c configs\normal_traffic.sumocfg"
timeout /t 3 /nobreak > nul

start "SUMO Rush Hour" cmd /k "sumo-gui -c configs\rush_hour.sumocfg"
timeout /t 3 /nobreak > nul

start "SUMO Emergency Vehicle" cmd /k "sumo-gui -c configs\emergency_vehicle.sumocfg"
timeout /t 3 /nobreak > nul

echo [6/6] Opening Web Dashboard...
cd ..
start "Dashboard" simulation_dashboard.html

echo.
echo ========================================
echo All services started successfully!
echo ========================================
echo.
echo Services running:
echo - Backend API: http://localhost:8000
echo - ML API: http://localhost:8001
echo - CV Service: http://localhost:5001
echo - Frontend: http://localhost:3000
echo - SUMO GUI: 3 simulation windows opened
echo - Dashboard: simulation_dashboard.html opened
echo.
echo Press any key to stop all services...
pause > nul

echo Stopping all services...
taskkill /f /im python.exe 2>nul
taskkill /f /im node.exe 2>nul
taskkill /f /im sumo-gui.exe 2>nul
echo All services stopped.
