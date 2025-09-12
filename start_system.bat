@echo off
echo Starting Smart Traffic Management System...

echo.
echo Starting Backend API...
start "Backend API" cmd /k "python src/backend/main.py"

echo.
echo Waiting for backend to start...
timeout /t 3 /nobreak > nul

echo.
echo Starting Frontend...
start "Frontend" cmd /k "cd src/frontend/smart-traffic-ui && npm run dev"

echo.
echo System is starting up...
echo.
echo Backend API: http://localhost:8000
echo Frontend: http://localhost:5173
echo API Health Check: http://localhost:8000/health
echo.
echo Press any key to exit...
pause > nul
