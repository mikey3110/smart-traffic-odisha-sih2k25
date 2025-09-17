@echo off
REM Smart Traffic Management System - Windows Startup Script
REM This script starts all components of the system on Windows

setlocal enabledelayedexpansion

REM Configuration
set PROJECT_ROOT=%~dp0..
set LOG_FILE=project_startup_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log
set LOG_FILE=%LOG_FILE: =0%

REM Colors (Windows doesn't support colors in batch, but we'll use echo for clarity)
echo ================================================
echo Smart Traffic Management System - Windows
echo ================================================
echo.

REM Check prerequisites
echo Checking prerequisites...
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Node.js not found. Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)
echo ✓ Node.js found

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found. Please install Python from https://python.org/
    pause
    exit /b 1
)
echo ✓ Python found

where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: Docker not found. Some services may not work properly.
) else (
    echo ✓ Docker found
)

echo.
echo Starting services...
echo.

REM Start Backend Services
echo Starting Backend Services...
cd /d "%PROJECT_ROOT%\src\backend"

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating Python virtual environment...
    python -m venv venv
)

REM Activate virtual environment and install dependencies
call venv\Scripts\activate.bat
pip install -r requirements.txt

REM Start PostgreSQL and Redis with Docker Compose
echo Starting PostgreSQL and Redis...
docker-compose up -d postgres redis

REM Wait for services to be ready
echo Waiting for database services...
timeout /t 10 /nobreak >nul

REM Start Backend API
echo Starting Backend API...
start "Backend API" cmd /k "call venv\Scripts\activate.bat && python main.py"

REM Start ML API
echo Starting ML API...
cd /d "%PROJECT_ROOT%\src\ml_engine"
start "ML API" cmd /k "python api\ml_api.py"

REM Start Computer Vision Service
echo Starting Computer Vision Service...
cd /d "%PROJECT_ROOT%\src\computer_vision"
pip install -r requirements.txt

echo Starting HLS Streaming Service...
start "HLS Streaming" cmd /k "python hls_streaming.py"

echo Starting CV Demo Integration...
start "CV Demo" cmd /k "python demo_integration.py"

REM Start Frontend
echo Starting Frontend...
cd /d "%PROJECT_ROOT%\src\frontend\smart-traffic-ui"

REM Install Node.js dependencies if needed
if not exist "node_modules" (
    echo Installing Node.js dependencies...
    npm install
)

echo Starting React development server...
start "Frontend" cmd /k "npm run dev"

REM Start SUMO Simulation (if available)
echo Starting SUMO Simulation...
cd /d "%PROJECT_ROOT%\sumo"
where sumo >nul 2>&1
if %errorlevel% equ 0 (
    echo Starting SUMO simulation...
    start "SUMO Simulation" cmd /k "python launch_scenarios.py"
) else (
    echo WARNING: SUMO not found. Simulation will not start.
    echo To install SUMO: https://sumo.dlr.de/docs/Downloads.php
)

REM Wait for services to start
echo.
echo Waiting for services to start...
timeout /t 15 /nobreak >nul

REM Run health checks
echo.
echo Running health checks...
echo.

REM Check Backend API
curl -s http://localhost:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Backend API is healthy
) else (
    echo ✗ Backend API is not responding
)

REM Check ML API
curl -s http://localhost:8001/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ ML API is healthy
) else (
    echo ✗ ML API is not responding
)

REM Check CV Service
curl -s http://localhost:5001/cv/streams >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ CV Service is healthy
) else (
    echo ✗ CV Service is not responding
)

REM Check Frontend
curl -s http://localhost:3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ Frontend is healthy
) else (
    echo ✗ Frontend is not responding
)

REM Display access information
echo.
echo ================================================
echo Smart Traffic Management System is running!
echo ================================================
echo.
echo Access URLs:
echo   Frontend Dashboard: http://localhost:3000
echo   Backend API: http://localhost:8000
echo   ML API: http://localhost:8001
echo   CV Service: http://localhost:5001
echo   API Documentation: http://localhost:8000/docs
echo.
echo Services are running in separate command windows.
echo Close those windows to stop the services.
echo.
echo To stop all services, run: scripts\stop_project.bat
echo.
echo System is ready for use!
echo.

REM Keep the main window open
pause
