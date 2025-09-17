@echo off
REM Simple startup script for Windows users
REM This script will start the project using the easiest method available

echo ================================================
echo Smart Traffic Management System
echo Quick Start for Windows
echo ================================================
echo.

REM Check if Docker is available
where docker >nul 2>&1
if %errorlevel% equ 0 (
    echo Docker found! Starting with Docker Compose...
    echo.
    
    REM Start with Docker Compose
    docker-compose up -d
    
    echo.
    echo ================================================
    echo Services are starting up...
    echo ================================================
    echo.
    echo Please wait 2-3 minutes for all services to start.
    echo.
    echo Access URLs:
    echo   Frontend: http://localhost:3000
    echo   Backend API: http://localhost:8000
    echo   API Docs: http://localhost:8000/docs
    echo   ML API: http://localhost:8001
    echo   CV Service: http://localhost:5001
    echo.
    echo To stop services: docker-compose down
    echo.
    pause
) else (
    echo Docker not found. Starting with individual services...
    echo.
    
    REM Check if Node.js is available
    where node >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Node.js not found!
        echo Please install Node.js from https://nodejs.org/
        pause
        exit /b 1
    )
    
    REM Check if Python is available
    where python >nul 2>&1
    if %errorlevel% neq 0 (
        echo ERROR: Python not found!
        echo Please install Python from https://python.org/
        pause
        exit /b 1
    )
    
    echo Starting individual services...
    echo.
    
    REM Start the project using the batch script
    call scripts\start_project.bat
)
