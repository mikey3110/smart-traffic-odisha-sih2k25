@echo off
REM Smart Traffic Management System - Windows Stop Script
REM This script stops all components of the system on Windows

setlocal enabledelayedexpansion

echo ================================================
echo Stopping Smart Traffic Management System
echo ================================================
echo.

REM Stop processes by port
echo Stopping services by port...

REM Stop Frontend (port 3000)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000') do (
    echo Stopping Frontend (PID: %%a)...
    taskkill /PID %%a /F >nul 2>&1
)

REM Stop Backend API (port 8000)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000') do (
    echo Stopping Backend API (PID: %%a)...
    taskkill /PID %%a /F >nul 2>&1
)

REM Stop ML API (port 8001)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8001') do (
    echo Stopping ML API (PID: %%a)...
    taskkill /PID %%a /F >nul 2>&1
)

REM Stop CV Service (port 5001)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :5001') do (
    echo Stopping CV Service (PID: %%a)...
    taskkill /PID %%a /F >nul 2>&1
)

REM Stop SUMO (port 8813)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8813') do (
    echo Stopping SUMO Simulation (PID: %%a)...
    taskkill /PID %%a /F >nul 2>&1
)

REM Stop Docker services
echo Stopping Docker services...
cd /d "%~dp0..\src\backend"
docker-compose down >nul 2>&1
cd /d "%~dp0.."

REM Kill any remaining Python processes related to our project
echo Cleaning up Python processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Backend API*" >nul 2>&1
taskkill /F /IM python.exe /FI "WINDOWTITLE eq ML API*" >nul 2>&1
taskkill /F /IM python.exe /FI "WINDOWTITLE eq HLS Streaming*" >nul 2>&1
taskkill /F /IM python.exe /FI "WINDOWTITLE eq CV Demo*" >nul 2>&1
taskkill /F /IM python.exe /FI "WINDOWTITLE eq SUMO Simulation*" >nul 2>&1

REM Kill any remaining Node.js processes
echo Cleaning up Node.js processes...
taskkill /F /IM node.exe /FI "WINDOWTITLE eq Frontend*" >nul 2>&1

REM Kill any remaining cmd windows with our titles
echo Cleaning up command windows...
taskkill /F /FI "WINDOWTITLE eq Backend API*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq ML API*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq HLS Streaming*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq CV Demo*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq Frontend*" >nul 2>&1
taskkill /F /FI "WINDOWTITLE eq SUMO Simulation*" >nul 2>&1

echo.
echo ✓ All services stopped
echo.

REM Check if services are still running
echo Checking remaining services...
echo.

set SERVICES_STOPPED=1

REM Check Frontend
netstat -ano | findstr :3000 >nul 2>&1
if %errorlevel% equ 0 (
    echo WARNING: Frontend is still running on port 3000
    set SERVICES_STOPPED=0
) else (
    echo ✓ Frontend stopped
)

REM Check Backend API
netstat -ano | findstr :8000 >nul 2>&1
if %errorlevel% equ 0 (
    echo WARNING: Backend API is still running on port 8000
    set SERVICES_STOPPED=0
) else (
    echo ✓ Backend API stopped
)

REM Check ML API
netstat -ano | findstr :8001 >nul 2>&1
if %errorlevel% equ 0 (
    echo WARNING: ML API is still running on port 8001
    set SERVICES_STOPPED=0
) else (
    echo ✓ ML API stopped
)

REM Check CV Service
netstat -ano | findstr :5001 >nul 2>&1
if %errorlevel% equ 0 (
    echo WARNING: CV Service is still running on port 5001
    set SERVICES_STOPPED=0
) else (
    echo ✓ CV Service stopped
)

REM Check SUMO
netstat -ano | findstr :8813 >nul 2>&1
if %errorlevel% equ 0 (
    echo WARNING: SUMO Simulation is still running on port 8813
    set SERVICES_STOPPED=0
) else (
    echo ✓ SUMO Simulation stopped
)

echo.
if %SERVICES_STOPPED% equ 1 (
    echo ✓ All services successfully stopped
) else (
    echo WARNING: Some services may still be running
    echo You may need to manually close any remaining command windows
)

echo.
pause
