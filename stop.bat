@echo off
REM Simple stop script for Windows users

echo ================================================
echo Stopping Smart Traffic Management System
echo ================================================
echo.

REM Check if Docker Compose is running
docker-compose ps >nul 2>&1
if %errorlevel% equ 0 (
    echo Stopping Docker Compose services...
    docker-compose down
    echo.
    echo ✓ Docker services stopped
) else (
    echo Docker Compose not running, stopping individual services...
    call scripts\stop_project.bat
)

echo.
echo ================================================
echo All services stopped!
echo ================================================
echo.
pause
