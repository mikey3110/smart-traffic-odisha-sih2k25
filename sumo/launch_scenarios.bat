@echo off
REM SUMO Scenario Launcher for Windows
REM Launches different traffic scenarios for testing ML optimization

echo SUMO Scenario Launcher
echo =====================

if "%1"=="--list" (
    echo Available Scenarios:
    echo   normal      - Normal traffic conditions (2 hours)
    echo   rush_hour   - Rush hour traffic (7 hours)
    echo   emergency   - Emergency vehicle priority testing (1 hour)
    echo.
    echo Usage: launch_scenarios.bat [scenario_name]
    echo Example: launch_scenarios.bat normal
    goto :eof
)

if "%1"=="--validate" (
    echo Validating scenarios...
    python test_scenarios.py
    goto :eof
)

if "%1"=="--batch" (
    echo Running all scenarios in batch...
    python launch_scenarios.py --batch
    goto :eof
)

if "%1"=="normal" (
    echo Launching normal traffic scenario...
    sumo-gui -c configs/normal_traffic.sumocfg --remote-port 8813
    goto :eof
)

if "%1"=="rush_hour" (
    echo Launching rush hour scenario...
    sumo-gui -c configs/rush_hour.sumocfg --remote-port 8813
    goto :eof
)

if "%1"=="emergency" (
    echo Launching emergency vehicle scenario...
    sumo-gui -c configs/emergency_vehicle.sumocfg --remote-port 8813
    goto :eof
)

if "%1"=="test" (
    echo Running scenario tests...
    python test_scenarios.py
    goto :eof
)

if "%1"=="" (
    echo SUMO Scenario Launcher
    echo =====================
    echo.
    echo Usage: launch_scenarios.bat [option]
    echo.
    echo Options:
    echo   normal      - Launch normal traffic scenario
    echo   rush_hour   - Launch rush hour scenario
    echo   emergency   - Launch emergency vehicle scenario
    echo   --list      - List available scenarios
    echo   --validate  - Validate all scenarios
    echo   --batch     - Run all scenarios in batch
    echo   test        - Run scenario tests
    echo.
    echo Example: launch_scenarios.bat normal
    goto :eof
)

echo Unknown option: %1
echo Use --list to see available options
