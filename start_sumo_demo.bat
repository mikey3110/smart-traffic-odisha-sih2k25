@echo off
echo ========================================
echo Smart Traffic Management System
echo SUMO Demo - Visual Traffic Simulation
echo ========================================
echo.

echo Starting SUMO Traffic Simulation...
cd sumo
start "SUMO Traffic Demo" sumo-gui -c configs/simple_demo.sumocfg

echo.
echo ========================================
echo SUMO-GUI should now be opening!
echo ========================================
echo.
echo What you should see:
echo - A traffic intersection with 4 roads
echo - Cars, buses, and motorcycles moving
echo - Different colored vehicles
echo - Traffic flowing in all directions
echo.
echo Controls:
echo - Click PLAY (▶️) to start simulation
echo - Use speed controls to slow down/speed up
echo - Watch vehicles move through intersection
echo.
echo Press any key to close this window...
pause > nul
