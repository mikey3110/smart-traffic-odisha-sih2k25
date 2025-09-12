@echo off
echo Installing SUMO Traffic Simulator...

REM Download SUMO installer
echo Downloading SUMO installer...
powershell -Command "Invoke-WebRequest -Uri 'https://sumo.dlr.de/releases/1.18.0/sumo-win64-1.18.0.msi' -OutFile 'sumo-installer.msi'"

REM Install SUMO
echo Installing SUMO...
msiexec /i sumo-installer.msi /quiet

REM Add SUMO to PATH
echo Adding SUMO to PATH...
setx PATH "%PATH%;C:\Program Files (x86)\Eclipse\Sumo\bin" /M

echo SUMO installation complete!
echo Please restart your command prompt and run: sumo-gui --version
pause
