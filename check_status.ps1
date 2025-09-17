# Quick Status Checker for Smart Traffic Management System
Write-Host "🚦 Smart Traffic Management System - Status Check" -ForegroundColor Green
Write-Host "=" * 60 -ForegroundColor Cyan

# Check if processes are running
$processes = Get-Process | Where-Object {$_.ProcessName -like "*python*" -or $_.ProcessName -like "*node*"}

Write-Host "`n📊 Running Processes:" -ForegroundColor Yellow
if ($processes) {
    $processes | ForEach-Object {
        Write-Host "  ✅ $($_.ProcessName) (PID: $($_.Id))" -ForegroundColor Green
    }
} else {
    Write-Host "  ❌ No Python/Node processes found" -ForegroundColor Red
}

# Check ports
Write-Host "`n🌐 Port Status:" -ForegroundColor Yellow
$ports = @(3000, 8000, 8001, 5001)
foreach ($port in $ports) {
    $connection = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connection) {
        Write-Host "  ✅ Port $port - $($connection.State)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Port $port - Not listening" -ForegroundColor Red
    }
}

Write-Host "`n🌐 Access URLs:" -ForegroundColor Cyan
Write-Host "  Frontend Dashboard: http://localhost:3000" -ForegroundColor White
Write-Host "  Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "  API Documentation: http://localhost:8000/docs" -ForegroundColor White
Write-Host "  ML API: http://localhost:8001" -ForegroundColor White
Write-Host "  CV Service: http://localhost:5001" -ForegroundColor White
Write-Host "  Simulation Dashboard: file:///$PWD/simulation_dashboard.html" -ForegroundColor White

Write-Host "`n🎯 Quick Actions:" -ForegroundColor Yellow
Write-Host "  1. Open simulation_dashboard.html in your browser" -ForegroundColor White
Write-Host "  2. Open http://localhost:3000 for the main dashboard" -ForegroundColor White
Write-Host "  3. Open http://localhost:8000/docs for API documentation" -ForegroundColor White

Write-Host "`nPress any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown')
