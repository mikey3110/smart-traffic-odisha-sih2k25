# Quick Start Script for Smart Traffic Management System
Write-Host "🚀 Starting Smart Traffic Management System..." -ForegroundColor Green

# Start Frontend
Write-Host "📱 Starting Frontend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'src\frontend\smart-traffic-ui'; npm run dev"

# Wait a moment
Start-Sleep -Seconds 3

# Start Backend
Write-Host "🔧 Starting Backend..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'src\backend'; python main.py"

# Wait a moment
Start-Sleep -Seconds 3

# Start ML API
Write-Host "🤖 Starting ML API..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'src\ml_engine'; python api\ml_api.py"

# Wait a moment
Start-Sleep -Seconds 3

# Start CV Service
Write-Host "👁️ Starting Computer Vision Service..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd 'src\computer_vision'; python demo_integration.py"

Write-Host "✅ All services started!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Access URLs:" -ForegroundColor Cyan
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "  Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host "  ML API: http://localhost:8001" -ForegroundColor White
Write-Host "  CV Service: http://localhost:5001" -ForegroundColor White
Write-Host ""
Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
