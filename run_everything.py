#!/usr/bin/env python3
"""
Smart Traffic Management System - Complete Startup Script
This script will start all services and open SUMO-GUI for you
"""

import subprocess
import time
import webbrowser
import os
import sys
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("=" * 60)
    print("🚦 Smart Traffic Management System")
    print("   Complete Startup Script")
    print("=" * 60)
    print()

def start_service(name, command, working_dir=None):
    """Start a service in a new window"""
    print(f"🚀 Starting {name}...")
    try:
        if working_dir:
            os.chdir(working_dir)
        
        # Start the service in a new window
        subprocess.Popen(command, shell=True, creationflags=subprocess.CREATE_NEW_CONSOLE)
        print(f"✅ {name} started successfully")
        time.sleep(2)  # Give it time to start
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")

def open_browser(url, description):
    """Open a URL in the default browser"""
    print(f"🌐 Opening {description}...")
    try:
        webbrowser.open(url)
        print(f"✅ {description} opened")
    except Exception as e:
        print(f"❌ Failed to open {description}: {e}")

def main():
    """Main function to start everything"""
    print_banner()
    
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    
    print("📋 Starting all services...")
    print()
    
    # 1. Start Backend API
    start_service(
        "Backend API",
        "python main.py",
        project_root / "src" / "backend"
    )
    
    # 2. Start ML API
    start_service(
        "ML API",
        "python api/ml_api.py",
        project_root / "src" / "ml_engine"
    )
    
    # 3. Start Computer Vision Service
    start_service(
        "Computer Vision Service",
        "python demo_integration.py",
        project_root / "src" / "computer_vision"
    )
    
    # 4. Start Frontend
    start_service(
        "Frontend",
        "npm run dev",
        project_root / "src" / "frontend" / "smart-traffic-ui"
    )
    
    # 5. Start SUMO Simulations
    print("🚦 Starting SUMO Simulations...")
    
    # Normal Traffic
    start_service(
        "SUMO Normal Traffic",
        "sumo-gui -c configs/normal_traffic.sumocfg",
        project_root / "sumo"
    )
    
    # Rush Hour
    start_service(
        "SUMO Rush Hour",
        "sumo-gui -c configs/rush_hour.sumocfg",
        project_root / "sumo"
    )
    
    # Emergency Vehicle
    start_service(
        "SUMO Emergency Vehicle",
        "sumo-gui -c configs/emergency_vehicle.sumocfg",
        project_root / "sumo"
    )
    
    # 6. Open Web Dashboard
    dashboard_path = project_root / "simulation_dashboard.html"
    if dashboard_path.exists():
        open_browser(f"file://{dashboard_path}", "Simulation Dashboard")
    
    # 7. Open API Documentation
    open_browser("http://localhost:8000/docs", "API Documentation")
    
    print()
    print("=" * 60)
    print("🎉 All services started successfully!")
    print("=" * 60)
    print()
    print("📊 Services running:")
    print("   • Backend API: http://localhost:8000")
    print("   • ML API: http://localhost:8001")
    print("   • CV Service: http://localhost:5001")
    print("   • Frontend: http://localhost:3000")
    print("   • SUMO GUI: 3 simulation windows opened")
    print("   • Dashboard: simulation_dashboard.html opened")
    print()
    print("🔍 What you can see:")
    print("   • SUMO-GUI windows showing traffic simulation")
    print("   • Web dashboard with real-time data")
    print("   • API documentation for testing")
    print()
    print("⏹️  To stop all services, close this window and all opened windows")
    print("=" * 60)
    
    # Keep the script running
    try:
        input("\nPress Enter to stop all services...")
    except KeyboardInterrupt:
        pass
    
    print("\n🛑 Stopping all services...")
    print("Please close all opened windows manually to stop services.")

if __name__ == "__main__":
    main()
