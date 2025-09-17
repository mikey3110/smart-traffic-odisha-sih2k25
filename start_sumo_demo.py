#!/usr/bin/env python3
"""
Simple SUMO Demo - Visual Traffic Simulation
This will show you a working traffic simulation
"""

import subprocess
import time
import webbrowser
import os
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("=" * 60)
    print("🚦 Smart Traffic Management System")
    print("   SUMO Visual Demo")
    print("=" * 60)
    print()

def main():
    """Main function to start SUMO demo"""
    print_banner()
    
    # Get the project root directory
    project_root = Path(__file__).parent.absolute()
    sumo_dir = project_root / "sumo"
    
    print("🚀 Starting SUMO Traffic Simulation...")
    print()
    
    # Change to sumo directory
    os.chdir(sumo_dir)
    
    # Start SUMO-GUI
    print("📱 Opening SUMO-GUI...")
    try:
        subprocess.Popen(["sumo-gui", "-c", "configs/simple_demo.sumocfg"])
        print("✅ SUMO-GUI started successfully!")
    except Exception as e:
        print(f"❌ Error starting SUMO-GUI: {e}")
        return
    
    print()
    print("=" * 60)
    print("🎉 SUMO-GUI should now be opening!")
    print("=" * 60)
    print()
    print("👀 What you should see:")
    print("   • A traffic intersection with 4 roads")
    print("   • Cars, buses, and motorcycles moving")
    print("   • Different colored vehicles (red, blue, green, yellow)")
    print("   • Traffic flowing in all directions")
    print()
    print("🎮 Controls:")
    print("   • Click PLAY (▶️) to start simulation")
    print("   • Use speed controls to slow down/speed up")
    print("   • Watch vehicles move through intersection")
    print("   • Try different speeds to see traffic patterns")
    print()
    print("🔍 What to look for:")
    print("   • Red cars going North to South")
    print("   • Blue cars going South to North")
    print("   • Green cars going East to West")
    print("   • Yellow cars going West to East")
    print("   • Buses (larger vehicles)")
    print("   • Motorcycles (smaller vehicles)")
    print()
    print("⏹️  To stop: Close the SUMO-GUI window")
    print("=" * 60)
    
    # Keep the script running
    try:
        input("\nPress Enter to close this window...")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
