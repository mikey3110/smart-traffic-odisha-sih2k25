"""
Quick Simulation Launcher
Run different types of SUMO simulations
"""

import os
import sys
import subprocess
from datetime import datetime

def run_gui_simulation():
    """Run GUI simulation"""
    print("🚦 Starting GUI Simulation...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, 'configs')
    sumo_bin = os.path.join(config_dir, 'Sumo', 'bin', 'sumo-gui.exe')
    
    os.chdir(config_dir)
    cmd = [sumo_bin, "-c", "working_simulation.sumocfg", "--start", "--quit-on-end"]
    
    try:
        process = subprocess.Popen(cmd)
        print("✅ GUI Simulation started!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_headless_simulation():
    """Run headless simulation"""
    print("🤖 Starting Headless Simulation...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, 'configs')
    sumo_cmd = os.path.join(config_dir, 'Sumo', 'bin', 'sumo.exe')
    
    os.chdir(config_dir)
    cmd = [sumo_cmd, "-c", "working_simulation.sumocfg", "--duration", "300"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Headless Simulation completed!")
            print("📊 Simulation Results:")
            print(result.stdout)
            return True
        else:
            print(f"❌ Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def run_demo_simulation():
    """Run demo simulation with slow motion"""
    print("🎬 Starting Demo Simulation...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, 'configs')
    sumo_gui = os.path.join(config_dir, 'Sumo', 'bin', 'sumo-gui.exe')
    
    os.chdir(config_dir)
    cmd = [
        sumo_gui, 
        "-c", "working_simulation.sumocfg",
        "--start",
        "--step-length", "0.2",
        "--delay", "200",
        "--quit-on-end"
    ]
    
    try:
        process = subprocess.Popen(cmd)
        print("✅ Demo Simulation started!")
        print("🎯 Watch the traffic lights and vehicle movements!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function"""
    print("🚦 Smart Traffic Management System - Simulation Launcher")
    print("=" * 60)
    print("Available simulation types:")
    print("1. GUI Simulation (Visual)")
    print("2. Headless Simulation (Command-line)")
    print("3. Demo Simulation (Slow motion)")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                run_gui_simulation()
                break
            elif choice == "2":
                run_headless_simulation()
                break
            elif choice == "3":
                run_demo_simulation()
                break
            elif choice == "4":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-4.")
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
