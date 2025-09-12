#!/usr/bin/env python3
"""
Working SUMO GUI Simulator
"""

import os
import sys
import subprocess
import time
from datetime import datetime

class WorkingSumoGUI:
    def __init__(self):
        # Get the correct paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_dir = os.path.join(self.script_dir, 'configs')
        self.sumo_bin_dir = os.path.join(self.config_dir, 'Sumo', 'bin')
        
        # SUMO executables
        self.sumo_gui = os.path.join(self.sumo_bin_dir, 'sumo-gui.exe')
        self.sumo_cmd = os.path.join(self.sumo_bin_dir, 'sumo.exe')
        self.config_file = os.path.join(self.config_dir, 'working_simulation.sumocfg')
        
        print(f"📁 Script Directory: {self.script_dir}")
        print(f"📁 Config Directory: {self.config_dir}")
        print(f"📁 SUMO Bin Directory: {self.sumo_bin_dir}")
        print(f"🎯 SUMO GUI: {self.sumo_gui}")
        print(f"⚙️ Config File: {self.config_file}")
        
    def check_requirements(self):
        """Check if all requirements are met"""
        print("\n🔍 CHECKING REQUIREMENTS...")
        
        # Check SUMO GUI
        if os.path.exists(self.sumo_gui):
            print("✅ SUMO GUI found")
        else:
            print("❌ SUMO GUI not found")
            return False
            
        # Check SUMO command line
        if os.path.exists(self.sumo_cmd):
            print("✅ SUMO command line found")
        else:
            print("❌ SUMO command line not found")
            return False
            
        # Check config file
        if os.path.exists(self.config_file):
            print("✅ Config file found")
        else:
            print("❌ Config file not found")
            return False
            
        return True
        
    def run_sumo_gui(self):
        """Run SUMO GUI directly"""
        print("\n🚀 STARTING SUMO GUI...")
        
        if not self.check_requirements():
            print("❌ Requirements not met!")
            return False
            
        try:
            # Change to config directory
            os.chdir(self.config_dir)
            print(f"📁 Changed to directory: {self.config_dir}")
            
            # Run SUMO GUI
            cmd = [self.sumo_gui, "-c", "working_simulation.sumocfg", "--start", "--quit-on-end"]
            print(f"🎯 Running command: {' '.join(cmd)}")
            
            # Start SUMO GUI
            process = subprocess.Popen(cmd, cwd=self.config_dir)
            print("✅ SUMO GUI started successfully!")
            print("🎮 GUI should open in a new window")
            
            return True
            
        except Exception as e:
            print(f"❌ Error starting SUMO GUI: {e}")
            return False
            
    def run_sumo_command_line(self):
        """Run SUMO command line simulation"""
        print("\n🚀 STARTING SUMO COMMAND LINE...")
        
        if not self.check_requirements():
            print("❌ Requirements not met!")
            return False
            
        try:
            # Change to config directory
            os.chdir(self.config_dir)
            print(f"📁 Changed to directory: {self.config_dir}")
            
            # Run SUMO command line
            cmd = [self.sumo_cmd, "-c", "working_simulation.sumocfg"]
            print(f"🎯 Running command: {' '.join(cmd)}")
            
            # Start SUMO
            result = subprocess.run(cmd, cwd=self.config_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ SUMO simulation completed successfully!")
                print("📊 Simulation output:")
                print(result.stdout)
            else:
                print("❌ SUMO simulation failed!")
                print("Error output:")
                print(result.stderr)
                
            return result.returncode == 0
            
        except Exception as e:
            print(f"❌ Error running SUMO: {e}")
            return False

def main():
    """Main function"""
    print("🚦 SMART TRAFFIC MANAGEMENT - SUMO GUI LAUNCHER")
    print("=" * 60)
    
    # Create launcher
    launcher = WorkingSumoGUI()
    
    # Check requirements
    if not launcher.check_requirements():
        print("\n❌ Cannot proceed - requirements not met!")
        return
    
    print("\n🎯 CHOOSE AN OPTION:")
    print("1. Run SUMO GUI (Visual Interface)")
    print("2. Run SUMO Command Line (Text Output)")
    print("3. Check Requirements Only")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        launcher.run_sumo_gui()
    elif choice == "2":
        launcher.run_sumo_command_line()
    elif choice == "3":
        print("✅ Requirements check complete!")
    else:
        print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
