"""
Live Vehicle Counter - Real-time vehicle counting during simulation
"""

import os
import subprocess
import time
import json
from datetime import datetime

class LiveVehicleCounter:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_dir = os.path.join(self.script_dir, 'configs')
        self.sumo_bin = os.path.join(self.config_dir, 'Sumo', 'bin', 'sumo.exe')
        self.config_file = os.path.join(self.config_dir, 'working_simulation.sumocfg')
        
        # Live counting data
        self.current_vehicles = 0
        self.total_spawned = 0
        self.total_completed = 0
        self.vehicle_types = {'car': 0, 'truck': 0, 'bus': 0}
        
    def run_live_counter(self):
        """Run live vehicle counting"""
        print("🚗 LIVE VEHICLE COUNTER")
        print("=" * 50)
        print("Press Ctrl+C to stop")
        print()
        
        try:
            # Start simulation with live output
            cmd = [
                self.sumo_bin,
                '-c', 'working_simulation.sumocfg',
                '--duration', '300',
                '--step-length', '1',
                '--verbose'
            ]
            
            print("🚦 Starting simulation...")
            print("📊 Live vehicle counting will begin...")
            print()
            
            # Change to config directory
            os.chdir(self.config_dir)
            
            # Run simulation and capture output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                     universal_newlines=True, bufsize=1)
            
            # Monitor output for vehicle events
            for line in iter(process.stdout.readline, ''):
                if line:
                    self.process_simulation_output(line)
                    
            process.wait()
            
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def process_simulation_output(self, line):
        """Process simulation output to count vehicles"""
        line = line.strip()
        
        # Count vehicle spawns
        if 'Inserted' in line and 'vehicle' in line:
            self.total_spawned += 1
            self.current_vehicles += 1
            
            # Determine vehicle type
            if 'type="car"' in line or 'car' in line.lower():
                self.vehicle_types['car'] += 1
            elif 'type="truck"' in line or 'truck' in line.lower():
                self.vehicle_types['truck'] += 1
            elif 'type="bus"' in line or 'bus' in line.lower():
                self.vehicle_types['bus'] += 1
            
            self.display_live_count()
        
        # Count vehicle completions
        elif 'Removed' in line and 'vehicle' in line:
            self.total_completed += 1
            self.current_vehicles -= 1
            self.display_live_count()
        
        # Show simulation progress
        elif 'Simulation ended' in line:
            self.display_final_summary()
    
    def display_live_count(self):
        """Display live vehicle count"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] 🚗 Current: {self.current_vehicles} | "
              f"Total Spawned: {self.total_spawned} | "
              f"Completed: {self.total_completed}")
        
        # Show vehicle type breakdown
        if self.total_spawned > 0:
            car_pct = round(self.vehicle_types['car'] / self.total_spawned * 100, 1)
            truck_pct = round(self.vehicle_types['truck'] / self.total_spawned * 100, 1)
            bus_pct = round(self.vehicle_types['bus'] / self.total_spawned * 100, 1)
            
            print(f"         🚗 Cars: {self.vehicle_types['car']} ({car_pct}%) | "
                  f"🚛 Trucks: {self.vehicle_types['truck']} ({truck_pct}%) | "
                  f"🚌 Buses: {self.vehicle_types['bus']} ({bus_pct}%)")
    
    def display_final_summary(self):
        """Display final summary"""
        print("\n" + "=" * 50)
        print("📊 FINAL VEHICLE COUNTING SUMMARY")
        print("=" * 50)
        print(f"🚗 Total Vehicles Spawned: {self.total_spawned}")
        print(f"✅ Total Vehicles Completed: {self.total_completed}")
        print(f"🔄 Vehicles Still in System: {self.current_vehicles}")
        print()
        print("📈 Vehicle Type Breakdown:")
        print(f"   🚗 Cars: {self.vehicle_types['car']} ({round(self.vehicle_types['car']/self.total_spawned*100, 1)}%)")
        print(f"   🚛 Trucks: {self.vehicle_types['truck']} ({round(self.vehicle_types['truck']/self.total_spawned*100, 1)}%)")
        print(f"   🚌 Buses: {self.vehicle_types['bus']} ({round(self.vehicle_types['bus']/self.total_spawned*100, 1)}%)")
        print()
        print("🎯 Traffic Flow Analysis:")
        print(f"   📊 Flow Rate: {round(self.total_spawned/5, 2)} vehicles/minute")
        print(f"   ⏱️ Average Processing Time: {round(300/self.total_spawned, 2)} seconds/vehicle")
        print(f"   🎯 System Efficiency: {round(self.total_completed/self.total_spawned*100, 1)}%")

def main():
    """Main function"""
    print("🚦 Smart Traffic Management System")
    print("🚗 Live Vehicle Counter")
    print("=" * 50)
    
    counter = LiveVehicleCounter()
    counter.run_live_counter()

if __name__ == "__main__":
    main()
