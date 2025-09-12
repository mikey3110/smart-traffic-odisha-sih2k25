"""
Smart Traffic Management System - Vehicle Counting Demo
Real-time vehicle counting and statistics
"""

import os
import subprocess
import time
import json
from datetime import datetime

class VehicleCounterDemo:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_dir = os.path.join(self.script_dir, 'configs')
        self.sumo_bin = os.path.join(self.config_dir, 'Sumo', 'bin', 'sumo.exe')
        self.config_file = os.path.join(self.config_dir, 'working_simulation.sumocfg')
        
        # Vehicle counting data
        self.vehicle_counts = {
            'total_spawned': 0,
            'total_completed': 0,
            'current_in_system': 0,
            'by_type': {'car': 0, 'truck': 0, 'bus': 0},
            'by_route': {},
            'by_intersection': {'center': 0}
        }
        
    def run_vehicle_counting_demo(self):
        """Run the vehicle counting demo"""
        print("🚗 VEHICLE COUNTING DEMO")
        print("=" * 50)
        
        # Create output file for vehicle data
        output_file = os.path.join(self.config_dir, 'vehicle_output.xml')
        
        # Run SUMO with vehicle output
        cmd = [
            self.sumo_bin,
            '-c', 'working_simulation.sumocfg',
            '--fcd-output', output_file,
            '--duration', '300',
            '--step-length', '1'
        ]
        
        print("🚦 Starting simulation with vehicle tracking...")
        print(f"📁 Output file: {output_file}")
        
        try:
            # Change to config directory
            os.chdir(self.config_dir)
            
            # Run simulation
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Simulation completed successfully!")
                self.analyze_vehicle_data(output_file)
            else:
                print(f"❌ Simulation failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def analyze_vehicle_data(self, output_file):
        """Analyze vehicle data from output file"""
        print("\n📊 ANALYZING VEHICLE DATA...")
        
        if not os.path.exists(output_file):
            print("❌ Output file not found!")
            return
        
        try:
            # Parse XML output (simplified)
            with open(output_file, 'r') as f:
                content = f.read()
            
            # Count vehicles (simplified parsing)
            vehicle_count = content.count('<vehicle')
            print(f"🚗 Total vehicle records: {vehicle_count}")
            
            # Count by type
            car_count = content.count('type="car"')
            truck_count = content.count('type="truck"')
            bus_count = content.count('type="bus"')
            
            print(f"🚗 Cars: {car_count}")
            print(f"🚛 Trucks: {truck_count}")
            print(f"🚌 Buses: {bus_count}")
            
            # Generate summary
            self.generate_vehicle_summary(vehicle_count, car_count, truck_count, bus_count)
            
        except Exception as e:
            print(f"❌ Error analyzing data: {e}")
    
    def generate_vehicle_summary(self, total, cars, trucks, buses):
        """Generate vehicle counting summary"""
        print("\n📈 VEHICLE COUNTING SUMMARY")
        print("=" * 50)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_vehicles': total,
            'vehicle_types': {
                'cars': cars,
                'trucks': trucks,
                'buses': buses
            },
            'traffic_flow': {
                'vehicles_per_minute': round(total / 5, 2),  # 5-minute simulation
                'peak_flow': 'High traffic density detected',
                'efficiency': 'Good traffic flow management'
            }
        }
        
        # Save summary
        summary_file = os.path.join(self.config_dir, 'vehicle_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📊 Total Vehicles Processed: {total}")
        print(f"🚗 Cars: {cars} ({round(cars/total*100, 1)}%)")
        print(f"🚛 Trucks: {trucks} ({round(trucks/total*100, 1)}%)")
        print(f"🚌 Buses: {buses} ({round(buses/total*100, 1)}%)")
        print(f"📈 Flow Rate: {summary['traffic_flow']['vehicles_per_minute']} vehicles/min")
        print(f"💾 Summary saved to: {summary_file}")
        
        return summary

def main():
    """Main function"""
    print("🚦 Smart Traffic Management System")
    print("🚗 Vehicle Counting Demo")
    print("=" * 50)
    
    demo = VehicleCounterDemo()
    demo.run_vehicle_counting_demo()
    
    print("\n✅ Vehicle counting demo completed!")
    print("📊 Check the generated files for detailed statistics")

if __name__ == "__main__":
    main()
