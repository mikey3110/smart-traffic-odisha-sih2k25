"""
Simple Vehicle Counter - Count vehicles in SUMO simulation
"""

import os
import subprocess
import time
import json
from datetime import datetime

def count_vehicles_in_simulation():
    """Count vehicles in the simulation"""
    print("🚗 VEHICLE COUNTING DEMO")
    print("=" * 50)
    
    # Get current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, 'configs')
    sumo_bin = os.path.join(config_dir, 'Sumo', 'bin', 'sumo.exe')
    
    # Change to config directory
    os.chdir(config_dir)
    
    print("🚦 Running simulation with vehicle counting...")
    print("📊 This will count all vehicles that spawn during the simulation")
    print()
    
    # Run simulation and count vehicles
    cmd = [sumo_bin, '-c', 'working_simulation.sumocfg', '--duration', '300']
    
    try:
        # Run simulation
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Count vehicles from output
        output = result.stdout + result.stderr
        
        # Count different types of vehicles
        total_vehicles = output.count('Inserted vehicle')
        cars = output.count('type="car"')
        trucks = output.count('type="truck"')
        buses = output.count('type="bus"')
        
        # Count completed vehicles
        completed = output.count('Removed vehicle')
        
        # Count collisions and warnings
        collisions = output.count('collision')
        teleports = output.count('teleport')
        emergency_braking = output.count('emergency braking')
        
        # Display results
        print("📊 VEHICLE COUNTING RESULTS")
        print("=" * 50)
        print(f"🚗 Total Vehicles Spawned: {total_vehicles}")
        print(f"✅ Vehicles Completed Journey: {completed}")
        print(f"🔄 Vehicles Still in System: {total_vehicles - completed}")
        print()
        print("📈 Vehicle Type Breakdown:")
        print(f"   🚗 Cars: {cars} ({round(cars/total_vehicles*100, 1)}%)")
        print(f"   🚛 Trucks: {trucks} ({round(trucks/total_vehicles*100, 1)}%)")
        print(f"   🚌 Buses: {buses} ({round(buses/total_vehicles*100, 1)}%)")
        print()
        print("⚠️ Traffic Events:")
        print(f"   💥 Collisions: {collisions}")
        print(f"   🚀 Teleports: {teleports}")
        print(f"   🛑 Emergency Braking: {emergency_braking}")
        print()
        print("🎯 Traffic Flow Analysis:")
        print(f"   📊 Flow Rate: {round(total_vehicles/5, 2)} vehicles/minute")
        print(f"   ⏱️ Average Processing Time: {round(300/total_vehicles, 2)} seconds/vehicle")
        print(f"   🎯 System Efficiency: {round(completed/total_vehicles*100, 1)}%")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_vehicles': total_vehicles,
            'completed_vehicles': completed,
            'vehicle_types': {
                'cars': cars,
                'trucks': trucks,
                'buses': buses
            },
            'traffic_events': {
                'collisions': collisions,
                'teleports': teleports,
                'emergency_braking': emergency_braking
            },
            'flow_rate': round(total_vehicles/5, 2),
            'efficiency': round(completed/total_vehicles*100, 1)
        }
        
        # Save to file
        results_file = os.path.join(config_dir, 'vehicle_counting_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {results_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main function"""
    print("🚦 Smart Traffic Management System")
    print("🚗 Vehicle Counting Demo")
    print("=" * 50)
    
    results = count_vehicles_in_simulation()
    
    if results:
        print("\n✅ Vehicle counting completed successfully!")
        print("📊 Check the generated JSON file for detailed statistics")
    else:
        print("\n❌ Vehicle counting failed!")

if __name__ == "__main__":
    main()
