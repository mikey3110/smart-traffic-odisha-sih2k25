"""
Smart Traffic Management System - Simulation Types
Different simulation scenarios for testing and demonstration
"""

import os
import subprocess
import time
from datetime import datetime

class SimulationTypes:
    def __init__(self):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.sumo_bin_dir = os.path.join(self.script_dir, 'Sumo', 'bin')
        self.sumo_gui = os.path.join(self.sumo_bin_dir, 'sumo-gui.exe')
        self.sumo_cmd = os.path.join(self.sumo_bin_dir, 'sumo.exe')
        
    def run_simulation(self, config_file, simulation_type="gui", duration=300):
        """Run a specific simulation type"""
        os.chdir(self.script_dir)
        
        if simulation_type == "gui":
            return self._run_gui_simulation(config_file)
        elif simulation_type == "headless":
            return self._run_headless_simulation(config_file, duration)
        elif simulation_type == "batch":
            return self._run_batch_simulation(config_file, duration)
        elif simulation_type == "demo":
            return self._run_demo_simulation(config_file)
        else:
            print(f"❌ Unknown simulation type: {simulation_type}")
            return False
    
    def _run_gui_simulation(self, config_file):
        """Run simulation with GUI"""
        print("🚦 Starting GUI Simulation...")
        try:
            cmd = [self.sumo_gui, "-c", config_file, "--start", "--quit-on-end"]
            process = subprocess.Popen(cmd, cwd=self.script_dir)
            print("✅ GUI Simulation started!")
            return True
        except Exception as e:
            print(f"❌ GUI Simulation failed: {e}")
            return False
    
    def _run_headless_simulation(self, config_file, duration):
        """Run simulation without GUI (headless)"""
        print("🤖 Starting Headless Simulation...")
        try:
            cmd = [self.sumo_cmd, "-c", config_file, "--duration", str(duration)]
            result = subprocess.run(cmd, cwd=self.script_dir, capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Headless Simulation completed!")
                return True
            else:
                print(f"❌ Headless Simulation failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Headless Simulation failed: {e}")
            return False
    
    def _run_batch_simulation(self, config_file, duration):
        """Run multiple simulations in batch"""
        print("📊 Starting Batch Simulation...")
        scenarios = [
            {"name": "Low Traffic", "period": 10, "probability": 0.1},
            {"name": "Medium Traffic", "period": 5, "probability": 0.3},
            {"name": "High Traffic", "period": 2, "probability": 0.6},
            {"name": "Rush Hour", "period": 1, "probability": 0.8}
        ]
        
        results = []
        for i, scenario in enumerate(scenarios):
            print(f"🔄 Running scenario {i+1}: {scenario['name']}")
            # Modify routes for this scenario
            self._modify_routes_for_scenario(scenario)
            
            # Run simulation
            success = self._run_headless_simulation(config_file, duration)
            results.append({
                "scenario": scenario['name'],
                "success": success,
                "timestamp": datetime.now().isoformat()
            })
            
            time.sleep(2)  # Wait between simulations
        
        print("📈 Batch Simulation Results:")
        for result in results:
            status = "✅" if result['success'] else "❌"
            print(f"  {status} {result['scenario']}: {result['timestamp']}")
        
        return all(r['success'] for r in results)
    
    def _run_demo_simulation(self, config_file):
        """Run demonstration simulation with visual effects"""
        print("🎬 Starting Demo Simulation...")
        try:
            # Run with GUI and special demo settings
            cmd = [
                self.sumo_gui, 
                "-c", config_file,
                "--start",
                "--step-length", "0.1",  # Slower for demo
                "--delay", "100",        # Delay between steps
                "--quit-on-end"
            ]
            process = subprocess.Popen(cmd, cwd=self.script_dir)
            print("✅ Demo Simulation started!")
            print("🎯 Watch the traffic lights and vehicle movements!")
            return True
        except Exception as e:
            print(f"❌ Demo Simulation failed: {e}")
            return False
    
    def _modify_routes_for_scenario(self, scenario):
        """Modify routes file for specific scenario"""
        routes_file = os.path.join(self.script_dir, 'routes.rou.xml')
        # This would modify the routes file based on scenario
        # For now, just print the scenario
        print(f"  📝 Applying scenario: {scenario['name']}")
    
    def list_simulation_types(self):
        """List available simulation types"""
        print("🚦 Available Simulation Types:")
        print("  1. GUI Simulation - Visual simulation with SUMO GUI")
        print("  2. Headless Simulation - Command-line simulation")
        print("  3. Batch Simulation - Multiple scenarios in sequence")
        print("  4. Demo Simulation - Slower, visual demonstration")
        print("  5. Custom Simulation - User-defined parameters")

def main():
    """Main function to run different simulation types"""
    simulator = SimulationTypes()
    
    print("🚦 Smart Traffic Management System - Simulation Types")
    print("=" * 60)
    
    # List available types
    simulator.list_simulation_types()
    
    print("\n🎯 Running GUI Simulation...")
    success = simulator.run_simulation("working_simulation.sumocfg", "gui")
    
    if success:
        print("\n✅ Simulation completed successfully!")
    else:
        print("\n❌ Simulation failed!")

if __name__ == "__main__":
    main()
