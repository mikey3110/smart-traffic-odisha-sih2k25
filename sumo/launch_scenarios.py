#!/usr/bin/env python3
"""
SUMO Scenario Launcher
Launches different traffic scenarios for testing ML optimization
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

class SUMOScenarioLauncher:
    def __init__(self, sumo_path=None):
        """
        Initialize the SUMO scenario launcher
        
        Args:
            sumo_path: Path to SUMO installation (optional)
        """
        self.sumo_path = sumo_path or self._find_sumo_path()
        self.scenarios = {
            'normal': {
                'config': 'configs/normal_traffic.sumocfg',
                'description': 'Normal traffic conditions (2 hours)',
                'duration': 7200
            },
            'rush_hour': {
                'config': 'configs/rush_hour.sumocfg',
                'description': 'Rush hour traffic (7 hours)',
                'duration': 25200
            },
            'emergency': {
                'config': 'configs/emergency_vehicle.sumocfg',
                'description': 'Emergency vehicle priority testing (1 hour)',
                'duration': 3600
            }
        }
    
    def _find_sumo_path(self):
        """Find SUMO installation path"""
        possible_paths = [
            'sumo',
            'sumo-gui',
            '/usr/bin/sumo',
            '/usr/bin/sumo-gui',
            '/usr/local/bin/sumo',
            '/usr/local/bin/sumo-gui',
            'C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo.exe',
            'C:\\Program Files (x86)\\Eclipse\\Sumo\\bin\\sumo-gui.exe',
            'C:\\Program Files\\Eclipse\\Sumo\\bin\\sumo.exe',
            'C:\\Program Files\\Eclipse\\Sumo\\bin\\sumo-gui.exe'
        ]
        
        for path in possible_paths:
            try:
                result = subprocess.run([path, '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        
        return None
    
    def list_scenarios(self):
        """List available scenarios"""
        print("Available SUMO Scenarios:")
        print("=" * 50)
        for name, info in self.scenarios.items():
            print(f"{name:12} - {info['description']}")
        print()
    
    def launch_scenario(self, scenario_name, gui=True, port=8813):
        """
        Launch a specific scenario
        
        Args:
            scenario_name: Name of the scenario to launch
            gui: Whether to launch with GUI
            port: TraCI server port
        """
        if scenario_name not in self.scenarios:
            print(f"Error: Scenario '{scenario_name}' not found!")
            self.list_scenarios()
            return False
        
        if not self.sumo_path:
            print("Error: SUMO not found! Please install SUMO or specify the path.")
            return False
        
        config_path = self.scenarios[scenario_name]['config']
        if not os.path.exists(config_path):
            print(f"Error: Configuration file '{config_path}' not found!")
            return False
        
        # Prepare command
        cmd = [self.sumo_path]
        if gui:
            cmd.append('-g')
        
        cmd.extend([
            '-c', config_path,
            '--remote-port', str(port),
            '--step-length', '0.1',
            '--verbose'
        ])
        
        print(f"Launching scenario: {scenario_name}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Description: {self.scenarios[scenario_name]['description']}")
        print(f"Duration: {self.scenarios[scenario_name]['duration']} seconds")
        print(f"TraCI Port: {port}")
        print("-" * 50)
        
        try:
            # Launch SUMO
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for SUMO to start
            time.sleep(2)
            
            if process.poll() is None:
                print(f"SUMO started successfully! PID: {process.pid}")
                print("Press Ctrl+C to stop the simulation")
                
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\nStopping simulation...")
                    process.terminate()
                    process.wait()
                    print("Simulation stopped.")
            else:
                stdout, stderr = process.communicate()
                print(f"Error launching SUMO:")
                print(f"STDOUT: {stdout.decode()}")
                print(f"STDERR: {stderr.decode()}")
                return False
                
        except Exception as e:
            print(f"Error launching scenario: {e}")
            return False
        
        return True
    
    def run_batch_scenarios(self, scenarios=None, gui=False, port=8813):
        """
        Run multiple scenarios in batch mode
        
        Args:
            scenarios: List of scenario names (default: all)
            gui: Whether to launch with GUI
            port: TraCI server port
        """
        if scenarios is None:
            scenarios = list(self.scenarios.keys())
        
        print(f"Running batch scenarios: {', '.join(scenarios)}")
        print("=" * 50)
        
        for i, scenario_name in enumerate(scenarios):
            if scenario_name not in self.scenarios:
                print(f"Skipping unknown scenario: {scenario_name}")
                continue
            
            print(f"\n[{i+1}/{len(scenarios)}] Running scenario: {scenario_name}")
            
            # Use different port for each scenario
            scenario_port = port + i
            
            success = self.launch_scenario(scenario_name, gui=gui, port=scenario_port)
            if not success:
                print(f"Failed to run scenario: {scenario_name}")
                continue
            
            print(f"Completed scenario: {scenario_name}")
            
            # Wait between scenarios
            if i < len(scenarios) - 1:
                print("Waiting 5 seconds before next scenario...")
                time.sleep(5)
        
        print("\nBatch execution completed!")
    
    def validate_scenarios(self):
        """Validate all scenario configurations"""
        print("Validating SUMO scenarios...")
        print("=" * 50)
        
        all_valid = True
        
        for name, info in self.scenarios.items():
            config_path = info['config']
            print(f"Validating {name}: {config_path}")
            
            if not os.path.exists(config_path):
                print(f"  ❌ Configuration file not found!")
                all_valid = False
                continue
            
            # Check if SUMO can parse the configuration
            try:
                cmd = [self.sumo_path, '-c', config_path, '--check-config']
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    print(f"  ✅ Configuration valid")
                else:
                    print(f"  ❌ Configuration invalid:")
                    print(f"     {result.stderr}")
                    all_valid = False
                    
            except Exception as e:
                print(f"  ❌ Error validating: {e}")
                all_valid = False
        
        if all_valid:
            print("\n✅ All scenarios are valid!")
        else:
            print("\n❌ Some scenarios have issues!")
        
        return all_valid

def main():
    parser = argparse.ArgumentParser(description='SUMO Scenario Launcher')
    parser.add_argument('scenario', nargs='?', help='Scenario name to launch')
    parser.add_argument('--list', action='store_true', help='List available scenarios')
    parser.add_argument('--batch', action='store_true', help='Run all scenarios in batch')
    parser.add_argument('--no-gui', action='store_true', help='Run without GUI')
    parser.add_argument('--port', type=int, default=8813, help='TraCI server port')
    parser.add_argument('--validate', action='store_true', help='Validate all scenarios')
    parser.add_argument('--sumo-path', help='Path to SUMO installation')
    
    args = parser.parse_args()
    
    launcher = SUMOScenarioLauncher(sumo_path=args.sumo_path)
    
    if args.list:
        launcher.list_scenarios()
    elif args.validate:
        launcher.validate_scenarios()
    elif args.batch:
        launcher.run_batch_scenarios(gui=not args.no_gui, port=args.port)
    elif args.scenario:
        launcher.launch_scenario(args.scenario, gui=not args.no_gui, port=args.port)
    else:
        print("SUMO Scenario Launcher")
        print("=" * 50)
        print("Use --help for usage information")
        print()
        launcher.list_scenarios()

if __name__ == "__main__":
    main()
