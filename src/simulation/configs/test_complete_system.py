import os
import subprocess

def test_sumo_system():
    """Test complete SUMO system"""
    print("🧪 Testing SUMO system...")
    
    # Check files exist
    required_files = [
        'configs/intersection.net.xml',
        'configs/traffic_lights.add.xml', 
        'configs/routes.rou.xml',
        'configs/simulation.sumocfg',
        'traffic_simulator.py',
        'demo_simulation.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All configuration files present")
    
    # Test SUMO installation
    try:
        result = subprocess.run(['sumo', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ SUMO installation working")
        else:
            print("❌ SUMO not working")
            return False
    except:
        print("❌ SUMO not installed or not in PATH")
        return False
    
    # Test Python imports
    try:
        import traci
        import sumolib
        import numpy as np
        import pandas as pd
        print("✅ Python libraries working")
    except ImportError as e:
        print(f"❌ Missing Python library: {e}")
        return False
    
    print("✅ SUMO system ready for demonstration!")
    return True

if __name__ == "__main__":
    test_sumo_system()
