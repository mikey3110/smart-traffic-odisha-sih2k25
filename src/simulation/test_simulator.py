#!/usr/bin/env python3
"""
Test script for Traffic Simulator
This will test the path fixes and run a quick simulation
"""

import os
import sys
import json
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_paths():
    """Test that the results directory can be created and files saved"""
    print("🧪 Testing Traffic Simulator Paths...")
    
    # Test results directory creation
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    
    print(f"📁 Script directory: {script_dir}")
    print(f"📁 Results directory: {results_dir}")
    
    # Create results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print("✅ Results directory created")
    else:
        print("✅ Results directory already exists")
    
    # Test file creation
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_file = os.path.join(results_dir, f"test_{timestamp}.json")
    
    test_data = {
        "test": True,
        "timestamp": timestamp,
        "message": "Path test successful",
        "results_dir": results_dir
    }
    
    try:
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        print(f"✅ Test file created: {test_file}")
        return True
    except Exception as e:
        print(f"❌ Error creating test file: {e}")
        return False

def run_mock_simulation():
    """Run a mock simulation to test the full workflow"""
    print("\n🚗 Running Mock Simulation...")
    
    try:
        from configs.traffic_simulator import TrafficSimulator
        
        # Create simulator instance
        simulator = TrafficSimulator()
        print("✅ TrafficSimulator created successfully")
        
        # Run mock simulation (without SUMO)
        print("🔄 Running baseline simulation...")
        simulator.run_baseline_simulation(duration=60, use_gui=False)
        
        print("🔄 Running optimized simulation...")
        simulator.run_optimized_simulation(duration=60, use_gui=False)
        
        print("🔄 Generating comparison...")
        simulator.generate_comparison_report()
        
        print("✅ Mock simulation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error running simulation: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🚦 SMART TRAFFIC SIMULATOR - PATH TEST")
    print("=" * 60)
    
    # Test 1: Path creation
    if not test_paths():
        print("❌ Path test failed!")
        return
    
    # Test 2: Mock simulation
    if not run_mock_simulation():
        print("❌ Simulation test failed!")
        return
    
    print("\n" + "=" * 60)
    print("🎯 ALL TESTS PASSED!")
    print("✅ The simulator is ready to use")
    print("📁 Results will be saved to: src/simulation/results/")
    print("=" * 60)

if __name__ == "__main__":
    main()
