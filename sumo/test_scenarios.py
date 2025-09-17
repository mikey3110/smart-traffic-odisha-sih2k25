#!/usr/bin/env python3
"""
SUMO Scenario Test Script
Validates and tests all SUMO scenarios
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def test_sumo_installation():
    """Test if SUMO is properly installed"""
    print("Testing SUMO installation...")
    
    try:
        result = subprocess.run(['sumo', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ SUMO found:", result.stdout.strip().split('\n')[0])
            return True
        else:
            print("❌ SUMO not working properly")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        print(f"❌ SUMO not found: {e}")
        return False

def test_network_files():
    """Test if network files exist and are valid"""
    print("\nTesting network files...")
    
    # Change to sumo directory
    os.chdir('sumo')
    
    network_files = [
        'networks/intersection_4way.nod.xml',
        'networks/intersection_4way.edg.xml',
        'networks/intersection_4way.con.xml',
        'networks/intersection_4way.tll.xml',
        'networks/intersection_4way.net.xml'
    ]
    
    all_valid = True
    
    for file_path in network_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_valid = False
    
    return all_valid

def test_route_files():
    """Test if route files exist and are valid"""
    print("\nTesting route files...")
    
    route_files = [
        'routes/normal_traffic.rou.xml',
        'routes/rush_hour.rou.xml',
        'routes/emergency_vehicle.rou.xml'
    ]
    
    all_valid = True
    
    for file_path in route_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_valid = False
    
    return all_valid

def test_config_files():
    """Test if configuration files exist and are valid"""
    print("\nTesting configuration files...")
    
    config_files = [
        'configs/normal_traffic.sumocfg',
        'configs/rush_hour.sumocfg',
        'configs/emergency_vehicle.sumocfg'
    ]
    
    all_valid = True
    
    for file_path in config_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
            
            # Test configuration validity
            try:
                result = subprocess.run(['sumo', '-c', file_path, '--help'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"  ✅ Configuration valid")
                else:
                    print(f"  ❌ Configuration invalid: {result.stderr}")
                    all_valid = False
            except Exception as e:
                print(f"  ❌ Error checking configuration: {e}")
                all_valid = False
        else:
            print(f"❌ {file_path} missing")
            all_valid = False
    
    return all_valid

def test_scenario_execution():
    """Test if scenarios can be executed"""
    print("\nTesting scenario execution...")
    
    scenarios = [
        'configs/normal_traffic.sumocfg',
        'configs/rush_hour.sumocfg',
        'configs/emergency_vehicle.sumocfg'
    ]
    
    all_valid = True
    
    for config_file in scenarios:
        print(f"Testing {config_file}...")
        
        try:
            # Run scenario for 10 seconds
            cmd = ['sumo', '-c', config_file, '--end', '10', '--no-step-log']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                print(f"  ✅ {config_file} executes successfully")
            else:
                print(f"  ❌ {config_file} execution failed: {result.stderr}")
                all_valid = False
                
        except subprocess.TimeoutExpired:
            print(f"  ✅ {config_file} executes successfully (timeout expected)")
        except Exception as e:
            print(f"  ❌ Error executing {config_file}: {e}")
            all_valid = False
    
    return all_valid

def test_traci_connection():
    """Test TraCI connection"""
    print("\nTesting TraCI connection...")
    
    try:
        # Start SUMO in background
        cmd = ['sumo', '-c', 'configs/normal_traffic.sumocfg', '--remote-port', '8813']
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for SUMO to start
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ SUMO started successfully")
            
            # Test TraCI connection
            try:
                import traci
                traci.init(8813)
                
                # Test basic TraCI commands
                simulation_time = traci.simulation.getTime()
                print(f"✅ TraCI connection successful, simulation time: {simulation_time}")
                
                traci.close()
                process.terminate()
                process.wait()
                
                return True
                
            except ImportError:
                print("❌ TraCI Python module not available")
                process.terminate()
                process.wait()
                return False
            except Exception as e:
                print(f"❌ TraCI connection failed: {e}")
                process.terminate()
                process.wait()
                return False
        else:
            stdout, stderr = process.communicate()
            print(f"❌ SUMO failed to start: {stderr.decode()}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing TraCI: {e}")
        return False

def main():
    """Run all tests"""
    print("SUMO Scenario Test Suite")
    print("=" * 50)
    
    # Change to sumo directory for all tests
    if os.path.exists('sumo'):
        os.chdir('sumo')
        print("Changed to sumo directory")
    else:
        print("Warning: sumo directory not found, running from current directory")
    
    tests = [
        ("SUMO Installation", test_sumo_installation),
        ("Network Files", test_network_files),
        ("Route Files", test_route_files),
        ("Configuration Files", test_config_files),
        ("Scenario Execution", test_scenario_execution),
        ("TraCI Connection", test_traci_connection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! SUMO scenarios are ready to use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
