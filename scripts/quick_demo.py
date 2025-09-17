#!/usr/bin/env python3
"""
Quick Demo Script for SIH Presentation

This script provides a simplified interface for running demos
during the Smart India Hackathon presentation.

Author: Smart Traffic Management System Team
Date: 2025
"""

import os
import sys
import subprocess
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_sumo():
    """Check if SUMO is available"""
    try:
        result = subprocess.run(['sumo', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ SUMO found: {result.stdout.strip()}")
            return True
        else:
            print("❌ SUMO not found")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ SUMO not found. Please install SUMO and set SUMO_HOME")
        return False

def run_demo(config_file, duration=300):
    """Run a demo with the given configuration"""
    print(f"\n🚀 Starting demo: {config_file}")
    print(f"⏱️  Duration: {duration} seconds")
    print("Press Ctrl+C to stop early\n")
    
    try:
        cmd = ['sumo-gui', '-c', config_file]
        process = subprocess.Popen(cmd)
        
        # Wait for duration or until interrupted
        try:
            process.wait(timeout=duration)
        except subprocess.TimeoutExpired:
            print(f"\n⏰ Demo duration reached ({duration}s), stopping...")
            process.terminate()
            process.wait(timeout=5)
        
        print("✅ Demo completed successfully")
        return True
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo stopped by user")
        process.terminate()
        return False
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("SMART TRAFFIC MANAGEMENT SYSTEM - QUICK DEMO")
    print("=" * 60)
    
    # Check SUMO
    if not check_sumo():
        print("\n❌ SUMO not available. Please install SUMO first.")
        return False
    
    # Demo configurations
    demos = {
        '1': {
            'name': 'Baseline Traffic Control',
            'file': 'sumo/configs/demo/demo_baseline.sumocfg',
            'duration': 300
        },
        '2': {
            'name': 'ML-Optimized Traffic Control',
            'file': 'sumo/configs/demo/demo_ml_optimized.sumocfg',
            'duration': 300
        },
        '3': {
            'name': 'Rush Hour Traffic',
            'file': 'sumo/configs/demo/demo_rush_hour.sumocfg',
            'duration': 600
        },
        '4': {
            'name': 'Emergency Vehicle Priority',
            'file': 'sumo/configs/demo/demo_emergency.sumocfg',
            'duration': 180
        }
    }
    
    while True:
        print("\nSelect a demo to run:")
        print("1. Baseline Traffic Control (5 min)")
        print("2. ML-Optimized Traffic Control (5 min)")
        print("3. Rush Hour Traffic (10 min)")
        print("4. Emergency Vehicle Priority (3 min)")
        print("5. Run All Demos")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '6':
            print("👋 Goodbye!")
            break
        elif choice == '5':
            # Run all demos
            print("\n🚀 Running all demos...")
            for key, demo in demos.items():
                print(f"\n{'='*40}")
                print(f"DEMO: {demo['name']}")
                print(f"{'='*40}")
                run_demo(demo['file'], demo['duration'])
                if key != '4':  # Not the last demo
                    input("\nPress Enter to continue to next demo...")
        elif choice in demos:
            demo = demos[choice]
            print(f"\n{'='*40}")
            print(f"DEMO: {demo['name']}")
            print(f"{'='*40}")
            run_demo(demo['file'], demo['duration'])
        else:
            print("❌ Invalid choice. Please try again.")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
