#!/usr/bin/env python3
"""
Demo Launcher for SIH Presentation

This script provides an easy-to-use interface for launching demo scenarios
during the Smart India Hackathon presentation.

Author: Smart Traffic Management System Team
Date: 2025
"""

import os
import sys
import subprocess
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/demo_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DemoLauncher:
    """Demo launcher for SIH presentation"""
    
    def __init__(self):
        """Initialize the demo launcher"""
        self.demo_configs = {
            'baseline': {
                'name': 'Baseline Traffic Control',
                'config': 'sumo/configs/demo/demo_baseline.sumocfg',
                'duration': 300,
                'description': 'Traditional fixed-timing traffic signal control'
            },
            'ml_optimized': {
                'name': 'ML-Optimized Traffic Control',
                'config': 'sumo/configs/demo/demo_ml_optimized.sumocfg',
                'duration': 300,
                'description': 'Machine learning-based traffic signal optimization'
            },
            'rush_hour': {
                'name': 'Rush Hour Scenario',
                'config': 'sumo/configs/demo/demo_rush_hour.sumocfg',
                'duration': 600,
                'description': 'High traffic volume scenario with ML optimization'
            },
            'emergency': {
                'name': 'Emergency Vehicle Priority',
                'config': 'sumo/configs/demo/demo_emergency.sumocfg',
                'duration': 180,
                'description': 'Emergency vehicle priority with ML optimization'
            }
        }
        
        self.current_demo = None
        self.demo_process = None
        self.metrics = {
            'demos_run': 0,
            'total_duration': 0,
            'start_time': None
        }
        
        # Create necessary directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results/demo', exist_ok=True)
        
        logger.info("Demo Launcher initialized")
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available"""
        logger.info("Checking dependencies...")
        
        # Check SUMO
        try:
            result = subprocess.run(['sumo', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✅ SUMO found: {result.stdout.strip()}")
            else:
                logger.error("❌ SUMO not found or not working")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("❌ SUMO not found. Please install SUMO and set SUMO_HOME")
            return False
        
        # Check Python packages
        required_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn']
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"✅ {package} found")
            except ImportError:
                logger.warning(f"⚠️ {package} not found, some features may not work")
        
        return True
    
    def list_demos(self) -> None:
        """List available demo scenarios"""
        print("\n" + "="*60)
        print("AVAILABLE DEMO SCENARIOS")
        print("="*60)
        
        for key, demo in self.demo_configs.items():
            print(f"\n{key.upper()}:")
            print(f"  Name: {demo['name']}")
            print(f"  Duration: {demo['duration']} seconds")
            print(f"  Description: {demo['description']}")
            print(f"  Config: {demo['config']}")
        
        print("\n" + "="*60)
    
    def run_demo(self, demo_key: str, headless: bool = False) -> bool:
        """
        Run a specific demo scenario
        
        Args:
            demo_key: Key of the demo to run
            headless: Run in headless mode (no GUI)
            
        Returns:
            True if demo ran successfully, False otherwise
        """
        if demo_key not in self.demo_configs:
            logger.error(f"Demo '{demo_key}' not found")
            return False
        
        demo = self.demo_configs[demo_key]
        config_file = demo['config']
        
        # Check if config file exists
        if not os.path.exists(config_file):
            logger.error(f"Config file not found: {config_file}")
            return False
        
        logger.info(f"Starting demo: {demo['name']}")
        logger.info(f"Config: {config_file}")
        logger.info(f"Duration: {demo['duration']} seconds")
        
        # Prepare SUMO command
        cmd = ['sumo-gui' if not headless else 'sumo', '-c', config_file]
        
        if headless:
            cmd.extend(['--no-step-log', '--duration-log.statistics'])
        
        try:
            # Start demo
            self.demo_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.current_demo = demo_key
            self.metrics['start_time'] = time.time()
            
            logger.info(f"Demo started successfully (PID: {self.demo_process.pid})")
            
            # Wait for demo to complete
            if not headless:
                logger.info("Demo is running in GUI mode. Press Ctrl+C to stop early.")
                try:
                    self.demo_process.wait(timeout=demo['duration'])
                except subprocess.TimeoutExpired:
                    logger.info("Demo duration reached, stopping...")
                    self.stop_demo()
            else:
                # For headless mode, wait for completion
                self.demo_process.wait(timeout=demo['duration'])
            
            # Update metrics
            self.metrics['demos_run'] += 1
            if self.metrics['start_time']:
                self.metrics['total_duration'] += time.time() - self.metrics['start_time']
            
            logger.info(f"Demo '{demo['name']}' completed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            logger.warning("Demo timed out")
            self.stop_demo()
            return False
        except KeyboardInterrupt:
            logger.info("Demo interrupted by user")
            self.stop_demo()
            return False
        except Exception as e:
            logger.error(f"Error running demo: {e}")
            return False
    
    def stop_demo(self) -> None:
        """Stop the current demo"""
        if self.demo_process:
            logger.info("Stopping demo...")
            self.demo_process.terminate()
            try:
                self.demo_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.demo_process.kill()
            finally:
                self.demo_process = None
                self.current_demo = None
    
    def run_demo_sequence(self, sequence: List[str], headless: bool = False) -> None:
        """
        Run a sequence of demos
        
        Args:
            sequence: List of demo keys to run
            headless: Run in headless mode
        """
        logger.info(f"Starting demo sequence: {sequence}")
        
        for i, demo_key in enumerate(sequence):
            if demo_key not in self.demo_configs:
                logger.error(f"Demo '{demo_key}' not found, skipping...")
                continue
            
            print(f"\n{'='*60}")
            print(f"DEMO {i+1}/{len(sequence)}: {self.demo_configs[demo_key]['name']}")
            print(f"{'='*60}")
            
            success = self.run_demo(demo_key, headless)
            
            if success:
                logger.info(f"Demo {i+1} completed successfully")
            else:
                logger.error(f"Demo {i+1} failed")
            
            # Pause between demos
            if i < len(sequence) - 1:
                print("\nPress Enter to continue to next demo...")
                input()
    
    def generate_demo_report(self) -> None:
        """Generate a demo execution report"""
        report_file = f"results/demo/demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Demo Execution Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Demo Statistics\n\n")
            f.write(f"- **Demos Run:** {self.metrics['demos_run']}\n")
            f.write(f"- **Total Duration:** {self.metrics['total_duration']:.2f} seconds\n")
            f.write(f"- **Average Duration:** {self.metrics['total_duration']/max(self.metrics['demos_run'], 1):.2f} seconds per demo\n\n")
            
            f.write("## Available Demos\n\n")
            for key, demo in self.demo_configs.items():
                f.write(f"### {demo['name']}\n")
                f.write(f"- **Key:** {key}\n")
                f.write(f"- **Duration:** {demo['duration']} seconds\n")
                f.write(f"- **Description:** {demo['description']}\n")
                f.write(f"- **Config:** {demo['config']}\n\n")
            
            f.write("## Usage Instructions\n\n")
            f.write("```bash\n")
            f.write("# Run individual demo\n")
            f.write("python scripts/demo_launcher.py --demo baseline\n")
            f.write("python scripts/demo_launcher.py --demo ml_optimized\n")
            f.write("\n# Run demo sequence\n")
            f.write("python scripts/demo_launcher.py --sequence baseline,ml_optimized,rush_hour\n")
            f.write("\n# Run in headless mode\n")
            f.write("python scripts/demo_launcher.py --demo baseline --headless\n")
            f.write("```\n")
        
        logger.info(f"Demo report generated: {report_file}")
    
    def interactive_mode(self) -> None:
        """Run in interactive mode for demo selection"""
        print("\n" + "="*60)
        print("SMART TRAFFIC MANAGEMENT SYSTEM - DEMO LAUNCHER")
        print("="*60)
        
        while True:
            print("\nSelect an option:")
            print("1. List available demos")
            print("2. Run single demo")
            print("3. Run demo sequence")
            print("4. Run all demos")
            print("5. Generate demo report")
            print("6. Check dependencies")
            print("7. Exit")
            
            choice = input("\nEnter your choice (1-7): ").strip()
            
            if choice == '1':
                self.list_demos()
            elif choice == '2':
                self.list_demos()
                demo_key = input("\nEnter demo key: ").strip()
                headless = input("Run in headless mode? (y/n): ").strip().lower() == 'y'
                self.run_demo(demo_key, headless)
            elif choice == '3':
                self.list_demos()
                sequence_input = input("\nEnter demo keys separated by commas: ").strip()
                sequence = [key.strip() for key in sequence_input.split(',')]
                headless = input("Run in headless mode? (y/n): ").strip().lower() == 'y'
                self.run_demo_sequence(sequence, headless)
            elif choice == '4':
                all_demos = list(self.demo_configs.keys())
                headless = input("Run in headless mode? (y/n): ").strip().lower() == 'y'
                self.run_demo_sequence(all_demos, headless)
            elif choice == '5':
                self.generate_demo_report()
            elif choice == '6':
                self.check_dependencies()
            elif choice == '7':
                print("Exiting demo launcher...")
                break
            else:
                print("Invalid choice. Please try again.")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo Launcher for SIH Presentation')
    parser.add_argument('--demo', help='Run specific demo')
    parser.add_argument('--sequence', help='Run demo sequence (comma-separated)')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode')
    parser.add_argument('--list', action='store_true', help='List available demos')
    parser.add_argument('--check', action='store_true', help='Check dependencies')
    
    args = parser.parse_args()
    
    # Create demo launcher
    launcher = DemoLauncher()
    
    # Check dependencies first
    if not launcher.check_dependencies():
        logger.error("Dependency check failed. Please install required software.")
        return False
    
    # Handle command line arguments
    if args.list:
        launcher.list_demos()
    elif args.check:
        launcher.check_dependencies()
    elif args.demo:
        launcher.run_demo(args.demo, args.headless)
    elif args.sequence:
        sequence = [key.strip() for key in args.sequence.split(',')]
        launcher.run_demo_sequence(sequence, args.headless)
    elif args.interactive:
        launcher.interactive_mode()
    else:
        # Default to interactive mode
        launcher.interactive_mode()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
