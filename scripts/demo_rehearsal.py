#!/usr/bin/env python3
"""
Demo Rehearsal Script for SIH Presentation

This script helps rehearse the demo scenarios before the actual presentation.

Author: Smart Traffic Management System Team
Date: 2025
"""

import os
import sys
import subprocess
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DemoRehearsal:
    """Demo rehearsal system for SIH presentation"""
    
    def __init__(self):
        """Initialize the demo rehearsal system"""
        self.rehearsal_log = []
        self.start_time = None
        self.current_demo = None
        
        # Demo scenarios for rehearsal
        self.demos = {
            'baseline': {
                'name': 'Baseline Traffic Control',
                'config': 'sumo/configs/demo/demo_baseline.sumocfg',
                'duration': 60,  # Shorter for rehearsal
                'key_points': [
                    "Show fixed timing traffic lights",
                    "Highlight queue buildup",
                    "Point out inefficiencies",
                    "Mention wait times"
                ]
            },
            'ml_optimized': {
                'name': 'ML-Optimized Traffic Control',
                'config': 'sumo/configs/demo/demo_ml_optimized.sumocfg',
                'duration': 60,
                'key_points': [
                    "Show adaptive control",
                    "Highlight reduced wait times",
                    "Point out better traffic flow",
                    "Mention 22% improvement"
                ]
            },
            'rush_hour': {
                'name': 'Rush Hour Traffic',
                'config': 'sumo/configs/demo/demo_rush_hour.sumocfg',
                'duration': 120,
                'key_points': [
                    "Show high traffic volume",
                    "Highlight queue management",
                    "Point out dynamic timing",
                    "Mention performance under load"
                ]
            },
            'emergency': {
                'name': 'Emergency Vehicle Priority',
                'config': 'sumo/configs/demo/demo_emergency.sumocfg',
                'duration': 60,
                'key_points': [
                    "Show emergency detection",
                    "Highlight priority override",
                    "Point out quick response",
                    "Mention safety features"
                ]
            }
        }
        
        logger.info("Demo Rehearsal initialized")
    
    def check_system(self):
        """Check if the system is ready for rehearsal"""
        logger.info("Checking system readiness...")
        
        checks = {
            'SUMO': self._check_sumo(),
            'Python': self._check_python(),
            'Files': self._check_files(),
            'Permissions': self._check_permissions()
        }
        
        all_good = all(checks.values())
        
        print("\n" + "="*50)
        print("SYSTEM READINESS CHECK")
        print("="*50)
        
        for check, status in checks.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {check}")
        
        if all_good:
            print("\n✅ System is ready for rehearsal!")
        else:
            print("\n❌ System issues detected. Please fix before rehearsal.")
        
        return all_good
    
    def _check_sumo(self):
        """Check SUMO installation"""
        try:
            result = subprocess.run(['sumo', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"SUMO found: {result.stdout.strip()}")
                return True
            else:
                logger.error("SUMO not working properly")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("SUMO not found")
            return False
    
    def _check_python(self):
        """Check Python packages"""
        try:
            import pandas, numpy, matplotlib, seaborn
            logger.info("Python packages found")
            return True
        except ImportError as e:
            logger.error(f"Python package missing: {e}")
            return False
    
    def _check_files(self):
        """Check if demo files exist"""
        required_files = [
            'sumo/configs/demo/demo_baseline.sumocfg',
            'sumo/configs/demo/demo_ml_optimized.sumocfg',
            'sumo/configs/demo/demo_rush_hour.sumocfg',
            'sumo/configs/demo/demo_emergency.sumocfg',
            'sumo/networks/simple_intersection.net.xml',
            'scripts/demo_launcher.py'
        ]
        
        missing_files = []
        for file in required_files:
            if not os.path.exists(file):
                missing_files.append(file)
        
        if missing_files:
            logger.error(f"Missing files: {missing_files}")
            return False
        else:
            logger.info("All required files found")
            return True
    
    def _check_permissions(self):
        """Check file permissions"""
        try:
            # Check if we can read demo files
            with open('sumo/configs/demo/demo_baseline.sumocfg', 'r') as f:
                f.read()
            logger.info("File permissions OK")
            return True
        except Exception as e:
            logger.error(f"Permission error: {e}")
            return False
    
    def rehearse_demo(self, demo_key, practice_mode=True):
        """Rehearse a specific demo scenario"""
        if demo_key not in self.demos:
            logger.error(f"Demo '{demo_key}' not found")
            return False
        
        demo = self.demos[demo_key]
        self.current_demo = demo_key
        self.start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"REHEARSING: {demo['name']}")
        print(f"{'='*60}")
        
        # Show key points
        print("\nKey Points to Cover:")
        for i, point in enumerate(demo['key_points'], 1):
            print(f"  {i}. {point}")
        
        # Practice timing
        if practice_mode:
            print(f"\n⏱️  Practice Duration: {demo['duration']} seconds")
            print("Press Enter when ready to start...")
            input()
        
        # Run demo
        print(f"\n🚀 Starting {demo['name']}...")
        success = self._run_demo(demo)
        
        # Record rehearsal
        duration = time.time() - self.start_time
        self.rehearsal_log.append({
            'demo': demo_key,
            'success': success,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
        
        if success:
            print(f"✅ {demo['name']} rehearsal completed successfully")
        else:
            print(f"❌ {demo['name']} rehearsal failed")
        
        return success
    
    def _run_demo(self, demo):
        """Run a demo scenario"""
        try:
            cmd = ['sumo-gui', '-c', demo['config']]
            process = subprocess.Popen(cmd)
            
            # Wait for demo duration
            try:
                process.wait(timeout=demo['duration'])
            except subprocess.TimeoutExpired:
                print(f"\n⏰ Demo duration reached ({demo['duration']}s), stopping...")
                process.terminate()
                process.wait(timeout=5)
            
            return True
            
        except KeyboardInterrupt:
            print("\n⏹️  Demo stopped by user")
            process.terminate()
            return False
        except Exception as e:
            logger.error(f"Error running demo: {e}")
            return False
    
    def rehearse_sequence(self, sequence, practice_mode=True):
        """Rehearse a sequence of demos"""
        print(f"\n{'='*60}")
        print(f"REHEARSING DEMO SEQUENCE")
        print(f"{'='*60}")
        
        total_duration = 0
        successful_demos = 0
        
        for i, demo_key in enumerate(sequence, 1):
            if demo_key not in self.demos:
                print(f"❌ Demo '{demo_key}' not found, skipping...")
                continue
            
            print(f"\nDemo {i}/{len(sequence)}: {self.demos[demo_key]['name']}")
            
            if practice_mode and i > 1:
                print("Press Enter to continue to next demo...")
                input()
            
            success = self.rehearse_demo(demo_key, practice_mode)
            if success:
                successful_demos += 1
                total_duration += self.demos[demo_key]['duration']
        
        print(f"\n{'='*60}")
        print("SEQUENCE REHEARSAL COMPLETE")
        print(f"{'='*60}")
        print(f"Successful demos: {successful_demos}/{len(sequence)}")
        print(f"Total duration: {total_duration} seconds")
        
        return successful_demos == len(sequence)
    
    def generate_rehearsal_report(self):
        """Generate a rehearsal report"""
        report_file = f"results/demo/rehearsal_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write("# Demo Rehearsal Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Rehearsal Summary\n\n")
            f.write(f"- **Total Demos Rehearsed:** {len(self.rehearsal_log)}\n")
            f.write(f"- **Successful Demos:** {sum(1 for log in self.rehearsal_log if log['success'])}\n")
            f.write(f"- **Failed Demos:** {sum(1 for log in self.rehearsal_log if not log['success'])}\n")
            
            f.write("\n## Demo Details\n\n")
            for log in self.rehearsal_log:
                demo = self.demos[log['demo']]
                status = "✅ Success" if log['success'] else "❌ Failed"
                f.write(f"### {demo['name']}\n")
                f.write(f"- **Status:** {status}\n")
                f.write(f"- **Duration:** {log['duration']:.2f} seconds\n")
                f.write(f"- **Timestamp:** {log['timestamp']}\n\n")
            
            f.write("## Recommendations\n\n")
            if all(log['success'] for log in self.rehearsal_log):
                f.write("✅ All demos rehearsed successfully. You're ready for the presentation!\n")
            else:
                f.write("⚠️ Some demos failed during rehearsal. Please practice more before the presentation.\n")
            
            f.write("\n## Next Steps\n\n")
            f.write("1. Review any failed demos\n")
            f.write("2. Practice timing and key points\n")
            f.write("3. Prepare backup plans\n")
            f.write("4. Test on presentation hardware\n")
        
        logger.info(f"Rehearsal report generated: {report_file}")
    
    def interactive_rehearsal(self):
        """Run interactive rehearsal mode"""
        print("\n" + "="*60)
        print("DEMO REHEARSAL - SIH PRESENTATION")
        print("="*60)
        
        while True:
            print("\nSelect rehearsal option:")
            print("1. Check system readiness")
            print("2. Rehearse single demo")
            print("3. Rehearse demo sequence")
            print("4. Rehearse all demos")
            print("5. Generate rehearsal report")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                self.check_system()
            elif choice == '2':
                print("\nAvailable demos:")
                for key, demo in self.demos.items():
                    print(f"  {key}: {demo['name']}")
                demo_key = input("\nEnter demo key: ").strip()
                self.rehearse_demo(demo_key)
            elif choice == '3':
                print("\nAvailable demos:")
                for key, demo in self.demos.items():
                    print(f"  {key}: {demo['name']}")
                sequence_input = input("\nEnter demo keys separated by commas: ").strip()
                sequence = [key.strip() for key in sequence_input.split(',')]
                self.rehearse_sequence(sequence)
            elif choice == '4':
                all_demos = list(self.demos.keys())
                self.rehearse_sequence(all_demos)
            elif choice == '5':
                self.generate_rehearsal_report()
            elif choice == '6':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo Rehearsal for SIH Presentation')
    parser.add_argument('--demo', help='Rehearse specific demo')
    parser.add_argument('--sequence', help='Rehearse demo sequence (comma-separated)')
    parser.add_argument('--all', action='store_true', help='Rehearse all demos')
    parser.add_argument('--check', action='store_true', help='Check system readiness')
    parser.add_argument('--interactive', action='store_true', help='Run interactive rehearsal')
    
    args = parser.parse_args()
    
    # Create rehearsal system
    rehearsal = DemoRehearsal()
    
    # Handle command line arguments
    if args.check:
        rehearsal.check_system()
    elif args.demo:
        rehearsal.rehearse_demo(args.demo)
    elif args.sequence:
        sequence = [key.strip() for key in args.sequence.split(',')]
        rehearsal.rehearse_sequence(sequence)
    elif args.all:
        all_demos = list(rehearsal.demos.keys())
        rehearsal.rehearse_sequence(all_demos)
    elif args.interactive:
        rehearsal.interactive_rehearsal()
    else:
        # Default to interactive mode
        rehearsal.interactive_rehearsal()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
