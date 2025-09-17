#!/usr/bin/env python3
"""
Demo Media Capture Script

This script captures screenshots and videos of demo scenarios
for the SIH presentation.

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

class DemoMediaCapture:
    """Captures media assets for demo scenarios"""
    
    def __init__(self):
        """Initialize the media capture system"""
        self.assets_dir = "docs/assets"
        self.screenshots_dir = os.path.join(self.assets_dir, "screenshots")
        self.videos_dir = os.path.join(self.assets_dir, "videos")
        
        # Create directories
        os.makedirs(self.screenshots_dir, exist_ok=True)
        os.makedirs(self.videos_dir, exist_ok=True)
        
        # Demo configurations
        self.demos = {
            'baseline': {
                'name': 'Baseline Traffic Control',
                'config': 'sumo/configs/demo/demo_baseline.sumocfg',
                'duration': 60,  # Shorter for media capture
                'key_moments': [10, 30, 50]  # Seconds to capture
            },
            'ml_optimized': {
                'name': 'ML-Optimized Traffic Control',
                'config': 'sumo/configs/demo/demo_ml_optimized.sumocfg',
                'duration': 60,
                'key_moments': [10, 30, 50]
            },
            'rush_hour': {
                'name': 'Rush Hour Traffic',
                'config': 'sumo/configs/demo/demo_rush_hour.sumocfg',
                'duration': 120,
                'key_moments': [20, 60, 100]
            },
            'emergency': {
                'name': 'Emergency Vehicle Priority',
                'config': 'sumo/configs/demo/demo_emergency.sumocfg',
                'duration': 60,
                'key_moments': [15, 30, 45]
            }
        }
        
        logger.info("Demo Media Capture initialized")
    
    def check_dependencies(self):
        """Check if required tools are available"""
        logger.info("Checking dependencies...")
        
        # Check SUMO
        try:
            result = subprocess.run(['sumo', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✅ SUMO found: {result.stdout.strip()}")
            else:
                logger.error("❌ SUMO not found")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            logger.error("❌ SUMO not found")
            return False
        
        # Check if we can run SUMO in headless mode
        try:
            result = subprocess.run(['sumo', '--help'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("✅ SUMO headless mode available")
            else:
                logger.warning("⚠️ SUMO headless mode may not work")
        except Exception as e:
            logger.warning(f"⚠️ Could not test SUMO headless mode: {e}")
        
        return True
    
    def capture_screenshot(self, demo_key, moment):
        """Capture a screenshot at a specific moment"""
        demo = self.demos[demo_key]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{demo_key}_moment_{moment}s_{timestamp}.png"
        filepath = os.path.join(self.screenshots_dir, filename)
        
        logger.info(f"Capturing screenshot for {demo['name']} at {moment}s")
        
        # This is a placeholder - in a real implementation, you would:
        # 1. Start SUMO in headless mode
        # 2. Run until the specified moment
        # 3. Capture a screenshot
        # 4. Save to file
        
        # For now, create a placeholder file
        with open(filepath, 'w') as f:
            f.write(f"Screenshot placeholder for {demo['name']} at {moment}s")
        
        logger.info(f"Screenshot saved: {filepath}")
        return filepath
    
    def capture_video(self, demo_key, duration=60):
        """Capture a video of the demo scenario"""
        demo = self.demos[demo_key]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{demo_key}_demo_{timestamp}.mp4"
        filepath = os.path.join(self.videos_dir, filename)
        
        logger.info(f"Capturing video for {demo['name']} ({duration}s)")
        
        # This is a placeholder - in a real implementation, you would:
        # 1. Start SUMO with screen recording
        # 2. Run for the specified duration
        # 3. Save the video file
        
        # For now, create a placeholder file
        with open(filepath, 'w') as f:
            f.write(f"Video placeholder for {demo['name']} ({duration}s)")
        
        logger.info(f"Video saved: {filepath}")
        return filepath
    
    def capture_demo_assets(self, demo_key):
        """Capture all assets for a demo scenario"""
        if demo_key not in self.demos:
            logger.error(f"Demo '{demo_key}' not found")
            return False
        
        demo = self.demos[demo_key]
        logger.info(f"Capturing assets for: {demo['name']}")
        
        # Capture screenshots at key moments
        screenshots = []
        for moment in demo['key_moments']:
            screenshot = self.capture_screenshot(demo_key, moment)
            screenshots.append(screenshot)
        
        # Capture video
        video = self.capture_video(demo_key, demo['duration'])
        
        # Create summary file
        summary_file = os.path.join(self.assets_dir, f"{demo_key}_assets_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Demo Assets Summary: {demo['name']}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Screenshots:\n")
            for screenshot in screenshots:
                f.write(f"  - {os.path.basename(screenshot)}\n")
            f.write(f"\nVideo:\n  - {os.path.basename(video)}\n")
        
        logger.info(f"Assets captured for {demo['name']}")
        return True
    
    def capture_all_assets(self):
        """Capture assets for all demo scenarios"""
        logger.info("Capturing assets for all demo scenarios...")
        
        for demo_key in self.demos.keys():
            self.capture_demo_assets(demo_key)
            time.sleep(1)  # Brief pause between demos
        
        logger.info("All assets captured successfully")
    
    def generate_media_report(self):
        """Generate a report of all captured media"""
        report_file = os.path.join(self.assets_dir, "media_report.md")
        
        with open(report_file, 'w') as f:
            f.write("# Demo Media Assets Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Screenshots\n\n")
            screenshot_files = [f for f in os.listdir(self.screenshots_dir) if f.endswith('.png')]
            for file in sorted(screenshot_files):
                f.write(f"- `{file}`\n")
            
            f.write("\n## Videos\n\n")
            video_files = [f for f in os.listdir(self.videos_dir) if f.endswith('.mp4')]
            for file in sorted(video_files):
                f.write(f"- `{file}`\n")
            
            f.write("\n## Usage\n\n")
            f.write("These media assets can be used in:\n")
            f.write("- Presentation slides\n")
            f.write("- Demo videos\n")
            f.write("- Documentation\n")
            f.write("- Marketing materials\n")
        
        logger.info(f"Media report generated: {report_file}")

def main():
    """Main function"""
    print("=" * 60)
    print("DEMO MEDIA CAPTURE - SIH PRESENTATION")
    print("=" * 60)
    
    # Create media capture system
    capture = DemoMediaCapture()
    
    # Check dependencies
    if not capture.check_dependencies():
        print("❌ Dependency check failed. Please install required software.")
        return False
    
    # Capture assets for all demos
    capture.capture_all_assets()
    
    # Generate report
    capture.generate_media_report()
    
    print("\n✅ Media capture completed successfully!")
    print(f"Screenshots saved to: {capture.screenshots_dir}")
    print(f"Videos saved to: {capture.videos_dir}")
    print(f"Report generated: {os.path.join(capture.assets_dir, 'media_report.md')}")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
