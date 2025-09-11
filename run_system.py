#!/usr/bin/env python3
"""
Smart Traffic Management System - Master Runner
Combines all components and runs the complete system
"""

import subprocess
import sys
import os
import time
import signal
import threading
from pathlib import Path

class SmartTrafficSystem:
    def __init__(self):
        self.processes = {}
        self.running = False
        
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        print("🔍 Checking system dependencies...")
        
        # Check Python packages
        required_packages = [
            'fastapi', 'uvicorn', 'opencv-python', 'ultralytics', 
            'requests', 'numpy', 'pandas'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print("📦 Install with: pip install -r requirements.txt")
            return False
        
        # Check Node.js for frontend
        try:
            subprocess.run(['node', '--version'], check=True, capture_output=True)
            print("✅ Node.js found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ Node.js not found. Please install Node.js 16+ from https://nodejs.org")
            return False
        
        # Check video files
        video_dir = Path("data/videos")
        if not video_dir.exists() or not list(video_dir.glob("*.mp4")):
            print("⚠️  No video files found in data/videos/. System will use mock data.")
        
        print("✅ All dependencies check passed!")
        return True
    
    def start_backend(self):
        """Start the FastAPI backend server"""
        print("🚀 Starting Backend API Server...")
        try:
            os.chdir("src/backend")
            process = subprocess.Popen([
                sys.executable, "main.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes['backend'] = process
            print("✅ Backend server started on http://localhost:8000")
            os.chdir("../..")
            return True
        except Exception as e:
            print(f"❌ Failed to start backend: {e}")
            return False
    
    def start_frontend(self):
        """Start the React frontend dashboard"""
        print("🎨 Starting Frontend Dashboard...")
        try:
            os.chdir("src/frontend/smart-traffic-ui")
            
            # Install dependencies if needed
            if not Path("node_modules").exists():
                print("📦 Installing frontend dependencies...")
                subprocess.run(["npm", "install"], check=True)
            
            process = subprocess.Popen([
                "npm", "run", "dev"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes['frontend'] = process
            print("✅ Frontend dashboard started on http://localhost:5173")
            os.chdir("../../..")
            return True
        except Exception as e:
            print(f"❌ Failed to start frontend: {e}")
            return False
    
    def start_computer_vision(self):
        """Start the computer vision module"""
        print("👁️  Starting Computer Vision Module...")
        try:
            os.chdir("src/computer_vision")
            process = subprocess.Popen([
                sys.executable, "vehicle_count.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes['computer_vision'] = process
            print("✅ Computer vision module started")
            os.chdir("../..")
            return True
        except Exception as e:
            print(f"❌ Failed to start computer vision: {e}")
            return False
    
    def start_ml_engine(self):
        """Start the ML optimization engine"""
        print("🤖 Starting ML Optimization Engine...")
        try:
            os.chdir("src/ml_engine")
            process = subprocess.Popen([
                sys.executable, "continuous_optimizer.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes['ml_engine'] = process
            print("✅ ML engine started")
            os.chdir("../..")
            return True
        except Exception as e:
            print(f"❌ Failed to start ML engine: {e}")
            return False
    
    def start_simulation(self):
        """Start the SUMO simulation (optional)"""
        print("🚗 Starting Traffic Simulation...")
        try:
            os.chdir("src/simulation")
            process = subprocess.Popen([
                sys.executable, "traffic_simulator_mock.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.processes['simulation'] = process
            print("✅ Simulation started")
            os.chdir("../..")
            return True
        except Exception as e:
            print(f"❌ Failed to start simulation: {e}")
            return False
    
    def start_all(self):
        """Start all system components"""
        print("🚦 Smart Traffic Management System - Starting All Components")
        print("=" * 70)
        
        if not self.check_dependencies():
            return False
        
        self.running = True
        
        # Start components in order
        components = [
            ("Backend API", self.start_backend),
            ("Frontend Dashboard", self.start_frontend),
            ("Computer Vision", self.start_computer_vision),
            ("ML Engine", self.start_ml_engine),
            ("Simulation", self.start_simulation)
        ]
        
        started_components = []
        
        for name, start_func in components:
            print(f"\n🔄 Starting {name}...")
            if start_func():
                started_components.append(name)
                time.sleep(2)  # Give each component time to start
            else:
                print(f"⚠️  {name} failed to start, continuing with other components...")
        
        print(f"\n✅ Successfully started {len(started_components)} components:")
        for component in started_components:
            print(f"   ✓ {component}")
        
        if started_components:
            print(f"\n🌐 System URLs:")
            if "Backend API" in started_components:
                print(f"   📡 Backend API: http://localhost:8000")
                print(f"   📚 API Docs: http://localhost:8000/docs")
            if "Frontend Dashboard" in started_components:
                print(f"   🎨 Dashboard: http://localhost:5173")
            
            print(f"\n🎯 System is running! Press Ctrl+C to stop all components.")
            return True
        else:
            print("❌ No components started successfully")
            return False
    
    def stop_all(self):
        """Stop all running components"""
        print("\n🛑 Stopping all components...")
        self.running = False
        
        for name, process in self.processes.items():
            try:
                process.terminate()
                process.wait(timeout=5)
                print(f"✅ {name} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"🔪 {name} force stopped")
            except Exception as e:
                print(f"⚠️  Error stopping {name}: {e}")
        
        self.processes.clear()
        print("🏁 All components stopped")
    
    def monitor_system(self):
        """Monitor system health"""
        while self.running:
            time.sleep(10)
            
            # Check if processes are still running
            for name, process in list(self.processes.items()):
                if process.poll() is not None:
                    print(f"⚠️  {name} has stopped unexpectedly")
                    del self.processes[name]
            
            if not self.processes:
                print("❌ All components have stopped")
                self.running = False
                break

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Shutdown signal received...")
    system.stop_all()
    sys.exit(0)

def main():
    """Main function"""
    global system
    system = SmartTrafficSystem()
    
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🚦 Smart Traffic Management System - SIH 2025")
    print("=" * 50)
    print("Choose startup mode:")
    print("1. Start all components (recommended)")
    print("2. Start backend only")
    print("3. Start frontend only")
    print("4. Start computer vision only")
    print("5. Start ML engine only")
    print("6. Check dependencies only")
    
    try:
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            if system.start_all():
                system.monitor_system()
        elif choice == "2":
            if system.start_backend():
                print("Backend running. Press Ctrl+C to stop.")
                system.monitor_system()
        elif choice == "3":
            if system.start_frontend():
                print("Frontend running. Press Ctrl+C to stop.")
                system.monitor_system()
        elif choice == "4":
            if system.start_computer_vision():
                print("Computer vision running. Press Ctrl+C to stop.")
                system.monitor_system()
        elif choice == "5":
            if system.start_ml_engine():
                print("ML engine running. Press Ctrl+C to stop.")
                system.monitor_system()
        elif choice == "6":
            system.check_dependencies()
        else:
            print("Invalid choice. Starting all components...")
            if system.start_all():
                system.monitor_system()
                
    except KeyboardInterrupt:
        system.stop_all()
    except Exception as e:
        print(f"❌ System error: {e}")
        system.stop_all()

if __name__ == "__main__":
    main()
