#!/usr/bin/env python3
"""
Simple Smart Traffic Management System Starter
This script will start the system components one by one
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path

def run_backend():
    """Start the backend API server"""
    print("🚀 Starting Backend API Server...")
    try:
        os.chdir("src/backend")
        subprocess.run([sys.executable, "main.py"], check=True)
    except KeyboardInterrupt:
        print("Backend stopped")
    except Exception as e:
        print(f"Backend error: {e}")

def run_frontend():
    """Start the frontend dashboard"""
    print("🎨 Starting Frontend Dashboard...")
    try:
        os.chdir("src/frontend/smart-traffic-ui")
        subprocess.run(["npm", "run", "dev"], check=True)
    except KeyboardInterrupt:
        print("Frontend stopped")
    except Exception as e:
        print(f"Frontend error: {e}")

def run_computer_vision():
    """Start the computer vision module"""
    print("👁️ Starting Computer Vision Module...")
    try:
        os.chdir("src/computer_vision")
        subprocess.run([sys.executable, "vehicle_count.py"], check=True)
    except KeyboardInterrupt:
        print("Computer vision stopped")
    except Exception as e:
        print(f"Computer vision error: {e}")

def run_ml_engine():
    """Start the ML optimization engine"""
    print("🤖 Starting ML Optimization Engine...")
    try:
        os.chdir("src/ml_engine")
        subprocess.run([sys.executable, "continuous_optimizer.py"], check=True)
    except KeyboardInterrupt:
        print("ML engine stopped")
    except Exception as e:
        print(f"ML engine error: {e}")

def main():
    """Main function to start the system"""
    print("🚦 Smart Traffic Management System - SIH 2025")
    print("=" * 60)
    print("Choose what to start:")
    print("1. Backend API only")
    print("2. Frontend Dashboard only")
    print("3. Computer Vision only")
    print("4. ML Engine only")
    print("5. Backend + Frontend")
    print("6. All components (recommended)")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice == "1":
        print("\n🚀 Starting Backend API Server...")
        print("📍 Backend will run on: http://localhost:8000")
        print("📍 API Docs: http://localhost:8000/docs")
        print("Press Ctrl+C to stop")
        run_backend()
        
    elif choice == "2":
        print("\n🎨 Starting Frontend Dashboard...")
        print("📍 Frontend will run on: http://localhost:5173")
        print("Press Ctrl+C to stop")
        run_frontend()
        
    elif choice == "3":
        print("\n👁️ Starting Computer Vision Module...")
        print("Press Ctrl+C to stop")
        run_computer_vision()
        
    elif choice == "4":
        print("\n🤖 Starting ML Optimization Engine...")
        print("Press Ctrl+C to stop")
        run_ml_engine()
        
    elif choice == "5":
        print("\n🚀 Starting Backend + Frontend...")
        print("📍 Backend: http://localhost:8000")
        print("📍 Frontend: http://localhost:5173")
        print("Press Ctrl+C to stop both")
        
        # Start backend in a thread
        backend_thread = threading.Thread(target=run_backend)
        backend_thread.daemon = True
        backend_thread.start()
        
        # Wait a bit for backend to start
        time.sleep(3)
        
        # Start frontend
        run_frontend()
        
    elif choice == "6":
        print("\n🚀 Starting ALL Components...")
        print("📍 Backend: http://localhost:8000")
        print("📍 Frontend: http://localhost:5173")
        print("Press Ctrl+C to stop all")
        
        # Start all components in threads
        threads = []
        
        backend_thread = threading.Thread(target=run_backend)
        backend_thread.daemon = True
        threads.append(backend_thread)
        
        frontend_thread = threading.Thread(target=run_frontend)
        frontend_thread.daemon = True
        threads.append(frontend_thread)
        
        cv_thread = threading.Thread(target=run_computer_vision)
        cv_thread.daemon = True
        threads.append(cv_thread)
        
        ml_thread = threading.Thread(target=run_ml_engine)
        ml_thread.daemon = True
        threads.append(ml_thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
            time.sleep(2)  # Stagger the starts
        
        # Wait for all threads
        try:
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            print("\n🛑 Stopping all components...")
            
    else:
        print("Invalid choice. Starting backend only...")
        run_backend()

if __name__ == "__main__":
    main()
