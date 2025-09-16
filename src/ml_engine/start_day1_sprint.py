#!/usr/bin/env python3
"""
ML Engineer Day 1 Sprint Quick Start
Start all ML components and run Day 1 objectives
"""

import asyncio
import subprocess
import time
import sys
import os
from pathlib import Path


def print_banner():
    """Print startup banner"""
    print("🚀 ML Engineer Day 1 Sprint - Quick Start")
    print("=" * 60)
    print("Starting ML optimization system and running Day 1 objectives")
    print("=" * 60)


def check_dependencies():
    """Check if all dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'numpy', 'pandas', 'requests',
        'asyncio', 'logging', 'json', 'time', 'datetime'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    return True


def start_backend_api():
    """Start the backend API server"""
    print("\n🚀 Starting backend API server...")
    
    try:
        # Change to backend directory
        backend_dir = Path(__file__).parent.parent / "backend"
        os.chdir(backend_dir)
        
        # Start the server in background
        process = subprocess.Popen([
            sys.executable, "main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        import requests
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend API server started successfully")
                return process
            else:
                print("❌ Backend API server failed to start")
                return None
        except:
            print("❌ Backend API server not responding")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend API: {e}")
        return None


def run_ml_optimizer():
    """Run the ML optimizer"""
    print("\n🧠 Starting ML optimizer...")
    
    try:
        # Change to ML engine directory
        ml_dir = Path(__file__).parent
        os.chdir(ml_dir)
        
        # Run the enhanced continuous optimizer
        print("  🔄 Starting 30-second optimization loop...")
        print("  📊 Monitoring intersections: junction-1, junction-2, junction-3")
        print("  ⏰ Press Ctrl+C to stop")
        
        # Import and run the optimizer
        from enhanced_continuous_optimizer import main as optimizer_main
        asyncio.run(optimizer_main())
        
    except KeyboardInterrupt:
        print("\n🛑 ML optimizer stopped by user")
    except Exception as e:
        print(f"❌ Error running ML optimizer: {e}")


def run_day1_tests():
    """Run Day 1 test suite"""
    print("\n🧪 Running Day 1 test suite...")
    
    try:
        # Change to ML engine directory
        ml_dir = Path(__file__).parent
        os.chdir(ml_dir)
        
        # Run the test suite
        from run_day1_tests import main as test_main
        asyncio.run(test_main())
        
    except Exception as e:
        print(f"❌ Error running tests: {e}")


def main():
    """Main function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependencies check failed. Please install missing packages.")
        sys.exit(1)
    
    # Start backend API
    backend_process = start_backend_api()
    if not backend_process:
        print("\n❌ Failed to start backend API. Exiting.")
        sys.exit(1)
    
    try:
        # Ask user what to do
        print("\n📋 What would you like to do?")
        print("1. Run ML optimizer (30-second loop)")
        print("2. Run Day 1 test suite")
        print("3. Both (recommended)")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            run_ml_optimizer()
        elif choice == "2":
            run_day1_tests()
        elif choice == "3":
            print("\n🚀 Running both ML optimizer and tests...")
            print("  (This will run tests first, then start optimizer)")
            
            # Run tests first
            run_day1_tests()
            
            # Ask if user wants to continue with optimizer
            continue_choice = input("\nContinue with ML optimizer? (y/n): ").strip().lower()
            if continue_choice in ['y', 'yes']:
                run_ml_optimizer()
        elif choice == "4":
            print("👋 Goodbye!")
        else:
            print("❌ Invalid choice. Exiting.")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        # Clean up backend process
        if backend_process:
            print("\n🧹 Cleaning up backend process...")
            backend_process.terminate()
            backend_process.wait()
            print("✅ Backend process terminated")


if __name__ == "__main__":
    main()
