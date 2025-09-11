#!/usr/bin/env python3
"""
Smart Traffic Management System - Installation Script
Automatically installs all dependencies and sets up the system
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing Python packages"):
        return False
    
    return True

def install_node_dependencies():
    """Install Node.js dependencies for frontend"""
    print("📦 Installing Node.js dependencies...")
    
    # Check if Node.js is installed
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js not found. Please install Node.js 16+ from https://nodejs.org")
        return False
    
    # Install frontend dependencies
    frontend_dir = Path("src/frontend/smart-traffic-ui")
    if frontend_dir.exists():
        os.chdir(frontend_dir)
        if not run_command("npm install", "Installing frontend dependencies"):
            os.chdir("../..")
            return False
        os.chdir("../..")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "data/videos",
        "src/simulation/results",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def download_yolo_model():
    """Download YOLO model (will happen automatically on first run)"""
    print("🤖 YOLO model will be downloaded automatically on first run")
    return True

def create_sample_video():
    """Create a sample video file for testing"""
    print("🎬 Creating sample video for testing...")
    
    # This would create a simple test video, but for now we'll just note it
    video_dir = Path("data/videos")
    if not list(video_dir.glob("*.mp4")):
        print("⚠️  No video files found in data/videos/")
        print("   Please add some traffic video files for testing")
        print("   Supported formats: .mp4, .avi, .mov")
    else:
        print(f"✅ Found {len(list(video_dir.glob('*.mp4')))} video files")
    
    return True

def test_installation():
    """Test if installation was successful"""
    print("🧪 Testing installation...")
    
    # Test Python imports
    test_imports = [
        "fastapi", "uvicorn", "cv2", "ultralytics", 
        "requests", "numpy", "pandas"
    ]
    
    failed_imports = []
    for module in test_imports:
        try:
            __import__(module.replace('-', '_'))
        except ImportError:
            failed_imports.append(module)
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All Python modules imported successfully")
    return True

def main():
    """Main installation function"""
    print("🚦 Smart Traffic Management System - Installation")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("❌ Python dependency installation failed")
        return False
    
    # Install Node.js dependencies
    if not install_node_dependencies():
        print("❌ Node.js dependency installation failed")
        return False
    
    # Download YOLO model info
    download_yolo_model()
    
    # Check for video files
    create_sample_video()
    
    # Test installation
    if not test_installation():
        print("❌ Installation test failed")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Installation completed successfully!")
    print("=" * 60)
    print("🚀 To start the system, run:")
    print("   python run_system.py")
    print("\n📚 System URLs (when running):")
    print("   🎨 Dashboard: http://localhost:5173")
    print("   📡 Backend API: http://localhost:8000")
    print("   📚 API Docs: http://localhost:8000/docs")
    print("\n📁 Project structure:")
    print("   src/backend/          - FastAPI server")
    print("   src/frontend/         - React dashboard")
    print("   src/computer_vision/  - Vehicle detection")
    print("   src/ml_engine/        - AI optimization")
    print("   src/simulation/       - Traffic simulation")
    print("\n🎯 Ready to optimize traffic! 🚦")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
