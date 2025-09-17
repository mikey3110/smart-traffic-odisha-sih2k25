#!/usr/bin/env python3
"""
Simple test script to check if all services are working
"""

import requests
import time
import subprocess
import webbrowser
from pathlib import Path

def test_service(name, url, timeout=5):
    """Test if a service is responding"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            print(f"✅ {name}: Working (Status: {response.status_code})")
            return True
        else:
            print(f"⚠️  {name}: Responding but status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {name}: Not responding ({e})")
        return False

def main():
    """Test all services"""
    print("🚦 Smart Traffic Management System - Service Test")
    print("=" * 60)
    print()
    
    # Wait a bit for services to start
    print("⏳ Waiting for services to start...")
    time.sleep(10)
    
    # Test services
    services = [
        ("Backend API", "http://localhost:8000/health"),
        ("ML API", "http://localhost:8001/health"),
        ("CV Service", "http://localhost:5001/cv/streams"),
        ("Frontend", "http://localhost:3000")
    ]
    
    working_services = 0
    total_services = len(services)
    
    for name, url in services:
        if test_service(name, url):
            working_services += 1
        time.sleep(1)
    
    print()
    print("=" * 60)
    print(f"📊 Results: {working_services}/{total_services} services working")
    
    if working_services == total_services:
        print("🎉 All services are working! Your system is ready!")
        print()
        print("🌐 You can now access:")
        print("   • Frontend Dashboard: http://localhost:3000")
        print("   • API Documentation: http://localhost:8000/docs")
        print("   • SUMO-GUI: Check the opened windows")
        print()
        
        # Open the frontend
        try:
            webbrowser.open("http://localhost:3000")
            print("✅ Frontend opened in your browser")
        except:
            print("⚠️  Could not open browser automatically")
            
    else:
        print("⚠️  Some services are not working. Please check the error messages above.")
        print("💡 Try running the services manually or check the logs.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
