import uvicorn
import os
import sys

def start_server():
    print("🚀 Starting Smart Traffic Management Backend Server")
    print("=" * 50)
    print("📍 Server URL: http://localhost:8000")
    print("📍 API Docs: http://localhost:8000/docs")
    print("📍 ReDoc: http://localhost:8000/redoc")
    print("=" * 50)
    
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")

if __name__ == "__main__":
    start_server()
