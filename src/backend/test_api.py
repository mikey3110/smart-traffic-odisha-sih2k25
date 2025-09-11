import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_all_endpoints():
    print("🧪 Testing Smart Traffic Management API")
    print("=" * 50)
    
    # Test root endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"GET / → Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    print("\n" + "-" * 30)
    
    # Test traffic data ingestion
    traffic_data = {
        "intersection_id": "junction-test",
        "timestamp": int(datetime.now().timestamp()),
        "lane_counts": {"north_lane": 15, "south_lane": 10, "east_lane": 12, "west_lane": 8},
        "avg_speed": 22.5,
        "weather_condition": "clear"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/traffic/ingest", json=traffic_data)
        print(f"POST /traffic/ingest → Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Traffic ingest failed: {e}")
    
    print("\n" + "-" * 30)
    
    # Test traffic status retrieval
    try:
        response = requests.get(f"{BASE_URL}/traffic/status/junction-test")
        print(f"GET /traffic/status/junction-test → Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Traffic status failed: {e}")
    
    print("\n" + "-" * 30)
    
    # Test signal optimization
    optimization_data = {
        "intersection_id": "junction-test",
        "optimized_timings": {"north_lane": 45, "south_lane": 35, "east_lane": 40, "west_lane": 30},
        "confidence_score": 0.92,
        "expected_improvement": 18.5
    }
    
    try:
        response = requests.put(f"{BASE_URL}/signal/optimize/junction-test", json=optimization_data)
        print(f"PUT /signal/optimize/junction-test → Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Signal optimization failed: {e}")
    
    print("\n" + "-" * 30)
    
    # Test signal status retrieval
    try:
        response = requests.get(f"{BASE_URL}/signal/status/junction-test")
        print(f"GET /signal/status/junction-test → Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Signal status failed: {e}")
    
    print("\n" + "-" * 30)
    
    # Test health check
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"GET /health → Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    print("\n" + "-" * 30)
    
    # Test intersections list
    try:
        response = requests.get(f"{BASE_URL}/intersections")
        print(f"GET /intersections → Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Intersections list failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ API Testing Complete!")

if __name__ == "__main__":
    print("⏰ Waiting 3 seconds for server to start...")
    time.sleep(3)
    test_all_endpoints()
