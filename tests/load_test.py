#!/usr/bin/env python3
"""
Load Testing Script for Smart Traffic Management System
Uses Locust for comprehensive load testing
"""

import time
import json
import random
from locust import HttpUser, task, between
from datetime import datetime

class TrafficManagementUser(HttpUser):
    """Simulates a user interacting with the traffic management system"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Called when a user starts"""
        self.intersection_id = f"test_intersection_{random.randint(1, 10)}"
        self.camera_id = f"test_cam_{random.randint(1, 5):03d}"
        
    @task(3)
    def check_health(self):
        """Check system health (most frequent)"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(2)
    def get_traffic_status(self):
        """Get traffic status for intersection"""
        with self.client.get(f"/api/v1/traffic/status/{self.intersection_id}", catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Traffic status failed: {response.status_code}")
    
    @task(2)
    def get_signal_status(self):
        """Get signal status for intersection"""
        with self.client.get(f"/api/v1/signals/status/{self.intersection_id}", catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Signal status failed: {response.status_code}")
    
    @task(1)
    def submit_traffic_data(self):
        """Submit traffic data (less frequent)"""
        traffic_data = {
            "intersection_id": self.intersection_id,
            "timestamp": int(time.time()),
            "vehicle_counts": {
                "car": random.randint(5, 25),
                "motorcycle": random.randint(2, 10),
                "bus": random.randint(0, 3),
                "truck": random.randint(0, 2)
            },
            "lane_occupancy": {
                "lane_1": random.uniform(0.2, 0.9),
                "lane_2": random.uniform(0.2, 0.9),
                "lane_3": random.uniform(0.2, 0.9)
            },
            "waiting_times": {
                "lane_1": random.uniform(10, 60),
                "lane_2": random.uniform(10, 60),
                "lane_3": random.uniform(10, 60)
            },
            "coordinates": {
                "lat": 20.2961 + random.uniform(-0.01, 0.01),
                "lng": 85.8245 + random.uniform(-0.01, 0.01)
            }
        }
        
        with self.client.post("/api/v1/traffic/ingest", 
                            json=traffic_data, 
                            catch_response=True) as response:
            if response.status_code in [200, 404, 422]:
                response.success()
            else:
                response.failure(f"Traffic data submission failed: {response.status_code}")
    
    @task(1)
    def submit_cv_data(self):
        """Submit computer vision data"""
        cv_data = {
            "camera_id": self.camera_id,
            "intersection_id": self.intersection_id,
            "timestamp": int(time.time()),
            "total_vehicles": random.randint(5, 30),
            "counts_by_class": {
                "car": random.randint(3, 20),
                "motorcycle": random.randint(1, 8),
                "bus": random.randint(0, 2),
                "truck": random.randint(0, 2)
            },
            "coordinates": {
                "lat": 20.2961 + random.uniform(-0.01, 0.01),
                "lng": 85.8245 + random.uniform(-0.01, 0.01)
            }
        }
        
        with self.client.post("/api/v1/cv/counts", 
                            json=cv_data, 
                            catch_response=True) as response:
            if response.status_code in [200, 404, 422]:
                response.success()
            else:
                response.failure(f"CV data submission failed: {response.status_code}")
    
    @task(1)
    def get_ml_metrics(self):
        """Get ML metrics"""
        with self.client.get("/api/v1/ml/metrics", catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"ML metrics failed: {response.status_code}")

class MLAPIUser(HttpUser):
    """Simulates ML API usage"""
    
    host = "http://localhost:8001"
    wait_time = between(2, 5)
    
    def on_start(self):
        self.intersection_id = f"test_intersection_{random.randint(1, 10)}"
    
    @task(3)
    def check_ml_health(self):
        """Check ML API health"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"ML health check failed: {response.status_code}")
    
    @task(2)
    def get_ml_metrics(self):
        """Get ML metrics"""
        with self.client.get("/api/v1/ml/metrics", catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"ML metrics failed: {response.status_code}")
    
    @task(1)
    def ml_prediction(self):
        """Test ML prediction"""
        prediction_data = {
            "intersection_id": self.intersection_id,
            "traffic_data": {
                "vehicle_counts": {
                    "car": random.randint(10, 30),
                    "motorcycle": random.randint(5, 15),
                    "bus": random.randint(0, 5)
                },
                "lane_occupancy": {
                    "lane_1": random.uniform(0.3, 0.9),
                    "lane_2": random.uniform(0.3, 0.9)
                },
                "waiting_times": {
                    "lane_1": random.uniform(20, 80),
                    "lane_2": random.uniform(20, 80)
                }
            }
        }
        
        with self.client.post("/api/v1/ml/predict", 
                            json=prediction_data, 
                            catch_response=True) as response:
            if response.status_code in [200, 404, 422]:
                response.success()
            else:
                response.failure(f"ML prediction failed: {response.status_code}")

class CVAPIUser(HttpUser):
    """Simulates Computer Vision API usage"""
    
    host = "http://localhost:5001"
    wait_time = between(1, 4)
    
    def on_start(self):
        self.intersection_id = f"test_intersection_{random.randint(1, 10)}"
        self.camera_id = f"test_cam_{random.randint(1, 5):03d}"
    
    @task(3)
    def get_cv_streams(self):
        """Get CV streams"""
        with self.client.get("/cv/streams", catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"CV streams failed: {response.status_code}")
    
    @task(2)
    def get_cv_counts(self):
        """Get CV counts"""
        with self.client.get(f"/cv/counts/{self.intersection_id}", catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"CV counts failed: {response.status_code}")
    
    @task(1)
    def submit_cv_data(self):
        """Submit CV data"""
        cv_data = {
            "camera_id": self.camera_id,
            "intersection_id": self.intersection_id,
            "timestamp": int(time.time()),
            "total_vehicles": random.randint(5, 25),
            "counts_by_class": {
                "car": random.randint(3, 18),
                "motorcycle": random.randint(1, 6),
                "bus": random.randint(0, 2),
                "truck": random.randint(0, 1)
            },
            "coordinates": {
                "lat": 20.2961 + random.uniform(-0.01, 0.01),
                "lng": 85.8245 + random.uniform(-0.01, 0.01)
            }
        }
        
        with self.client.post("/cv/counts", 
                            json=cv_data, 
                            catch_response=True) as response:
            if response.status_code in [200, 404, 422]:
                response.success()
            else:
                response.failure(f"CV data submission failed: {response.status_code}")

# Load test configuration
class LoadTestConfig:
    """Configuration for load tests"""
    
    @staticmethod
    def get_test_scenarios():
        """Get different test scenarios"""
        return {
            "light_load": {
                "users": 10,
                "spawn_rate": 2,
                "duration": "2m",
                "description": "Light load test - 10 concurrent users"
            },
            "medium_load": {
                "users": 50,
                "spawn_rate": 5,
                "duration": "5m",
                "description": "Medium load test - 50 concurrent users"
            },
            "heavy_load": {
                "users": 100,
                "spawn_rate": 10,
                "duration": "10m",
                "description": "Heavy load test - 100 concurrent users"
            },
            "stress_test": {
                "users": 200,
                "spawn_rate": 20,
                "duration": "15m",
                "description": "Stress test - 200 concurrent users"
            }
        }
    
    @staticmethod
    def run_load_test(scenario_name: str = "medium_load"):
        """Run load test with specified scenario"""
        scenarios = LoadTestConfig.get_test_scenarios()
        
        if scenario_name not in scenarios:
            print(f"❌ Unknown scenario: {scenario_name}")
            print(f"Available scenarios: {list(scenarios.keys())}")
            return False
        
        scenario = scenarios[scenario_name]
        print(f"🚀 Starting {scenario['description']}")
        print(f"   Users: {scenario['users']}")
        print(f"   Spawn Rate: {scenario['spawn_rate']}/second")
        print(f"   Duration: {scenario['duration']}")
        
        # Generate locust command
        cmd = [
            "locust",
            "-f", "tests/load_test.py",
            "--host", "http://localhost:8000",
            "--users", str(scenario['users']),
            "--spawn-rate", str(scenario['spawn_rate']),
            "--run-time", scenario['duration'],
            "--headless",
            "--html", f"load_test_report_{scenario_name}.html",
            "--csv", f"load_test_results_{scenario_name}"
        ]
        
        print(f"📊 Running command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)  # 20 min timeout
            if result.returncode == 0:
                print(f"✅ Load test completed successfully")
                print(f"📊 Report saved to: load_test_report_{scenario_name}.html")
                return True
            else:
                print(f"❌ Load test failed: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            print("⏰ Load test timed out")
            return False
        except Exception as e:
            print(f"❌ Error running load test: {e}")
            return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        scenario = sys.argv[1]
    else:
        scenario = "medium_load"
    
    print("🚦 Smart Traffic Management System - Load Testing")
    print("=" * 60)
    
    success = LoadTestConfig.run_load_test(scenario)
    sys.exit(0 if success else 1)
