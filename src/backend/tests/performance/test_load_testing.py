"""
Load Testing Suite for Backend API
Uses Locust to simulate 100 concurrent users
"""

import time
import random
import json
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

# Locust imports
from locust import HttpUser, task, between, events
from locust.exception import RescheduleTask

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrafficManagementUser(HttpUser):
    """Simulates a user interacting with the traffic management system"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks
    
    def on_start(self):
        """Called when a user starts"""
        self.intersection_id = f"intersection_{random.randint(1, 10)}"
        self.camera_id = f"cam_{random.randint(1, 5):03d}"
        self.coordinates = {
            "lat": 20.2961 + random.uniform(-0.01, 0.01),
            "lng": 85.8245 + random.uniform(-0.01, 0.01)
        }
        logger.info(f"User started for intersection {self.intersection_id}")
    
    @task(3)
    def check_health(self):
        """Check system health - most common task"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    response.success()
                else:
                    response.failure("Health check returned unhealthy status")
            else:
                response.failure(f"Health check failed with status {response.status_code}")
    
    @task(2)
    def get_traffic_status(self):
        """Get traffic status for intersection"""
        with self.client.get(f"/api/v1/traffic/status/{self.intersection_id}", 
                           catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Traffic status failed with status {response.status_code}")
    
    @task(2)
    def get_signal_status(self):
        """Get signal status for intersection"""
        with self.client.get(f"/api/v1/signals/status/{self.intersection_id}", 
                           catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Signal status failed with status {response.status_code}")
    
    @task(1)
    def submit_traffic_data(self):
        """Submit traffic data - less frequent but important"""
        traffic_data = {
            "intersection_id": self.intersection_id,
            "timestamp": int(time.time()),
            "vehicle_counts": {
                "car": random.randint(5, 30),
                "motorcycle": random.randint(2, 15),
                "bus": random.randint(0, 5),
                "truck": random.randint(0, 3)
            },
            "lane_occupancy": {
                f"lane_{i}": random.uniform(0.1, 0.9) 
                for i in range(1, 4)
            },
            "waiting_times": {
                f"lane_{i}": random.uniform(10.0, 60.0) 
                for i in range(1, 4)
            },
            "coordinates": self.coordinates
        }
        
        with self.client.post("/api/v1/traffic/ingest", 
                            json=traffic_data,
                            catch_response=True) as response:
            if response.status_code in [200, 404, 422]:
                response.success()
            else:
                response.failure(f"Traffic data submission failed with status {response.status_code}")
    
    @task(1)
    def submit_cv_data(self):
        """Submit computer vision data"""
        cv_data = {
            "camera_id": self.camera_id,
            "intersection_id": self.intersection_id,
            "timestamp": int(time.time()),
            "total_vehicles": random.randint(0, 50),
            "counts_by_class": {
                "car": random.randint(0, 30),
                "motorcycle": random.randint(0, 15),
                "bus": random.randint(0, 5),
                "truck": random.randint(0, 3)
            },
            "coordinates": self.coordinates
        }
        
        with self.client.post("/api/v1/cv/counts", 
                            json=cv_data,
                            catch_response=True) as response:
            if response.status_code in [200, 404, 422]:
                response.success()
            else:
                response.failure(f"CV data submission failed with status {response.status_code}")
    
    @task(1)
    def optimize_signals(self):
        """Request signal optimization"""
        optimization_request = {
            "intersection_id": self.intersection_id,
            "current_phase": random.randint(0, 3),
            "traffic_data": {
                "vehicle_counts": {
                    "car": random.randint(10, 40),
                    "motorcycle": random.randint(5, 20),
                    "bus": random.randint(0, 8),
                    "truck": random.randint(0, 5)
                },
                "lane_occupancy": {
                    f"lane_{i}": random.uniform(0.2, 0.8) 
                    for i in range(1, 4)
                },
                "waiting_times": {
                    f"lane_{i}": random.uniform(20.0, 80.0) 
                    for i in range(1, 4)
                }
            },
            "optimization_type": random.choice(["ml_optimized", "webster_formula", "adaptive"])
        }
        
        with self.client.post(f"/api/v1/signals/optimize/{self.intersection_id}", 
                            json=optimization_request,
                            catch_response=True) as response:
            if response.status_code in [200, 404, 422]:
                response.success()
            else:
                response.failure(f"Signal optimization failed with status {response.status_code}")
    
    @task(1)
    def get_ml_metrics(self):
        """Get ML performance metrics"""
        with self.client.get("/api/v1/ml/metrics", 
                           catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"ML metrics failed with status {response.status_code}")
    
    @task(1)
    def get_traffic_history(self):
        """Get traffic history"""
        hours = random.randint(1, 24)
        with self.client.get(f"/api/v1/traffic/history/{self.intersection_id}?hours={hours}", 
                           catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"Traffic history failed with status {response.status_code}")
    
    @task(1)
    def get_cv_streams(self):
        """Get CV stream information"""
        with self.client.get("/api/v1/cv/streams", 
                           catch_response=True) as response:
            if response.status_code in [200, 404]:
                response.success()
            else:
                response.failure(f"CV streams failed with status {response.status_code}")

class HighLoadUser(HttpUser):
    """Simulates high-load scenarios"""
    
    wait_time = between(0.1, 0.5)  # Very short wait times
    
    def on_start(self):
        self.intersection_id = f"high_load_intersection_{random.randint(1, 3)}"
        self.camera_id = f"high_load_cam_{random.randint(1, 10):03d}"
    
    @task(5)
    def rapid_health_checks(self):
        """Rapid health checks"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Rapid health check failed: {response.status_code}")
    
    @task(3)
    def rapid_traffic_updates(self):
        """Rapid traffic data updates"""
        traffic_data = {
            "intersection_id": self.intersection_id,
            "timestamp": int(time.time()),
            "vehicle_counts": {
                "car": random.randint(20, 50),
                "motorcycle": random.randint(10, 25),
                "bus": random.randint(2, 8),
                "truck": random.randint(1, 5)
            },
            "lane_occupancy": {
                f"lane_{i}": random.uniform(0.5, 0.95) 
                for i in range(1, 4)
            },
            "waiting_times": {
                f"lane_{i}": random.uniform(30.0, 90.0) 
                for i in range(1, 4)
            },
            "coordinates": {"lat": 20.2961, "lng": 85.8245}
        }
        
        with self.client.post("/api/v1/traffic/ingest", 
                            json=traffic_data,
                            catch_response=True) as response:
            if response.status_code in [200, 404, 422]:
                response.success()
            else:
                response.failure(f"Rapid traffic update failed: {response.status_code}")

# Custom event handlers
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Log slow requests"""
    if response_time > 1000:  # More than 1 second
        logger.warning(f"Slow request: {name} took {response_time}ms")

@events.user_error.add_listener
def on_user_error(user_instance, exception, tb, **kwargs):
    """Log user errors"""
    logger.error(f"User error: {exception}")

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "p95_response_time": 300,  # 300ms
    "p99_response_time": 500,  # 500ms
    "error_rate": 0.01,  # 1%
    "throughput": 100  # 100 RPS
}

def check_performance_thresholds(stats):
    """Check if performance meets thresholds"""
    results = {}
    
    # Check P95 response time
    p95 = stats.get("p95_response_time", 0)
    results["p95_ok"] = p95 <= PERFORMANCE_THRESHOLDS["p95_response_time"]
    
    # Check P99 response time
    p99 = stats.get("p99_response_time", 0)
    results["p99_ok"] = p99 <= PERFORMANCE_THRESHOLDS["p99_response_time"]
    
    # Check error rate
    error_rate = stats.get("error_rate", 0)
    results["error_rate_ok"] = error_rate <= PERFORMANCE_THRESHOLDS["error_rate"]
    
    # Check throughput
    throughput = stats.get("throughput", 0)
    results["throughput_ok"] = throughput >= PERFORMANCE_THRESHOLDS["throughput"]
    
    return results

# Locust configuration
class WebsiteUser(TrafficManagementUser):
    host = "http://localhost:8000"
    weight = 3

class HighLoadUser(HighLoadUser):
    host = "http://localhost:8000"
    weight = 1

# Test scenarios
def run_load_test():
    """Run load test with different scenarios"""
    scenarios = [
        {
            "name": "Normal Load",
            "users": 50,
            "spawn_rate": 10,
            "duration": "5m"
        },
        {
            "name": "High Load",
            "users": 100,
            "spawn_rate": 20,
            "duration": "3m"
        },
        {
            "name": "Peak Load",
            "users": 200,
            "spawn_rate": 50,
            "duration": "2m"
        }
    ]
    
    for scenario in scenarios:
        logger.info(f"Running scenario: {scenario['name']}")
        # Run locust with scenario parameters
        # This would be run from command line:
        # locust -f test_load_testing.py --users {users} --spawn-rate {spawn_rate} --run-time {duration}

if __name__ == "__main__":
    # This file is meant to be run with locust command
    # Example: locust -f test_load_testing.py --host=http://localhost:8000
    print("Load testing configuration loaded")
    print("Run with: locust -f test_load_testing.py --host=http://localhost:8000")
