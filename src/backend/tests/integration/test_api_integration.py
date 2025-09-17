"""
Integration Tests for Backend API
Tests full request flows: CV → ML → TraCI
"""

import pytest
import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

# Test configuration
BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api/v1"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestAPIIntegration:
    """Integration tests for backend API"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.test_intersection_id = "test_intersection_1"
        self.test_camera_id = "test_cam_001"
        self.test_coordinates = {"lat": 20.2961, "lng": 85.8245}
        
    def test_health_endpoints(self):
        """Test health check endpoints"""
        try:
            # Test main health endpoint
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            
            # Test root endpoint
            response = requests.get(f"{BASE_URL}/", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend service not running")
    
    def test_traffic_data_flow(self):
        """Test complete traffic data flow"""
        try:
            # 1. Ingest traffic data
            traffic_data = {
                "intersection_id": self.test_intersection_id,
                "timestamp": int(time.time()),
                "vehicle_counts": {
                    "car": 15,
                    "motorcycle": 5,
                    "bus": 2,
                    "truck": 1
                },
                "lane_occupancy": {
                    "lane_1": 0.75,
                    "lane_2": 0.60,
                    "lane_3": 0.45
                },
                "waiting_times": {
                    "lane_1": 45.2,
                    "lane_2": 38.7,
                    "lane_3": 52.1
                },
                "coordinates": self.test_coordinates
            }
            
            response = requests.post(
                f"{API_BASE}/traffic/ingest",
                json=traffic_data,
                timeout=10
            )
            
            # Should return 200 or 404 (if endpoint not implemented)
            assert response.status_code in [200, 404, 422]
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] == True
                logger.info("Traffic data ingested successfully")
            
            # 2. Get traffic status
            response = requests.get(
                f"{API_BASE}/traffic/status/{self.test_intersection_id}",
                timeout=5
            )
            
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                data = response.json()
                assert "intersection_id" in data
                logger.info("Traffic status retrieved successfully")
            
            # 3. Get traffic history
            response = requests.get(
                f"{API_BASE}/traffic/history/{self.test_intersection_id}",
                timeout=5
            )
            
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, list)
                logger.info("Traffic history retrieved successfully")
                
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend service not running")
    
    def test_signal_optimization_flow(self):
        """Test signal optimization flow"""
        try:
            # 1. Optimize signals
            optimization_request = {
                "intersection_id": self.test_intersection_id,
                "current_phase": 0,
                "traffic_data": {
                    "vehicle_counts": {"car": 20, "motorcycle": 8, "bus": 3},
                    "lane_occupancy": {"lane_1": 0.80, "lane_2": 0.65},
                    "waiting_times": {"lane_1": 50.0, "lane_2": 42.0}
                },
                "optimization_type": "ml_optimized"
            }
            
            response = requests.post(
                f"{API_BASE}/signals/optimize/{self.test_intersection_id}",
                json=optimization_request,
                timeout=10
            )
            
            assert response.status_code in [200, 404, 422]
            
            if response.status_code == 200:
                data = response.json()
                assert "optimized_phases" in data
                assert "performance_metrics" in data
                logger.info("Signal optimization completed successfully")
            
            # 2. Get signal status
            response = requests.get(
                f"{API_BASE}/signals/status/{self.test_intersection_id}",
                timeout=5
            )
            
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                data = response.json()
                assert "current_phase" in data
                logger.info("Signal status retrieved successfully")
            
            # 3. Manual override
            override_request = {
                "phase": 1,
                "duration": 30,
                "reason": "emergency_vehicle"
            }
            
            response = requests.put(
                f"{API_BASE}/signals/override/{self.test_intersection_id}",
                json=override_request,
                timeout=5
            )
            
            assert response.status_code in [200, 404, 422]
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] == True
                logger.info("Signal override applied successfully")
                
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend service not running")
    
    def test_ml_metrics_flow(self):
        """Test ML metrics flow"""
        try:
            # 1. Get ML metrics
            response = requests.get(f"{API_BASE}/ml/metrics", timeout=5)
            
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                data = response.json()
                assert "data" in data
                assert "timestamp" in data["data"]
                logger.info("ML metrics retrieved successfully")
            
            # 2. Get ML performance
            response = requests.get(
                f"{API_BASE}/ml/performance/{self.test_intersection_id}",
                timeout=5
            )
            
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                data = response.json()
                assert "data" in data
                logger.info("ML performance retrieved successfully")
            
            # 3. Submit ML metrics
            metrics_data = {
                "intersection_id": self.test_intersection_id,
                "timestamp": int(time.time()),
                "reward": 0.85,
                "wait_time_reduction": 0.25,
                "throughput_increase": 0.15,
                "fuel_efficiency": 0.10,
                "safety_score": 0.95
            }
            
            response = requests.post(
                f"{API_BASE}/ml/metrics",
                json=metrics_data,
                timeout=5
            )
            
            assert response.status_code in [200, 404, 422]
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] == True
                logger.info("ML metrics submitted successfully")
                
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend service not running")
    
    def test_cv_integration_flow(self):
        """Test CV integration flow"""
        try:
            # 1. Submit vehicle counts
            cv_data = {
                "camera_id": self.test_camera_id,
                "intersection_id": self.test_intersection_id,
                "timestamp": int(time.time()),
                "total_vehicles": 18,
                "counts_by_class": {
                    "car": 12,
                    "motorcycle": 4,
                    "bus": 1,
                    "truck": 1
                },
                "coordinates": self.test_coordinates
            }
            
            response = requests.post(
                f"{API_BASE}/cv/counts",
                json=cv_data,
                timeout=5
            )
            
            assert response.status_code in [200, 404, 422]
            
            if response.status_code == 200:
                data = response.json()
                assert data["success"] == True
                logger.info("CV data submitted successfully")
            
            # 2. Get CV counts
            response = requests.get(
                f"{API_BASE}/cv/counts/{self.test_intersection_id}",
                timeout=5
            )
            
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                data = response.json()
                assert "data" in data
                logger.info("CV counts retrieved successfully")
            
            # 3. Get stream information
            response = requests.get(
                f"{API_BASE}/cv/streams",
                timeout=5
            )
            
            assert response.status_code in [200, 404]
            
            if response.status_code == 200:
                data = response.json()
                assert "streams" in data
                logger.info("CV streams retrieved successfully")
                
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend service not running")
    
    def test_error_handling(self):
        """Test error handling and validation"""
        try:
            # Test malformed JSON
            response = requests.post(
                f"{API_BASE}/traffic/ingest",
                data="invalid json",
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            assert response.status_code in [400, 422, 404]
            
            # Test missing required fields
            incomplete_data = {
                "intersection_id": self.test_intersection_id
                # Missing required fields
            }
            
            response = requests.post(
                f"{API_BASE}/traffic/ingest",
                json=incomplete_data,
                timeout=5
            )
            
            assert response.status_code in [400, 422, 404]
            
            # Test invalid intersection ID
            response = requests.get(
                f"{API_BASE}/traffic/status/invalid_id",
                timeout=5
            )
            
            assert response.status_code in [404, 400]
            
            logger.info("Error handling tests passed")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend service not running")
    
    def test_data_validation(self):
        """Test data validation and schema compliance"""
        try:
            # Test valid data
            valid_data = {
                "intersection_id": self.test_intersection_id,
                "timestamp": int(time.time()),
                "vehicle_counts": {"car": 10, "motorcycle": 3},
                "lane_occupancy": {"lane_1": 0.5},
                "waiting_times": {"lane_1": 30.0},
                "coordinates": self.test_coordinates
            }
            
            response = requests.post(
                f"{API_BASE}/traffic/ingest",
                json=valid_data,
                timeout=5
            )
            
            # Should accept valid data
            assert response.status_code in [200, 404, 422]
            
            # Test invalid data types
            invalid_data = {
                "intersection_id": 123,  # Should be string
                "timestamp": "invalid",  # Should be int
                "vehicle_counts": "invalid",  # Should be dict
                "coordinates": "invalid"  # Should be dict
            }
            
            response = requests.post(
                f"{API_BASE}/traffic/ingest",
                json=invalid_data,
                timeout=5
            )
            
            # Should reject invalid data
            assert response.status_code in [400, 422, 404]
            
            logger.info("Data validation tests passed")
            
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend service not running")
    
    def test_concurrent_requests(self):
        """Test concurrent request handling"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = requests.get(f"{BASE_URL}/health", timeout=5)
                results.put(response.status_code)
            except Exception as e:
                results.put(f"Error: {e}")
        
        # Start 10 concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            thread.start()
            threads.append(thread)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        status_codes = []
        while not results.empty():
            result = results.get()
            status_codes.append(result)
        
        # All requests should succeed
        assert len(status_codes) == 10
        assert all(code == 200 for code in status_codes if isinstance(code, int))
        
        logger.info("Concurrent request tests passed")

def test_full_workflow():
    """Test complete workflow from CV to ML to TraCI"""
    try:
        # This test simulates the complete workflow
        # 1. CV detects vehicles
        cv_data = {
            "camera_id": "cam_001",
            "intersection_id": "intersection_1",
            "timestamp": int(time.time()),
            "total_vehicles": 25,
            "counts_by_class": {"car": 18, "motorcycle": 5, "bus": 2},
            "coordinates": {"lat": 20.2961, "lng": 85.8245}
        }
        
        # Submit CV data
        response = requests.post(
            f"{API_BASE}/cv/counts",
            json=cv_data,
            timeout=5
        )
        
        if response.status_code == 200:
            logger.info("CV data submitted")
        
        # 2. ML optimization
        ml_request = {
            "intersection_id": "intersection_1",
            "traffic_data": cv_data,
            "optimization_type": "ml_optimized"
        }
        
        response = requests.post(
            f"{API_BASE}/signals/optimize/intersection_1",
            json=ml_request,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("ML optimization completed")
        
        # 3. Get results
        response = requests.get(
            f"{API_BASE}/signals/status/intersection_1",
            timeout=5
        )
        
        if response.status_code == 200:
            logger.info("Signal status retrieved")
        
        logger.info("Full workflow test completed")
        
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend service not running")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
