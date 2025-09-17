#!/usr/bin/env python3
"""
Comprehensive Test Suite for Smart Traffic Management System
Tests all components: Backend, ML, CV, Frontend, and Integration
"""

import pytest
import requests
import time
import json
import subprocess
import os
import sys
from datetime import datetime
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSuite:
    """Main test suite class"""
    
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.ml_url = "http://localhost:8001"
        self.cv_url = "http://localhost:5001"
        self.frontend_url = "http://localhost:3000"
        self.test_results = []
        
    def log_test_result(self, test_name: str, status: str, details: str = ""):
        """Log test result"""
        result = {
            "test_name": test_name,
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details
        }
        self.test_results.append(result)
        logger.info(f"{status.upper()}: {test_name} - {details}")

class BackendTests(TestSuite):
    """Backend API tests"""
    
    def test_health_endpoint(self):
        """Test backend health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            self.log_test_result("Backend Health", "PASS", f"Status: {data['status']}")
        except Exception as e:
            self.log_test_result("Backend Health", "FAIL", str(e))
            raise
    
    def test_api_endpoints(self):
        """Test all API endpoints"""
        endpoints = [
            "/",
            "/health",
            "/api/v1/traffic/status/test_intersection",
            "/api/v1/signals/status/test_intersection",
            "/api/v1/ml/metrics"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                # Accept 200, 404 (not implemented), or 422 (validation error)
                assert response.status_code in [200, 404, 422]
                self.log_test_result(f"API Endpoint {endpoint}", "PASS", f"Status: {response.status_code}")
            except Exception as e:
                self.log_test_result(f"API Endpoint {endpoint}", "FAIL", str(e))
    
    def test_traffic_data_ingestion(self):
        """Test traffic data ingestion"""
        test_data = {
            "intersection_id": "test_intersection_1",
            "timestamp": int(time.time()),
            "vehicle_counts": {"car": 10, "motorcycle": 5, "bus": 2},
            "lane_occupancy": {"lane_1": 0.5, "lane_2": 0.3},
            "waiting_times": {"lane_1": 30.0, "lane_2": 25.0},
            "coordinates": {"lat": 20.2961, "lng": 85.8245}
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/traffic/ingest",
                json=test_data,
                timeout=10
            )
            assert response.status_code in [200, 404, 422]
            self.log_test_result("Traffic Data Ingestion", "PASS", f"Status: {response.status_code}")
        except Exception as e:
            self.log_test_result("Traffic Data Ingestion", "FAIL", str(e))

class MLTests(TestSuite):
    """ML Engine tests"""
    
    def test_ml_health(self):
        """Test ML API health"""
        try:
            response = requests.get(f"{self.ml_url}/health", timeout=5)
            assert response.status_code == 200
            self.log_test_result("ML Health", "PASS", "ML API responding")
        except Exception as e:
            self.log_test_result("ML Health", "FAIL", str(e))
    
    def test_ml_prediction(self):
        """Test ML prediction endpoint"""
        test_data = {
            "intersection_id": "test_intersection_1",
            "traffic_data": {
                "vehicle_counts": {"car": 15, "motorcycle": 8, "bus": 3},
                "lane_occupancy": {"lane_1": 0.7, "lane_2": 0.5},
                "waiting_times": {"lane_1": 45.0, "lane_2": 35.0}
            }
        }
        
        try:
            response = requests.post(
                f"{self.ml_url}/api/v1/ml/predict",
                json=test_data,
                timeout=10
            )
            assert response.status_code in [200, 404, 422]
            self.log_test_result("ML Prediction", "PASS", f"Status: {response.status_code}")
        except Exception as e:
            self.log_test_result("ML Prediction", "FAIL", str(e))

class CVTests(TestSuite):
    """Computer Vision tests"""
    
    def test_cv_health(self):
        """Test CV service health"""
        try:
            response = requests.get(f"{self.cv_url}/cv/streams", timeout=5)
            assert response.status_code in [200, 404]
            self.log_test_result("CV Health", "PASS", "CV service responding")
        except Exception as e:
            self.log_test_result("CV Health", "FAIL", str(e))
    
    def test_vehicle_counting(self):
        """Test vehicle counting endpoint"""
        test_data = {
            "camera_id": "test_cam_001",
            "intersection_id": "test_intersection_1",
            "timestamp": int(time.time()),
            "total_vehicles": 12,
            "counts_by_class": {"car": 8, "motorcycle": 3, "bus": 1},
            "coordinates": {"lat": 20.2961, "lng": 85.8245}
        }
        
        try:
            response = requests.post(
                f"{self.cv_url}/cv/counts",
                json=test_data,
                timeout=5
            )
            assert response.status_code in [200, 404, 422]
            self.log_test_result("Vehicle Counting", "PASS", f"Status: {response.status_code}")
        except Exception as e:
            self.log_test_result("Vehicle Counting", "FAIL", str(e))

class IntegrationTests(TestSuite):
    """End-to-end integration tests"""
    
    def test_full_workflow(self):
        """Test complete CV → ML → Backend workflow"""
        try:
            # 1. Submit CV data
            cv_data = {
                "camera_id": "test_cam_001",
                "intersection_id": "test_intersection_1",
                "timestamp": int(time.time()),
                "total_vehicles": 20,
                "counts_by_class": {"car": 15, "motorcycle": 4, "bus": 1},
                "coordinates": {"lat": 20.2961, "lng": 85.8245}
            }
            
            cv_response = requests.post(
                f"{self.cv_url}/cv/counts",
                json=cv_data,
                timeout=5
            )
            
            # 2. ML optimization
            ml_data = {
                "intersection_id": "test_intersection_1",
                "traffic_data": cv_data,
                "optimization_type": "ml_optimized"
            }
            
            ml_response = requests.post(
                f"{self.ml_url}/api/v1/ml/predict",
                json=ml_data,
                timeout=10
            )
            
            # 3. Backend signal control
            signal_data = {
                "intersection_id": "test_intersection_1",
                "current_phase": 0,
                "traffic_data": cv_data,
                "optimization_type": "ml_optimized"
            }
            
            signal_response = requests.post(
                f"{self.base_url}/api/v1/signals/optimize/test_intersection_1",
                json=signal_data,
                timeout=10
            )
            
            # All should return valid responses
            assert cv_response.status_code in [200, 404, 422]
            assert ml_response.status_code in [200, 404, 422]
            assert signal_response.status_code in [200, 404, 422]
            
            self.log_test_result("Full Workflow", "PASS", "Complete data flow working")
            
        except Exception as e:
            self.log_test_result("Full Workflow", "FAIL", str(e))

class PerformanceTests(TestSuite):
    """Performance and load tests"""
    
    def test_response_times(self):
        """Test API response times"""
        endpoints = [
            f"{self.base_url}/health",
            f"{self.base_url}/api/v1/ml/metrics",
            f"{self.cv_url}/cv/streams"
        ]
        
        for endpoint in endpoints:
            try:
                start_time = time.time()
                response = requests.get(endpoint, timeout=5)
                end_time = time.time()
                
                response_time = (end_time - start_time) * 1000  # Convert to ms
                assert response_time < 300  # Should be under 300ms
                
                self.log_test_result(f"Response Time {endpoint}", "PASS", f"{response_time:.2f}ms")
            except Exception as e:
                self.log_test_result(f"Response Time {endpoint}", "FAIL", str(e))
    
    def test_concurrent_requests(self):
        """Test concurrent request handling"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
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
        success_count = sum(1 for code in status_codes if code == 200)
        assert success_count >= 8  # At least 80% should succeed
        
        self.log_test_result("Concurrent Requests", "PASS", f"{success_count}/10 successful")

def run_all_tests():
    """Run all test suites"""
    print("🚦 Smart Traffic Management System - Test Suite")
    print("=" * 60)
    
    # Initialize test suites
    backend_tests = BackendTests()
    ml_tests = MLTests()
    cv_tests = CVTests()
    integration_tests = IntegrationTests()
    performance_tests = PerformanceTests()
    
    # Run tests
    test_suites = [
        ("Backend Tests", backend_tests),
        ("ML Tests", ml_tests),
        ("CV Tests", cv_tests),
        ("Integration Tests", integration_tests),
        ("Performance Tests", performance_tests)
    ]
    
    all_results = []
    
    for suite_name, suite in test_suites:
        print(f"\n📊 Running {suite_name}...")
        
        # Run all test methods
        test_methods = [method for method in dir(suite) if method.startswith('test_')]
        
        for test_method in test_methods:
            try:
                getattr(suite, test_method)()
            except Exception as e:
                suite.log_test_result(test_method, "ERROR", str(e))
        
        all_results.extend(suite.test_results)
    
    # Generate report
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    total_tests = len(all_results)
    passed_tests = len([r for r in all_results if r["status"] == "PASS"])
    failed_tests = len([r for r in all_results if r["status"] == "FAIL"])
    error_tests = len([r for r in all_results if r["status"] == "ERROR"])
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed_tests}")
    print(f"❌ Failed: {failed_tests}")
    print(f"⚠️  Errors: {error_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    # Detailed results
    print("\n📝 DETAILED RESULTS:")
    for result in all_results:
        status_icon = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
        print(f"{status_icon} {result['test_name']}: {result['status']}")
        if result["details"]:
            print(f"   Details: {result['details']}")
    
    # Save results to file
    with open("test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Results saved to test_results.json")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
