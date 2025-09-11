"""
Performance and load testing for Smart Traffic Management System
"""

import pytest
import asyncio
import time
import statistics
import concurrent.futures
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import httpx
from fastapi.testclient import TestClient

from src.backend.main import app

class TestLoadTesting:
    """Load testing for system performance"""
    
    @pytest.fixture
    def client(self):
        """Create test client for load testing"""
        return TestClient(app)
    
    def test_concurrent_traffic_light_creation(self, client):
        """Test concurrent creation of traffic lights"""
        def create_traffic_light(light_id):
            light_data = {
                "id": f"load_light_{light_id}",
                "name": f"Load Light {light_id}",
                "location": {"lat": 40.7128 + light_id * 0.001, "lng": -74.0060 + light_id * 0.001},
                "status": "normal",
                "current_phase": light_id % 4,
                "phase_duration": 30,
                "program": "adaptive",
                "vehicle_count": light_id % 20,
                "waiting_time": light_id * 0.5
            }
            
            start_time = time.time()
            response = client.post("/api/traffic/lights", json=light_data)
            end_time = time.time()
            
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 201
            }
        
        # Test with 50 concurrent requests
        num_requests = 50
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_traffic_light, i) for i in range(num_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        response_times = [r["response_time"] for r in results]
        
        assert len(successful_requests) >= num_requests * 0.95  # 95% success rate
        assert statistics.mean(response_times) < 2.0  # Average response time < 2 seconds
        assert max(response_times) < 5.0  # Max response time < 5 seconds
    
    def test_concurrent_vehicle_creation(self, client):
        """Test concurrent creation of vehicles"""
        def create_vehicle(vehicle_id):
            vehicle_data = {
                "id": f"load_vehicle_{vehicle_id}",
                "type": "passenger",
                "position": {"lat": 40.7128 + vehicle_id * 0.001, "lng": -74.0060 + vehicle_id * 0.001},
                "speed": 25.0 + vehicle_id % 10,
                "lane": f"lane_{vehicle_id % 4}",
                "route": ["test_route"],
                "waiting_time": vehicle_id * 0.1,
                "co2_emission": 0.1,
                "fuel_consumption": 0.05
            }
            
            start_time = time.time()
            response = client.post("/api/traffic/vehicles", json=vehicle_data)
            end_time = time.time()
            
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 201
            }
        
        # Test with 100 concurrent requests
        num_requests = 100
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(create_vehicle, i) for i in range(num_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        response_times = [r["response_time"] for r in results]
        
        assert len(successful_requests) >= num_requests * 0.95  # 95% success rate
        assert statistics.mean(response_times) < 1.5  # Average response time < 1.5 seconds
        assert max(response_times) < 3.0  # Max response time < 3 seconds
    
    def test_high_frequency_data_updates(self, client):
        """Test high frequency data updates"""
        # Create initial traffic light
        light_data = {
            "id": "high_freq_light",
            "name": "High Frequency Light",
            "location": {"lat": 40.7128, "lng": -74.0060},
            "status": "normal",
            "current_phase": 0,
            "phase_duration": 30,
            "program": "adaptive",
            "vehicle_count": 10,
            "waiting_time": 20.0
        }
        
        response = client.post("/api/traffic/lights", json=light_data)
        assert response.status_code == 201
        
        # Perform high frequency updates
        num_updates = 200
        response_times = []
        
        for i in range(num_updates):
            update_data = {
                "vehicle_count": 10 + i % 20,
                "waiting_time": 20.0 + i * 0.1,
                "current_phase": i % 4
            }
            
            start_time = time.time()
            response = client.put("/api/traffic/lights/high_freq_light", json=update_data)
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append(end_time - start_time)
        
        # Analyze performance
        avg_response_time = statistics.mean(response_times)
        max_response_time = max(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        
        assert avg_response_time < 0.5  # Average response time < 0.5 seconds
        assert max_response_time < 2.0  # Max response time < 2 seconds
        assert p95_response_time < 1.0  # 95th percentile < 1 second
    
    def test_database_query_performance(self, client):
        """Test database query performance under load"""
        # Create test data
        for i in range(100):
            light_data = {
                "id": f"query_light_{i}",
                "name": f"Query Light {i}",
                "location": {"lat": 40.7128 + i * 0.001, "lng": -74.0060 + i * 0.001},
                "status": "normal",
                "current_phase": i % 4,
                "phase_duration": 30,
                "program": "adaptive",
                "vehicle_count": i % 20,
                "waiting_time": i * 0.5
            }
            client.post("/api/traffic/lights", json=light_data)
        
        # Test query performance
        query_times = []
        
        for _ in range(50):
            start_time = time.time()
            response = client.get("/api/traffic/lights")
            end_time = time.time()
            
            assert response.status_code == 200
            query_times.append(end_time - start_time)
        
        # Analyze query performance
        avg_query_time = statistics.mean(query_times)
        max_query_time = max(query_times)
        
        assert avg_query_time < 1.0  # Average query time < 1 second
        assert max_query_time < 2.0  # Max query time < 2 seconds
    
    def test_memory_usage_under_load(self, client):
        """Test memory usage under load"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large amount of data
        for i in range(1000):
            vehicle_data = {
                "id": f"memory_vehicle_{i}",
                "type": "passenger",
                "position": {"lat": 40.7128 + i * 0.001, "lng": -74.0060 + i * 0.001},
                "speed": 25.0 + i % 10,
                "lane": f"lane_{i % 4}",
                "route": ["test_route"],
                "waiting_time": i * 0.1,
                "co2_emission": 0.1,
                "fuel_consumption": 0.05
            }
            client.post("/api/traffic/vehicles", json=vehicle_data)
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for 1000 records)
        assert memory_increase < 100  # MB
    
    def test_concurrent_optimization_requests(self, client):
        """Test concurrent optimization requests"""
        def run_optimization(request_id):
            optimization_data = {
                "intersection_id": f"opt_intersection_{request_id}",
                "algorithm": "q_learning",
                "traffic_data": {
                    "vehicles": [
                        {
                            "id": f"vehicle_{request_id}",
                            "type": "passenger",
                            "speed": 25.0,
                            "waiting_time": 5.0
                        }
                    ],
                    "traffic_lights": [
                        {
                            "id": f"light_{request_id}",
                            "phase": 0,
                            "duration": 30,
                            "vehicle_count": 10,
                            "waiting_time": 20.0
                        }
                    ]
                }
            }
            
            start_time = time.time()
            response = client.post("/api/optimization/optimize", json=optimization_data)
            end_time = time.time()
            
            return {
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "success": response.status_code == 200
            }
        
        # Test with 20 concurrent optimization requests
        num_requests = 20
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_optimization, i) for i in range(num_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        response_times = [r["response_time"] for r in results]
        
        assert len(successful_requests) >= num_requests * 0.9  # 90% success rate
        assert statistics.mean(response_times) < 5.0  # Average response time < 5 seconds
        assert max(response_times) < 10.0  # Max response time < 10 seconds

class TestStressTesting:
    """Stress testing for system limits"""
    
    @pytest.fixture
    def client(self):
        """Create test client for stress testing"""
        return TestClient(app)
    
    def test_maximum_concurrent_connections(self, client):
        """Test maximum concurrent connections"""
        def make_request(request_id):
            try:
                start_time = time.time()
                response = client.get("/health")
                end_time = time.time()
                
                return {
                    "success": response.status_code == 200,
                    "response_time": end_time - start_time,
                    "request_id": request_id
                }
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "request_id": request_id
                }
        
        # Test with increasing number of concurrent connections
        max_connections = 100
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_connections) as executor:
            futures = [executor.submit(make_request, i) for i in range(max_connections)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        success_rate = len(successful_requests) / len(results)
        
        assert success_rate >= 0.95  # 95% success rate
        assert len(successful_requests) >= max_connections * 0.95
    
    def test_large_data_handling(self, client):
        """Test handling of large data payloads"""
        # Create large traffic data
        large_traffic_data = {
            "intersection_id": "large_intersection",
            "traffic_lights": [
                {
                    "id": f"light_{i}",
                    "phase": i % 4,
                    "duration": 30,
                    "vehicle_count": i % 20,
                    "waiting_time": i * 0.5
                }
                for i in range(1000)  # 1000 traffic lights
            ],
            "vehicles": [
                {
                    "id": f"vehicle_{i}",
                    "type": "passenger",
                    "speed": 25.0 + i % 10,
                    "waiting_time": i * 0.1
                }
                for i in range(5000)  # 5000 vehicles
            ]
        }
        
        start_time = time.time()
        response = client.post("/api/optimization/optimize", json=large_traffic_data)
        end_time = time.time()
        
        # Should handle large data within reasonable time
        assert response.status_code == 200
        assert end_time - start_time < 30  # Less than 30 seconds
    
    def test_rapid_successive_requests(self, client):
        """Test rapid successive requests"""
        # Create initial data
        light_data = {
            "id": "rapid_light",
            "name": "Rapid Light",
            "location": {"lat": 40.7128, "lng": -74.0060},
            "status": "normal",
            "current_phase": 0,
            "phase_duration": 30,
            "program": "adaptive",
            "vehicle_count": 10,
            "waiting_time": 20.0
        }
        client.post("/api/traffic/lights", json=light_data)
        
        # Make rapid successive requests
        num_requests = 500
        response_times = []
        errors = 0
        
        for i in range(num_requests):
            try:
                start_time = time.time()
                response = client.get("/api/traffic/lights/rapid_light")
                end_time = time.time()
                
                if response.status_code == 200:
                    response_times.append(end_time - start_time)
                else:
                    errors += 1
            except Exception:
                errors += 1
        
        # Analyze results
        success_rate = (num_requests - errors) / num_requests
        avg_response_time = statistics.mean(response_times) if response_times else float('inf')
        
        assert success_rate >= 0.99  # 99% success rate
        assert avg_response_time < 0.1  # Average response time < 0.1 seconds

class TestPerformanceBenchmarks:
    """Performance benchmarks for system components"""
    
    @pytest.fixture
    def client(self):
        """Create test client for benchmarking"""
        return TestClient(app)
    
    def test_api_response_time_benchmarks(self, client):
        """Benchmark API response times"""
        benchmarks = {}
        
        # Health check benchmark
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        benchmarks["health_check"] = end_time - start_time
        assert response.status_code == 200
        
        # Traffic lights list benchmark
        start_time = time.time()
        response = client.get("/api/traffic/lights")
        end_time = time.time()
        benchmarks["traffic_lights_list"] = end_time - start_time
        assert response.status_code == 200
        
        # Vehicles list benchmark
        start_time = time.time()
        response = client.get("/api/traffic/vehicles")
        end_time = time.time()
        benchmarks["vehicles_list"] = end_time - start_time
        assert response.status_code == 200
        
        # Performance metrics benchmark
        start_time = time.time()
        response = client.get("/api/traffic/metrics")
        end_time = time.time()
        benchmarks["performance_metrics"] = end_time - start_time
        assert response.status_code == 200
        
        # Verify benchmarks meet requirements
        assert benchmarks["health_check"] < 0.1  # Health check < 100ms
        assert benchmarks["traffic_lights_list"] < 0.5  # Traffic lights list < 500ms
        assert benchmarks["vehicles_list"] < 0.5  # Vehicles list < 500ms
        assert benchmarks["performance_metrics"] < 1.0  # Performance metrics < 1s
    
    def test_database_performance_benchmarks(self, client):
        """Benchmark database performance"""
        # Create test data
        for i in range(100):
            light_data = {
                "id": f"benchmark_light_{i}",
                "name": f"Benchmark Light {i}",
                "location": {"lat": 40.7128 + i * 0.001, "lng": -74.0060 + i * 0.001},
                "status": "normal",
                "current_phase": i % 4,
                "phase_duration": 30,
                "program": "adaptive",
                "vehicle_count": i % 20,
                "waiting_time": i * 0.5
            }
            client.post("/api/traffic/lights", json=light_data)
        
        # Benchmark different query types
        benchmarks = {}
        
        # Simple select benchmark
        start_time = time.time()
        response = client.get("/api/traffic/lights?limit=10")
        end_time = time.time()
        benchmarks["simple_select"] = end_time - start_time
        assert response.status_code == 200
        
        # Filtered query benchmark
        start_time = time.time()
        response = client.get("/api/traffic/lights?status=normal")
        end_time = time.time()
        benchmarks["filtered_query"] = end_time - start_time
        assert response.status_code == 200
        
        # Count query benchmark
        start_time = time.time()
        response = client.get("/api/traffic/lights?count=true")
        end_time = time.time()
        benchmarks["count_query"] = end_time - start_time
        assert response.status_code == 200
        
        # Verify database performance
        assert benchmarks["simple_select"] < 0.2  # Simple select < 200ms
        assert benchmarks["filtered_query"] < 0.3  # Filtered query < 300ms
        assert benchmarks["count_query"] < 0.1  # Count query < 100ms
    
    def test_optimization_performance_benchmarks(self, client):
        """Benchmark optimization performance"""
        # Create test traffic data
        traffic_data = {
            "intersection_id": "benchmark_intersection",
            "traffic_lights": [
                {
                    "id": f"light_{i}",
                    "phase": i % 4,
                    "duration": 30,
                    "vehicle_count": i % 20,
                    "waiting_time": i * 0.5
                }
                for i in range(10)
            ],
            "vehicles": [
                {
                    "id": f"vehicle_{i}",
                    "type": "passenger",
                    "speed": 25.0 + i % 10,
                    "waiting_time": i * 0.1
                }
                for i in range(50)
            ]
        }
        
        # Benchmark different optimization algorithms
        algorithms = ["q_learning", "dynamic_programming", "websters_formula"]
        benchmarks = {}
        
        for algorithm in algorithms:
            optimization_data = {
                "intersection_id": "benchmark_intersection",
                "algorithm": algorithm,
                "traffic_data": traffic_data
            }
            
            start_time = time.time()
            response = client.post("/api/optimization/optimize", json=optimization_data)
            end_time = time.time()
            
            benchmarks[algorithm] = end_time - start_time
            assert response.status_code == 200
        
        # Verify optimization performance
        for algorithm, time_taken in benchmarks.items():
            assert time_taken < 10.0  # All algorithms < 10 seconds
    
    def test_memory_usage_benchmarks(self, client):
        """Benchmark memory usage patterns"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create data in batches and measure memory usage
        batch_sizes = [100, 500, 1000]
        memory_usage = []
        
        for batch_size in batch_sizes:
            # Create batch of data
            for i in range(batch_size):
                vehicle_data = {
                    "id": f"memory_benchmark_vehicle_{i}",
                    "type": "passenger",
                    "position": {"lat": 40.7128 + i * 0.001, "lng": -74.0060 + i * 0.001},
                    "speed": 25.0 + i % 10,
                    "lane": f"lane_{i % 4}",
                    "route": ["test_route"],
                    "waiting_time": i * 0.1,
                    "co2_emission": 0.1,
                    "fuel_consumption": 0.05
                }
                client.post("/api/traffic/vehicles", json=vehicle_data)
            
            # Measure memory usage
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage.append(current_memory - initial_memory)
        
        # Verify memory usage is reasonable
        for i, usage in enumerate(memory_usage):
            records = batch_sizes[i]
            memory_per_record = usage / records
            assert memory_per_record < 0.1  # Less than 0.1MB per record
