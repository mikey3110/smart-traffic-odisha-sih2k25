"""
Unit tests for Backend API endpoints and services
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from src.backend.main import app
from src.backend.database.models import TrafficLight, Vehicle, Intersection, PerformanceMetrics
from src.backend.services.traffic_service import TrafficService
from src.backend.services.optimization_service import OptimizationService
from src.backend.api.traffic import router as traffic_router
from src.backend.api.signals import router as signals_router

class TestTrafficLightAPI:
    """Test traffic light API endpoints"""
    
    def test_get_traffic_lights_empty(self, client):
        """Test getting traffic lights when none exist"""
        response = client.get("/api/traffic/lights")
        assert response.status_code == 200
        assert response.json() == {"success": True, "data": [], "timestamp": response.json()["timestamp"]}
    
    def test_get_traffic_lights_with_data(self, client, sample_traffic_light_data):
        """Test getting traffic lights with data"""
        # Mock database response
        with patch('src.backend.database.connection.get_db') as mock_db:
            mock_session = Mock()
            mock_light = Mock()
            mock_light.id = sample_traffic_light_data["id"]
            mock_light.name = sample_traffic_light_data["name"]
            mock_light.status = sample_traffic_light_data["status"]
            mock_light.current_phase = sample_traffic_light_data["current_phase"]
            mock_light.phase_duration = sample_traffic_light_data["phase_duration"]
            mock_light.vehicle_count = sample_traffic_light_data["vehicle_count"]
            mock_light.waiting_time = sample_traffic_light_data["waiting_time"]
            mock_light.location = sample_traffic_light_data["location"]
            mock_light.program = sample_traffic_light_data["program"]
            mock_light.last_update = datetime.now()
            
            mock_session.query.return_value.all.return_value = [mock_light]
            mock_db.return_value = mock_session
            
            response = client.get("/api/traffic/lights")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["data"]) == 1
            assert data["data"][0]["id"] == sample_traffic_light_data["id"]
    
    def test_get_traffic_light_by_id(self, client, sample_traffic_light_data):
        """Test getting a specific traffic light by ID"""
        with patch('src.backend.database.connection.get_db') as mock_db:
            mock_session = Mock()
            mock_light = Mock()
            mock_light.id = sample_traffic_light_data["id"]
            mock_light.name = sample_traffic_light_data["name"]
            mock_light.status = sample_traffic_light_data["status"]
            mock_light.current_phase = sample_traffic_light_data["current_phase"]
            mock_light.phase_duration = sample_traffic_light_data["phase_duration"]
            mock_light.vehicle_count = sample_traffic_light_data["vehicle_count"]
            mock_light.waiting_time = sample_traffic_light_data["waiting_time"]
            mock_light.location = sample_traffic_light_data["location"]
            mock_light.program = sample_traffic_light_data["program"]
            mock_light.last_update = datetime.now()
            
            mock_session.query.return_value.filter.return_value.first.return_value = mock_light
            mock_db.return_value = mock_session
            
            response = client.get(f"/api/traffic/lights/{sample_traffic_light_data['id']}")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["id"] == sample_traffic_light_data["id"]
    
    def test_get_traffic_light_not_found(self, client):
        """Test getting a non-existent traffic light"""
        with patch('src.backend.database.connection.get_db') as mock_db:
            mock_session = Mock()
            mock_session.query.return_value.filter.return_value.first.return_value = None
            mock_db.return_value = mock_session
            
            response = client.get("/api/traffic/lights/non_existent")
            assert response.status_code == 404
            data = response.json()
            assert data["success"] is False
            assert "not found" in data["error"].lower()
    
    def test_update_traffic_light(self, client, sample_traffic_light_data):
        """Test updating a traffic light"""
        update_data = {
            "name": "Updated Light",
            "phase_duration": 45,
            "program": "fixed"
        }
        
        with patch('src.backend.database.connection.get_db') as mock_db:
            mock_session = Mock()
            mock_light = Mock()
            mock_light.id = sample_traffic_light_data["id"]
            mock_light.name = sample_traffic_light_data["name"]
            mock_light.phase_duration = sample_traffic_light_data["phase_duration"]
            mock_light.program = sample_traffic_light_data["program"]
            
            mock_session.query.return_value.filter.return_value.first.return_value = mock_light
            mock_db.return_value = mock_session
            
            response = client.put(
                f"/api/traffic/lights/{sample_traffic_light_data['id']}",
                json=update_data
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
    
    def test_control_traffic_light(self, client, sample_traffic_light_data):
        """Test controlling a traffic light"""
        control_data = {
            "phase": 2,
            "duration": 30
        }
        
        with patch('src.backend.database.connection.get_db') as mock_db:
            mock_session = Mock()
            mock_light = Mock()
            mock_light.id = sample_traffic_light_data["id"]
            mock_light.current_phase = sample_traffic_light_data["current_phase"]
            mock_light.phase_duration = sample_traffic_light_data["phase_duration"]
            
            mock_session.query.return_value.filter.return_value.first.return_value = mock_light
            mock_db.return_value = mock_session
            
            response = client.post(
                f"/api/traffic/lights/{sample_traffic_light_data['id']}/control",
                json=control_data
            )
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

class TestVehicleAPI:
    """Test vehicle API endpoints"""
    
    def test_get_vehicles_empty(self, client):
        """Test getting vehicles when none exist"""
        response = client.get("/api/traffic/vehicles")
        assert response.status_code == 200
        assert response.json() == {"success": True, "data": [], "timestamp": response.json()["timestamp"]}
    
    def test_get_vehicles_with_filters(self, client, sample_vehicle_data):
        """Test getting vehicles with filters"""
        with patch('src.backend.database.connection.get_db') as mock_db:
            mock_session = Mock()
            mock_vehicle = Mock()
            mock_vehicle.id = sample_vehicle_data["id"]
            mock_vehicle.type = sample_vehicle_data["type"]
            mock_vehicle.position = sample_vehicle_data["position"]
            mock_vehicle.speed = sample_vehicle_data["speed"]
            mock_vehicle.lane = sample_vehicle_data["lane"]
            mock_vehicle.waiting_time = sample_vehicle_data["waiting_time"]
            mock_vehicle.timestamp = datetime.now()
            
            mock_session.query.return_value.filter.return_value.all.return_value = [mock_vehicle]
            mock_db.return_value = mock_session
            
            response = client.get("/api/traffic/vehicles?type=passenger&min_speed=20")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["data"]) == 1
    
    def test_get_vehicle_by_id(self, client, sample_vehicle_data):
        """Test getting a specific vehicle by ID"""
        with patch('src.backend.database.connection.get_db') as mock_db:
            mock_session = Mock()
            mock_vehicle = Mock()
            mock_vehicle.id = sample_vehicle_data["id"]
            mock_vehicle.type = sample_vehicle_data["type"]
            mock_vehicle.position = sample_vehicle_data["position"]
            mock_vehicle.speed = sample_vehicle_data["speed"]
            mock_vehicle.timestamp = datetime.now()
            
            mock_session.query.return_value.filter.return_value.first.return_value = mock_vehicle
            mock_db.return_value = mock_session
            
            response = client.get(f"/api/traffic/vehicles/{sample_vehicle_data['id']}")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["id"] == sample_vehicle_data["id"]

class TestIntersectionAPI:
    """Test intersection API endpoints"""
    
    def test_get_intersections_empty(self, client):
        """Test getting intersections when none exist"""
        response = client.get("/api/traffic/intersections")
        assert response.status_code == 200
        assert response.json() == {"success": True, "data": [], "timestamp": response.json()["timestamp"]}
    
    def test_get_intersection_by_id(self, client, sample_intersection_data):
        """Test getting a specific intersection by ID"""
        with patch('src.backend.database.connection.get_db') as mock_db:
            mock_session = Mock()
            mock_intersection = Mock()
            mock_intersection.id = sample_intersection_data["id"]
            mock_intersection.name = sample_intersection_data["name"]
            mock_intersection.location = sample_intersection_data["location"]
            mock_intersection.total_vehicles = sample_intersection_data["total_vehicles"]
            mock_intersection.waiting_vehicles = sample_intersection_data["waiting_vehicles"]
            mock_intersection.average_speed = sample_intersection_data["average_speed"]
            mock_intersection.throughput = sample_intersection_data["throughput"]
            mock_intersection.last_update = datetime.now()
            
            mock_session.query.return_value.filter.return_value.first.return_value = mock_intersection
            mock_db.return_value = mock_session
            
            response = client.get(f"/api/traffic/intersections/{sample_intersection_data['id']}")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["id"] == sample_intersection_data["id"]

class TestPerformanceMetricsAPI:
    """Test performance metrics API endpoints"""
    
    def test_get_performance_metrics(self, client, sample_performance_metrics):
        """Test getting performance metrics"""
        with patch('src.backend.database.connection.get_db') as mock_db:
            mock_session = Mock()
            mock_metrics = Mock()
            mock_metrics.timestamp = datetime.fromisoformat(sample_performance_metrics["timestamp"])
            mock_metrics.total_vehicles = sample_performance_metrics["total_vehicles"]
            mock_metrics.running_vehicles = sample_performance_metrics["running_vehicles"]
            mock_metrics.waiting_vehicles = sample_performance_metrics["waiting_vehicles"]
            mock_metrics.total_waiting_time = sample_performance_metrics["total_waiting_time"]
            mock_metrics.average_speed = sample_performance_metrics["average_speed"]
            mock_metrics.total_co2_emission = sample_performance_metrics["total_co2_emission"]
            mock_metrics.total_fuel_consumption = sample_performance_metrics["total_fuel_consumption"]
            mock_metrics.throughput = sample_performance_metrics["throughput"]
            
            mock_session.query.return_value.order_by.return_value.limit.return_value.all.return_value = [mock_metrics]
            mock_db.return_value = mock_session
            
            response = client.get("/api/traffic/metrics")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["data"]) == 1
    
    def test_get_performance_metrics_with_time_range(self, client):
        """Test getting performance metrics with time range"""
        start_time = (datetime.now() - timedelta(hours=1)).isoformat()
        end_time = datetime.now().isoformat()
        
        with patch('src.backend.database.connection.get_db') as mock_db:
            mock_session = Mock()
            mock_session.query.return_value.filter.return_value.order_by.return_value.all.return_value = []
            mock_db.return_value = mock_session
            
            response = client.get(f"/api/traffic/metrics?start_time={start_time}&end_time={end_time}")
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True

class TestHealthAPI:
    """Test health check API endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_health_check_detailed(self, client):
        """Test detailed health check endpoint"""
        response = client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "database" in data["components"]
        assert "redis" in data["components"]

class TestErrorHandling:
    """Test error handling in API endpoints"""
    
    def test_invalid_json_request(self, client):
        """Test handling of invalid JSON in request body"""
        response = client.post(
            "/api/traffic/lights/test_id/control",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields"""
        response = client.post(
            "/api/traffic/lights/test_id/control",
            json={"phase": 1}  # Missing duration
        )
        assert response.status_code == 422
    
    def test_invalid_phase_value(self, client):
        """Test handling of invalid phase value"""
        response = client.post(
            "/api/traffic/lights/test_id/control",
            json={"phase": 5, "duration": 30}  # Invalid phase (should be 0-3)
        )
        assert response.status_code == 422
    
    def test_database_connection_error(self, client):
        """Test handling of database connection errors"""
        with patch('src.backend.database.connection.get_db') as mock_db:
            mock_db.side_effect = Exception("Database connection failed")
            
            response = client.get("/api/traffic/lights")
            assert response.status_code == 500
            data = response.json()
            assert data["success"] is False
            assert "error" in data

class TestAuthentication:
    """Test authentication and authorization"""
    
    def test_protected_endpoint_without_auth(self, client):
        """Test accessing protected endpoint without authentication"""
        response = client.get("/api/admin/users")
        assert response.status_code == 401
    
    def test_protected_endpoint_with_invalid_token(self, client):
        """Test accessing protected endpoint with invalid token"""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/api/admin/users", headers=headers)
        assert response.status_code == 401
    
    def test_protected_endpoint_with_valid_token(self, client, auth_headers):
        """Test accessing protected endpoint with valid token"""
        with patch('src.backend.api.dependencies.verify_token') as mock_verify:
            mock_verify.return_value = {"user_id": "test_user", "role": "admin"}
            
            response = client.get("/api/admin/users", headers=auth_headers)
            assert response.status_code == 200

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limiting_exceeded(self, client):
        """Test rate limiting when limit is exceeded"""
        # Make multiple requests quickly
        for _ in range(101):  # Assuming limit is 100 per minute
            response = client.get("/api/traffic/lights")
            if response.status_code == 429:
                break
        
        # Should eventually get rate limited
        assert response.status_code == 429
        data = response.json()
        assert "rate limit" in data["error"].lower()

class TestDataValidation:
    """Test data validation in API endpoints"""
    
    def test_traffic_light_creation_validation(self, client):
        """Test validation when creating traffic light"""
        invalid_data = {
            "name": "",  # Empty name
            "location": {"lat": 200, "lng": -200},  # Invalid coordinates
            "status": "invalid_status"  # Invalid status
        }
        
        response = client.post("/api/traffic/lights", json=invalid_data)
        assert response.status_code == 422
        data = response.json()
        assert "validation error" in data["error"].lower()
    
    def test_vehicle_data_validation(self, client):
        """Test validation when creating vehicle data"""
        invalid_data = {
            "id": "",  # Empty ID
            "type": "invalid_type",  # Invalid vehicle type
            "speed": -10,  # Negative speed
            "position": {"lat": "invalid", "lng": "invalid"}  # Invalid position
        }
        
        response = client.post("/api/traffic/vehicles", json=invalid_data)
        assert response.status_code == 422
        data = response.json()
        assert "validation error" in data["error"].lower()

@pytest.mark.unit
class TestTrafficService:
    """Test TrafficService class"""
    
    def test_get_traffic_lights(self, mock_redis):
        """Test getting traffic lights from service"""
        with patch('src.backend.services.traffic_service.TrafficService') as mock_service:
            mock_instance = Mock()
            mock_instance.get_traffic_lights.return_value = []
            mock_service.return_value = mock_instance
            
            service = TrafficService()
            result = service.get_traffic_lights()
            assert result == []
    
    def test_update_traffic_light(self, mock_redis):
        """Test updating traffic light in service"""
        with patch('src.backend.services.traffic_service.TrafficService') as mock_service:
            mock_instance = Mock()
            mock_instance.update_traffic_light.return_value = {"id": "test", "updated": True}
            mock_service.return_value = mock_instance
            
            service = TrafficService()
            result = service.update_traffic_light("test_id", {"phase": 2})
            assert result["id"] == "test"
            assert result["updated"] is True

@pytest.mark.unit
class TestOptimizationService:
    """Test OptimizationService class"""
    
    def test_optimize_traffic_lights(self, mock_ml_optimizer):
        """Test traffic light optimization"""
        with patch('src.backend.services.optimization_service.OptimizationService') as mock_service:
            mock_instance = Mock()
            mock_instance.optimize_traffic_lights.return_value = {
                "optimized_phases": [30, 25, 35, 20],
                "efficiency_improvement": 0.15
            }
            mock_service.return_value = mock_instance
            
            service = OptimizationService()
            result = service.optimize_traffic_lights([])
            assert "optimized_phases" in result
            assert "efficiency_improvement" in result
