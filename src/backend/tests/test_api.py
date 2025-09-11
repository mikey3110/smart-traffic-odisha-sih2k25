"""
Unit tests for API endpoints
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from main import app
from database.connection import get_db
from services.redis_service import redis_service


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_db():
    """Mock database session"""
    return Mock()


@pytest.fixture
def mock_redis():
    """Mock Redis service"""
    mock_redis = Mock()
    mock_redis.connected = True
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    return mock_redis


class TestTrafficAPI:
    """Test traffic API endpoints"""
    
    def test_ingest_traffic_data_success(self, client, mock_db, mock_redis):
        """Test successful traffic data ingestion"""
        with patch('api.dependencies.get_db', return_value=mock_db), \
             patch('services.redis_service.redis_service', mock_redis):
            
            data = {
                "intersection_id": "junction-1",
                "timestamp": int(datetime.now().timestamp()),
                "lane_counts": {"north_lane": 10, "south_lane": 8},
                "avg_speed": 25.5,
                "weather_condition": "clear",
                "confidence_score": 0.85
            }
            
            # Mock database operations
            mock_intersection = Mock()
            mock_intersection.id = "junction-1"
            mock_db.query.return_value.filter.return_value.first.return_value = mock_intersection
            
            mock_traffic_record = Mock()
            mock_traffic_record.id = 1
            mock_db.add.return_value = None
            mock_db.commit.return_value = None
            mock_db.refresh.return_value = None
            
            response = client.post("/api/v1/traffic/ingest", json=data)
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["status"] == "success"
            assert "Traffic data ingested" in response_data["message"]
    
    def test_ingest_traffic_data_invalid_intersection(self, client, mock_db):
        """Test traffic data ingestion with invalid intersection"""
        with patch('api.dependencies.get_db', return_value=mock_db):
            data = {
                "intersection_id": "nonexistent",
                "timestamp": int(datetime.now().timestamp()),
                "lane_counts": {"north_lane": 10}
            }
            
            # Mock intersection not found
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            response = client.post("/api/v1/traffic/ingest", json=data)
            
            assert response.status_code == 404
            response_data = response.json()
            assert response_data["detail"]["error_code"] == "INTERSECTION_NOT_FOUND"
    
    def test_get_traffic_status_success(self, client, mock_db, mock_redis):
        """Test successful traffic status retrieval"""
        with patch('api.dependencies.get_db', return_value=mock_db), \
             patch('services.redis_service.redis_service', mock_redis):
            
            # Mock intersection exists
            mock_intersection = Mock()
            mock_intersection.id = "junction-1"
            mock_db.query.return_value.filter.return_value.first.return_value = mock_intersection
            
            # Mock Redis data
            mock_redis.get.return_value = {
                "intersection_id": "junction-1",
                "timestamp": int(datetime.now().timestamp()),
                "lane_counts": {"north_lane": 10, "south_lane": 8},
                "avg_speed": 25.5,
                "weather_condition": "clear"
            }
            
            response = client.get("/api/v1/traffic/status/junction-1")
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["status"] == "success"
            assert response_data["data"]["intersection_id"] == "junction-1"
    
    def test_get_traffic_status_not_found(self, client, mock_db):
        """Test traffic status retrieval for non-existent intersection"""
        with patch('api.dependencies.get_db', return_value=mock_db):
            # Mock intersection not found
            mock_db.query.return_value.filter.return_value.first.return_value = None
            
            response = client.get("/api/v1/traffic/status/nonexistent")
            
            assert response.status_code == 404
            response_data = response.json()
            assert response_data["detail"]["error_code"] == "INTERSECTION_NOT_FOUND"


class TestSignalsAPI:
    """Test signals API endpoints"""
    
    def test_optimize_signal_success(self, client, mock_db, mock_redis):
        """Test successful signal optimization"""
        with patch('api.dependencies.get_db', return_value=mock_db), \
             patch('services.redis_service.redis_service', mock_redis):
            
            data = {
                "intersection_id": "junction-1",
                "optimized_timings": [
                    {"lane": "north_lane", "duration": 30, "state": "green", "priority": 2},
                    {"lane": "south_lane", "duration": 30, "state": "green", "priority": 2}
                ],
                "confidence_score": 0.85,
                "expected_improvement": 15.0,
                "algorithm_used": "q_learning"
            }
            
            # Mock intersection exists
            mock_intersection = Mock()
            mock_intersection.id = "junction-1"
            mock_db.query.return_value.filter.return_value.first.return_value = mock_intersection
            
            # Mock database operations
            mock_optimization_record = Mock()
            mock_optimization_record.id = 1
            mock_db.add.return_value = None
            mock_db.commit.return_value = None
            mock_db.refresh.return_value = None
            
            response = client.put("/api/v1/signals/optimize/junction-1", json=data)
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["status"] == "success"
            assert "Signal optimization applied" in response_data["message"]
    
    def test_get_signal_status_success(self, client, mock_db, mock_redis):
        """Test successful signal status retrieval"""
        with patch('api.dependencies.get_db', return_value=mock_db), \
             patch('services.redis_service.redis_service', mock_redis):
            
            # Mock intersection exists
            mock_intersection = Mock()
            mock_intersection.id = "junction-1"
            mock_db.query.return_value.filter.return_value.first.return_value = mock_intersection
            
            # Mock Redis data
            mock_redis.get.return_value = {
                "intersection_id": "junction-1",
                "optimized_timings": [
                    {"lane": "north_lane", "duration": 30, "state": "green", "priority": 2}
                ],
                "confidence_score": 0.85,
                "status": "active"
            }
            
            response = client.get("/api/v1/signals/status/junction-1")
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["status"] == "success"
            assert response_data["data"]["intersection_id"] == "junction-1"


class TestHealthAPI:
    """Test health check API endpoints"""
    
    def test_health_check_success(self, client, mock_db, mock_redis):
        """Test successful health check"""
        with patch('api.dependencies.get_db', return_value=mock_db), \
             patch('services.redis_service.redis_service', mock_redis), \
             patch('database.connection.health_check') as mock_db_health, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.cpu_percent') as mock_cpu:
            
            # Mock database health
            mock_db_health.return_value = {
                "status": "connected",
                "pool_info": {"size": 10, "checked_in": 5}
            }
            
            # Mock Redis health
            mock_redis.health_check.return_value = {
                "status": "connected",
                "ping_time_ms": 1.5
            }
            
            # Mock system metrics
            mock_memory.return_value = Mock(percent=50.0, available=1024**3)
            mock_cpu.return_value = 25.0
            
            response = client.get("/api/v1/health/")
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["status"] == "healthy"
            assert "database" in response_data["components"]
            assert "redis" in response_data["components"]
    
    def test_readiness_check_ready(self, client, mock_db, mock_redis):
        """Test readiness check when service is ready"""
        with patch('api.dependencies.get_db', return_value=mock_db), \
             patch('services.redis_service.redis_service', mock_redis), \
             patch('database.connection.health_check') as mock_db_health:
            
            # Mock healthy services
            mock_db_health.return_value = {"status": "connected"}
            mock_redis.health_check.return_value = {"status": "connected"}
            
            response = client.get("/api/v1/health/ready")
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["data"]["ready"] is True
    
    def test_readiness_check_not_ready(self, client, mock_db, mock_redis):
        """Test readiness check when service is not ready"""
        with patch('api.dependencies.get_db', return_value=mock_db), \
             patch('services.redis_service.redis_service', mock_redis), \
             patch('database.connection.health_check') as mock_db_health:
            
            # Mock unhealthy services
            mock_db_health.return_value = {"status": "error"}
            mock_redis.health_check.return_value = {"status": "disconnected"}
            
            response = client.get("/api/v1/health/ready")
            
            assert response.status_code == 200
            response_data = response.json()
            assert response_data["data"]["ready"] is False
    
    def test_liveness_check(self, client):
        """Test liveness check"""
        response = client.get("/api/v1/health/live")
        
        assert response.status_code == 200
        response_data = response.json()
        assert response_data["data"]["alive"] is True


class TestErrorHandling:
    """Test error handling"""
    
    def test_validation_error(self, client):
        """Test validation error handling"""
        data = {
            "intersection_id": "",  # Invalid empty string
            "timestamp": int(datetime.now().timestamp()),
            "lane_counts": {"north_lane": 10}
        }
        
        response = client.post("/api/v1/traffic/ingest", json=data)
        
        assert response.status_code == 422
        response_data = response.json()
        assert response_data["error_code"] == "VALIDATION_ERROR"
    
    def test_internal_server_error(self, client, mock_db):
        """Test internal server error handling"""
        with patch('api.dependencies.get_db', return_value=mock_db):
            # Mock database error
            mock_db.query.side_effect = Exception("Database error")
            
            response = client.get("/api/v1/traffic/status/junction-1")
            
            assert response.status_code == 500
            response_data = response.json()
            assert response_data["error_code"] == "INTERNAL_SERVER_ERROR"