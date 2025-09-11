"""
Comprehensive unit tests for enhanced backend features
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json

from main import app
from database.connection import get_db, Base
from database.models import Intersection, TrafficData, SignalTiming, OptimizationResult
from models.schemas import TrafficDataSchema, SignalOptimizationSchema
from services.redis_service import redis_service
from config.settings import settings
from exceptions.custom_exceptions import (
    ValidationError, DatabaseError, RedisError, 
    IntersectionNotFoundError, SignalOptimizationError
)


# Test database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    """Override database dependency for testing"""
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="function")
def setup_database():
    """Set up test database for each test"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def mock_redis():
    """Mock Redis service for testing"""
    with patch.object(redis_service, 'connected', True):
        with patch.object(redis_service, 'set') as mock_set:
            with patch.object(redis_service, 'get') as mock_get:
                with patch.object(redis_service, 'health_check') as mock_health:
                    mock_health.return_value = {
                        "status": "connected",
                        "ping_time_ms": 1.0,
                        "redis_version": "6.2.0"
                    }
                    yield mock_set, mock_get, mock_health


@pytest.fixture
def sample_intersection():
    """Create a sample intersection for testing"""
    return {
        "id": "test-intersection-1",
        "name": "Test Intersection",
        "location_lat": 20.2961,
        "location_lng": 85.8245,
        "lanes": ["north_lane", "south_lane", "east_lane", "west_lane"],
        "status": "operational"
    }


@pytest.fixture
def sample_traffic_data():
    """Create sample traffic data for testing"""
    return {
        "intersection_id": "test-intersection-1",
        "timestamp": int(datetime.now(timezone.utc).timestamp()),
        "lane_counts": {
            "north_lane": 15,
            "south_lane": 12,
            "east_lane": 8,
            "west_lane": 10
        },
        "avg_speed": 25.5,
        "weather_condition": "clear",
        "vehicle_types": {
            "car": 35,
            "truck": 5,
            "motorcycle": 8
        },
        "confidence_score": 0.85
    }


class TestLoggingConfiguration:
    """Test logging configuration and functionality"""
    
    def test_logger_creation(self):
        """Test that loggers are created correctly"""
        from config.logging_config import get_logger
        
        logger = get_logger("test_logger")
        assert logger.name == "traffic_management.test_logger"
        assert logger.level <= 20  # INFO level or lower
    
    def test_logging_middleware(self):
        """Test logging middleware functionality"""
        from middleware.logging_middleware import LoggingMiddleware
        
        middleware = LoggingMiddleware(
            app=app,
            log_requests=True,
            log_responses=True,
            log_request_body=False,
            log_response_body=False
        )
        
        assert middleware.log_requests is True
        assert middleware.log_responses is True
        assert middleware.max_body_size == 1024


class TestErrorHandling:
    """Test custom exception handling"""
    
    def test_validation_error(self):
        """Test validation error creation"""
        error = ValidationError("Test validation error", field="test_field")
        
        assert error.message == "Test validation error"
        assert error.error_code == "VALIDATION_ERROR"
        assert error.details["field"] == "test_field"
    
    def test_database_error(self):
        """Test database error creation"""
        error = DatabaseError("Test database error", operation="SELECT")
        
        assert error.message == "Test database error"
        assert error.error_code == "DATABASE_ERROR"
        assert error.details["operation"] == "SELECT"
    
    def test_intersection_not_found_error(self):
        """Test intersection not found error"""
        error = IntersectionNotFoundError("test-intersection-1")
        
        assert "test-intersection-1" in error.message
        assert error.error_code == "INTERSECTION_NOT_FOUND"
        assert error.details["intersection_id"] == "test-intersection-1"
    
    def test_http_exception_creation(self):
        """Test HTTP exception creation from custom exceptions"""
        from exceptions.custom_exceptions import create_http_exception
        
        error = ValidationError("Test error")
        http_exc = create_http_exception(error)
        
        assert http_exc.status_code == 422
        assert "error_code" in http_exc.detail


class TestDataValidation:
    """Test Pydantic model validation"""
    
    def test_traffic_data_validation_success(self, sample_traffic_data):
        """Test successful traffic data validation"""
        traffic_data = TrafficDataSchema(**sample_traffic_data)
        
        assert traffic_data.intersection_id == "test-intersection-1"
        assert traffic_data.lane_counts["north_lane"] == 15
        assert traffic_data.avg_speed == 25.5
        assert traffic_data.confidence_score == 0.85
    
    def test_traffic_data_validation_failure(self):
        """Test traffic data validation failure"""
        invalid_data = {
            "intersection_id": "",  # Empty ID should fail
            "timestamp": int(datetime.now(timezone.utc).timestamp()),
            "lane_counts": {"north_lane": 15}
        }
        
        with pytest.raises(ValueError):
            TrafficDataSchema(**invalid_data)
    
    def test_traffic_data_timestamp_validation(self):
        """Test timestamp validation"""
        # Test future timestamp (should fail)
        future_data = {
            "intersection_id": "test-intersection-1",
            "timestamp": int(datetime.now(timezone.utc).timestamp()) + 1000,  # 1000 seconds in future
            "lane_counts": {"north_lane": 15}
        }
        
        with pytest.raises(ValueError, match="Timestamp is in the future"):
            TrafficDataSchema(**future_data)
    
    def test_signal_optimization_validation(self):
        """Test signal optimization validation"""
        optimization_data = {
            "intersection_id": "test-intersection-1",
            "optimized_timings": [
                {
                    "lane": "north_lane",
                    "duration": 30,
                    "state": "green",
                    "priority": 2
                }
            ],
            "confidence_score": 0.9,
            "expected_improvement": 15.0,
            "algorithm_used": "ai_optimizer"
        }
        
        optimization = SignalOptimizationSchema(**optimization_data)
        assert optimization.intersection_id == "test-intersection-1"
        assert len(optimization.optimized_timings) == 1
        assert optimization.confidence_score == 0.9


class TestRedisService:
    """Test Redis service functionality"""
    
    def test_redis_connection(self, mock_redis):
        """Test Redis connection"""
        mock_set, mock_get, mock_health = mock_redis
        
        # Test connection
        result = redis_service.connect()
        assert result is True
        
        # Test health check
        health = redis_service.health_check()
        assert health["status"] == "connected"
    
    def test_redis_set_get(self, mock_redis):
        """Test Redis set and get operations"""
        mock_set, mock_get, mock_health = mock_redis
        
        # Test set operation
        result = redis_service.set("test_key", "test_value", ttl=60)
        mock_set.assert_called_once()
        
        # Test get operation
        mock_get.return_value = "test_value"
        value = redis_service.get("test_key")
        assert value == "test_value"
    
    def test_redis_json_handling(self, mock_redis):
        """Test Redis JSON serialization/deserialization"""
        mock_set, mock_get, mock_health = mock_redis
        
        test_data = {"key": "value", "number": 123}
        
        # Test setting JSON data
        redis_service.set("json_key", test_data)
        mock_set.assert_called_with("json_key", json.dumps(test_data), ttl=None)
        
        # Test getting JSON data
        mock_get.return_value = json.dumps(test_data)
        result = redis_service.get("json_key")
        assert result == test_data
    
    def test_redis_retry_logic(self, mock_redis):
        """Test Redis retry logic on failures"""
        mock_set, mock_get, mock_health = mock_redis
        
        # Mock Redis connection error
        mock_set.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception):
            redis_service.set("test_key", "test_value")


class TestDatabaseModels:
    """Test SQLAlchemy database models"""
    
    def test_intersection_model(self, setup_database, sample_intersection):
        """Test intersection model creation and relationships"""
        db = TestingSessionLocal()
        
        # Create intersection
        intersection = Intersection(**sample_intersection)
        db.add(intersection)
        db.commit()
        db.refresh(intersection)
        
        assert intersection.id == "test-intersection-1"
        assert intersection.name == "Test Intersection"
        assert intersection.status == "operational"
        assert len(intersection.lanes) == 4
    
    def test_traffic_data_model(self, setup_database, sample_intersection, sample_traffic_data):
        """Test traffic data model creation"""
        db = TestingSessionLocal()
        
        # Create intersection first
        intersection = Intersection(**sample_intersection)
        db.add(intersection)
        db.commit()
        
        # Create traffic data
        traffic_data = TrafficData(
            intersection_id=sample_traffic_data["intersection_id"],
            timestamp=datetime.fromtimestamp(sample_traffic_data["timestamp"]),
            lane_counts=sample_traffic_data["lane_counts"],
            avg_speed=sample_traffic_data["avg_speed"],
            weather_condition=sample_traffic_data["weather_condition"],
            vehicle_types=sample_traffic_data["vehicle_types"],
            confidence_score=sample_traffic_data["confidence_score"]
        )
        
        db.add(traffic_data)
        db.commit()
        db.refresh(traffic_data)
        
        assert traffic_data.intersection_id == "test-intersection-1"
        assert traffic_data.lane_counts["north_lane"] == 15
        assert traffic_data.avg_speed == 25.5
    
    def test_signal_timing_model(self, setup_database, sample_intersection):
        """Test signal timing model creation"""
        db = TestingSessionLocal()
        
        # Create intersection first
        intersection = Intersection(**sample_intersection)
        db.add(intersection)
        db.commit()
        
        # Create signal timing
        signal_timing = SignalTiming(
            intersection_id="test-intersection-1",
            lane="north_lane",
            duration=30,
            state="green",
            priority=2,
            is_active=True
        )
        
        db.add(signal_timing)
        db.commit()
        db.refresh(signal_timing)
        
        assert signal_timing.intersection_id == "test-intersection-1"
        assert signal_timing.lane == "north_lane"
        assert signal_timing.duration == 30
        assert signal_timing.state == "green"
        assert signal_timing.is_active is True


class TestAPIRoutes:
    """Test API route functionality"""
    
    def test_health_check_endpoint(self, mock_redis):
        """Test health check endpoint"""
        client = TestClient(app)
        
        response = client.get("/api/v1/health/")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "components" in data
    
    def test_traffic_ingest_endpoint(self, setup_database, sample_intersection, sample_traffic_data, mock_redis):
        """Test traffic data ingestion endpoint"""
        client = TestClient(app)
        
        # Create intersection first
        db = TestingSessionLocal()
        intersection = Intersection(**sample_intersection)
        db.add(intersection)
        db.commit()
        db.close()
        
        # Test traffic data ingestion
        response = client.post("/api/v1/traffic/ingest", json=sample_traffic_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "record_id" in data["data"]
    
    def test_traffic_status_endpoint(self, setup_database, sample_intersection, mock_redis):
        """Test traffic status endpoint"""
        client = TestClient(app)
        
        # Create intersection first
        db = TestingSessionLocal()
        intersection = Intersection(**sample_intersection)
        db.add(intersection)
        db.commit()
        db.close()
        
        # Test traffic status retrieval
        response = client.get("/api/v1/traffic/status/test-intersection-1")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
    
    def test_signal_optimization_endpoint(self, setup_database, sample_intersection, mock_redis):
        """Test signal optimization endpoint"""
        client = TestClient(app)
        
        # Create intersection first
        db = TestingSessionLocal()
        intersection = Intersection(**sample_intersection)
        db.add(intersection)
        db.commit()
        db.close()
        
        # Test signal optimization
        optimization_data = {
            "intersection_id": "test-intersection-1",
            "optimized_timings": [
                {
                    "lane": "north_lane",
                    "duration": 30,
                    "state": "green",
                    "priority": 2
                }
            ],
            "confidence_score": 0.9,
            "expected_improvement": 15.0,
            "algorithm_used": "ai_optimizer"
        }
        
        response = client.put("/api/v1/signals/optimize/test-intersection-1", json=optimization_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "optimization_id" in data["data"]


class TestMiddleware:
    """Test middleware functionality"""
    
    def test_rate_limiting_middleware(self, mock_redis):
        """Test rate limiting middleware"""
        from middleware.rate_limiting import RateLimitingMiddleware
        
        middleware = RateLimitingMiddleware(
            app=app,
            requests_per_minute=10,
            window_size=60
        )
        
        assert middleware.max_requests == 10
        assert middleware.window_seconds == 60
    
    def test_request_timing_middleware(self):
        """Test request timing middleware"""
        from middleware.request_timing import RequestTimingMiddleware
        
        middleware = RequestTimingMiddleware(app=app)
        assert middleware.skip_paths is not None
        assert "/health" in middleware.skip_paths


class TestConfiguration:
    """Test configuration management"""
    
    def test_settings_loading(self):
        """Test settings loading from environment"""
        assert settings.api.title == "Smart Traffic Management API"
        assert settings.api.version == "1.0.0"
        assert settings.database.url is not None
        assert settings.redis.host == "localhost"
    
    def test_database_settings(self):
        """Test database configuration"""
        assert settings.database.pool_size == 10
        assert settings.database.max_overflow == 20
        assert settings.database.pool_timeout == 30
    
    def test_redis_settings(self):
        """Test Redis configuration"""
        assert settings.redis.port == 6379
        assert settings.redis.max_connections == 20
        assert settings.redis.socket_timeout == 5
    
    def test_api_settings(self):
        """Test API configuration"""
        assert settings.api.host == "0.0.0.0"
        assert settings.api.port == 8000
        assert settings.api.rate_limit_requests == 100
        assert settings.api.rate_limit_window == 60


class TestErrorHandlingIntegration:
    """Test error handling integration with FastAPI"""
    
    def test_validation_error_response(self, setup_database, sample_intersection):
        """Test validation error response format"""
        client = TestClient(app)
        
        # Create intersection first
        db = TestingSessionLocal()
        intersection = Intersection(**sample_intersection)
        db.add(intersection)
        db.commit()
        db.close()
        
        # Test with invalid data
        invalid_data = {
            "intersection_id": "",  # Invalid empty ID
            "timestamp": int(datetime.now(timezone.utc).timestamp()),
            "lane_counts": {"north_lane": 15}
        }
        
        response = client.post("/api/v1/traffic/ingest", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_not_found_error_response(self, setup_database, mock_redis):
        """Test not found error response"""
        client = TestClient(app)
        
        # Test with non-existent intersection
        response = client.get("/api/v1/traffic/status/non-existent-intersection")
        assert response.status_code == 404
        
        data = response.json()
        assert "error_code" in data["detail"]
        assert data["detail"]["error_code"] == "NOT_FOUND"


class TestPerformance:
    """Test performance-related functionality"""
    
    def test_database_connection_pooling(self, setup_database):
        """Test database connection pooling"""
        from database.connection import create_database_engine
        
        engine = create_database_engine()
        assert engine.pool.size() > 0
        assert engine.pool.checkedin() >= 0
    
    def test_redis_connection_pooling(self, mock_redis):
        """Test Redis connection pooling"""
        mock_set, mock_get, mock_health = mock_redis
        
        # Test multiple operations
        for i in range(10):
            redis_service.set(f"key_{i}", f"value_{i}")
        
        # Verify all operations were called
        assert mock_set.call_count == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
