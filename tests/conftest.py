"""
Pytest configuration and shared fixtures for Smart Traffic Management System tests
"""

import asyncio
import pytest
import tempfile
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Generator
from unittest.mock import Mock, patch

import httpx
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Import application components
from src.backend.main import app as backend_app
from src.backend.database.connection import get_db
from src.backend.database.models import Base
from src.ml_engine.signal_optimizer import SignalOptimizer
from src.simulation.sumo_integration.sumo_integration_manager import SumoIntegrationManager
from src.orchestration.system_orchestrator import SystemOrchestrator
from src.orchestration.data_flow_manager import DataFlowManager
from src.orchestration.monitoring_system import MonitoringSystem

# Test database URL
TEST_DATABASE_URL = "sqlite:///./test_traffic.db"

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_db():
    """Create test database."""
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()
    
    backend_app.dependency_overrides[get_db] = override_get_db
    yield TestingSessionLocal()
    
    # Cleanup
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client(test_db):
    """Create test client for backend API."""
    with TestClient(backend_app) as test_client:
        yield test_client

@pytest.fixture
def auth_headers():
    """Create authentication headers for API tests."""
    return {"Authorization": "Bearer test-jwt-token"}

@pytest.fixture
def sample_traffic_light_data():
    """Sample traffic light data for testing."""
    return {
        "id": "test_light_1",
        "name": "Test Intersection Light",
        "location": {"lat": 40.7128, "lng": -74.0060},
        "status": "normal",
        "current_phase": 0,
        "phase_duration": 30,
        "program": "adaptive",
        "vehicle_count": 15,
        "waiting_time": 45.5
    }

@pytest.fixture
def sample_vehicle_data():
    """Sample vehicle data for testing."""
    return {
        "id": "test_vehicle_1",
        "type": "passenger",
        "position": {"lat": 40.7128, "lng": -74.0060},
        "speed": 25.5,
        "lane": "north_approach_0",
        "route": ["north_approach", "center_junction", "south_exit"],
        "waiting_time": 5.2,
        "co2_emission": 0.1,
        "fuel_consumption": 0.05
    }

@pytest.fixture
def sample_intersection_data():
    """Sample intersection data for testing."""
    return {
        "id": "test_intersection_1",
        "name": "Test Intersection",
        "location": {"lat": 40.7128, "lng": -74.0060},
        "traffic_lights": ["test_light_1", "test_light_2"],
        "total_vehicles": 23,
        "waiting_vehicles": 8,
        "average_speed": 22.5,
        "throughput": 45
    }

@pytest.fixture
def sample_performance_metrics():
    """Sample performance metrics for testing."""
    return {
        "timestamp": datetime.now().isoformat(),
        "total_vehicles": 156,
        "running_vehicles": 142,
        "waiting_vehicles": 14,
        "total_waiting_time": 234.5,
        "average_speed": 28.3,
        "total_co2_emission": 45.2,
        "total_fuel_consumption": 23.1,
        "throughput": 89
    }

@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    mock_redis = Mock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.delete.return_value = True
    mock_redis.lpush.return_value = True
    mock_redis.lpop.return_value = None
    mock_redis.llen.return_value = 0
    return mock_redis

@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection for testing."""
    mock_ws = Mock()
    mock_ws.send.return_value = None
    mock_ws.recv.return_value = json.dumps({"type": "test", "data": {}})
    return mock_ws

@pytest.fixture
def sample_config():
    """Sample system configuration for testing."""
    return {
        "system": {
            "name": "Smart Traffic Management System",
            "version": "2.1.0",
            "environment": "test",
            "log_level": "DEBUG"
        },
        "components": {
            "backend": {
                "port": 8000,
                "health_check_url": "http://localhost:8000/health",
                "max_restarts": 3
            },
            "ml_optimizer": {
                "port": 8001,
                "health_check_url": "http://localhost:8001/health",
                "max_restarts": 3
            },
            "sumo_simulation": {
                "port": 8002,
                "health_check_url": "http://localhost:8002/health",
                "max_restarts": 3
            },
            "frontend": {
                "port": 3000,
                "health_check_url": "http://localhost:3000",
                "max_restarts": 3
            }
        },
        "monitoring": {
            "health_check_interval": 10,
            "metrics_collection_interval": 30,
            "alert_thresholds": {
                "cpu_usage": 80,
                "memory_usage": 85,
                "response_time": 5.0
            }
        }
    }

@pytest.fixture
def temp_config_file(sample_config):
    """Create temporary configuration file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(sample_config, f)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    os.unlink(temp_file)

@pytest.fixture
def mock_system_orchestrator(temp_config_file):
    """Mock system orchestrator for testing."""
    with patch('src.orchestration.system_orchestrator.SystemOrchestrator') as mock:
        mock_instance = Mock()
        mock_instance.config = temp_config_file
        mock_instance.components = {}
        mock_instance.running = False
        mock_instance.get_system_status.return_value = {
            "system": {"name": "Test System", "version": "2.1.0"},
            "components": {},
            "monitoring": {}
        }
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_data_flow_manager():
    """Mock data flow manager for testing."""
    with patch('src.orchestration.data_flow_manager.DataFlowManager') as mock:
        mock_instance = Mock()
        mock_instance.get_metrics.return_value = {
            "metrics": {
                "messages_processed": 100,
                "messages_failed": 5,
                "processing_time": 1.5
            }
        }
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_monitoring_system():
    """Mock monitoring system for testing."""
    with patch('src.orchestration.monitoring_system.MonitoringSystem') as mock:
        mock_instance = Mock()
        mock_instance.get_system_status.return_value = {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "disk_usage": 23.1
            },
            "component_status": {
                "backend": "healthy",
                "ml_optimizer": "healthy",
                "sumo_simulation": "healthy"
            },
            "active_alerts": 0
        }
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_ml_optimizer():
    """Mock ML optimizer for testing."""
    with patch('src.ml_engine.signal_optimizer.SignalOptimizer') as mock:
        mock_instance = Mock()
        mock_instance.optimize.return_value = {
            "optimized_phases": [30, 25, 35, 20],
            "efficiency_improvement": 0.15,
            "waiting_time_reduction": 12.5
        }
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_sumo_integration():
    """Mock SUMO integration for testing."""
    with patch('src.simulation.sumo_integration.sumo_integration_manager.SumoIntegrationManager') as mock:
        mock_instance = Mock()
        mock_instance.get_traffic_data.return_value = {
            "vehicles": [],
            "traffic_lights": [],
            "intersections": []
        }
        mock_instance.get_performance_metrics.return_value = {
            "total_vehicles": 100,
            "average_speed": 25.5,
            "throughput": 45
        }
        mock.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def sample_api_response():
    """Sample API response for testing."""
    return {
        "success": True,
        "data": {"message": "Test response"},
        "timestamp": datetime.now().isoformat()
    }

@pytest.fixture
def sample_error_response():
    """Sample error response for testing."""
    return {
        "success": False,
        "error": "Test error message",
        "timestamp": datetime.now().isoformat()
    }

@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for external API calls."""
    with patch('httpx.AsyncClient') as mock:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "ok"}
        mock_client.get.return_value = mock_response
        mock_client.post.return_value = mock_response
        mock.return_value.__aenter__.return_value = mock_client
        yield mock_client

@pytest.fixture
def test_data_generator():
    """Generate test data for various scenarios."""
    def generate_traffic_data(count: int = 10):
        data = []
        for i in range(count):
            data.append({
                "id": f"test_vehicle_{i}",
                "type": "passenger",
                "position": {
                    "lat": 40.7128 + (i * 0.001),
                    "lng": -74.0060 + (i * 0.001)
                },
                "speed": 20 + (i * 2),
                "lane": f"lane_{i % 4}",
                "waiting_time": i * 5.5
            })
        return data
    
    return generate_traffic_data

@pytest.fixture
def performance_test_data():
    """Generate performance test data."""
    def generate_metrics(hours: int = 24):
        metrics = []
        base_time = datetime.now() - timedelta(hours=hours)
        
        for i in range(hours * 60):  # Every minute
            timestamp = base_time + timedelta(minutes=i)
            metrics.append({
                "timestamp": timestamp.isoformat(),
                "total_vehicles": 100 + (i % 50),
                "running_vehicles": 80 + (i % 30),
                "waiting_vehicles": 20 + (i % 20),
                "average_speed": 25 + (i % 10),
                "throughput": 40 + (i % 20)
            })
        return metrics
    
    return generate_metrics

# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as a performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file paths."""
    for item in items:
        # Add markers based on file path
        if "unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
        elif "performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        
        # Mark slow tests
        if "load" in item.nodeid or "stress" in item.nodeid:
            item.add_marker(pytest.mark.slow)
