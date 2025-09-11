"""
Integration tests for API endpoints and database interactions
"""

import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.backend.main import app
from src.backend.database.models import Base, TrafficLight, Vehicle, Intersection, PerformanceMetrics
from src.backend.database.connection import get_db

# Test database setup
TEST_DATABASE_URL = "sqlite:///./test_integration.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture(scope="function")
def test_db():
    """Create test database for each test"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def client(test_db):
    """Create test client with database dependency override"""
    def override_get_db():
        try:
            yield test_db
        finally:
            pass
    
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

class TestTrafficLightIntegration:
    """Integration tests for traffic light operations"""
    
    def test_create_and_retrieve_traffic_light(self, client, test_db):
        """Test creating and retrieving a traffic light"""
        # Create traffic light data
        traffic_light_data = {
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
        
        # Create traffic light
        response = client.post("/api/traffic/lights", json=traffic_light_data)
        assert response.status_code == 201
        created_data = response.json()
        assert created_data["success"] is True
        assert created_data["data"]["id"] == traffic_light_data["id"]
        
        # Retrieve traffic light
        response = client.get(f"/api/traffic/lights/{traffic_light_data['id']}")
        assert response.status_code == 200
        retrieved_data = response.json()
        assert retrieved_data["success"] is True
        assert retrieved_data["data"]["name"] == traffic_light_data["name"]
        
        # Verify in database
        db_light = test_db.query(TrafficLight).filter(TrafficLight.id == traffic_light_data["id"]).first()
        assert db_light is not None
        assert db_light.name == traffic_light_data["name"]
    
    def test_update_traffic_light(self, client, test_db):
        """Test updating a traffic light"""
        # Create initial traffic light
        traffic_light = TrafficLight(
            id="test_light_2",
            name="Initial Light",
            location={"lat": 40.7128, "lng": -74.0060},
            status="normal",
            current_phase=0,
            phase_duration=30,
            program="adaptive",
            vehicle_count=10,
            waiting_time=30.0,
            last_update=datetime.now()
        )
        test_db.add(traffic_light)
        test_db.commit()
        
        # Update traffic light
        update_data = {
            "name": "Updated Light",
            "phase_duration": 45,
            "program": "fixed"
        }
        
        response = client.put("/api/traffic/lights/test_light_2", json=update_data)
        assert response.status_code == 200
        updated_data = response.json()
        assert updated_data["success"] is True
        
        # Verify update in database
        db_light = test_db.query(TrafficLight).filter(TrafficLight.id == "test_light_2").first()
        assert db_light.name == "Updated Light"
        assert db_light.phase_duration == 45
        assert db_light.program == "fixed"
    
    def test_control_traffic_light(self, client, test_db):
        """Test controlling a traffic light"""
        # Create traffic light
        traffic_light = TrafficLight(
            id="test_light_3",
            name="Controllable Light",
            location={"lat": 40.7128, "lng": -74.0060},
            status="normal",
            current_phase=0,
            phase_duration=30,
            program="adaptive",
            vehicle_count=5,
            waiting_time=15.0,
            last_update=datetime.now()
        )
        test_db.add(traffic_light)
        test_db.commit()
        
        # Control traffic light
        control_data = {
            "phase": 2,
            "duration": 25
        }
        
        response = client.post("/api/traffic/lights/test_light_3/control", json=control_data)
        assert response.status_code == 200
        control_result = response.json()
        assert control_result["success"] is True
        
        # Verify control in database
        db_light = test_db.query(TrafficLight).filter(TrafficLight.id == "test_light_3").first()
        assert db_light.current_phase == 2
        assert db_light.phase_duration == 25
    
    def test_get_traffic_lights_with_filters(self, client, test_db):
        """Test getting traffic lights with filters"""
        # Create multiple traffic lights
        lights_data = [
            {
                "id": "light_1",
                "name": "Light 1",
                "status": "normal",
                "current_phase": 0,
                "phase_duration": 30,
                "vehicle_count": 10,
                "waiting_time": 20.0
            },
            {
                "id": "light_2",
                "name": "Light 2",
                "status": "maintenance",
                "current_phase": 2,
                "phase_duration": 25,
                "vehicle_count": 5,
                "waiting_time": 15.0
            }
        ]
        
        for light_data in lights_data:
            traffic_light = TrafficLight(
                id=light_data["id"],
                name=light_data["name"],
                location={"lat": 40.7128, "lng": -74.0060},
                status=light_data["status"],
                current_phase=light_data["current_phase"],
                phase_duration=light_data["phase_duration"],
                program="adaptive",
                vehicle_count=light_data["vehicle_count"],
                waiting_time=light_data["waiting_time"],
                last_update=datetime.now()
            )
            test_db.add(traffic_light)
        test_db.commit()
        
        # Test filter by status
        response = client.get("/api/traffic/lights?status=normal")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 1
        assert data["data"][0]["status"] == "normal"
        
        # Test filter by phase
        response = client.get("/api/traffic/lights?phase=2")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 1
        assert data["data"][0]["current_phase"] == 2

class TestVehicleIntegration:
    """Integration tests for vehicle operations"""
    
    def test_create_and_retrieve_vehicle(self, client, test_db):
        """Test creating and retrieving a vehicle"""
        # Create vehicle data
        vehicle_data = {
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
        
        # Create vehicle
        response = client.post("/api/traffic/vehicles", json=vehicle_data)
        assert response.status_code == 201
        created_data = response.json()
        assert created_data["success"] is True
        assert created_data["data"]["id"] == vehicle_data["id"]
        
        # Retrieve vehicle
        response = client.get(f"/api/traffic/vehicles/{vehicle_data['id']}")
        assert response.status_code == 200
        retrieved_data = response.json()
        assert retrieved_data["success"] is True
        assert retrieved_data["data"]["type"] == vehicle_data["type"]
        
        # Verify in database
        db_vehicle = test_db.query(Vehicle).filter(Vehicle.id == vehicle_data["id"]).first()
        assert db_vehicle is not None
        assert db_vehicle.type == vehicle_data["type"]
    
    def test_get_vehicles_with_filters(self, client, test_db):
        """Test getting vehicles with filters"""
        # Create multiple vehicles
        vehicles_data = [
            {
                "id": "vehicle_1",
                "type": "passenger",
                "speed": 25.5,
                "waiting_time": 5.2
            },
            {
                "id": "vehicle_2",
                "type": "truck",
                "speed": 18.0,
                "waiting_time": 12.8
            },
            {
                "id": "vehicle_3",
                "type": "passenger",
                "speed": 30.0,
                "waiting_time": 3.5
            }
        ]
        
        for vehicle_data in vehicles_data:
            vehicle = Vehicle(
                id=vehicle_data["id"],
                type=vehicle_data["type"],
                position={"lat": 40.7128, "lng": -74.0060},
                speed=vehicle_data["speed"],
                lane="test_lane",
                route=["test_route"],
                waiting_time=vehicle_data["waiting_time"],
                co2_emission=0.1,
                fuel_consumption=0.05,
                timestamp=datetime.now()
            )
            test_db.add(vehicle)
        test_db.commit()
        
        # Test filter by type
        response = client.get("/api/traffic/vehicles?type=passenger")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 2
        assert all(v["type"] == "passenger" for v in data["data"])
        
        # Test filter by speed range
        response = client.get("/api/traffic/vehicles?min_speed=20&max_speed=25")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 1
        assert data["data"][0]["speed"] == 25.5

class TestIntersectionIntegration:
    """Integration tests for intersection operations"""
    
    def test_create_and_retrieve_intersection(self, client, test_db):
        """Test creating and retrieving an intersection"""
        # Create intersection data
        intersection_data = {
            "id": "test_intersection_1",
            "name": "Test Intersection",
            "location": {"lat": 40.7128, "lng": -74.0060},
            "traffic_lights": ["light_1", "light_2"],
            "total_vehicles": 23,
            "waiting_vehicles": 8,
            "average_speed": 22.5,
            "throughput": 45
        }
        
        # Create intersection
        response = client.post("/api/traffic/intersections", json=intersection_data)
        assert response.status_code == 201
        created_data = response.json()
        assert created_data["success"] is True
        assert created_data["data"]["id"] == intersection_data["id"]
        
        # Retrieve intersection
        response = client.get(f"/api/traffic/intersections/{intersection_data['id']}")
        assert response.status_code == 200
        retrieved_data = response.json()
        assert retrieved_data["success"] is True
        assert retrieved_data["data"]["name"] == intersection_data["name"]
        
        # Verify in database
        db_intersection = test_db.query(Intersection).filter(Intersection.id == intersection_data["id"]).first()
        assert db_intersection is not None
        assert db_intersection.name == intersection_data["name"]
    
    def test_update_intersection_metrics(self, client, test_db):
        """Test updating intersection metrics"""
        # Create intersection
        intersection = Intersection(
            id="test_intersection_2",
            name="Test Intersection 2",
            location={"lat": 40.7128, "lng": -74.0060},
            traffic_lights=["light_1", "light_2"],
            total_vehicles=10,
            waiting_vehicles=3,
            average_speed=25.0,
            throughput=30,
            last_update=datetime.now()
        )
        test_db.add(intersection)
        test_db.commit()
        
        # Update metrics
        update_data = {
            "total_vehicles": 35,
            "waiting_vehicles": 12,
            "average_speed": 20.5,
            "throughput": 55
        }
        
        response = client.put("/api/traffic/intersections/test_intersection_2/metrics", json=update_data)
        assert response.status_code == 200
        updated_data = response.json()
        assert updated_data["success"] is True
        
        # Verify update in database
        db_intersection = test_db.query(Intersection).filter(Intersection.id == "test_intersection_2").first()
        assert db_intersection.total_vehicles == 35
        assert db_intersection.waiting_vehicles == 12
        assert db_intersection.average_speed == 20.5
        assert db_intersection.throughput == 55

class TestPerformanceMetricsIntegration:
    """Integration tests for performance metrics operations"""
    
    def test_create_and_retrieve_metrics(self, client, test_db):
        """Test creating and retrieving performance metrics"""
        # Create metrics data
        metrics_data = {
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
        
        # Create metrics
        response = client.post("/api/traffic/metrics", json=metrics_data)
        assert response.status_code == 201
        created_data = response.json()
        assert created_data["success"] is True
        
        # Retrieve metrics
        response = client.get("/api/traffic/metrics")
        assert response.status_code == 200
        retrieved_data = response.json()
        assert retrieved_data["success"] is True
        assert len(retrieved_data["data"]) == 1
        assert retrieved_data["data"][0]["total_vehicles"] == metrics_data["total_vehicles"]
        
        # Verify in database
        db_metrics = test_db.query(PerformanceMetrics).first()
        assert db_metrics is not None
        assert db_metrics.total_vehicles == metrics_data["total_vehicles"]
    
    def test_get_metrics_with_time_range(self, client, test_db):
        """Test getting metrics with time range filter"""
        # Create metrics with different timestamps
        base_time = datetime.now() - timedelta(hours=2)
        
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=base_time + timedelta(minutes=i*30),
                total_vehicles=100 + i*10,
                running_vehicles=80 + i*8,
                waiting_vehicles=20 + i*2,
                total_waiting_time=200 + i*20,
                average_speed=25 + i,
                total_co2_emission=40 + i*2,
                total_fuel_consumption=20 + i,
                throughput=40 + i*5
            )
            test_db.add(metrics)
        test_db.commit()
        
        # Test time range filter
        start_time = (base_time + timedelta(minutes=30)).isoformat()
        end_time = (base_time + timedelta(minutes=90)).isoformat()
        
        response = client.get(f"/api/traffic/metrics?start_time={start_time}&end_time={end_time}")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 3  # 3 metrics in the time range

class TestDataConsistency:
    """Test data consistency across operations"""
    
    def test_traffic_light_vehicle_count_consistency(self, client, test_db):
        """Test consistency between traffic light and vehicle counts"""
        # Create traffic light
        traffic_light = TrafficLight(
            id="consistency_light",
            name="Consistency Light",
            location={"lat": 40.7128, "lng": -74.0060},
            status="normal",
            current_phase=0,
            phase_duration=30,
            program="adaptive",
            vehicle_count=0,
            waiting_time=0.0,
            last_update=datetime.now()
        )
        test_db.add(traffic_light)
        test_db.commit()
        
        # Create vehicles for this intersection
        vehicles_data = [
            {"id": "v1", "type": "passenger", "speed": 25.0, "waiting_time": 5.0},
            {"id": "v2", "type": "truck", "speed": 18.0, "waiting_time": 10.0},
            {"id": "v3", "type": "passenger", "speed": 30.0, "waiting_time": 3.0}
        ]
        
        for vehicle_data in vehicles_data:
            vehicle = Vehicle(
                id=vehicle_data["id"],
                type=vehicle_data["type"],
                position={"lat": 40.7128, "lng": -74.0060},
                speed=vehicle_data["speed"],
                lane="test_lane",
                route=["test_route"],
                waiting_time=vehicle_data["waiting_time"],
                co2_emission=0.1,
                fuel_consumption=0.05,
                timestamp=datetime.now()
            )
            test_db.add(vehicle)
        test_db.commit()
        
        # Update traffic light vehicle count
        update_data = {"vehicle_count": len(vehicles_data)}
        response = client.put("/api/traffic/lights/consistency_light", json=update_data)
        assert response.status_code == 200
        
        # Verify consistency
        db_light = test_db.query(TrafficLight).filter(TrafficLight.id == "consistency_light").first()
        assert db_light.vehicle_count == len(vehicles_data)
    
    def test_intersection_metrics_consistency(self, client, test_db):
        """Test consistency between intersection and traffic light metrics"""
        # Create traffic lights
        lights_data = [
            {"id": "light_1", "vehicle_count": 10, "waiting_time": 20.0},
            {"id": "light_2", "vehicle_count": 15, "waiting_time": 30.0},
            {"id": "light_3", "vehicle_count": 8, "waiting_time": 15.0}
        ]
        
        for light_data in lights_data:
            traffic_light = TrafficLight(
                id=light_data["id"],
                name=f"Light {light_data['id']}",
                location={"lat": 40.7128, "lng": -74.0060},
                status="normal",
                current_phase=0,
                phase_duration=30,
                program="adaptive",
                vehicle_count=light_data["vehicle_count"],
                waiting_time=light_data["waiting_time"],
                last_update=datetime.now()
            )
            test_db.add(traffic_light)
        test_db.commit()
        
        # Create intersection
        intersection = Intersection(
            id="consistency_intersection",
            name="Consistency Intersection",
            location={"lat": 40.7128, "lng": -74.0060},
            traffic_lights=[light["id"] for light in lights_data],
            total_vehicles=0,  # Will be calculated
            waiting_vehicles=0,  # Will be calculated
            average_speed=0.0,  # Will be calculated
            throughput=0,  # Will be calculated
            last_update=datetime.now()
        )
        test_db.add(intersection)
        test_db.commit()
        
        # Update intersection metrics based on traffic lights
        total_vehicles = sum(light["vehicle_count"] for light in lights_data)
        total_waiting_time = sum(light["waiting_time"] for light in lights_data)
        
        update_data = {
            "total_vehicles": total_vehicles,
            "waiting_vehicles": total_vehicles,  # Simplified
            "average_speed": 25.0,
            "throughput": total_vehicles * 2
        }
        
        response = client.put("/api/traffic/intersections/consistency_intersection/metrics", json=update_data)
        assert response.status_code == 200
        
        # Verify consistency
        db_intersection = test_db.query(Intersection).filter(Intersection.id == "consistency_intersection").first()
        assert db_intersection.total_vehicles == total_vehicles

class TestErrorHandlingIntegration:
    """Test error handling in integration scenarios"""
    
    def test_database_constraint_violation(self, client, test_db):
        """Test handling of database constraint violations"""
        # Create traffic light with duplicate ID
        traffic_light1 = TrafficLight(
            id="duplicate_id",
            name="Light 1",
            location={"lat": 40.7128, "lng": -74.0060},
            status="normal",
            current_phase=0,
            phase_duration=30,
            program="adaptive",
            vehicle_count=10,
            waiting_time=20.0,
            last_update=datetime.now()
        )
        test_db.add(traffic_light1)
        test_db.commit()
        
        # Try to create another traffic light with same ID
        traffic_light_data = {
            "id": "duplicate_id",
            "name": "Light 2",
            "location": {"lat": 40.7128, "lng": -74.0060},
            "status": "normal",
            "current_phase": 0,
            "phase_duration": 30,
            "program": "adaptive",
            "vehicle_count": 5,
            "waiting_time": 15.0
        }
        
        response = client.post("/api/traffic/lights", json=traffic_light_data)
        assert response.status_code == 409  # Conflict
        error_data = response.json()
        assert error_data["success"] is False
        assert "already exists" in error_data["error"].lower()
    
    def test_foreign_key_constraint_violation(self, client, test_db):
        """Test handling of foreign key constraint violations"""
        # Try to create intersection with non-existent traffic light
        intersection_data = {
            "id": "invalid_intersection",
            "name": "Invalid Intersection",
            "location": {"lat": 40.7128, "lng": -74.0060},
            "traffic_lights": ["non_existent_light"],
            "total_vehicles": 10,
            "waiting_vehicles": 3,
            "average_speed": 25.0,
            "throughput": 30
        }
        
        response = client.post("/api/traffic/intersections", json=intersection_data)
        assert response.status_code == 400  # Bad Request
        error_data = response.json()
        assert error_data["success"] is False
        assert "not found" in error_data["error"].lower()
    
    def test_concurrent_update_handling(self, client, test_db):
        """Test handling of concurrent updates"""
        # Create traffic light
        traffic_light = TrafficLight(
            id="concurrent_light",
            name="Concurrent Light",
            location={"lat": 40.7128, "lng": -74.0060},
            status="normal",
            current_phase=0,
            phase_duration=30,
            program="adaptive",
            vehicle_count=10,
            waiting_time=20.0,
            last_update=datetime.now()
        )
        test_db.add(traffic_light)
        test_db.commit()
        
        # Simulate concurrent updates
        update1 = {"phase_duration": 45}
        update2 = {"phase_duration": 60}
        
        # First update
        response1 = client.put("/api/traffic/lights/concurrent_light", json=update1)
        assert response1.status_code == 200
        
        # Second update (should handle gracefully)
        response2 = client.put("/api/traffic/lights/concurrent_light", json=update2)
        assert response2.status_code == 200
        
        # Verify final state
        db_light = test_db.query(TrafficLight).filter(TrafficLight.id == "concurrent_light").first()
        assert db_light.phase_duration == 60  # Last update should win
