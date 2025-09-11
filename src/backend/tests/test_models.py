"""
Unit tests for Pydantic models and validation
"""

import pytest
from datetime import datetime, timezone
from pydantic import ValidationError

from models.schemas import (
    TrafficDataSchema, SignalTimingSchema, SignalOptimizationSchema,
    TrafficStatusSchema, HealthCheckSchema, IntersectionSchema,
    APIResponseSchema, ErrorResponseSchema, PaginationSchema
)
from models.enums import (
    TrafficLightState, LaneDirection, VehicleType, WeatherCondition,
    OptimizationStatus, SystemStatus, EventType, Priority
)


class TestTrafficDataSchema:
    """Test TrafficDataSchema validation"""
    
    def test_valid_traffic_data(self):
        """Test valid traffic data"""
        data = {
            "intersection_id": "junction-1",
            "timestamp": int(datetime.now().timestamp()),
            "lane_counts": {"north_lane": 10, "south_lane": 8, "east_lane": 12, "west_lane": 6},
            "avg_speed": 25.5,
            "weather_condition": "clear",
            "vehicle_types": {"car": 20, "truck": 5, "motorcycle": 3},
            "confidence_score": 0.85
        }
        
        traffic_data = TrafficDataSchema(**data)
        assert traffic_data.intersection_id == "junction-1"
        assert traffic_data.lane_counts["north_lane"] == 10
        assert traffic_data.avg_speed == 25.5
        assert traffic_data.weather_condition == "clear"
        assert traffic_data.confidence_score == 0.85
    
    def test_invalid_intersection_id(self):
        """Test invalid intersection ID"""
        data = {
            "intersection_id": "",  # Empty string
            "timestamp": int(datetime.now().timestamp()),
            "lane_counts": {"north_lane": 10}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TrafficDataSchema(**data)
        
        assert "min_length" in str(exc_info.value)
    
    def test_invalid_timestamp(self):
        """Test invalid timestamp"""
        data = {
            "intersection_id": "junction-1",
            "timestamp": -1,  # Negative timestamp
            "lane_counts": {"north_lane": 10}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TrafficDataSchema(**data)
        
        assert "greater than 0" in str(exc_info.value)
    
    def test_invalid_lane_counts(self):
        """Test invalid lane counts"""
        data = {
            "intersection_id": "junction-1",
            "timestamp": int(datetime.now().timestamp()),
            "lane_counts": {"invalid_lane": 10}  # Invalid lane name
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TrafficDataSchema(**data)
        
        assert "Invalid lane name" in str(exc_info.value)
    
    def test_invalid_confidence_score(self):
        """Test invalid confidence score"""
        data = {
            "intersection_id": "junction-1",
            "timestamp": int(datetime.now().timestamp()),
            "lane_counts": {"north_lane": 10},
            "confidence_score": 1.5  # Invalid confidence score
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TrafficDataSchema(**data)
        
        assert "less than or equal to 1" in str(exc_info.value)
    
    def test_old_timestamp(self):
        """Test timestamp that's too old"""
        old_timestamp = int((datetime.now() - timezone.utc.localize(datetime(2020, 1, 1))).timestamp())
        data = {
            "intersection_id": "junction-1",
            "timestamp": old_timestamp,
            "lane_counts": {"north_lane": 10}
        }
        
        with pytest.raises(ValidationError) as exc_info:
            TrafficDataSchema(**data)
        
        assert "too old" in str(exc_info.value)


class TestSignalTimingSchema:
    """Test SignalTimingSchema validation"""
    
    def test_valid_signal_timing(self):
        """Test valid signal timing"""
        data = {
            "lane": "north_lane",
            "duration": 30,
            "state": "green",
            "priority": 2
        }
        
        signal_timing = SignalTimingSchema(**data)
        assert signal_timing.lane == "north_lane"
        assert signal_timing.duration == 30
        assert signal_timing.state == "green"
        assert signal_timing.priority == 2
    
    def test_invalid_duration(self):
        """Test invalid duration"""
        data = {
            "lane": "north_lane",
            "duration": 2,  # Too short
            "state": "green"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            SignalTimingSchema(**data)
        
        assert "greater than or equal to 5" in str(exc_info.value)
    
    def test_invalid_state(self):
        """Test invalid state"""
        data = {
            "lane": "north_lane",
            "duration": 30,
            "state": "invalid_state"
        }
        
        with pytest.raises(ValidationError) as exc_info:
            SignalTimingSchema(**data)
        
        assert "not a valid enumeration member" in str(exc_info.value)


class TestSignalOptimizationSchema:
    """Test SignalOptimizationSchema validation"""
    
    def test_valid_optimization(self):
        """Test valid optimization"""
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
        
        optimization = SignalOptimizationSchema(**data)
        assert optimization.intersection_id == "junction-1"
        assert len(optimization.optimized_timings) == 2
        assert optimization.confidence_score == 0.85
        assert optimization.expected_improvement == 15.0
    
    def test_duplicate_lanes(self):
        """Test duplicate lane configurations"""
        data = {
            "intersection_id": "junction-1",
            "optimized_timings": [
                {"lane": "north_lane", "duration": 30, "state": "green"},
                {"lane": "north_lane", "duration": 45, "state": "green"}  # Duplicate lane
            ],
            "confidence_score": 0.85
        }
        
        with pytest.raises(ValidationError) as exc_info:
            SignalOptimizationSchema(**data)
        
        assert "Duplicate lane configurations" in str(exc_info.value)
    
    def test_excessive_cycle_time(self):
        """Test excessive total cycle time"""
        data = {
            "intersection_id": "junction-1",
            "optimized_timings": [
                {"lane": "north_lane", "duration": 300, "state": "green"},  # 5 minutes
                {"lane": "south_lane", "duration": 300, "state": "green"},  # 5 minutes
                {"lane": "east_lane": 300, "state": "green"},  # 5 minutes
                {"lane": "west_lane": 300, "state": "green"}   # 5 minutes
            ],
            "confidence_score": 0.85
        }
        
        with pytest.raises(ValidationError) as exc_info:
            SignalOptimizationSchema(**data)
        
        assert "exceeds maximum allowed duration" in str(exc_info.value)


class TestHealthCheckSchema:
    """Test HealthCheckSchema validation"""
    
    def test_valid_health_check(self):
        """Test valid health check"""
        data = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc),
            "version": "1.0.0",
            "components": {
                "database": {"status": "connected"},
                "redis": {"status": "connected"}
            }
        }
        
        health_check = HealthCheckSchema(**data)
        assert health_check.status == "healthy"
        assert health_check.version == "1.0.0"
        assert "database" in health_check.components
    
    def test_invalid_memory_usage(self):
        """Test invalid memory usage"""
        data = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc),
            "version": "1.0.0",
            "components": {},
            "memory_usage": 150.0  # Invalid memory usage > 100%
        }
        
        with pytest.raises(ValidationError) as exc_info:
            HealthCheckSchema(**data)
        
        assert "less than or equal to 100" in str(exc_info.value)


class TestIntersectionSchema:
    """Test IntersectionSchema validation"""
    
    def test_valid_intersection(self):
        """Test valid intersection"""
        data = {
            "id": "junction-1",
            "name": "Main Street & First Avenue",
            "location": {"lat": 40.7128, "lng": -74.0060},
            "lanes": ["north_lane", "south_lane", "east_lane", "west_lane"],
            "status": "operational"
        }
        
        intersection = IntersectionSchema(**data)
        assert intersection.id == "junction-1"
        assert intersection.name == "Main Street & First Avenue"
        assert intersection.location["lat"] == 40.7128
        assert len(intersection.lanes) == 4
    
    def test_invalid_location(self):
        """Test invalid GPS coordinates"""
        data = {
            "id": "junction-1",
            "lanes": ["north_lane"],
            "location": {"lat": 200.0, "lng": -74.0060}  # Invalid latitude
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IntersectionSchema(**data)
        
        assert "Latitude must be between -90 and 90" in str(exc_info.value)
    
    def test_missing_location_keys(self):
        """Test missing location keys"""
        data = {
            "id": "junction-1",
            "lanes": ["north_lane"],
            "location": {"latitude": 40.7128, "longitude": -74.0060}  # Wrong keys
        }
        
        with pytest.raises(ValidationError) as exc_info:
            IntersectionSchema(**data)
        
        assert "must contain lat and lng keys" in str(exc_info.value)


class TestPaginationSchema:
    """Test PaginationSchema validation"""
    
    def test_valid_pagination(self):
        """Test valid pagination"""
        data = {
            "page": 1,
            "per_page": 20,
            "total": 100,
            "pages": 5,
            "has_next": True,
            "has_prev": False
        }
        
        pagination = PaginationSchema(**data)
        assert pagination.page == 1
        assert pagination.per_page == 20
        assert pagination.total == 100
        assert pagination.pages == 5
        assert pagination.has_next is True
        assert pagination.has_prev is False
    
    def test_invalid_page(self):
        """Test invalid page number"""
        data = {
            "page": 0,  # Invalid page number
            "per_page": 20,
            "total": 100,
            "pages": 5,
            "has_next": True,
            "has_prev": False
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PaginationSchema(**data)
        
        assert "greater than or equal to 1" in str(exc_info.value)