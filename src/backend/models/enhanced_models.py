"""
Enhanced Pydantic models with advanced validation for Smart Traffic Management System
Includes field validation, custom validators, and comprehensive data models
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Optional, Union, Literal
from datetime import datetime, timezone
from enum import Enum
import re
from .exceptions import DataValidationError


class WeatherCondition(str, Enum):
    """Weather condition enumeration"""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    FOGGY = "foggy"
    STORMY = "stormy"
    SNOWY = "snowy"


class VehicleType(str, Enum):
    """Vehicle type enumeration"""
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    EMERGENCY = "emergency"


class CongestionLevel(str, Enum):
    """Traffic congestion level enumeration"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    SEVERE = "severe"


class SignalStatus(str, Enum):
    """Traffic signal status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class OptimizationAlgorithm(str, Enum):
    """Signal optimization algorithm enumeration"""
    AI_OPTIMIZER = "ai_optimizer"
    GENETIC_ALGORITHM = "genetic_algorithm"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    HEURISTIC = "heuristic"


class LaneDirection(str, Enum):
    """Lane direction enumeration"""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    NORTHEAST = "northeast"
    NORTHWEST = "northwest"
    SOUTHEAST = "southeast"
    SOUTHWEST = "southwest"


class BaseTrafficModel(BaseModel):
    """Base model for traffic-related data with common validation"""
    
    class Config:
        # Enable validation on assignment
        validate_assignment = True
        # Use enum values instead of names
        use_enum_values = True
        # Allow population by field name
        allow_population_by_field_name = True
        # Validate default values
        validate_default = True
        # Extra fields are forbidden
        extra = "forbid"


class IntersectionIdValidator(BaseModel):
    """Mixin for intersection ID validation"""
    
    @validator('intersection_id')
    def validate_intersection_id(cls, v):
        """Validate intersection ID format"""
        if not v:
            raise ValueError("Intersection ID cannot be empty")
        
        # Check format: should be alphanumeric with optional hyphens/underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Intersection ID must contain only alphanumeric characters, hyphens, and underscores")
        
        # Check length
        if len(v) < 3 or len(v) > 50:
            raise ValueError("Intersection ID must be between 3 and 50 characters")
        
        return v.lower()  # Normalize to lowercase


class TimestampValidator(BaseModel):
    """Mixin for timestamp validation"""
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp is reasonable"""
        if not isinstance(v, (int, float)):
            raise ValueError("Timestamp must be a number")
        
        # Check if timestamp is in reasonable range (last 10 years to future 1 year)
        current_time = datetime.now(timezone.utc).timestamp()
        min_time = current_time - (10 * 365 * 24 * 3600)  # 10 years ago
        max_time = current_time + (1 * 365 * 24 * 3600)   # 1 year from now
        
        if v < min_time or v > max_time:
            raise ValueError("Timestamp is outside reasonable range")
        
        return int(v)


class LaneCountsValidator(BaseModel):
    """Mixin for lane counts validation"""
    
    @validator('lane_counts')
    def validate_lane_counts(cls, v):
        """Validate lane counts data"""
        if not v:
            raise ValueError("Lane counts cannot be empty")
        
        if not isinstance(v, dict):
            raise ValueError("Lane counts must be a dictionary")
        
        # Validate each lane count
        for lane, count in v.items():
            if not isinstance(lane, str):
                raise ValueError("Lane names must be strings")
            
            if not isinstance(count, int):
                raise ValueError("Lane counts must be integers")
            
            if count < 0:
                raise ValueError("Lane counts cannot be negative")
            
            if count > 1000:  # Reasonable upper limit
                raise ValueError("Lane counts seem unreasonably high")
        
        return v


class SpeedValidator(BaseModel):
    """Mixin for speed validation"""
    
    @validator('avg_speed')
    def validate_avg_speed(cls, v):
        """Validate average speed"""
        if v is not None:
            if not isinstance(v, (int, float)):
                raise ValueError("Average speed must be a number")
            
            if v < 0:
                raise ValueError("Average speed cannot be negative")
            
            if v > 200:  # Reasonable upper limit in km/h
                raise ValueError("Average speed seems unreasonably high")
        
        return v


class TrafficData(BaseTrafficModel, IntersectionIdValidator, TimestampValidator, LaneCountsValidator, SpeedValidator):
    """Enhanced traffic data model with comprehensive validation"""
    
    intersection_id: str = Field(..., description="Unique identifier for the intersection")
    timestamp: int = Field(..., description="Unix timestamp of the data collection")
    lane_counts: Dict[str, int] = Field(..., description="Vehicle counts per lane")
    avg_speed: Optional[float] = Field(None, ge=0, le=200, description="Average speed in km/h")
    weather_condition: Optional[WeatherCondition] = Field(None, description="Current weather condition")
    vehicle_types: Optional[Dict[VehicleType, int]] = Field(None, description="Count of different vehicle types")
    congestion_level: Optional[CongestionLevel] = Field(None, description="Current congestion level")
    temperature: Optional[float] = Field(None, ge=-50, le=60, description="Temperature in Celsius")
    humidity: Optional[float] = Field(None, ge=0, le=100, description="Humidity percentage")
    visibility: Optional[float] = Field(None, ge=0, le=100, description="Visibility in km")
    
    @validator('vehicle_types')
    def validate_vehicle_types(cls, v):
        """Validate vehicle types data"""
        if v is not None:
            for vehicle_type, count in v.items():
                if not isinstance(count, int):
                    raise ValueError("Vehicle type counts must be integers")
                if count < 0:
                    raise ValueError("Vehicle type counts cannot be negative")
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        """Validate temperature range"""
        if v is not None and (v < -50 or v > 60):
            raise ValueError("Temperature must be between -50°C and 60°C")
        return v
    
    @validator('humidity')
    def validate_humidity(cls, v):
        """Validate humidity percentage"""
        if v is not None and (v < 0 or v > 100):
            raise ValueError("Humidity must be between 0% and 100%")
        return v
    
    @validator('visibility')
    def validate_visibility(cls, v):
        """Validate visibility range"""
        if v is not None and (v < 0 or v > 100):
            raise ValueError("Visibility must be between 0km and 100km")
        return v
    
    @root_validator
    def validate_congestion_consistency(cls, values):
        """Validate that congestion level is consistent with traffic counts"""
        lane_counts = values.get('lane_counts')
        congestion_level = values.get('congestion_level')
        
        if lane_counts and congestion_level:
            total_vehicles = sum(lane_counts.values())
            
            # Define congestion thresholds
            if total_vehicles < 10 and congestion_level in ['high', 'severe']:
                raise ValueError("Congestion level seems inconsistent with low vehicle count")
            elif total_vehicles > 50 and congestion_level == 'low':
                raise ValueError("Congestion level seems inconsistent with high vehicle count")
        
        return values


class SignalTiming(BaseModel):
    """Model for individual signal timing"""
    
    lane: str = Field(..., description="Lane identifier")
    green_duration: int = Field(..., ge=5, le=300, description="Green light duration in seconds")
    yellow_duration: int = Field(..., ge=2, le=10, description="Yellow light duration in seconds")
    red_duration: int = Field(..., ge=5, le=300, description="Red light duration in seconds")
    
    @validator('lane')
    def validate_lane(cls, v):
        """Validate lane identifier"""
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError("Lane identifier must contain only alphanumeric characters, hyphens, and underscores")
        return v.lower()


class SignalOptimization(BaseTrafficModel, IntersectionIdValidator):
    """Enhanced signal optimization model with comprehensive validation"""
    
    intersection_id: str = Field(..., description="Unique identifier for the intersection")
    optimized_timings: Dict[str, SignalTiming] = Field(..., description="Optimized timing for each lane")
    confidence_score: Optional[float] = Field(None, ge=0, le=1, description="Confidence score (0-1)")
    expected_improvement: Optional[float] = Field(None, ge=0, le=100, description="Expected improvement percentage")
    algorithm_used: OptimizationAlgorithm = Field(OptimizationAlgorithm.AI_OPTIMIZER, description="Algorithm used for optimization")
    optimization_metadata: Optional[Dict[str, Union[str, int, float]]] = Field(None, description="Additional optimization metadata")
    
    @validator('optimized_timings')
    def validate_optimized_timings(cls, v):
        """Validate optimized timings structure"""
        if not v:
            raise ValueError("Optimized timings cannot be empty")
        
        # Check that all lanes have valid timings
        for lane, timing in v.items():
            if not isinstance(timing, dict):
                raise ValueError(f"Timing for lane '{lane}' must be a dictionary")
            
            required_fields = ['green_duration', 'yellow_duration', 'red_duration']
            for field in required_fields:
                if field not in timing:
                    raise ValueError(f"Timing for lane '{lane}' missing required field: {field}")
        
        return v
    
    @root_validator
    def validate_cycle_consistency(cls, values):
        """Validate that signal cycle is consistent across lanes"""
        optimized_timings = values.get('optimized_timings')
        
        if optimized_timings and len(optimized_timings) > 1:
            # Calculate cycle times for each lane
            cycle_times = []
            for timing in optimized_timings.values():
                cycle_time = timing['green_duration'] + timing['yellow_duration'] + timing['red_duration']
                cycle_times.append(cycle_time)
            
            # Check if cycle times are similar (within 10% tolerance)
            if cycle_times:
                avg_cycle = sum(cycle_times) / len(cycle_times)
                for cycle_time in cycle_times:
                    if abs(cycle_time - avg_cycle) / avg_cycle > 0.1:
                        raise ValueError("Signal cycle times are inconsistent across lanes")
        
        return values


class TrafficStatus(BaseTrafficModel, IntersectionIdValidator):
    """Enhanced traffic status model with comprehensive validation"""
    
    intersection_id: str = Field(..., description="Unique identifier for the intersection")
    current_counts: Dict[str, int] = Field(..., description="Current vehicle counts per lane")
    current_timings: Dict[str, SignalTiming] = Field(..., description="Current signal timings")
    last_updated: datetime = Field(..., description="Last update timestamp")
    optimization_status: SignalStatus = Field(..., description="Current optimization status")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="Performance metrics")
    alerts: Optional[List[str]] = Field(None, description="Active alerts")
    
    @validator('last_updated')
    def validate_last_updated(cls, v):
        """Validate last updated timestamp"""
        if isinstance(v, str):
            try:
                v = datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                raise ValueError("Invalid datetime format for last_updated")
        
        # Check if timestamp is not too far in the future
        now = datetime.now(timezone.utc)
        if v > now:
            raise ValueError("Last updated timestamp cannot be in the future")
        
        return v
    
    @validator('performance_metrics')
    def validate_performance_metrics(cls, v):
        """Validate performance metrics"""
        if v is not None:
            valid_metrics = ['efficiency', 'throughput', 'wait_time', 'queue_length']
            for metric in v.keys():
                if metric not in valid_metrics:
                    raise ValueError(f"Invalid performance metric: {metric}")
            
            for metric, value in v.items():
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Performance metric '{metric}' must be a number")
                if value < 0:
                    raise ValueError(f"Performance metric '{metric}' cannot be negative")
        
        return v


class APIResponse(BaseModel):
    """Standard API response model"""
    
    status: Literal["success", "error"] = Field(..., description="Response status")
    message: Optional[str] = Field(None, description="Response message")
    data: Optional[Union[Dict, List]] = Field(None, description="Response data")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Standard error response model"""
    
    status: Literal["error"] = Field(..., description="Response status")
    error: Dict[str, Union[str, Dict]] = Field(..., description="Error details")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    
    status: Literal["healthy", "unhealthy"] = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Check timestamp")
    version: str = Field(..., description="Application version")
    services: Dict[str, Dict[str, Union[str, bool]]] = Field(..., description="Service health status")
    uptime: Optional[float] = Field(None, description="Application uptime in seconds")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

