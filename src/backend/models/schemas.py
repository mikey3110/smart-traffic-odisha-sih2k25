"""
Advanced Pydantic schemas with comprehensive validation
for the Smart Traffic Management System
"""

from pydantic import BaseModel, Field, validator, root_validator
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timezone
import re
from enum import Enum

from .enums import (
    TrafficLightState, LaneDirection, VehicleType, WeatherCondition,
    OptimizationStatus, SystemStatus, EventType, Priority,
    MIN_SIGNAL_TIMING, MAX_SIGNAL_TIMING, DEFAULT_SIGNAL_TIMING,
    MIN_CONFIDENCE_SCORE, MAX_CONFIDENCE_SCORE, DEFAULT_CONFIDENCE_SCORE,
    MIN_SPEED, MAX_SPEED, MIN_VEHICLE_COUNT, MAX_VEHICLE_COUNT,
    INTERSECTION_ID_PATTERN, VALID_LANE_NAMES
)


class BaseSchema(BaseModel):
    """Base schema with common configuration"""
    
    class Config:
        # Use enum values instead of enum objects
        use_enum_values = True
        # Validate assignment
        validate_assignment = True
        # Allow population by field name or alias
        allow_population_by_field_name = True
        # Generate JSON schema
        schema_extra = {
            "examples": []
        }


class TrafficDataSchema(BaseSchema):
    """Schema for traffic data with comprehensive validation"""
    
    intersection_id: str = Field(
        ...,
        description="Unique identifier for the intersection",
        min_length=1,
        max_length=50,
        regex=INTERSECTION_ID_PATTERN
    )
    
    timestamp: int = Field(
        ...,
        description="Unix timestamp when data was collected",
        gt=0
    )
    
    lane_counts: Dict[str, int] = Field(
        ...,
        description="Vehicle counts per lane",
        min_items=1
    )
    
    avg_speed: Optional[float] = Field(
        None,
        description="Average speed in km/h",
        ge=MIN_SPEED,
        le=MAX_SPEED
    )
    
    weather_condition: Optional[WeatherCondition] = Field(
        None,
        description="Current weather condition"
    )
    
    vehicle_types: Optional[Dict[str, int]] = Field(
        None,
        description="Count of different vehicle types"
    )
    
    confidence_score: Optional[float] = Field(
        None,
        description="Confidence score for the data",
        ge=MIN_CONFIDENCE_SCORE,
        le=MAX_CONFIDENCE_SCORE
    )
    
    @validator('timestamp')
    def validate_timestamp(cls, v):
        """Validate timestamp is not too old or in the future"""
        now = datetime.now(timezone.utc).timestamp()
        # Allow data up to 1 hour old or 5 minutes in the future
        if v < now - 3600:
            raise ValueError('Timestamp is too old (more than 1 hour)')
        if v > now + 300:
            raise ValueError('Timestamp is in the future (more than 5 minutes)')
        return v
    
    @validator('lane_counts')
    def validate_lane_counts(cls, v):
        """Validate lane counts are within reasonable bounds"""
        for lane, count in v.items():
            if lane not in VALID_LANE_NAMES:
                raise ValueError(f'Invalid lane name: {lane}')
            if not (MIN_VEHICLE_COUNT <= count <= MAX_VEHICLE_COUNT):
                raise ValueError(f'Vehicle count for {lane} must be between {MIN_VEHICLE_COUNT} and {MAX_VEHICLE_COUNT}')
        return v
    
    @validator('vehicle_types')
    def validate_vehicle_types(cls, v):
        """Validate vehicle type counts"""
        if v:
            for vehicle_type, count in v.items():
                if vehicle_type not in [vt.value for vt in VehicleType]:
                    raise ValueError(f'Invalid vehicle type: {vehicle_type}')
                if not (MIN_VEHICLE_COUNT <= count <= MAX_VEHICLE_COUNT):
                    raise ValueError(f'Vehicle count for {vehicle_type} must be between {MIN_VEHICLE_COUNT} and {MAX_VEHICLE_COUNT}')
        return v
    
    @root_validator
    def validate_data_consistency(cls, values):
        """Validate overall data consistency"""
        lane_counts = values.get('lane_counts', {})
        vehicle_types = values.get('vehicle_types', {})
        
        # If both lane_counts and vehicle_types are provided, check consistency
        if lane_counts and vehicle_types:
            total_lane_count = sum(lane_counts.values())
            total_vehicle_count = sum(vehicle_types.values())
            
            # Allow some variance due to different counting methods
            if abs(total_lane_count - total_vehicle_count) > total_lane_count * 0.1:
                raise ValueError('Lane counts and vehicle type counts are inconsistent')
        
        return values


class SignalTimingSchema(BaseSchema):
    """Schema for signal timing configuration"""
    
    lane: str = Field(
        ...,
        description="Lane identifier",
        regex=r"^[a-zA-Z0-9_-]+$"
    )
    
    duration: int = Field(
        ...,
        description="Signal duration in seconds",
        ge=MIN_SIGNAL_TIMING,
        le=MAX_SIGNAL_TIMING
    )
    
    state: TrafficLightState = Field(
        ...,
        description="Traffic light state"
    )
    
    priority: Priority = Field(
        default=Priority.NORMAL,
        description="Priority level for this timing"
    )


class SignalOptimizationSchema(BaseSchema):
    """Schema for signal optimization requests and responses"""
    
    intersection_id: str = Field(
        ...,
        description="Intersection identifier",
        min_length=1,
        max_length=50,
        regex=INTERSECTION_ID_PATTERN
    )
    
    optimized_timings: List[SignalTimingSchema] = Field(
        ...,
        description="Optimized signal timings for each lane",
        min_items=1
    )
    
    confidence_score: float = Field(
        ...,
        description="Confidence score for the optimization",
        ge=MIN_CONFIDENCE_SCORE,
        le=MAX_CONFIDENCE_SCORE
    )
    
    expected_improvement: Optional[float] = Field(
        None,
        description="Expected improvement percentage",
        ge=0.0,
        le=100.0
    )
    
    algorithm_used: Optional[str] = Field(
        None,
        description="Algorithm used for optimization",
        max_length=100
    )
    
    optimization_time: Optional[float] = Field(
        None,
        description="Time taken for optimization in seconds",
        ge=0.0
    )
    
    @validator('optimized_timings')
    def validate_timings_consistency(cls, v):
        """Validate that all lanes have valid timings"""
        if not v:
            raise ValueError('At least one timing configuration is required')
        
        # Check for duplicate lanes
        lanes = [timing.lane for timing in v]
        if len(lanes) != len(set(lanes)):
            raise ValueError('Duplicate lane configurations found')
        
        # Validate total cycle time is reasonable
        total_duration = sum(timing.duration for timing in v)
        if total_duration > 600:  # 10 minutes max cycle
            raise ValueError('Total cycle time exceeds maximum allowed duration')
        
        return v


class TrafficStatusSchema(BaseSchema):
    """Schema for traffic status responses"""
    
    intersection_id: str = Field(
        ...,
        description="Intersection identifier"
    )
    
    current_counts: Dict[str, int] = Field(
        ...,
        description="Current vehicle counts per lane"
    )
    
    current_timings: Dict[str, int] = Field(
        ...,
        description="Current signal timings per lane"
    )
    
    last_updated: datetime = Field(
        ...,
        description="Last update timestamp"
    )
    
    optimization_status: OptimizationStatus = Field(
        ...,
        description="Current optimization status"
    )
    
    system_status: SystemStatus = Field(
        default=SystemStatus.OPERATIONAL,
        description="System status"
    )
    
    avg_wait_time: Optional[float] = Field(
        None,
        description="Average wait time in seconds",
        ge=0.0
    )
    
    throughput: Optional[float] = Field(
        None,
        description="Vehicles processed per hour",
        ge=0.0
    )


class HealthCheckSchema(BaseSchema):
    """Schema for health check responses"""
    
    status: SystemStatus = Field(
        ...,
        description="Overall system status"
    )
    
    timestamp: datetime = Field(
        ...,
        description="Health check timestamp"
    )
    
    version: str = Field(
        ...,
        description="API version"
    )
    
    components: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Status of individual components"
    )
    
    uptime: Optional[float] = Field(
        None,
        description="System uptime in seconds"
    )
    
    memory_usage: Optional[float] = Field(
        None,
        description="Memory usage percentage",
        ge=0.0,
        le=100.0
    )
    
    cpu_usage: Optional[float] = Field(
        None,
        description="CPU usage percentage",
        ge=0.0,
        le=100.0
    )


class IntersectionSchema(BaseSchema):
    """Schema for intersection information"""
    
    id: str = Field(
        ...,
        description="Intersection identifier",
        regex=INTERSECTION_ID_PATTERN
    )
    
    name: Optional[str] = Field(
        None,
        description="Human-readable intersection name",
        max_length=200
    )
    
    location: Optional[Dict[str, float]] = Field(
        None,
        description="GPS coordinates (lat, lng)"
    )
    
    lanes: List[str] = Field(
        ...,
        description="List of lane identifiers",
        min_items=1
    )
    
    status: SystemStatus = Field(
        default=SystemStatus.OPERATIONAL,
        description="Current intersection status"
    )
    
    created_at: Optional[datetime] = Field(
        None,
        description="Creation timestamp"
    )
    
    updated_at: Optional[datetime] = Field(
        None,
        description="Last update timestamp"
    )
    
    @validator('location')
    def validate_location(cls, v):
        """Validate GPS coordinates"""
        if v:
            if 'lat' not in v or 'lng' not in v:
                raise ValueError('Location must contain lat and lng keys')
            if not (-90 <= v['lat'] <= 90):
                raise ValueError('Latitude must be between -90 and 90')
            if not (-180 <= v['lng'] <= 180):
                raise ValueError('Longitude must be between -180 and 180')
        return v


class APIResponseSchema(BaseSchema):
    """Standard API response schema"""
    
    status: str = Field(
        ...,
        description="Response status (success, error)"
    )
    
    message: str = Field(
        ...,
        description="Response message"
    )
    
    data: Optional[Any] = Field(
        None,
        description="Response data"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Response timestamp"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Request identifier for tracking"
    )


class ErrorResponseSchema(BaseSchema):
    """Error response schema"""
    
    error_code: str = Field(
        ...,
        description="Error code"
    )
    
    message: str = Field(
        ...,
        description="Error message"
    )
    
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details"
    )
    
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Error timestamp"
    )
    
    request_id: Optional[str] = Field(
        None,
        description="Request identifier for tracking"
    )


class PaginationSchema(BaseSchema):
    """Pagination schema for list responses"""
    
    page: int = Field(
        default=1,
        description="Current page number",
        ge=1
    )
    
    per_page: int = Field(
        default=20,
        description="Items per page",
        ge=1,
        le=100
    )
    
    total: int = Field(
        ...,
        description="Total number of items",
        ge=0
    )
    
    pages: int = Field(
        ...,
        description="Total number of pages",
        ge=0
    )
    
    has_next: bool = Field(
        ...,
        description="Whether there is a next page"
    )
    
    has_prev: bool = Field(
        ...,
        description="Whether there is a previous page"
    )


class PaginatedResponseSchema(BaseSchema):
    """Paginated response schema"""
    
    items: List[Any] = Field(
        ...,
        description="List of items"
    )
    
    pagination: PaginationSchema = Field(
        ...,
        description="Pagination information"
    )


