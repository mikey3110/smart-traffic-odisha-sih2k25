"""
Models module for Smart Traffic Management System
"""

from .schemas import (
    BaseSchema,
    TrafficDataSchema,
    SignalTimingSchema,
    SignalOptimizationSchema,
    TrafficStatusSchema,
    HealthCheckSchema,
    IntersectionSchema,
    APIResponseSchema,
    ErrorResponseSchema,
    PaginationSchema,
    PaginatedResponseSchema
)

from .enums import (
    TrafficLightState,
    LaneDirection,
    VehicleType,
    WeatherCondition,
    OptimizationStatus,
    SystemStatus,
    EventType,
    Priority,
    MIN_SIGNAL_TIMING,
    MAX_SIGNAL_TIMING,
    DEFAULT_SIGNAL_TIMING,
    MIN_CONFIDENCE_SCORE,
    MAX_CONFIDENCE_SCORE,
    DEFAULT_CONFIDENCE_SCORE,
    MIN_SPEED,
    MAX_SPEED,
    MIN_VEHICLE_COUNT,
    MAX_VEHICLE_COUNT,
    INTERSECTION_ID_PATTERN,
    VALID_LANE_NAMES,
    DEFAULT_RATE_LIMIT,
    DEFAULT_RATE_WINDOW,
    CACHE_TTL_SHORT,
    CACHE_TTL_MEDIUM,
    CACHE_TTL_LONG,
    CACHE_TTL_VERY_LONG
)

__all__ = [
    # Schemas
    "BaseSchema",
    "TrafficDataSchema",
    "SignalTimingSchema",
    "SignalOptimizationSchema",
    "TrafficStatusSchema",
    "HealthCheckSchema",
    "IntersectionSchema",
    "APIResponseSchema",
    "ErrorResponseSchema",
    "PaginationSchema",
    "PaginatedResponseSchema",
    
    # Enums
    "TrafficLightState",
    "LaneDirection",
    "VehicleType",
    "WeatherCondition",
    "OptimizationStatus",
    "SystemStatus",
    "EventType",
    "Priority",
    
    # Constants
    "MIN_SIGNAL_TIMING",
    "MAX_SIGNAL_TIMING",
    "DEFAULT_SIGNAL_TIMING",
    "MIN_CONFIDENCE_SCORE",
    "MAX_CONFIDENCE_SCORE",
    "DEFAULT_CONFIDENCE_SCORE",
    "MIN_SPEED",
    "MAX_SPEED",
    "MIN_VEHICLE_COUNT",
    "MAX_VEHICLE_COUNT",
    "INTERSECTION_ID_PATTERN",
    "VALID_LANE_NAMES",
    "DEFAULT_RATE_LIMIT",
    "DEFAULT_RATE_WINDOW",
    "CACHE_TTL_SHORT",
    "CACHE_TTL_MEDIUM",
    "CACHE_TTL_LONG",
    "CACHE_TTL_VERY_LONG"
]