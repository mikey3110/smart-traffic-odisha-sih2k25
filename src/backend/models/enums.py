"""
Enums and constants for the Smart Traffic Management System
"""

from enum import Enum, IntEnum
from typing import List


class TrafficLightState(str, Enum):
    """Traffic light states"""
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    FLASHING_RED = "flashing_red"
    FLASHING_YELLOW = "flashing_yellow"
    OFF = "off"


class LaneDirection(str, Enum):
    """Lane directions"""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"
    NORTHEAST = "northeast"
    NORTHWEST = "northwest"
    SOUTHEAST = "southeast"
    SOUTHWEST = "southwest"


class VehicleType(str, Enum):
    """Vehicle types for detection"""
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"


class WeatherCondition(str, Enum):
    """Weather conditions"""
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    FOGGY = "foggy"
    SNOWY = "snowy"
    STORMY = "stormy"


class OptimizationStatus(str, Enum):
    """Signal optimization status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SystemStatus(str, Enum):
    """System status indicators"""
    OPERATIONAL = "operational"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"
    ERROR = "error"


class EventType(str, Enum):
    """Event types for logging and monitoring"""
    TRAFFIC_DATA_RECEIVED = "traffic_data_received"
    SIGNAL_OPTIMIZED = "signal_optimized"
    INTERSECTION_STATUS_CHANGED = "intersection_status_changed"
    SYSTEM_ERROR = "system_error"
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    HEALTH_CHECK = "health_check"


class Priority(IntEnum):
    """Priority levels for operations"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


# Constants
MIN_SIGNAL_TIMING = 5  # Minimum signal timing in seconds
MAX_SIGNAL_TIMING = 300  # Maximum signal timing in seconds
DEFAULT_SIGNAL_TIMING = 30  # Default signal timing in seconds

MIN_CONFIDENCE_SCORE = 0.0
MAX_CONFIDENCE_SCORE = 1.0
DEFAULT_CONFIDENCE_SCORE = 0.8

MIN_SPEED = 0.0  # Minimum speed in km/h
MAX_SPEED = 200.0  # Maximum speed in km/h

MIN_VEHICLE_COUNT = 0
MAX_VEHICLE_COUNT = 1000

# Valid intersection ID patterns
INTERSECTION_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"

# Valid lane names
VALID_LANE_NAMES = [
    "north_lane", "south_lane", "east_lane", "west_lane",
    "northeast_lane", "northwest_lane", "southeast_lane", "southwest_lane",
    "left_turn", "right_turn", "straight", "u_turn"
]

# API rate limiting
DEFAULT_RATE_LIMIT = 100  # requests per minute
DEFAULT_RATE_WINDOW = 60  # seconds

# Cache TTL values (in seconds)
CACHE_TTL_SHORT = 60  # 1 minute
CACHE_TTL_MEDIUM = 300  # 5 minutes
CACHE_TTL_LONG = 3600  # 1 hour
CACHE_TTL_VERY_LONG = 86400  # 24 hours

