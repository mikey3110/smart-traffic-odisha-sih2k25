"""
Exception handling module for Smart Traffic Management System
"""

from .custom_exceptions import (
    TrafficManagementException,
    ValidationError,
    DatabaseError,
    RedisError,
    IntersectionNotFoundError,
    SignalOptimizationError,
    TrafficDataError,
    RateLimitExceededError,
    ServiceUnavailableError,
    ConfigurationError,
    create_http_exception,
    create_validation_error,
    create_not_found_error,
    create_rate_limit_error
)

from .handlers import (
    register_exception_handlers,
    create_error_response
)

__all__ = [
    # Custom exceptions
    "TrafficManagementException",
    "ValidationError",
    "DatabaseError",
    "RedisError",
    "IntersectionNotFoundError",
    "SignalOptimizationError",
    "TrafficDataError",
    "RateLimitExceededError",
    "ServiceUnavailableError",
    "ConfigurationError",
    
    # Helper functions
    "create_http_exception",
    "create_validation_error",
    "create_not_found_error",
    "create_rate_limit_error",
    
    # Handlers
    "register_exception_handlers",
    "create_error_response"
]



