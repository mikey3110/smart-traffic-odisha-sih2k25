"""
Custom exception classes for the Smart Traffic Management System
Provides specific error types for different failure scenarios
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException, status


class TrafficManagementException(Exception):
    """Base exception for all traffic management errors"""
    
    def __init__(
        self, 
        message: str, 
        error_code: str = "TRAFFIC_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(TrafficManagementException):
    """Raised when data validation fails"""
    
    def __init__(self, message: str, field: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details={**details, "field": field} if field else details
        )


class DatabaseError(TrafficManagementException):
    """Raised when database operations fail"""
    
    def __init__(self, message: str, operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details={**details, "operation": operation} if operation else details
        )


class RedisError(TrafficManagementException):
    """Raised when Redis operations fail"""
    
    def __init__(self, message: str, operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="REDIS_ERROR",
            details={**details, "operation": operation} if operation else details
        )


class IntersectionNotFoundError(TrafficManagementException):
    """Raised when an intersection is not found"""
    
    def __init__(self, intersection_id: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=f"Intersection '{intersection_id}' not found",
            error_code="INTERSECTION_NOT_FOUND",
            details={**details, "intersection_id": intersection_id}
        )


class SignalOptimizationError(TrafficManagementException):
    """Raised when signal optimization fails"""
    
    def __init__(self, message: str, intersection_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="SIGNAL_OPTIMIZATION_ERROR",
            details={**details, "intersection_id": intersection_id} if intersection_id else details
        )


class TrafficDataError(TrafficManagementException):
    """Raised when traffic data processing fails"""
    
    def __init__(self, message: str, intersection_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="TRAFFIC_DATA_ERROR",
            details={**details, "intersection_id": intersection_id} if intersection_id else details
        )


class RateLimitExceededError(TrafficManagementException):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, message: str, retry_after: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details={**details, "retry_after": retry_after} if retry_after else details
        )


class ServiceUnavailableError(TrafficManagementException):
    """Raised when a service is unavailable"""
    
    def __init__(self, message: str, service: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE",
            details={**details, "service": service} if service else details
        )


class ConfigurationError(TrafficManagementException):
    """Raised when configuration is invalid"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details={**details, "config_key": config_key} if config_key else details
        )


# HTTP Exception mappings
EXCEPTION_TO_HTTP_STATUS = {
    ValidationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
    IntersectionNotFoundError: status.HTTP_404_NOT_FOUND,
    RateLimitExceededError: status.HTTP_429_TOO_MANY_REQUESTS,
    ServiceUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
    ConfigurationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    DatabaseError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    RedisError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    SignalOptimizationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    TrafficDataError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    TrafficManagementException: status.HTTP_500_INTERNAL_SERVER_ERROR,
}


def create_http_exception(exc: TrafficManagementException) -> HTTPException:
    """
    Convert a custom exception to an HTTPException
    """
    status_code = EXCEPTION_TO_HTTP_STATUS.get(type(exc), status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    return HTTPException(
        status_code=status_code,
        detail={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )


def create_validation_error(field: str, message: str, value: Any = None) -> ValidationError:
    """
    Create a validation error with standardized format
    """
    return ValidationError(
        message=f"Validation error in field '{field}': {message}",
        field=field,
        details={"invalid_value": value}
    )


def create_not_found_error(resource: str, resource_id: str) -> HTTPException:
    """
    Create a standardized not found error
    """
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={
            "error_code": "NOT_FOUND",
            "message": f"{resource} with ID '{resource_id}' not found",
            "details": {"resource": resource, "resource_id": resource_id}
        }
    )


def create_rate_limit_error(retry_after: int = 60) -> HTTPException:
    """
    Create a standardized rate limit error
    """
    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail={
            "error_code": "RATE_LIMIT_EXCEEDED",
            "message": "Rate limit exceeded. Please try again later.",
            "details": {"retry_after": retry_after}
        },
        headers={"Retry-After": str(retry_after)}
    )



