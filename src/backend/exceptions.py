"""
Custom exception classes for Smart Traffic Management System
Provides structured error handling with proper HTTP status codes
"""

from typing import Optional, Dict, Any
from fastapi import HTTPException


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


class DataValidationError(TrafficManagementException):
    """Raised when data validation fails"""
    
    def __init__(self, message: str, field: str = None, details: Optional[Dict[str, Any]] = None):
        self.field = field
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            details=details or {}
        )


class DatabaseError(TrafficManagementException):
    """Raised when database operations fail"""
    
    def __init__(self, message: str, operation: str = None, details: Optional[Dict[str, Any]] = None):
        self.operation = operation
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            details=details or {}
        )


class RedisConnectionError(TrafficManagementException):
    """Raised when Redis connection fails"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="REDIS_CONNECTION_ERROR",
            details=details or {}
        )


class RedisOperationError(TrafficManagementException):
    """Raised when Redis operations fail"""
    
    def __init__(self, message: str, operation: str = None, details: Optional[Dict[str, Any]] = None):
        self.operation = operation
        super().__init__(
            message=message,
            error_code="REDIS_OPERATION_ERROR",
            details=details or {}
        )


class IntersectionNotFoundError(TrafficManagementException):
    """Raised when intersection is not found"""
    
    def __init__(self, intersection_id: str, details: Optional[Dict[str, Any]] = None):
        self.intersection_id = intersection_id
        super().__init__(
            message=f"Intersection '{intersection_id}' not found",
            error_code="INTERSECTION_NOT_FOUND",
            details=details or {}
        )


class SignalOptimizationError(TrafficManagementException):
    """Raised when signal optimization fails"""
    
    def __init__(self, message: str, intersection_id: str = None, details: Optional[Dict[str, Any]] = None):
        self.intersection_id = intersection_id
        super().__init__(
            message=message,
            error_code="SIGNAL_OPTIMIZATION_ERROR",
            details=details or {}
        )


class RateLimitExceededError(TrafficManagementException):
    """Raised when rate limit is exceeded"""
    
    def __init__(self, message: str, retry_after: int = None, details: Optional[Dict[str, Any]] = None):
        self.retry_after = retry_after
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details or {}
        )


class ServiceUnavailableError(TrafficManagementException):
    """Raised when a service is unavailable"""
    
    def __init__(self, message: str, service: str = None, details: Optional[Dict[str, Any]] = None):
        self.service = service
        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE",
            details=details or {}
        )


class ConfigurationError(TrafficManagementException):
    """Raised when configuration is invalid"""
    
    def __init__(self, message: str, config_key: str = None, details: Optional[Dict[str, Any]] = None):
        self.config_key = config_key
        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            details=details or {}
        )


# HTTP Exception Mappers
def map_to_http_exception(exception: TrafficManagementException) -> HTTPException:
    """
    Map custom exceptions to appropriate HTTP exceptions
    
    Args:
        exception: Custom exception instance
    
    Returns:
        HTTPException with appropriate status code
    """
    
    status_code_mapping = {
        "VALIDATION_ERROR": 422,
        "DATABASE_ERROR": 500,
        "REDIS_CONNECTION_ERROR": 503,
        "REDIS_OPERATION_ERROR": 500,
        "INTERSECTION_NOT_FOUND": 404,
        "SIGNAL_OPTIMIZATION_ERROR": 500,
        "RATE_LIMIT_EXCEEDED": 429,
        "SERVICE_UNAVAILABLE": 503,
        "CONFIGURATION_ERROR": 500,
        "TRAFFIC_ERROR": 500
    }
    
    status_code = status_code_mapping.get(exception.error_code, 500)
    
    # Prepare error response
    error_response = {
        "error": {
            "code": exception.error_code,
            "message": exception.message,
            "details": exception.details
        }
    }
    
    # Add retry-after header for rate limiting
    headers = {}
    if isinstance(exception, RateLimitExceededError) and exception.retry_after:
        headers["Retry-After"] = str(exception.retry_after)
    
    return HTTPException(
        status_code=status_code,
        detail=error_response,
        headers=headers
    )


# Exception handlers for FastAPI
def create_exception_handlers():
    """Create exception handlers for FastAPI app"""
    
    from fastapi import Request
    from fastapi.responses import JSONResponse
    from config.logging_config import get_logger
    
    logger = get_logger(__name__)
    
    async def traffic_management_exception_handler(request: Request, exc: TrafficManagementException):
        """Handle custom traffic management exceptions"""
        logger.error(f"Traffic management error: {exc.message}", extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method
        })
        
        http_exc = map_to_http_exception(exc)
        return JSONResponse(
            status_code=http_exc.status_code,
            content=http_exc.detail,
            headers=http_exc.headers
        )
    
    async def validation_exception_handler(request: Request, exc: Exception):
        """Handle validation exceptions"""
        logger.error(f"Validation error: {str(exc)}", extra={
            "path": request.url.path,
            "method": request.method
        })
        
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request validation failed",
                    "details": {"validation_error": str(exc)}
                }
            }
        )
    
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unexpected error: {str(exc)}", extra={
            "path": request.url.path,
            "method": request.method,
            "exception_type": type(exc).__name__
        })
        
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred",
                    "details": {}
                }
            }
        )
    
    return {
        TrafficManagementException: traffic_management_exception_handler,
        Exception: general_exception_handler
    }

