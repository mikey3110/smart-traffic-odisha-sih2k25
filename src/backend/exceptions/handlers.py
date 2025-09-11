"""
Exception handlers for the Smart Traffic Management System
Provides centralized error handling and response formatting
"""

from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
from typing import Union

from .custom_exceptions import (
    TrafficManagementException,
    create_http_exception,
    EXCEPTION_TO_HTTP_STATUS
)
from config.logging_config import get_logger

logger = get_logger("exception_handlers")


async def traffic_management_exception_handler(
    request: Request, 
    exc: TrafficManagementException
) -> JSONResponse:
    """
    Handle custom TrafficManagementException
    """
    logger.error(
        f"Traffic Management Exception: {exc.message}",
        extra={
            'request_id': getattr(request.state, 'request_id', 'unknown'),
            'error_code': exc.error_code,
            'details': exc.details,
            'path': str(request.url.path),
            'method': request.method
        }
    )
    
    http_exc = create_http_exception(exc)
    return JSONResponse(
        status_code=http_exc.status_code,
        content=http_exc.detail
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """
    Handle FastAPI HTTPException
    """
    logger.warning(
        f"HTTP Exception: {exc.detail}",
        extra={
            'request_id': getattr(request.state, 'request_id', 'unknown'),
            'status_code': exc.status_code,
            'path': str(request.url.path),
            'method': request.method
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": "HTTP_ERROR",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


async def validation_exception_handler(
    request: Request, 
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors
    """
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(loc) for loc in error["loc"])
        errors.append({
            "field": field,
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    logger.warning(
        f"Validation Error: {len(errors)} validation errors",
        extra={
            'request_id': getattr(request.state, 'request_id', 'unknown'),
            'validation_errors': errors,
            'path': str(request.url.path),
            'method': request.method
        }
    )
    
    return JSONResponse(
        status_code=422,
        content={
            "error_code": "VALIDATION_ERROR",
            "message": "Request validation failed",
            "details": {
                "validation_errors": errors,
                "total_errors": len(errors)
            }
        }
    )


async def starlette_http_exception_handler(
    request: Request, 
    exc: StarletteHTTPException
) -> JSONResponse:
    """
    Handle Starlette HTTPException
    """
    logger.warning(
        f"Starlette HTTP Exception: {exc.detail}",
        extra={
            'request_id': getattr(request.state, 'request_id', 'unknown'),
            'status_code': exc.status_code,
            'path': str(request.url.path),
            'method': request.method
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error_code": "HTTP_ERROR",
            "message": exc.detail,
            "status_code": exc.status_code
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle unexpected exceptions
    """
    logger.error(
        f"Unexpected Exception: {str(exc)}",
        extra={
            'request_id': getattr(request.state, 'request_id', 'unknown'),
            'exception_type': type(exc).__name__,
            'path': str(request.url.path),
            'method': request.method,
            'exc_info': True
        }
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "details": {
                "exception_type": type(exc).__name__,
                "request_id": getattr(request.state, 'request_id', 'unknown')
            }
        }
    )


def register_exception_handlers(app):
    """
    Register all exception handlers with the FastAPI app
    """
    # Custom exception handlers
    app.add_exception_handler(
        TrafficManagementException,
        traffic_management_exception_handler
    )
    
    # Standard exception handlers
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(StarletteHTTPException, starlette_http_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    logger.info("Exception handlers registered successfully")


def create_error_response(
    error_code: str,
    message: str,
    status_code: int = 500,
    details: dict = None
) -> JSONResponse:
    """
    Create a standardized error response
    """
    return JSONResponse(
        status_code=status_code,
        content={
            "error_code": error_code,
            "message": message,
            "details": details or {}
        }
    )

