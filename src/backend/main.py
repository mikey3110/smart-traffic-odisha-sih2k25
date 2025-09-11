"""
Enhanced Smart Traffic Management System - Main Application
Comprehensive FastAPI application with all enhancements
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# Import configuration and logging
from config.settings import settings
from config.logging_config import get_logger, setup_logging

# Import exception handlers
from exceptions import register_exception_handlers

# Import middleware
from middleware import (
    RequestTimingMiddleware, 
    RateLimitingMiddleware, 
    LoggingMiddleware,
    SecurityLoggingMiddleware
)

# Import API routers
from api.v1 import traffic_router, signals_router, health_router

# Import database
from database.connection import init_database, close_database

# Import services
from services.redis_service import redis_service

# Set up logging
logger = get_logger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events
    """
    # Startup
    logger.info("Starting Smart Traffic Management System...")
    
    try:
        # Initialize database
        if settings.enable_database:
            init_database()
            logger.info("Database initialized successfully")
        
        # Connect to Redis
        if settings.enable_redis:
            redis_connected = redis_service.connect()
            if redis_connected:
                logger.info("Redis connected successfully")
            else:
                logger.warning("Redis connection failed, using fallback storage")
        
        logger.info("Smart Traffic Management System started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Smart Traffic Management System...")
    
    try:
        # Close Redis connection
        if settings.enable_redis and redis_service.connected:
            redis_service.disconnect()
            logger.info("Redis disconnected")
        
        # Close database connections
        if settings.enable_database:
            close_database()
            logger.info("Database connections closed")
        
        logger.info("Smart Traffic Management System shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    lifespan=lifespan,
    docs_url="/docs" if settings.api.debug else None,
    redoc_url="/redoc" if settings.api.debug else None,
    openapi_url="/openapi.json" if settings.api.debug else None
)

# Register exception handlers
register_exception_handlers(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=settings.api.cors_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
if settings.api.debug:
    # Request timing middleware
    app.add_middleware(RequestTimingMiddleware)
    
    # Comprehensive logging middleware
    app.add_middleware(
        LoggingMiddleware,
        log_requests=True,
        log_responses=True,
        log_request_body=False,  # Disable for performance
        log_response_body=False,  # Disable for performance
        max_body_size=1024
    )
    
    # Security logging middleware
    app.add_middleware(SecurityLoggingMiddleware)

# Add rate limiting middleware
app.add_middleware(
    RateLimitingMiddleware,
    requests_per_minute=settings.api.rate_limit_requests,
    window_size=settings.api.rate_limit_window
)

# Include API routers
app.include_router(traffic_router, prefix="/api/v1")
app.include_router(signals_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """
    Root endpoint with system information
    """
    return {
        "message": "Smart Traffic Management System API",
        "version": settings.api.version,
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": settings.environment,
        "endpoints": {
            "traffic": "/api/v1/traffic",
            "signals": "/api/v1/signals", 
            "health": "/api/v1/health",
            "docs": "/docs" if settings.api.debug else "disabled"
        },
        "features": {
            "database": settings.enable_database,
            "redis": settings.enable_redis,
            "metrics": settings.enable_metrics,
            "tracing": settings.enable_tracing
        }
    }


# Legacy endpoints for backward compatibility
@app.get("/health", tags=["legacy"])
async def legacy_health():
    """
    Legacy health check endpoint
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.api.version,
        "message": "Use /api/v1/health/ for comprehensive health checks"
    }


@app.get("/intersections", tags=["legacy"])
async def legacy_intersections():
    """
    Legacy intersections endpoint
    """
    return {
        "status": "success",
        "intersections": ["junction-1", "junction-2", "junction-3"],
        "total_count": 3,
        "message": "Use /api/v1/traffic/status/{id} for detailed intersection data"
    }


# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled exceptions
    """
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            'request_id': getattr(request.state, 'request_id', 'unknown'),
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
                "request_id": getattr(request.state, 'request_id', 'unknown'),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        }
    )


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """
    Add request ID to all requests
    """
    import uuid
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


# Performance monitoring middleware
@app.middleware("http")
async def performance_monitoring(request: Request, call_next):
    """
    Monitor API performance
    """
    start_time = datetime.now()
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds()
    
    # Log slow requests
    if process_time > 1.0:  # Log requests taking more than 1 second
        logger.warning(
            f"Slow request detected: {request.method} {request.url.path} took {process_time:.3f}s",
            extra={
                'request_id': getattr(request.state, 'request_id', 'unknown'),
                'method': request.method,
                'path': str(request.url.path),
                'process_time': process_time,
                'status_code': response.status_code,
                'event_type': 'slow_request'
            }
        )
    
    return response


if __name__ == "__main__":
    # Configure uvicorn
    uvicorn_config = {
        "app": "main:app",
        "host": settings.api.host,
        "port": settings.api.port,
        "reload": settings.api.reload,
        "log_level": settings.logging.level.lower(),
        "access_log": settings.api.debug,
        "use_colors": True
    }
    
    logger.info(f"Starting server on {settings.api.host}:{settings.api.port}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.api.debug}")
    logger.info(f"Database enabled: {settings.enable_database}")
    logger.info(f"Redis enabled: {settings.enable_redis}")
    
    uvicorn.run(**uvicorn_config)