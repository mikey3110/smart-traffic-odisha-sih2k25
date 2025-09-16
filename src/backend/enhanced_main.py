"""
Enhanced main application for Smart Traffic Management System
Integrates all enhanced features including logging, error handling, middleware, and API versioning
"""

import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
import uvicorn

# Import configuration and logging
from config import settings, setup_logging, get_logger
from config.logging_config import app_logger

# Import exception handlers
from exceptions import create_exception_handlers

# Import middleware
from middleware import RequestTimingMiddleware, RateLimitingMiddleware

# Import API routers
from api.v1 import traffic_router, signals_router, health_router, admin_router

# Import services
from services.enhanced_redis_service import enhanced_redis_service
from database import db_manager

# Setup logging
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting Smart Traffic Management System...")
    
    try:
        # Initialize database
        await db_manager.initialize_async()
        await db_manager.create_tables()
        logger.info("✅ Database initialized successfully")
        
        # Initialize Redis (already done in service)
        if enhanced_redis_service.connected:
            logger.info("✅ Redis service connected successfully")
        else:
            logger.warning("⚠️  Redis service not available")
        
        logger.info("🎉 Application startup completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Application startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Smart Traffic Management System...")
    
    try:
        # Close database connections
        await db_manager.close()
        logger.info("✅ Database connections closed")
        
        # Close Redis connections
        await enhanced_redis_service.close()
        logger.info("✅ Redis connections closed")
        
        logger.info("👋 Application shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="AI-based traffic signal optimization system for SIH 2025",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)

# Add custom middleware
app.add_middleware(
    RequestTimingMiddleware,
    track_slow_requests=True,
    slow_request_threshold=1.0
)

app.add_middleware(
    RateLimitingMiddleware,
    requests_per_minute=settings.rate_limit_requests,
    burst_capacity=settings.rate_limit_requests
)

# Add exception handlers
exception_handlers = create_exception_handlers()
for exception_type, handler in exception_handlers.items():
    app.add_exception_handler(exception_type, handler)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Smart Traffic Management System API",
        "version": settings.app_version,
        "status": "operational",
        "environment": settings.environment,
        "docs": "/docs",
        "health": "/api/v1/health",
        "endpoints": {
            "traffic": "/api/v1/traffic",
            "signals": "/api/v1/signals",
            "health": "/api/v1/health",
            "admin": "/api/v1/admin"
        }
    }


# Include API routers
app.include_router(traffic_router, prefix=settings.api_v1_prefix)
app.include_router(signals_router, prefix=settings.api_v1_prefix)
app.include_router(health_router, prefix=settings.api_v1_prefix)
app.include_router(admin_router, prefix=settings.api_v1_prefix)

# Legacy endpoints for backward compatibility
@app.get("/health", tags=["legacy"])
async def legacy_health_check():
    """Legacy health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "version": settings.app_version,
        "note": "This is a legacy endpoint. Use /api/v1/health for enhanced health checks."
    }


# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=settings.app_name,
        version=settings.app_version,
        description="AI-based traffic signal optimization system for SIH 2025",
        routes=app.routes,
    )
    
    # Add custom information
    openapi_schema["info"]["x-logo"] = {
        "url": "https://via.placeholder.com/150x150/4F46E5/FFFFFF?text=STMS"
    }
    
    openapi_schema["info"]["contact"] = {
        "name": "Smart Traffic Management System",
        "email": "support@smarttraffic.com",
        "url": "https://smarttraffic.com"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    # Add server information
    openapi_schema["servers"] = [
        {
            "url": f"http://{settings.host}:{settings.port}",
            "description": f"{settings.environment.title()} server"
        }
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Custom documentation
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with enhanced styling"""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{settings.app_name} - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        swagger_ui_parameters={
            "defaultModelsExpandDepth": 2,
            "defaultModelExpandDepth": 2,
            "displayRequestDuration": True,
            "docExpansion": "list",
            "filter": True,
            "showExtensions": True,
            "showCommonExtensions": True
        }
    )


# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}", extra={
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


# Application startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("🚀 Application startup event triggered")


# Application shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("🛑 Application shutdown event triggered")


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        "enhanced_main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )




