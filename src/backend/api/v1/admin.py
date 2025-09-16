"""
Admin API endpoints for Smart Traffic Management System v1
Provides system administration and monitoring capabilities
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import asyncio

from config.logging_config import get_logger
from config.settings import settings
from models.enhanced_models import APIResponse
from services.enhanced_redis_service import enhanced_redis_service
from database import db_manager
from database.enhanced_database import get_async_session
from middleware.request_timing import RequestTimingMiddleware
from middleware.rate_limiting import RateLimitingMiddleware
from exceptions import ServiceUnavailableError

logger = get_logger(__name__)

# Create router
router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    responses={
        403: {"description": "Forbidden"},
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)


@router.get("/stats", response_model=APIResponse)
async def get_system_stats() -> APIResponse:
    """
    Get comprehensive system statistics
    
    Returns:
        System statistics
    """
    try:
        logger.info("Retrieving system statistics")
        
        # Get Redis statistics
        redis_stats = enhanced_redis_service.get_stats()
        
        # Get database statistics
        db_stats = db_manager.get_stats()
        
        # Get middleware statistics (if available)
        middleware_stats = {
            "request_timing": {
                "message": "Request timing middleware statistics not directly accessible",
                "note": "Check logs for detailed request timing information"
            },
            "rate_limiting": {
                "message": "Rate limiting statistics not directly accessible",
                "note": "Check logs for rate limiting information"
            }
        }
        
        system_stats = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "redis": redis_stats,
            "database": db_stats,
            "middleware": middleware_stats,
            "application": {
                "version": settings.app_version,
                "environment": settings.environment,
                "debug": settings.debug,
                "uptime": "calculated_from_start_time"
            }
        }
        
        logger.info("Successfully retrieved system statistics")
        
        return APIResponse(
            status="success",
            data=system_stats
        )
        
    except Exception as e:
        logger.error(f"Error retrieving system statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system statistics: {str(e)}")


@router.post("/redis/clear", response_model=APIResponse)
async def clear_redis_data(
    pattern: str = Query("traffic:*", description="Redis key pattern to clear"),
    confirm: bool = Query(False, description="Confirmation flag")
) -> APIResponse:
    """
    Clear Redis data matching a pattern
    
    Args:
        pattern: Redis key pattern to clear
        confirm: Confirmation flag
    
    Returns:
        Clear operation result
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Clear operation requires confirmation. Add '?confirm=true' to the request."
        )
    
    try:
        logger.info(f"Clearing Redis data with pattern: {pattern}")
        
        # Clear data matching pattern
        deleted_count = await enhanced_redis_service.delete_data(pattern)
        
        logger.info(f"Successfully cleared {deleted_count} Redis keys matching pattern: {pattern}")
        
        return APIResponse(
            status="success",
            message=f"Cleared {deleted_count} Redis keys matching pattern: {pattern}",
            data={
                "pattern": pattern,
                "deleted_count": deleted_count
            }
        )
        
    except Exception as e:
        logger.error(f"Error clearing Redis data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear Redis data: {str(e)}")


@router.post("/redis/reset-stats", response_model=APIResponse)
async def reset_redis_stats() -> APIResponse:
    """
    Reset Redis service statistics
    
    Returns:
        Reset operation result
    """
    try:
        logger.info("Resetting Redis service statistics")
        
        # Reset Redis statistics
        enhanced_redis_service.reset_stats()
        
        logger.info("Successfully reset Redis service statistics")
        
        return APIResponse(
            status="success",
            message="Redis service statistics reset successfully"
        )
        
    except Exception as e:
        logger.error(f"Error resetting Redis statistics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset Redis statistics: {str(e)}")


@router.get("/database/tables", response_model=APIResponse)
async def get_database_tables() -> APIResponse:
    """
    Get database table information
    
    Returns:
        Database table information
    """
    try:
        logger.info("Retrieving database table information")
        
        # This would typically query the database metadata
        # For now, we'll return the known table structure
        tables = [
            {
                "name": "intersections",
                "description": "Intersection metadata and configuration",
                "columns": ["id", "intersection_id", "name", "location", "latitude", "longitude", "lanes", "is_active", "created_at", "updated_at"]
            },
            {
                "name": "traffic_data",
                "description": "Real-time traffic data",
                "columns": ["id", "intersection_id", "timestamp", "lane_counts", "avg_speed", "weather_condition", "vehicle_types", "congestion_level", "temperature", "humidity", "visibility", "created_at"]
            },
            {
                "name": "signal_optimizations",
                "description": "Signal optimization data",
                "columns": ["id", "intersection_id", "optimized_timings", "confidence_score", "expected_improvement", "algorithm_used", "optimization_metadata", "status", "is_applied", "applied_at", "created_at", "updated_at"]
            },
            {
                "name": "traffic_alerts",
                "description": "System alerts and notifications",
                "columns": ["id", "intersection_id", "alert_type", "severity", "message", "metadata", "is_resolved", "resolved_at", "created_at"]
            },
            {
                "name": "system_metrics",
                "description": "System performance metrics",
                "columns": ["id", "metric_name", "metric_value", "metric_unit", "tags", "timestamp"]
            },
            {
                "name": "api_logs",
                "description": "API request/response logs",
                "columns": ["id", "request_id", "method", "path", "status_code", "response_time", "client_ip", "user_agent", "request_size", "response_size", "error_message", "created_at"]
            },
            {
                "name": "configurations",
                "description": "System configuration",
                "columns": ["id", "key", "value", "value_type", "description", "is_system", "created_at", "updated_at"]
            }
        ]
        
        logger.info(f"Successfully retrieved information for {len(tables)} database tables")
        
        return APIResponse(
            status="success",
            data={
                "tables": tables,
                "total_count": len(tables)
            }
        )
        
    except Exception as e:
        logger.error(f"Error retrieving database table information: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve database table information: {str(e)}")


@router.post("/database/create-tables", response_model=APIResponse)
async def create_database_tables() -> APIResponse:
    """
    Create all database tables
    
    Returns:
        Table creation result
    """
    try:
        logger.info("Creating database tables")
        
        # Create all tables
        await db_manager.create_tables()
        
        logger.info("Successfully created all database tables")
        
        return APIResponse(
            status="success",
            message="All database tables created successfully"
        )
        
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create database tables: {str(e)}")


@router.post("/database/drop-tables", response_model=APIResponse)
async def drop_database_tables(
    confirm: bool = Query(False, description="Confirmation flag")
) -> APIResponse:
    """
    Drop all database tables
    
    Args:
        confirm: Confirmation flag
    
    Returns:
        Table drop result
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Drop operation requires confirmation. Add '?confirm=true' to the request."
        )
    
    try:
        logger.warning("Dropping all database tables")
        
        # Drop all tables
        await db_manager.drop_tables()
        
        logger.warning("Successfully dropped all database tables")
        
        return APIResponse(
            status="success",
            message="All database tables dropped successfully"
        )
        
    except Exception as e:
        logger.error(f"Error dropping database tables: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to drop database tables: {str(e)}")


@router.get("/logs", response_model=APIResponse)
async def get_recent_logs(
    level: str = Query("INFO", description="Log level filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of log entries")
) -> APIResponse:
    """
    Get recent log entries
    
    Args:
        level: Log level to filter by
        limit: Maximum number of entries to return
    
    Returns:
        Recent log entries
    """
    try:
        logger.info(f"Retrieving recent logs with level: {level}, limit: {limit}")
        
        # This would typically read from log files or a log database
        # For now, we'll return a placeholder response
        log_entries = [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": "INFO",
                "message": "Sample log entry",
                "module": "admin",
                "function": "get_recent_logs"
            }
        ]
        
        logger.info(f"Successfully retrieved {len(log_entries)} log entries")
        
        return APIResponse(
            status="success",
            data={
                "logs": log_entries,
                "level": level,
                "limit": limit,
                "note": "This is a placeholder response. Implement actual log retrieval based on your logging setup."
            }
        )
        
    except Exception as e:
        logger.error(f"Error retrieving recent logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve recent logs: {str(e)}")


@router.get("/config", response_model=APIResponse)
async def get_system_config() -> APIResponse:
    """
    Get current system configuration
    
    Returns:
        System configuration
    """
    try:
        logger.info("Retrieving system configuration")
        
        # Get configuration (excluding sensitive data)
        config = {
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "environment": settings.environment,
            "debug": settings.debug,
            "api_v1_prefix": settings.api_v1_prefix,
            "host": settings.host,
            "port": settings.port,
            "log_level": settings.log_level,
            "rate_limit_requests": settings.rate_limit_requests,
            "rate_limit_window": settings.rate_limit_window,
            "traffic_data_ttl": settings.traffic_data_ttl,
            "signal_data_ttl": settings.signal_data_ttl,
            "health_check_timeout": settings.health_check_timeout
        }
        
        logger.info("Successfully retrieved system configuration")
        
        return APIResponse(
            status="success",
            data=config
        )
        
    except Exception as e:
        logger.error(f"Error retrieving system configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system configuration: {str(e)}")


@router.post("/restart", response_model=APIResponse)
async def restart_application(
    confirm: bool = Query(False, description="Confirmation flag")
) -> APIResponse:
    """
    Restart the application (graceful shutdown)
    
    Args:
        confirm: Confirmation flag
    
    Returns:
        Restart operation result
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Restart operation requires confirmation. Add '?confirm=true' to the request."
        )
    
    try:
        logger.warning("Application restart requested")
        
        # This would typically trigger a graceful shutdown
        # For now, we'll just log the request
        logger.warning("Application restart requested via admin API")
        
        return APIResponse(
            status="success",
            message="Application restart initiated. The application will shut down gracefully.",
            data={
                "restart_time": datetime.now(timezone.utc).isoformat(),
                "note": "This is a placeholder response. Implement actual restart logic based on your deployment setup."
            }
        )
        
    except Exception as e:
        logger.error(f"Error initiating application restart: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initiate application restart: {str(e)}")



