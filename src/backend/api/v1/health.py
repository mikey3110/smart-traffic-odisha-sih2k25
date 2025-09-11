"""
Health check API endpoints for v1
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import psutil
import time
import logging

from database.connection import get_db, health_check as db_health_check
from database.models import Intersection, TrafficData, OptimizationResult
from models.schemas import HealthCheckSchema, APIResponseSchema
from services.redis_service import redis_service
from config.settings import settings
from config.logging_config import get_logger, log_system_event

logger = get_logger("health_api")

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthCheckSchema)
async def health_check():
    """
    Comprehensive health check endpoint
    """
    try:
        start_time = time.time()
        
        # Check system components
        components = {}
        
        # Database health
        db_health = db_health_check()
        components["database"] = {
            "status": db_health["status"],
            "details": db_health
        }
        
        # Redis health
        redis_health = redis_service.health_check()
        components["redis"] = {
            "status": redis_health["status"],
            "details": redis_health
        }
        
        # System metrics
        try:
            memory_info = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=1)
            
            components["system"] = {
                "status": "healthy",
                "details": {
                    "memory_usage_percent": memory_info.percent,
                    "memory_available_gb": round(memory_info.available / (1024**3), 2),
                    "cpu_usage_percent": cpu_percent,
                    "disk_usage_percent": psutil.disk_usage('/').percent
                }
            }
        except Exception as e:
            components["system"] = {
                "status": "error",
                "details": {"error": str(e)}
            }
        
        # Determine overall status
        component_statuses = [comp["status"] for comp in components.values()]
        if "error" in component_statuses:
            overall_status = "error"
        elif "disconnected" in component_statuses:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # Convert to ms
        
        health_data = HealthCheckSchema(
            status=overall_status,
            timestamp=datetime.now(timezone.utc),
            version=settings.api.version,
            components=components,
            uptime=time.time() - start_time,
            memory_usage=components.get("system", {}).get("details", {}).get("memory_usage_percent"),
            cpu_usage=components.get("system", {}).get("details", {}).get("cpu_usage_percent")
        )
        
        # Log health check
        log_system_event(
            logger=logger,
            event_type="health_check",
            message=f"Health check completed with status: {overall_status}",
            level=logging.INFO if overall_status == "healthy" else logging.WARNING,
            component="health_api",
            response_time=response_time
        )
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckSchema(
            status="error",
            timestamp=datetime.now(timezone.utc),
            version=settings.api.version,
            components={"error": {"status": "error", "details": {"error": str(e)}}}
        )


@router.get("/database", response_model=APIResponseSchema)
async def database_health_check(db: Session = Depends(get_db)):
    """
    Detailed database health check
    """
    try:
        # Test basic connection
        db.execute("SELECT 1")
        
        # Get database statistics
        intersection_count = db.query(Intersection).count()
        traffic_data_count = db.query(TrafficData).count()
        optimization_count = db.query(OptimizationResult).count()
        
        # Get recent activity
        recent_traffic = (
            db.query(TrafficData)
            .filter(TrafficData.created_at >= datetime.now() - timedelta(hours=1))
            .count()
        )
        
        db_health = {
            "status": "connected",
            "intersection_count": intersection_count,
            "traffic_data_count": traffic_data_count,
            "optimization_count": optimization_count,
            "recent_traffic_data": recent_traffic,
            "last_check": datetime.now().isoformat()
        }
        
        return APIResponseSchema(
            status="success",
            message="Database health check completed",
            data=db_health
        )
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return APIResponseSchema(
            status="error",
            message="Database health check failed",
            data={
                "status": "error",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
        )


@router.get("/redis", response_model=APIResponseSchema)
async def redis_health_check():
    """
    Detailed Redis health check
    """
    try:
        redis_health = redis_service.health_check()
        
        return APIResponseSchema(
            status="success",
            message="Redis health check completed",
            data=redis_health
        )
        
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return APIResponseSchema(
            status="error",
            message="Redis health check failed",
            data={
                "status": "error",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
        )


@router.get("/system", response_model=APIResponseSchema)
async def system_health_check():
    """
    Detailed system health check
    """
    try:
        # System metrics
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk_info = psutil.disk_usage('/')
        
        # Process information
        process = psutil.Process()
        process_memory = process.memory_info()
        
        system_health = {
            "status": "healthy",
            "memory": {
                "total_gb": round(memory_info.total / (1024**3), 2),
                "available_gb": round(memory_info.available / (1024**3), 2),
                "used_percent": memory_info.percent,
                "process_memory_mb": round(process_memory.rss / (1024**2), 2)
            },
            "cpu": {
                "usage_percent": cpu_percent,
                "count": psutil.cpu_count()
            },
            "disk": {
                "total_gb": round(disk_info.total / (1024**3), 2),
                "free_gb": round(disk_info.free / (1024**3), 2),
                "used_percent": round((disk_info.used / disk_info.total) * 100, 2)
            },
            "uptime_seconds": time.time() - psutil.boot_time(),
            "last_check": datetime.now().isoformat()
        }
        
        return APIResponseSchema(
            status="success",
            message="System health check completed",
            data=system_health
        )
        
    except Exception as e:
        logger.error(f"System health check failed: {e}")
        return APIResponseSchema(
            status="error",
            message="System health check failed",
            data={
                "status": "error",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
        )


@router.get("/ready", response_model=APIResponseSchema)
async def readiness_check():
    """
    Kubernetes readiness probe endpoint
    """
    try:
        # Check critical dependencies
        db_health = db_health_check()
        redis_health = redis_service.health_check()
        
        # Determine if service is ready
        is_ready = (
            db_health["status"] in ["connected"] and
            redis_health["status"] in ["connected"]
        )
        
        if is_ready:
            return APIResponseSchema(
                status="success",
                message="Service is ready",
                data={"ready": True, "timestamp": datetime.now().isoformat()}
            )
        else:
            return APIResponseSchema(
                status="error",
                message="Service is not ready",
                data={
                    "ready": False,
                    "database_status": db_health["status"],
                    "redis_status": redis_health["status"],
                    "timestamp": datetime.now().isoformat()
                }
            )
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return APIResponseSchema(
            status="error",
            message="Readiness check failed",
            data={"ready": False, "error": str(e)}
        )


@router.get("/live", response_model=APIResponseSchema)
async def liveness_check():
    """
    Kubernetes liveness probe endpoint
    """
    try:
        # Simple liveness check - just verify the service is responding
        return APIResponseSchema(
            status="success",
            message="Service is alive",
            data={"alive": True, "timestamp": datetime.now().isoformat()}
        )
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        return APIResponseSchema(
            status="error",
            message="Liveness check failed",
            data={"alive": False, "error": str(e)}
        )