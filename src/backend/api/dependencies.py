"""
API dependencies for dependency injection
"""

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Generator, Optional
import logging

from database.connection import get_db
from database.models import Intersection, TrafficData, SignalTiming
from services.redis_service import redis_service
from config.logging_config import get_logger

logger = get_logger("api_dependencies")


def get_database_session() -> Generator[Session, None, None]:
    """
    Get database session dependency
    """
    return get_db()


def get_redis_service():
    """
    Get Redis service dependency
    """
    if not redis_service.connected:
        logger.warning("Redis service not connected")
    return redis_service


def get_intersection_by_id(
    intersection_id: str,
    db: Session = Depends(get_database_session)
) -> Intersection:
    """
    Get intersection by ID or raise 404
    """
    intersection = db.query(Intersection).filter(Intersection.id == intersection_id).first()
    if not intersection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "INTERSECTION_NOT_FOUND",
                "message": f"Intersection '{intersection_id}' not found",
                "details": {"intersection_id": intersection_id}
            }
        )
    return intersection


def get_intersection_optional(
    intersection_id: str,
    db: Session = Depends(get_database_session)
) -> Optional[Intersection]:
    """
    Get intersection by ID or return None
    """
    return db.query(Intersection).filter(Intersection.id == intersection_id).first()


def validate_intersection_exists(
    intersection_id: str,
    db: Session = Depends(get_database_session)
) -> bool:
    """
    Validate that intersection exists
    """
    intersection = db.query(Intersection).filter(Intersection.id == intersection_id).first()
    if not intersection:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": "INTERSECTION_NOT_FOUND",
                "message": f"Intersection '{intersection_id}' not found",
                "details": {"intersection_id": intersection_id}
            }
        )
    return True


def get_latest_traffic_data(
    intersection_id: str,
    db: Session = Depends(get_database_session)
) -> Optional[TrafficData]:
    """
    Get latest traffic data for intersection
    """
    return (
        db.query(TrafficData)
        .filter(TrafficData.intersection_id == intersection_id)
        .order_by(TrafficData.timestamp.desc())
        .first()
    )


def get_latest_signal_timing(
    intersection_id: str,
    db: Session = Depends(get_database_session)
) -> Optional[SignalTiming]:
    """
    Get latest signal timing for intersection
    """
    return (
        db.query(SignalTiming)
        .filter(
            SignalTiming.intersection_id == intersection_id,
            SignalTiming.is_active == True
        )
        .order_by(SignalTiming.updated_at.desc())
        .first()
    )


def require_redis_connection():
    """
    Require Redis connection to be available
    """
    if not redis_service.connected:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "REDIS_UNAVAILABLE",
                "message": "Redis service is not available",
                "details": {"service": "redis"}
            }
        )
    return True


def require_database_connection(db: Session = Depends(get_database_session)):
    """
    Require database connection to be available
    """
    try:
        # Test database connection
        db.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error_code": "DATABASE_UNAVAILABLE",
                "message": "Database service is not available",
                "details": {"service": "database", "error": str(e)}
            }
        )


def get_request_id(request_id: Optional[str] = None) -> str:
    """
    Get or generate request ID
    """
    if request_id:
        return request_id
    
    import uuid
    return str(uuid.uuid4())


def log_api_call(
    method: str,
    path: str,
    status_code: int,
    response_time: float,
    request_id: str,
    user_id: Optional[str] = None,
    db: Session = Depends(get_database_session)
):
    """
    Log API call to database
    """
    try:
        from database.models import APILog
        
        api_log = APILog(
            request_id=request_id,
            method=method,
            path=path,
            status_code=status_code,
            response_time=response_time,
            user_id=user_id
        )
        
        db.add(api_log)
        db.commit()
        
    except Exception as e:
        logger.error(f"Failed to log API call: {e}")
        # Don't raise exception to avoid breaking the main request


def get_pagination_params(
    page: int = 1,
    per_page: int = 20,
    max_per_page: int = 100
) -> dict:
    """
    Get and validate pagination parameters
    """
    if page < 1:
        page = 1
    if per_page < 1:
        per_page = 20
    if per_page > max_per_page:
        per_page = max_per_page
    
    return {
        "page": page,
        "per_page": per_page,
        "offset": (page - 1) * per_page
    }


def get_sorting_params(
    sort_by: Optional[str] = None,
    sort_order: str = "asc",
    allowed_sort_fields: list = None
) -> dict:
    """
    Get and validate sorting parameters
    """
    if allowed_sort_fields and sort_by and sort_by not in allowed_sort_fields:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "INVALID_SORT_FIELD",
                "message": f"Invalid sort field: {sort_by}",
                "details": {
                    "sort_by": sort_by,
                    "allowed_fields": allowed_sort_fields
                }
            }
        )
    
    if sort_order.lower() not in ["asc", "desc"]:
        sort_order = "asc"
    
    return {
        "sort_by": sort_by,
        "sort_order": sort_order
    }


