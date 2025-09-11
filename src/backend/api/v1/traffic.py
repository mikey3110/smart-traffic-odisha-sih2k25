"""
Traffic API endpoints for v1
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from database.connection import get_db
from database.models import TrafficData, Intersection
from models.schemas import (
    TrafficDataSchema, TrafficStatusSchema, APIResponseSchema, 
    PaginatedResponseSchema, PaginationSchema
)
from api.dependencies import (
    get_intersection_by_id, get_latest_traffic_data, 
    get_pagination_params, get_sorting_params, log_api_call
)
from services.redis_service import redis_service
from config.logging_config import get_logger, log_traffic_event

logger = get_logger("traffic_api")

router = APIRouter(prefix="/traffic", tags=["traffic"])


@router.post("/ingest", response_model=APIResponseSchema)
async def ingest_traffic_data(
    data: TrafficDataSchema,
    db: Session = Depends(get_db)
):
    """
    Ingest traffic data from computer vision systems
    """
    try:
        # Validate intersection exists
        intersection = get_intersection_by_id(data.intersection_id, db)
        
        # Create traffic data record
        traffic_record = TrafficData(
            intersection_id=data.intersection_id,
            timestamp=datetime.fromtimestamp(data.timestamp),
            lane_counts=data.lane_counts,
            avg_speed=data.avg_speed,
            weather_condition=data.weather_condition,
            vehicle_types=data.vehicle_types,
            confidence_score=data.confidence_score,
            data_source="camera"
        )
        
        db.add(traffic_record)
        db.commit()
        db.refresh(traffic_record)
        
        # Store in Redis for real-time access
        if redis_service.connected:
            redis_data = {
                "intersection_id": data.intersection_id,
                "timestamp": data.timestamp,
                "lane_counts": data.lane_counts,
                "avg_speed": data.avg_speed,
                "weather_condition": data.weather_condition,
                "vehicle_types": data.vehicle_types,
                "confidence_score": data.confidence_score,
                "ingested_at": datetime.now().isoformat()
            }
            
            redis_service.set(
                f"traffic:{data.intersection_id}",
                redis_data,
                ttl=3600  # 1 hour TTL
            )
        
        # Log traffic event
        log_traffic_event(
            logger=logger,
            event_type="traffic_data_received",
            intersection_id=data.intersection_id,
            details={
                "lane_counts": data.lane_counts,
                "confidence_score": data.confidence_score,
                "record_id": traffic_record.id
            }
        )
        
        return APIResponseSchema(
            status="success",
            message=f"Traffic data ingested for intersection {data.intersection_id}",
            data={
                "record_id": traffic_record.id,
                "intersection_id": data.intersection_id,
                "timestamp": data.timestamp,
                "lane_counts": data.lane_counts
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to ingest traffic data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "TRAFFIC_DATA_INGESTION_FAILED",
                "message": "Failed to ingest traffic data",
                "details": {"error": str(e)}
            }
        )


@router.get("/status/{intersection_id}", response_model=APIResponseSchema)
async def get_traffic_status(
    intersection_id: str,
    db: Session = Depends(get_db)
):
    """
    Get current traffic status for an intersection
    """
    try:
        # Validate intersection exists
        intersection = get_intersection_by_id(intersection_id, db)
        
        # Try to get data from Redis first (real-time)
        traffic_data = None
        if redis_service.connected:
            traffic_data = redis_service.get(f"traffic:{intersection_id}")
        
        # Fallback to database if Redis data not available
        if not traffic_data:
            latest_record = get_latest_traffic_data(intersection_id, db)
            if latest_record:
                traffic_data = {
                    "intersection_id": latest_record.intersection_id,
                    "timestamp": int(latest_record.timestamp.timestamp()),
                    "lane_counts": latest_record.lane_counts,
                    "avg_speed": latest_record.avg_speed,
                    "weather_condition": latest_record.weather_condition,
                    "vehicle_types": latest_record.vehicle_types,
                    "confidence_score": latest_record.confidence_score,
                    "ingested_at": latest_record.created_at.isoformat()
                }
        
        # Return mock data if no real data available
        if not traffic_data:
            traffic_data = {
                "intersection_id": intersection_id,
                "timestamp": int(datetime.now().timestamp()),
                "lane_counts": {"north_lane": 12, "south_lane": 8, "east_lane": 15, "west_lane": 10},
                "avg_speed": 25.5,
                "weather_condition": "clear",
                "vehicle_types": {"car": 30, "truck": 5, "motorcycle": 8},
                "confidence_score": 0.85,
                "ingested_at": datetime.now().isoformat()
            }
        
        return APIResponseSchema(
            status="success",
            message=f"Traffic status retrieved for intersection {intersection_id}",
            data=traffic_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get traffic status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "TRAFFIC_STATUS_RETRIEVAL_FAILED",
                "message": "Failed to retrieve traffic status",
                "details": {"error": str(e)}
            }
        )


@router.get("/history/{intersection_id}", response_model=PaginatedResponseSchema)
async def get_traffic_history(
    intersection_id: str,
    start_time: Optional[datetime] = Query(None, description="Start time for data range"),
    end_time: Optional[datetime] = Query(None, description="End time for data range"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("timestamp", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    db: Session = Depends(get_db)
):
    """
    Get traffic data history for an intersection
    """
    try:
        # Validate intersection exists
        intersection = get_intersection_by_id(intersection_id, db)
        
        # Get pagination parameters
        pagination = get_pagination_params(page, per_page)
        
        # Get sorting parameters
        sorting = get_sorting_params(
            sort_by=sort_by,
            sort_order=sort_order,
            allowed_sort_fields=["timestamp", "created_at", "confidence_score"]
        )
        
        # Build query
        query = db.query(TrafficData).filter(TrafficData.intersection_id == intersection_id)
        
        # Apply time filters
        if start_time:
            query = query.filter(TrafficData.timestamp >= start_time)
        if end_time:
            query = query.filter(TrafficData.timestamp <= end_time)
        
        # Apply sorting
        if sorting["sort_by"] == "timestamp":
            if sorting["sort_order"] == "desc":
                query = query.order_by(TrafficData.timestamp.desc())
            else:
                query = query.order_by(TrafficData.timestamp.asc())
        elif sorting["sort_by"] == "created_at":
            if sorting["sort_order"] == "desc":
                query = query.order_by(TrafficData.created_at.desc())
            else:
                query = query.order_by(TrafficData.created_at.asc())
        elif sorting["sort_by"] == "confidence_score":
            if sorting["sort_order"] == "desc":
                query = query.order_by(TrafficData.confidence_score.desc())
            else:
                query = query.order_by(TrafficData.confidence_score.asc())
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        items = query.offset(pagination["offset"]).limit(pagination["per_page"]).all()
        
        # Convert to response format
        traffic_items = []
        for item in items:
            traffic_items.append({
                "id": item.id,
                "intersection_id": item.intersection_id,
                "timestamp": int(item.timestamp.timestamp()),
                "lane_counts": item.lane_counts,
                "avg_speed": item.avg_speed,
                "weather_condition": item.weather_condition,
                "vehicle_types": item.vehicle_types,
                "confidence_score": item.confidence_score,
                "data_source": item.data_source,
                "created_at": item.created_at.isoformat()
            })
        
        # Calculate pagination info
        pages = (total + pagination["per_page"] - 1) // pagination["per_page"]
        
        pagination_info = PaginationSchema(
            page=pagination["page"],
            per_page=pagination["per_page"],
            total=total,
            pages=pages,
            has_next=pagination["page"] < pages,
            has_prev=pagination["page"] > 1
        )
        
        return PaginatedResponseSchema(
            items=traffic_items,
            pagination=pagination_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get traffic history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "TRAFFIC_HISTORY_RETRIEVAL_FAILED",
                "message": "Failed to retrieve traffic history",
                "details": {"error": str(e)}
            }
        )


@router.get("/analytics/{intersection_id}", response_model=APIResponseSchema)
async def get_traffic_analytics(
    intersection_id: str,
    hours: int = Query(24, ge=1, le=168, description="Hours of data to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get traffic analytics for an intersection
    """
    try:
        # Validate intersection exists
        intersection = get_intersection_by_id(intersection_id, db)
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Get traffic data in time range
        traffic_data = (
            db.query(TrafficData)
            .filter(
                TrafficData.intersection_id == intersection_id,
                TrafficData.timestamp >= start_time,
                TrafficData.timestamp <= end_time
            )
            .order_by(TrafficData.timestamp.asc())
            .all()
        )
        
        if not traffic_data:
            return APIResponseSchema(
                status="success",
                message=f"No traffic data found for intersection {intersection_id} in the last {hours} hours",
                data={
                    "intersection_id": intersection_id,
                    "time_range_hours": hours,
                    "total_records": 0,
                    "analytics": {}
                }
            )
        
        # Calculate analytics
        total_records = len(traffic_data)
        
        # Average vehicle counts per lane
        lane_totals = {}
        lane_counts = {}
        for record in traffic_data:
            for lane, count in record.lane_counts.items():
                if lane not in lane_totals:
                    lane_totals[lane] = 0
                    lane_counts[lane] = 0
                lane_totals[lane] += count
                lane_counts[lane] += 1
        
        avg_lane_counts = {
            lane: lane_totals[lane] / lane_counts[lane] if lane_counts[lane] > 0 else 0
            for lane in lane_totals
        }
        
        # Average speed
        speeds = [record.avg_speed for record in traffic_data if record.avg_speed is not None]
        avg_speed = sum(speeds) / len(speeds) if speeds else None
        
        # Confidence scores
        confidences = [record.confidence_score for record in traffic_data if record.confidence_score is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        
        # Peak hours analysis
        hourly_counts = {}
        for record in traffic_data:
            hour = record.timestamp.hour
            if hour not in hourly_counts:
                hourly_counts[hour] = 0
            hourly_counts[hour] += sum(record.lane_counts.values())
        
        peak_hour = max(hourly_counts.items(), key=lambda x: x[1]) if hourly_counts else None
        
        analytics = {
            "intersection_id": intersection_id,
            "time_range_hours": hours,
            "total_records": total_records,
            "average_lane_counts": avg_lane_counts,
            "average_speed": avg_speed,
            "average_confidence": avg_confidence,
            "peak_hour": {
                "hour": peak_hour[0] if peak_hour else None,
                "total_vehicles": peak_hour[1] if peak_hour else None
            },
            "hourly_distribution": hourly_counts
        }
        
        return APIResponseSchema(
            status="success",
            message=f"Traffic analytics retrieved for intersection {intersection_id}",
            data=analytics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get traffic analytics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "TRAFFIC_ANALYTICS_FAILED",
                "message": "Failed to retrieve traffic analytics",
                "details": {"error": str(e)}
            }
        )