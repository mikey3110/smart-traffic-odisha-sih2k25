"""
Signal optimization API endpoints for v1
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from database.connection import get_db
from database.models import SignalTiming, OptimizationResult, Intersection
from models.schemas import (
    SignalOptimizationSchema, TrafficStatusSchema, APIResponseSchema,
    PaginatedResponseSchema, PaginationSchema
)
from api.dependencies import (
    get_intersection_by_id, get_latest_signal_timing,
    get_pagination_params, get_sorting_params, log_api_call
)
from services.redis_service import redis_service
from config.logging_config import get_logger, log_traffic_event

logger = get_logger("signals_api")

router = APIRouter(prefix="/signals", tags=["signals"])


@router.put("/optimize/{intersection_id}", response_model=APIResponseSchema)
async def optimize_signal(
    intersection_id: str,
    optimization: SignalOptimizationSchema,
    db: Session = Depends(get_db)
):
    """
    Optimize traffic signal timings for an intersection
    """
    try:
        # Validate intersection exists
        intersection = get_intersection_by_id(intersection_id, db)
        
        # Create optimization result record
        optimization_record = OptimizationResult(
            intersection_id=intersection_id,
            algorithm_used=optimization.algorithm_used,
            confidence_score=optimization.confidence_score,
            expected_improvement=optimization.expected_improvement,
            optimization_time=optimization.optimization_time,
            optimized_timings=optimization.optimized_timings,
            status="completed"
        )
        
        db.add(optimization_record)
        db.commit()
        db.refresh(optimization_record)
        
        # Update signal timings in database
        for timing in optimization.optimized_timings:
            # Deactivate existing timings for this lane
            existing_timings = (
                db.query(SignalTiming)
                .filter(
                    SignalTiming.intersection_id == intersection_id,
                    SignalTiming.lane == timing.lane,
                    SignalTiming.is_active == True
                )
                .all()
            )
            
            for existing in existing_timings:
                existing.is_active = False
            
            # Create new timing
            new_timing = SignalTiming(
                intersection_id=intersection_id,
                lane=timing.lane,
                duration=timing.duration,
                state=timing.state,
                priority=timing.priority,
                is_active=True
            )
            
            db.add(new_timing)
        
        db.commit()
        
        # Store in Redis for real-time access
        if redis_service.connected:
            signal_data = {
                "intersection_id": intersection_id,
                "optimized_timings": optimization.optimized_timings,
                "confidence_score": optimization.confidence_score,
                "expected_improvement": optimization.expected_improvement,
                "algorithm_used": optimization.algorithm_used,
                "optimization_time": optimization.optimization_time,
                "optimized_at": datetime.now().isoformat(),
                "status": "active"
            }
            
            redis_service.set(
                f"signal:{intersection_id}",
                signal_data,
                ttl=3600  # 1 hour TTL
            )
        
        # Log optimization event
        log_traffic_event(
            logger=logger,
            event_type="signal_optimized",
            intersection_id=intersection_id,
            details={
                "optimization_id": optimization_record.id,
                "confidence_score": optimization.confidence_score,
                "expected_improvement": optimization.expected_improvement,
                "algorithm_used": optimization.algorithm_used
            }
        )
        
        return APIResponseSchema(
            status="success",
            message=f"Signal optimization applied to intersection {intersection_id}",
            data={
                "optimization_id": optimization_record.id,
                "intersection_id": intersection_id,
                "optimized_timings": optimization.optimized_timings,
                "confidence_score": optimization.confidence_score,
                "expected_improvement": optimization.expected_improvement
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to optimize signal: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "SIGNAL_OPTIMIZATION_FAILED",
                "message": "Failed to optimize signal timings",
                "details": {"error": str(e)}
            }
        )


@router.get("/status/{intersection_id}", response_model=APIResponseSchema)
async def get_signal_status(
    intersection_id: str,
    db: Session = Depends(get_db)
):
    """
    Get current signal status for an intersection
    """
    try:
        # Validate intersection exists
        intersection = get_intersection_by_id(intersection_id, db)
        
        # Try to get data from Redis first (real-time)
        signal_data = None
        if redis_service.connected:
            signal_data = redis_service.get(f"signal:{intersection_id}")
        
        # Fallback to database if Redis data not available
        if not signal_data:
            # Get latest optimization result
            latest_optimization = (
                db.query(OptimizationResult)
                .filter(OptimizationResult.intersection_id == intersection_id)
                .order_by(OptimizationResult.created_at.desc())
                .first()
            )
            
            # Get current signal timings
            current_timings = (
                db.query(SignalTiming)
                .filter(
                    SignalTiming.intersection_id == intersection_id,
                    SignalTiming.is_active == True
                )
                .all()
            )
            
            if latest_optimization and current_timings:
                signal_data = {
                    "intersection_id": intersection_id,
                    "optimized_timings": latest_optimization.optimized_timings,
                    "confidence_score": latest_optimization.confidence_score,
                    "expected_improvement": latest_optimization.expected_improvement,
                    "algorithm_used": latest_optimization.algorithm_used,
                    "optimization_time": latest_optimization.optimization_time,
                    "optimized_at": latest_optimization.created_at.isoformat(),
                    "status": latest_optimization.status
                }
            else:
                # Return default signal timings
                signal_data = {
                    "intersection_id": intersection_id,
                    "optimized_timings": [
                        {"lane": "north_lane", "duration": 30, "state": "green", "priority": 2},
                        {"lane": "south_lane", "duration": 30, "state": "green", "priority": 2},
                        {"lane": "east_lane", "duration": 30, "state": "red", "priority": 2},
                        {"lane": "west_lane", "duration": 30, "state": "red", "priority": 2}
                    ],
                    "confidence_score": 0.8,
                    "expected_improvement": 15.0,
                    "algorithm_used": "default",
                    "optimization_time": 0.0,
                    "optimized_at": datetime.now().isoformat(),
                    "status": "default"
                }
        
        return APIResponseSchema(
            status="success",
            message=f"Signal status retrieved for intersection {intersection_id}",
            data=signal_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get signal status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "SIGNAL_STATUS_RETRIEVAL_FAILED",
                "message": "Failed to retrieve signal status",
                "details": {"error": str(e)}
            }
        )


@router.get("/history/{intersection_id}", response_model=PaginatedResponseSchema)
async def get_signal_history(
    intersection_id: str,
    start_time: Optional[datetime] = Query(None, description="Start time for data range"),
    end_time: Optional[datetime] = Query(None, description="End time for data range"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(20, ge=1, le=100, description="Items per page"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    db: Session = Depends(get_db)
):
    """
    Get signal optimization history for an intersection
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
            allowed_sort_fields=["created_at", "confidence_score", "expected_improvement"]
        )
        
        # Build query
        query = db.query(OptimizationResult).filter(OptimizationResult.intersection_id == intersection_id)
        
        # Apply time filters
        if start_time:
            query = query.filter(OptimizationResult.created_at >= start_time)
        if end_time:
            query = query.filter(OptimizationResult.created_at <= end_time)
        
        # Apply sorting
        if sorting["sort_by"] == "created_at":
            if sorting["sort_order"] == "desc":
                query = query.order_by(OptimizationResult.created_at.desc())
            else:
                query = query.order_by(OptimizationResult.created_at.asc())
        elif sorting["sort_by"] == "confidence_score":
            if sorting["sort_order"] == "desc":
                query = query.order_by(OptimizationResult.confidence_score.desc())
            else:
                query = query.order_by(OptimizationResult.confidence_score.asc())
        elif sorting["sort_by"] == "expected_improvement":
            if sorting["sort_order"] == "desc":
                query = query.order_by(OptimizationResult.expected_improvement.desc())
            else:
                query = query.order_by(OptimizationResult.expected_improvement.asc())
        
        # Get total count
        total = query.count()
        
        # Apply pagination
        items = query.offset(pagination["offset"]).limit(pagination["per_page"]).all()
        
        # Convert to response format
        signal_items = []
        for item in items:
            signal_items.append({
                "id": item.id,
                "intersection_id": item.intersection_id,
                "algorithm_used": item.algorithm_used,
                "confidence_score": item.confidence_score,
                "expected_improvement": item.expected_improvement,
                "optimization_time": item.optimization_time,
                "optimized_timings": item.optimized_timings,
                "status": item.status,
                "error_message": item.error_message,
                "created_at": item.created_at.isoformat(),
                "applied_at": item.applied_at.isoformat() if item.applied_at else None
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
            items=signal_items,
            pagination=pagination_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get signal history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "SIGNAL_HISTORY_RETRIEVAL_FAILED",
                "message": "Failed to retrieve signal history",
                "details": {"error": str(e)}
            }
        )


@router.get("/performance/{intersection_id}", response_model=APIResponseSchema)
async def get_signal_performance(
    intersection_id: str,
    days: int = Query(7, ge=1, le=30, description="Days of data to analyze"),
    db: Session = Depends(get_db)
):
    """
    Get signal performance metrics for an intersection
    """
    try:
        # Validate intersection exists
        intersection = get_intersection_by_id(intersection_id, db)
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        # Get optimization results in time range
        optimizations = (
            db.query(OptimizationResult)
            .filter(
                OptimizationResult.intersection_id == intersection_id,
                OptimizationResult.created_at >= start_time,
                OptimizationResult.created_at <= end_time,
                OptimizationResult.status == "completed"
            )
            .order_by(OptimizationResult.created_at.asc())
            .all()
        )
        
        if not optimizations:
            return APIResponseSchema(
                status="success",
                message=f"No optimization data found for intersection {intersection_id} in the last {days} days",
                data={
                    "intersection_id": intersection_id,
                    "time_range_days": days,
                    "total_optimizations": 0,
                    "performance_metrics": {}
                }
            )
        
        # Calculate performance metrics
        total_optimizations = len(optimizations)
        
        # Average confidence score
        confidences = [opt.confidence_score for opt in optimizations if opt.confidence_score is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        
        # Average expected improvement
        improvements = [opt.expected_improvement for opt in optimizations if opt.expected_improvement is not None]
        avg_improvement = sum(improvements) / len(improvements) if improvements else None
        
        # Average optimization time
        times = [opt.optimization_time for opt in optimizations if opt.optimization_time is not None]
        avg_optimization_time = sum(times) / len(times) if times else None
        
        # Success rate
        successful_optimizations = len([opt for opt in optimizations if opt.status == "completed"])
        success_rate = (successful_optimizations / total_optimizations) * 100 if total_optimizations > 0 else 0
        
        # Algorithm usage
        algorithm_usage = {}
        for opt in optimizations:
            if opt.algorithm_used:
                algorithm_usage[opt.algorithm_used] = algorithm_usage.get(opt.algorithm_used, 0) + 1
        
        performance_metrics = {
            "intersection_id": intersection_id,
            "time_range_days": days,
            "total_optimizations": total_optimizations,
            "successful_optimizations": successful_optimizations,
            "success_rate": round(success_rate, 2),
            "average_confidence": round(avg_confidence, 3) if avg_confidence else None,
            "average_improvement": round(avg_improvement, 2) if avg_improvement else None,
            "average_optimization_time": round(avg_optimization_time, 3) if avg_optimization_time else None,
            "algorithm_usage": algorithm_usage
        }
        
        return APIResponseSchema(
            status="success",
            message=f"Signal performance metrics retrieved for intersection {intersection_id}",
            data=performance_metrics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get signal performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error_code": "SIGNAL_PERFORMANCE_FAILED",
                "message": "Failed to retrieve signal performance metrics",
                "details": {"error": str(e)}
            }
        )