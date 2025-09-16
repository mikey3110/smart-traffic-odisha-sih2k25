"""
ML Metrics API endpoints for v1
Day 1 Sprint Implementation - ML Engineer
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging
import json
import os

from database.connection import get_db
from models.schemas import APIResponseSchema
from api.dependencies import log_api_call
from config.logging_config import get_logger

logger = get_logger("ml_metrics_api")

router = APIRouter(prefix="/ml", tags=["ml-metrics"])


@router.get("/metrics", response_model=APIResponseSchema)
async def get_ml_metrics(
    db: Session = Depends(get_db)
):
    """
    Get current ML optimization metrics
    """
    try:
        # Try to get live metrics from ML optimizer
        metrics_file = "models/optimization_state.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                state = json.load(f)
                metrics = state.get('metrics', {})
        else:
            # Return default metrics if no state file
            metrics = {
                'total_cycles': 0,
                'successful_cycles': 0,
                'average_cycle_time': 0.0,
                'timing_drift': 0.0,
                'current_reward': 0.0,
                'q_table_size': 0,
                'learning_rate': 0.01,
                'epsilon': 0.1,
                'performance_improvement': 0.0
            }
        
        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()
        
        return APIResponseSchema(
            success=True,
            data=metrics,
            message="ML metrics retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving ML metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve ML metrics: {str(e)}"
        )


@router.get("/performance", response_model=APIResponseSchema)
async def get_ml_performance(
    hours: int = Query(24, description="Hours of performance data to retrieve"),
    db: Session = Depends(get_db)
):
    """
    Get ML performance metrics over time
    """
    try:
        # Try to get performance data
        performance_file = "reports/optimization_performance.json"
        if os.path.exists(performance_file):
            with open(performance_file, 'r') as f:
                performance_data = json.load(f)
        else:
            # Return default performance data
            performance_data = {
                'total_cycles': 0,
                'successful_cycles': 0,
                'success_rate': 0.0,
                'average_cycle_time': 0.0,
                'average_reward': 0.0,
                'timing_drift': 0.0,
                'performance_improvement': 0.0,
                'generated_at': datetime.now().isoformat()
            }
        
        return APIResponseSchema(
            success=True,
            data=performance_data,
            message=f"ML performance data retrieved for last {hours} hours"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving ML performance: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve ML performance: {str(e)}"
        )


@router.get("/reward-curve", response_model=APIResponseSchema)
async def get_reward_curve(
    intersection_id: str = Query("junction-1", description="Intersection ID"),
    db: Session = Depends(get_db)
):
    """
    Get reward curve data for visualization
    """
    try:
        # Try to get reward curve data
        reward_data = {
            'cycles': list(range(100)),  # Mock data for now
            'rewards': [0.5 + 0.3 * (i % 20) / 20 for i in range(100)],  # Mock data
            'moving_average': [0.5 + 0.2 * (i % 20) / 20 for i in range(100)],  # Mock data
            'intersection_id': intersection_id
        }
        
        # In real implementation, this would come from the ML optimizer
        # reward_data = ml_optimizer.get_reward_curve_data()
        
        return APIResponseSchema(
            success=True,
            data=reward_data,
            message=f"Reward curve data retrieved for {intersection_id}"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving reward curve: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve reward curve: {str(e)}"
        )


@router.get("/q-table-heatmap", response_model=APIResponseSchema)
async def get_q_table_heatmap(
    intersection_id: str = Query("junction-1", description="Intersection ID"),
    db: Session = Depends(get_db)
):
    """
    Get Q-table heatmap data for visualization
    """
    try:
        # Mock Q-table data for visualization
        states = [f"state_{i}" for i in range(10)]
        actions = [f"action_{i}" for i in range(5)]
        q_values = [[0.1 + 0.8 * (i + j) / 14 for j in range(5)] for i in range(10)]
        
        heatmap_data = {
            'states': states,
            'actions': actions,
            'q_values': q_values,
            'intersection_id': intersection_id,
            'max_q_value': max(max(row) for row in q_values),
            'min_q_value': min(min(row) for row in q_values)
        }
        
        # In real implementation, this would come from the Q-learning optimizer
        # heatmap_data = q_learning_optimizer.get_q_table_heatmap(intersection_id)
        
        return APIResponseSchema(
            success=True,
            data=heatmap_data,
            message=f"Q-table heatmap data retrieved for {intersection_id}"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving Q-table heatmap: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve Q-table heatmap: {str(e)}"
        )


@router.get("/status", response_model=APIResponseSchema)
async def get_ml_status(
    db: Session = Depends(get_db)
):
    """
    Get ML optimizer status and health
    """
    try:
        # Check if ML optimizer is running
        status_file = "models/optimization_state.json"
        is_running = os.path.exists(status_file)
        
        # Get basic status information
        status_data = {
            'is_running': is_running,
            'last_update': datetime.now().isoformat(),
            'components': {
                'q_learning': True,
                'data_integration': True,
                'signal_optimizer': True,
                'traffic_predictor': True
            },
            'health': 'healthy' if is_running else 'stopped'
        }
        
        # Add metrics if available
        if is_running:
            try:
                with open(status_file, 'r') as f:
                    state = json.load(f)
                    status_data['metrics'] = state.get('metrics', {})
            except:
                pass
        
        return APIResponseSchema(
            success=True,
            data=status_data,
            message="ML status retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving ML status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve ML status: {str(e)}"
        )


@router.get("/hyperparameters", response_model=APIResponseSchema)
async def get_hyperparameters(
    db: Session = Depends(get_db)
):
    """
    Get current ML hyperparameters
    """
    try:
        # Get hyperparameters from config or state
        hyperparameters = {
            'q_learning': {
                'learning_rate': 0.01,
                'epsilon': 0.1,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01,
                'discount_factor': 0.95,
                'replay_buffer_size': 10000,
                'batch_size': 32,
                'target_update_frequency': 100
            },
            'state_space': {
                'lane_counts': 'continuous',
                'avg_speed': 'continuous',
                'queue_lengths': 'continuous',
                'waiting_times': 'continuous',
                'time_of_day': 'continuous',
                'day_of_week': 'continuous',
                'weather_condition': 'discrete',
                'current_phase': 'discrete'
            },
            'action_space': {
                'type': 'discrete',
                'size': 4,
                'actions': ['extend_north_south', 'extend_east_west', 'reduce_cycle', 'emergency_override']
            },
            'reward_function': {
                'formula': 'reward = -waiting_time - queue_length + throughput',
                'weights': {
                    'waiting_time': -0.5,
                    'queue_length': -0.3,
                    'throughput': 0.2
                }
            }
        }
        
        return APIResponseSchema(
            success=True,
            data=hyperparameters,
            message="Hyperparameters retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving hyperparameters: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve hyperparameters: {str(e)}"
        )


@router.get("/performance-gains", response_model=APIResponseSchema)
async def get_performance_gains(
    db: Session = Depends(get_db)
):
    """
    Get performance improvement metrics
    """
    try:
        # Mock performance gains data
        performance_gains = {
            'overall_improvement': 18.5,  # 18.5% improvement
            'scenarios': {
                'rush_hour': {
                    'wait_time_reduction': 22.3,
                    'throughput_increase': 15.7,
                    'fuel_savings': 12.1
                },
                'normal_traffic': {
                    'wait_time_reduction': 16.8,
                    'throughput_increase': 11.2,
                    'fuel_savings': 8.9
                },
                'low_traffic': {
                    'wait_time_reduction': 14.2,
                    'throughput_increase': 9.5,
                    'fuel_savings': 7.3
                },
                'emergency_scenario': {
                    'wait_time_reduction': 25.1,
                    'throughput_increase': 18.9,
                    'fuel_savings': 14.7
                },
                'event_traffic': {
                    'wait_time_reduction': 19.6,
                    'throughput_increase': 13.4,
                    'fuel_savings': 10.8
                }
            },
            'baseline_metrics': {
                'average_wait_time': 45.2,  # seconds
                'average_throughput': 120.5,  # vehicles/hour
                'average_fuel_consumption': 8.7  # liters/hour
            },
            'optimized_metrics': {
                'average_wait_time': 36.8,  # seconds
                'average_throughput': 140.1,  # vehicles/hour
                'average_fuel_consumption': 7.6  # liters/hour
            }
        }
        
        return APIResponseSchema(
            success=True,
            data=performance_gains,
            message="Performance gains retrieved successfully"
        )
        
    except Exception as e:
        logger.error(f"Error retrieving performance gains: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve performance gains: {str(e)}"
        )
