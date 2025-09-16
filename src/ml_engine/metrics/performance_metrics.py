"""
Performance Metrics Calculation for ML Traffic Signal Optimization
Comprehensive metrics for evaluating optimization performance and system effectiveness
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import math

from config import PerformanceMetricsConfig, get_config
from data import TrafficData


class MetricType(Enum):
    """Types of performance metrics"""
    WAIT_TIME = "wait_time"
    THROUGHPUT = "throughput"
    FUEL_CONSUMPTION = "fuel_consumption"
    EMISSIONS = "emissions"
    SAFETY = "safety"
    COMFORT = "comfort"
    EFFICIENCY = "efficiency"
    EQUITY = "equity"


@dataclass
class MetricValue:
    """Individual metric value with metadata"""
    metric_type: MetricType
    value: float
    unit: str
    timestamp: datetime
    intersection_id: str
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Complete performance snapshot at a point in time"""
    timestamp: datetime
    intersection_id: str
    metrics: Dict[MetricType, MetricValue]
    overall_score: float
    optimization_algorithm: str
    traffic_conditions: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'intersection_id': self.intersection_id,
            'overall_score': self.overall_score,
            'optimization_algorithm': self.optimization_algorithm,
            'traffic_conditions': self.traffic_conditions,
            'metrics': {
                metric_type.value: {
                    'value': metric.value,
                    'unit': metric.unit,
                    'confidence': metric.confidence,
                    'metadata': metric.metadata
                }
                for metric_type, metric in self.metrics.items()
            }
        }


class PerformanceMetricsCalculator:
    """
    Comprehensive performance metrics calculator for traffic signal optimization
    
    Calculates various performance metrics including:
    - Wait time reduction
    - Throughput improvement
    - Fuel consumption reduction
    - Emissions reduction
    - Safety indicators
    - Comfort metrics
    """
    
    def __init__(self, config: Optional[PerformanceMetricsConfig] = None):
        self.config = config or get_config().performance_metrics
        self.logger = logging.getLogger(__name__)
        
        # Metric weights
        self.weights = {
            MetricType.WAIT_TIME: self.config.wait_time_weight,
            MetricType.THROUGHPUT: self.config.throughput_weight,
            MetricType.FUEL_CONSUMPTION: self.config.fuel_consumption_weight,
            MetricType.EMISSIONS: self.config.emission_weight,
            MetricType.SAFETY: self.config.safety_weight,
            MetricType.COMFORT: self.config.comfort_weight
        }
        
        # Historical data for comparison
        self.baseline_metrics: List[PerformanceSnapshot] = []
        self.optimized_metrics: List[PerformanceSnapshot] = []
        
        # Performance tracking
        self.improvement_history: List[Dict[str, Any]] = []
        
        # Constants for calculations
        self.vehicle_length = 4.5  # meters
        self.vehicle_width = 2.0   # meters
        self.fuel_consumption_idle = 0.5  # liters per hour
        self.fuel_consumption_moving = 0.3  # liters per hour per km
        self.emission_factor_co2 = 2.3  # kg CO2 per liter fuel
        self.emission_factor_nox = 0.02  # kg NOx per liter fuel
        
    def calculate_wait_time_metrics(self, traffic_data: TrafficData, 
                                  signal_timings: Dict[str, int]) -> MetricValue:
        """
        Calculate wait time related metrics
        
        Args:
            traffic_data: Current traffic data
            signal_timings: Current signal timings
            
        Returns:
            Wait time metric value
        """
        # Calculate average wait time based on queue lengths and cycle time
        total_queue = sum(traffic_data.lane_counts.values())
        cycle_time = sum(signal_timings.values())
        
        if cycle_time == 0:
            avg_wait_time = 0.0
        else:
            # Simplified wait time calculation
            # In practice, this would use more sophisticated queuing theory
            avg_wait_time = (total_queue * cycle_time) / (4 * 60)  # seconds
        
        # Calculate wait time per vehicle
        if total_queue > 0:
            wait_time_per_vehicle = avg_wait_time / total_queue
        else:
            wait_time_per_vehicle = 0.0
        
        return MetricValue(
            metric_type=MetricType.WAIT_TIME,
            value=wait_time_per_vehicle,
            unit="seconds per vehicle",
            timestamp=traffic_data.timestamp,
            intersection_id=traffic_data.intersection_id,
            confidence=0.8,
            metadata={
                'total_queue': total_queue,
                'cycle_time': cycle_time,
                'avg_wait_time': avg_wait_time
            }
        )
    
    def calculate_throughput_metrics(self, traffic_data: TrafficData, 
                                   signal_timings: Dict[str, int]) -> MetricValue:
        """
        Calculate throughput related metrics
        
        Args:
            traffic_data: Current traffic data
            signal_timings: Current signal timings
            
        Returns:
            Throughput metric value
        """
        # Calculate vehicles processed per hour
        cycle_time = sum(signal_timings.values())
        if cycle_time == 0:
            throughput = 0.0
        else:
            # Calculate throughput based on green time allocation
            total_green_time = sum(signal_timings.values())
            saturation_flow_rate = 1800  # vehicles per hour per lane
            num_lanes = len(signal_timings)
            
            # Effective green time ratio
            green_ratio = total_green_time / cycle_time if cycle_time > 0 else 0
            
            # Throughput calculation
            throughput = saturation_flow_rate * num_lanes * green_ratio
        
        return MetricValue(
            metric_type=MetricType.THROUGHPUT,
            value=throughput,
            unit="vehicles per hour",
            timestamp=traffic_data.timestamp,
            intersection_id=traffic_data.intersection_id,
            confidence=0.9,
            metadata={
                'cycle_time': cycle_time,
                'total_green_time': total_green_time,
                'green_ratio': green_ratio,
                'num_lanes': num_lanes
            }
        )
    
    def calculate_fuel_consumption_metrics(self, traffic_data: TrafficData, 
                                         signal_timings: Dict[str, int]) -> MetricValue:
        """
        Calculate fuel consumption metrics
        
        Args:
            traffic_data: Current traffic data
            signal_timings: Current signal timings
            
        Returns:
            Fuel consumption metric value
        """
        # Calculate fuel consumption based on queue lengths and wait times
        total_queue = sum(traffic_data.lane_counts.values())
        cycle_time = sum(signal_timings.values())
        
        if cycle_time == 0:
            fuel_consumption = 0.0
        else:
            # Calculate average wait time
            avg_wait_time = (total_queue * cycle_time) / (4 * 60)  # seconds
            
            # Fuel consumption calculation
            # Idle fuel consumption (vehicles waiting)
            idle_fuel = total_queue * (avg_wait_time / 3600) * self.fuel_consumption_idle
            
            # Moving fuel consumption (vehicles processing)
            vehicles_processed = min(total_queue, sum(signal_timings.values()) * 0.5)  # Simplified
            moving_fuel = vehicles_processed * 0.1 * self.fuel_consumption_moving  # 0.1 km per vehicle
            
            fuel_consumption = idle_fuel + moving_fuel
        
        return MetricValue(
            metric_type=MetricType.FUEL_CONSUMPTION,
            value=fuel_consumption,
            unit="liters per hour",
            timestamp=traffic_data.timestamp,
            intersection_id=traffic_data.intersection_id,
            confidence=0.7,
            metadata={
                'total_queue': total_queue,
                'avg_wait_time': avg_wait_time,
                'vehicles_processed': vehicles_processed if cycle_time > 0 else 0
            }
        )
    
    def calculate_emissions_metrics(self, traffic_data: TrafficData, 
                                  signal_timings: Dict[str, int]) -> MetricValue:
        """
        Calculate emissions metrics
        
        Args:
            traffic_data: Current traffic data
            signal_timings: Current signal timings
            
        Returns:
            Emissions metric value
        """
        # Calculate CO2 emissions based on fuel consumption
        fuel_metric = self.calculate_fuel_consumption_metrics(traffic_data, signal_timings)
        fuel_consumption = fuel_metric.value
        
        # CO2 emissions calculation
        co2_emissions = fuel_consumption * self.emission_factor_co2
        
        return MetricValue(
            metric_type=MetricType.EMISSIONS,
            value=co2_emissions,
            unit="kg CO2 per hour",
            timestamp=traffic_data.timestamp,
            intersection_id=traffic_data.intersection_id,
            confidence=fuel_metric.confidence,
            metadata={
                'fuel_consumption': fuel_consumption,
                'co2_factor': self.emission_factor_co2,
                'nox_emissions': fuel_consumption * self.emission_factor_nox
            }
        )
    
    def calculate_safety_metrics(self, traffic_data: TrafficData, 
                               signal_timings: Dict[str, int]) -> MetricValue:
        """
        Calculate safety related metrics
        
        Args:
            traffic_data: Current traffic data
            signal_timings: Current signal timings
            
        Returns:
            Safety metric value
        """
        # Calculate safety score based on various factors
        total_queue = sum(traffic_data.lane_counts.values())
        cycle_time = sum(signal_timings.values())
        
        # Safety factors
        safety_score = 1.0
        
        # Queue length factor (longer queues = higher risk)
        if total_queue > 20:
            safety_score *= 0.8
        elif total_queue > 10:
            safety_score *= 0.9
        
        # Cycle time factor (very long cycles = higher risk)
        if cycle_time > 120:
            safety_score *= 0.7
        elif cycle_time > 90:
            safety_score *= 0.9
        
        # Weather factor
        if traffic_data.weather_condition == 'rainy':
            safety_score *= 0.8
        elif traffic_data.weather_condition == 'foggy':
            safety_score *= 0.6
        elif traffic_data.weather_condition == 'stormy':
            safety_score *= 0.5
        
        # Visibility factor
        if traffic_data.visibility is not None and traffic_data.visibility < 5:
            safety_score *= 0.7
        
        return MetricValue(
            metric_type=MetricType.SAFETY,
            value=safety_score,
            unit="safety index (0-1)",
            timestamp=traffic_data.timestamp,
            intersection_id=traffic_data.intersection_id,
            confidence=0.6,
            metadata={
                'total_queue': total_queue,
                'cycle_time': cycle_time,
                'weather_condition': traffic_data.weather_condition,
                'visibility': traffic_data.visibility
            }
        )
    
    def calculate_comfort_metrics(self, traffic_data: TrafficData, 
                                signal_timings: Dict[str, int]) -> MetricValue:
        """
        Calculate comfort related metrics
        
        Args:
            traffic_data: Current traffic data
            signal_timings: Current signal timings
            
        Returns:
            Comfort metric value
        """
        # Calculate comfort score based on smoothness of traffic flow
        total_queue = sum(traffic_data.lane_counts.values())
        cycle_time = sum(signal_timings.values())
        
        # Comfort factors
        comfort_score = 1.0
        
        # Queue length factor (shorter queues = more comfort)
        if total_queue < 5:
            comfort_score *= 1.2
        elif total_queue < 10:
            comfort_score *= 1.0
        else:
            comfort_score *= 0.8
        
        # Cycle time factor (moderate cycles = more comfort)
        if 60 <= cycle_time <= 90:
            comfort_score *= 1.1
        elif cycle_time > 120 or cycle_time < 30:
            comfort_score *= 0.8
        
        # Speed factor (higher speeds = more comfort)
        if traffic_data.avg_speed is not None:
            if traffic_data.avg_speed > 30:
                comfort_score *= 1.1
            elif traffic_data.avg_speed < 15:
                comfort_score *= 0.9
        
        return MetricValue(
            metric_type=MetricType.COMFORT,
            value=comfort_score,
            unit="comfort index (0-1)",
            timestamp=traffic_data.timestamp,
            intersection_id=traffic_data.intersection_id,
            confidence=0.7,
            metadata={
                'total_queue': total_queue,
                'cycle_time': cycle_time,
                'avg_speed': traffic_data.avg_speed
            }
        )
    
    def calculate_efficiency_metrics(self, traffic_data: TrafficData, 
                                   signal_timings: Dict[str, int]) -> MetricValue:
        """
        Calculate overall efficiency metrics
        
        Args:
            traffic_data: Current traffic data
            signal_timings: Current signal timings
            
        Returns:
            Efficiency metric value
        """
        # Calculate efficiency as ratio of actual throughput to maximum possible
        throughput_metric = self.calculate_throughput_metrics(traffic_data, signal_timings)
        actual_throughput = throughput_metric.value
        
        # Maximum possible throughput
        cycle_time = sum(signal_timings.values())
        if cycle_time == 0:
            efficiency = 0.0
        else:
            max_throughput = 1800 * len(signal_timings)  # vehicles per hour
            efficiency = min(1.0, actual_throughput / max_throughput)
        
        return MetricValue(
            metric_type=MetricType.EFFICIENCY,
            value=efficiency,
            unit="efficiency ratio (0-1)",
            timestamp=traffic_data.timestamp,
            intersection_id=traffic_data.intersection_id,
            confidence=0.8,
            metadata={
                'actual_throughput': actual_throughput,
                'max_throughput': max_throughput if cycle_time > 0 else 0,
                'cycle_time': cycle_time
            }
        )
    
    def calculate_comprehensive_metrics(self, traffic_data: TrafficData, 
                                      signal_timings: Dict[str, int], 
                                      algorithm: str = "unknown") -> PerformanceSnapshot:
        """
        Calculate comprehensive performance metrics
        
        Args:
            traffic_data: Current traffic data
            signal_timings: Current signal timings
            algorithm: Optimization algorithm used
            
        Returns:
            Complete performance snapshot
        """
        # Calculate all metrics
        metrics = {
            MetricType.WAIT_TIME: self.calculate_wait_time_metrics(traffic_data, signal_timings),
            MetricType.THROUGHPUT: self.calculate_throughput_metrics(traffic_data, signal_timings),
            MetricType.FUEL_CONSUMPTION: self.calculate_fuel_consumption_metrics(traffic_data, signal_timings),
            MetricType.EMISSIONS: self.calculate_emissions_metrics(traffic_data, signal_timings),
            MetricType.SAFETY: self.calculate_safety_metrics(traffic_data, signal_timings),
            MetricType.COMFORT: self.calculate_comfort_metrics(traffic_data, signal_timings),
            MetricType.EFFICIENCY: self.calculate_efficiency_metrics(traffic_data, signal_timings)
        }
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(metrics)
        
        # Prepare traffic conditions summary
        traffic_conditions = {
            'total_vehicles': sum(traffic_data.lane_counts.values()),
            'avg_speed': traffic_data.avg_speed,
            'weather_condition': traffic_data.weather_condition,
            'congestion_level': traffic_data.congestion_level,
            'temperature': traffic_data.temperature,
            'humidity': traffic_data.humidity,
            'visibility': traffic_data.visibility
        }
        
        snapshot = PerformanceSnapshot(
            timestamp=traffic_data.timestamp,
            intersection_id=traffic_data.intersection_id,
            metrics=metrics,
            overall_score=overall_score,
            optimization_algorithm=algorithm,
            traffic_conditions=traffic_conditions
        )
        
        return snapshot
    
    def _calculate_overall_score(self, metrics: Dict[MetricType, MetricValue]) -> float:
        """Calculate weighted overall performance score"""
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric_type, metric in metrics.items():
            if metric_type in self.weights:
                weight = self.weights[metric_type]
                
                # Normalize metric value (different metrics have different scales)
                normalized_value = self._normalize_metric_value(metric_type, metric.value)
                
                weighted_sum += normalized_value * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def _normalize_metric_value(self, metric_type: MetricType, value: float) -> float:
        """Normalize metric value to 0-1 scale"""
        if metric_type == MetricType.WAIT_TIME:
            # Lower wait time is better, normalize to 0-1 (inverse)
            return max(0, min(1, 1 - (value / 60)))  # Assume max wait time is 60 seconds
        
        elif metric_type == MetricType.THROUGHPUT:
            # Higher throughput is better, normalize to 0-1
            return max(0, min(1, value / 2000))  # Assume max throughput is 2000 vph
        
        elif metric_type == MetricType.FUEL_CONSUMPTION:
            # Lower fuel consumption is better, normalize to 0-1 (inverse)
            return max(0, min(1, 1 - (value / 10)))  # Assume max fuel consumption is 10 L/h
        
        elif metric_type == MetricType.EMISSIONS:
            # Lower emissions is better, normalize to 0-1 (inverse)
            return max(0, min(1, 1 - (value / 20)))  # Assume max emissions is 20 kg CO2/h
        
        elif metric_type in [MetricType.SAFETY, MetricType.COMFORT, MetricType.EFFICIENCY]:
            # These are already in 0-1 scale
            return max(0, min(1, value))
        
        else:
            return 0.0
    
    def compare_with_baseline(self, optimized_snapshot: PerformanceSnapshot, 
                            baseline_snapshot: PerformanceSnapshot) -> Dict[str, Any]:
        """
        Compare optimized performance with baseline
        
        Args:
            optimized_snapshot: Performance snapshot with optimization
            baseline_snapshot: Baseline performance snapshot
            
        Returns:
            Comparison results
        """
        comparison = {
            'timestamp': optimized_snapshot.timestamp.isoformat(),
            'intersection_id': optimized_snapshot.intersection_id,
            'overall_improvement': optimized_snapshot.overall_score - baseline_snapshot.overall_score,
            'metric_improvements': {},
            'percentage_improvements': {}
        }
        
        # Compare individual metrics
        for metric_type in optimized_snapshot.metrics:
            if metric_type in baseline_snapshot.metrics:
                opt_value = optimized_snapshot.metrics[metric_type].value
                base_value = baseline_snapshot.metrics[metric_type].value
                
                improvement = opt_value - base_value
                percentage_improvement = (improvement / base_value * 100) if base_value != 0 else 0
                
                comparison['metric_improvements'][metric_type.value] = improvement
                comparison['percentage_improvements'][metric_type.value] = percentage_improvement
        
        return comparison
    
    def get_performance_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the specified time window"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Filter recent snapshots
        recent_snapshots = [
            snapshot for snapshot in self.optimized_metrics
            if snapshot.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            return {'error': 'No recent performance data available'}
        
        # Calculate summary statistics
        overall_scores = [snapshot.overall_score for snapshot in recent_snapshots]
        
        summary = {
            'time_window_hours': time_window_hours,
            'total_snapshots': len(recent_snapshots),
            'avg_overall_score': np.mean(overall_scores),
            'min_overall_score': np.min(overall_scores),
            'max_overall_score': np.max(overall_scores),
            'std_overall_score': np.std(overall_scores),
            'metric_averages': {},
            'algorithm_performance': {}
        }
        
        # Calculate average metrics
        for metric_type in MetricType:
            values = []
            for snapshot in recent_snapshots:
                if metric_type in snapshot.metrics:
                    values.append(snapshot.metrics[metric_type].value)
            
            if values:
                summary['metric_averages'][metric_type.value] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Calculate algorithm performance
        algorithms = set(snapshot.optimization_algorithm for snapshot in recent_snapshots)
        for algorithm in algorithms:
            alg_snapshots = [s for s in recent_snapshots if s.optimization_algorithm == algorithm]
            alg_scores = [s.overall_score for s in alg_snapshots]
            
            summary['algorithm_performance'][algorithm] = {
                'count': len(alg_snapshots),
                'avg_score': np.mean(alg_scores),
                'std_score': np.std(alg_scores)
            }
        
        return summary
    
    def add_performance_snapshot(self, snapshot: PerformanceSnapshot, is_baseline: bool = False):
        """Add a performance snapshot to the history"""
        if is_baseline:
            self.baseline_metrics.append(snapshot)
        else:
            self.optimized_metrics.append(snapshot)
        
        # Keep only recent data (last 1000 snapshots)
        if len(self.optimized_metrics) > 1000:
            self.optimized_metrics = self.optimized_metrics[-1000:]
        if len(self.baseline_metrics) > 1000:
            self.baseline_metrics = self.baseline_metrics[-1000:]
    
    def export_metrics_to_csv(self, filepath: str, time_window_hours: int = 24):
        """Export performance metrics to CSV file"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Filter recent snapshots
        recent_snapshots = [
            snapshot for snapshot in self.optimized_metrics
            if snapshot.timestamp >= cutoff_time
        ]
        
        if not recent_snapshots:
            self.logger.warning("No recent performance data to export")
            return
        
        # Prepare data for CSV
        data = []
        for snapshot in recent_snapshots:
            row = {
                'timestamp': snapshot.timestamp.isoformat(),
                'intersection_id': snapshot.intersection_id,
                'overall_score': snapshot.overall_score,
                'optimization_algorithm': snapshot.optimization_algorithm
            }
            
            # Add individual metrics
            for metric_type, metric in snapshot.metrics.items():
                row[f"{metric_type.value}_value"] = metric.value
                row[f"{metric_type.value}_unit"] = metric.unit
                row[f"{metric_type.value}_confidence"] = metric.confidence
            
            data.append(row)
        
        # Create DataFrame and save
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        self.logger.info(f"Performance metrics exported to {filepath}")



