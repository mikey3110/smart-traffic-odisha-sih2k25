"""
Enhanced Performance Metrics for ML Traffic Signal Optimization
Comprehensive metrics calculation for wait time reduction, throughput improvement, and system efficiency
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import defaultdict, deque

from config.ml_config import PerformanceMetricsConfig, get_config


class MetricType(Enum):
    """Types of performance metrics"""
    WAIT_TIME = "wait_time"
    THROUGHPUT = "throughput"
    FUEL_CONSUMPTION = "fuel_consumption"
    EMISSIONS = "emissions"
    SAFETY = "safety"
    COMFORT = "comfort"
    EFFICIENCY = "efficiency"
    QUEUE_LENGTH = "queue_length"
    DELAY = "delay"
    STOP_DELAY = "stop_delay"


@dataclass
class TrafficMetrics:
    """Traffic performance metrics for a single measurement"""
    timestamp: datetime
    intersection_id: str
    wait_time: float = 0.0
    throughput: float = 0.0
    fuel_consumption: float = 0.0
    emissions: float = 0.0
    safety_score: float = 0.0
    comfort_score: float = 0.0
    efficiency: float = 0.0
    queue_length: float = 0.0
    delay: float = 0.0
    stop_delay: float = 0.0
    total_vehicles: int = 0
    processed_vehicles: int = 0
    avg_speed: float = 0.0
    signal_cycle_time: float = 0.0
    green_time_ratio: float = 0.0


@dataclass
class OptimizationResult:
    """Result of signal optimization"""
    intersection_id: str
    timestamp: datetime
    algorithm_used: str
    before_metrics: TrafficMetrics
    after_metrics: TrafficMetrics
    improvement_percentage: Dict[str, float]
    optimization_time: float
    confidence: float


@dataclass
class PerformanceSummary:
    """Summary of performance metrics over a time period"""
    intersection_id: str
    start_time: datetime
    end_time: datetime
    total_measurements: int
    avg_wait_time: float
    avg_throughput: float
    avg_fuel_consumption: float
    avg_emissions: float
    avg_safety_score: float
    avg_comfort_score: float
    avg_efficiency: float
    total_vehicles_processed: int
    improvement_over_baseline: Dict[str, float]
    peak_performance_time: Optional[datetime] = None
    worst_performance_time: Optional[datetime] = None


class TrafficFlowCalculator:
    """Calculate traffic flow characteristics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_wait_time(self, queue_length: float, departure_rate: float) -> float:
        """Calculate average wait time for vehicles in queue"""
        if departure_rate <= 0:
            return float('inf')
        return queue_length / departure_rate
    
    def calculate_throughput(self, processed_vehicles: int, time_period: float) -> float:
        """Calculate throughput (vehicles per hour)"""
        if time_period <= 0:
            return 0.0
        return (processed_vehicles / time_period) * 3600  # Convert to per hour
    
    def calculate_fuel_consumption(self, queue_length: float, avg_speed: float, 
                                 idle_fuel_rate: float = 0.5, moving_fuel_rate: float = 0.1) -> float:
        """Calculate fuel consumption based on queue length and speed"""
        # Idle fuel consumption (vehicles in queue)
        idle_consumption = queue_length * idle_fuel_rate
        
        # Moving fuel consumption (based on speed)
        moving_consumption = max(0, avg_speed * moving_fuel_rate)
        
        return idle_consumption + moving_consumption
    
    def calculate_emissions(self, fuel_consumption: float, emission_factor: float = 2.3) -> float:
        """Calculate CO2 emissions based on fuel consumption"""
        return fuel_consumption * emission_factor
    
    def calculate_safety_score(self, avg_speed: float, queue_length: float, 
                              weather_condition: str) -> float:
        """Calculate safety score (0-1, higher is safer)"""
        # Base safety score
        safety = 1.0
        
        # Speed factor (lower speed = safer)
        if avg_speed > 50:
            safety *= 0.7
        elif avg_speed > 40:
            safety *= 0.85
        elif avg_speed < 10:
            safety *= 0.9  # Very slow can be unsafe too
        
        # Queue length factor (longer queues = less safe)
        if queue_length > 20:
            safety *= 0.8
        elif queue_length > 10:
            safety *= 0.9
        
        # Weather factor
        weather_factors = {
            'clear': 1.0,
            'cloudy': 0.95,
            'rainy': 0.8,
            'foggy': 0.6,
            'stormy': 0.5,
            'snowy': 0.4
        }
        safety *= weather_factors.get(weather_condition, 1.0)
        
        return max(0.0, min(1.0, safety))
    
    def calculate_comfort_score(self, avg_speed: float, stop_frequency: float, 
                               acceleration_changes: int) -> float:
        """Calculate comfort score (0-1, higher is more comfortable)"""
        comfort = 1.0
        
        # Speed factor (moderate speed = comfortable)
        if avg_speed < 15 or avg_speed > 45:
            comfort *= 0.8
        elif avg_speed < 20 or avg_speed > 40:
            comfort *= 0.9
        
        # Stop frequency factor
        if stop_frequency > 0.5:  # More than 50% of time stopped
            comfort *= 0.7
        elif stop_frequency > 0.3:
            comfort *= 0.85
        
        # Acceleration changes factor
        if acceleration_changes > 10:
            comfort *= 0.8
        elif acceleration_changes > 5:
            comfort *= 0.9
        
        return max(0.0, min(1.0, comfort))
    
    def calculate_efficiency(self, throughput: float, wait_time: float, 
                           fuel_consumption: float) -> float:
        """Calculate overall efficiency score (0-1, higher is more efficient)"""
        # Normalize metrics (higher throughput and lower wait/fuel = better)
        throughput_score = min(1.0, throughput / 1000)  # Normalize to 1000 vehicles/hour
        wait_score = max(0.0, 1.0 - (wait_time / 60))  # Normalize to 60 seconds
        fuel_score = max(0.0, 1.0 - (fuel_consumption / 100))  # Normalize to 100 units
        
        # Weighted average
        efficiency = (0.4 * throughput_score + 0.4 * wait_score + 0.2 * fuel_score)
        return max(0.0, min(1.0, efficiency))


class EnhancedPerformanceMetrics:
    """
    Enhanced performance metrics calculator for traffic signal optimization
    
    Features:
    - Comprehensive traffic flow metrics
    - Real-time performance monitoring
    - Historical trend analysis
    - Optimization impact assessment
    - Multi-dimensional performance evaluation
    - Statistical analysis and reporting
    """
    
    def __init__(self, config: Optional[PerformanceMetricsConfig] = None):
        self.config = config or get_config().performance_metrics
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.flow_calculator = TrafficFlowCalculator()
        
        # Data storage
        self.metrics_history: List[TrafficMetrics] = []
        self.optimization_results: List[OptimizationResult] = []
        self.baseline_metrics: Dict[str, TrafficMetrics] = {}
        
        # Performance tracking
        self.current_metrics: Dict[str, TrafficMetrics] = {}
        self.performance_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Statistical analysis
        self.performance_trends: Dict[str, List[float]] = defaultdict(list)
        
        self.logger.info("Enhanced performance metrics initialized")
    
    def calculate_metrics(self, traffic_data: Dict[str, Any], 
                         signal_timings: Dict[str, int],
                         intersection_id: str) -> TrafficMetrics:
        """Calculate comprehensive performance metrics"""
        timestamp = datetime.now()
        
        # Extract basic data
        lane_counts = traffic_data.get('lane_counts', {})
        total_vehicles = sum(lane_counts.values())
        avg_speed = traffic_data.get('avg_speed', 0.0)
        weather_condition = traffic_data.get('weather_condition', 'clear')
        
        # Calculate queue length (total vehicles waiting)
        queue_length = total_vehicles
        
        # Calculate departure rate (vehicles per second)
        total_green_time = sum(signal_timings.values())
        if total_green_time > 0:
            departure_rate = total_vehicles / total_green_time
        else:
            departure_rate = 0.0
        
        # Calculate wait time
        wait_time = self.flow_calculator.calculate_wait_time(queue_length, departure_rate)
        
        # Calculate throughput
        time_period = 60  # Assume 1-minute measurement period
        processed_vehicles = max(0, total_vehicles - queue_length)
        throughput = self.flow_calculator.calculate_throughput(processed_vehicles, time_period)
        
        # Calculate fuel consumption
        fuel_consumption = self.flow_calculator.calculate_fuel_consumption(
            queue_length, avg_speed
        )
        
        # Calculate emissions
        emissions = self.flow_calculator.calculate_emissions(fuel_consumption)
        
        # Calculate safety score
        safety_score = self.flow_calculator.calculate_safety_score(
            avg_speed, queue_length, weather_condition
        )
        
        # Calculate comfort score
        stop_frequency = 1.0 if avg_speed < 5 else 0.0  # Simplified
        acceleration_changes = 0  # Would need more detailed data
        comfort_score = self.flow_calculator.calculate_comfort_score(
            avg_speed, stop_frequency, acceleration_changes
        )
        
        # Calculate efficiency
        efficiency = self.flow_calculator.calculate_efficiency(
            throughput, wait_time, fuel_consumption
        )
        
        # Calculate delays
        delay = wait_time  # Simplified
        stop_delay = max(0, wait_time - 5)  # Stop delay is wait time minus 5 seconds
        
        # Calculate signal cycle metrics
        signal_cycle_time = sum(signal_timings.values())
        green_time_ratio = sum(signal_timings.values()) / max(signal_cycle_time, 1)
        
        # Create metrics object
        metrics = TrafficMetrics(
            timestamp=timestamp,
            intersection_id=intersection_id,
            wait_time=wait_time,
            throughput=throughput,
            fuel_consumption=fuel_consumption,
            emissions=emissions,
            safety_score=safety_score,
            comfort_score=comfort_score,
            efficiency=efficiency,
            queue_length=queue_length,
            delay=delay,
            stop_delay=stop_delay,
            total_vehicles=total_vehicles,
            processed_vehicles=processed_vehicles,
            avg_speed=avg_speed,
            signal_cycle_time=signal_cycle_time,
            green_time_ratio=green_time_ratio
        )
        
        # Store metrics
        self.metrics_history.append(metrics)
        self.current_metrics[intersection_id] = metrics
        self.performance_windows[intersection_id].append(metrics)
        
        # Update trends
        self._update_trends(intersection_id, metrics)
        
        return metrics
    
    def _update_trends(self, intersection_id: str, metrics: TrafficMetrics):
        """Update performance trends for statistical analysis"""
        trend_metrics = ['wait_time', 'throughput', 'efficiency', 'safety_score']
        
        for metric_name in trend_metrics:
            trend_key = f"{intersection_id}_{metric_name}"
            value = getattr(metrics, metric_name)
            self.performance_trends[trend_key].append(value)
            
            # Keep only recent trends (last 1000 measurements)
            if len(self.performance_trends[trend_key]) > 1000:
                self.performance_trends[trend_key] = self.performance_trends[trend_key][-1000:]
    
    def calculate_optimization_impact(self, before_metrics: TrafficMetrics,
                                    after_metrics: TrafficMetrics,
                                    algorithm_used: str,
                                    optimization_time: float) -> OptimizationResult:
        """Calculate the impact of signal optimization"""
        
        # Calculate improvement percentages
        improvement_percentage = {}
        metric_names = ['wait_time', 'throughput', 'fuel_consumption', 'emissions',
                       'safety_score', 'comfort_score', 'efficiency']
        
        for metric_name in metric_names:
            before_value = getattr(before_metrics, metric_name)
            after_value = getattr(after_metrics, metric_name)
            
            if before_value != 0:
                if metric_name in ['wait_time', 'fuel_consumption', 'emissions']:
                    # For these metrics, lower is better
                    improvement = ((before_value - after_value) / before_value) * 100
                else:
                    # For these metrics, higher is better
                    improvement = ((after_value - before_value) / before_value) * 100
            else:
                improvement = 0.0
            
            improvement_percentage[metric_name] = improvement
        
        # Calculate overall confidence
        confidence = self._calculate_optimization_confidence(before_metrics, after_metrics)
        
        # Create optimization result
        result = OptimizationResult(
            intersection_id=before_metrics.intersection_id,
            timestamp=datetime.now(),
            algorithm_used=algorithm_used,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_percentage=improvement_percentage,
            optimization_time=optimization_time,
            confidence=confidence
        )
        
        # Store result
        self.optimization_results.append(result)
        
        return result
    
    def _calculate_optimization_confidence(self, before_metrics: TrafficMetrics,
                                         after_metrics: TrafficMetrics) -> float:
        """Calculate confidence in optimization results"""
        # Base confidence
        confidence = 0.5
        
        # Check for significant improvements
        significant_improvements = 0
        total_metrics = 0
        
        metric_names = ['wait_time', 'throughput', 'efficiency']
        for metric_name in metric_names:
            before_value = getattr(before_metrics, metric_name)
            after_value = getattr(after_metrics, metric_name)
            
            if before_value != 0:
                if metric_name == 'wait_time':
                    improvement = (before_value - after_value) / before_value
                else:
                    improvement = (after_value - before_value) / before_value
                
                if improvement > 0.1:  # 10% improvement
                    significant_improvements += 1
                total_metrics += 1
        
        if total_metrics > 0:
            confidence += (significant_improvements / total_metrics) * 0.5
        
        return min(1.0, confidence)
    
    def get_performance_summary(self, intersection_id: str, 
                              start_time: Optional[datetime] = None,
                              end_time: Optional[datetime] = None) -> PerformanceSummary:
        """Get performance summary for an intersection over a time period"""
        
        # Filter metrics by time period
        if start_time is None:
            start_time = datetime.now() - timedelta(hours=1)
        if end_time is None:
            end_time = datetime.now()
        
        filtered_metrics = [
            m for m in self.metrics_history
            if (m.intersection_id == intersection_id and
                start_time <= m.timestamp <= end_time)
        ]
        
        if not filtered_metrics:
            return PerformanceSummary(
                intersection_id=intersection_id,
                start_time=start_time,
                end_time=end_time,
                total_measurements=0,
                avg_wait_time=0.0,
                avg_throughput=0.0,
                avg_fuel_consumption=0.0,
                avg_emissions=0.0,
                avg_safety_score=0.0,
                avg_comfort_score=0.0,
                avg_efficiency=0.0,
                total_vehicles_processed=0,
                improvement_over_baseline={}
            )
        
        # Calculate averages
        avg_wait_time = np.mean([m.wait_time for m in filtered_metrics])
        avg_throughput = np.mean([m.throughput for m in filtered_metrics])
        avg_fuel_consumption = np.mean([m.fuel_consumption for m in filtered_metrics])
        avg_emissions = np.mean([m.emissions for m in filtered_metrics])
        avg_safety_score = np.mean([m.safety_score for m in filtered_metrics])
        avg_comfort_score = np.mean([m.comfort_score for m in filtered_metrics])
        avg_efficiency = np.mean([m.efficiency for m in filtered_metrics])
        
        total_vehicles_processed = sum([m.processed_vehicles for m in filtered_metrics])
        
        # Find peak and worst performance times
        efficiency_scores = [m.efficiency for m in filtered_metrics]
        if efficiency_scores:
            peak_idx = np.argmax(efficiency_scores)
            worst_idx = np.argmin(efficiency_scores)
            peak_performance_time = filtered_metrics[peak_idx].timestamp
            worst_performance_time = filtered_metrics[worst_idx].timestamp
        else:
            peak_performance_time = None
            worst_performance_time = None
        
        # Calculate improvement over baseline
        improvement_over_baseline = {}
        if intersection_id in self.baseline_metrics:
            baseline = self.baseline_metrics[intersection_id]
            improvement_over_baseline = {
                'wait_time': ((baseline.wait_time - avg_wait_time) / baseline.wait_time) * 100,
                'throughput': ((avg_throughput - baseline.throughput) / baseline.throughput) * 100,
                'efficiency': ((avg_efficiency - baseline.efficiency) / baseline.efficiency) * 100
            }
        
        return PerformanceSummary(
            intersection_id=intersection_id,
            start_time=start_time,
            end_time=end_time,
            total_measurements=len(filtered_metrics),
            avg_wait_time=avg_wait_time,
            avg_throughput=avg_throughput,
            avg_fuel_consumption=avg_fuel_consumption,
            avg_emissions=avg_emissions,
            avg_safety_score=avg_safety_score,
            avg_comfort_score=avg_comfort_score,
            avg_efficiency=avg_efficiency,
            total_vehicles_processed=total_vehicles_processed,
            improvement_over_baseline=improvement_over_baseline,
            peak_performance_time=peak_performance_time,
            worst_performance_time=worst_performance_time
        )
    
    def set_baseline_metrics(self, intersection_id: str, metrics: TrafficMetrics):
        """Set baseline metrics for comparison"""
        self.baseline_metrics[intersection_id] = metrics
        self.logger.info(f"Baseline metrics set for {intersection_id}")
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get statistics about optimization results"""
        if not self.optimization_results:
            return {}
        
        # Calculate average improvements
        avg_improvements = {}
        metric_names = ['wait_time', 'throughput', 'efficiency', 'safety_score']
        
        for metric_name in metric_names:
            improvements = [r.improvement_percentage.get(metric_name, 0) 
                          for r in self.optimization_results]
            avg_improvements[metric_name] = np.mean(improvements)
        
        # Calculate success rate (improvements > 0)
        successful_optimizations = sum(1 for r in self.optimization_results 
                                     if any(imp > 0 for imp in r.improvement_percentage.values()))
        success_rate = successful_optimizations / len(self.optimization_results)
        
        # Calculate average optimization time
        avg_optimization_time = np.mean([r.optimization_time for r in self.optimization_results])
        
        # Calculate average confidence
        avg_confidence = np.mean([r.confidence for r in self.optimization_results])
        
        return {
            'total_optimizations': len(self.optimization_results),
            'successful_optimizations': successful_optimizations,
            'success_rate': success_rate,
            'avg_optimization_time': avg_optimization_time,
            'avg_confidence': avg_confidence,
            'avg_improvements': avg_improvements
        }
    
    def get_trend_analysis(self, intersection_id: str, metric_name: str) -> Dict[str, Any]:
        """Get trend analysis for a specific metric"""
        trend_key = f"{intersection_id}_{metric_name}"
        
        if trend_key not in self.performance_trends:
            return {}
        
        values = self.performance_trends[trend_key]
        
        if len(values) < 2:
            return {}
        
        # Calculate trend statistics
        trend_slope = np.polyfit(range(len(values)), values, 1)[0]
        trend_direction = "improving" if trend_slope > 0 else "degrading" if trend_slope < 0 else "stable"
        
        # Calculate volatility
        volatility = np.std(values)
        
        # Calculate recent vs historical performance
        if len(values) >= 10:
            recent_avg = np.mean(values[-10:])
            historical_avg = np.mean(values[:-10])
            change = ((recent_avg - historical_avg) / historical_avg) * 100
        else:
            change = 0.0
        
        return {
            'trend_slope': trend_slope,
            'trend_direction': trend_direction,
            'volatility': volatility,
            'recent_change_percentage': change,
            'current_value': values[-1] if values else 0,
            'data_points': len(values)
        }
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics to file"""
        export_data = {
            'metrics_history': [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'intersection_id': m.intersection_id,
                    'wait_time': m.wait_time,
                    'throughput': m.throughput,
                    'efficiency': m.efficiency,
                    'safety_score': m.safety_score
                }
                for m in self.metrics_history
            ],
            'optimization_results': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'intersection_id': r.intersection_id,
                    'algorithm_used': r.algorithm_used,
                    'improvement_percentage': r.improvement_percentage,
                    'confidence': r.confidence
                }
                for r in self.optimization_results
            ]
        }
        
        with open(filepath, 'w') as f:
            if format.lower() == "json":
                json.dump(export_data, f, indent=2)
            else:
                # CSV format
                import csv
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'intersection_id', 'wait_time', 'throughput', 'efficiency'])
                for m in self.metrics_history:
                    writer.writerow([m.timestamp.isoformat(), m.intersection_id, 
                                   m.wait_time, m.throughput, m.efficiency])
        
        self.logger.info(f"Metrics exported to {filepath}")
    
    def reset_metrics(self):
        """Reset all metrics data"""
        self.metrics_history.clear()
        self.optimization_results.clear()
        self.current_metrics.clear()
        self.performance_windows.clear()
        self.performance_trends.clear()
        self.baseline_metrics.clear()
        self.logger.info("All metrics data reset")


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the performance metrics
    metrics_calculator = EnhancedPerformanceMetrics()
    
    # Test data
    traffic_data = {
        'lane_counts': {'north_lane': 10, 'south_lane': 8, 'east_lane': 12, 'west_lane': 6},
        'avg_speed': 35.0,
        'weather_condition': 'clear'
    }
    
    signal_timings = {'north_lane': 30, 'south_lane': 30, 'east_lane': 25, 'west_lane': 25}
    
    # Calculate metrics
    print("Calculating performance metrics...")
    metrics = metrics_calculator.calculate_metrics(traffic_data, signal_timings, "junction-1")
    print(f"Metrics: {metrics}")
    
    # Test optimization impact
    print("\nTesting optimization impact...")
    before_metrics = metrics
    after_metrics = TrafficMetrics(
        timestamp=datetime.now(),
        intersection_id="junction-1",
        wait_time=15.0,  # Improved
        throughput=800.0,  # Improved
        efficiency=0.8  # Improved
    )
    
    result = metrics_calculator.calculate_optimization_impact(
        before_metrics, after_metrics, "q_learning", 0.5
    )
    print(f"Optimization result: {result}")
    
    # Test performance summary
    print("\nTesting performance summary...")
    summary = metrics_calculator.get_performance_summary("junction-1")
    print(f"Performance summary: {summary}")
    
    # Test statistics
    print("\nTesting optimization statistics...")
    stats = metrics_calculator.get_optimization_statistics()
    print(f"Optimization statistics: {stats}")
