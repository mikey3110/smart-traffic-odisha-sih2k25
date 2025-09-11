"""
Enhanced Webster's Formula with ML Predictions for Traffic Signal Optimization
Implements the classic Webster's formula enhanced with machine learning predictions
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import math

from config import WebstersFormulaConfig, get_config
from data import TrafficData


@dataclass
class PhaseData:
    """Data for a signal phase"""
    approach: str
    lane_count: int
    flow_rate: float  # vehicles per hour
    saturation_flow_rate: float  # vehicles per hour per lane
    lost_time: float  # seconds
    critical_flow_ratio: float


@dataclass
class CycleTiming:
    """Complete cycle timing information"""
    cycle_time: float
    phase_timings: Dict[str, float]
    lost_time: float
    efficiency: float
    capacity: float


class WebstersFormulaOptimizer:
    """
    Enhanced Webster's Formula optimizer with ML predictions
    
    Webster's formula is a classical method for calculating optimal signal timing
    based on traffic flow characteristics. This implementation enhances it with:
    - ML-based traffic flow predictions
    - Dynamic adjustment based on real-time conditions
    - Weather and environmental factor consideration
    - Historical pattern analysis
    """
    
    def __init__(self, config: Optional[WebstersFormulaConfig] = None):
        self.config = config or get_config().websters_formula
        self.logger = logging.getLogger(__name__)
        
        # Webster's formula parameters
        self.base_cycle_time = self.config.base_cycle_time
        self.min_cycle_time = self.config.min_cycle_time
        self.max_cycle_time = self.config.max_cycle_time
        self.lost_time = self.config.lost_time
        self.saturation_flow_rate = self.config.saturation_flow_rate
        self.critical_flow_ratio = self.config.critical_flow_ratio
        
        # ML enhancement parameters
        self.ml_enhancement = self.config.ml_enhancement
        self.prediction_horizon = self.config.prediction_horizon
        
        # Historical data for ML predictions
        self.historical_flows: List[Dict[str, float]] = []
        self.historical_cycle_times: List[float] = []
        
        # Performance tracking
        self.optimization_history: List[Dict[str, Any]] = []
        
    def _estimate_flow_rates(self, traffic_data: TrafficData) -> Dict[str, float]:
        """
        Estimate flow rates for each approach using ML predictions
        
        Args:
            traffic_data: Current traffic data
            
        Returns:
            Dictionary of flow rates for each approach
        """
        # Base flow rate estimation from current counts
        base_flows = {}
        for approach, count in traffic_data.lane_counts.items():
            # Convert lane count to flow rate (vehicles per hour)
            # Assume counts are over 1 minute, so multiply by 60
            base_flows[approach] = count * 60
        
        if not self.ml_enhancement:
            return base_flows
        
        # ML-enhanced flow prediction
        enhanced_flows = self._ml_enhance_flow_prediction(traffic_data, base_flows)
        return enhanced_flows
    
    def _ml_enhance_flow_prediction(self, traffic_data: TrafficData, base_flows: Dict[str, float]) -> Dict[str, float]:
        """
        Enhance flow prediction using ML techniques
        
        This is a simplified ML enhancement. In practice, this would use
        trained models for traffic flow prediction.
        """
        enhanced_flows = base_flows.copy()
        
        # Time-based adjustment
        hour = traffic_data.timestamp.hour
        day_of_week = traffic_data.timestamp.weekday()
        
        # Rush hour patterns
        if 7 <= hour <= 9:  # Morning rush
            time_factor = 1.5
        elif 17 <= hour <= 19:  # Evening rush
            time_factor = 1.3
        elif 10 <= hour <= 16:  # Daytime
            time_factor = 1.1
        else:  # Night time
            time_factor = 0.7
        
        # Weekend adjustment
        if day_of_week >= 5:  # Weekend
            time_factor *= 0.8
        
        # Weather adjustment
        weather_factor = 1.0
        if traffic_data.weather_condition == 'rainy':
            weather_factor = 0.8
        elif traffic_data.weather_condition == 'foggy':
            weather_factor = 0.6
        elif traffic_data.weather_condition == 'stormy':
            weather_factor = 0.5
        elif traffic_data.weather_condition == 'clear':
            weather_factor = 1.1
        
        # Environmental factors
        if traffic_data.temperature is not None:
            if traffic_data.temperature < 0 or traffic_data.temperature > 35:
                weather_factor *= 0.9
        
        if traffic_data.visibility is not None and traffic_data.visibility < 5:
            weather_factor *= 0.8
        
        # Apply adjustments
        for approach in enhanced_flows:
            enhanced_flows[approach] *= time_factor * weather_factor
        
        # Smooth with historical data
        if self.historical_flows:
            # Simple exponential smoothing
            alpha = 0.3
            for approach in enhanced_flows:
                if approach in self.historical_flows[-1]:
                    historical_flow = self.historical_flows[-1][approach]
                    enhanced_flows[approach] = alpha * enhanced_flows[approach] + (1 - alpha) * historical_flow
        
        return enhanced_flows
    
    def _calculate_critical_flow_ratios(self, flow_rates: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate critical flow ratios for each approach
        
        Critical flow ratio = flow_rate / saturation_flow_rate
        """
        critical_ratios = {}
        
        for approach, flow_rate in flow_rates.items():
            # Estimate number of lanes (simplified)
            num_lanes = 1  # Assume single lane per approach
            if 'north' in approach or 'south' in approach:
                num_lanes = 2  # Assume 2 lanes for major approaches
            
            saturation_flow = self.saturation_flow_rate * num_lanes
            critical_ratio = min(flow_rate / saturation_flow, self.critical_flow_ratio)
            critical_ratios[approach] = critical_ratio
        
        return critical_ratios
    
    def _websters_cycle_time(self, critical_flow_ratios: Dict[str, float]) -> float:
        """
        Calculate optimal cycle time using Webster's formula
        
        C = (1.5 * L + 5) / (1 - Y)
        where:
        - C = cycle time
        - L = total lost time
        - Y = sum of critical flow ratios
        """
        # Calculate total critical flow ratio
        Y = sum(critical_flow_ratios.values())
        
        if Y >= 1.0:
            self.logger.warning("Critical flow ratio >= 1.0, using maximum cycle time")
            return self.max_cycle_time
        
        # Webster's formula
        cycle_time = (1.5 * self.lost_time + 5) / (1 - Y)
        
        # Apply constraints
        cycle_time = max(self.min_cycle_time, min(self.max_cycle_time, cycle_time))
        
        return cycle_time
    
    def _allocate_green_times(self, cycle_time: float, critical_flow_ratios: Dict[str, float]) -> Dict[str, float]:
        """
        Allocate green times to each approach based on critical flow ratios
        
        Green time = (Critical flow ratio / Total critical flow ratio) * (Cycle time - Lost time)
        """
        total_critical_ratio = sum(critical_flow_ratios.values())
        
        if total_critical_ratio == 0:
            # Equal allocation if no flow data
            available_time = cycle_time - self.lost_time
            num_approaches = len(critical_flow_ratios)
            return {approach: available_time / num_approaches for approach in critical_flow_ratios}
        
        green_times = {}
        available_time = cycle_time - self.lost_time
        
        for approach, critical_ratio in critical_flow_ratios.items():
            green_time = (critical_ratio / total_critical_ratio) * available_time
            green_times[approach] = max(5, green_time)  # Minimum 5 seconds
        
        return green_times
    
    def _calculate_cycle_efficiency(self, cycle_time: float, critical_flow_ratios: Dict[str, float]) -> float:
        """
        Calculate cycle efficiency
        
        Efficiency = (1 - Lost time / Cycle time) * (1 - Sum of critical flow ratios)
        """
        if cycle_time <= 0:
            return 0.0
        
        lost_time_ratio = self.lost_time / cycle_time
        critical_flow_sum = sum(critical_flow_ratios.values())
        
        efficiency = (1 - lost_time_ratio) * (1 - critical_flow_sum)
        return max(0.0, efficiency)
    
    def _calculate_cycle_capacity(self, green_times: Dict[str, float], flow_rates: Dict[str, float]) -> float:
        """
        Calculate cycle capacity (vehicles per hour)
        """
        total_capacity = 0.0
        
        for approach, green_time in green_times.items():
            if approach in flow_rates:
                # Capacity = flow rate * green time / cycle time
                approach_capacity = flow_rates[approach] * green_time / sum(green_times.values())
                total_capacity += approach_capacity
        
        return total_capacity
    
    def optimize_signal_timing(self, traffic_data: TrafficData, current_timings: Dict[str, int]) -> Dict[str, int]:
        """
        Optimize signal timing using enhanced Webster's formula
        
        Args:
            traffic_data: Current traffic data
            current_timings: Current signal timings
            
        Returns:
            Optimized signal timings
        """
        self.logger.info("Starting Webster's formula optimization")
        
        # Estimate flow rates
        flow_rates = self._estimate_flow_rates(traffic_data)
        self.logger.debug(f"Estimated flow rates: {flow_rates}")
        
        # Calculate critical flow ratios
        critical_flow_ratios = self._calculate_critical_flow_ratios(flow_rates)
        self.logger.debug(f"Critical flow ratios: {critical_flow_ratios}")
        
        # Calculate optimal cycle time
        cycle_time = self._websters_cycle_time(critical_flow_ratios)
        self.logger.debug(f"Optimal cycle time: {cycle_time:.1f} seconds")
        
        # Allocate green times
        green_times = self._allocate_green_times(cycle_time, critical_flow_ratios)
        self.logger.debug(f"Allocated green times: {green_times}")
        
        # Calculate cycle efficiency and capacity
        efficiency = self._calculate_cycle_efficiency(cycle_time, critical_flow_ratios)
        capacity = self._calculate_cycle_capacity(green_times, flow_rates)
        
        # Convert to integer timings
        optimized_timings = {}
        for approach, green_time in green_times.items():
            optimized_timings[approach] = int(round(green_time))
        
        # Store optimization results
        optimization_result = {
            'timestamp': traffic_data.timestamp,
            'cycle_time': cycle_time,
            'efficiency': efficiency,
            'capacity': capacity,
            'flow_rates': flow_rates,
            'critical_flow_ratios': critical_flow_ratios,
            'green_times': green_times
        }
        self.optimization_history.append(optimization_result)
        
        # Update historical data
        self.historical_flows.append(flow_rates)
        self.historical_cycle_times.append(cycle_time)
        
        # Keep only recent history (last 100 optimizations)
        if len(self.historical_flows) > 100:
            self.historical_flows = self.historical_flows[-100:]
            self.historical_cycle_times = self.historical_cycle_times[-100:]
        
        self.logger.info(f"Webster's optimization completed: cycle_time={cycle_time:.1f}s, efficiency={efficiency:.3f}")
        
        return optimized_timings
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.optimization_history:
            return {}
        
        recent_optimizations = self.optimization_history[-10:]  # Last 10 optimizations
        
        avg_cycle_time = np.mean([opt['cycle_time'] for opt in recent_optimizations])
        avg_efficiency = np.mean([opt['efficiency'] for opt in recent_optimizations])
        avg_capacity = np.mean([opt['capacity'] for opt in recent_optimizations])
        
        return {
            'total_optimizations': len(self.optimization_history),
            'avg_cycle_time': avg_cycle_time,
            'avg_efficiency': avg_efficiency,
            'avg_capacity': avg_capacity,
            'ml_enhancement_enabled': self.ml_enhancement,
            'prediction_horizon': self.prediction_horizon,
            'base_cycle_time': self.base_cycle_time,
            'saturation_flow_rate': self.saturation_flow_rate
        }
    
    def reset_historical_data(self):
        """Reset historical data"""
        self.historical_flows = []
        self.historical_cycle_times = []
        self.optimization_history = []
        self.logger.info("Historical data reset")
    
    def get_flow_prediction_accuracy(self) -> float:
        """
        Calculate flow prediction accuracy based on historical data
        
        This is a simplified accuracy calculation. In practice, this would
        compare predicted flows with actual observed flows.
        """
        if len(self.historical_flows) < 2:
            return 0.0
        
        # Simple accuracy based on consistency of predictions
        recent_flows = self.historical_flows[-10:]
        if len(recent_flows) < 2:
            return 0.0
        
        # Calculate variance in predictions (lower variance = higher accuracy)
        total_variance = 0.0
        num_approaches = len(recent_flows[0])
        
        for approach in recent_flows[0]:
            flows = [flow[approach] for flow in recent_flows if approach in flow]
            if len(flows) > 1:
                variance = np.var(flows)
                total_variance += variance
        
        # Convert variance to accuracy (simplified)
        avg_variance = total_variance / num_approaches if num_approaches > 0 else 1.0
        accuracy = max(0.0, 1.0 - (avg_variance / 1000.0))  # Normalize variance
        
        return accuracy


