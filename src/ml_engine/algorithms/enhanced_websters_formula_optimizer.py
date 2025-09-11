"""
Enhanced Webster's Formula Traffic Signal Optimization
ML-enhanced implementation with traffic prediction and adaptive parameters
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from config.ml_config import WebstersFormulaConfig, get_config
from data.enhanced_data_integration import TrafficData
from prediction.traffic_predictor import PredictionResult, TrafficPredictor


@dataclass
class PhaseData:
    """Data for a signal phase"""
    phase_id: int
    approach_lanes: List[str]
    flow_rate: float  # vehicles per hour
    saturation_flow_rate: float  # vehicles per hour
    critical_flow_ratio: float
    green_time: float  # seconds
    lost_time: float  # seconds
    priority_factor: float  # ML-enhanced priority


@dataclass
class CycleOptimization:
    """Webster's formula cycle optimization result"""
    cycle_time: float
    phases: List[PhaseData]
    total_lost_time: float
    critical_flow_ratio: float
    efficiency: float
    ml_adjustments: Dict[str, float]
    confidence_score: float


class TrafficFlowAnalyzer:
    """Traffic flow analysis for Webster's formula"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Traffic flow parameters
        self.saturation_flow_rate = 1800  # vehicles per hour per lane
        self.lost_time_per_phase = 4  # seconds
        self.minimum_green_time = 15  # seconds
        self.maximum_green_time = 60  # seconds
        
        # ML enhancement parameters
        self.prediction_horizon = 15  # minutes
        self.adaptation_factor = 0.1  # How much to adapt based on predictions
        
    def analyze_approach_flows(self, traffic_data: TrafficData, 
                             prediction: Optional[PredictionResult] = None) -> Dict[str, float]:
        """
        Analyze traffic flows for each approach
        
        Args:
            traffic_data: Current traffic data
            prediction: ML prediction for future flows
            
        Returns:
            Dictionary of approach flows (vehicles per hour)
        """
        flows = {}
        
        # Calculate current flows from lane counts
        for lane, count in traffic_data.lane_counts.items():
            # Convert counts to flow rate (vehicles per hour)
            # Assume counts are per minute
            current_flow = count * 60.0
            flows[lane] = current_flow
        
        # Apply ML predictions if available
        if prediction and prediction.confidence > 0.7:
            self.logger.info(f"Applying ML predictions with confidence {prediction.confidence:.2f}")
            
            for lane in flows:
                if lane in prediction.predicted_flows:
                    # Blend current and predicted flows
                    predicted_flow = prediction.predicted_flows[lane]
                    current_flow = flows[lane]
                    
                    # Weighted average based on prediction confidence
                    weight = prediction.confidence * self.adaptation_factor
                    flows[lane] = (1 - weight) * current_flow + weight * predicted_flow
                    
                    self.logger.debug(f"{lane}: current={current_flow:.1f}, predicted={predicted_flow:.1f}, "
                                    f"blended={flows[lane]:.1f}")
        
        return flows
    
    def calculate_saturation_flow_rates(self, traffic_data: TrafficData) -> Dict[str, float]:
        """
        Calculate saturation flow rates for each approach
        
        Args:
            traffic_data: Current traffic data
            
        Returns:
            Dictionary of saturation flow rates
        """
        saturation_rates = {}
        
        for lane in traffic_data.lane_counts:
            # Base saturation flow rate
            base_rate = self.saturation_flow_rate
            
            # Adjust for weather conditions
            weather_factor = self._get_weather_factor(traffic_data.weather_condition)
            
            # Adjust for time of day
            time_factor = self._get_time_factor(traffic_data.timestamp)
            
            # Adjust for congestion level
            congestion_factor = self._get_congestion_factor(traffic_data.lane_counts[lane])
            
            # Calculate adjusted saturation flow rate
            adjusted_rate = base_rate * weather_factor * time_factor * congestion_factor
            saturation_rates[lane] = adjusted_rate
        
        return saturation_rates
    
    def _get_weather_factor(self, weather_condition: str) -> float:
        """Get weather adjustment factor for saturation flow rate"""
        weather_factors = {
            'clear': 1.0,
            'cloudy': 0.95,
            'rainy': 0.85,
            'foggy': 0.75,
            'stormy': 0.70,
            'snowy': 0.60
        }
        return weather_factors.get(weather_condition, 1.0)
    
    def _get_time_factor(self, timestamp: datetime) -> float:
        """Get time of day adjustment factor"""
        hour = timestamp.hour
        
        # Rush hour periods have higher saturation flow rates
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            return 1.1  # Rush hour
        elif 10 <= hour <= 16:
            return 1.0  # Normal hours
        else:
            return 0.9  # Off-peak hours
    
    def _get_congestion_factor(self, vehicle_count: int) -> float:
        """Get congestion adjustment factor"""
        if vehicle_count < 5:
            return 0.9  # Low traffic
        elif vehicle_count < 15:
            return 1.0  # Normal traffic
        elif vehicle_count < 25:
            return 1.05  # High traffic
        else:
            return 1.1  # Very high traffic


class EnhancedWebstersFormulaOptimizer:
    """
    Enhanced Webster's Formula optimizer with ML predictions
    
    Features:
    - ML-enhanced traffic flow prediction
    - Adaptive parameter adjustment
    - Multi-objective optimization
    - Real-time adaptation
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[WebstersFormulaConfig] = None):
        self.config = config or get_config().websters_formula
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.flow_analyzer = TrafficFlowAnalyzer()
        self.traffic_predictor = None  # Will be set by external predictor
        
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
        
        # Phase definitions
        self.phase_definitions = {
            0: ['north_lane', 'south_lane'],  # North-South
            1: ['east_lane', 'west_lane'],    # East-West
            2: ['north_lane'],                # North only
            3: ['south_lane']                 # South only
        }
        
        # Optimization history
        self.optimization_history = []
        self.performance_metrics = []
        
        self.logger.info("Enhanced Webster's Formula optimizer initialized")
    
    def set_traffic_predictor(self, predictor: TrafficPredictor):
        """Set the traffic predictor for ML enhancement"""
        self.traffic_predictor = predictor
        self.logger.info("Traffic predictor set for ML enhancement")
    
    def optimize_signal_timing(self, traffic_data: TrafficData, 
                             current_timings: Dict[str, int],
                             historical_data: List[TrafficData] = None) -> Dict[str, int]:
        """
        Optimize signal timing using enhanced Webster's formula
        
        Args:
            traffic_data: Current traffic data
            current_timings: Current signal timings
            historical_data: Historical data for ML prediction
            
        Returns:
            Optimized signal timings
        """
        start_time = datetime.now()
        
        # Get ML prediction if available
        prediction = None
        if self.ml_enhancement and self.traffic_predictor and historical_data:
            try:
                prediction = self.traffic_predictor.predict(traffic_data)
                self.logger.info(f"ML prediction obtained with confidence {prediction.confidence:.2f}")
            except Exception as e:
                self.logger.warning(f"ML prediction failed: {e}")
        
        # Analyze traffic flows
        approach_flows = self.flow_analyzer.analyze_approach_flows(traffic_data, prediction)
        saturation_rates = self.flow_analyzer.calculate_saturation_flow_rates(traffic_data)
        
        # Calculate critical flow ratios
        critical_ratios = self._calculate_critical_flow_ratios(approach_flows, saturation_rates)
        
        # Optimize cycle time
        cycle_optimization = self._optimize_cycle_time(critical_ratios, approach_flows, prediction)
        
        # Calculate phase timings
        phase_timings = self._calculate_phase_timings(cycle_optimization, approach_flows)
        
        # Convert to signal timings
        optimized_timings = self._convert_to_signal_timings(phase_timings, current_timings)
        
        # Record optimization
        optimization_time = (datetime.now() - start_time).total_seconds()
        self._record_optimization(traffic_data, cycle_optimization, optimization_time)
        
        self.logger.info(
            f"Webster's formula optimization completed in {optimization_time:.3f}s. "
            f"Cycle time: {cycle_optimization.cycle_time:.1f}s, "
            f"Efficiency: {cycle_optimization.efficiency:.2f}"
        )
        
        return optimized_timings
    
    def _calculate_critical_flow_ratios(self, approach_flows: Dict[str, float], 
                                      saturation_rates: Dict[str, float]) -> Dict[int, float]:
        """Calculate critical flow ratios for each phase"""
        critical_ratios = {}
        
        for phase_id, lanes in self.phase_definitions.items():
            max_ratio = 0.0
            
            for lane in lanes:
                if lane in approach_flows and lane in saturation_rates:
                    flow_ratio = approach_flows[lane] / saturation_rates[lane]
                    max_ratio = max(max_ratio, flow_ratio)
            
            critical_ratios[phase_id] = max_ratio
        
        return critical_ratios
    
    def _optimize_cycle_time(self, critical_ratios: Dict[int, float], 
                           approach_flows: Dict[str, float],
                           prediction: Optional[PredictionResult] = None) -> CycleOptimization:
        """Optimize cycle time using Webster's formula"""
        
        # Calculate total critical flow ratio
        total_critical_ratio = sum(critical_ratios.values())
        
        if total_critical_ratio == 0:
            # Fallback to base cycle time
            cycle_time = self.base_cycle_time
        else:
            # Webster's formula: C = (1.5 * L + 5) / (1 - Y)
            # where L = total lost time, Y = critical flow ratio
            total_lost_time = len(critical_ratios) * self.lost_time
            cycle_time = (1.5 * total_lost_time + 5) / (1 - total_critical_ratio)
            
            # Apply constraints
            cycle_time = max(self.min_cycle_time, min(self.max_cycle_time, cycle_time))
        
        # ML enhancement: adjust cycle time based on predictions
        ml_adjustments = {}
        if prediction and prediction.confidence > 0.7:
            # Predict future traffic conditions
            future_flows = self._predict_future_flows(approach_flows, prediction)
            future_critical_ratio = self._calculate_future_critical_ratio(future_flows)
            
            # Adjust cycle time based on predicted changes
            flow_change = (future_critical_ratio - total_critical_ratio) / total_critical_ratio
            cycle_adjustment = cycle_time * flow_change * 0.1  # 10% of predicted change
            cycle_time += cycle_adjustment
            ml_adjustments['cycle_time_adjustment'] = cycle_adjustment
            
            # Ensure constraints are still met
            cycle_time = max(self.min_cycle_time, min(self.max_cycle_time, cycle_time))
        
        # Calculate efficiency
        efficiency = total_critical_ratio / (cycle_time / (cycle_time - total_lost_time))
        
        # Create phase data
        phases = []
        for phase_id, critical_ratio in critical_ratios.items():
            lanes = self.phase_definitions[phase_id]
            phase_flow = sum(approach_flows.get(lane, 0) for lane in lanes)
            phase_saturation = sum(saturation_rates.get(lane, self.saturation_flow_rate) for lane in lanes)
            
            phase_data = PhaseData(
                phase_id=phase_id,
                approach_lanes=lanes,
                flow_rate=phase_flow,
                saturation_flow_rate=phase_saturation,
                critical_flow_ratio=critical_ratio,
                green_time=0.0,  # Will be calculated later
                lost_time=self.lost_time,
                priority_factor=1.0  # Will be enhanced with ML
            )
            phases.append(phase_data)
        
        return CycleOptimization(
            cycle_time=cycle_time,
            phases=phases,
            total_lost_time=total_lost_time,
            critical_flow_ratio=total_critical_ratio,
            efficiency=efficiency,
            ml_adjustments=ml_adjustments,
            confidence_score=prediction.confidence if prediction else 0.5
        )
    
    def _predict_future_flows(self, current_flows: Dict[str, float], 
                            prediction: PredictionResult) -> Dict[str, float]:
        """Predict future flows based on ML prediction"""
        future_flows = {}
        
        for lane in current_flows:
            if lane in prediction.predicted_flows:
                # Use ML prediction
                future_flows[lane] = prediction.predicted_flows[lane]
            else:
                # Use current flow as fallback
                future_flows[lane] = current_flows[lane]
        
        return future_flows
    
    def _calculate_future_critical_ratio(self, future_flows: Dict[str, float]) -> float:
        """Calculate critical flow ratio for future flows"""
        total_ratio = 0.0
        
        for phase_id, lanes in self.phase_definitions.items():
            max_ratio = 0.0
            
            for lane in lanes:
                if lane in future_flows:
                    # Use base saturation flow rate for prediction
                    flow_ratio = future_flows[lane] / self.saturation_flow_rate
                    max_ratio = max(max_ratio, flow_ratio)
            
            total_ratio += max_ratio
        
        return total_ratio
    
    def _calculate_phase_timings(self, cycle_optimization: CycleOptimization, 
                               approach_flows: Dict[str, float]) -> Dict[int, float]:
        """Calculate green times for each phase"""
        phase_timings = {}
        
        # Calculate total effective green time
        total_lost_time = cycle_optimization.total_lost_time
        effective_green_time = cycle_optimization.cycle_time - total_lost_time
        
        # Calculate green times proportional to critical flow ratios
        total_critical_ratio = sum(phase.critical_flow_ratio for phase in cycle_optimization.phases)
        
        if total_critical_ratio > 0:
            for phase in cycle_optimization.phases:
                # Proportional allocation
                green_time = (phase.critical_flow_ratio / total_critical_ratio) * effective_green_time
                
                # Apply constraints
                green_time = max(self.flow_analyzer.minimum_green_time, 
                               min(self.flow_analyzer.maximum_green_time, green_time))
                
                phase_timings[phase.phase_id] = green_time
        else:
            # Equal allocation if no critical ratios
            green_time = effective_green_time / len(cycle_optimization.phases)
            for phase in cycle_optimization.phases:
                phase_timings[phase.phase_id] = green_time
        
        return phase_timings
    
    def _convert_to_signal_timings(self, phase_timings: Dict[int, float], 
                                 current_timings: Dict[str, int]) -> Dict[str, int]:
        """Convert phase timings to signal timings for each lane"""
        optimized_timings = current_timings.copy()
        
        # Map phases to lanes
        for phase_id, green_time in phase_timings.items():
            lanes = self.phase_definitions[phase_id]
            for lane in lanes:
                if lane in optimized_timings:
                    optimized_timings[lane] = int(green_time)
        
        # Ensure all lanes have timing
        for lane in optimized_timings:
            if lane not in [l for lanes in self.phase_definitions.values() for l in lanes]:
                # Use average timing for unmapped lanes
                avg_timing = int(np.mean(list(phase_timings.values())))
                optimized_timings[lane] = max(15, min(90, avg_timing))
        
        return optimized_timings
    
    def _record_optimization(self, traffic_data: TrafficData, 
                           cycle_optimization: CycleOptimization, 
                           optimization_time: float):
        """Record optimization for analysis"""
        record = {
            'timestamp': datetime.now(),
            'intersection_id': traffic_data.intersection_id,
            'cycle_time': cycle_optimization.cycle_time,
            'efficiency': cycle_optimization.efficiency,
            'critical_flow_ratio': cycle_optimization.critical_flow_ratio,
            'ml_adjustments': cycle_optimization.ml_adjustments,
            'confidence_score': cycle_optimization.confidence_score,
            'optimization_time': optimization_time
        }
        
        self.optimization_history.append(record)
        
        # Keep only last 1000 records
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-1000:]
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization performance statistics"""
        if not self.optimization_history:
            return {}
        
        cycle_times = [record['cycle_time'] for record in self.optimization_history]
        efficiencies = [record['efficiency'] for record in self.optimization_history]
        optimization_times = [record['optimization_time'] for record in self.optimization_history]
        confidence_scores = [record['confidence_score'] for record in self.optimization_history]
        
        return {
            'total_optimizations': len(self.optimization_history),
            'avg_cycle_time': np.mean(cycle_times),
            'avg_efficiency': np.mean(efficiencies),
            'avg_optimization_time': np.mean(optimization_times),
            'avg_confidence_score': np.mean(confidence_scores),
            'ml_enhancement_usage': sum(1 for record in self.optimization_history 
                                      if record['ml_adjustments']) / len(self.optimization_history)
        }
    
    def get_phase_analysis(self, traffic_data: TrafficData) -> Dict[str, Any]:
        """Get detailed phase analysis for current traffic conditions"""
        approach_flows = self.flow_analyzer.analyze_approach_flows(traffic_data)
        saturation_rates = self.flow_analyzer.calculate_saturation_flow_rates(traffic_data)
        critical_ratios = self._calculate_critical_flow_ratios(approach_flows, saturation_rates)
        
        phase_analysis = {}
        for phase_id, lanes in self.phase_definitions.items():
            phase_flow = sum(approach_flows.get(lane, 0) for lane in lanes)
            phase_saturation = sum(saturation_rates.get(lane, self.saturation_flow_rate) for lane in lanes)
            
            phase_analysis[f'phase_{phase_id}'] = {
                'lanes': lanes,
                'flow_rate': phase_flow,
                'saturation_rate': phase_saturation,
                'critical_ratio': critical_ratios.get(phase_id, 0),
                'utilization': phase_flow / phase_saturation if phase_saturation > 0 else 0
            }
        
        return phase_analysis
    
    def update_parameters(self, new_config: WebstersFormulaConfig):
        """Update optimization parameters"""
        self.config = new_config
        self.base_cycle_time = new_config.base_cycle_time
        self.min_cycle_time = new_config.min_cycle_time
        self.max_cycle_time = new_config.max_cycle_time
        self.lost_time = new_config.lost_time
        self.saturation_flow_rate = new_config.saturation_flow_rate
        self.critical_flow_ratio = new_config.critical_flow_ratio
        self.ml_enhancement = new_config.ml_enhancement
        self.prediction_horizon = new_config.prediction_horizon
        
        self.logger.info("Webster's formula parameters updated")
    
    def clear_history(self):
        """Clear optimization history"""
        self.optimization_history.clear()
        self.performance_metrics.clear()
        self.logger.info("Webster's formula optimization history cleared")

