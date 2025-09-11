"""
Enhanced ML-based Signal Optimizer
Integrates all ML components for comprehensive traffic signal optimization
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd

# Import ML components
from data.enhanced_data_integration import EnhancedDataIntegration, TrafficDataPoint
from prediction.enhanced_traffic_predictor import EnhancedTrafficPredictor
from metrics.enhanced_performance_metrics import EnhancedPerformanceMetrics, TrafficMetrics
from ab_testing.ab_testing_framework import ABTestingFramework, ABTestConfig, TestVariant
from algorithms.enhanced_q_learning_optimizer import EnhancedQLearningOptimizer
from algorithms.enhanced_dynamic_programming_optimizer import EnhancedDynamicProgrammingOptimizer
from algorithms.enhanced_websters_formula_optimizer import EnhancedWebstersFormulaOptimizer
from config.ml_config import MLConfig, get_config


class OptimizationMode(Enum):
    """Optimization modes"""
    SINGLE_ALGORITHM = "single_algorithm"
    ENSEMBLE = "ensemble"
    ADAPTIVE = "adaptive"
    AB_TESTING = "ab_testing"


@dataclass
class OptimizationRequest:
    """Request for signal optimization"""
    intersection_id: str
    current_timings: Dict[str, int]
    traffic_data: Optional[TrafficDataPoint] = None
    algorithm_preference: Optional[str] = None
    optimization_mode: OptimizationMode = OptimizationMode.ADAPTIVE
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class OptimizationResponse:
    """Response from signal optimization"""
    intersection_id: str
    optimized_timings: Dict[str, int]
    algorithm_used: str
    confidence: float
    improvement_prediction: Dict[str, float]
    processing_time: float
    timestamp: datetime
    metadata: Dict[str, Any]


class EnhancedSignalOptimizer:
    """
    Enhanced ML-based Signal Optimizer
    
    Features:
    - Real-time data integration with fallback strategies
    - Multiple optimization algorithms (Q-Learning, DP, Webster's)
    - Traffic flow prediction using ML models
    - Performance metrics calculation and monitoring
    - A/B testing framework for algorithm comparison
    - Adaptive algorithm selection based on performance
    - Comprehensive logging and monitoring
    - Error handling and recovery
    """
    
    def __init__(self, config: Optional[MLConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_integration = EnhancedDataIntegration(self.config.data_integration)
        self.traffic_predictor = EnhancedTrafficPredictor(self.config.traffic_prediction)
        self.performance_metrics = EnhancedPerformanceMetrics(self.config.performance_metrics)
        self.ab_testing = ABTestingFramework(self.config.ab_testing)
        
        # Initialize optimization algorithms
        self.algorithms = {
            'q_learning': EnhancedQLearningOptimizer(self.config.q_learning),
            'dynamic_programming': EnhancedDynamicProgrammingOptimizer(self.config.dynamic_programming),
            'websters_formula': EnhancedWebstersFormulaOptimizer(self.config.websters_formula)
        }
        
        # Performance tracking
        self.optimization_history: deque = deque(maxlen=1000)
        self.algorithm_performance: Dict[str, List[float]] = defaultdict(list)
        self.intersection_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Optimization queue and processing
        self.optimization_queue = asyncio.Queue()
        self.processing_thread = None
        self.is_running = False
        
        # A/B testing
        self.active_tests: Dict[str, str] = {}  # intersection_id -> test_id
        
        self.logger.info("Enhanced signal optimizer initialized")
    
    async def start(self):
        """Start the optimization system"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start data integration
        await self.data_integration.__aenter__()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start A/B testing monitoring
        self.ab_testing.start_monitoring()
        
        self.logger.info("Enhanced signal optimizer started")
    
    async def stop(self):
        """Stop the optimization system"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Stop processing thread
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        # Stop A/B testing monitoring
        self.ab_testing.stop_monitoring()
        
        # Close data integration
        await self.data_integration.__aexit__(None, None, None)
        
        self.logger.info("Enhanced signal optimizer stopped")
    
    def _processing_loop(self):
        """Main processing loop for optimization requests"""
        while self.is_running:
            try:
                # Process optimization requests
                asyncio.run(self._process_optimization_requests())
                time.sleep(self.config.optimization_interval)
            except Exception as e:
                self.logger.error(f"Error in processing loop: {e}")
                time.sleep(5)
    
    async def _process_optimization_requests(self):
        """Process pending optimization requests"""
        # This would typically process requests from a queue
        # For now, we'll process based on a schedule
        pass
    
    async def optimize_intersection(self, request: OptimizationRequest) -> OptimizationResponse:
        """
        Optimize traffic signals for an intersection
        
        Args:
            request: Optimization request
            
        Returns:
            Optimization response with optimized timings
        """
        start_time = time.time()
        self.logger.info(f"Optimizing intersection {request.intersection_id}")
        
        try:
            # Get traffic data
            if request.traffic_data is None:
                traffic_data = await self.data_integration.fetch_traffic_data_async(
                    request.intersection_id
                )
            else:
                traffic_data = request.traffic_data
            
            # Calculate current performance metrics
            current_metrics = self.performance_metrics.calculate_metrics(
                {
                    'lane_counts': traffic_data.lane_counts,
                    'avg_speed': traffic_data.avg_speed,
                    'weather_condition': traffic_data.weather_condition,
                    'temperature': traffic_data.temperature,
                    'humidity': traffic_data.humidity,
                    'visibility': traffic_data.visibility
                },
                request.current_timings,
                request.intersection_id
            )
            
            # Select optimization algorithm
            algorithm_name = self._select_algorithm(
                request, traffic_data, current_metrics
            )
            
            # Perform optimization
            optimized_timings = await self._perform_optimization(
                algorithm_name, traffic_data, request.current_timings
            )
            
            # Calculate predicted improvement
            improvement_prediction = self._predict_improvement(
                current_metrics, optimized_timings, algorithm_name
            )
            
            # Calculate confidence
            confidence = self._calculate_confidence(
                algorithm_name, traffic_data, improvement_prediction
            )
            
            # Create response
            response = OptimizationResponse(
                intersection_id=request.intersection_id,
                optimized_timings=optimized_timings,
                algorithm_used=algorithm_name,
                confidence=confidence,
                improvement_prediction=improvement_prediction,
                processing_time=time.time() - start_time,
                timestamp=datetime.now(),
                metadata={
                    'traffic_data_source': traffic_data.source.value,
                    'traffic_data_confidence': traffic_data.confidence,
                    'optimization_mode': request.optimization_mode.value
                }
            )
            
            # Record optimization
            self._record_optimization(request, response, current_metrics)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error optimizing intersection {request.intersection_id}: {e}")
            # Return fallback optimization
            return self._create_fallback_response(request, str(e))
    
    def _select_algorithm(self, request: OptimizationRequest, 
                         traffic_data: TrafficDataPoint,
                         current_metrics: TrafficMetrics) -> str:
        """Select the best algorithm for optimization"""
        
        # Check if A/B testing is active for this intersection
        if request.intersection_id in self.active_tests:
            test_id = self.active_tests[request.intersection_id]
            # A/B testing will handle algorithm selection
            return self._get_ab_test_algorithm(test_id, request.intersection_id)
        
        # Check for algorithm preference
        if request.algorithm_preference and request.algorithm_preference in self.algorithms:
            return request.algorithm_preference
        
        # Adaptive algorithm selection based on performance
        if request.optimization_mode == OptimizationMode.ADAPTIVE:
            return self._select_adaptive_algorithm(traffic_data, current_metrics)
        
        # Use primary algorithm from config
        return self.config.primary_algorithm.value
    
    def _select_adaptive_algorithm(self, traffic_data: TrafficDataPoint,
                                 current_metrics: TrafficMetrics) -> str:
        """Select algorithm based on current traffic conditions and historical performance"""
        
        # Analyze traffic conditions
        total_vehicles = sum(traffic_data.lane_counts.values())
        avg_speed = traffic_data.avg_speed
        weather_condition = traffic_data.weather_condition
        
        # Algorithm selection logic based on conditions
        if total_vehicles > 50:  # High traffic
            if weather_condition in ['rainy', 'foggy', 'stormy']:
                return 'websters_formula'  # More conservative in bad weather
            else:
                return 'q_learning'  # Adaptive for high traffic
        elif total_vehicles < 10:  # Low traffic
            return 'dynamic_programming'  # Efficient for low traffic
        else:  # Medium traffic
            # Use historical performance
            best_algorithm = self._get_best_performing_algorithm()
            return best_algorithm or 'websters_formula'
    
    def _get_best_performing_algorithm(self) -> Optional[str]:
        """Get the best performing algorithm based on historical data"""
        if not self.algorithm_performance:
            return None
        
        # Calculate average performance for each algorithm
        avg_performance = {}
        for algorithm, scores in self.algorithm_performance.items():
            if scores:
                avg_performance[algorithm] = np.mean(scores[-10:])  # Last 10 scores
        
        if not avg_performance:
            return None
        
        return max(avg_performance, key=avg_performance.get)
    
    def _get_ab_test_algorithm(self, test_id: str, intersection_id: str) -> str:
        """Get algorithm from A/B test"""
        # This would integrate with the A/B testing framework
        # For now, return a default
        return 'websters_formula'
    
    async def _perform_optimization(self, algorithm_name: str, 
                                  traffic_data: TrafficDataPoint,
                                  current_timings: Dict[str, int]) -> Dict[str, int]:
        """Perform optimization using the selected algorithm"""
        
        if algorithm_name not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        algorithm = self.algorithms[algorithm_name]
        
        # Convert TrafficDataPoint to algorithm format
        traffic_data_dict = {
            'lane_counts': traffic_data.lane_counts,
            'avg_speed': traffic_data.avg_speed,
            'weather_condition': traffic_data.weather_condition,
            'temperature': traffic_data.temperature,
            'humidity': traffic_data.humidity,
            'visibility': traffic_data.visibility,
            'timestamp': traffic_data.timestamp
        }
        
        # Perform optimization
        optimized_timings = algorithm.optimize_signal_timing(
            traffic_data_dict, current_timings
        )
        
        return optimized_timings
    
    def _predict_improvement(self, current_metrics: TrafficMetrics,
                           optimized_timings: Dict[str, int],
                           algorithm_name: str) -> Dict[str, float]:
        """Predict improvement from optimization"""
        
        # Use traffic predictor to estimate future conditions
        prediction = self.traffic_predictor.predict_traffic_flow(
            current_metrics.intersection_id, 5  # 5 minutes ahead
        )
        
        # Calculate predicted metrics with optimized timings
        predicted_metrics = self.performance_metrics.calculate_metrics(
            {
                'lane_counts': prediction['lane_counts'],
                'avg_speed': prediction['avg_speed'],
                'weather_condition': prediction['weather_condition'],
                'temperature': prediction.get('temperature'),
                'humidity': prediction.get('humidity'),
                'visibility': prediction.get('visibility')
            },
            optimized_timings,
            current_metrics.intersection_id
        )
        
        # Calculate improvement percentages
        improvements = {}
        metric_names = ['wait_time', 'throughput', 'efficiency', 'safety_score']
        
        for metric_name in metric_names:
            current_value = getattr(current_metrics, metric_name)
            predicted_value = getattr(predicted_metrics, metric_name)
            
            if current_value != 0:
                if metric_name == 'wait_time':
                    improvement = ((current_value - predicted_value) / current_value) * 100
                else:
                    improvement = ((predicted_value - current_value) / current_value) * 100
            else:
                improvement = 0.0
            
            improvements[metric_name] = improvement
        
        return improvements
    
    def _calculate_confidence(self, algorithm_name: str, 
                            traffic_data: TrafficDataPoint,
                            improvement_prediction: Dict[str, float]) -> float:
        """Calculate confidence in optimization results"""
        
        base_confidence = 0.5
        
        # Algorithm-specific confidence
        algorithm_confidence = {
            'q_learning': 0.8,
            'dynamic_programming': 0.9,
            'websters_formula': 0.7
        }.get(algorithm_name, 0.5)
        
        # Data quality confidence
        data_confidence = traffic_data.confidence
        
        # Improvement confidence (based on predicted improvements)
        avg_improvement = np.mean(list(improvement_prediction.values()))
        improvement_confidence = min(1.0, avg_improvement / 20.0)  # Normalize to 20% improvement
        
        # Weather confidence
        weather_confidence = {
            'clear': 1.0,
            'cloudy': 0.9,
            'rainy': 0.7,
            'foggy': 0.6,
            'stormy': 0.5,
            'snowy': 0.4
        }.get(traffic_data.weather_condition, 0.8)
        
        # Combine confidences
        confidence = (algorithm_confidence * 0.3 + 
                     data_confidence * 0.3 + 
                     improvement_confidence * 0.2 + 
                     weather_confidence * 0.2)
        
        return min(1.0, max(0.0, confidence))
    
    def _record_optimization(self, request: OptimizationRequest, 
                           response: OptimizationResponse,
                           current_metrics: TrafficMetrics):
        """Record optimization for performance tracking"""
        
        # Record in history
        self.optimization_history.append({
            'timestamp': response.timestamp,
            'intersection_id': request.intersection_id,
            'algorithm_used': response.algorithm_used,
            'confidence': response.confidence,
            'processing_time': response.processing_time,
            'improvement_prediction': response.improvement_prediction
        })
        
        # Update algorithm performance
        if response.improvement_prediction:
            avg_improvement = np.mean(list(response.improvement_prediction.values()))
            self.algorithm_performance[response.algorithm_used].append(avg_improvement)
            
            # Keep only recent performance data
            if len(self.algorithm_performance[response.algorithm_used]) > 100:
                self.algorithm_performance[response.algorithm_used] = \
                    self.algorithm_performance[response.algorithm_used][-100:]
        
        # Update intersection statistics
        self.intersection_stats[request.intersection_id] = {
            'last_optimization': response.timestamp,
            'algorithm_used': response.algorithm_used,
            'confidence': response.confidence,
            'total_optimizations': self.intersection_stats[request.intersection_id].get(
                'total_optimizations', 0) + 1
        }
    
    def _create_fallback_response(self, request: OptimizationRequest, 
                                error_message: str) -> OptimizationResponse:
        """Create fallback response when optimization fails"""
        
        # Use Webster's formula as fallback
        fallback_timings = {
            'north_lane': 30,
            'south_lane': 30,
            'east_lane': 30,
            'west_lane': 30
        }
        
        return OptimizationResponse(
            intersection_id=request.intersection_id,
            optimized_timings=fallback_timings,
            algorithm_used='websters_formula_fallback',
            confidence=0.3,
            improvement_prediction={},
            processing_time=0.0,
            timestamp=datetime.now(),
            metadata={'error': error_message, 'fallback': True}
        )
    
    def create_ab_test(self, test_config: ABTestConfig) -> str:
        """Create an A/B test for algorithm comparison"""
        test_id = self.ab_testing.create_test(test_config)
        
        # Map intersections to test
        for variant in test_config.variants:
            if variant.is_control:
                # This would be handled by the A/B testing framework
                pass
        
        return test_id
    
    def start_ab_test(self, test_id: str, intersection_ids: List[str]):
        """Start A/B test for specific intersections"""
        self.ab_testing.start_test(test_id)
        
        # Map intersections to test
        for intersection_id in intersection_ids:
            self.active_tests[intersection_id] = test_id
    
    def stop_ab_test(self, test_id: str):
        """Stop A/B test"""
        self.ab_testing.stop_test(test_id)
        
        # Remove intersection mappings
        intersections_to_remove = [
            iid for iid, tid in self.active_tests.items() if tid == test_id
        ]
        for intersection_id in intersections_to_remove:
            del self.active_tests[intersection_id]
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.optimization_history:
            return {}
        
        # Calculate statistics
        total_optimizations = len(self.optimization_history)
        algorithms_used = [opt['algorithm_used'] for opt in self.optimization_history]
        algorithm_counts = {alg: algorithms_used.count(alg) for alg in set(algorithms_used)}
        
        avg_confidence = np.mean([opt['confidence'] for opt in self.optimization_history])
        avg_processing_time = np.mean([opt['processing_time'] for opt in self.optimization_history])
        
        # Algorithm performance
        algorithm_performance = {}
        for alg, scores in self.algorithm_performance.items():
            if scores:
                algorithm_performance[alg] = {
                    'avg_score': np.mean(scores),
                    'recent_score': np.mean(scores[-10:]) if len(scores) >= 10 else np.mean(scores),
                    'total_optimizations': len(scores)
                }
        
        return {
            'total_optimizations': total_optimizations,
            'algorithm_usage': algorithm_counts,
            'avg_confidence': avg_confidence,
            'avg_processing_time': avg_processing_time,
            'algorithm_performance': algorithm_performance,
            'active_ab_tests': len(self.active_tests),
            'intersection_stats': dict(self.intersection_stats)
        }
    
    def get_intersection_performance(self, intersection_id: str) -> Dict[str, Any]:
        """Get performance metrics for a specific intersection"""
        if intersection_id not in self.intersection_stats:
            return {}
        
        # Get recent optimizations for this intersection
        recent_optimizations = [
            opt for opt in self.optimization_history
            if opt['intersection_id'] == intersection_id
        ][-10:]  # Last 10 optimizations
        
        if not recent_optimizations:
            return self.intersection_stats[intersection_id]
        
        # Calculate performance metrics
        avg_confidence = np.mean([opt['confidence'] for opt in recent_optimizations])
        avg_processing_time = np.mean([opt['processing_time'] for opt in recent_optimizations])
        
        # Get improvement predictions
        improvements = [opt['improvement_prediction'] for opt in recent_optimizations]
        if improvements:
            avg_improvements = {}
            for metric in ['wait_time', 'throughput', 'efficiency']:
                values = [imp.get(metric, 0) for imp in improvements if imp]
                if values:
                    avg_improvements[metric] = np.mean(values)
        else:
            avg_improvements = {}
        
        return {
            **self.intersection_stats[intersection_id],
            'avg_confidence': avg_confidence,
            'avg_processing_time': avg_processing_time,
            'avg_improvements': avg_improvements,
            'recent_optimizations': len(recent_optimizations)
        }
    
    def export_optimization_data(self, filepath: str):
        """Export optimization data for analysis"""
        export_data = {
            'optimization_history': list(self.optimization_history),
            'algorithm_performance': dict(self.algorithm_performance),
            'intersection_stats': dict(self.intersection_stats),
            'active_tests': self.active_tests,
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Optimization data exported to {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the enhanced signal optimizer
    optimizer = EnhancedSignalOptimizer()
    
    # Test optimization request
    request = OptimizationRequest(
        intersection_id="junction-1",
        current_timings={'north_lane': 30, 'south_lane': 30, 'east_lane': 30, 'west_lane': 30},
        optimization_mode=OptimizationMode.ADAPTIVE
    )
    
    # Run optimization (would need to be run in async context)
    print("Testing enhanced signal optimizer...")
    
    # Test statistics
    stats = optimizer.get_optimization_statistics()
    print(f"Optimization statistics: {stats}")
    
    # Test intersection performance
    performance = optimizer.get_intersection_performance("junction-1")
    print(f"Intersection performance: {performance}")
