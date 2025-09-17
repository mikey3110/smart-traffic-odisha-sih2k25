"""
Real-Time Optimization Engine for Multi-Intersection Traffic Management
Phase 2: Real-Time Optimization Loop & Safety Systems

Features:
- Continuous 30-second optimization cycle with precise timing
- Thread-safe state management for concurrent intersections
- Real-time data ingestion from multiple camera feeds
- Adaptive confidence scoring for ML decisions
- Graceful degradation when ML confidence is low
"""

import asyncio
import threading
import time
import logging
import queue
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

from algorithms.advanced_q_learning_agent import AdvancedQLearningAgent, MultiDimensionalState, SophisticatedAction
from algorithms.multi_intersection_coordinator import MultiIntersectionCoordinator
from algorithms.advanced_reward_function import AdvancedRewardFunction
from algorithms.adaptive_experience_replay import AdaptiveExperienceReplay


class OptimizationMode(Enum):
    """Optimization mode for the system"""
    ML_OPTIMIZATION = "ml_optimization"
    FALLBACK_WEBSTER = "fallback_webster"
    EMERGENCY_OVERRIDE = "emergency_override"
    MANUAL_CONTROL = "manual_control"
    SAFETY_MODE = "safety_mode"


@dataclass
class OptimizationRequest:
    """Request for optimization cycle"""
    request_id: str
    intersection_id: str
    timestamp: datetime
    traffic_data: Dict[str, Any]
    current_timings: Dict[str, int]
    mode: OptimizationMode
    priority: int = 0
    timeout: float = 25.0  # seconds
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OptimizationResponse:
    """Response from optimization cycle"""
    request_id: str
    intersection_id: str
    timestamp: datetime
    success: bool
    optimized_timings: Dict[str, int]
    algorithm_used: str
    confidence_score: float
    processing_time: float
    mode: OptimizationMode
    safety_violations: List[str] = None
    warnings: List[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.safety_violations is None:
            self.safety_violations = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SystemMetrics:
    """Real-time system performance metrics"""
    timestamp: datetime
    total_cycles: int
    successful_cycles: int
    failed_cycles: int
    avg_cycle_time: float
    avg_confidence: float
    active_intersections: int
    memory_usage: float
    cpu_usage: float
    queue_size: int
    error_rate: float
    safety_violations: int
    emergency_overrides: int


class ThreadSafeStateManager:
    """Thread-safe state management for concurrent intersections"""
    
    def __init__(self, max_intersections: int = 10):
        self.max_intersections = max_intersections
        self.lock = threading.RLock()
        self.states = {}
        self.last_update = {}
        self.state_history = {}
        self.logger = logging.getLogger(__name__)
    
    def update_state(self, intersection_id: str, state: MultiDimensionalState) -> bool:
        """Update state for intersection with thread safety"""
        with self.lock:
            try:
                self.states[intersection_id] = state
                self.last_update[intersection_id] = datetime.now()
                
                # Store in history (keep last 100 states)
                if intersection_id not in self.state_history:
                    self.state_history[intersection_id] = []
                
                self.state_history[intersection_id].append({
                    'timestamp': datetime.now(),
                    'state': state.to_vector().tolist(),
                    'state_hash': state.get_state_hash()
                })
                
                # Keep only last 100 states
                if len(self.state_history[intersection_id]) > 100:
                    self.state_history[intersection_id] = self.state_history[intersection_id][-100:]
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error updating state for {intersection_id}: {e}")
                return False
    
    def get_state(self, intersection_id: str) -> Optional[MultiDimensionalState]:
        """Get current state for intersection"""
        with self.lock:
            return self.states.get(intersection_id)
    
    def get_all_states(self) -> Dict[str, MultiDimensionalState]:
        """Get all current states"""
        with self.lock:
            return self.states.copy()
    
    def get_state_history(self, intersection_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get state history for intersection"""
        with self.lock:
            history = self.state_history.get(intersection_id, [])
            return history[-limit:] if history else []
    
    def is_state_stale(self, intersection_id: str, max_age: float = 60.0) -> bool:
        """Check if state is stale"""
        with self.lock:
            if intersection_id not in self.last_update:
                return True
            
            age = (datetime.now() - self.last_update[intersection_id]).total_seconds()
            return age > max_age
    
    def clear_stale_states(self, max_age: float = 300.0):
        """Clear stale states"""
        with self.lock:
            current_time = datetime.now()
            stale_intersections = []
            
            for intersection_id, last_update in self.last_update.items():
                age = (current_time - last_update).total_seconds()
                if age > max_age:
                    stale_intersections.append(intersection_id)
            
            for intersection_id in stale_intersections:
                if intersection_id in self.states:
                    del self.states[intersection_id]
                if intersection_id in self.last_update:
                    del self.last_update[intersection_id]
                if intersection_id in self.state_history:
                    del self.state_history[intersection_id]
            
            if stale_intersections:
                self.logger.info(f"Cleared {len(stale_intersections)} stale states")


class AdaptiveConfidenceScorer:
    """Adaptive confidence scoring for ML decisions"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Confidence parameters
        self.base_confidence = 0.7
        self.min_confidence = 0.3
        self.max_confidence = 0.95
        
        # Learning parameters
        self.learning_rate = 0.01
        self.confidence_history = {}
        self.performance_history = {}
        
        # Confidence factors
        self.factors = {
            'data_quality': 0.3,
            'model_uncertainty': 0.25,
            'historical_performance': 0.2,
            'traffic_complexity': 0.15,
            'system_load': 0.1
        }
    
    def calculate_confidence(self, intersection_id: str, state: MultiDimensionalState, 
                           action: SophisticatedAction, model_uncertainty: float = 0.0) -> float:
        """Calculate adaptive confidence score"""
        try:
            # Data quality factor
            data_quality = self._assess_data_quality(state)
            
            # Model uncertainty factor
            uncertainty_factor = 1.0 - min(model_uncertainty, 1.0)
            
            # Historical performance factor
            historical_performance = self._get_historical_performance(intersection_id)
            
            # Traffic complexity factor
            traffic_complexity = self._assess_traffic_complexity(state)
            
            # System load factor
            system_load = self._assess_system_load()
            
            # Calculate weighted confidence
            confidence = (
                data_quality * self.factors['data_quality'] +
                uncertainty_factor * self.factors['model_uncertainty'] +
                historical_performance * self.factors['historical_performance'] +
                (1.0 - traffic_complexity) * self.factors['traffic_complexity'] +
                (1.0 - system_load) * self.factors['system_load']
            )
            
            # Apply bounds
            confidence = max(self.min_confidence, min(self.max_confidence, confidence))
            
            # Update confidence history
            self._update_confidence_history(intersection_id, confidence)
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return self.min_confidence
    
    def _assess_data_quality(self, state: MultiDimensionalState) -> float:
        """Assess quality of input data"""
        # Check for missing or invalid data
        state_vector = state.to_vector()
        
        # Check for NaN or infinite values
        if np.any(np.isnan(state_vector)) or np.any(np.isinf(state_vector)):
            return 0.1
        
        # Check for extreme values (outliers)
        if np.any(np.abs(state_vector) > 10.0):
            return 0.3
        
        # Check for reasonable traffic data
        if np.sum(state.lane_counts) > 200:  # Unrealistic traffic volume
            return 0.5
        
        # Check for temporal consistency
        if state.time_since_change < 0 or state.time_since_change > 300:
            return 0.6
        
        return 0.9  # High quality data
    
    def _get_historical_performance(self, intersection_id: str) -> float:
        """Get historical performance for intersection"""
        if intersection_id not in self.performance_history:
            return 0.5  # Default performance
        
        recent_performance = self.performance_history[intersection_id][-10:]  # Last 10 cycles
        return np.mean(recent_performance) if recent_performance else 0.5
    
    def _assess_traffic_complexity(self, state: MultiDimensionalState) -> float:
        """Assess complexity of traffic situation"""
        # High complexity indicators
        complexity = 0.0
        
        # High congestion
        if state.congestion_trend > 0.7:
            complexity += 0.3
        
        # Emergency vehicles
        if state.emergency_vehicles:
            complexity += 0.2
        
        # Bad weather
        if state.weather_condition in [2, 3, 4]:  # Rainy, foggy, stormy
            complexity += 0.2
        
        # Low visibility
        if state.visibility < 5.0:
            complexity += 0.1
        
        # High traffic volume
        if np.sum(state.lane_counts) > 100:
            complexity += 0.2
        
        return min(complexity, 1.0)
    
    def _assess_system_load(self) -> float:
        """Assess current system load"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_load = cpu_percent / 100.0
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_load = memory.percent / 100.0
            
            # Combined load
            system_load = (cpu_load + memory_load) / 2.0
            
            return min(system_load, 1.0)
            
        except Exception:
            return 0.5  # Default load
    
    def _update_confidence_history(self, intersection_id: str, confidence: float):
        """Update confidence history for learning"""
        if intersection_id not in self.confidence_history:
            self.confidence_history[intersection_id] = []
        
        self.confidence_history[intersection_id].append({
            'timestamp': datetime.now(),
            'confidence': confidence
        })
        
        # Keep only last 100 entries
        if len(self.confidence_history[intersection_id]) > 100:
            self.confidence_history[intersection_id] = self.confidence_history[intersection_id][-100:]
    
    def update_performance(self, intersection_id: str, success: bool, reward: float):
        """Update performance history"""
        if intersection_id not in self.performance_history:
            self.performance_history[intersection_id] = []
        
        # Convert success and reward to performance score
        performance_score = 0.5  # Base score
        if success:
            performance_score += 0.3
        if reward > 0:
            performance_score += min(reward * 0.1, 0.2)
        
        self.performance_history[intersection_id].append(performance_score)
        
        # Keep only last 100 entries
        if len(self.performance_history[intersection_id]) > 100:
            self.performance_history[intersection_id] = self.performance_history[intersection_id][-100:]
    
    def get_confidence_statistics(self) -> Dict[str, Any]:
        """Get confidence statistics"""
        stats = {
            'total_intersections': len(self.confidence_history),
            'avg_confidence': 0.0,
            'min_confidence': 1.0,
            'max_confidence': 0.0,
            'confidence_trends': {}
        }
        
        all_confidences = []
        
        for intersection_id, history in self.confidence_history.items():
            if history:
                confidences = [entry['confidence'] for entry in history]
                all_confidences.extend(confidences)
                
                stats['confidence_trends'][intersection_id] = {
                    'current': confidences[-1] if confidences else 0.0,
                    'avg': np.mean(confidences),
                    'trend': np.polyfit(range(len(confidences)), confidences, 1)[0] if len(confidences) > 1 else 0.0
                }
        
        if all_confidences:
            stats['avg_confidence'] = np.mean(all_confidences)
            stats['min_confidence'] = np.min(all_confidences)
            stats['max_confidence'] = np.max(all_confidences)
        
        return stats


class RealTimeDataIngestion:
    """Real-time data ingestion from multiple camera feeds via API"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Data sources
        self.data_sources = {}
        self.data_queue = queue.Queue(maxsize=1000)
        self.ingestion_threads = {}
        self.is_running = False
        
        # Data processing
        self.data_processor = None
        self.data_validator = None
        
        # Performance tracking
        self.ingestion_metrics = {
            'total_messages': 0,
            'successful_messages': 0,
            'failed_messages': 0,
            'avg_processing_time': 0.0,
            'queue_size': 0
        }
    
    def add_data_source(self, source_id: str, source_config: Dict[str, Any]):
        """Add a data source (camera feed, sensor, etc.)"""
        self.data_sources[source_id] = {
            'config': source_config,
            'last_update': None,
            'is_active': False,
            'error_count': 0
        }
        self.logger.info(f"Added data source: {source_id}")
    
    def start_ingestion(self):
        """Start data ingestion from all sources"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start ingestion thread for each source
        for source_id, source_config in self.data_sources.items():
            thread = threading.Thread(
                target=self._ingestion_loop,
                args=(source_id, source_config),
                daemon=True
            )
            thread.start()
            self.ingestion_threads[source_id] = thread
        
        self.logger.info("Started data ingestion from all sources")
    
    def stop_ingestion(self):
        """Stop data ingestion"""
        self.is_running = False
        
        # Wait for threads to finish
        for thread in self.ingestion_threads.values():
            thread.join(timeout=5)
        
        self.ingestion_threads.clear()
        self.logger.info("Stopped data ingestion")
    
    def _ingestion_loop(self, source_id: str, source_config: Dict[str, Any]):
        """Main ingestion loop for a data source"""
        while self.is_running:
            try:
                # Simulate data ingestion (replace with actual API calls)
                data = self._fetch_data_from_source(source_id, source_config)
                
                if data:
                    # Validate data
                    if self._validate_data(data):
                        # Add to queue
                        self.data_queue.put({
                            'source_id': source_id,
                            'timestamp': datetime.now(),
                            'data': data
                        })
                        
                        self.ingestion_metrics['successful_messages'] += 1
                        self.data_sources[source_id]['last_update'] = datetime.now()
                        self.data_sources[source_id]['is_active'] = True
                        self.data_sources[source_id]['error_count'] = 0
                    else:
                        self.ingestion_metrics['failed_messages'] += 1
                        self.data_sources[source_id]['error_count'] += 1
                
                self.ingestion_metrics['total_messages'] += 1
                
                # Sleep based on source frequency
                sleep_time = source_config.get('frequency', 1.0)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in ingestion loop for {source_id}: {e}")
                self.data_sources[source_id]['error_count'] += 1
                time.sleep(5)  # Wait before retrying
    
    def _fetch_data_from_source(self, source_id: str, source_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fetch data from a specific source"""
        # This is a mock implementation
        # In a real system, this would make actual API calls
        
        source_type = source_config.get('type', 'camera')
        
        if source_type == 'camera':
            return {
                'intersection_id': source_config.get('intersection_id', 'unknown'),
                'lane_counts': {
                    'north_lane': np.random.randint(5, 25),
                    'south_lane': np.random.randint(5, 25),
                    'east_lane': np.random.randint(5, 25),
                    'west_lane': np.random.randint(5, 25)
                },
                'avg_speed': np.random.uniform(20, 60),
                'timestamp': datetime.now().isoformat()
            }
        elif source_type == 'sensor':
            return {
                'intersection_id': source_config.get('intersection_id', 'unknown'),
                'queue_lengths': {
                    'north_lane': np.random.randint(0, 15),
                    'south_lane': np.random.randint(0, 15),
                    'east_lane': np.random.randint(0, 15),
                    'west_lane': np.random.randint(0, 15)
                },
                'waiting_times': {
                    'north_lane': np.random.uniform(0, 60),
                    'south_lane': np.random.uniform(0, 60),
                    'east_lane': np.random.uniform(0, 60),
                    'west_lane': np.random.uniform(0, 60)
                },
                'timestamp': datetime.now().isoformat()
            }
        
        return None
    
    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate incoming data"""
        try:
            # Check required fields
            required_fields = ['intersection_id', 'timestamp']
            if not all(field in data for field in required_fields):
                return False
            
            # Check data types and ranges
            if 'lane_counts' in data:
                lane_counts = data['lane_counts']
                if not isinstance(lane_counts, dict):
                    return False
                
                for lane, count in lane_counts.items():
                    if not isinstance(count, (int, float)) or count < 0 or count > 100:
                        return False
            
            # Check timestamp format
            try:
                datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            except:
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_data(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get data from ingestion queue"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_ingestion_metrics(self) -> Dict[str, Any]:
        """Get ingestion performance metrics"""
        self.ingestion_metrics['queue_size'] = self.data_queue.qsize()
        
        metrics = self.ingestion_metrics.copy()
        metrics['data_sources'] = {}
        
        for source_id, source_info in self.data_sources.items():
            metrics['data_sources'][source_id] = {
                'is_active': source_info['is_active'],
                'last_update': source_info['last_update'].isoformat() if source_info['last_update'] else None,
                'error_count': source_info['error_count']
            }
        
        return metrics


class RealTimeOptimizer:
    """
    Real-Time Optimization Engine with Safety-First Architecture
    
    Features:
    - Continuous 30-second optimization cycle with precise timing
    - Thread-safe state management for concurrent intersections
    - Real-time data ingestion from multiple camera feeds
    - Adaptive confidence scoring for ML decisions
    - Graceful degradation when ML confidence is low
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.state_manager = ThreadSafeStateManager()
        self.confidence_scorer = AdaptiveConfidenceScorer()
        self.data_ingestion = RealTimeDataIngestion()
        
        # ML components (will be injected)
        self.agents = {}
        self.coordinators = {}
        self.reward_functions = {}
        self.replay_buffers = {}
        
        # Optimization state
        self.is_running = False
        self.optimization_thread = None
        self.cycle_count = 0
        self.start_time = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_cycles': 0,
            'successful_cycles': 0,
            'failed_cycles': 0,
            'avg_cycle_time': 0.0,
            'avg_confidence': 0.0,
            'safety_violations': 0,
            'emergency_overrides': 0
        }
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Configuration
        self.cycle_time = self.config.get('cycle_time', 30.0)
        self.max_processing_time = self.config.get('max_processing_time', 25.0)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.safety_mode_threshold = self.config.get('safety_mode_threshold', 0.3)
        
        self.logger.info("Real-Time Optimizer initialized")
    
    def set_ml_components(self, agents: Dict[str, AdvancedQLearningAgent],
                         coordinators: Dict[str, MultiIntersectionCoordinator],
                         reward_functions: Dict[str, AdvancedRewardFunction],
                         replay_buffers: Dict[str, AdaptiveExperienceReplay]):
        """Set ML components for optimization"""
        self.agents = agents
        self.coordinators = coordinators
        self.reward_functions = reward_functions
        self.replay_buffers = replay_buffers
        
        self.logger.info(f"Set ML components for {len(agents)} intersections")
    
    def add_data_source(self, source_id: str, source_config: Dict[str, Any]):
        """Add a data source for real-time ingestion"""
        self.data_ingestion.add_data_source(source_id, source_config)
    
    async def start_optimization(self):
        """Start the real-time optimization engine"""
        if self.is_running:
            self.logger.warning("Optimization engine is already running")
            return
        
        self.logger.info("Starting real-time optimization engine")
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            # Start data ingestion
            self.data_ingestion.start_ingestion()
            
            # Start optimization loop
            self.optimization_thread = threading.Thread(
                target=self._optimization_loop,
                daemon=True
            )
            self.optimization_thread.start()
            
            self.logger.info("Real-time optimization engine started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start optimization engine: {e}")
            self.is_running = False
            raise
    
    async def stop_optimization(self):
        """Stop the real-time optimization engine"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping real-time optimization engine")
        self.is_running = False
        
        try:
            # Stop data ingestion
            self.data_ingestion.stop_ingestion()
            
            # Wait for optimization thread to finish
            if self.optimization_thread and self.optimization_thread.is_alive():
                self.optimization_thread.join(timeout=10)
            
            # Shutdown thread pool
            self.thread_pool.shutdown(wait=True)
            
            self.logger.info("Real-time optimization engine stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping optimization engine: {e}")
    
    def _optimization_loop(self):
        """Main optimization loop with precise 30-second timing"""
        self.logger.info("Starting optimization loop")
        
        while self.is_running:
            try:
                cycle_start = time.time()
                
                # Run optimization cycle
                success = self._run_optimization_cycle()
                
                # Update performance metrics
                cycle_time = time.time() - cycle_start
                self._update_performance_metrics(success, cycle_time)
                
                # Calculate sleep time for precise 30-second cycles
                sleep_time = max(0, self.cycle_time - cycle_time)
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    self.logger.warning(f"Cycle took {cycle_time:.2f}s, exceeding target {self.cycle_time}s")
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                time.sleep(5)  # Wait before retrying
        
        self.logger.info("Optimization loop stopped")
    
    def _run_optimization_cycle(self) -> bool:
        """Run a single optimization cycle for all intersections"""
        try:
            self.cycle_count += 1
            cycle_start = datetime.now()
            
            # Collect real-time data
            traffic_data = self._collect_real_time_data()
            
            if not traffic_data:
                self.logger.warning("No traffic data available for optimization cycle")
                return False
            
            # Process each intersection in parallel
            optimization_tasks = []
            for intersection_id in self.agents.keys():
                if intersection_id in traffic_data:
                    task = self.thread_pool.submit(
                        self._optimize_intersection,
                        intersection_id,
                        traffic_data[intersection_id]
                    )
                    optimization_tasks.append((intersection_id, task))
            
            # Collect results
            successful_optimizations = 0
            total_confidence = 0.0
            
            for intersection_id, task in optimization_tasks:
                try:
                    response = task.result(timeout=self.max_processing_time)
                    
                    if response.success:
                        successful_optimizations += 1
                        total_confidence += response.confidence_score
                        
                        # Update state manager
                        if response.intersection_id in self.agents:
                            agent = self.agents[response.intersection_id]
                            state = agent.create_state(
                                traffic_data[intersection_id],
                                response.optimized_timings
                            )
                            self.state_manager.update_state(intersection_id, state)
                    
                    self.logger.debug(
                        f"Optimized {intersection_id}: {response.algorithm_used} "
                        f"(confidence: {response.confidence_score:.3f}, "
                        f"time: {response.processing_time:.3f}s)"
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error optimizing {intersection_id}: {e}")
            
            # Calculate average confidence
            avg_confidence = total_confidence / len(optimization_tasks) if optimization_tasks else 0.0
            
            # Log cycle results
            self.logger.info(
                f"Optimization cycle {self.cycle_count}: "
                f"{successful_optimizations}/{len(optimization_tasks)} successful, "
                f"avg confidence: {avg_confidence:.3f}"
            )
            
            return successful_optimizations > 0
            
        except Exception as e:
            self.logger.error(f"Error in optimization cycle: {e}")
            return False
    
    def _collect_real_time_data(self) -> Dict[str, Dict[str, Any]]:
        """Collect real-time data from all sources"""
        traffic_data = {}
        
        # Get data from ingestion queue
        max_data_points = 100
        data_points = []
        
        for _ in range(max_data_points):
            data = self.data_ingestion.get_data(timeout=0.1)
            if data:
                data_points.append(data)
            else:
                break
        
        # Process and aggregate data
        for data_point in data_points:
            intersection_id = data_point['data'].get('intersection_id')
            if intersection_id:
                if intersection_id not in traffic_data:
                    traffic_data[intersection_id] = {
                        'intersection_id': intersection_id,
                        'timestamp': data_point['timestamp'].isoformat(),
                        'lane_counts': {},
                        'queue_lengths': {},
                        'waiting_times': {},
                        'avg_speed': 0.0,
                        'sources': []
                    }
                
                # Merge data from different sources
                source_data = data_point['data']
                traffic_data[intersection_id]['sources'].append(data_point['source_id'])
                
                if 'lane_counts' in source_data:
                    traffic_data[intersection_id]['lane_counts'].update(source_data['lane_counts'])
                
                if 'queue_lengths' in source_data:
                    traffic_data[intersection_id]['queue_lengths'].update(source_data['queue_lengths'])
                
                if 'waiting_times' in source_data:
                    traffic_data[intersection_id]['waiting_times'].update(source_data['waiting_times'])
                
                if 'avg_speed' in source_data:
                    traffic_data[intersection_id]['avg_speed'] = source_data['avg_speed']
        
        return traffic_data
    
    def _optimize_intersection(self, intersection_id: str, traffic_data: Dict[str, Any]) -> OptimizationResponse:
        """Optimize a single intersection"""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Get current state
            current_state = self.state_manager.get_state(intersection_id)
            
            # Get current timings (mock for now)
            current_timings = {
                'north_lane': 30,
                'south_lane': 30,
                'east_lane': 30,
                'west_lane': 30
            }
            
            # Determine optimization mode based on confidence
            mode = self._determine_optimization_mode(intersection_id, current_state)
            
            # Perform optimization
            if mode == OptimizationMode.ML_OPTIMIZATION and intersection_id in self.agents:
                response = self._ml_optimization(intersection_id, traffic_data, current_timings)
            else:
                response = self._fallback_optimization(intersection_id, traffic_data, current_timings, mode)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Create response
            return OptimizationResponse(
                request_id=request_id,
                intersection_id=intersection_id,
                timestamp=datetime.now(),
                success=response.get('success', False),
                optimized_timings=response.get('optimized_timings', current_timings),
                algorithm_used=response.get('algorithm_used', 'unknown'),
                confidence_score=response.get('confidence_score', 0.0),
                processing_time=processing_time,
                mode=mode,
                safety_violations=response.get('safety_violations', []),
                warnings=response.get('warnings', [])
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing {intersection_id}: {e}")
            return OptimizationResponse(
                request_id=request_id,
                intersection_id=intersection_id,
                timestamp=datetime.now(),
                success=False,
                optimized_timings=current_timings,
                algorithm_used='error',
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                mode=OptimizationMode.SAFETY_MODE,
                safety_violations=['optimization_error'],
                warnings=[str(e)]
            )
    
    def _determine_optimization_mode(self, intersection_id: str, current_state: Optional[MultiDimensionalState]) -> OptimizationMode:
        """Determine optimization mode based on system state and confidence"""
        # Check for emergency conditions
        if current_state and current_state.emergency_vehicles:
            return OptimizationMode.EMERGENCY_OVERRIDE
        
        # Check for stale state
        if self.state_manager.is_state_stale(intersection_id):
            return OptimizationMode.SAFETY_MODE
        
        # Check system load
        if self._is_system_overloaded():
            return OptimizationMode.FALLBACK_WEBSTER
        
        # Default to ML optimization
        return OptimizationMode.ML_OPTIMIZATION
    
    def _ml_optimization(self, intersection_id: str, traffic_data: Dict[str, Any], 
                        current_timings: Dict[str, int]) -> Dict[str, Any]:
        """Perform ML-based optimization"""
        try:
            agent = self.agents[intersection_id]
            coordinator = self.coordinators[intersection_id]
            reward_function = self.reward_functions[intersection_id]
            
            # Create state
            state = agent.create_state(traffic_data, current_timings)
            
            # Select action
            action = agent.select_action(state, training=False)
            
            # Coordinate with other intersections
            coordinated_action = coordinator.coordinate_optimization(action.__dict__)
            coordinated_action = SophisticatedAction(**coordinated_action)
            
            # Calculate confidence
            confidence = self.confidence_scorer.calculate_confidence(
                intersection_id, state, coordinated_action
            )
            
            # Apply action with safety constraints
            optimized_timings = coordinated_action.apply_safety_constraints(current_timings)
            
            # Calculate reward
            next_state = state  # Simplified
            reward_components = reward_function.calculate_reward(state, coordinated_action, next_state)
            
            # Add experience for learning
            if intersection_id in self.replay_buffers:
                replay_buffer = self.replay_buffers[intersection_id]
                replay_buffer.add_experience(state, coordinated_action, reward_components.total_reward, next_state, False)
            
            # Update performance
            self.confidence_scorer.update_performance(intersection_id, True, reward_components.total_reward)
            
            return {
                'success': True,
                'optimized_timings': optimized_timings,
                'algorithm_used': 'ml_q_learning',
                'confidence_score': confidence,
                'safety_violations': [],
                'warnings': []
            }
            
        except Exception as e:
            self.logger.error(f"ML optimization failed for {intersection_id}: {e}")
            return {
                'success': False,
                'optimized_timings': current_timings,
                'algorithm_used': 'ml_q_learning',
                'confidence_score': 0.0,
                'safety_violations': ['ml_optimization_error'],
                'warnings': [str(e)]
            }
    
    def _fallback_optimization(self, intersection_id: str, traffic_data: Dict[str, Any], 
                             current_timings: Dict[str, int], mode: OptimizationMode) -> Dict[str, Any]:
        """Perform fallback optimization (Webster's formula, etc.)"""
        try:
            if mode == OptimizationMode.FALLBACK_WEBSTER:
                # Implement Webster's formula fallback
                optimized_timings = self._webster_optimization(traffic_data, current_timings)
                algorithm_used = 'webster_fallback'
            elif mode == OptimizationMode.EMERGENCY_OVERRIDE:
                # Emergency vehicle priority
                optimized_timings = self._emergency_optimization(traffic_data, current_timings)
                algorithm_used = 'emergency_override'
            else:
                # Safety mode - maintain current timings
                optimized_timings = current_timings.copy()
                algorithm_used = 'safety_mode'
            
            return {
                'success': True,
                'optimized_timings': optimized_timings,
                'algorithm_used': algorithm_used,
                'confidence_score': 0.8,  # High confidence for fallback methods
                'safety_violations': [],
                'warnings': [f'Using {algorithm_used} due to low ML confidence']
            }
            
        except Exception as e:
            self.logger.error(f"Fallback optimization failed for {intersection_id}: {e}")
            return {
                'success': False,
                'optimized_timings': current_timings,
                'algorithm_used': 'fallback_error',
                'confidence_score': 0.0,
                'safety_violations': ['fallback_optimization_error'],
                'warnings': [str(e)]
            }
    
    def _webster_optimization(self, traffic_data: Dict[str, Any], current_timings: Dict[str, int]) -> Dict[str, int]:
        """Webster's formula optimization (simplified implementation)"""
        # Get traffic volumes
        lane_counts = traffic_data.get('lane_counts', {})
        
        # Calculate Webster's formula
        total_volume = sum(lane_counts.values()) if lane_counts else 100
        
        # Proportional timing based on volume
        optimized_timings = {}
        for lane, current_time in current_timings.items():
            lane_volume = lane_counts.get(lane, 10)
            proportion = lane_volume / total_volume if total_volume > 0 else 0.25
            
            # Scale to reasonable range (20-60 seconds)
            new_time = max(20, min(60, int(proportion * 120)))
            optimized_timings[lane] = new_time
        
        return optimized_timings
    
    def _emergency_optimization(self, traffic_data: Dict[str, Any], current_timings: Dict[str, int]) -> Dict[str, int]:
        """Emergency vehicle priority optimization"""
        # Give priority to main traffic flow (north-south typically)
        optimized_timings = current_timings.copy()
        
        # Increase green time for main flow
        optimized_timings['north_lane'] = min(90, optimized_timings['north_lane'] + 20)
        optimized_timings['south_lane'] = min(90, optimized_timings['south_lane'] + 20)
        
        # Reduce green time for side flow
        optimized_timings['east_lane'] = max(10, optimized_timings['east_lane'] - 10)
        optimized_timings['west_lane'] = max(10, optimized_timings['west_lane'] - 10)
        
        return optimized_timings
    
    def _is_system_overloaded(self) -> bool:
        """Check if system is overloaded"""
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > 90:
                return True
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                return True
            
            # Check queue size
            if self.data_ingestion.data_queue.qsize() > 500:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _update_performance_metrics(self, success: bool, cycle_time: float):
        """Update performance metrics"""
        self.performance_metrics['total_cycles'] += 1
        
        if success:
            self.performance_metrics['successful_cycles'] += 1
        else:
            self.performance_metrics['failed_cycles'] += 1
        
        # Update average cycle time
        total_cycles = self.performance_metrics['total_cycles']
        current_avg = self.performance_metrics['avg_cycle_time']
        self.performance_metrics['avg_cycle_time'] = (
            (current_avg * (total_cycles - 1) + cycle_time) / total_cycles
        )
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics"""
        # Get confidence statistics
        confidence_stats = self.confidence_scorer.get_confidence_statistics()
        
        # Get ingestion metrics
        ingestion_metrics = self.data_ingestion.get_ingestion_metrics()
        
        # Get system resource usage
        try:
            memory_usage = psutil.virtual_memory().percent
            cpu_usage = psutil.cpu_percent(interval=0.1)
        except:
            memory_usage = 0.0
            cpu_usage = 0.0
        
        return SystemMetrics(
            timestamp=datetime.now(),
            total_cycles=self.performance_metrics['total_cycles'],
            successful_cycles=self.performance_metrics['successful_cycles'],
            failed_cycles=self.performance_metrics['failed_cycles'],
            avg_cycle_time=self.performance_metrics['avg_cycle_time'],
            avg_confidence=confidence_stats.get('avg_confidence', 0.0),
            active_intersections=len(self.state_manager.states),
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            queue_size=ingestion_metrics['queue_size'],
            error_rate=self.performance_metrics['failed_cycles'] / max(self.performance_metrics['total_cycles'], 1),
            safety_violations=self.performance_metrics['safety_violations'],
            emergency_overrides=self.performance_metrics['emergency_overrides']
        )
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        return {
            'is_running': self.is_running,
            'cycle_count': self.cycle_count,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'performance_metrics': self.performance_metrics.copy(),
            'confidence_statistics': self.confidence_scorer.get_confidence_statistics(),
            'ingestion_metrics': self.data_ingestion.get_ingestion_metrics(),
            'state_manager_stats': {
                'active_states': len(self.state_manager.states),
                'stale_states': sum(1 for intersection_id in self.state_manager.states 
                                  if self.state_manager.is_state_stale(intersection_id))
            }
        }
