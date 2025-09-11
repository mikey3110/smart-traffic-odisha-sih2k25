"""
Enhanced Dynamic Programming Traffic Signal Optimization
Advanced implementation with multi-objective optimization and real-time adaptation
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import time
from collections import defaultdict
import heapq

from config.ml_config import DynamicProgrammingConfig, get_config
from data.enhanced_data_integration import TrafficData
from prediction.traffic_predictor import PredictionResult


@dataclass
class TrafficState:
    """Traffic state representation for dynamic programming"""
    intersection_id: str
    timestamp: datetime
    lane_counts: Dict[str, int]
    queue_lengths: Dict[str, float]
    waiting_times: Dict[str, float]
    flow_rates: Dict[str, float]
    congestion_level: float
    phase: int
    phase_duration: float


@dataclass
class SignalAction:
    """Signal action representation"""
    phase_sequence: List[int]  # Sequence of phases
    phase_durations: List[float]  # Duration for each phase
    cycle_time: float
    total_cost: float
    priority_boost: Dict[str, bool]  # Priority boost for each lane


@dataclass
class OptimizationResult:
    """Dynamic programming optimization result"""
    optimal_actions: List[SignalAction]
    total_cost: float
    performance_metrics: Dict[str, float]
    computation_time: float
    convergence_iterations: int


class TrafficFlowModel:
    """Traffic flow model for dynamic programming"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Traffic flow parameters
        self.saturation_flow_rate = 1800  # vehicles per hour per lane
        self.jam_density = 200  # vehicles per km per lane
        self.free_flow_speed = 50  # km/h
        self.lane_capacity = 50  # vehicles per lane
        
        # Queue dynamics parameters
        self.queue_discharge_rate = 0.8  # fraction of saturation flow
        self.queue_buildup_rate = 1.2  # fraction of arrival rate
        
    def predict_queue_evolution(self, current_state: TrafficState, 
                              action: SignalAction, time_horizon: int) -> List[TrafficState]:
        """
        Predict queue evolution over time horizon
        
        Args:
            current_state: Current traffic state
            action: Signal action to apply
            time_horizon: Prediction horizon in time steps
            
        Returns:
            List of predicted states
        """
        predicted_states = []
        state = current_state
        
        for t in range(time_horizon):
            # Calculate current phase
            phase_index = t % len(action.phase_sequence)
            current_phase = action.phase_sequence[phase_index]
            
            # Calculate flow rates for each lane
            new_flow_rates = {}
            new_queue_lengths = {}
            new_waiting_times = {}
            
            for lane in state.lane_counts:
                # Determine if lane has green signal
                has_green = self._lane_has_green(lane, current_phase)
                
                if has_green:
                    # Discharge rate (vehicles per time step)
                    discharge_rate = min(
                        state.queue_lengths[lane],
                        self.saturation_flow_rate / 3600 * 5  # 5-second time step
                    )
                    new_flow_rates[lane] = discharge_rate
                    new_queue_lengths[lane] = max(0, state.queue_lengths[lane] - discharge_rate)
                else:
                    # No discharge, queue builds up
                    arrival_rate = state.flow_rates[lane] / 3600 * 5  # 5-second time step
                    new_flow_rates[lane] = 0
                    new_queue_lengths[lane] = state.queue_lengths[lane] + arrival_rate
                
                # Calculate waiting time (simplified)
                new_waiting_times[lane] = new_queue_lengths[lane] * 2.0  # 2 seconds per queued vehicle
            
            # Calculate new lane counts (vehicles in queue)
            new_lane_counts = {
                lane: int(round(new_queue_lengths[lane]))
                for lane in state.lane_counts
            }
            
            # Calculate congestion level
            total_vehicles = sum(new_lane_counts.values())
            congestion_level = min(1.0, total_vehicles / (self.lane_capacity * 4))  # 4 lanes
            
            # Create new state
            new_state = TrafficState(
                intersection_id=state.intersection_id,
                timestamp=state.timestamp + timedelta(seconds=t * 5),
                lane_counts=new_lane_counts,
                queue_lengths=new_queue_lengths,
                waiting_times=new_waiting_times,
                flow_rates=new_flow_rates,
                congestion_level=congestion_level,
                phase=current_phase,
                phase_duration=action.phase_durations[phase_index]
            )
            
            predicted_states.append(new_state)
            state = new_state
        
        return predicted_states
    
    def _lane_has_green(self, lane: str, phase: int) -> bool:
        """Determine if lane has green signal in given phase"""
        # Simplified phase mapping
        phase_mapping = {
            0: ['north_lane', 'south_lane'],  # North-South
            1: ['east_lane', 'west_lane'],    # East-West
            2: ['north_lane'],                # North only
            3: ['south_lane']                 # South only
        }
        
        return lane in phase_mapping.get(phase, [])


class CostFunction:
    """Multi-objective cost function for dynamic programming"""
    
    def __init__(self, config: DynamicProgrammingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cost weights
        self.wait_time_weight = config.cost_weights.get('wait_time', 1.0)
        self.queue_length_weight = config.cost_weights.get('queue_length', 0.5)
        self.fuel_consumption_weight = config.cost_weights.get('fuel_consumption', 0.3)
        self.emissions_weight = config.cost_weights.get('emissions', 0.2)
        
    def calculate_cost(self, state: TrafficState, action: SignalAction) -> float:
        """
        Calculate total cost for a state-action pair
        
        Args:
            state: Current traffic state
            action: Signal action
            
        Returns:
            Total cost (lower is better)
        """
        total_cost = 0.0
        
        # Wait time cost (primary objective)
        wait_time_cost = sum(state.waiting_times.values()) * self.wait_time_weight
        total_cost += wait_time_cost
        
        # Queue length cost
        queue_length_cost = sum(state.queue_lengths.values()) * self.queue_length_weight
        total_cost += queue_length_cost
        
        # Fuel consumption cost (estimated from idling vehicles)
        idling_vehicles = sum(state.queue_lengths.values())
        fuel_cost = idling_vehicles * 0.1 * self.fuel_consumption_weight  # 0.1 L per idling vehicle
        total_cost += fuel_cost
        
        # Emissions cost (estimated)
        emissions_cost = idling_vehicles * 0.05 * self.emissions_weight  # 0.05 kg CO2 per idling vehicle
        total_cost += emissions_cost
        
        # Signal switching cost (penalty for frequent changes)
        if hasattr(self, 'last_action') and self.last_action.phase_sequence[0] != action.phase_sequence[0]:
            switching_cost = 0.5
            total_cost += switching_cost
        
        # Congestion penalty
        if state.congestion_level > 0.8:
            congestion_penalty = state.congestion_level * 10.0
            total_cost += congestion_penalty
        
        self.last_action = action
        
        return total_cost


class EnhancedDynamicProgrammingOptimizer:
    """
    Enhanced Dynamic Programming optimizer for traffic signal control
    
    Features:
    - Multi-objective optimization
    - Real-time adaptation
    - Traffic flow prediction
    - Constraint handling
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[DynamicProgrammingConfig] = None):
        self.config = config or get_config().dynamic_programming
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.traffic_model = TrafficFlowModel()
        self.cost_function = CostFunction(self.config)
        
        # Optimization parameters
        self.time_horizon = self.config.time_horizon
        self.time_step = self.config.time_step
        self.max_cycle_time = self.config.max_cycle_time
        self.min_cycle_time = self.config.min_cycle_time
        self.min_green_time = self.config.min_green_time
        self.max_green_time = self.config.max_green_time
        self.yellow_time = self.config.yellow_time
        self.all_red_time = self.config.all_red_time
        
        # State space discretization
        self.queue_levels = 10  # Discretize queue lengths
        self.congestion_levels = 5  # Discretize congestion levels
        
        # Action space
        self.phase_sequences = self._generate_phase_sequences()
        self.duration_options = self._generate_duration_options()
        
        # Optimization state
        self.value_function = {}  # State -> optimal value
        self.policy = {}  # State -> optimal action
        self.optimization_history = []
        
        # Performance tracking
        self.optimization_times = []
        self.convergence_iterations = []
        self.cost_reductions = []
        
        self.logger.info("Enhanced Dynamic Programming optimizer initialized")
    
    def _generate_phase_sequences(self) -> List[List[int]]:
        """Generate possible phase sequences"""
        sequences = [
            [0, 1],  # North-South, East-West
            [0, 1, 2, 3],  # All phases
            [0, 2, 1, 3],  # North-South, North only, East-West, South only
            [1, 3, 0, 2],  # East-West, South only, North-South, North only
        ]
        return sequences
    
    def _generate_duration_options(self) -> List[float]:
        """Generate possible phase durations"""
        durations = []
        for duration in range(self.min_green_time, self.max_green_time + 1, 5):
            durations.append(float(duration))
        return durations
    
    def _discretize_state(self, state: TrafficState) -> Tuple:
        """Discretize continuous state for DP"""
        # Discretize queue lengths
        queue_levels = []
        for lane in ['north_lane', 'south_lane', 'east_lane', 'west_lane']:
            queue_length = state.queue_lengths.get(lane, 0)
            level = min(self.queue_levels - 1, int(queue_length / 10))  # 10 vehicles per level
            queue_levels.append(level)
        
        # Discretize congestion level
        congestion_level = min(self.congestion_levels - 1, int(state.congestion_level * self.congestion_levels))
        
        # Discretize time of day (4 periods)
        hour = state.timestamp.hour
        if 6 <= hour < 12:
            time_period = 0  # Morning
        elif 12 <= hour < 18:
            time_period = 1  # Afternoon
        elif 18 <= hour < 22:
            time_period = 2  # Evening
        else:
            time_period = 3  # Night
        
        return tuple(queue_levels + [congestion_level, time_period, state.phase])
    
    def _generate_actions(self, state: TrafficState) -> List[SignalAction]:
        """Generate feasible actions for given state"""
        actions = []
        
        for phase_sequence in self.phase_sequences:
            for duration in self.duration_options:
                # Calculate total cycle time
                cycle_time = len(phase_sequence) * duration + len(phase_sequence) * (self.yellow_time + self.all_red_time)
                
                # Check constraints
                if self.min_cycle_time <= cycle_time <= self.max_cycle_time:
                    # Create action
                    action = SignalAction(
                        phase_sequence=phase_sequence,
                        phase_durations=[duration] * len(phase_sequence),
                        cycle_time=cycle_time,
                        total_cost=0.0,
                        priority_boost={lane: False for lane in state.lane_counts}
                    )
                    actions.append(action)
        
        return actions
    
    def optimize_signal_timing(self, traffic_data: TrafficData, 
                             current_timings: Dict[str, int],
                             historical_data: List[TrafficData] = None) -> Dict[str, int]:
        """
        Optimize signal timing using dynamic programming
        
        Args:
            traffic_data: Current traffic data
            current_timings: Current signal timings
            historical_data: Historical data for context
            
        Returns:
            Optimized signal timings
        """
        start_time = time.time()
        
        # Convert to traffic state
        state = self._traffic_data_to_state(traffic_data, current_timings)
        
        # Run dynamic programming optimization
        optimal_action = self._solve_dp(state)
        
        # Convert action to signal timings
        optimized_timings = self._action_to_timings(optimal_action, current_timings)
        
        # Record performance
        optimization_time = time.time() - start_time
        self.optimization_times.append(optimization_time)
        
        # Log optimization result
        self.logger.info(
            f"DP optimization completed in {optimization_time:.3f}s. "
            f"Cycle time: {optimal_action.cycle_time:.1f}s, "
            f"Cost: {optimal_action.total_cost:.2f}"
        )
        
        return optimized_timings
    
    def _traffic_data_to_state(self, traffic_data: TrafficData, 
                              current_timings: Dict[str, int]) -> TrafficState:
        """Convert TrafficData to TrafficState"""
        
        # Calculate queue lengths (simplified)
        queue_lengths = {}
        for lane, count in traffic_data.lane_counts.items():
            queue_lengths[lane] = float(count)
        
        # Calculate waiting times (simplified)
        waiting_times = {}
        for lane, count in traffic_data.lane_counts.items():
            waiting_times[lane] = float(count) * 2.0  # 2 seconds per vehicle
        
        # Calculate flow rates (simplified)
        flow_rates = {}
        for lane, count in traffic_data.lane_counts.items():
            flow_rates[lane] = float(count) * 60.0  # vehicles per hour
        
        # Calculate congestion level
        total_vehicles = sum(traffic_data.lane_counts.values())
        congestion_level = min(1.0, total_vehicles / 100.0)  # Normalize by max expected
        
        return TrafficState(
            intersection_id=traffic_data.intersection_id,
            timestamp=traffic_data.timestamp,
            lane_counts=traffic_data.lane_counts,
            queue_lengths=queue_lengths,
            waiting_times=waiting_times,
            flow_rates=flow_rates,
            congestion_level=congestion_level,
            phase=0,  # Assume starting phase
            phase_duration=current_timings.get('north_lane', 30)
        )
    
    def _solve_dp(self, initial_state: TrafficState) -> SignalAction:
        """Solve dynamic programming problem"""
        
        # Discretize initial state
        state_key = self._discretize_state(initial_state)
        
        # Check if already solved
        if state_key in self.policy:
            return self.policy[state_key]
        
        # Generate all possible actions
        actions = self._generate_actions(initial_state)
        
        if not actions:
            # Fallback to simple action
            return SignalAction(
                phase_sequence=[0, 1],
                phase_durations=[30.0, 30.0],
                cycle_time=60.0,
                total_cost=0.0,
                priority_boost={lane: False for lane in initial_state.lane_counts}
            )
        
        # Evaluate each action
        best_action = None
        best_cost = float('inf')
        
        for action in actions:
            # Predict traffic evolution
            predicted_states = self.traffic_model.predict_queue_evolution(
                initial_state, action, self.time_horizon
            )
            
            # Calculate total cost
            total_cost = 0.0
            for i, state in enumerate(predicted_states):
                cost = self.cost_function.calculate_cost(state, action)
                # Apply discount factor
                discount = 0.95 ** i
                total_cost += cost * discount
            
            action.total_cost = total_cost
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_action = action
        
        # Cache result
        self.policy[state_key] = best_action
        self.value_function[state_key] = best_cost
        
        return best_action
    
    def _action_to_timings(self, action: SignalAction, 
                          current_timings: Dict[str, int]) -> Dict[str, int]:
        """Convert action to signal timings"""
        
        # Calculate total cycle time
        total_cycle = action.cycle_time
        
        # Distribute time among phases
        phase_times = {}
        for i, phase in enumerate(action.phase_sequence):
            duration = action.phase_durations[i]
            
            # Map phases to lanes
            if phase == 0:  # North-South
                phase_times['north_lane'] = duration
                phase_times['south_lane'] = duration
            elif phase == 1:  # East-West
                phase_times['east_lane'] = duration
                phase_times['west_lane'] = duration
            elif phase == 2:  # North only
                phase_times['north_lane'] = duration
            elif phase == 3:  # South only
                phase_times['south_lane'] = duration
        
        # Ensure all lanes have timing
        optimized_timings = current_timings.copy()
        for lane in optimized_timings:
            if lane in phase_times:
                optimized_timings[lane] = int(phase_times[lane])
            else:
                # Use average timing for lanes not in current phase
                avg_timing = int(total_cycle / len(action.phase_sequence))
                optimized_timings[lane] = max(15, min(90, avg_timing))
        
        return optimized_timings
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization performance statistics"""
        return {
            "total_optimizations": len(self.optimization_times),
            "avg_optimization_time": np.mean(self.optimization_times) if self.optimization_times else 0,
            "max_optimization_time": np.max(self.optimization_times) if self.optimization_times else 0,
            "min_optimization_time": np.min(self.optimization_times) if self.optimization_times else 0,
            "cached_states": len(self.policy),
            "avg_convergence_iterations": np.mean(self.convergence_iterations) if self.convergence_iterations else 0,
            "avg_cost_reduction": np.mean(self.cost_reductions) if self.cost_reductions else 0
        }
    
    def clear_cache(self):
        """Clear optimization cache"""
        self.value_function.clear()
        self.policy.clear()
        self.logger.info("DP optimization cache cleared")
    
    def update_parameters(self, new_config: DynamicProgrammingConfig):
        """Update optimization parameters"""
        self.config = new_config
        self.cost_function = CostFunction(new_config)
        self.clear_cache()
        self.logger.info("DP parameters updated")
    
    def save_optimization_state(self, filepath: str):
        """Save optimization state for persistence"""
        import pickle
        
        state_data = {
            'value_function': self.value_function,
            'policy': self.policy,
            'optimization_times': self.optimization_times,
            'convergence_iterations': self.convergence_iterations,
            'cost_reductions': self.cost_reductions,
            'config': self.config.__dict__
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state_data, f)
        
        self.logger.info(f"DP optimization state saved to {filepath}")
    
    def load_optimization_state(self, filepath: str):
        """Load optimization state from file"""
        import pickle
        
        try:
            with open(filepath, 'rb') as f:
                state_data = pickle.load(f)
            
            self.value_function = state_data.get('value_function', {})
            self.policy = state_data.get('policy', {})
            self.optimization_times = state_data.get('optimization_times', [])
            self.convergence_iterations = state_data.get('convergence_iterations', [])
            self.cost_reductions = state_data.get('cost_reductions', [])
            
            self.logger.info(f"DP optimization state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load DP state: {e}")

