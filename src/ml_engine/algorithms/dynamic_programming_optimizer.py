"""
Dynamic Programming based Traffic Signal Optimization
Implements optimal control theory for traffic signal timing optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum
import math

from config import DynamicProgrammingConfig, get_config
from data import TrafficData


class Phase(Enum):
    """Traffic signal phases"""
    NORTH_SOUTH = 0
    EAST_WEST = 1
    ALL_RED = 2


@dataclass
class TrafficState:
    """State representation for dynamic programming"""
    queue_lengths: Dict[str, float]  # Queue length for each approach
    arrival_rates: Dict[str, float]  # Arrival rate for each approach
    departure_rates: Dict[str, float]  # Departure rate for each approach
    current_phase: Phase
    phase_duration: float
    time_step: int


@dataclass
class ControlAction:
    """Control action for dynamic programming"""
    phase_duration: float
    next_phase: Phase
    green_time_allocation: Dict[str, float]


class DynamicProgrammingOptimizer:
    """
    Dynamic Programming based traffic signal optimizer
    
    This optimizer uses optimal control theory to find the optimal signal timing
    sequence that minimizes total system cost over a finite time horizon.
    
    The cost function considers:
    - Queue length (waiting time)
    - Throughput (vehicles processed)
    - Fuel consumption
    - Emissions
    """
    
    def __init__(self, config: Optional[DynamicProgrammingConfig] = None):
        self.config = config or get_config().dynamic_programming
        self.logger = logging.getLogger(__name__)
        
        # Time horizon parameters
        self.time_horizon = self.config.time_horizon
        self.time_step = self.config.time_step
        self.num_steps = self.time_horizon // self.time_step
        
        # Signal timing constraints
        self.max_cycle_time = self.config.max_cycle_time
        self.min_cycle_time = self.config.min_cycle_time
        self.min_green_time = self.config.min_green_time
        self.max_green_time = self.config.max_green_time
        self.yellow_time = self.config.yellow_time
        self.all_red_time = self.config.all_red_time
        
        # Cost function weights
        self.cost_weights = self.config.cost_weights
        
        # Optimization results
        self.optimal_policy: Optional[Dict] = None
        self.value_function: Optional[np.ndarray] = None
        
        # Traffic flow parameters
        self.saturation_flow_rate = 1800  # vehicles per hour per lane
        self.jam_density = 200  # vehicles per kilometer per lane
        self.free_flow_speed = 50  # kilometers per hour
        
    def traffic_data_to_state(self, traffic_data: TrafficData, current_phase: Phase = Phase.NORTH_SOUTH, phase_duration: float = 0.0) -> TrafficState:
        """Convert TrafficData to TrafficState"""
        # Extract queue lengths from lane counts
        queue_lengths = {
            'north': float(traffic_data.lane_counts.get('north_lane', 0)),
            'south': float(traffic_data.lane_counts.get('south_lane', 0)),
            'east': float(traffic_data.lane_counts.get('east_lane', 0)),
            'west': float(traffic_data.lane_counts.get('west_lane', 0))
        }
        
        # Estimate arrival rates based on historical patterns
        # In practice, this would be based on historical data analysis
        arrival_rates = self._estimate_arrival_rates(traffic_data)
        
        # Calculate departure rates based on current phase
        departure_rates = self._calculate_departure_rates(current_phase, traffic_data)
        
        return TrafficState(
            queue_lengths=queue_lengths,
            arrival_rates=arrival_rates,
            departure_rates=departure_rates,
            current_phase=current_phase,
            phase_duration=phase_duration,
            time_step=0
        )
    
    def _estimate_arrival_rates(self, traffic_data: TrafficData) -> Dict[str, float]:
        """Estimate arrival rates for each approach"""
        # Base arrival rate estimation
        # In practice, this would use historical data and prediction models
        base_rate = 0.5  # vehicles per second
        
        # Adjust based on time of day
        hour = traffic_data.timestamp.hour
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # Rush hours
            time_factor = 2.0
        elif 10 <= hour <= 16:  # Daytime
            time_factor = 1.5
        else:  # Night time
            time_factor = 0.5
        
        # Adjust based on weather
        weather_factor = 1.0
        if traffic_data.weather_condition == 'rainy':
            weather_factor = 0.8
        elif traffic_data.weather_condition == 'foggy':
            weather_factor = 0.6
        
        arrival_rates = {}
        for approach in ['north', 'south', 'east', 'west']:
            arrival_rates[approach] = base_rate * time_factor * weather_factor
        
        return arrival_rates
    
    def _calculate_departure_rates(self, phase: Phase, traffic_data: TrafficData) -> Dict[str, float]:
        """Calculate departure rates based on current phase"""
        departure_rates = {'north': 0.0, 'south': 0.0, 'east': 0.0, 'west': 0.0}
        
        if phase == Phase.NORTH_SOUTH:
            # North and south approaches can discharge
            departure_rates['north'] = self.saturation_flow_rate / 3600  # Convert to vehicles per second
            departure_rates['south'] = self.saturation_flow_rate / 3600
        elif phase == Phase.EAST_WEST:
            # East and west approaches can discharge
            departure_rates['east'] = self.saturation_flow_rate / 3600
            departure_rates['west'] = self.saturation_flow_rate / 3600
        # During all-red phase, no departures
        
        return departure_rates
    
    def _calculate_immediate_cost(self, state: TrafficState, action: ControlAction) -> float:
        """Calculate immediate cost for a state-action pair"""
        cost = 0.0
        
        # Queue length cost (waiting time)
        total_queue = sum(state.queue_lengths.values())
        queue_cost = self.cost_weights['wait_time'] * total_queue
        cost += queue_cost
        
        # Throughput cost (negative reward for processed vehicles)
        total_departure = sum(action.green_time_allocation.values()) * sum(state.departure_rates.values())
        throughput_cost = -self.cost_weights['throughput'] * total_departure
        cost += throughput_cost
        
        # Fuel consumption cost (proportional to queue length)
        fuel_cost = self.cost_weights['fuel_consumption'] * total_queue * 0.1
        cost += fuel_cost
        
        # Emissions cost (proportional to queue length)
        emission_cost = self.cost_weights['emissions'] * total_queue * 0.05
        cost += emission_cost
        
        return cost
    
    def _transition_function(self, state: TrafficState, action: ControlAction) -> TrafficState:
        """Calculate next state given current state and action"""
        # Update queue lengths based on arrivals and departures
        new_queue_lengths = {}
        for approach in state.queue_lengths:
            current_queue = state.queue_lengths[approach]
            arrival_rate = state.arrival_rates[approach]
            departure_rate = action.green_time_allocation.get(approach, 0) * state.departure_rates[approach]
            
            # Queue evolution: new_queue = max(0, current_queue + arrivals - departures)
            new_queue = max(0, current_queue + arrival_rate * self.time_step - departure_rate * self.time_step)
            new_queue_lengths[approach] = new_queue
        
        # Update phase duration
        new_phase_duration = action.phase_duration
        
        # Determine next phase
        next_phase = action.next_phase
        
        return TrafficState(
            queue_lengths=new_queue_lengths,
            arrival_rates=state.arrival_rates,  # Assume arrival rates remain constant
            departure_rates=self._calculate_departure_rates(next_phase, None),  # Will need traffic data
            current_phase=next_phase,
            phase_duration=new_phase_duration,
            time_step=state.time_step + 1
        )
    
    def _get_feasible_actions(self, state: TrafficState) -> List[ControlAction]:
        """Get all feasible actions for a given state"""
        actions = []
        
        # Determine possible phase transitions
        if state.current_phase == Phase.NORTH_SOUTH:
            next_phases = [Phase.EAST_WEST, Phase.ALL_RED]
        elif state.current_phase == Phase.EAST_WEST:
            next_phases = [Phase.NORTH_SOUTH, Phase.ALL_RED]
        else:  # ALL_RED
            next_phases = [Phase.NORTH_SOUTH, Phase.EAST_WEST]
        
        # Generate actions for each possible phase transition
        for next_phase in next_phases:
            # Calculate minimum green time needed
            min_green_time = self.min_green_time
            
            # Calculate maximum green time allowed
            max_green_time = min(self.max_green_time, self.max_cycle_time - state.phase_duration)
            
            # Generate actions with different green time allocations
            for green_time in range(int(min_green_time), int(max_green_time) + 1, 5):
                # Allocate green time to appropriate approaches
                green_time_allocation = {}
                if next_phase == Phase.NORTH_SOUTH:
                    green_time_allocation = {
                        'north': green_time,
                        'south': green_time,
                        'east': 0,
                        'west': 0
                    }
                elif next_phase == Phase.EAST_WEST:
                    green_time_allocation = {
                        'north': 0,
                        'south': 0,
                        'east': green_time,
                        'west': green_time
                    }
                else:  # ALL_RED
                    green_time_allocation = {'north': 0, 'south': 0, 'east': 0, 'west': 0}
                
                action = ControlAction(
                    phase_duration=green_time,
                    next_phase=next_phase,
                    green_time_allocation=green_time_allocation
                )
                actions.append(action)
        
        return actions
    
    def solve_optimal_control(self, initial_state: TrafficState) -> Tuple[Dict, np.ndarray]:
        """
        Solve the optimal control problem using dynamic programming
        
        Returns:
            optimal_policy: Dictionary mapping states to optimal actions
            value_function: Value function for each state
        """
        self.logger.info("Solving optimal control problem using dynamic programming...")
        
        # Initialize value function
        # For simplicity, we'll use a discretized state space
        max_queue = 50  # Maximum queue length to consider
        queue_discretization = 5  # Discretization step
        num_queue_levels = max_queue // queue_discretization + 1
        
        # Value function: V[time_step][north_queue][south_queue][east_queue][west_queue][phase]
        value_function = np.full((
            self.num_steps + 1,
            num_queue_levels,
            num_queue_levels,
            num_queue_levels,
            num_queue_levels,
            len(Phase)
        ), float('inf'))
        
        # Terminal condition: V(T, s) = 0 for all states
        value_function[-1, :, :, :, :, :] = 0
        
        # Backward induction
        for t in range(self.num_steps - 1, -1, -1):
            self.logger.debug(f"Solving time step {t}/{self.num_steps}")
            
            for north_q in range(num_queue_levels):
                for south_q in range(num_queue_levels):
                    for east_q in range(num_queue_levels):
                        for west_q in range(num_queue_levels):
                            for phase_idx, phase in enumerate(Phase):
                                # Create state
                                state = TrafficState(
                                    queue_lengths={
                                        'north': north_q * queue_discretization,
                                        'south': south_q * queue_discretization,
                                        'east': east_q * queue_discretization,
                                        'west': west_q * queue_discretization
                                    },
                                    arrival_rates={'north': 0.5, 'south': 0.5, 'east': 0.5, 'west': 0.5},
                                    departure_rates=self._calculate_departure_rates(phase, None),
                                    current_phase=phase,
                                    phase_duration=0,
                                    time_step=t
                                )
                                
                                # Get feasible actions
                                actions = self._get_feasible_actions(state)
                                
                                if not actions:
                                    continue
                                
                                # Find optimal action
                                min_cost = float('inf')
                                optimal_action = None
                                
                                for action in actions:
                                    # Calculate immediate cost
                                    immediate_cost = self._calculate_immediate_cost(state, action)
                                    
                                    # Calculate expected future cost
                                    next_state = self._transition_function(state, action)
                                    
                                    # Discretize next state
                                    next_north_q = min(num_queue_levels - 1, int(next_state.queue_lengths['north'] // queue_discretization))
                                    next_south_q = min(num_queue_levels - 1, int(next_state.queue_lengths['south'] // queue_discretization))
                                    next_east_q = min(num_queue_levels - 1, int(next_state.queue_lengths['east'] // queue_discretization))
                                    next_west_q = min(num_queue_levels - 1, int(next_state.queue_lengths['west'] // queue_discretization))
                                    next_phase_idx = next_state.current_phase.value
                                    
                                    # Future cost
                                    future_cost = value_function[t + 1, next_north_q, next_south_q, next_east_q, next_west_q, next_phase_idx]
                                    
                                    # Total cost
                                    total_cost = immediate_cost + future_cost
                                    
                                    if total_cost < min_cost:
                                        min_cost = total_cost
                                        optimal_action = action
                                
                                # Update value function
                                value_function[t, north_q, south_q, east_q, west_q, phase_idx] = min_cost
        
        self.logger.info("Optimal control problem solved")
        return {}, value_function  # Simplified return for now
    
    def optimize_signal_timing(self, traffic_data: TrafficData, current_timings: Dict[str, int]) -> Dict[str, int]:
        """
        Optimize signal timing using dynamic programming
        
        Args:
            traffic_data: Current traffic data
            current_timings: Current signal timings
            
        Returns:
            Optimized signal timings
        """
        # Convert traffic data to state
        initial_state = self.traffic_data_to_state(traffic_data)
        
        # Solve optimal control problem
        optimal_policy, value_function = self.solve_optimal_control(initial_state)
        
        # For now, use a simplified optimization based on current queue lengths
        optimized_timings = self._simplified_optimization(traffic_data, current_timings)
        
        self.logger.info("Dynamic programming optimization completed")
        return optimized_timings
    
    def _simplified_optimization(self, traffic_data: TrafficData, current_timings: Dict[str, int]) -> Dict[str, int]:
        """Simplified optimization based on queue lengths"""
        optimized_timings = current_timings.copy()
        
        # Calculate total queue length for each direction
        north_south_queue = traffic_data.lane_counts.get('north_lane', 0) + traffic_data.lane_counts.get('south_lane', 0)
        east_west_queue = traffic_data.lane_counts.get('east_lane', 0) + traffic_data.lane_counts.get('west_lane', 0)
        
        # Allocate green time based on queue lengths
        total_queue = north_south_queue + east_west_queue
        if total_queue > 0:
            north_south_ratio = north_south_queue / total_queue
            east_west_ratio = east_west_queue / total_queue
            
            # Calculate optimal green times
            total_cycle_time = self.min_cycle_time + (self.max_cycle_time - self.min_cycle_time) * min(1.0, total_queue / 50)
            
            north_south_green = max(self.min_green_time, min(self.max_green_time, total_cycle_time * north_south_ratio))
            east_west_green = max(self.min_green_time, min(self.max_green_time, total_cycle_time * east_west_ratio))
            
            # Update timings
            optimized_timings['north_lane'] = int(north_south_green)
            optimized_timings['south_lane'] = int(north_south_green)
            optimized_timings['east_lane'] = int(east_west_green)
            optimized_timings['west_lane'] = int(east_west_green)
        
        return optimized_timings
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            "time_horizon": self.time_horizon,
            "time_step": self.time_step,
            "num_steps": self.num_steps,
            "cost_weights": self.cost_weights,
            "saturation_flow_rate": self.saturation_flow_rate
        }


