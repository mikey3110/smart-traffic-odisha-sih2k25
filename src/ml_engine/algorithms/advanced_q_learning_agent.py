"""
Advanced Q-Learning Agent for Multi-Intersection Traffic Signal Optimization
Production-ready implementation with sophisticated state/action spaces and coordination
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
import json
import os
import time
import threading
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
from queue import Queue, Empty
import pickle
import hashlib

from config.ml_config import get_config


@dataclass
class MultiDimensionalState:
    """
    Enhanced multi-dimensional state space for Q-Learning
    Includes traffic data, temporal features, environmental factors, and adjacent signal states
    """
    # Core traffic data (4 lanes)
    lane_counts: np.ndarray  # [north, south, east, west]
    avg_speed: float
    queue_lengths: np.ndarray
    waiting_times: np.ndarray
    flow_rates: np.ndarray  # vehicles per hour
    
    # Current signal state
    current_phase: int  # 0-3: north-south, east-west, left turns, pedestrian
    phase_duration: float
    cycle_progress: float  # 0-1 progress through cycle
    time_since_change: float
    
    # Temporal features
    time_of_day: float  # 0-1 normalized hour
    day_of_week: float  # 0-1 normalized day
    is_weekend: bool
    is_holiday: bool
    season: int  # 0-3: spring, summer, fall, winter
    
    # Environmental factors
    weather_condition: int  # 0-5: clear, cloudy, rainy, foggy, stormy, snowy
    temperature: float
    visibility: float
    precipitation_intensity: float
    
    # Adjacent intersections state (for coordination)
    adjacent_signals: Dict[str, Dict[str, Any]]  # {intersection_id: {phase, duration, congestion}}
    upstream_flow: np.ndarray  # Flow from upstream intersections
    downstream_capacity: np.ndarray  # Available capacity downstream
    
    # Historical context
    recent_performance: Dict[str, float]  # Recent wait times, throughput, etc.
    congestion_trend: float  # -1 to 1: decreasing to increasing
    emergency_vehicles: bool
    
    def to_vector(self) -> np.ndarray:
        """Convert state to normalized feature vector for neural network"""
        features = []
        
        # Traffic features (normalized)
        features.extend(self.lane_counts / 50.0)  # Max 50 vehicles per lane
        features.append(self.avg_speed / 100.0)  # Max 100 km/h
        features.extend(self.queue_lengths / 30.0)  # Max 30 vehicles in queue
        features.extend(self.waiting_times / 120.0)  # Max 2 minutes wait
        features.extend(self.flow_rates / 1000.0)  # Max 1000 veh/h
        
        # Signal state features
        features.append(self.current_phase / 4.0)  # 4 phases
        features.append(self.phase_duration / 120.0)  # Max 2 minutes
        features.append(self.cycle_progress)
        features.append(self.time_since_change / 60.0)  # Max 1 minute
        
        # Temporal features
        features.append(self.time_of_day)
        features.append(self.day_of_week)
        features.append(1.0 if self.is_weekend else 0.0)
        features.append(1.0 if self.is_holiday else 0.0)
        features.append(self.season / 4.0)
        
        # Environmental features
        weather_one_hot = [0.0] * 6
        weather_one_hot[self.weather_condition] = 1.0
        features.extend(weather_one_hot)
        features.append(self.temperature / 50.0)  # Max 50°C
        features.append(self.visibility / 20.0)  # Max 20km visibility
        features.append(self.precipitation_intensity)
        
        # Adjacent signals features
        for intersection_id in ['upstream_1', 'upstream_2', 'downstream_1', 'downstream_2']:
            if intersection_id in self.adjacent_signals:
                adj_data = self.adjacent_signals[intersection_id]
                features.append(adj_data.get('phase', 0) / 4.0)
                features.append(adj_data.get('duration', 0) / 120.0)
                features.append(adj_data.get('congestion', 0))
            else:
                features.extend([0.0, 0.0, 0.0])  # No data available
        
        # Flow and capacity features
        features.extend(self.upstream_flow / 1000.0)
        features.extend(self.downstream_capacity / 1000.0)
        
        # Historical context
        features.append(self.recent_performance.get('avg_wait_time', 0) / 120.0)
        features.append(self.recent_performance.get('throughput', 0) / 1000.0)
        features.append(self.congestion_trend)
        features.append(1.0 if self.emergency_vehicles else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def get_state_hash(self) -> str:
        """Get unique hash for state (for Q-table lookup)"""
        state_str = json.dumps({
            'lane_counts': self.lane_counts.tolist(),
            'current_phase': self.current_phase,
            'phase_duration': round(self.phase_duration, 1),
            'time_of_day': round(self.time_of_day, 2)
        }, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()[:16]


@dataclass
class SophisticatedAction:
    """
    Sophisticated action space with dynamic phase durations and safety constraints
    """
    action_type: int  # 0-7: different optimization strategies
    green_time_adjustments: np.ndarray  # [north, south, east, west] adjustments
    cycle_time_adjustment: int  # Overall cycle time change
    phase_sequence_change: bool  # Whether to change phase order
    priority_boost: bool  # Emergency vehicle priority
    coordination_signal: str  # Signal to send to adjacent intersections
    safety_override: bool  # Safety constraint override
    
    # Safety constraints
    min_green_time: int = 10  # Minimum green time in seconds
    max_green_time: int = 120  # Maximum green time in seconds
    min_cycle_time: int = 60  # Minimum cycle time
    max_cycle_time: int = 180  # Maximum cycle time
    
    def to_vector(self) -> np.ndarray:
        """Convert action to vector representation"""
        return np.array([
            self.action_type / 7.0,
            np.mean(self.green_time_adjustments) / 30.0,
            self.cycle_time_adjustment / 30.0,
            1.0 if self.phase_sequence_change else 0.0,
            1.0 if self.priority_boost else 0.0,
            1.0 if self.safety_override else 0.0
        ], dtype=np.float32)
    
    def apply_safety_constraints(self, current_timings: Dict[str, int]) -> Dict[str, int]:
        """Apply safety constraints to action"""
        new_timings = current_timings.copy()
        
        for i, lane in enumerate(['north_lane', 'south_lane', 'east_lane', 'west_lane']):
            current_time = new_timings.get(lane, 30)
            adjustment = self.green_time_adjustments[i]
            new_time = current_time + adjustment
            
            # Apply safety constraints
            new_time = max(self.min_green_time, min(self.max_green_time, new_time))
            new_timings[lane] = new_time
        
        # Apply cycle time constraints
        total_cycle = sum(new_timings.values())
        if total_cycle < self.min_cycle_time:
            # Scale up proportionally
            scale_factor = self.min_cycle_time / total_cycle
            for lane in new_timings:
                new_timings[lane] = int(new_timings[lane] * scale_factor)
        elif total_cycle > self.max_cycle_time:
            # Scale down proportionally
            scale_factor = self.max_cycle_time / total_cycle
            for lane in new_timings:
                new_timings[lane] = int(new_timings[lane] * scale_factor)
        
        return new_timings


class AdvancedDQNNetwork(nn.Module):
    """
    Advanced Deep Q-Network with attention mechanism and residual connections
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.1, use_attention: bool = True, use_residual: bool = True):
        super(AdvancedDQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Input projection
        self.input_projection = nn.Linear(state_size, hidden_layers[0])
        
        # Main network with residual connections
        self.layers = nn.ModuleList()
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_layers[-1],
                num_heads=8,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_layers[-1])
        
        # Output layers for different action components
        self.q_value_head = nn.Linear(hidden_layers[-1], action_size)
        self.advantage_head = nn.Linear(hidden_layers[-1], action_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)
        
        # Main network with residual connections
        residual = x
        for i in range(0, len(self.layers), 3):  # Skip by 3 for layer, activation, dropout
            if i < len(self.layers):
                x = self.layers[i](x)  # Linear layer
                if i + 1 < len(self.layers):
                    x = self.layers[i + 1](x)  # Activation
                if i + 2 < len(self.layers):
                    x = self.layers[i + 2](x)  # Dropout
                
                # Residual connection
                if self.use_residual and x.shape == residual.shape:
                    x = x + residual
                    residual = x
        
        # Attention mechanism
        if self.use_attention and x.dim() == 2:
            x_attn = x.unsqueeze(1)  # Add sequence dimension
            attended, _ = self.attention(x_attn, x_attn, x_attn)
            x = self.attention_norm(attended.squeeze(1))
        
        # Dueling DQN: separate value and advantage streams
        value = self.q_value_head(x)
        advantage = self.advantage_head(x)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class PrioritizedExperienceReplay:
    """
    Prioritized Experience Replay Buffer with importance sampling
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done, td_error: float = None):
        """Add experience with priority"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        # Calculate priority based on TD error
        if td_error is not None:
            priority = (abs(td_error) + 1e-6) ** self.alpha
        else:
            priority = self.max_priority
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = priority
        self.max_priority = max(self.max_priority, priority)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch with importance sampling weights"""
        if len(self.buffer) == 0:
            return [], np.array([]), np.array([])
        
        # Calculate sampling probabilities
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize weights
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities for sampled experiences"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class MultiIntersectionCoordinator:
    """
    Handles communication and coordination between adjacent intersections
    """
    
    def __init__(self, intersection_id: str, adjacent_intersections: List[str]):
        self.intersection_id = intersection_id
        self.adjacent_intersections = adjacent_intersections
        self.coordination_data = {}
        self.coordination_lock = threading.Lock()
        self.logger = logging.getLogger(f"coordinator_{intersection_id}")
    
    def send_coordination_signal(self, action: SophisticatedAction, target_intersections: List[str] = None):
        """Send coordination signal to adjacent intersections"""
        if target_intersections is None:
            target_intersections = self.adjacent_intersections
        
        coordination_data = {
            'sender': self.intersection_id,
            'timestamp': datetime.now().isoformat(),
            'action_type': action.action_type,
            'cycle_adjustment': action.cycle_time_adjustment,
            'priority_boost': action.priority_boost,
            'coordination_signal': action.coordination_signal
        }
        
        with self.coordination_lock:
            for target in target_intersections:
                self.coordination_data[target] = coordination_data
        
        self.logger.debug(f"Sent coordination signal to {target_intersections}")
    
    def receive_coordination_data(self, sender: str, data: Dict[str, Any]):
        """Receive coordination data from adjacent intersection"""
        with self.coordination_lock:
            self.coordination_data[sender] = data
        
        self.logger.debug(f"Received coordination data from {sender}")
    
    def get_coordination_context(self) -> Dict[str, Any]:
        """Get current coordination context from adjacent intersections"""
        with self.coordination_lock:
            return self.coordination_data.copy()
    
    def calculate_coordination_reward(self, action: SophisticatedAction, 
                                    adjacent_states: Dict[str, MultiDimensionalState]) -> float:
        """Calculate reward component for coordination"""
        coordination_reward = 0.0
        
        # Green wave coordination
        for adj_id, adj_state in adjacent_states.items():
            if adj_id in self.coordination_data:
                adj_action = self.coordination_data[adj_id].get('action_type', 0)
                
                # Reward synchronized actions
                if action.action_type == adj_action:
                    coordination_reward += 0.1
                
                # Reward complementary timing
                if (action.cycle_time_adjustment > 0 and 
                    self.coordination_data[adj_id].get('cycle_adjustment', 0) > 0):
                    coordination_reward += 0.05
        
        return coordination_reward


class AdvancedQLearningAgent:
    """
    Production-ready Q-Learning agent for multi-intersection traffic optimization
    """
    
    def __init__(self, intersection_id: str, config: Optional[Dict] = None):
        self.intersection_id = intersection_id
        self.config = config or get_config()
        self.logger = logging.getLogger(f"qlearning_{intersection_id}")
        
        # State and action dimensions
        self.state_size = 45  # Enhanced state representation
        self.action_size = 8  # 8 different action types
        
        # Initialize networks
        self.q_network = AdvancedDQNNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_layers=[256, 128, 64],
            dropout_rate=0.1,
            use_attention=True,
            use_residual=True
        )
        
        self.target_network = AdvancedDQNNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_layers=[256, 128, 64],
            dropout_rate=0.1,
            use_attention=True,
            use_residual=True
        )
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer with adaptive learning rate
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=50,
            verbose=True
        )
        
        # Experience replay
        self.replay_buffer = PrioritizedExperienceReplay(
            capacity=self.config.get('replay_buffer_size', 50000),
            alpha=0.6,
            beta=0.4
        )
        
        # Multi-intersection coordination
        adjacent_intersections = self.config.get('adjacent_intersections', [])
        self.coordinator = MultiIntersectionCoordinator(intersection_id, adjacent_intersections)
        
        # Training parameters
        self.epsilon = self.config.get('epsilon', 0.1)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.batch_size = self.config.get('batch_size', 64)
        self.target_update_frequency = self.config.get('target_update_frequency', 200)
        self.discount_factor = self.config.get('discount_factor', 0.95)
        
        # Training state
        self.training_step = 0
        self.episode_rewards = []
        self.loss_history = []
        self.performance_history = []
        
        # Action mapping
        self.action_mapping = {
            0: {"strategy": "maintain", "description": "Maintain current timing"},
            1: {"strategy": "reduce_congestion", "description": "Reduce congestion in high-traffic lanes"},
            2: {"strategy": "increase_throughput", "description": "Increase overall throughput"},
            3: {"strategy": "coordinate_green_wave", "description": "Coordinate with adjacent intersections"},
            4: {"strategy": "emergency_priority", "description": "Give priority to emergency vehicles"},
            5: {"strategy": "pedestrian_friendly", "description": "Optimize for pedestrian safety"},
            6: {"strategy": "fuel_efficient", "description": "Minimize fuel consumption"},
            7: {"strategy": "adaptive_cycle", "description": "Adapt cycle time to traffic demand"}
        }
        
        # Performance tracking
        self.optimization_history = []
        self.reward_components = {
            'wait_time_reduction': [],
            'throughput_increase': [],
            'fuel_efficiency': [],
            'pedestrian_safety': [],
            'coordination': [],
            'stability': []
        }
        
        self.logger.info(f"Advanced Q-Learning agent initialized for {intersection_id}")
    
    def create_state(self, traffic_data: Dict[str, Any], 
                    current_timings: Dict[str, int],
                    historical_data: List[Dict] = None,
                    adjacent_states: Dict[str, MultiDimensionalState] = None) -> MultiDimensionalState:
        """Create enhanced multi-dimensional state from traffic data"""
        
        # Extract lane data
        lane_counts = np.array([
            traffic_data.get('lane_counts', {}).get('north_lane', 0),
            traffic_data.get('lane_counts', {}).get('south_lane', 0),
            traffic_data.get('lane_counts', {}).get('east_lane', 0),
            traffic_data.get('lane_counts', {}).get('west_lane', 0)
        ], dtype=np.float32)
        
        # Calculate derived metrics
        avg_speed = traffic_data.get('avg_speed', 0.0)
        queue_lengths = lane_counts * 0.8  # Estimate queue length
        waiting_times = queue_lengths * 2.0  # Estimate waiting time
        flow_rates = lane_counts * 60.0  # Estimate flow rate (veh/h)
        
        # Signal state
        current_phase = traffic_data.get('current_phase', 0)
        phase_duration = traffic_data.get('phase_duration', 30.0)
        cycle_progress = phase_duration / sum(current_timings.values())
        time_since_change = traffic_data.get('time_since_change', 0.0)
        
        # Temporal features
        timestamp = datetime.fromisoformat(traffic_data.get('timestamp', datetime.now().isoformat()))
        time_of_day = timestamp.hour / 24.0
        day_of_week = timestamp.weekday() / 7.0
        is_weekend = timestamp.weekday() >= 5
        is_holiday = self._is_holiday(timestamp)
        season = self._get_season(timestamp)
        
        # Environmental features
        weather_condition = self._encode_weather(traffic_data.get('weather_condition', 'clear'))
        temperature = traffic_data.get('temperature', 20.0)
        visibility = traffic_data.get('visibility', 10.0)
        precipitation_intensity = traffic_data.get('precipitation_intensity', 0.0)
        
        # Adjacent signals state
        adjacent_signals = {}
        if adjacent_states:
            for adj_id, adj_state in adjacent_states.items():
                adjacent_signals[adj_id] = {
                    'phase': adj_state.current_phase,
                    'duration': adj_state.phase_duration,
                    'congestion': adj_state.congestion_trend
                }
        
        # Flow and capacity estimates
        upstream_flow = np.array([0.0, 0.0, 0.0, 0.0])  # Placeholder
        downstream_capacity = np.array([1000.0, 1000.0, 1000.0, 1000.0])  # Placeholder
        
        # Historical context
        recent_performance = self._calculate_recent_performance(historical_data)
        congestion_trend = self._calculate_congestion_trend(historical_data)
        emergency_vehicles = traffic_data.get('emergency_vehicles', False)
        
        return MultiDimensionalState(
            lane_counts=lane_counts,
            avg_speed=avg_speed,
            queue_lengths=queue_lengths,
            waiting_times=waiting_times,
            flow_rates=flow_rates,
            current_phase=current_phase,
            phase_duration=phase_duration,
            cycle_progress=cycle_progress,
            time_since_change=time_since_change,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            is_holiday=is_holiday,
            season=season,
            weather_condition=weather_condition,
            temperature=temperature,
            visibility=visibility,
            precipitation_intensity=precipitation_intensity,
            adjacent_signals=adjacent_signals,
            upstream_flow=upstream_flow,
            downstream_capacity=downstream_capacity,
            recent_performance=recent_performance,
            congestion_trend=congestion_trend,
            emergency_vehicles=emergency_vehicles
        )
    
    def select_action(self, state: MultiDimensionalState, training: bool = True) -> SophisticatedAction:
        """Select action using epsilon-greedy policy with exploration bonus"""
        
        if training and random.random() < self.epsilon:
            # Random action for exploration
            action_type = random.randint(0, self.action_size - 1)
        else:
            # Greedy action
            state_vector = torch.FloatTensor(state.to_vector()).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_vector)
                action_type = q_values.argmax().item()
        
        # Create sophisticated action based on action type
        action = self._create_sophisticated_action(action_type, state)
        
        # Send coordination signal
        self.coordinator.send_coordination_signal(action)
        
        return action
    
    def _create_sophisticated_action(self, action_type: int, state: MultiDimensionalState) -> SophisticatedAction:
        """Create sophisticated action based on action type and current state"""
        
        # Base adjustments for different strategies
        base_adjustments = {
            0: np.array([0, 0, 0, 0]),  # maintain
            1: np.array([5, 5, -5, -5]),  # reduce congestion
            2: np.array([10, 10, 10, 10]),  # increase throughput
            3: np.array([0, 0, 0, 0]),  # coordinate green wave
            4: np.array([15, 15, -10, -10]),  # emergency priority
            5: np.array([-5, -5, 5, 5]),  # pedestrian friendly
            6: np.array([-10, -10, -10, -10]),  # fuel efficient
            7: np.array([5, 5, 5, 5])  # adaptive cycle
        }
        
        # Adjust based on congestion levels
        congestion_factor = state.congestion_trend
        adjustments = base_adjustments[action_type] * (1 + congestion_factor)
        
        # Emergency vehicle priority
        priority_boost = state.emergency_vehicles or action_type == 4
        
        # Coordination signal
        coordination_signals = {
            0: "maintain",
            1: "reduce_congestion",
            2: "increase_throughput",
            3: "green_wave",
            4: "emergency",
            5: "pedestrian",
            6: "fuel_efficient",
            7: "adaptive"
        }
        
        return SophisticatedAction(
            action_type=action_type,
            green_time_adjustments=adjustments.astype(int),
            cycle_time_adjustment=int(adjustments.sum() / 4),  # Average adjustment
            phase_sequence_change=action_type == 3,  # Green wave coordination
            priority_boost=priority_boost,
            coordination_signal=coordination_signals[action_type],
            safety_override=priority_boost
        )
    
    def calculate_advanced_reward(self, state: MultiDimensionalState, action: SophisticatedAction,
                                next_state: MultiDimensionalState, performance_metrics: Dict[str, float] = None) -> float:
        """
        Calculate advanced multi-objective reward function
        """
        reward = 0.0
        
        # Wait time reduction (primary objective)
        wait_time_reduction = np.sum(state.waiting_times) - np.sum(next_state.waiting_times)
        wait_time_reward = wait_time_reduction * 0.2
        reward += wait_time_reward
        self.reward_components['wait_time_reduction'].append(wait_time_reward)
        
        # Throughput increase
        throughput_increase = np.sum(next_state.flow_rates) - np.sum(state.flow_rates)
        throughput_reward = throughput_increase * 0.1
        reward += throughput_reward
        self.reward_components['throughput_increase'].append(throughput_reward)
        
        # Fuel efficiency (penalty for idling)
        idling_vehicles = np.sum(state.queue_lengths)
        fuel_efficiency_reward = -idling_vehicles * 0.05
        reward += fuel_efficiency_reward
        self.reward_components['fuel_efficiency'].append(fuel_efficiency_reward)
        
        # Pedestrian safety (bonus for pedestrian-friendly actions)
        if action.action_type == 5:  # Pedestrian friendly
            pedestrian_reward = 0.1
            reward += pedestrian_reward
        else:
            pedestrian_reward = 0.0
        self.reward_components['pedestrian_safety'].append(pedestrian_reward)
        
        # Coordination reward
        coordination_reward = self.coordinator.calculate_coordination_reward(action, {})
        reward += coordination_reward
        self.reward_components['coordination'].append(coordination_reward)
        
        # Stability reward (penalty for frequent changes)
        if hasattr(self, 'last_action') and self.last_action.action_type != action.action_type:
            stability_reward = -0.02
            reward += stability_reward
        else:
            stability_reward = 0.01
        self.reward_components['stability'].append(stability_reward)
        
        # Performance metrics bonus
        if performance_metrics:
            if 'wait_time_improvement' in performance_metrics:
                reward += performance_metrics['wait_time_improvement'] * 0.15
            if 'throughput_improvement' in performance_metrics:
                reward += performance_metrics['throughput_improvement'] * 0.1
            if 'fuel_savings' in performance_metrics:
                reward += performance_metrics['fuel_savings'] * 0.05
        
        self.last_action = action
        
        return reward
    
    def optimize_signal_timing(self, traffic_data: Dict[str, Any], 
                             current_timings: Dict[str, int],
                             historical_data: List[Dict] = None,
                             adjacent_states: Dict[str, MultiDimensionalState] = None) -> Dict[str, int]:
        """
        Optimize signal timing using advanced Q-Learning
        """
        # Create state
        state = self.create_state(traffic_data, current_timings, historical_data, adjacent_states)
        
        # Select action
        action = self.select_action(state, training=False)
        
        # Apply action with safety constraints
        optimized_timings = action.apply_safety_constraints(current_timings)
        
        # Log optimization decision
        self.logger.info(
            f"Q-Learning optimization: Action {action.action_type} -> "
            f"Adjustments: {action.green_time_adjustments}, "
            f"Cycle: {action.cycle_time_adjustment}s, "
            f"Priority: {action.priority_boost}"
        )
        
        # Store for learning
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'state': state,
            'action': action,
            'optimized_timings': optimized_timings.copy()
        })
        
        return optimized_timings
    
    def train_step(self):
        """Perform one training step with prioritized experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch, weights, indices = self.replay_buffer.sample(self.batch_size)
        
        if not batch:
            return
        
        # Prepare batch data
        states = torch.FloatTensor([s.to_vector() for s, _, _, _, _ in batch])
        actions = torch.LongTensor([a.action_type for _, a, _, _, _ in batch])
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch])
        next_states = torch.FloatTensor([s.to_vector() for _, _, _, s, _ in batch])
        dones = torch.BoolTensor([d for _, _, _, _, d in batch])
        weights = torch.FloatTensor(weights)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.discount_factor * next_q_values * ~dones.unsqueeze(1))
        
        # Calculate TD errors
        td_errors = torch.abs(target_q_values.squeeze() - current_q_values.squeeze())
        
        # Calculate loss with importance sampling
        loss = torch.nn.functional.smooth_l1_loss(current_q_values.squeeze(), target_q_values.squeeze(), reduction='none')
        weighted_loss = (loss * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors.detach().numpy())
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.logger.info(f"Target network updated at step {self.training_step}")
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update learning rate
        self.scheduler.step(weighted_loss.item())
        
        # Log training progress
        self.loss_history.append(weighted_loss.item())
        if self.training_step % 100 == 0:
            avg_loss = np.mean(self.loss_history[-100:])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(
                f"Training step {self.training_step}, Loss: {avg_loss:.4f}, "
                f"Epsilon: {self.epsilon:.4f}, LR: {current_lr:.6f}"
            )
    
    def add_experience(self, state: MultiDimensionalState, action: SophisticatedAction,
                      reward: float, next_state: MultiDimensionalState, done: bool):
        """Add experience to replay buffer"""
        # Calculate TD error for prioritization
        with torch.no_grad():
            current_q = self.q_network(torch.FloatTensor(state.to_vector()).unsqueeze(0))
            current_q_value = current_q[0][action.action_type].item()
            
            next_q = self.target_network(torch.FloatTensor(next_state.to_vector()).unsqueeze(0))
            next_q_value = next_q.max().item()
            target_q = reward + (self.discount_factor * next_q_value * (1 - done))
            
            td_error = abs(target_q - current_q_value)
        
        self.replay_buffer.push(state, action, reward, next_state, done, td_error)
    
    def save_model(self, filepath: str):
        """Save the trained model with all components"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'config': self.config,
            'loss_history': self.loss_history,
            'episode_rewards': self.episode_rewards,
            'performance_history': self.performance_history,
            'reward_components': self.reward_components,
            'action_mapping': self.action_mapping
        }
        
        torch.save(model_data, filepath)
        self.logger.info(f"Advanced Q-Learning model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if not os.path.exists(filepath):
            self.logger.warning(f"Model file {filepath} not found")
            return
        
        model_data = torch.load(filepath, map_location='cpu')
        
        self.q_network.load_state_dict(model_data['q_network_state_dict'])
        self.target_network.load_state_dict(model_data['target_network_state_dict'])
        self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        self.scheduler.load_state_dict(model_data['scheduler_state_dict'])
        self.epsilon = model_data.get('epsilon', self.epsilon)
        self.training_step = model_data.get('training_step', 0)
        self.loss_history = model_data.get('loss_history', [])
        self.episode_rewards = model_data.get('episode_rewards', [])
        self.performance_history = model_data.get('performance_history', [])
        self.reward_components = model_data.get('reward_components', self.reward_components)
        
        self.logger.info(f"Advanced Q-Learning model loaded from {filepath}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        return {
            "intersection_id": self.intersection_id,
            "training_step": self.training_step,
            "epsilon": self.epsilon,
            "replay_buffer_size": len(self.replay_buffer),
            "avg_loss": np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            "total_episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            "current_lr": self.optimizer.param_groups[0]['lr'],
            "optimization_count": len(self.optimization_history),
            "reward_components": {
                component: np.mean(values[-100:]) if values else 0
                for component, values in self.reward_components.items()
            }
        }
    
    # Helper methods
    def _is_holiday(self, date: datetime) -> bool:
        """Check if date is a holiday (simplified)"""
        # Simplified holiday detection
        holidays = [(1, 1), (7, 4), (12, 25)]  # New Year, Independence Day, Christmas
        return (date.month, date.day) in holidays
    
    def _get_season(self, date: datetime) -> int:
        """Get season (0-3: spring, summer, fall, winter)"""
        month = date.month
        if month in [3, 4, 5]:
            return 0  # Spring
        elif month in [6, 7, 8]:
            return 1  # Summer
        elif month in [9, 10, 11]:
            return 2  # Fall
        else:
            return 3  # Winter
    
    def _encode_weather(self, weather: str) -> int:
        """Encode weather condition to integer"""
        weather_map = {
            'clear': 0, 'cloudy': 1, 'rainy': 2,
            'foggy': 3, 'stormy': 4, 'snowy': 5
        }
        return weather_map.get(weather.lower(), 0)
    
    def _calculate_recent_performance(self, historical_data: List[Dict]) -> Dict[str, float]:
        """Calculate recent performance metrics"""
        if not historical_data or len(historical_data) < 2:
            return {'avg_wait_time': 0.0, 'throughput': 0.0}
        
        recent_data = historical_data[-5:]  # Last 5 data points
        
        avg_wait_times = []
        throughputs = []
        
        for data in recent_data:
            if 'waiting_times' in data:
                avg_wait_times.append(np.mean(data['waiting_times']))
            if 'flow_rates' in data:
                throughputs.append(np.sum(data['flow_rates']))
        
        return {
            'avg_wait_time': np.mean(avg_wait_times) if avg_wait_times else 0.0,
            'throughput': np.mean(throughputs) if throughputs else 0.0
        }
    
    def _calculate_congestion_trend(self, historical_data: List[Dict]) -> float:
        """Calculate congestion trend (-1 to 1)"""
        if not historical_data or len(historical_data) < 3:
            return 0.0
        
        recent_data = historical_data[-3:]
        congestion_levels = []
        
        for data in recent_data:
            if 'lane_counts' in data:
                total_vehicles = sum(data['lane_counts'].values()) if isinstance(data['lane_counts'], dict) else np.sum(data['lane_counts'])
                congestion_levels.append(total_vehicles)
        
        if len(congestion_levels) < 2:
            return 0.0
        
        # Calculate trend
        trend = np.polyfit(range(len(congestion_levels)), congestion_levels, 1)[0]
        return np.clip(trend / 10.0, -1.0, 1.0)  # Normalize trend
