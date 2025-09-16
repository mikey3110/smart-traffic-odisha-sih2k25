"""
Q-Learning based Traffic Signal Optimization
Implements Deep Q-Network (DQN) for adaptive signal timing optimization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import json
import os

from config import QLearningConfig, get_config
from data import TrafficData


@dataclass
class QLearningState:
    """State representation for Q-Learning"""
    lane_counts: np.ndarray
    avg_speed: float
    time_features: np.ndarray
    weather_features: np.ndarray
    environmental_features: np.ndarray
    current_phase: int
    phase_duration: float
    
    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector"""
        features = []
        features.extend(self.lane_counts)
        features.append(self.avg_speed / 100.0)  # Normalize speed
        features.extend(self.time_features)
        features.extend(self.weather_features)
        features.extend(self.environmental_features)
        features.append(self.current_phase / 4.0)  # Normalize phase
        features.append(self.phase_duration / 60.0)  # Normalize duration
        return np.array(features, dtype=np.float32)


@dataclass
class QLearningAction:
    """Action representation for Q-Learning"""
    action_type: int  # 0: reduce_green, 1: maintain, 2: increase_small, 3: increase_medium, 4: increase_large
    green_time_adjustment: int  # seconds to adjust green time
    phase_extension: bool  # whether to extend current phase
    
    def to_vector(self) -> np.ndarray:
        """Convert action to vector representation"""
        return np.array([
            self.action_type / 4.0,  # Normalize action type
            self.green_time_adjustment / 30.0,  # Normalize adjustment
            1.0 if self.phase_extension else 0.0
        ], dtype=np.float32)


class DQNNetwork(nn.Module):
    """Deep Q-Network for traffic signal optimization"""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int], activation: str = "relu"):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Build network layers
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "sigmoid":
                layers.append(nn.Sigmoid())
            layers.append(nn.Dropout(0.1))
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample random batch from buffer"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class QLearningOptimizer:
    """
    Q-Learning based traffic signal optimizer using Deep Q-Network (DQN)
    
    This optimizer learns optimal signal timing strategies through reinforcement learning,
    adapting to traffic patterns and environmental conditions in real-time.
    """
    
    def __init__(self, config: Optional[QLearningConfig] = None):
        self.config = config or get_config().q_learning
        self.logger = logging.getLogger(__name__)
        
        # Initialize neural networks
        self.q_network = DQNNetwork(
            state_size=self.config.state_size,
            action_size=self.config.action_size,
            hidden_layers=self.config.hidden_layers,
            activation=self.config.activation
        )
        
        self.target_network = DQNNetwork(
            state_size=self.config.state_size,
            action_size=self.config.action_size,
            hidden_layers=self.config.hidden_layers,
            activation=self.config.activation
        )
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Initialize optimizer
        if self.config.optimizer == "adam":
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.q_network.parameters(), lr=self.config.learning_rate)
        else:
            self.optimizer = optim.SGD(self.q_network.parameters(), lr=self.config.learning_rate)
        
        # Initialize loss function
        if self.config.loss_function == "mse":
            self.criterion = nn.MSELoss()
        elif self.config.loss_function == "huber":
            self.criterion = nn.SmoothL1Loss()
        else:
            self.criterion = nn.MSELoss()
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.config.memory_size)
        
        # Training parameters
        self.epsilon = self.config.epsilon
        self.epsilon_decay = self.config.epsilon_decay
        self.epsilon_min = self.config.epsilon_min
        self.batch_size = self.config.batch_size
        self.target_update_frequency = self.config.target_update_frequency
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.loss_history = []
        
        # Action mapping
        self.action_mapping = {
            0: {"adjustment": -10, "description": "Reduce green time"},
            1: {"adjustment": 0, "description": "Maintain current timing"},
            2: {"adjustment": 5, "description": "Small increase"},
            3: {"adjustment": 15, "description": "Medium increase"},
            4: {"adjustment": 25, "description": "Large increase"}
        }
    
    def state_to_ql_state(self, traffic_data: TrafficData, current_phase: int = 0, phase_duration: float = 0.0) -> QLearningState:
        """Convert TrafficData to QLearningState"""
        # Extract features from traffic data
        lane_counts = np.array([
            traffic_data.lane_counts.get('north_lane', 0),
            traffic_data.lane_counts.get('south_lane', 0),
            traffic_data.lane_counts.get('east_lane', 0),
            traffic_data.lane_counts.get('west_lane', 0)
        ], dtype=np.float32)
        
        avg_speed = traffic_data.avg_speed or 0.0
        
        # Time features (cyclical encoding)
        hour = traffic_data.timestamp.hour
        time_features = np.array([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * traffic_data.timestamp.weekday() / 7),
            np.cos(2 * np.pi * traffic_data.timestamp.weekday() / 7)
        ], dtype=np.float32)
        
        # Weather features (one-hot encoding)
        weather_conditions = ['clear', 'cloudy', 'rainy', 'foggy', 'stormy', 'snowy']
        weather_features = np.zeros(len(weather_conditions), dtype=np.float32)
        if traffic_data.weather_condition in weather_conditions:
            weather_features[weather_conditions.index(traffic_data.weather_condition)] = 1.0
        
        # Environmental features
        environmental_features = np.array([
            (traffic_data.temperature or 20) / 50.0,  # Normalized temperature
            (traffic_data.humidity or 50) / 100.0,    # Normalized humidity
            (traffic_data.visibility or 10) / 20.0    # Normalized visibility
        ], dtype=np.float32)
        
        return QLearningState(
            lane_counts=lane_counts,
            avg_speed=avg_speed,
            time_features=time_features,
            weather_features=weather_features,
            environmental_features=environmental_features,
            current_phase=current_phase,
            phase_duration=phase_duration
        )
    
    def select_action(self, state: QLearningState, training: bool = True) -> QLearningAction:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action
            action_type = random.randint(0, self.config.action_size - 1)
        else:
            # Greedy action
            state_vector = torch.FloatTensor(state.to_vector()).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_vector)
                action_type = q_values.argmax().item()
        
        # Map action type to actual action
        action_info = self.action_mapping[action_type]
        green_time_adjustment = action_info["adjustment"]
        phase_extension = green_time_adjustment > 0
        
        return QLearningAction(
            action_type=action_type,
            green_time_adjustment=green_time_adjustment,
            phase_extension=phase_extension
        )
    
    def calculate_reward(self, state: QLearningState, action: QLearningAction, next_state: QLearningState) -> float:
        """
        Calculate reward based on traffic conditions and optimization goals
        
        Reward function considers:
        - Wait time reduction
        - Throughput improvement
        - Queue length reduction
        - Fuel consumption reduction
        """
        reward = 0.0
        
        # Wait time component (negative reward for high wait times)
        total_vehicles = np.sum(state.lane_counts)
        wait_time_penalty = -total_vehicles * 0.1
        reward += wait_time_penalty
        
        # Throughput component (positive reward for moving vehicles)
        vehicles_processed = max(0, total_vehicles - np.sum(next_state.lane_counts))
        throughput_reward = vehicles_processed * 0.2
        reward += throughput_reward
        
        # Queue length component (negative reward for long queues)
        max_queue = np.max(state.lane_counts)
        queue_penalty = -max_queue * 0.05
        reward += queue_penalty
        
        # Action efficiency component
        if action.green_time_adjustment > 0:
            # Reward for increasing green time when there's traffic
            if total_vehicles > 10:
                efficiency_reward = action.green_time_adjustment * 0.01
                reward += efficiency_reward
        elif action.green_time_adjustment < 0:
            # Reward for reducing green time when there's little traffic
            if total_vehicles < 5:
                efficiency_reward = abs(action.green_time_adjustment) * 0.01
                reward += efficiency_reward
        
        # Stability component (penalty for frequent changes)
        if hasattr(self, 'last_action') and self.last_action.action_type != action.action_type:
            stability_penalty = -0.1
            reward += stability_penalty
        
        self.last_action = action
        
        return reward
    
    def optimize_signal_timing(self, traffic_data: TrafficData, current_timings: Dict[str, int]) -> Dict[str, int]:
        """
        Optimize signal timing using Q-Learning
        
        Args:
            traffic_data: Current traffic data
            current_timings: Current signal timings
            
        Returns:
            Optimized signal timings
        """
        # Convert to Q-Learning state
        current_phase = 0  # Assume starting from phase 0
        phase_duration = current_timings.get('north_lane', 30)  # Use north lane as reference
        
        state = self.state_to_ql_state(traffic_data, current_phase, phase_duration)
        
        # Select action
        action = self.select_action(state, training=False)
        
        # Apply action to current timings
        optimized_timings = current_timings.copy()
        
        for lane in optimized_timings:
            current_time = optimized_timings[lane]
            new_time = max(15, min(60, current_time + action.green_time_adjustment))
            optimized_timings[lane] = new_time
        
        self.logger.info(f"Q-Learning optimization: {action.action_type} -> {action.green_time_adjustment}s adjustment")
        
        return optimized_timings
    
    def train_step(self, state: QLearningState, action: QLearningAction, reward: float, next_state: QLearningState, done: bool):
        """Perform one training step"""
        # Store experience in replay buffer
        self.replay_buffer.push(state, action, reward, next_state, done)
        
        # Only train if we have enough experiences
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Prepare batch data
        states = torch.FloatTensor([s.to_vector() for s, _, _, _, _ in batch])
        actions = torch.LongTensor([a.action_type for _, a, _, _, _ in batch])
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch])
        next_states = torch.FloatTensor([s.to_vector() for _, _, _, s, _ in batch])
        dones = torch.BoolTensor([d for _, _, _, _, d in batch])
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.config.discount_factor * next_q_values * ~dones)
        
        # Calculate loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Log training progress
        self.loss_history.append(loss.item())
        if self.training_step % 100 == 0:
            avg_loss = np.mean(self.loss_history[-100:])
            self.logger.info(f"Training step {self.training_step}, Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.4f}")
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'config': self.config.__dict__,
            'loss_history': self.loss_history,
            'episode_rewards': self.episode_rewards
        }
        
        torch.save(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if not os.path.exists(filepath):
            self.logger.warning(f"Model file {filepath} not found")
            return
        
        model_data = torch.load(filepath, map_location='cpu')
        
        self.q_network.load_state_dict(model_data['q_network_state_dict'])
        self.target_network.load_state_dict(model_data['target_network_state_dict'])
        self.optimizer.load_state_dict(model_data['optimizer_state_dict'])
        self.epsilon = model_data.get('epsilon', self.epsilon)
        self.training_step = model_data.get('training_step', 0)
        self.loss_history = model_data.get('loss_history', [])
        self.episode_rewards = model_data.get('episode_rewards', [])
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            "training_step": self.training_step,
            "epsilon": self.epsilon,
            "replay_buffer_size": len(self.replay_buffer),
            "avg_loss": np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            "total_episodes": len(self.episode_rewards),
            "avg_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0
        }
    
    def reset_training(self):
        """Reset training state"""
        self.training_step = 0
        self.epsilon = self.config.epsilon
        self.loss_history = []
        self.episode_rewards = []
        self.replay_buffer = ReplayBuffer(self.config.memory_size)
        self.logger.info("Training state reset")



