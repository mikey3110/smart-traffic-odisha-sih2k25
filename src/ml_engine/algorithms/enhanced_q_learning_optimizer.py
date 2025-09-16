"""
Enhanced Q-Learning based Traffic Signal Optimization
Advanced implementation with real-time data integration, experience replay, and adaptive learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import logging
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
import os
import time
from datetime import datetime, timedelta
import threading
from queue import Queue, Empty

from config.ml_config import QLearningConfig, get_config
from data.enhanced_data_integration import data_service, TrafficData
from prediction.traffic_predictor import PredictionResult


@dataclass
class QLearningState:
    """Enhanced state representation for Q-Learning"""
    # Traffic data
    lane_counts: np.ndarray
    avg_speed: float
    queue_lengths: np.ndarray
    waiting_times: np.ndarray
    
    # Temporal features
    time_of_day: float  # 0-1 normalized hour
    day_of_week: float  # 0-1 normalized day
    is_weekend: bool
    
    # Environmental features
    weather_condition: int  # Encoded weather
    temperature: float
    visibility: float
    
    # Signal state
    current_phase: int
    phase_duration: float
    cycle_progress: float  # 0-1 progress through current cycle
    
    # Historical context
    recent_flow_rates: np.ndarray
    congestion_level: float
    
    def to_vector(self) -> np.ndarray:
        """Convert state to normalized feature vector"""
        features = []
        
        # Traffic features (normalized)
        features.extend(self.lane_counts / 50.0)  # Normalize by max expected count
        features.append(self.avg_speed / 100.0)  # Normalize speed
        features.extend(self.queue_lengths / 30.0)  # Normalize queue lengths
        features.extend(self.waiting_times / 60.0)  # Normalize waiting times
        
        # Temporal features
        features.append(self.time_of_day)
        features.append(self.day_of_week)
        features.append(1.0 if self.is_weekend else 0.0)
        
        # Environmental features
        weather_one_hot = [0.0] * 6  # 6 weather conditions
        weather_one_hot[self.weather_condition] = 1.0
        features.extend(weather_one_hot)
        features.append(self.temperature / 50.0)  # Normalize temperature
        features.append(self.visibility / 20.0)  # Normalize visibility
        
        # Signal state features
        features.append(self.current_phase / 4.0)  # Normalize phase
        features.append(self.phase_duration / 120.0)  # Normalize duration
        features.append(self.cycle_progress)
        
        # Historical context
        features.extend(self.recent_flow_rates / 1000.0)  # Normalize flow rates
        features.append(self.congestion_level)
        
        return np.array(features, dtype=np.float32)


@dataclass
class QLearningAction:
    """Enhanced action representation for Q-Learning"""
    action_type: int  # 0-4: reduce, maintain, small_increase, medium_increase, large_increase
    green_time_adjustment: int  # seconds to adjust
    phase_extension: bool  # whether to extend current phase
    cycle_adjustment: int  # seconds to adjust total cycle time
    priority_boost: bool  # whether to give priority to congested lanes
    
    def to_vector(self) -> np.ndarray:
        """Convert action to vector representation"""
        return np.array([
            self.action_type / 4.0,
            self.green_time_adjustment / 30.0,
            1.0 if self.phase_extension else 0.0,
            self.cycle_adjustment / 30.0,
            1.0 if self.priority_boost else 0.0
        ], dtype=np.float32)


class DQNNetwork(nn.Module):
    """Enhanced Deep Q-Network with attention mechanism"""
    
    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int], 
                 dropout_rate: float = 0.1, use_attention: bool = True):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.use_attention = use_attention
        
        # Build main network
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_size = hidden_size
        
        self.main_network = nn.Sequential(*layers)
        
        # Attention mechanism for important features
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_layers[-1],
                num_heads=4,
                dropout=dropout_rate,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_layers[-1])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], action_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Main network
        features = self.main_network(x)
        
        # Apply attention if enabled
        if self.use_attention and features.dim() == 2:
            # Reshape for attention: (batch, seq_len, embed_dim)
            features = features.unsqueeze(1)
            attended, _ = self.attention(features, features, features)
            features = self.attention_norm(attended.squeeze(1))
        
        # Output layer
        q_values = self.output_layer(features)
        return q_values


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer for better learning"""
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
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
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray]:
        """Sample batch with importance sampling weights"""
        if len(self.buffer) == 0:
            return [], np.array([])
        
        # Calculate sampling probabilities
        probs = self.priorities[:len(self.buffer)] ** self.alpha
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize weights
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        return experiences, weights
    
    def __len__(self):
        return len(self.buffer)


class EnhancedQLearningOptimizer:
    """
    Enhanced Q-Learning optimizer with advanced features:
    - Real-time data integration
    - Prioritized experience replay
    - Double DQN for stability
    - Adaptive learning rate
    - Multi-objective reward function
    - Online learning capabilities
    """
    
    def __init__(self, config: Optional[QLearningConfig] = None):
        self.config = config or get_config().q_learning
        self.logger = logging.getLogger(__name__)
        
        # State and action dimensions
        self.state_size = 25  # Enhanced state representation
        self.action_size = 5
        
        # Initialize networks
        self.q_network = DQNNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_layers=self.config.hidden_layers,
            dropout_rate=0.1,
            use_attention=True
        )
        
        self.target_network = DQNNetwork(
            state_size=self.state_size,
            action_size=self.action_size,
            hidden_layers=self.config.hidden_layers,
            dropout_rate=0.1,
            use_attention=True
        )
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer with adaptive learning rate
        self.optimizer = optim.AdamW(
            self.q_network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=100,
            verbose=True
        )
        
        # Loss function
        self.criterion = nn.SmoothL1Loss(reduction='none')
        
        # Experience replay with prioritization
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.memory_size,
            alpha=0.6,
            beta=0.4
        )
        
        # Training parameters
        self.epsilon = self.config.epsilon
        self.epsilon_decay = self.config.epsilon_decay
        self.epsilon_min = self.config.epsilon_min
        self.batch_size = self.config.batch_size
        self.target_update_frequency = self.config.target_update_frequency
        
        # Training state
        self.training_step = 0
        self.episode_rewards = []
        self.loss_history = []
        self.performance_history = []
        
        # Real-time learning
        self.online_learning = True
        self.learning_thread = None
        self.learning_queue = Queue(maxsize=1000)
        
        # Performance tracking
        self.optimization_history = []
        self.reward_components = {
            'wait_time': [],
            'throughput': [],
            'queue_length': [],
            'fuel_consumption': [],
            'stability': []
        }
        
        # Action mapping with enhanced actions
        self.action_mapping = {
            0: {"adjustment": -15, "cycle_adjustment": -10, "description": "Reduce green time"},
            1: {"adjustment": 0, "cycle_adjustment": 0, "description": "Maintain current timing"},
            2: {"adjustment": 5, "cycle_adjustment": 5, "description": "Small increase"},
            3: {"adjustment": 15, "cycle_adjustment": 10, "description": "Medium increase"},
            4: {"adjustment": 25, "cycle_adjustment": 20, "description": "Large increase"}
        }
        
        self.logger.info("Enhanced Q-Learning optimizer initialized")
    
    def start_online_learning(self):
        """Start online learning thread"""
        if self.learning_thread and self.learning_thread.is_alive():
            return
        
        self.learning_thread = threading.Thread(target=self._online_learning_loop, daemon=True)
        self.learning_thread.start()
        self.logger.info("Online learning started")
    
    def stop_online_learning(self):
        """Stop online learning thread"""
        self.online_learning = False
        if self.learning_thread:
            self.learning_thread.join(timeout=5)
        self.logger.info("Online learning stopped")
    
    def _online_learning_loop(self):
        """Online learning loop for continuous improvement"""
        while self.online_learning:
            try:
                # Process learning queue
                if not self.learning_queue.empty():
                    experience = self.learning_queue.get_nowait()
                    self._process_experience(experience)
                
                # Periodic training
                if len(self.replay_buffer) >= self.batch_size and self.training_step % 10 == 0:
                    self._train_step()
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
                
            except Exception as e:
                self.logger.error(f"Error in online learning loop: {e}")
                time.sleep(1)
    
    def _process_experience(self, experience: Tuple):
        """Process a single experience for learning"""
        state, action, reward, next_state, done = experience
        
        # Calculate TD error for prioritization
        with torch.no_grad():
            current_q = self.q_network(torch.FloatTensor(state.to_vector()).unsqueeze(0))
            current_q_value = current_q[0][action.action_type].item()
            
            next_q = self.target_network(torch.FloatTensor(next_state.to_vector()).unsqueeze(0))
            next_q_value = next_q.max().item()
            target_q = reward + (self.config.discount_factor * next_q_value * (1 - done))
            
            td_error = abs(target_q - current_q_value)
        
        # Add to replay buffer with priority
        self.replay_buffer.push(state, action, reward, next_state, done, td_error)
    
    def state_to_ql_state(self, traffic_data: TrafficData, current_phase: int = 0, 
                         phase_duration: float = 0.0, historical_data: List[TrafficData] = None) -> QLearningState:
        """Convert TrafficData to enhanced QLearningState"""
        
        # Extract lane counts
        lane_counts = np.array([
            traffic_data.lane_counts.get('north_lane', 0),
            traffic_data.lane_counts.get('south_lane', 0),
            traffic_data.lane_counts.get('east_lane', 0),
            traffic_data.lane_counts.get('west_lane', 0)
        ], dtype=np.float32)
        
        # Estimate queue lengths (simplified model)
        queue_lengths = lane_counts * 0.8  # Assume 80% of vehicles are queued
        
        # Estimate waiting times (simplified model)
        waiting_times = queue_lengths * 2.0  # Assume 2 seconds per queued vehicle
        
        # Temporal features
        hour = traffic_data.timestamp.hour
        day_of_week = traffic_data.timestamp.weekday()
        time_of_day = hour / 24.0
        day_of_week_norm = day_of_week / 7.0
        is_weekend = day_of_week >= 5
        
        # Weather encoding
        weather_conditions = ['clear', 'cloudy', 'rainy', 'foggy', 'stormy', 'snowy']
        weather_condition = 0  # Default to clear
        if traffic_data.weather_condition in weather_conditions:
            weather_condition = weather_conditions.index(traffic_data.weather_condition)
        
        # Environmental features
        temperature = getattr(traffic_data, 'temperature', 20.0)
        visibility = getattr(traffic_data, 'visibility', 10.0)
        
        # Signal state
        cycle_progress = phase_duration / 120.0  # Assume 2-minute cycle
        
        # Historical context
        recent_flow_rates = np.array([0.0, 0.0, 0.0, 0.0])  # Placeholder
        if historical_data and len(historical_data) >= 2:
            # Calculate flow rates from recent data
            for i, lane in enumerate(['north_lane', 'south_lane', 'east_lane', 'west_lane']):
                if len(historical_data) >= 2:
                    recent_count = historical_data[-1].lane_counts.get(lane, 0)
                    prev_count = historical_data[-2].lane_counts.get(lane, 0)
                    time_diff = (historical_data[-1].timestamp - historical_data[-2].timestamp).total_seconds() / 3600  # hours
                    if time_diff > 0:
                        recent_flow_rates[i] = (recent_count - prev_count) / time_diff
        
        # Congestion level (0-1)
        total_vehicles = np.sum(lane_counts)
        congestion_level = min(1.0, total_vehicles / 100.0)  # Normalize by max expected vehicles
        
        return QLearningState(
            lane_counts=lane_counts,
            avg_speed=traffic_data.avg_speed or 0.0,
            queue_lengths=queue_lengths,
            waiting_times=waiting_times,
            time_of_day=time_of_day,
            day_of_week=day_of_week_norm,
            is_weekend=is_weekend,
            weather_condition=weather_condition,
            temperature=temperature,
            visibility=visibility,
            current_phase=current_phase,
            phase_duration=phase_duration,
            cycle_progress=cycle_progress,
            recent_flow_rates=recent_flow_rates,
            congestion_level=congestion_level
        )
    
    def select_action(self, state: QLearningState, training: bool = True) -> QLearningAction:
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
        
        # Map action type to actual action
        action_info = self.action_mapping[action_type]
        
        # Add exploration bonus for congested states
        if state.congestion_level > 0.7 and training:
            # Boost action for high congestion
            action_type = min(4, action_type + 1)
            action_info = self.action_mapping[action_type]
        
        return QLearningAction(
            action_type=action_type,
            green_time_adjustment=action_info["adjustment"],
            phase_extension=action_info["adjustment"] > 0,
            cycle_adjustment=action_info["cycle_adjustment"],
            priority_boost=state.congestion_level > 0.6
        )
    
    def calculate_reward(self, state: QLearningState, action: QLearningAction, 
                        next_state: QLearningState, performance_metrics: Dict[str, float] = None) -> float:
        """
        Calculate multi-objective reward function
        
        Reward components:
        - Wait time reduction (primary)
        - Throughput improvement
        - Queue length reduction
        - Fuel consumption reduction
        - Signal stability
        """
        reward = 0.0
        
        # Wait time component (most important)
        wait_time_reduction = np.sum(state.waiting_times) - np.sum(next_state.waiting_times)
        wait_time_reward = wait_time_reduction * 0.1
        reward += wait_time_reward
        self.reward_components['wait_time'].append(wait_time_reward)
        
        # Throughput component
        vehicles_processed = np.sum(state.lane_counts) - np.sum(next_state.lane_counts)
        throughput_reward = vehicles_processed * 0.05
        reward += throughput_reward
        self.reward_components['throughput'].append(throughput_reward)
        
        # Queue length component
        queue_reduction = np.sum(state.queue_lengths) - np.sum(next_state.queue_lengths)
        queue_reward = queue_reduction * 0.03
        reward += queue_reward
        self.reward_components['queue_length'].append(queue_reward)
        
        # Fuel consumption component (estimated)
        idling_vehicles = np.sum(state.queue_lengths)
        fuel_penalty = -idling_vehicles * 0.01
        reward += fuel_penalty
        self.reward_components['fuel_consumption'].append(fuel_penalty)
        
        # Stability component (penalty for frequent changes)
        if hasattr(self, 'last_action') and self.last_action.action_type != action.action_type:
            stability_penalty = -0.05
            reward += stability_penalty
        self.reward_components['stability'].append(stability_penalty if 'stability_penalty' in locals() else 0.0)
        
        # Action efficiency bonus
        if action.green_time_adjustment > 0 and state.congestion_level > 0.5:
            efficiency_bonus = action.green_time_adjustment * 0.02
            reward += efficiency_bonus
        elif action.green_time_adjustment < 0 and state.congestion_level < 0.3:
            efficiency_bonus = abs(action.green_time_adjustment) * 0.01
            reward += efficiency_bonus
        
        # Performance metrics bonus (if available)
        if performance_metrics:
            if 'wait_time_improvement' in performance_metrics:
                reward += performance_metrics['wait_time_improvement'] * 0.1
            if 'throughput_improvement' in performance_metrics:
                reward += performance_metrics['throughput_improvement'] * 0.05
        
        self.last_action = action
        
        return reward
    
    def optimize_signal_timing(self, traffic_data: TrafficData, current_timings: Dict[str, int],
                             historical_data: List[TrafficData] = None) -> Dict[str, int]:
        """
        Optimize signal timing using enhanced Q-Learning
        
        Args:
            traffic_data: Current traffic data
            current_timings: Current signal timings
            historical_data: Historical traffic data for context
            
        Returns:
            Optimized signal timings
        """
        # Convert to Q-Learning state
        current_phase = 0  # Assume starting from phase 0
        phase_duration = current_timings.get('north_lane', 30)
        
        state = self.state_to_ql_state(traffic_data, current_phase, phase_duration, historical_data)
        
        # Select action
        action = self.select_action(state, training=False)
        
        # Apply action to current timings
        optimized_timings = current_timings.copy()
        
        # Calculate cycle adjustment
        cycle_adjustment = action.cycle_adjustment
        total_cycle_time = sum(optimized_timings.values())
        new_total_cycle = max(60, min(180, total_cycle_time + cycle_adjustment))
        
        # Scale all timings proportionally
        if total_cycle_time > 0:
            scale_factor = new_total_cycle / total_cycle_time
            for lane in optimized_timings:
                optimized_timings[lane] = max(15, min(90, int(optimized_timings[lane] * scale_factor)))
        
        # Apply individual lane adjustments
        for lane in optimized_timings:
            current_time = optimized_timings[lane]
            new_time = max(15, min(90, current_time + action.green_time_adjustment))
            optimized_timings[lane] = new_time
        
        # Log optimization decision
        self.logger.info(
            f"Q-Learning optimization: Action {action.action_type} -> "
            f"{action.green_time_adjustment}s adjustment, "
            f"Cycle: {cycle_adjustment}s, Priority boost: {action.priority_boost}"
        )
        
        # Store optimization for learning
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'intersection_id': traffic_data.intersection_id,
            'state': state,
            'action': action,
            'optimized_timings': optimized_timings.copy()
        })
        
        return optimized_timings
    
    def _train_step(self):
        """Perform one training step with prioritized experience replay"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch with importance sampling
        batch, weights = self.replay_buffer.sample(self.batch_size)
        
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
        
        # Double DQN: use main network for action selection, target network for evaluation
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.config.discount_factor * next_q_values * ~dones.unsqueeze(1))
        
        # Calculate loss with importance sampling weights
        loss = self.criterion(current_q_values.squeeze(), target_q_values.squeeze())
        weighted_loss = (loss * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
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
    
    def add_experience(self, state: QLearningState, action: QLearningAction, 
                      reward: float, next_state: QLearningState, done: bool):
        """Add experience for online learning"""
        if self.online_learning:
            self.learning_queue.put((state, action, reward, next_state, done))
    
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
            'config': self.config.__dict__,
            'loss_history': self.loss_history,
            'episode_rewards': self.episode_rewards,
            'performance_history': self.performance_history,
            'reward_components': self.reward_components
        }
        
        torch.save(model_data, filepath)
        self.logger.info(f"Enhanced Q-Learning model saved to {filepath}")
    
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
        
        self.logger.info(f"Enhanced Q-Learning model loaded from {filepath}")
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        return {
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
    
    def reset_training(self):
        """Reset training state"""
        self.training_step = 0
        self.epsilon = self.config.epsilon
        self.loss_history = []
        self.episode_rewards = []
        self.performance_history = []
        self.optimization_history = []
        self.reward_components = {component: [] for component in self.reward_components}
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config.memory_size,
            alpha=0.6,
            beta=0.4
        )
        self.logger.info("Enhanced Q-Learning training state reset")


