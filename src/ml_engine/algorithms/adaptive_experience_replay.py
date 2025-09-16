"""
Adaptive Experience Replay Buffer with Advanced Sampling Strategies
Implements prioritized experience replay with adaptive decay and curriculum learning
"""

import numpy as np
import random
import logging
from typing import List, Tuple, Optional, Dict, Any, Union
from collections import deque
import threading
import time
from dataclasses import dataclass
import heapq
import math

from algorithms.advanced_q_learning_agent import MultiDimensionalState, SophisticatedAction


@dataclass
class Experience:
    """Single experience tuple"""
    state: MultiDimensionalState
    action: SophisticatedAction
    reward: float
    next_state: MultiDimensionalState
    done: bool
    timestamp: float
    td_error: float = 0.0
    priority: float = 1.0
    importance_weight: float = 1.0
    
    def __lt__(self, other):
        """For priority queue ordering"""
        return self.priority < other.priority


class AdaptiveExperienceReplay:
    """
    Advanced experience replay buffer with adaptive sampling and curriculum learning
    
    Features:
    - Prioritized experience replay
    - Curriculum learning
    - Adaptive sampling strategies
    - Experience clustering
    - Dynamic buffer sizing
    - Multi-objective sampling
    """
    
    def __init__(self, capacity: int = 100000, alpha: float = 0.6, beta: float = 0.4,
                 beta_increment: float = 0.001, epsilon: float = 1e-6):
        self.capacity = capacity
        self.alpha = alpha  # Prioritization exponent
        self.beta = beta    # Importance sampling exponent
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.logger = logging.getLogger(__name__)
        
        # Experience storage
        self.experiences = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
        # Adaptive parameters
        self.adaptive_alpha = alpha
        self.adaptive_beta = beta
        self.curriculum_threshold = 0.1
        self.cluster_centers = []
        self.experience_clusters = defaultdict(list)
        
        # Sampling strategies
        self.sampling_strategies = ['prioritized', 'uniform', 'curriculum', 'balanced']
        self.current_strategy = 'prioritized'
        self.strategy_weights = {
            'prioritized': 0.4,
            'uniform': 0.2,
            'curriculum': 0.2,
            'balanced': 0.2
        }
        
        # Performance tracking
        self.sampling_stats = {
            'total_samples': 0,
            'strategy_usage': {strategy: 0 for strategy in self.sampling_strategies},
            'avg_priority': 0.0,
            'avg_importance_weight': 0.0
        }
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Experience analysis
        self.reward_history = deque(maxlen=1000)
        self.difficulty_scores = deque(maxlen=1000)
        
        self.logger.info(f"Adaptive experience replay initialized with capacity {capacity}")
    
    def add_experience(self, state: MultiDimensionalState, action: SophisticatedAction,
                      reward: float, next_state: MultiDimensionalState, done: bool,
                      td_error: float = None) -> None:
        """Add experience to replay buffer"""
        with self.lock:
            # Calculate priority
            if td_error is not None:
                priority = (abs(td_error) + self.epsilon) ** self.adaptive_alpha
            else:
                priority = 1.0
            
            # Create experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                timestamp=time.time(),
                td_error=td_error or 0.0,
                priority=priority
            )
            
            # Add to buffer
            if self.size < self.capacity:
                self.experiences.append(experience)
                self.priorities[self.size] = priority
                self.size += 1
            else:
                # Replace oldest experience
                self.experiences[self.position] = experience
                self.priorities[self.position] = priority
                self.position = (self.position + 1) % self.capacity
            
            # Update statistics
            self.reward_history.append(reward)
            self._update_difficulty_score(experience)
            
            # Update adaptive parameters
            self._update_adaptive_parameters()
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample batch of experiences using adaptive strategy
        
        Returns:
            experiences: List of sampled experiences
            weights: Importance sampling weights
            indices: Indices of sampled experiences
        """
        with self.lock:
            if self.size < batch_size:
                return [], np.array([]), np.array([])
            
            # Select sampling strategy
            strategy = self._select_sampling_strategy()
            
            # Sample based on strategy
            if strategy == 'prioritized':
                experiences, weights, indices = self._prioritized_sampling(batch_size)
            elif strategy == 'uniform':
                experiences, weights, indices = self._uniform_sampling(batch_size)
            elif strategy == 'curriculum':
                experiences, weights, indices = self._curriculum_sampling(batch_size)
            elif strategy == 'balanced':
                experiences, weights, indices = self._balanced_sampling(batch_size)
            else:
                experiences, weights, indices = self._prioritized_sampling(batch_size)
            
            # Update statistics
            self.sampling_stats['total_samples'] += len(experiences)
            self.sampling_stats['strategy_usage'][strategy] += 1
            
            if experiences:
                self.sampling_stats['avg_priority'] = np.mean([exp.priority for exp in experiences])
                self.sampling_stats['avg_importance_weight'] = np.mean(weights)
            
            return experiences, weights, indices
    
    def _select_sampling_strategy(self) -> str:
        """Select sampling strategy based on current performance"""
        # Simple strategy selection based on recent performance
        if len(self.reward_history) < 100:
            return 'prioritized'
        
        recent_rewards = list(self.reward_history)[-100:]
        reward_variance = np.var(recent_rewards)
        
        # High variance suggests need for curriculum learning
        if reward_variance > 0.5:
            return 'curriculum'
        
        # Low variance suggests need for exploration
        if reward_variance < 0.1:
            return 'uniform'
        
        # Balanced approach
        return 'balanced'
    
    def _prioritized_sampling(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Prioritized experience replay sampling"""
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size] ** self.adaptive_alpha
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.adaptive_beta)
        weights = weights / weights.max()  # Normalize
        
        # Get experiences
        experiences = [self.experiences[i] for i in indices]
        
        return experiences, weights, indices
    
    def _uniform_sampling(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Uniform random sampling"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        experiences = [self.experiences[i] for i in indices]
        weights = np.ones(batch_size)
        
        return experiences, weights, indices
    
    def _curriculum_sampling(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Curriculum learning sampling - easier experiences first"""
        if not self.difficulty_scores:
            return self._uniform_sampling(batch_size)
        
        # Sort experiences by difficulty (easier first)
        difficulty_scores = np.array(self.difficulty_scores)
        sorted_indices = np.argsort(difficulty_scores)
        
        # Sample from easier experiences
        easy_threshold = int(self.size * self.curriculum_threshold)
        easy_indices = sorted_indices[:easy_threshold]
        
        if len(easy_indices) >= batch_size:
            selected_indices = np.random.choice(easy_indices, batch_size, replace=False)
        else:
            # Mix easy and medium difficulty
            medium_indices = sorted_indices[easy_threshold:int(self.size * 0.5)]
            all_indices = np.concatenate([easy_indices, medium_indices])
            selected_indices = np.random.choice(all_indices, batch_size, replace=True)
        
        experiences = [self.experiences[i] for i in selected_indices]
        weights = np.ones(len(experiences))
        
        return experiences, weights, selected_indices
    
    def _balanced_sampling(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Balanced sampling across different experience types"""
        # Categorize experiences by reward
        positive_experiences = []
        negative_experiences = []
        neutral_experiences = []
        
        for i, exp in enumerate(self.experiences[:self.size]):
            if exp.reward > 0.1:
                positive_experiences.append(i)
            elif exp.reward < -0.1:
                negative_experiences.append(i)
            else:
                neutral_experiences.append(i)
        
        # Sample proportionally from each category
        total_positive = len(positive_experiences)
        total_negative = len(negative_experiences)
        total_neutral = len(neutral_experiences)
        total_experiences = total_positive + total_negative + total_neutral
        
        if total_experiences == 0:
            return self._uniform_sampling(batch_size)
        
        # Calculate sampling proportions
        pos_ratio = total_positive / total_experiences
        neg_ratio = total_negative / total_experiences
        neu_ratio = total_neutral / total_experiences
        
        pos_samples = int(batch_size * pos_ratio)
        neg_samples = int(batch_size * neg_ratio)
        neu_samples = batch_size - pos_samples - neg_samples
        
        # Sample from each category
        selected_indices = []
        
        if pos_samples > 0 and positive_experiences:
            pos_indices = np.random.choice(positive_experiences, 
                                         min(pos_samples, len(positive_experiences)), 
                                         replace=False)
            selected_indices.extend(pos_indices)
        
        if neg_samples > 0 and negative_experiences:
            neg_indices = np.random.choice(negative_experiences, 
                                         min(neg_samples, len(negative_experiences)), 
                                         replace=False)
            selected_indices.extend(neg_indices)
        
        if neu_samples > 0 and neutral_experiences:
            neu_indices = np.random.choice(neutral_experiences, 
                                         min(neu_samples, len(neutral_experiences)), 
                                         replace=False)
            selected_indices.extend(neu_indices)
        
        # Fill remaining slots with random sampling
        while len(selected_indices) < batch_size:
            random_idx = np.random.randint(0, self.size)
            if random_idx not in selected_indices:
                selected_indices.append(random_idx)
        
        experiences = [self.experiences[i] for i in selected_indices]
        weights = np.ones(len(experiences))
        
        return experiences, weights, np.array(selected_indices)
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities for sampled experiences"""
        with self.lock:
            for idx, td_error in zip(indices, td_errors):
                if idx < self.size:
                    priority = (abs(td_error) + self.epsilon) ** self.adaptive_alpha
                    self.priorities[idx] = priority
                    self.experiences[idx].priority = priority
                    self.experiences[idx].td_error = td_error
    
    def _update_difficulty_score(self, experience: Experience):
        """Update difficulty score for experience"""
        # Calculate difficulty based on reward magnitude and state complexity
        reward_magnitude = abs(experience.reward)
        state_complexity = self._calculate_state_complexity(experience.state)
        
        difficulty = reward_magnitude * 0.7 + state_complexity * 0.3
        self.difficulty_scores.append(difficulty)
    
    def _calculate_state_complexity(self, state: MultiDimensionalState) -> float:
        """Calculate complexity of state"""
        # Simple complexity measure based on congestion and traffic density
        congestion = state.congestion_trend
        traffic_density = np.sum(state.lane_counts) / 100.0  # Normalize
        
        complexity = congestion * 0.6 + traffic_density * 0.4
        return min(complexity, 1.0)
    
    def _update_adaptive_parameters(self):
        """Update adaptive parameters based on recent performance"""
        if len(self.reward_history) < 50:
            return
        
        recent_rewards = list(self.reward_history)[-50:]
        reward_std = np.std(recent_rewards)
        
        # Adjust alpha based on reward variance
        if reward_std > 0.5:  # High variance - increase prioritization
            self.adaptive_alpha = min(0.8, self.adaptive_alpha + 0.01)
        else:  # Low variance - decrease prioritization
            self.adaptive_alpha = max(0.2, self.adaptive_alpha - 0.01)
        
        # Adjust beta based on learning progress
        self.adaptive_beta = min(1.0, self.adaptive_beta + self.beta_increment)
        
        # Adjust curriculum threshold
        if len(self.difficulty_scores) > 100:
            recent_difficulty = list(self.difficulty_scores)[-100:]
            avg_difficulty = np.mean(recent_difficulty)
            
            if avg_difficulty > 0.7:  # High difficulty - increase curriculum threshold
                self.curriculum_threshold = min(0.3, self.curriculum_threshold + 0.01)
            else:  # Low difficulty - decrease curriculum threshold
                self.curriculum_threshold = max(0.05, self.curriculum_threshold - 0.01)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get replay buffer statistics"""
        with self.lock:
            stats = {
                'size': self.size,
                'capacity': self.capacity,
                'utilization': self.size / self.capacity,
                'adaptive_alpha': self.adaptive_alpha,
                'adaptive_beta': self.adaptive_beta,
                'curriculum_threshold': self.curriculum_threshold,
                'current_strategy': self.current_strategy,
                'sampling_stats': self.sampling_stats.copy()
            }
            
            if self.size > 0:
                priorities = self.priorities[:self.size]
                stats['priority_stats'] = {
                    'mean': np.mean(priorities),
                    'std': np.std(priorities),
                    'min': np.min(priorities),
                    'max': np.max(priorities)
                }
            
            if self.reward_history:
                recent_rewards = list(self.reward_history)[-100:]
                stats['reward_stats'] = {
                    'mean': np.mean(recent_rewards),
                    'std': np.std(recent_rewards),
                    'min': np.min(recent_rewards),
                    'max': np.max(recent_rewards)
                }
            
            if self.difficulty_scores:
                recent_difficulty = list(self.difficulty_scores)[-100:]
                stats['difficulty_stats'] = {
                    'mean': np.mean(recent_difficulty),
                    'std': np.std(recent_difficulty),
                    'min': np.min(recent_difficulty),
                    'max': np.max(recent_difficulty)
                }
            
            return stats
    
    def clear(self):
        """Clear replay buffer"""
        with self.lock:
            self.experiences = []
            self.priorities = np.zeros(self.capacity, dtype=np.float32)
            self.position = 0
            self.size = 0
            self.reward_history.clear()
            self.difficulty_scores.clear()
            self.sampling_stats = {
                'total_samples': 0,
                'strategy_usage': {strategy: 0 for strategy in self.sampling_strategies},
                'avg_priority': 0.0,
                'avg_importance_weight': 0.0
            }
    
    def __len__(self):
        """Return current buffer size"""
        return self.size


class ExperienceReplayAnalyzer:
    """Analyzes experience replay buffer for insights and optimization"""
    
    def __init__(self, replay_buffer: AdaptiveExperienceReplay):
        self.replay_buffer = replay_buffer
        self.logger = logging.getLogger(__name__)
    
    def analyze_experience_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of experiences in buffer"""
        if self.replay_buffer.size == 0:
            return {}
        
        experiences = self.replay_buffer.experiences[:self.replay_buffer.size]
        
        # Analyze reward distribution
        rewards = [exp.reward for exp in experiences]
        reward_analysis = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'min': np.min(rewards),
            'max': np.max(rewards),
            'positive_ratio': sum(1 for r in rewards if r > 0) / len(rewards),
            'negative_ratio': sum(1 for r in rewards if r < 0) / len(rewards)
        }
        
        # Analyze action distribution
        actions = [exp.action.action_type for exp in experiences]
        action_counts = np.bincount(actions, minlength=8)
        action_analysis = {
            'action_distribution': action_counts.tolist(),
            'most_common_action': int(np.argmax(action_counts)),
            'action_diversity': len(np.nonzero(action_counts)[0]) / len(action_counts)
        }
        
        # Analyze state complexity
        complexities = [self.replay_buffer._calculate_state_complexity(exp.state) for exp in experiences]
        complexity_analysis = {
            'mean_complexity': np.mean(complexities),
            'std_complexity': np.std(complexities),
            'high_complexity_ratio': sum(1 for c in complexities if c > 0.7) / len(complexities)
        }
        
        return {
            'reward_analysis': reward_analysis,
            'action_analysis': action_analysis,
            'complexity_analysis': complexity_analysis,
            'total_experiences': len(experiences)
        }
    
    def suggest_optimizations(self) -> List[str]:
        """Suggest optimizations for the replay buffer"""
        suggestions = []
        
        stats = self.replay_buffer.get_statistics()
        
        # Check utilization
        if stats['utilization'] < 0.5:
            suggestions.append("Buffer utilization is low. Consider reducing buffer size or increasing experience collection.")
        
        # Check strategy usage
        strategy_usage = stats['sampling_stats']['strategy_usage']
        total_samples = stats['sampling_stats']['total_samples']
        
        if total_samples > 0:
            for strategy, count in strategy_usage.items():
                usage_ratio = count / total_samples
                if usage_ratio > 0.8:
                    suggestions.append(f"Strategy '{strategy}' is overused ({usage_ratio:.2%}). Consider rebalancing strategy weights.")
        
        # Check priority distribution
        if 'priority_stats' in stats:
            priority_std = stats['priority_stats']['std']
            if priority_std < 0.1:
                suggestions.append("Priority distribution is too narrow. Consider increasing alpha for more prioritization.")
            elif priority_std > 1.0:
                suggestions.append("Priority distribution is too wide. Consider decreasing alpha for more uniform sampling.")
        
        # Check reward distribution
        if 'reward_stats' in stats:
            reward_std = stats['reward_stats']['std']
            if reward_std < 0.1:
                suggestions.append("Reward variance is low. Consider increasing exploration or reward shaping.")
        
        return suggestions
    
    def generate_curriculum_schedule(self, num_episodes: int) -> List[Dict[str, Any]]:
        """Generate curriculum learning schedule"""
        schedule = []
        
        # Phase 1: Easy experiences (first 30% of episodes)
        phase1_episodes = int(num_episodes * 0.3)
        for episode in range(phase1_episodes):
            schedule.append({
                'episode': episode,
                'phase': 'easy',
                'curriculum_threshold': 0.1,
                'sampling_strategy': 'curriculum',
                'description': 'Focus on easy experiences'
            })
        
        # Phase 2: Mixed experiences (next 40% of episodes)
        phase2_episodes = int(num_episodes * 0.4)
        for episode in range(phase1_episodes, phase1_episodes + phase2_episodes):
            schedule.append({
                'episode': episode,
                'phase': 'mixed',
                'curriculum_threshold': 0.3,
                'sampling_strategy': 'balanced',
                'description': 'Mix of easy and medium experiences'
            })
        
        # Phase 3: All experiences (final 30% of episodes)
        phase3_episodes = num_episodes - phase1_episodes - phase2_episodes
        for episode in range(phase1_episodes + phase2_episodes, num_episodes):
            schedule.append({
                'episode': episode,
                'phase': 'all',
                'curriculum_threshold': 0.5,
                'sampling_strategy': 'prioritized',
                'description': 'All experiences with prioritization'
            })
        
        return schedule
