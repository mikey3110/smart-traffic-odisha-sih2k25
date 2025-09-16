"""
Advanced Multi-Objective Reward Function for Q-Learning
Combines wait time reduction, throughput increase, fuel efficiency, and pedestrian safety
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from algorithms.advanced_q_learning_agent import MultiDimensionalState, SophisticatedAction


@dataclass
class RewardComponents:
    """Individual components of the reward function"""
    wait_time_reduction: float = 0.0
    throughput_increase: float = 0.0
    fuel_efficiency: float = 0.0
    pedestrian_safety: float = 0.0
    coordination_bonus: float = 0.0
    stability_penalty: float = 0.0
    emergency_priority: float = 0.0
    environmental_adaptation: float = 0.0
    total_reward: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'wait_time_reduction': self.wait_time_reduction,
            'throughput_increase': self.throughput_increase,
            'fuel_efficiency': self.fuel_efficiency,
            'pedestrian_safety': self.pedestrian_safety,
            'coordination_bonus': self.coordination_bonus,
            'stability_penalty': self.stability_penalty,
            'emergency_priority': self.emergency_priority,
            'environmental_adaptation': self.environmental_adaptation,
            'total_reward': self.total_reward
        }


class AdvancedRewardFunction:
    """
    Advanced multi-objective reward function for traffic signal optimization
    
    Combines multiple objectives:
    1. Wait time reduction (primary)
    2. Throughput increase
    3. Fuel efficiency
    4. Pedestrian safety
    5. Coordination with adjacent intersections
    6. Signal stability
    7. Emergency vehicle priority
    8. Environmental adaptation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Reward weights (configurable)
        self.weights = {
            'wait_time_reduction': self.config.get('wait_time_weight', 0.4),
            'throughput_increase': self.config.get('throughput_weight', 0.2),
            'fuel_efficiency': self.config.get('fuel_efficiency_weight', 0.15),
            'pedestrian_safety': self.config.get('pedestrian_safety_weight', 0.1),
            'coordination_bonus': self.config.get('coordination_weight', 0.05),
            'stability_penalty': self.config.get('stability_weight', 0.05),
            'emergency_priority': self.config.get('emergency_weight', 0.03),
            'environmental_adaptation': self.config.get('environmental_weight', 0.02)
        }
        
        # Normalization factors
        self.normalization_factors = {
            'wait_time': 120.0,  # Max wait time in seconds
            'throughput': 1000.0,  # Max throughput in veh/h
            'fuel_consumption': 50.0,  # Max fuel consumption penalty
            'pedestrian_delay': 60.0,  # Max pedestrian delay in seconds
            'coordination': 1.0,  # Binary coordination bonus
            'stability': 1.0,  # Binary stability penalty
            'emergency': 1.0,  # Binary emergency priority
            'environmental': 1.0  # Environmental adaptation factor
        }
        
        # Historical data for trend analysis
        self.historical_rewards = []
        self.reward_trends = {}
        
        # Performance baselines
        self.baseline_metrics = {
            'wait_time': 60.0,  # Baseline wait time in seconds
            'throughput': 500.0,  # Baseline throughput in veh/h
            'fuel_consumption': 25.0,  # Baseline fuel consumption
            'pedestrian_delay': 30.0  # Baseline pedestrian delay
        }
    
    def calculate_reward(self, state: MultiDimensionalState, action: SophisticatedAction,
                        next_state: MultiDimensionalState, 
                        performance_metrics: Optional[Dict[str, float]] = None,
                        coordination_data: Optional[Dict[str, Any]] = None) -> RewardComponents:
        """
        Calculate comprehensive reward for state-action-next_state transition
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state after action
            performance_metrics: Additional performance metrics
            coordination_data: Data from adjacent intersections
            
        Returns:
            RewardComponents with detailed breakdown
        """
        components = RewardComponents()
        
        # 1. Wait time reduction (primary objective)
        components.wait_time_reduction = self._calculate_wait_time_reward(state, next_state)
        
        # 2. Throughput increase
        components.throughput_increase = self._calculate_throughput_reward(state, next_state)
        
        # 3. Fuel efficiency
        components.fuel_efficiency = self._calculate_fuel_efficiency_reward(state, action)
        
        # 4. Pedestrian safety
        components.pedestrian_safety = self._calculate_pedestrian_safety_reward(state, action)
        
        # 5. Coordination bonus
        components.coordination_bonus = self._calculate_coordination_reward(action, coordination_data)
        
        # 6. Stability penalty
        components.stability_penalty = self._calculate_stability_penalty(action)
        
        # 7. Emergency vehicle priority
        components.emergency_priority = self._calculate_emergency_priority_reward(state, action)
        
        # 8. Environmental adaptation
        components.environmental_adaptation = self._calculate_environmental_adaptation_reward(state, action)
        
        # Calculate total weighted reward
        components.total_reward = self._calculate_total_reward(components)
        
        # Store for trend analysis
        self._update_reward_history(components)
        
        return components
    
    def _calculate_wait_time_reward(self, state: MultiDimensionalState, 
                                  next_state: MultiDimensionalState) -> float:
        """Calculate reward for wait time reduction"""
        # Current wait times
        current_wait_times = state.waiting_times
        next_wait_times = next_state.waiting_times
        
        # Calculate wait time reduction
        wait_time_reduction = np.sum(current_wait_times) - np.sum(next_wait_times)
        
        # Normalize by maximum expected wait time
        normalized_reduction = wait_time_reduction / self.normalization_factors['wait_time']
        
        # Apply non-linear scaling for better learning
        if normalized_reduction > 0:
            # Positive reward for reduction (logarithmic scaling)
            reward = math.log(1 + normalized_reduction) * 0.5
        else:
            # Negative reward for increase (exponential scaling)
            reward = -math.exp(abs(normalized_reduction)) * 0.3
        
        return reward
    
    def _calculate_throughput_reward(self, state: MultiDimensionalState, 
                                   next_state: MultiDimensionalState) -> float:
        """Calculate reward for throughput increase"""
        # Current and next flow rates
        current_throughput = np.sum(state.flow_rates)
        next_throughput = np.sum(next_state.flow_rates)
        
        # Calculate throughput increase
        throughput_increase = next_throughput - current_throughput
        
        # Normalize by maximum expected throughput
        normalized_increase = throughput_increase / self.normalization_factors['throughput']
        
        # Apply scaling
        if normalized_increase > 0:
            reward = math.sqrt(normalized_increase) * 0.4
        else:
            reward = normalized_increase * 0.2  # Linear penalty for decrease
        
        return reward
    
    def _calculate_fuel_efficiency_reward(self, state: MultiDimensionalState, 
                                        action: SophisticatedAction) -> float:
        """Calculate reward for fuel efficiency"""
        # Estimate fuel consumption based on idling vehicles
        idling_vehicles = np.sum(state.queue_lengths)
        
        # Fuel consumption penalty (vehicles idling consume fuel)
        fuel_penalty = idling_vehicles * 0.1  # 0.1 fuel units per idling vehicle
        
        # Normalize penalty
        normalized_penalty = fuel_penalty / self.normalization_factors['fuel_consumption']
        
        # Reward for actions that reduce idling
        if action.action_type in [2, 3]:  # Actions that increase throughput
            efficiency_bonus = 0.1
        else:
            efficiency_bonus = 0.0
        
        # Total fuel efficiency reward (negative penalty + positive bonus)
        reward = -normalized_penalty + efficiency_bonus
        
        return reward
    
    def _calculate_pedestrian_safety_reward(self, state: MultiDimensionalState, 
                                          action: SophisticatedAction) -> float:
        """Calculate reward for pedestrian safety"""
        # Base pedestrian safety score
        pedestrian_safety_score = 0.0
        
        # Check if action is pedestrian-friendly
        if action.action_type == 5:  # Pedestrian-friendly action
            pedestrian_safety_score += 0.3
        
        # Check for pedestrian crossing phases
        if state.current_phase == 3:  # Pedestrian phase
            # Reward for maintaining pedestrian phase
            if action.action_type == 1:  # Maintain current timing
                pedestrian_safety_score += 0.2
            # Penalty for reducing pedestrian time
            elif action.action_type == 0:  # Reduce timing
                pedestrian_safety_score -= 0.1
        
        # Weather-based pedestrian safety
        if state.weather_condition in [2, 3, 4]:  # Rainy, foggy, stormy
            # Higher reward for pedestrian-friendly actions in bad weather
            if action.action_type == 5:
                pedestrian_safety_score += 0.2
        
        # Visibility-based safety
        if state.visibility < 10.0:  # Low visibility
            if action.action_type == 5:
                pedestrian_safety_score += 0.1
        
        return pedestrian_safety_score
    
    def _calculate_coordination_reward(self, action: SophisticatedAction, 
                                     coordination_data: Optional[Dict[str, Any]]) -> float:
        """Calculate reward for coordination with adjacent intersections"""
        if not coordination_data:
            return 0.0
        
        coordination_reward = 0.0
        
        # Green wave coordination
        if action.action_type == 3:  # Green wave coordination
            coordination_reward += 0.2
        
        # Check for synchronized actions
        if 'synchronized_actions' in coordination_data:
            synchronized_count = coordination_data['synchronized_actions']
            coordination_reward += synchronized_count * 0.05
        
        # Check for complementary timing
        if 'complementary_timing' in coordination_data:
            if coordination_data['complementary_timing']:
                coordination_reward += 0.1
        
        # Check for conflict resolution
        if 'conflicts_resolved' in coordination_data:
            conflicts_resolved = coordination_data['conflicts_resolved']
            coordination_reward += conflicts_resolved * 0.03
        
        return coordination_reward
    
    def _calculate_stability_penalty(self, action: SophisticatedAction) -> float:
        """Calculate penalty for signal instability"""
        stability_penalty = 0.0
        
        # Penalty for frequent action changes
        if hasattr(self, 'last_action_type'):
            if self.last_action_type != action.action_type:
                stability_penalty -= 0.05
        
        # Penalty for extreme timing changes
        if abs(action.cycle_time_adjustment) > 20:  # Large cycle time change
            stability_penalty -= 0.1
        
        # Penalty for frequent phase changes
        if action.phase_sequence_change:
            stability_penalty -= 0.03
        
        # Store current action for next comparison
        self.last_action_type = action.action_type
        
        return stability_penalty
    
    def _calculate_emergency_priority_reward(self, state: MultiDimensionalState, 
                                           action: SophisticatedAction) -> float:
        """Calculate reward for emergency vehicle priority"""
        if not state.emergency_vehicles:
            return 0.0
        
        emergency_reward = 0.0
        
        # High reward for emergency priority actions
        if action.action_type == 4:  # Emergency priority action
            emergency_reward += 0.5
        
        # Reward for priority boost
        if action.priority_boost:
            emergency_reward += 0.3
        
        # Reward for safety override
        if action.safety_override:
            emergency_reward += 0.2
        
        # Penalty for not giving priority when emergency vehicles present
        if not action.priority_boost and state.emergency_vehicles:
            emergency_reward -= 0.4
        
        return emergency_reward
    
    def _calculate_environmental_adaptation_reward(self, state: MultiDimensionalState, 
                                                 action: SophisticatedAction) -> float:
        """Calculate reward for environmental adaptation"""
        environmental_reward = 0.0
        
        # Weather-based adaptation
        if state.weather_condition == 2:  # Rainy
            # Reward for more conservative actions in rain
            if action.action_type in [0, 1]:  # Conservative actions
                environmental_reward += 0.1
        elif state.weather_condition == 3:  # Foggy
            # Reward for pedestrian-friendly actions in fog
            if action.action_type == 5:
                environmental_reward += 0.15
        elif state.weather_condition == 4:  # Stormy
            # Reward for emergency-ready actions in storms
            if action.action_type == 4:
                environmental_reward += 0.2
        
        # Temperature-based adaptation
        if state.temperature < 0:  # Freezing conditions
            # Reward for actions that reduce idling (prevent engine cooling)
            if action.action_type in [2, 3]:  # Actions that increase flow
                environmental_reward += 0.1
        
        # Visibility-based adaptation
        if state.visibility < 5.0:  # Very low visibility
            # Reward for more conservative actions
            if action.action_type in [0, 1]:
                environmental_reward += 0.1
        
        return environmental_reward
    
    def _calculate_total_reward(self, components: RewardComponents) -> float:
        """Calculate total weighted reward"""
        total_reward = 0.0
        
        # Apply weights to each component
        total_reward += components.wait_time_reduction * self.weights['wait_time_reduction']
        total_reward += components.throughput_increase * self.weights['throughput_increase']
        total_reward += components.fuel_efficiency * self.weights['fuel_efficiency']
        total_reward += components.pedestrian_safety * self.weights['pedestrian_safety']
        total_reward += components.coordination_bonus * self.weights['coordination_bonus']
        total_reward += components.stability_penalty * self.weights['stability_penalty']
        total_reward += components.emergency_priority * self.weights['emergency_priority']
        total_reward += components.environmental_adaptation * self.weights['environmental_adaptation']
        
        # Apply reward shaping (optional)
        total_reward = self._apply_reward_shaping(total_reward, components)
        
        return total_reward
    
    def _apply_reward_shaping(self, reward: float, components: RewardComponents) -> float:
        """Apply reward shaping techniques"""
        # Potential-based reward shaping
        potential = self._calculate_potential(components)
        shaped_reward = reward + potential
        
        # Reward clipping to prevent extreme values
        shaped_reward = np.clip(shaped_reward, -2.0, 2.0)
        
        return shaped_reward
    
    def _calculate_potential(self, components: RewardComponents) -> float:
        """Calculate potential-based reward shaping"""
        # Simple potential based on wait time reduction
        potential = components.wait_time_reduction * 0.1
        
        return potential
    
    def _update_reward_history(self, components: RewardComponents):
        """Update historical reward data for trend analysis"""
        self.historical_rewards.append(components.total_reward)
        
        # Keep only recent history (last 1000 rewards)
        if len(self.historical_rewards) > 1000:
            self.historical_rewards = self.historical_rewards[-1000:]
        
        # Update component trends
        for component_name, value in components.to_dict().items():
            if component_name not in self.reward_trends:
                self.reward_trends[component_name] = []
            
            self.reward_trends[component_name].append(value)
            
            # Keep only recent history
            if len(self.reward_trends[component_name]) > 1000:
                self.reward_trends[component_name] = self.reward_trends[component_name][-1000:]
    
    def get_reward_statistics(self) -> Dict[str, Any]:
        """Get reward statistics and trends"""
        if not self.historical_rewards:
            return {}
        
        stats = {
            'total_rewards': len(self.historical_rewards),
            'avg_reward': np.mean(self.historical_rewards),
            'std_reward': np.std(self.historical_rewards),
            'min_reward': np.min(self.historical_rewards),
            'max_reward': np.max(self.historical_rewards),
            'recent_avg_reward': np.mean(self.historical_rewards[-100:]) if len(self.historical_rewards) >= 100 else np.mean(self.historical_rewards)
        }
        
        # Component statistics
        for component_name, values in self.reward_trends.items():
            if values:
                stats[f'{component_name}_avg'] = np.mean(values)
                stats[f'{component_name}_std'] = np.std(values)
                stats[f'{component_name}_recent_avg'] = np.mean(values[-100:]) if len(values) >= 100 else np.mean(values)
        
        return stats
    
    def update_weights(self, new_weights: Dict[str, float]):
        """Update reward function weights"""
        for weight_name, weight_value in new_weights.items():
            if weight_name in self.weights:
                self.weights[weight_name] = weight_value
                self.logger.info(f"Updated weight {weight_name} to {weight_value}")
    
    def get_weights(self) -> Dict[str, float]:
        """Get current reward function weights"""
        return self.weights.copy()
    
    def reset_history(self):
        """Reset reward history"""
        self.historical_rewards = []
        self.reward_trends = {}
        self.logger.info("Reward history reset")


class RewardFunctionValidator:
    """Validates and tests reward function performance"""
    
    def __init__(self, reward_function: AdvancedRewardFunction):
        self.reward_function = reward_function
        self.logger = logging.getLogger(__name__)
    
    def validate_reward_function(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate reward function with test cases"""
        validation_results = {
            'total_tests': len(test_cases),
            'passed_tests': 0,
            'failed_tests': 0,
            'test_results': []
        }
        
        for i, test_case in enumerate(test_cases):
            try:
                result = self._run_test_case(test_case)
                validation_results['test_results'].append(result)
                
                if result['passed']:
                    validation_results['passed_tests'] += 1
                else:
                    validation_results['failed_tests'] += 1
                    
            except Exception as e:
                self.logger.error(f"Test case {i} failed with error: {e}")
                validation_results['test_results'].append({
                    'test_id': i,
                    'passed': False,
                    'error': str(e)
                })
                validation_results['failed_tests'] += 1
        
        return validation_results
    
    def _run_test_case(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        # Extract test data
        state = test_case['state']
        action = test_case['action']
        next_state = test_case['next_state']
        expected_reward_range = test_case.get('expected_reward_range', (-2.0, 2.0))
        
        # Calculate reward
        reward_components = self.reward_function.calculate_reward(
            state, action, next_state
        )
        
        # Validate reward
        reward = reward_components.total_reward
        min_expected, max_expected = expected_reward_range
        
        passed = min_expected <= reward <= max_expected
        
        return {
            'test_id': test_case.get('test_id', 0),
            'passed': passed,
            'actual_reward': reward,
            'expected_range': expected_reward_range,
            'reward_components': reward_components.to_dict()
        }
    
    def create_test_cases(self) -> List[Dict[str, Any]]:
        """Create test cases for reward function validation"""
        test_cases = []
        
        # Test case 1: Normal traffic, good action
        test_cases.append({
            'test_id': 1,
            'description': 'Normal traffic with good action',
            'state': self._create_mock_state(congestion_level=0.5, emergency_vehicles=False),
            'action': self._create_mock_action(action_type=2),  # Good action
            'next_state': self._create_mock_state(congestion_level=0.3, emergency_vehicles=False),
            'expected_reward_range': (0.0, 1.0)
        })
        
        # Test case 2: High congestion, emergency vehicles
        test_cases.append({
            'test_id': 2,
            'description': 'High congestion with emergency vehicles',
            'state': self._create_mock_state(congestion_level=0.9, emergency_vehicles=True),
            'action': self._create_mock_action(action_type=4, priority_boost=True),  # Emergency action
            'next_state': self._create_mock_state(congestion_level=0.7, emergency_vehicles=False),
            'expected_reward_range': (0.5, 2.0)
        })
        
        # Test case 3: Bad weather, pedestrian safety
        test_cases.append({
            'test_id': 3,
            'description': 'Bad weather with pedestrian safety action',
            'state': self._create_mock_state(weather_condition=2, visibility=5.0),  # Rainy, low visibility
            'action': self._create_mock_action(action_type=5),  # Pedestrian safety action
            'next_state': self._create_mock_state(weather_condition=2, visibility=5.0),
            'expected_reward_range': (0.2, 1.5)
        })
        
        return test_cases
    
    def _create_mock_state(self, **kwargs) -> MultiDimensionalState:
        """Create mock state for testing"""
        defaults = {
            'lane_counts': np.array([10, 10, 10, 10]),
            'avg_speed': 30.0,
            'queue_lengths': np.array([5, 5, 5, 5]),
            'waiting_times': np.array([20, 20, 20, 20]),
            'flow_rates': np.array([100, 100, 100, 100]),
            'current_phase': 0,
            'phase_duration': 30.0,
            'cycle_progress': 0.5,
            'time_since_change': 15.0,
            'time_of_day': 0.5,
            'day_of_week': 0.5,
            'is_weekend': False,
            'is_holiday': False,
            'season': 1,
            'weather_condition': 0,
            'temperature': 20.0,
            'visibility': 10.0,
            'precipitation_intensity': 0.0,
            'adjacent_signals': {},
            'upstream_flow': np.array([0, 0, 0, 0]),
            'downstream_capacity': np.array([1000, 1000, 1000, 1000]),
            'recent_performance': {'avg_wait_time': 20.0, 'throughput': 400.0},
            'congestion_trend': 0.0,
            'emergency_vehicles': False
        }
        
        # Update with provided values
        defaults.update(kwargs)
        
        return MultiDimensionalState(**defaults)
    
    def _create_mock_action(self, **kwargs) -> SophisticatedAction:
        """Create mock action for testing"""
        defaults = {
            'action_type': 0,
            'green_time_adjustments': np.array([0, 0, 0, 0]),
            'cycle_time_adjustment': 0,
            'phase_sequence_change': False,
            'priority_boost': False,
            'coordination_signal': 'maintain',
            'safety_override': False
        }
        
        # Update with provided values
        defaults.update(kwargs)
        
        return SophisticatedAction(**defaults)
