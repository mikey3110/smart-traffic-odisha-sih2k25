#!/usr/bin/env python3
"""
Test script for Advanced Multi-Intersection Q-Learning System
Verifies all components are working correctly
"""

import sys
import os
import logging
import asyncio
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from algorithms.advanced_q_learning_agent import AdvancedQLearningAgent, MultiDimensionalState, SophisticatedAction
from algorithms.multi_intersection_coordinator import MultiIntersectionCoordinator
from algorithms.advanced_reward_function import AdvancedRewardFunction
from algorithms.adaptive_experience_replay import AdaptiveExperienceReplay
from production_q_learning_system import ProductionQLearningSystem


def test_advanced_q_learning_agent():
    """Test Advanced Q-Learning Agent"""
    print("Testing Advanced Q-Learning Agent...")
    
    try:
        # Create agent
        config = {
            'learning_rate': 0.001,
            'epsilon': 0.1,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'discount_factor': 0.95,
            'replay_buffer_size': 1000,
            'batch_size': 32,
            'target_update_frequency': 100,
            'hidden_layers': [128, 64],
            'intersection_id': 'test-junction',
            'adjacent_intersections': []
        }
        
        agent = AdvancedQLearningAgent('test-junction', config)
        
        # Test state creation
        traffic_data = {
            'intersection_id': 'test-junction',
            'timestamp': datetime.now().isoformat(),
            'lane_counts': {'north_lane': 10, 'south_lane': 15, 'east_lane': 8, 'west_lane': 12},
            'avg_speed': 30.0,
            'weather_condition': 'clear',
            'temperature': 20.0,
            'visibility': 10.0,
            'emergency_vehicles': False
        }
        
        state = agent.create_state(traffic_data, {}, [])
        assert isinstance(state, MultiDimensionalState)
        assert len(state.to_vector()) == 45
        print("✓ State creation successful")
        
        # Test action selection
        action = agent.select_action(state, training=True)
        assert isinstance(action, SophisticatedAction)
        assert 0 <= action.action_type < 8
        print("✓ Action selection successful")
        
        # Test signal optimization
        current_timings = {'north_lane': 30, 'south_lane': 30, 'east_lane': 30, 'west_lane': 30}
        optimized_timings = agent.optimize_signal_timing(traffic_data, current_timings)
        assert isinstance(optimized_timings, dict)
        assert all(lane in optimized_timings for lane in current_timings.keys())
        print("✓ Signal optimization successful")
        
        # Test training statistics
        stats = agent.get_training_statistics()
        assert 'training_step' in stats
        assert 'epsilon' in stats
        print("✓ Training statistics successful")
        
        print("✅ Advanced Q-Learning Agent test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Advanced Q-Learning Agent test failed: {e}")
        return False


def test_multi_intersection_coordinator():
    """Test Multi-Intersection Coordinator"""
    print("Testing Multi-Intersection Coordinator...")
    
    try:
        # Create coordinator
        config = {
            'intersections': [
                {'id': 'junction-1', 'adjacent_intersections': ['junction-2']},
                {'id': 'junction-2', 'adjacent_intersections': ['junction-1']}
            ]
        }
        
        coordinator = MultiIntersectionCoordinator('junction-1', config)
        
        # Test message sending
        message_id = coordinator.send_message(
            'junction-2',
            'state_update',
            {'phase': 1, 'duration': 30}
        )
        assert message_id is not None
        print("✓ Message sending successful")
        
        # Test coordination metrics
        metrics = coordinator.get_coordination_metrics()
        assert 'intersection_id' in metrics
        assert 'is_coordinating' in metrics
        print("✓ Coordination metrics successful")
        
        # Test network topology
        topology = coordinator.get_network_topology()
        assert 'nodes' in topology
        assert 'edges' in topology
        print("✓ Network topology successful")
        
        print("✅ Multi-Intersection Coordinator test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Multi-Intersection Coordinator test failed: {e}")
        return False


def test_advanced_reward_function():
    """Test Advanced Reward Function"""
    print("Testing Advanced Reward Function...")
    
    try:
        # Create reward function
        config = {
            'wait_time_weight': 0.4,
            'throughput_weight': 0.2,
            'fuel_efficiency_weight': 0.15,
            'pedestrian_safety_weight': 0.1,
            'coordination_weight': 0.05,
            'stability_weight': 0.05,
            'emergency_weight': 0.03,
            'environmental_weight': 0.02
        }
        
        reward_function = AdvancedRewardFunction(config)
        
        # Create test states and action
        state = MultiDimensionalState(
            lane_counts=np.array([10, 15, 8, 12]),
            avg_speed=30.0,
            queue_lengths=np.array([5, 8, 4, 6]),
            waiting_times=np.array([20, 25, 15, 18]),
            flow_rates=np.array([100, 150, 80, 120]),
            current_phase=0,
            phase_duration=30.0,
            cycle_progress=0.5,
            time_since_change=15.0,
            time_of_day=0.5,
            day_of_week=0.5,
            is_weekend=False,
            is_holiday=False,
            season=1,
            weather_condition=0,
            temperature=20.0,
            visibility=10.0,
            precipitation_intensity=0.0,
            adjacent_signals={},
            upstream_flow=np.array([0, 0, 0, 0]),
            downstream_capacity=np.array([1000, 1000, 1000, 1000]),
            recent_performance={'avg_wait_time': 20.0, 'throughput': 400.0},
            congestion_trend=0.0,
            emergency_vehicles=False
        )
        
        action = SophisticatedAction(
            action_type=2,
            green_time_adjustments=np.array([5, 5, -5, -5]),
            cycle_time_adjustment=10,
            phase_sequence_change=False,
            priority_boost=False,
            coordination_signal='increase_throughput',
            safety_override=False
        )
        
        next_state = state  # Simplified for testing
        
        # Test reward calculation
        reward_components = reward_function.calculate_reward(state, action, next_state)
        assert hasattr(reward_components, 'total_reward')
        assert hasattr(reward_components, 'wait_time_reduction')
        assert hasattr(reward_components, 'throughput_increase')
        print("✓ Reward calculation successful")
        
        # Test reward statistics
        stats = reward_function.get_reward_statistics()
        assert isinstance(stats, dict)
        print("✓ Reward statistics successful")
        
        print("✅ Advanced Reward Function test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Advanced Reward Function test failed: {e}")
        return False


def test_adaptive_experience_replay():
    """Test Adaptive Experience Replay"""
    print("Testing Adaptive Experience Replay...")
    
    try:
        # Create replay buffer
        replay_buffer = AdaptiveExperienceReplay(
            capacity=1000,
            alpha=0.6,
            beta=0.4
        )
        
        # Create test experiences
        state = MultiDimensionalState(
            lane_counts=np.array([10, 15, 8, 12]),
            avg_speed=30.0,
            queue_lengths=np.array([5, 8, 4, 6]),
            waiting_times=np.array([20, 25, 15, 18]),
            flow_rates=np.array([100, 150, 80, 120]),
            current_phase=0,
            phase_duration=30.0,
            cycle_progress=0.5,
            time_since_change=15.0,
            time_of_day=0.5,
            day_of_week=0.5,
            is_weekend=False,
            is_holiday=False,
            season=1,
            weather_condition=0,
            temperature=20.0,
            visibility=10.0,
            precipitation_intensity=0.0,
            adjacent_signals={},
            upstream_flow=np.array([0, 0, 0, 0]),
            downstream_capacity=np.array([1000, 1000, 1000, 1000]),
            recent_performance={'avg_wait_time': 20.0, 'throughput': 400.0},
            congestion_trend=0.0,
            emergency_vehicles=False
        )
        
        action = SophisticatedAction(
            action_type=2,
            green_time_adjustments=np.array([5, 5, -5, -5]),
            cycle_time_adjustment=10,
            phase_sequence_change=False,
            priority_boost=False,
            coordination_signal='increase_throughput',
            safety_override=False
        )
        
        # Add experiences
        for i in range(10):
            reward = np.random.uniform(-1, 1)
            replay_buffer.add_experience(state, action, reward, state, False, reward)
        
        assert len(replay_buffer) == 10
        print("✓ Experience addition successful")
        
        # Test sampling
        experiences, weights, indices = replay_buffer.sample(5)
        assert len(experiences) == 5
        assert len(weights) == 5
        assert len(indices) == 5
        print("✓ Experience sampling successful")
        
        # Test statistics
        stats = replay_buffer.get_statistics()
        assert 'size' in stats
        assert 'capacity' in stats
        print("✓ Statistics successful")
        
        print("✅ Adaptive Experience Replay test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Adaptive Experience Replay test failed: {e}")
        return False


async def test_production_system():
    """Test Production Q-Learning System"""
    print("Testing Production Q-Learning System...")
    
    try:
        # Create minimal config
        config = {
            'intersections': [
                {'id': 'junction-1', 'adjacent_intersections': ['junction-2']},
                {'id': 'junction-2', 'adjacent_intersections': ['junction-1']}
            ],
            'q_learning': {
                'learning_rate': 0.001,
                'epsilon': 0.1,
                'epsilon_decay': 0.995,
                'epsilon_min': 0.01,
                'discount_factor': 0.95,
                'replay_buffer_size': 1000,
                'batch_size': 32,
                'target_update_frequency': 100,
                'hidden_layers': [128, 64]
            },
            'reward_function': {
                'wait_time_weight': 0.4,
                'throughput_weight': 0.2,
                'fuel_efficiency_weight': 0.15,
                'pedestrian_safety_weight': 0.1,
                'coordination_weight': 0.05,
                'stability_weight': 0.05,
                'emergency_weight': 0.03,
                'environmental_weight': 0.02
            },
            'experience_replay': {
                'capacity': 1000,
                'alpha': 0.6,
                'beta': 0.4
            }
        }
        
        # Create system
        system = ProductionQLearningSystem()
        
        # Test system status
        status = system.get_system_status()
        assert 'is_running' in status
        assert 'is_training' in status
        print("✓ System status successful")
        
        # Test performance metrics
        metrics = system.get_performance_metrics()
        assert 'total_cycles' in metrics
        print("✓ Performance metrics successful")
        
        print("✅ Production Q-Learning System test passed\n")
        return True
        
    except Exception as e:
        print(f"❌ Production Q-Learning System test failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Advanced Multi-Intersection Q-Learning System Tests")
    print("=" * 60)
    
    tests = [
        test_advanced_q_learning_agent,
        test_multi_intersection_coordinator,
        test_advanced_reward_function,
        test_adaptive_experience_replay,
        test_production_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if asyncio.iscoroutinefunction(test):
                result = asyncio.run(test())
            else:
                result = test()
            
            if result:
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.WARNING,  # Reduce log noise during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run tests
    success = run_all_tests()
    sys.exit(0 if success else 1)
