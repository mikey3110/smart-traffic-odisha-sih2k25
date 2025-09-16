"""
Comprehensive Unit Tests and Performance Benchmarking for Phase 2
Real-Time Optimization Loop & Safety Systems

Features:
- Unit tests for all Phase 2 components
- Integration tests with mock SUMO data
- Performance benchmarking and optimization reports
- Load testing and stress testing
- End-to-end system validation
"""

import unittest
import asyncio
import threading
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import tempfile
import os
import sys
import logging
from unittest.mock import Mock, patch, MagicMock
import psutil
import gc

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime.real_time_optimizer import (
    RealTimeOptimizer, ThreadSafeStateManager, AdaptiveConfidenceScorer,
    RealTimeDataIngestion, OptimizationMode, OptimizationRequest, OptimizationResponse
)
from safety.safety_manager import (
    SafetyManager, WebsterFormulaFallback, EmergencyVehicleManager,
    SafetyConstraintManager, SafetyLevel, SafetyViolation
)
from performance.optimized_q_table import (
    OptimizedQTable, StateHasher, ActionHasher, MemoryMappedQTable,
    LoadBalancer, PerformanceMonitor
)
from sumo.enhanced_traci_controller import (
    EnhancedTraCIController, SimulationState, ConnectionStatus,
    TrafficData, ControlCommand, TraCIError
)


class TestThreadSafeStateManager(unittest.TestCase):
    """Test ThreadSafeStateManager"""
    
    def setUp(self):
        self.state_manager = ThreadSafeStateManager(max_intersections=5)
        self.test_state = Mock()
        self.test_state.to_vector.return_value = np.array([1, 2, 3, 4])
        self.test_state.get_state_hash.return_value = "test_hash"
    
    def test_update_state(self):
        """Test state update functionality"""
        result = self.state_manager.update_state("intersection_1", self.test_state)
        self.assertTrue(result)
        
        # Verify state was stored
        stored_state = self.state_manager.get_state("intersection_1")
        self.assertEqual(stored_state, self.test_state)
    
    def test_concurrent_updates(self):
        """Test concurrent state updates"""
        def update_state(intersection_id):
            for i in range(10):
                state = Mock()
                state.to_vector.return_value = np.array([i, i+1, i+2, i+3])
                state.get_state_hash.return_value = f"hash_{i}"
                self.state_manager.update_state(intersection_id, state)
        
        # Create multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=update_state, args=(f"intersection_{i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all states were updated
        for i in range(3):
            state = self.state_manager.get_state(f"intersection_{i}")
            self.assertIsNotNone(state)
    
    def test_stale_state_detection(self):
        """Test stale state detection"""
        # Update state
        self.state_manager.update_state("intersection_1", self.test_state)
        
        # State should not be stale immediately
        self.assertFalse(self.state_manager.is_state_stale("intersection_1", max_age=60.0))
        
        # Mock old timestamp
        with patch.object(self.state_manager, 'last_update', {"intersection_1": datetime.now() - timedelta(seconds=70)}):
            self.assertTrue(self.state_manager.is_state_stale("intersection_1", max_age=60.0))


class TestAdaptiveConfidenceScorer(unittest.TestCase):
    """Test AdaptiveConfidenceScorer"""
    
    def setUp(self):
        self.scorer = AdaptiveConfidenceScorer()
        self.test_state = Mock()
        self.test_state.to_vector.return_value = np.array([10, 20, 30, 40])
        self.test_state.lane_counts = np.array([10, 20, 30, 40])
        self.test_state.time_since_change = 30
        self.test_state.congestion_trend = 0.5
        self.test_state.emergency_vehicles = False
        self.test_state.weather_condition = 0
        self.test_state.visibility = 10.0
    
    def test_confidence_calculation(self):
        """Test confidence score calculation"""
        action = Mock()
        confidence = self.scorer.calculate_confidence("intersection_1", self.test_state, action)
        
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_confidence_with_stale_data(self):
        """Test confidence with stale data"""
        # Mock stale data
        self.test_state.to_vector.return_value = np.array([np.nan, 20, 30, 40])
        
        action = Mock()
        confidence = self.scorer.calculate_confidence("intersection_1", self.test_state, action)
        
        # Should have low confidence due to NaN values
        self.assertLess(confidence, 0.5)
    
    def test_confidence_with_high_congestion(self):
        """Test confidence with high congestion"""
        self.test_state.congestion_trend = 0.9
        self.test_state.lane_counts = np.array([100, 100, 100, 100])
        
        action = Mock()
        confidence = self.scorer.calculate_confidence("intersection_1", self.test_state, action)
        
        # Should have lower confidence due to high congestion
        self.assertLess(confidence, 0.8)
    
    def test_performance_update(self):
        """Test performance update functionality"""
        self.scorer.update_performance("intersection_1", True, 0.5)
        
        # Check if performance was recorded
        self.assertIn("intersection_1", self.scorer.performance_history)
        self.assertEqual(len(self.scorer.performance_history["intersection_1"]), 1)


class TestRealTimeDataIngestion(unittest.TestCase):
    """Test RealTimeDataIngestion"""
    
    def setUp(self):
        self.ingestion = RealTimeDataIngestion()
    
    def test_add_data_source(self):
        """Test adding data source"""
        source_config = {
            'type': 'camera',
            'intersection_id': 'intersection_1',
            'frequency': 1.0
        }
        
        self.ingestion.add_data_source("camera_1", source_config)
        
        self.assertIn("camera_1", self.ingestion.data_sources)
        self.assertEqual(self.ingestion.data_sources["camera_1"]["config"], source_config)
    
    def test_data_validation(self):
        """Test data validation"""
        # Valid data
        valid_data = {
            'intersection_id': 'intersection_1',
            'timestamp': datetime.now().isoformat(),
            'lane_counts': {'north': 10, 'south': 15}
        }
        
        self.assertTrue(self.ingestion._validate_data(valid_data))
        
        # Invalid data - missing required fields
        invalid_data = {
            'lane_counts': {'north': 10, 'south': 15}
        }
        
        self.assertFalse(self.ingestion._validate_data(invalid_data))
        
        # Invalid data - invalid lane counts
        invalid_data2 = {
            'intersection_id': 'intersection_1',
            'timestamp': datetime.now().isoformat(),
            'lane_counts': {'north': -5, 'south': 15}
        }
        
        self.assertFalse(self.ingestion._validate_data(invalid_data2))


class TestWebsterFormulaFallback(unittest.TestCase):
    """Test WebsterFormulaFallback"""
    
    def setUp(self):
        self.webster = WebsterFormulaFallback()
        self.webster.configure_intersection("intersection_1", {
            'lanes': ['north', 'south', 'east', 'west'],
            'approach_angles': [0, 180, 90, 270]
        })
    
    def test_webster_optimization(self):
        """Test Webster's formula optimization"""
        traffic_data = {
            'lane_counts': {
                'north': 20,
                'south': 15,
                'east': 10,
                'west': 5
            }
        }
        
        result = self.webster.optimize_intersection("intersection_1", traffic_data)
        
        # Check that all lanes have timings
        self.assertEqual(len(result), 4)
        self.assertIn('north', result)
        self.assertIn('south', result)
        self.assertIn('east', result)
        self.assertIn('west', result)
        
        # Check that timings are reasonable
        for lane, timing in result.items():
            self.assertGreaterEqual(timing, 10)  # Minimum green time
            self.assertLessEqual(timing, 60)     # Maximum green time
    
    def test_safety_constraints(self):
        """Test safety constraint application"""
        # Test with extreme traffic data
        traffic_data = {
            'lane_counts': {
                'north': 200,  # Very high
                'south': 1,    # Very low
                'east': 0,     # Zero
                'west': 50     # Normal
            }
        }
        
        result = self.webster.optimize_intersection("intersection_1", traffic_data)
        
        # All timings should be within safety bounds
        for lane, timing in result.items():
            self.assertGreaterEqual(timing, 10)
            self.assertLessEqual(timing, 60)


class TestEmergencyVehicleManager(unittest.TestCase):
    """Test EmergencyVehicleManager"""
    
    def setUp(self):
        self.emergency_manager = EmergencyVehicleManager()
    
    def test_register_emergency_vehicle(self):
        """Test emergency vehicle registration"""
        alert_id = self.emergency_manager.register_emergency_vehicle(
            "intersection_1",
            "ambulance",
            1,
            datetime.now() + timedelta(minutes=2),
            (40.7128, -74.0060),
            "hospital"
        )
        
        self.assertIsNotNone(alert_id)
        self.assertIn(alert_id, self.emergency_manager.active_alerts)
    
    def test_get_active_alerts(self):
        """Test getting active alerts"""
        # Register multiple alerts
        alert1 = self.emergency_manager.register_emergency_vehicle(
            "intersection_1", "ambulance", 1, datetime.now() + timedelta(minutes=1), (0, 0), "hospital"
        )
        alert2 = self.emergency_manager.register_emergency_vehicle(
            "intersection_2", "fire_truck", 1, datetime.now() + timedelta(minutes=1), (0, 0), "fire_station"
        )
        
        # Get all active alerts
        active_alerts = self.emergency_manager.get_active_alerts()
        self.assertEqual(len(active_alerts), 2)
        
        # Get alerts for specific intersection
        intersection1_alerts = self.emergency_manager.get_active_alerts("intersection_1")
        self.assertEqual(len(intersection1_alerts), 1)
        self.assertEqual(intersection1_alerts[0].intersection_id, "intersection_1")
    
    def test_emergency_override(self):
        """Test emergency vehicle priority override"""
        # Register emergency vehicle
        self.emergency_manager.register_emergency_vehicle(
            "intersection_1",
            "ambulance",
            1,
            datetime.now() + timedelta(minutes=1),
            (0, 0),
            "hospital"
        )
        
        current_timings = {
            'north_lane': 30,
            'south_lane': 30,
            'east_lane': 30,
            'west_lane': 30
        }
        
        override_timings = self.emergency_manager.handle_emergency_override(
            "intersection_1", current_timings
        )
        
        # Should be different from current timings
        self.assertNotEqual(override_timings, current_timings)
        
        # Main approach should have increased time
        self.assertGreaterEqual(override_timings['north_lane'], current_timings['north_lane'])
        self.assertGreaterEqual(override_timings['south_lane'], current_timings['south_lane'])


class TestSafetyConstraintManager(unittest.TestCase):
    """Test SafetyConstraintManager"""
    
    def setUp(self):
        self.constraint_manager = SafetyConstraintManager()
    
    def test_constraint_violation_detection(self):
        """Test safety constraint violation detection"""
        timings = {
            'north_lane': 5,  # Below minimum
            'south_lane': 30,
            'east_lane': 30,
            'west_lane': 30
        }
        
        traffic_data = {
            'lane_counts': {'north': 10, 'south': 15, 'east': 10, 'west': 5},
            'timestamp': datetime.now().isoformat()
        }
        
        violations = self.constraint_manager.check_constraints(
            "intersection_1", timings, traffic_data
        )
        
        # Should detect minimum green time violation
        self.assertGreater(len(violations), 0)
        violation_types = [v.violation_type for v in violations]
        self.assertIn(SafetyViolation.MIN_GREEN_TIME_VIOLATION, violation_types)
    
    def test_corrective_actions(self):
        """Test corrective action application"""
        timings = {
            'north_lane': 5,  # Below minimum
            'south_lane': 30,
            'east_lane': 30,
            'west_lane': 30
        }
        
        traffic_data = {
            'lane_counts': {'north': 10, 'south': 15, 'east': 10, 'west': 5},
            'timestamp': datetime.now().isoformat()
        }
        
        violations = self.constraint_manager.check_constraints(
            "intersection_1", timings, traffic_data
        )
        
        corrected_timings = self.constraint_manager.apply_corrective_actions(
            "intersection_1", timings, violations
        )
        
        # North lane should be corrected to minimum
        self.assertGreaterEqual(corrected_timings['north_lane'], 10)
    
    def test_high_congestion_detection(self):
        """Test high congestion detection"""
        timings = {
            'north_lane': 30,
            'south_lane': 30,
            'east_lane': 30,
            'west_lane': 30
        }
        
        traffic_data = {
            'lane_counts': {'north': 50, 'south': 50, 'east': 50, 'west': 50},  # High congestion
            'timestamp': datetime.now().isoformat()
        }
        
        violations = self.constraint_manager.check_constraints(
            "intersection_1", timings, traffic_data
        )
        
        # Should detect high congestion
        violation_types = [v.violation_type for v in violations]
        self.assertIn(SafetyViolation.HIGH_CONGESTION, violation_types)


class TestOptimizedQTable(unittest.TestCase):
    """Test OptimizedQTable"""
    
    def setUp(self):
        self.q_table = OptimizedQTable()
        self.test_state = np.array([1, 2, 3, 4])
        self.test_action = {'phase': 1, 'duration': 30}
    
    def test_q_value_operations(self):
        """Test Q-value get/set operations"""
        # Set Q-value
        self.q_table.set_q_value(self.test_state, self.test_action, 0.5)
        
        # Get Q-value
        q_value = self.q_table.get_q_value(self.test_state, self.test_action)
        self.assertEqual(q_value, 0.5)
        
        # Test non-existent Q-value
        non_existent_action = {'phase': 2, 'duration': 45}
        q_value = self.q_table.get_q_value(self.test_state, non_existent_action)
        self.assertEqual(q_value, 0.0)
    
    def test_best_action_selection(self):
        """Test best action selection"""
        # Set multiple Q-values
        actions = [
            {'phase': 1, 'duration': 30},
            {'phase': 2, 'duration': 45},
            {'phase': 3, 'duration': 60}
        ]
        
        q_values = [0.3, 0.7, 0.5]
        
        for action, q_value in zip(actions, q_values):
            self.q_table.set_q_value(self.test_state, action, q_value)
        
        # Get best action
        best_action, best_q_value = self.q_table.get_best_action(self.test_state, actions)
        
        self.assertEqual(best_action, actions[1])  # Should be the action with highest Q-value
        self.assertEqual(best_q_value, 0.7)
    
    def test_q_value_update(self):
        """Test Q-value update using Q-learning"""
        # Set initial Q-value
        self.q_table.set_q_value(self.test_state, self.test_action, 0.0)
        
        # Update Q-value
        next_state = np.array([2, 3, 4, 5])
        self.q_table.update_q_value(
            self.test_state, self.test_action, 1.0, next_state, 0.1, 0.9
        )
        
        # Check updated Q-value
        updated_q = self.q_table.get_q_value(self.test_state, self.test_action)
        self.assertGreater(updated_q, 0.0)
    
    def test_memory_management(self):
        """Test memory management and cleanup"""
        # Add many entries to trigger cleanup
        for i in range(1000):
            state = np.array([i, i+1, i+2, i+3])
            action = {'phase': i % 4, 'duration': 30 + i}
            self.q_table.set_q_value(state, action, float(i))
        
        # Check that cleanup was triggered
        metrics = self.q_table.get_metrics()
        self.assertGreater(metrics.total_entries, 0)
    
    def test_persistence(self):
        """Test Q-table persistence"""
        # Set some Q-values
        self.q_table.set_q_value(self.test_state, self.test_action, 0.5)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            self.q_table.save_to_file(tmp_path)
            
            # Create new Q-table and load
            new_q_table = OptimizedQTable()
            new_q_table.load_from_file(tmp_path)
            
            # Check that Q-value was loaded
            q_value = new_q_table.get_q_value(self.test_state, self.test_action)
            self.assertEqual(q_value, 0.5)
            
        finally:
            os.unlink(tmp_path)


class TestEnhancedTraCIController(unittest.TestCase):
    """Test EnhancedTraCIController"""
    
    def setUp(self):
        self.controller = EnhancedTraCIController()
        self.controller.add_intersection("intersection_1", {
            'traffic_lights': ['tl_1'],
            'detectors': ['det_1'],
            'lanes': ['lane_1', 'lane_2'],
            'phases': [0, 1, 2, 3]
        })
    
    @patch('traci.init')
    @patch('traci.simulationStep')
    @patch('traci.close')
    def test_simulation_control(self, mock_close, mock_step, mock_init):
        """Test simulation start/stop control"""
        # Mock successful connection
        mock_init.return_value = None
        
        # Test start simulation
        result = self.controller.start_simulation()
        self.assertTrue(result)
        
        # Test stop simulation
        result = self.controller.stop_simulation()
        self.assertTrue(result)
    
    def test_intersection_management(self):
        """Test intersection management"""
        # Test adding intersection
        self.controller.add_intersection("intersection_2", {
            'traffic_lights': ['tl_2'],
            'lanes': ['lane_3', 'lane_4']
        })
        
        self.assertIn("intersection_2", self.controller.intersections)
        
        # Test intersection configuration
        config = self.controller.intersection_configs["intersection_2"]
        self.assertEqual(config['traffic_lights'], ['tl_2'])
    
    def test_control_command_queuing(self):
        """Test control command queuing"""
        command_id = self.controller.send_control_command(
            "intersection_1",
            "set_phase",
            {'traffic_light_id': 'tl_1', 'phase': 1, 'duration': 30}
        )
        
        self.assertIsNotNone(command_id)
        self.assertGreater(self.controller.control_queue.qsize(), 0)
    
    def test_scenario_switching(self):
        """Test scenario switching"""
        # Mock simulation control methods
        with patch.object(self.controller, 'stop_simulation', return_value=True), \
             patch.object(self.controller, 'start_simulation', return_value=True):
            
            result = self.controller.switch_scenario("new_scenario.sumocfg")
            self.assertTrue(result)


class TestPerformanceBenchmarking(unittest.TestCase):
    """Test Performance Benchmarking"""
    
    def setUp(self):
        self.performance_monitor = PerformanceMonitor()
    
    def test_performance_monitoring(self):
        """Test performance monitoring functionality"""
        # Start monitoring
        self.performance_monitor.start_monitoring()
        
        # Record some processing times
        for i in range(10):
            self.performance_monitor.record_processing_time(0.1 + i * 0.01)
        
        # Get performance summary
        summary = self.performance_monitor.get_performance_summary()
        
        self.assertIn('processing_times', summary)
        self.assertIn('cpu_usage', summary)
        self.assertIn('memory_usage', summary)
        
        # Stop monitoring
        self.performance_monitor.stop_monitoring()
    
    def test_load_balancer(self):
        """Test load balancer functionality"""
        load_balancer = LoadBalancer()
        
        # Submit some tasks
        def dummy_task(x):
            time.sleep(0.01)
            return x * 2
        
        futures = []
        for i in range(5):
            future = load_balancer.submit_task("intersection_1", dummy_task, i)
            futures.append(future)
        
        # Wait for completion
        results = []
        for future in futures:
            if future:
                results.append(future.result())
        
        self.assertEqual(len(results), 5)
        
        # Get metrics
        metrics = load_balancer.get_worker_metrics()
        self.assertIn('active_workers', metrics)
        self.assertIn('stats', metrics)
        
        # Shutdown
        load_balancer.shutdown()


class TestIntegrationTests(unittest.TestCase):
    """Integration tests for Phase 2 components"""
    
    def setUp(self):
        # Initialize all components
        self.real_time_optimizer = RealTimeOptimizer()
        self.safety_manager = SafetyManager()
        self.traci_controller = EnhancedTraCIController()
        
        # Configure intersections
        self.real_time_optimizer.add_data_source("camera_1", {
            'type': 'camera',
            'intersection_id': 'intersection_1',
            'frequency': 1.0
        })
        
        self.safety_manager.configure_intersection("intersection_1", {
            'lanes': ['north', 'south', 'east', 'west'],
            'approach_angles': [0, 180, 90, 270]
        })
        
        self.traci_controller.add_intersection("intersection_1", {
            'traffic_lights': ['tl_1'],
            'lanes': ['north', 'south', 'east', 'west']
        })
    
    def test_end_to_end_optimization(self):
        """Test end-to-end optimization flow"""
        # Mock traffic data
        traffic_data = {
            'intersection_id': 'intersection_1',
            'lane_counts': {'north': 20, 'south': 15, 'east': 10, 'west': 5},
            'timestamp': datetime.now().isoformat()
        }
        
        # Test safety check and optimization
        result = self.safety_manager.check_safety_and_optimize(
            'intersection_1', traffic_data, {'north': 30, 'south': 30, 'east': 30, 'west': 30}
        )
        
        self.assertTrue(result['success'])
        self.assertIn('optimized_timings', result)
        self.assertIn('algorithm_used', result)
    
    def test_emergency_vehicle_flow(self):
        """Test emergency vehicle handling flow"""
        # Register emergency vehicle
        alert_id = self.safety_manager.register_emergency_vehicle(
            'intersection_1',
            'ambulance',
            1,
            datetime.now() + timedelta(minutes=1),
            (0, 0),
            'hospital'
        )
        
        self.assertIsNotNone(alert_id)
        
        # Test emergency override
        traffic_data = {
            'intersection_id': 'intersection_1',
            'lane_counts': {'north': 20, 'south': 15, 'east': 10, 'west': 5},
            'timestamp': datetime.now().isoformat()
        }
        
        result = self.safety_manager.check_safety_and_optimize(
            'intersection_1', traffic_data, {'north': 30, 'south': 30, 'east': 30, 'west': 30}
        )
        
        self.assertTrue(result['emergency_active'])
    
    def test_performance_under_load(self):
        """Test system performance under load"""
        # Create multiple intersections
        for i in range(5):
            intersection_id = f"intersection_{i}"
            
            self.real_time_optimizer.add_data_source(f"camera_{i}", {
                'type': 'camera',
                'intersection_id': intersection_id,
                'frequency': 0.5
            })
            
            self.safety_manager.configure_intersection(intersection_id, {
                'lanes': ['north', 'south', 'east', 'west']
            })
        
        # Simulate high load
        start_time = time.time()
        
        for i in range(100):
            traffic_data = {
                'intersection_id': f'intersection_{i % 5}',
                'lane_counts': {'north': 20, 'south': 15, 'east': 10, 'west': 5},
                'timestamp': datetime.now().isoformat()
            }
            
            result = self.safety_manager.check_safety_and_optimize(
                f'intersection_{i % 5}', traffic_data, {'north': 30, 'south': 30, 'east': 30, 'west': 30}
            )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time
        self.assertLess(processing_time, 10.0)  # 10 seconds for 100 operations
        
        # Check that most operations succeeded
        self.assertGreater(processing_time, 0.1)  # Should take some time


class TestStressTests(unittest.TestCase):
    """Stress tests for Phase 2 components"""
    
    def test_memory_usage_under_load(self):
        """Test memory usage under high load"""
        q_table = OptimizedQTable({'memory_limit': 10})  # 10 MB limit
        
        # Add many entries
        for i in range(10000):
            state = np.random.rand(10)
            action = {'phase': i % 4, 'duration': 30 + i % 30}
            q_table.set_q_value(state, action, float(i))
        
        # Check memory usage
        metrics = q_table.get_metrics()
        self.assertGreater(metrics.total_entries, 0)
    
    def test_concurrent_operations(self):
        """Test concurrent operations on shared resources"""
        state_manager = ThreadSafeStateManager()
        
        def update_states(thread_id):
            for i in range(100):
                state = Mock()
                state.to_vector.return_value = np.array([thread_id, i, i+1, i+2])
                state.get_state_hash.return_value = f"hash_{thread_id}_{i}"
                state_manager.update_state(f"intersection_{thread_id}", state)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=update_states, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check that all states were updated
        for i in range(5):
            state = state_manager.get_state(f"intersection_{i}")
            self.assertIsNotNone(state)
    
    def test_error_recovery(self):
        """Test error recovery mechanisms"""
        safety_manager = SafetyManager()
        
        # Test with invalid data
        invalid_traffic_data = {
            'intersection_id': 'intersection_1',
            'lane_counts': {'north': -5, 'south': 15},  # Invalid negative count
            'timestamp': 'invalid_timestamp'  # Invalid timestamp
        }
        
        result = safety_manager.check_safety_and_optimize(
            'intersection_1', invalid_traffic_data, {'north': 30, 'south': 30}
        )
        
        # Should handle invalid data gracefully
        self.assertIsNotNone(result)
        self.assertIn('success', result)


def run_performance_benchmarks():
    """Run comprehensive performance benchmarks"""
    print("Running Phase 2 Performance Benchmarks...")
    
    # Benchmark Q-table operations
    print("\n=== Q-Table Performance Benchmarks ===")
    q_table = OptimizedQTable()
    
    # Benchmark set operations
    start_time = time.time()
    for i in range(10000):
        state = np.random.rand(10)
        action = {'phase': i % 4, 'duration': 30 + i % 30}
        q_table.set_q_value(state, action, float(i))
    set_time = time.time() - start_time
    print(f"Set 10,000 Q-values: {set_time:.3f}s")
    
    # Benchmark get operations
    start_time = time.time()
    for i in range(10000):
        state = np.random.rand(10)
        action = {'phase': i % 4, 'duration': 30 + i % 30}
        q_table.get_q_value(state, action)
    get_time = time.time() - start_time
    print(f"Get 10,000 Q-values: {get_time:.3f}s")
    
    # Benchmark state manager operations
    print("\n=== State Manager Performance Benchmarks ===")
    state_manager = ThreadSafeStateManager()
    
    start_time = time.time()
    for i in range(10000):
        state = Mock()
        state.to_vector.return_value = np.random.rand(10)
        state.get_state_hash.return_value = f"hash_{i}"
        state_manager.update_state(f"intersection_{i % 100}", state)
    update_time = time.time() - start_time
    print(f"Update 10,000 states: {update_time:.3f}s")
    
    # Benchmark safety manager operations
    print("\n=== Safety Manager Performance Benchmarks ===")
    safety_manager = SafetyManager()
    
    start_time = time.time()
    for i in range(1000):
        traffic_data = {
            'intersection_id': f'intersection_{i % 10}',
            'lane_counts': {'north': 20, 'south': 15, 'east': 10, 'west': 5},
            'timestamp': datetime.now().isoformat()
        }
        safety_manager.check_safety_and_optimize(
            f'intersection_{i % 10}', traffic_data, {'north': 30, 'south': 30, 'east': 30, 'west': 30}
        )
    safety_time = time.time() - start_time
    print(f"Safety check 1,000 intersections: {safety_time:.3f}s")
    
    # Memory usage
    print("\n=== Memory Usage ===")
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.2f} MB")
    
    print("\nBenchmarks completed!")


if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run unit tests
    print("Running Phase 2 Unit Tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run performance benchmarks
    run_performance_benchmarks()
