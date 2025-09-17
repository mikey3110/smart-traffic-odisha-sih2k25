#!/usr/bin/env python3
"""
Unit tests for RobustTraCIController

Tests error scenarios, connection handling, and control logic
"""

import unittest
import unittest.mock as mock
import time
import threading
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from simulation.traci_controller import (
    RobustTraCIController, 
    ConnectionState, 
    SignalPhase, 
    IntersectionConfig,
    TrafficData,
    ControlCommand,
    WebsterFormula
)

class TestWebsterFormula(unittest.TestCase):
    """Test Webster's formula calculations"""
    
    def test_calculate_cycle_time(self):
        """Test cycle time calculation"""
        # Test with normal flows
        approach_flows = {
            'north_approach': 300,
            'south_approach': 280,
            'east_approach': 250,
            'west_approach': 270
        }
        
        cycle_time = WebsterFormula.calculate_cycle_time(approach_flows)
        
        # Should be between 40 and 120 seconds
        self.assertGreaterEqual(cycle_time, 40.0)
        self.assertLessEqual(cycle_time, 120.0)
    
    def test_calculate_cycle_time_empty(self):
        """Test cycle time calculation with empty flows"""
        cycle_time = WebsterFormula.calculate_cycle_time({})
        self.assertEqual(cycle_time, 60.0)
    
    def test_calculate_green_times(self):
        """Test green time calculation"""
        approach_flows = {
            'north_approach': 300,
            'south_approach': 200,
            'east_approach': 100,
            'west_approach': 100
        }
        
        cycle_time = 60.0
        green_times = WebsterFormula.calculate_green_times(approach_flows, cycle_time)
        
        # Should have green times for all approaches
        self.assertEqual(len(green_times), 4)
        
        # North approach should have highest green time (highest flow)
        self.assertGreater(green_times['north_approach'], green_times['east_approach'])
        
        # All green times should be between 10 and 60 seconds
        for approach, green_time in green_times.items():
            self.assertGreaterEqual(green_time, 10.0)
            self.assertLessEqual(green_time, 60.0)

class TestTraCIController(unittest.TestCase):
    """Test TraCI controller functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
        
        # Create test configuration
        self.test_config = {
            'max_retries': 3,
            'retry_delay': 0.1,
            'port': 8813,
            'host': 'localhost',
            'monitoring_interval': 0.1,
            'control_interval': 0.1,
            'log_level': 'DEBUG',
            'intersections': [
                {
                    'id': 'test_intersection',
                    'phases': ['GGrrGGrr', 'yyrryyrr', 'rrrrrrrr', 'rrGGrrGG', 'rryyrryy'],
                    'min_green_time': 5.0,
                    'max_green_time': 30.0,
                    'yellow_time': 2.0,
                    'all_red_time': 1.0,
                    'approach_edges': ['north_approach', 'south_approach'],
                    'exit_edges': ['north_exit', 'south_exit'],
                    'detectors': []
                }
            ]
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
        
        # Create controller
        self.controller = RobustTraCIController(self.config_file)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'controller'):
            self.controller.stop()
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test controller initialization"""
        self.assertEqual(self.controller.connection_state, ConnectionState.DISCONNECTED)
        self.assertEqual(len(self.controller.intersections), 1)
        self.assertIn('test_intersection', self.controller.intersections)
        self.assertFalse(self.controller.running)
    
    def test_config_loading(self):
        """Test configuration loading"""
        self.assertEqual(self.controller.config['max_retries'], 3)
        self.assertEqual(self.controller.config['retry_delay'], 0.1)
        self.assertEqual(len(self.controller.config['intersections']), 1)
    
    def test_intersection_initialization(self):
        """Test intersection configuration initialization"""
        intersection = self.controller.intersections['test_intersection']
        
        self.assertEqual(intersection.id, 'test_intersection')
        self.assertEqual(intersection.min_green_time, 5.0)
        self.assertEqual(intersection.max_green_time, 30.0)
        self.assertEqual(len(intersection.phases), 5)
        self.assertEqual(len(intersection.approach_edges), 2)
    
    @patch('traci.init')
    def test_successful_connection(self, mock_traci_init):
        """Test successful connection to SUMO"""
        mock_traci_init.return_value = None
        
        result = self.controller.connect()
        
        self.assertTrue(result)
        self.assertEqual(self.controller.connection_state, ConnectionState.CONNECTED)
        mock_traci_init.assert_called_once_with(8813, host='localhost')
    
    @patch('traci.init')
    def test_connection_failure(self, mock_traci_init):
        """Test connection failure handling"""
        mock_traci_init.side_effect = Exception("Connection failed")
        
        result = self.controller.connect()
        
        self.assertFalse(result)
        self.assertEqual(self.controller.connection_state, ConnectionState.ERROR)
    
    @patch('traci.init')
    def test_reconnection_logic(self, mock_traci_init):
        """Test reconnection logic"""
        # First call fails, second succeeds
        mock_traci_init.side_effect = [Exception("Connection failed"), None]
        
        # First connection attempt should fail
        result1 = self.controller.connect()
        self.assertFalse(result1)
        self.assertEqual(self.controller.connection_retries, 1)
        
        # Reconnection should succeed
        result2 = self.controller.reconnect()
        self.assertTrue(result2)
        self.assertEqual(self.controller.connection_retries, 2)
    
    @patch('traci.init')
    def test_max_retries_exceeded(self, mock_traci_init):
        """Test max retries exceeded"""
        mock_traci_init.side_effect = Exception("Connection failed")
        
        # Exceed max retries
        for _ in range(4):  # max_retries + 1
            self.controller.reconnect()
        
        self.assertEqual(self.controller.connection_retries, 4)
        self.assertEqual(self.controller.connection_state, ConnectionState.ERROR)
    
    def test_control_command_creation(self):
        """Test control command creation and queuing"""
        # Test set phase command
        self.controller.set_phase('test_intersection', 'GGrrGGrr', priority=1)
        
        # Check command was queued
        self.assertFalse(self.controller.control_queue.empty())
        
        # Test extend phase command
        self.controller.extend_phase('test_intersection', 10.0, priority=2)
        
        # Test emergency override
        self.controller.emergency_override('test_intersection')
        
        # Should have 3 commands in queue
        self.assertEqual(self.controller.control_queue.qsize(), 3)
    
    def test_phase_transition_logic(self):
        """Test phase transition logic"""
        intersection_id = 'test_intersection'
        
        # Test phase change conditions
        self.controller.current_phases[intersection_id] = 'GGrrGGrr'
        self.controller.phase_timers[intersection_id] = 0.0
        
        # Should not change phase immediately (min green time)
        should_change = self.controller._should_change_phase(intersection_id, 0.0)
        self.assertFalse(should_change)
        
        # Should change after max green time
        self.controller.phase_timers[intersection_id] = 35.0  # > max_green_time
        should_change = self.controller._should_change_phase(intersection_id, 35.0)
        self.assertTrue(should_change)
    
    def test_emergency_override(self):
        """Test emergency override functionality"""
        intersection_id = 'test_intersection'
        
        # Test emergency override
        self.controller.emergency_override(intersection_id)
        
        # Check command was queued with high priority
        self.assertFalse(self.controller.control_queue.empty())
        
        # Test clearing emergency override
        self.controller.emergency_intersections.add(intersection_id)
        self.controller.clear_emergency_override(intersection_id)
        self.assertNotIn(intersection_id, self.controller.emergency_intersections)
    
    def test_webster_optimization(self):
        """Test Webster's formula optimization"""
        intersection_id = 'test_intersection'
        
        # Create mock traffic data
        traffic_data = TrafficData(
            intersection_id=intersection_id,
            timestamp=0.0,
            vehicle_counts={'north_approach': 10, 'south_approach': 8},
            lane_occupancy={'north_approach': 0.5, 'south_approach': 0.4},
            queue_lengths={'north_approach': 5, 'south_approach': 4},
            waiting_times={'north_approach': 10.0, 'south_approach': 8.0},
            current_phase='GGrrGGrr',
            phase_remaining_time=15.0
        )
        
        self.controller.traffic_data[intersection_id] = traffic_data
        
        # Test optimization
        result = self.controller.optimize_with_webster(intersection_id)
        
        self.assertIsNotNone(result)
        self.assertIn('cycle_time', result)
        self.assertIn('green_times', result)
        self.assertGreater(result['cycle_time'], 0)
    
    def test_statistics_collection(self):
        """Test statistics collection"""
        stats = self.controller.get_statistics()
        
        self.assertIn('connection_state', stats)
        self.assertIn('uptime', stats)
        self.assertIn('commands_sent', stats)
        self.assertIn('commands_failed', stats)
        self.assertIn('reconnections', stats)
        self.assertIn('errors', stats)
        self.assertIn('intersections', stats)
        self.assertIn('emergency_intersections', stats)
        self.assertIn('queue_size', stats)
    
    def test_health_check(self):
        """Test health check functionality"""
        # Test when disconnected
        health = self.controller.health_check()
        self.assertFalse(health)
        
        # Test when connected (mocked)
        with patch('traci.simulation.getTime') as mock_get_time:
            mock_get_time.return_value = 0.0
            self.controller.connection_state = ConnectionState.CONNECTED
            
            health = self.controller.health_check()
            self.assertTrue(health)
    
    @patch('traci.init')
    @patch('traci.close')
    def test_start_stop_cycle(self, mock_traci_close, mock_traci_init):
        """Test controller start/stop cycle"""
        mock_traci_init.return_value = None
        
        # Start controller
        self.controller.start()
        
        # Check it's running
        self.assertTrue(self.controller.running)
        self.assertIsNotNone(self.controller.control_thread)
        self.assertIsNotNone(self.controller.monitoring_thread)
        
        # Stop controller
        self.controller.stop()
        
        # Check it's stopped
        self.assertFalse(self.controller.running)
        mock_traci_close.assert_called_once()

class TestErrorScenarios(unittest.TestCase):
    """Test error scenarios and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
        
        # Create minimal config
        config = {
            'max_retries': 2,
            'retry_delay': 0.1,
            'port': 8813,
            'host': 'localhost',
            'intersections': []
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
        
        self.controller = RobustTraCIController(self.config_file)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'controller'):
            self.controller.stop()
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_invalid_config_file(self):
        """Test handling of invalid config file"""
        invalid_config_file = os.path.join(self.temp_dir, 'invalid.json')
        
        with open(invalid_config_file, 'w') as f:
            f.write('invalid json content')
        
        controller = RobustTraCIController(invalid_config_file)
        
        # Should use default config
        self.assertEqual(controller.config['max_retries'], 5)
        self.assertEqual(controller.config['port'], 8813)
    
    def test_missing_config_file(self):
        """Test handling of missing config file"""
        missing_config_file = os.path.join(self.temp_dir, 'missing.json')
        
        controller = RobustTraCIController(missing_config_file)
        
        # Should use default config
        self.assertEqual(controller.config['max_retries'], 5)
        self.assertEqual(controller.config['port'], 8813)
    
    def test_unknown_intersection_commands(self):
        """Test handling of commands for unknown intersections"""
        # Test set phase for unknown intersection
        self.controller.set_phase('unknown_intersection', 'GGrrGGrr')
        
        # Command should be queued but will fail when executed
        self.assertFalse(self.controller.control_queue.empty())
    
    def test_control_loop_error_handling(self):
        """Test error handling in control loop"""
        with patch.object(self.controller, '_process_control_commands') as mock_process:
            mock_process.side_effect = Exception("Control loop error")
            
            # Start controller
            self.controller.start()
            time.sleep(0.2)  # Let it run briefly
            self.controller.stop()
            
            # Should have recorded error
            self.assertGreater(self.controller.stats['errors'], 0)
    
    def test_monitoring_loop_error_handling(self):
        """Test error handling in monitoring loop"""
        with patch.object(self.controller, '_collect_traffic_data') as mock_collect:
            mock_collect.side_effect = Exception("Monitoring loop error")
            
            # Start controller
            self.controller.start()
            time.sleep(0.2)  # Let it run briefly
            self.controller.stop()
            
            # Should have recorded error
            self.assertGreater(self.controller.stats['errors'], 0)
    
    @patch('traci.init')
    def test_traci_connection_drop(self, mock_traci_init):
        """Test handling of TraCI connection drops"""
        # First connection succeeds, then fails
        mock_traci_init.side_effect = [None, Exception("Connection lost")]
        
        # Connect successfully
        self.controller.connect()
        self.assertEqual(self.controller.connection_state, ConnectionState.CONNECTED)
        
        # Simulate connection drop by trying to reconnect
        result = self.controller.reconnect()
        self.assertFalse(result)
        self.assertEqual(self.controller.connection_state, ConnectionState.ERROR)

class TestIntegration(unittest.TestCase):
    """Integration tests with mocked SUMO"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
        
        # Create test configuration
        config = {
            'max_retries': 2,
            'retry_delay': 0.1,
            'port': 8813,
            'host': 'localhost',
            'monitoring_interval': 0.1,
            'control_interval': 0.1,
            'intersections': [
                {
                    'id': 'center',
                    'phases': ['GGrrGGrr', 'yyrryyrr', 'rrrrrrrr', 'rrGGrrGG', 'rryyrryy'],
                    'min_green_time': 5.0,
                    'max_green_time': 30.0,
                    'yellow_time': 2.0,
                    'all_red_time': 1.0,
                    'approach_edges': ['north_approach', 'south_approach'],
                    'exit_edges': ['north_exit', 'south_exit'],
                    'detectors': []
                }
            ]
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
        
        self.controller = RobustTraCIController(self.config_file)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if hasattr(self, 'controller'):
            self.controller.stop()
        
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('traci.init')
    @patch('traci.close')
    @patch('traci.trafficlight.setRedYellowGreenState')
    @patch('traci.simulation.getTime')
    @patch('traci.edge.getLastStepVehicleNumber')
    @patch('traci.edge.getLastStepOccupancy')
    @patch('traci.edge.getLastStepHaltingNumber')
    @patch('traci.edge.getWaitingTime')
    def test_full_operation_cycle(self, mock_waiting_time, mock_halting, mock_occupancy, 
                                 mock_vehicle_count, mock_get_time, mock_set_phase, 
                                 mock_close, mock_init):
        """Test full operation cycle with mocked SUMO"""
        # Setup mocks
        mock_init.return_value = None
        mock_get_time.return_value = 0.0
        mock_vehicle_count.return_value = 5
        mock_occupancy.return_value = 0.3
        mock_halting.return_value = 2
        mock_waiting_time.return_value = 10.0
        
        # Start controller
        self.controller.start()
        
        # Let it run briefly
        time.sleep(0.5)
        
        # Check that monitoring collected data
        traffic_data = self.controller.get_traffic_data('center')
        self.assertIsNotNone(traffic_data)
        self.assertEqual(traffic_data.intersection_id, 'center')
        
        # Test phase change
        self.controller.set_phase('center', 'rrGGrrGG')
        time.sleep(0.1)  # Let command process
        
        # Stop controller
        self.controller.stop()
        
        # Verify mocks were called
        mock_init.assert_called_once()
        mock_close.assert_called_once()
        mock_get_time.assert_called()

if __name__ == '__main__':
    # Create logs directory for tests
    os.makedirs('logs', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)
