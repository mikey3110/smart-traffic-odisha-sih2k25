#!/usr/bin/env python3
"""
Robust TraCI Controller for Multi-Intersection Traffic Signal Control

This module provides a fault-tolerant Python TraCI controller for managing
real-time traffic signal control through SUMO with multi-intersection
synchronization, error handling, and fallback mechanisms.

Author: Smart Traffic Management System Team
Date: 2025
"""

import traci
import time
import logging
import threading
import queue
import json
import math
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import traceback
from datetime import datetime
import socket
import subprocess
import os
import signal
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/traci_controller.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConnectionState(Enum):
    """TraCI connection states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"

class SignalPhase(Enum):
    """Traffic signal phases"""
    NORTH_SOUTH_GREEN = "GGrrGGrr"
    NORTH_SOUTH_YELLOW = "yyrryyrr"
    ALL_RED = "rrrrrrrr"
    EAST_WEST_GREEN = "rrGGrrGG"
    EAST_WEST_YELLOW = "rryyrryy"

@dataclass
class IntersectionConfig:
    """Configuration for a single intersection"""
    id: str
    phases: List[SignalPhase]
    min_green_time: float = 10.0
    max_green_time: float = 60.0
    yellow_time: float = 3.0
    all_red_time: float = 2.0
    approach_edges: List[str] = None
    exit_edges: List[str] = None
    detectors: List[str] = None

@dataclass
class TrafficData:
    """Real-time traffic data for an intersection"""
    intersection_id: str
    timestamp: float
    vehicle_counts: Dict[str, int]
    lane_occupancy: Dict[str, float]
    queue_lengths: Dict[str, int]
    waiting_times: Dict[str, float]
    current_phase: str
    phase_remaining_time: float

@dataclass
class ControlCommand:
    """Command for traffic signal control"""
    intersection_id: str
    action: str  # 'set_phase', 'extend_phase', 'emergency_override'
    phase: Optional[str] = None
    duration: Optional[float] = None
    priority: int = 0  # Higher priority for emergency commands

class WebsterFormula:
    """Webster's formula for traffic signal timing optimization"""
    
    @staticmethod
    def calculate_cycle_time(approach_flows: Dict[str, float], 
                           lost_time: float = 4.0) -> float:
        """
        Calculate optimal cycle time using Webster's formula
        
        Args:
            approach_flows: Dictionary of approach flows (vehicles/hour)
            lost_time: Lost time per cycle (seconds)
            
        Returns:
            Optimal cycle time in seconds
        """
        if not approach_flows:
            return 60.0
        
        # Calculate critical flow ratio
        critical_ratio = sum(approach_flows.values()) / 3600.0  # Convert to veh/sec
        
        if critical_ratio == 0:
            return 60.0
        
        # Webster's formula: C = (1.5 * L + 5) / (1 - Y)
        # where L = lost time, Y = critical flow ratio
        cycle_time = (1.5 * lost_time + 5) / (1 - critical_ratio)
        
        # Constrain between 40 and 120 seconds
        return max(40.0, min(120.0, cycle_time))
    
    @staticmethod
    def calculate_green_times(approach_flows: Dict[str, float], 
                            cycle_time: float,
                            lost_time: float = 4.0) -> Dict[str, float]:
        """
        Calculate green times for each approach
        
        Args:
            approach_flows: Dictionary of approach flows
            cycle_time: Total cycle time
            lost_time: Lost time per cycle
            
        Returns:
            Dictionary of green times for each approach
        """
        if not approach_flows:
            return {}
        
        total_flow = sum(approach_flows.values())
        if total_flow == 0:
            return {approach: 10.0 for approach in approach_flows.keys()}
        
        green_times = {}
        available_time = cycle_time - lost_time
        
        for approach, flow in approach_flows.items():
            # Proportional green time based on flow
            green_time = (flow / total_flow) * available_time
            # Constrain between 10 and 60 seconds
            green_times[approach] = max(10.0, min(60.0, green_time))
        
        return green_times

class RobustTraCIController:
    """
    Robust TraCI controller for multi-intersection traffic signal control
    """
    
    def __init__(self, config_file: str = "config/traci_config.json"):
        """
        Initialize the TraCI controller
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        # Connection management
        self.connection_state = ConnectionState.DISCONNECTED
        self.connection_retries = 0
        self.max_retries = self.config.get('max_retries', 5)
        self.retry_delay = self.config.get('retry_delay', 5.0)
        self.port = self.config.get('port', 8813)
        self.host = self.config.get('host', 'localhost')
        
        # Intersection management
        self.intersections: Dict[str, IntersectionConfig] = {}
        self.current_phases: Dict[str, str] = {}
        self.phase_timers: Dict[str, float] = {}
        self.traffic_data: Dict[str, TrafficData] = {}
        
        # Control management
        self.control_queue = queue.PriorityQueue()
        self.emergency_override = False
        self.emergency_intersections: set = set()
        
        # Threading
        self.control_thread = None
        self.monitoring_thread = None
        self.running = False
        self.lock = threading.RLock()
        
        # Statistics
        self.stats = {
            'commands_sent': 0,
            'commands_failed': 0,
            'reconnections': 0,
            'errors': 0,
            'start_time': time.time()
        }
        
        # Initialize intersections
        self._initialize_intersections()
        
        logger.info("TraCI Controller initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'max_retries': 5,
            'retry_delay': 5.0,
            'port': 8813,
            'host': 'localhost',
            'monitoring_interval': 1.0,
            'control_interval': 0.1,
            'log_level': 'INFO',
            'intersections': []
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return {**default_config, **config}
            else:
                logger.warning(f"Configuration file {self.config_file} not found, using defaults")
                return default_config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return default_config
    
    def _initialize_intersections(self):
        """Initialize intersection configurations"""
        for intersection_config in self.config.get('intersections', []):
            intersection = IntersectionConfig(
                id=intersection_config['id'],
                phases=[SignalPhase(phase) for phase in intersection_config.get('phases', [])],
                min_green_time=intersection_config.get('min_green_time', 10.0),
                max_green_time=intersection_config.get('max_green_time', 60.0),
                yellow_time=intersection_config.get('yellow_time', 3.0),
                all_red_time=intersection_config.get('all_red_time', 2.0),
                approach_edges=intersection_config.get('approach_edges', []),
                exit_edges=intersection_config.get('exit_edges', []),
                detectors=intersection_config.get('detectors', [])
            )
            self.intersections[intersection.id] = intersection
            self.current_phases[intersection.id] = "GGrrGGrr"
            self.phase_timers[intersection.id] = 0.0
        
        logger.info(f"Initialized {len(self.intersections)} intersections")
    
    def connect(self) -> bool:
        """
        Connect to SUMO via TraCI
        
        Returns:
            True if connection successful, False otherwise
        """
        with self.lock:
            if self.connection_state == ConnectionState.CONNECTED:
                return True
            
            self.connection_state = ConnectionState.CONNECTING
            logger.info(f"Connecting to SUMO at {self.host}:{self.port}")
            
            try:
                traci.init(self.port, host=self.host)
                self.connection_state = ConnectionState.CONNECTED
                self.connection_retries = 0
                logger.info("Successfully connected to SUMO")
                return True
                
            except Exception as e:
                self.connection_state = ConnectionState.ERROR
                logger.error(f"Failed to connect to SUMO: {e}")
                return False
    
    def disconnect(self):
        """Disconnect from SUMO"""
        with self.lock:
            if self.connection_state == ConnectionState.CONNECTED:
                try:
                    traci.close()
                    logger.info("Disconnected from SUMO")
                except Exception as e:
                    logger.error(f"Error disconnecting from SUMO: {e}")
                finally:
                    self.connection_state = ConnectionState.DISCONNECTED
    
    def reconnect(self) -> bool:
        """
        Reconnect to SUMO with retry logic
        
        Returns:
            True if reconnection successful, False otherwise
        """
        with self.lock:
            if self.connection_retries >= self.max_retries:
                logger.error(f"Max reconnection attempts ({self.max_retries}) exceeded")
                return False
            
            self.connection_state = ConnectionState.RECONNECTING
            self.connection_retries += 1
            self.stats['reconnections'] += 1
            
            logger.info(f"Reconnection attempt {self.connection_retries}/{self.max_retries}")
            
            # Disconnect if connected
            if self.connection_state == ConnectionState.CONNECTED:
                self.disconnect()
            
            # Wait before retry
            time.sleep(self.retry_delay)
            
            # Attempt reconnection
            if self.connect():
                logger.info("Reconnection successful")
                return True
            else:
                logger.warning(f"Reconnection attempt {self.connection_retries} failed")
                return False
    
    def start(self):
        """Start the controller"""
        if self.running:
            logger.warning("Controller is already running")
            return
        
        if not self.connect():
            logger.error("Failed to connect to SUMO, cannot start controller")
            return
        
        self.running = True
        
        # Start control thread
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("TraCI Controller started")
    
    def stop(self):
        """Stop the controller"""
        if not self.running:
            return
        
        self.running = False
        
        # Wait for threads to finish
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=5.0)
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        self.disconnect()
        logger.info("TraCI Controller stopped")
    
    def _control_loop(self):
        """Main control loop for traffic signal management"""
        while self.running:
            try:
                if self.connection_state != ConnectionState.CONNECTED:
                    if not self.reconnect():
                        time.sleep(1.0)
                        continue
                
                # Process control commands
                self._process_control_commands()
                
                # Update signal phases
                self._update_signal_phases()
                
                # Check for phase transitions
                self._check_phase_transitions()
                
                time.sleep(self.config.get('control_interval', 0.1))
                
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                self.stats['errors'] += 1
                time.sleep(1.0)
    
    def _monitoring_loop(self):
        """Monitoring loop for traffic data collection"""
        while self.running:
            try:
                if self.connection_state == ConnectionState.CONNECTED:
                    self._collect_traffic_data()
                
                time.sleep(self.config.get('monitoring_interval', 1.0))
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.stats['errors'] += 1
                time.sleep(1.0)
    
    def _process_control_commands(self):
        """Process queued control commands"""
        try:
            while not self.control_queue.empty():
                priority, command = self.control_queue.get_nowait()
                
                if self._execute_command(command):
                    self.stats['commands_sent'] += 1
                    logger.debug(f"Executed command: {command.action} for {command.intersection_id}")
                else:
                    self.stats['commands_failed'] += 1
                    logger.warning(f"Failed to execute command: {command.action} for {command.intersection_id}")
                
        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error processing control commands: {e}")
            self.stats['errors'] += 1
    
    def _execute_command(self, command: ControlCommand) -> bool:
        """
        Execute a control command
        
        Args:
            command: Control command to execute
            
        Returns:
            True if command executed successfully, False otherwise
        """
        try:
            if command.action == 'set_phase':
                return self._set_signal_phase(command.intersection_id, command.phase)
            elif command.action == 'extend_phase':
                return self._extend_phase(command.intersection_id, command.duration)
            elif command.action == 'emergency_override':
                return self._emergency_override(command.intersection_id)
            else:
                logger.warning(f"Unknown command action: {command.action}")
                return False
                
        except Exception as e:
            logger.error(f"Error executing command {command.action}: {e}")
            return False
    
    def _set_signal_phase(self, intersection_id: str, phase: str) -> bool:
        """
        Set traffic signal phase for an intersection
        
        Args:
            intersection_id: ID of the intersection
            phase: Signal phase to set
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if intersection_id not in self.intersections:
                logger.warning(f"Unknown intersection: {intersection_id}")
                return False
            
            traci.trafficlight.setRedYellowGreenState(intersection_id, phase)
            self.current_phases[intersection_id] = phase
            self.phase_timers[intersection_id] = 0.0
            
            logger.info(f"Set phase {phase} for intersection {intersection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting signal phase: {e}")
            return False
    
    def _extend_phase(self, intersection_id: str, duration: float) -> bool:
        """
        Extend current phase duration
        
        Args:
            intersection_id: ID of the intersection
            duration: Duration to extend (seconds)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if intersection_id not in self.intersections:
                return False
            
            # Extend phase by updating timer
            self.phase_timers[intersection_id] += duration
            
            logger.info(f"Extended phase for {intersection_id} by {duration}s")
            return True
            
        except Exception as e:
            logger.error(f"Error extending phase: {e}")
            return False
    
    def _emergency_override(self, intersection_id: str) -> bool:
        """
        Set emergency override for an intersection
        
        Args:
            intersection_id: ID of the intersection
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if intersection_id not in self.intersections:
                return False
            
            # Set all green for emergency
            traci.trafficlight.setRedYellowGreenState(intersection_id, "GGGGGGGG")
            self.emergency_intersections.add(intersection_id)
            self.emergency_override = True
            
            logger.warning(f"Emergency override activated for {intersection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting emergency override: {e}")
            return False
    
    def _update_signal_phases(self):
        """Update signal phases based on current state"""
        current_time = traci.simulation.getTime()
        
        for intersection_id, intersection in self.intersections.items():
            if intersection_id in self.emergency_intersections:
                continue  # Skip emergency intersections
            
            # Check if phase needs to change
            if self._should_change_phase(intersection_id, current_time):
                self._change_phase(intersection_id)
    
    def _should_change_phase(self, intersection_id: str, current_time: float) -> bool:
        """
        Check if phase should change for an intersection
        
        Args:
            intersection_id: ID of the intersection
            current_time: Current simulation time
            
        Returns:
            True if phase should change, False otherwise
        """
        intersection = self.intersections[intersection_id]
        current_phase = self.current_phases[intersection_id]
        phase_time = self.phase_timers[intersection_id]
        
        # Check minimum green time
        if current_phase in ["GGrrGGrr", "rrGGrrGG"] and phase_time < intersection.min_green_time:
            return False
        
        # Check maximum green time
        if current_phase in ["GGrrGGrr", "rrGGrrGG"] and phase_time >= intersection.max_green_time:
            return True
        
        # Check yellow time
        if current_phase in ["yyrryyrr", "rryyrryy"] and phase_time >= intersection.yellow_time:
            return True
        
        # Check all red time
        if current_phase == "rrrrrrrr" and phase_time >= intersection.all_red_time:
            return True
        
        return False
    
    def _change_phase(self, intersection_id: str):
        """Change phase for an intersection"""
        intersection = self.intersections[intersection_id]
        current_phase = self.current_phases[intersection_id]
        
        # Determine next phase
        if current_phase == "GGrrGGrr":  # North-South Green
            next_phase = "yyrryyrr"  # North-South Yellow
        elif current_phase == "yyrryyrr":  # North-South Yellow
            next_phase = "rrrrrrrr"  # All Red
        elif current_phase == "rrrrrrrr":  # All Red
            next_phase = "rrGGrrGG"  # East-West Green
        elif current_phase == "rrGGrrGG":  # East-West Green
            next_phase = "rryyrryy"  # East-West Yellow
        elif current_phase == "rryyrryy":  # East-West Yellow
            next_phase = "rrrrrrrr"  # All Red
        else:
            next_phase = "GGrrGGrr"  # Default to North-South Green
        
        # Set new phase
        self._set_signal_phase(intersection_id, next_phase)
        self.phase_timers[intersection_id] = 0.0
        
        logger.info(f"Changed phase for {intersection_id} from {current_phase} to {next_phase}")
    
    def _check_phase_transitions(self):
        """Check for automatic phase transitions"""
        current_time = traci.simulation.getTime()
        
        for intersection_id in self.intersections:
            if intersection_id in self.emergency_intersections:
                continue
            
            # Update phase timer
            self.phase_timers[intersection_id] += self.config.get('control_interval', 0.1)
    
    def _collect_traffic_data(self):
        """Collect real-time traffic data from all intersections"""
        current_time = traci.simulation.getTime()
        
        for intersection_id, intersection in self.intersections.items():
            try:
                # Get vehicle counts for approach edges
                vehicle_counts = {}
                for edge in intersection.approach_edges:
                    vehicle_counts[edge] = traci.edge.getLastStepVehicleNumber(edge)
                
                # Get lane occupancy
                lane_occupancy = {}
                for edge in intersection.approach_edges:
                    occupancy = traci.edge.getLastStepOccupancy(edge)
                    lane_occupancy[edge] = occupancy
                
                # Get queue lengths
                queue_lengths = {}
                for edge in intersection.approach_edges:
                    queue_length = traci.edge.getLastStepHaltingNumber(edge)
                    queue_lengths[edge] = queue_length
                
                # Get waiting times
                waiting_times = {}
                for edge in intersection.approach_edges:
                    waiting_time = traci.edge.getWaitingTime(edge)
                    waiting_times[edge] = waiting_time
                
                # Create traffic data object
                traffic_data = TrafficData(
                    intersection_id=intersection_id,
                    timestamp=current_time,
                    vehicle_counts=vehicle_counts,
                    lane_occupancy=lane_occupancy,
                    queue_lengths=queue_lengths,
                    waiting_times=waiting_times,
                    current_phase=self.current_phases[intersection_id],
                    phase_remaining_time=self.phase_timers[intersection_id]
                )
                
                self.traffic_data[intersection_id] = traffic_data
                
            except Exception as e:
                logger.error(f"Error collecting traffic data for {intersection_id}: {e}")
                self.stats['errors'] += 1
    
    def get_traffic_data(self, intersection_id: str) -> Optional[TrafficData]:
        """
        Get traffic data for a specific intersection
        
        Args:
            intersection_id: ID of the intersection
            
        Returns:
            TrafficData object or None if not available
        """
        return self.traffic_data.get(intersection_id)
    
    def get_all_traffic_data(self) -> Dict[str, TrafficData]:
        """
        Get traffic data for all intersections
        
        Returns:
            Dictionary of TrafficData objects
        """
        return self.traffic_data.copy()
    
    def set_phase(self, intersection_id: str, phase: str, priority: int = 0):
        """
        Set signal phase for an intersection
        
        Args:
            intersection_id: ID of the intersection
            phase: Signal phase to set
            priority: Command priority (higher = more important)
        """
        command = ControlCommand(
            intersection_id=intersection_id,
            action='set_phase',
            phase=phase,
            priority=priority
        )
        self.control_queue.put((priority, command))
    
    def extend_phase(self, intersection_id: str, duration: float, priority: int = 0):
        """
        Extend current phase duration
        
        Args:
            intersection_id: ID of the intersection
            duration: Duration to extend (seconds)
            priority: Command priority
        """
        command = ControlCommand(
            intersection_id=intersection_id,
            action='extend_phase',
            duration=duration,
            priority=priority
        )
        self.control_queue.put((priority, command))
    
    def emergency_override(self, intersection_id: str):
        """
        Activate emergency override for an intersection
        
        Args:
            intersection_id: ID of the intersection
        """
        command = ControlCommand(
            intersection_id=intersection_id,
            action='emergency_override',
            priority=1000  # Highest priority
        )
        self.control_queue.put((1000, command))
    
    def clear_emergency_override(self, intersection_id: str):
        """
        Clear emergency override for an intersection
        
        Args:
            intersection_id: ID of the intersection
        """
        if intersection_id in self.emergency_intersections:
            self.emergency_intersections.remove(intersection_id)
            logger.info(f"Cleared emergency override for {intersection_id}")
    
    def optimize_with_webster(self, intersection_id: str):
        """
        Optimize signal timing using Webster's formula
        
        Args:
            intersection_id: ID of the intersection
        """
        if intersection_id not in self.intersections:
            logger.warning(f"Unknown intersection: {intersection_id}")
            return
        
        traffic_data = self.traffic_data.get(intersection_id)
        if not traffic_data:
            logger.warning(f"No traffic data available for {intersection_id}")
            return
        
        # Calculate approach flows (vehicles per hour)
        approach_flows = {}
        for edge, count in traffic_data.vehicle_counts.items():
            # Convert to vehicles per hour (assuming 1-second monitoring interval)
            approach_flows[edge] = count * 3600
        
        # Calculate optimal cycle time
        cycle_time = WebsterFormula.calculate_cycle_time(approach_flows)
        
        # Calculate green times
        green_times = WebsterFormula.calculate_green_times(approach_flows, cycle_time)
        
        logger.info(f"Webster optimization for {intersection_id}: cycle_time={cycle_time:.1f}s, green_times={green_times}")
        
        # Apply optimization (simplified - in practice, this would be more complex)
        # For now, just log the results
        return {
            'cycle_time': cycle_time,
            'green_times': green_times
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get controller statistics
        
        Returns:
            Dictionary of statistics
        """
        uptime = time.time() - self.stats['start_time']
        
        return {
            'connection_state': self.connection_state.value,
            'uptime': uptime,
            'commands_sent': self.stats['commands_sent'],
            'commands_failed': self.stats['commands_failed'],
            'reconnections': self.stats['reconnections'],
            'errors': self.stats['errors'],
            'intersections': len(self.intersections),
            'emergency_intersections': len(self.emergency_intersections),
            'queue_size': self.control_queue.qsize()
        }
    
    def health_check(self) -> bool:
        """
        Perform health check
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Check connection
            if self.connection_state != ConnectionState.CONNECTED:
                return False
            
            # Check if SUMO is responding
            traci.simulation.getTime()
            
            # Check for too many errors
            if self.stats['errors'] > 100:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

def create_default_config() -> Dict[str, Any]:
    """Create default configuration"""
    return {
        'max_retries': 5,
        'retry_delay': 5.0,
        'port': 8813,
        'host': 'localhost',
        'monitoring_interval': 1.0,
        'control_interval': 0.1,
        'log_level': 'INFO',
        'intersections': [
            {
                'id': 'center',
                'phases': ['GGrrGGrr', 'yyrryyrr', 'rrrrrrrr', 'rrGGrrGG', 'rryyrryy'],
                'min_green_time': 10.0,
                'max_green_time': 60.0,
                'yellow_time': 3.0,
                'all_red_time': 2.0,
                'approach_edges': ['north_approach', 'south_approach', 'east_approach', 'west_approach'],
                'exit_edges': ['north_exit', 'south_exit', 'east_exit', 'west_exit'],
                'detectors': []
            }
        ]
    }

def main():
    """Main function for testing"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create default config
    config = create_default_config()
    with open('config/traci_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create controller
    controller = RobustTraCIController()
    
    try:
        # Start controller
        controller.start()
        
        # Run for 60 seconds
        time.sleep(60)
        
        # Print statistics
        stats = controller.get_statistics()
        print(f"Controller Statistics: {stats}")
        
    except KeyboardInterrupt:
        print("Stopping controller...")
    finally:
        controller.stop()

if __name__ == "__main__":
    main()
