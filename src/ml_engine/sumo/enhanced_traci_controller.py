"""
Enhanced TraCI Controller with Robust Error Handling and Bidirectional Communication
Phase 2: Real-Time Optimization Loop & Safety Systems

Features:
- Robust error handling and recovery mechanisms
- Bidirectional communication: SUMO → ML → SUMO
- Simulation state synchronization and recovery
- Scenario switching capability during runtime
- Real-time data extraction and control
"""

import logging
import threading
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import json
import uuid
import subprocess
import os
import signal
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import traci
import traci.constants as tc


class SimulationState(Enum):
    """Simulation state enumeration"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"
    RECOVERING = "recovering"


class ConnectionStatus(Enum):
    """TraCI connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass
class SimulationMetrics:
    """Simulation performance metrics"""
    timestamp: datetime
    simulation_time: float
    step_count: int
    vehicle_count: int
    connection_status: ConnectionStatus
    data_latency: float
    control_latency: float
    error_count: int
    recovery_count: int
    scenario_name: str
    is_running: bool


@dataclass
class TrafficData:
    """Traffic data extracted from SUMO"""
    intersection_id: str
    timestamp: datetime
    simulation_time: float
    vehicle_counts: Dict[str, int]
    queue_lengths: Dict[str, int]
    waiting_times: Dict[str, float]
    avg_speeds: Dict[str, float]
    signal_states: Dict[str, str]
    phase_timings: Dict[str, int]
    emergency_vehicles: List[str]
    weather_condition: str
    visibility: float
    congestion_level: float


@dataclass
class ControlCommand:
    """Control command to send to SUMO"""
    command_id: str
    intersection_id: str
    timestamp: datetime
    command_type: str
    parameters: Dict[str, Any]
    priority: int = 1
    timeout: float = 5.0
    retry_count: int = 0
    max_retries: int = 3


class TraCIError(Exception):
    """Custom TraCI error"""
    pass


class SimulationConnectionError(TraCIError):
    """Simulation connection error"""
    pass


class ControlCommandError(TraCIError):
    """Control command error"""
    pass


class EnhancedTraCIController:
    """
    Enhanced TraCI Controller with robust error handling and bidirectional communication
    
    Features:
    - Robust error handling and recovery mechanisms
    - Bidirectional communication: SUMO → ML → SUMO
    - Simulation state synchronization and recovery
    - Scenario switching capability during runtime
    - Real-time data extraction and control
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Simulation state
        self.simulation_state = SimulationState.STOPPED
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.simulation_process = None
        self.simulation_thread = None
        
        # Configuration
        self.sumo_binary = self.config.get('sumo_binary', 'sumo')
        self.sumo_config = self.config.get('sumo_config', 'config.sumocfg')
        self.port = self.config.get('port', 8813)
        self.host = self.config.get('host', 'localhost')
        
        # Data extraction
        self.data_queue = queue.Queue(maxsize=1000)
        self.control_queue = queue.Queue(maxsize=1000)
        
        # Intersection management
        self.intersections = {}
        self.intersection_configs = {}
        
        # Error handling
        self.error_count = 0
        self.recovery_count = 0
        self.last_error = None
        self.error_threshold = self.config.get('error_threshold', 10)
        
        # Performance monitoring
        self.metrics = {
            'data_latency': deque(maxlen=100),
            'control_latency': deque(maxlen=100),
            'step_times': deque(maxlen=1000),
            'error_times': deque(maxlen=100)
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Callbacks
        self.data_callbacks = []
        self.error_callbacks = []
        self.state_callbacks = []
        
        self.logger.info("Enhanced TraCI Controller initialized")
    
    def add_intersection(self, intersection_id: str, config: Dict[str, Any]):
        """Add intersection configuration"""
        with self.lock:
            self.intersections[intersection_id] = {
                'id': intersection_id,
                'traffic_lights': config.get('traffic_lights', []),
                'detectors': config.get('detectors', []),
                'lanes': config.get('lanes', []),
                'phases': config.get('phases', []),
                'is_active': True
            }
            
            self.intersection_configs[intersection_id] = config
            
            self.logger.info(f"Added intersection: {intersection_id}")
    
    def start_simulation(self, scenario_path: Optional[str] = None) -> bool:
        """Start SUMO simulation with error handling"""
        try:
            with self.lock:
                if self.simulation_state != SimulationState.STOPPED:
                    self.logger.warning("Simulation is already running or starting")
                    return False
                
                self.simulation_state = SimulationState.STARTING
                self.connection_status = ConnectionStatus.CONNECTING
                
                # Start simulation process
                if scenario_path:
                    self.sumo_config = scenario_path
                
                success = self._start_sumo_process()
                
                if success:
                    # Start simulation thread
                    self.simulation_thread = threading.Thread(
                        target=self._simulation_loop,
                        daemon=True
                    )
                    self.simulation_thread.start()
                    
                    self.logger.info("Simulation started successfully")
                    return True
                else:
                    self.simulation_state = SimulationState.ERROR
                    self.connection_status = ConnectionStatus.ERROR
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error starting simulation: {e}")
            self.simulation_state = SimulationState.ERROR
            self.connection_status = ConnectionStatus.ERROR
            return False
    
    def stop_simulation(self) -> bool:
        """Stop SUMO simulation gracefully"""
        try:
            with self.lock:
                if self.simulation_state == SimulationState.STOPPED:
                    return True
                
                self.simulation_state = SimulationState.STOPPING
                
                # Close TraCI connection
                if self.connection_status == ConnectionStatus.CONNECTED:
                    try:
                        traci.close()
                    except:
                        pass
                
                # Terminate simulation process
                if self.simulation_process:
                    try:
                        self.simulation_process.terminate()
                        self.simulation_process.wait(timeout=10)
                    except:
                        try:
                            self.simulation_process.kill()
                        except:
                            pass
                
                # Wait for simulation thread to finish
                if self.simulation_thread and self.simulation_thread.is_alive():
                    self.simulation_thread.join(timeout=5)
                
                self.simulation_state = SimulationState.STOPPED
                self.connection_status = ConnectionStatus.DISCONNECTED
                
                self.logger.info("Simulation stopped successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Error stopping simulation: {e}")
            return False
    
    def _start_sumo_process(self) -> bool:
        """Start SUMO process with configuration"""
        try:
            # Build SUMO command
            cmd = [
                self.sumo_binary,
                '-c', self.sumo_config,
                '--remote-port', str(self.port),
                '--step-length', '1.0',
                '--no-warnings', 'true',
                '--no-step-log', 'true'
            ]
            
            # Start process
            self.simulation_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Wait for process to start
            time.sleep(2)
            
            # Check if process is running
            if self.simulation_process.poll() is None:
                self.logger.info(f"SUMO process started with PID: {self.simulation_process.pid}")
                return True
            else:
                self.logger.error("SUMO process failed to start")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting SUMO process: {e}")
            return False
    
    def _simulation_loop(self):
        """Main simulation loop with error handling"""
        self.logger.info("Starting simulation loop")
        
        while self.simulation_state in [SimulationState.STARTING, SimulationState.RUNNING]:
            try:
                # Connect to TraCI
                if self.connection_status != ConnectionStatus.CONNECTED:
                    if not self._connect_to_traci():
                        time.sleep(5)
                        continue
                
                # Run simulation step
                self._run_simulation_step()
                
                # Extract traffic data
                self._extract_traffic_data()
                
                # Process control commands
                self._process_control_commands()
                
                # Update metrics
                self._update_metrics()
                
                # Check for errors
                self._check_simulation_health()
                
            except Exception as e:
                self.logger.error(f"Error in simulation loop: {e}")
                self._handle_simulation_error(e)
                time.sleep(1)
        
        self.logger.info("Simulation loop ended")
    
    def _connect_to_traci(self) -> bool:
        """Connect to TraCI with retry logic"""
        try:
            # Try to connect
            traci.init(port=self.port, host=self.host)
            
            self.connection_status = ConnectionStatus.CONNECTED
            self.simulation_state = SimulationState.RUNNING
            
            self.logger.info("Connected to TraCI successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to TraCI: {e}")
            self.connection_status = ConnectionStatus.ERROR
            self.last_error = str(e)
            return False
    
    def _run_simulation_step(self):
        """Run single simulation step"""
        try:
            start_time = time.time()
            
            # Advance simulation
            traci.simulationStep()
            
            # Record step time
            step_time = time.time() - start_time
            self.metrics['step_times'].append(step_time)
            
        except Exception as e:
            raise TraCIError(f"Simulation step failed: {e}")
    
    def _extract_traffic_data(self):
        """Extract traffic data from SUMO"""
        try:
            for intersection_id, intersection in self.intersections.items():
                if not intersection['is_active']:
                    continue
                
                # Extract data for intersection
                traffic_data = self._extract_intersection_data(intersection_id, intersection)
                
                if traffic_data:
                    # Add to data queue
                    try:
                        self.data_queue.put_nowait(traffic_data)
                    except queue.Full:
                        self.logger.warning("Data queue is full, dropping data")
                
        except Exception as e:
            self.logger.error(f"Error extracting traffic data: {e}")
    
    def _extract_intersection_data(self, intersection_id: str, intersection: Dict[str, Any]) -> Optional[TrafficData]:
        """Extract traffic data for specific intersection"""
        try:
            # Get simulation time
            sim_time = traci.simulation.getTime()
            
            # Extract vehicle counts
            vehicle_counts = {}
            queue_lengths = {}
            waiting_times = {}
            avg_speeds = {}
            
            for lane in intersection['lanes']:
                try:
                    # Get vehicle count
                    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                    vehicle_counts[lane] = len(vehicle_ids)
                    
                    # Get queue length
                    queue_lengths[lane] = traci.lane.getLastStepHaltingNumber(lane)
                    
                    # Get waiting time
                    waiting_times[lane] = traci.lane.getWaitingTime(lane)
                    
                    # Get average speed
                    avg_speeds[lane] = traci.lane.getLastStepMeanSpeed(lane)
                    
                except Exception as e:
                    self.logger.warning(f"Error extracting data for lane {lane}: {e}")
                    vehicle_counts[lane] = 0
                    queue_lengths[lane] = 0
                    waiting_times[lane] = 0.0
                    avg_speeds[lane] = 0.0
            
            # Extract signal states
            signal_states = {}
            phase_timings = {}
            
            for tl_id in intersection['traffic_lights']:
                try:
                    # Get current phase
                    current_phase = traci.trafficlight.getPhase(tl_id)
                    signal_states[tl_id] = str(current_phase)
                    
                    # Get phase timing
                    phase_timings[tl_id] = traci.trafficlight.getPhaseDuration(tl_id)
                    
                except Exception as e:
                    self.logger.warning(f"Error extracting signal data for {tl_id}: {e}")
                    signal_states[tl_id] = "unknown"
                    phase_timings[tl_id] = 30
            
            # Extract emergency vehicles
            emergency_vehicles = []
            try:
                all_vehicles = traci.vehicle.getIDList()
                for vehicle_id in all_vehicles:
                    try:
                        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
                        if vehicle_type in ['emergency', 'ambulance', 'fire_truck', 'police']:
                            emergency_vehicles.append(vehicle_id)
                    except:
                        pass
            except:
                pass
            
            # Calculate congestion level
            total_vehicles = sum(vehicle_counts.values())
            max_capacity = len(intersection['lanes']) * 20  # Assume 20 vehicles per lane max
            congestion_level = min(1.0, total_vehicles / max_capacity) if max_capacity > 0 else 0.0
            
            return TrafficData(
                intersection_id=intersection_id,
                timestamp=datetime.now(),
                simulation_time=sim_time,
                vehicle_counts=vehicle_counts,
                queue_lengths=queue_lengths,
                waiting_times=waiting_times,
                avg_speeds=avg_speeds,
                signal_states=signal_states,
                phase_timings=phase_timings,
                emergency_vehicles=emergency_vehicles,
                weather_condition="clear",  # Would be extracted from SUMO weather
                visibility=1.0,  # Would be extracted from SUMO weather
                congestion_level=congestion_level
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting intersection data for {intersection_id}: {e}")
            return None
    
    def _process_control_commands(self):
        """Process control commands from queue"""
        try:
            # Process up to 10 commands per step
            for _ in range(10):
                try:
                    command = self.control_queue.get_nowait()
                    self._execute_control_command(command)
                except queue.Empty:
                    break
                except Exception as e:
                    self.logger.error(f"Error processing control command: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error processing control commands: {e}")
    
    def _execute_control_command(self, command: ControlCommand):
        """Execute control command on SUMO"""
        try:
            start_time = time.time()
            
            if command.command_type == "set_phase":
                # Set traffic light phase
                tl_id = command.parameters.get('traffic_light_id')
                phase = command.parameters.get('phase')
                duration = command.parameters.get('duration', 30)
                
                if tl_id and phase is not None:
                    traci.trafficlight.setPhase(tl_id, phase)
                    traci.trafficlight.setPhaseDuration(tl_id, duration)
                    
            elif command.command_type == "set_timing":
                # Set phase timings
                tl_id = command.parameters.get('traffic_light_id')
                timings = command.parameters.get('timings', {})
                
                if tl_id and timings:
                    for phase, duration in timings.items():
                        traci.trafficlight.setPhaseDuration(tl_id, int(phase), float(duration))
                        
            elif command.command_type == "set_program":
                # Set traffic light program
                tl_id = command.parameters.get('traffic_light_id')
                program = command.parameters.get('program')
                
                if tl_id and program:
                    traci.trafficlight.setProgram(tl_id, program)
            
            # Record control latency
            control_latency = time.time() - start_time
            self.metrics['control_latency'].append(control_latency)
            
            self.logger.debug(f"Executed control command: {command.command_type}")
            
        except Exception as e:
            self.logger.error(f"Error executing control command {command.command_type}: {e}")
            
            # Retry command if retries available
            if command.retry_count < command.max_retries:
                command.retry_count += 1
                self.control_queue.put(command)
    
    def _check_simulation_health(self):
        """Check simulation health and handle errors"""
        try:
            # Check if simulation process is still running
            if self.simulation_process and self.simulation_process.poll() is not None:
                self.logger.error("Simulation process has terminated unexpectedly")
                self._handle_simulation_error("Simulation process terminated")
                return
            
            # Check for excessive errors
            if self.error_count > self.error_threshold:
                self.logger.error(f"Too many errors ({self.error_count}), stopping simulation")
                self.simulation_state = SimulationState.ERROR
                return
            
            # Check connection health
            if self.connection_status == ConnectionStatus.CONNECTED:
                try:
                    # Test connection with simple query
                    traci.simulation.getTime()
                except:
                    self.logger.warning("TraCI connection lost, attempting recovery")
                    self.connection_status = ConnectionStatus.RECONNECTING
                    self._handle_connection_error()
            
        except Exception as e:
            self.logger.error(f"Error checking simulation health: {e}")
    
    def _handle_simulation_error(self, error: Union[str, Exception]):
        """Handle simulation errors with recovery"""
        try:
            self.error_count += 1
            self.last_error = str(error)
            self.metrics['error_times'].append(time.time())
            
            self.logger.error(f"Simulation error: {error}")
            
            # Attempt recovery
            if self.error_count <= self.error_threshold:
                self.simulation_state = SimulationState.RECOVERING
                self.recovery_count += 1
                
                # Wait before recovery attempt
                time.sleep(5)
                
                # Try to reconnect
                if self._connect_to_traci():
                    self.simulation_state = SimulationState.RUNNING
                    self.error_count = 0  # Reset error count on successful recovery
                    self.logger.info("Simulation recovered successfully")
                else:
                    self.simulation_state = SimulationState.ERROR
            else:
                self.simulation_state = SimulationState.ERROR
                self.logger.error("Maximum error threshold reached, stopping simulation")
            
        except Exception as e:
            self.logger.error(f"Error handling simulation error: {e}")
            self.simulation_state = SimulationState.ERROR
    
    def _handle_connection_error(self):
        """Handle TraCI connection errors"""
        try:
            # Close existing connection
            try:
                traci.close()
            except:
                pass
            
            # Wait before reconnection
            time.sleep(2)
            
            # Attempt reconnection
            if self._connect_to_traci():
                self.logger.info("TraCI connection recovered")
            else:
                self.logger.error("Failed to recover TraCI connection")
                self.connection_status = ConnectionStatus.ERROR
                
        except Exception as e:
            self.logger.error(f"Error handling connection error: {e}")
            self.connection_status = ConnectionStatus.ERROR
    
    def _update_metrics(self):
        """Update performance metrics"""
        try:
            # Calculate data latency
            if self.data_queue.qsize() > 0:
                # Estimate data latency based on queue size
                estimated_latency = self.data_queue.qsize() * 0.1  # Rough estimate
                self.metrics['data_latency'].append(estimated_latency)
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")
    
    def send_control_command(self, intersection_id: str, command_type: str, 
                           parameters: Dict[str, Any], priority: int = 1) -> str:
        """Send control command to SUMO"""
        try:
            command = ControlCommand(
                command_id=str(uuid.uuid4()),
                intersection_id=intersection_id,
                timestamp=datetime.now(),
                command_type=command_type,
                parameters=parameters,
                priority=priority
            )
            
            # Add to control queue
            self.control_queue.put(command)
            
            self.logger.debug(f"Queued control command: {command_type} for {intersection_id}")
            return command.command_id
            
        except Exception as e:
            self.logger.error(f"Error sending control command: {e}")
            return ""
    
    def get_traffic_data(self, timeout: float = 1.0) -> Optional[TrafficData]:
        """Get traffic data from queue"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def get_simulation_metrics(self) -> SimulationMetrics:
        """Get current simulation metrics"""
        try:
            sim_time = 0.0
            step_count = 0
            vehicle_count = 0
            
            if self.connection_status == ConnectionStatus.CONNECTED:
                try:
                    sim_time = traci.simulation.getTime()
                    step_count = traci.simulation.getMinExpectedNumber()
                    vehicle_count = len(traci.vehicle.getIDList())
                except:
                    pass
            
            # Calculate average latencies
            avg_data_latency = np.mean(self.metrics['data_latency']) if self.metrics['data_latency'] else 0.0
            avg_control_latency = np.mean(self.metrics['control_latency']) if self.metrics['control_latency'] else 0.0
            
            return SimulationMetrics(
                timestamp=datetime.now(),
                simulation_time=sim_time,
                step_count=step_count,
                vehicle_count=vehicle_count,
                connection_status=self.connection_status,
                data_latency=avg_data_latency,
                control_latency=avg_control_latency,
                error_count=self.error_count,
                recovery_count=self.recovery_count,
                scenario_name=os.path.basename(self.sumo_config),
                is_running=self.simulation_state == SimulationState.RUNNING
            )
            
        except Exception as e:
            self.logger.error(f"Error getting simulation metrics: {e}")
            return SimulationMetrics(
                timestamp=datetime.now(),
                simulation_time=0.0,
                step_count=0,
                vehicle_count=0,
                connection_status=ConnectionStatus.ERROR,
                data_latency=0.0,
                control_latency=0.0,
                error_count=self.error_count,
                recovery_count=self.recovery_count,
                scenario_name="unknown",
                is_running=False
            )
    
    def switch_scenario(self, new_scenario_path: str) -> bool:
        """Switch simulation scenario during runtime"""
        try:
            self.logger.info(f"Switching to scenario: {new_scenario_path}")
            
            # Stop current simulation
            if not self.stop_simulation():
                return False
            
            # Wait for complete shutdown
            time.sleep(2)
            
            # Start new scenario
            self.sumo_config = new_scenario_path
            return self.start_simulation()
            
        except Exception as e:
            self.logger.error(f"Error switching scenario: {e}")
            return False
    
    def add_data_callback(self, callback: Callable[[TrafficData], None]):
        """Add callback for traffic data"""
        self.data_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str], None]):
        """Add callback for errors"""
        self.error_callbacks.append(callback)
    
    def add_state_callback(self, callback: Callable[[SimulationState], None]):
        """Add callback for state changes"""
        self.state_callbacks.append(callback)
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get comprehensive simulation status"""
        return {
            'simulation_state': self.simulation_state.value,
            'connection_status': self.connection_status.value,
            'intersections': len(self.intersections),
            'active_intersections': len([i for i in self.intersections.values() if i['is_active']]),
            'data_queue_size': self.data_queue.qsize(),
            'control_queue_size': self.control_queue.qsize(),
            'error_count': self.error_count,
            'recovery_count': self.recovery_count,
            'last_error': self.last_error,
            'metrics': {
                'avg_step_time': np.mean(self.metrics['step_times']) if self.metrics['step_times'] else 0.0,
                'avg_data_latency': np.mean(self.metrics['data_latency']) if self.metrics['data_latency'] else 0.0,
                'avg_control_latency': np.mean(self.metrics['control_latency']) if self.metrics['control_latency'] else 0.0
            }
        }
