"""
SUMO Traffic Simulation Controller with TraCI Integration
Real-time simulation control and traffic light management
"""

import traci
import traci.constants as tc
import time
import threading
import logging
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

from config.sumo_config import SumoConfig, get_sumo_config


class SimulationState(Enum):
    """Simulation states"""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


class TrafficLightState(Enum):
    """Traffic light states"""
    RED = "r"
    YELLOW = "y"
    GREEN = "G"
    RED_YELLOW = "u"


@dataclass
class VehicleData:
    """Vehicle data structure"""
    id: str
    position: Tuple[float, float]
    speed: float
    lane: str
    route: List[str]
    waiting_time: float
    co2_emission: float
    fuel_consumption: float
    timestamp: datetime


@dataclass
class IntersectionData:
    """Intersection data structure"""
    id: str
    position: Tuple[float, float]
    current_phase: int
    phase_duration: float
    program_id: str
    controlled_lanes: List[str]
    vehicle_counts: Dict[str, int]
    waiting_vehicles: int
    timestamp: datetime


@dataclass
class SimulationMetrics:
    """Simulation performance metrics"""
    timestamp: datetime
    total_vehicles: int
    running_vehicles: int
    waiting_vehicles: int
    total_waiting_time: float
    total_co2_emission: float
    total_fuel_consumption: float
    average_speed: float
    simulation_time: float


class SumoController:
    """
    SUMO Traffic Simulation Controller with TraCI Integration
    
    Features:
    - Real-time simulation control
    - Dynamic traffic light management
    - Vehicle detection and counting
    - Data collection and export
    - Error handling and recovery
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[SumoConfig] = None):
        self.config = config or get_sumo_config()
        self.logger = logging.getLogger(__name__)
        
        # Simulation state
        self.state = SimulationState.STOPPED
        self.simulation_time = 0.0
        self.step_size = 1.0
        
        # Data collection
        self.vehicles: Dict[str, VehicleData] = {}
        self.intersections: Dict[str, IntersectionData] = {}
        self.metrics_history: List[SimulationMetrics] = []
        
        # API integration
        self.api_client = None
        self.data_export_interval = 10.0  # seconds
        self.last_export_time = 0.0
        
        # Control flags
        self.is_running = False
        self.control_thread = None
        self.export_thread = None
        
        # Error handling
        self.connection_retries = 0
        self.max_retries = 5
        self.retry_delay = 5.0
        
        self.logger.info("SUMO Controller initialized")
    
    async def start_simulation(self, scenario_file: str = None) -> bool:
        """Start SUMO simulation"""
        try:
            self.logger.info("Starting SUMO simulation...")
            
            # Start TraCI connection
            if not self._connect_to_sumo(scenario_file):
                return False
            
            # Initialize simulation
            self._initialize_simulation()
            
            # Start control threads
            self.is_running = True
            self.state = SimulationState.RUNNING
            
            self.control_thread = threading.Thread(target=self._simulation_loop)
            self.control_thread.daemon = True
            self.control_thread.start()
            
            self.export_thread = threading.Thread(target=self._data_export_loop)
            self.export_thread.daemon = True
            self.export_thread.start()
            
            self.logger.info("SUMO simulation started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start simulation: {e}")
            self.state = SimulationState.ERROR
            return False
    
    def stop_simulation(self):
        """Stop SUMO simulation"""
        self.logger.info("Stopping SUMO simulation...")
        
        self.is_running = False
        self.state = SimulationState.STOPPED
        
        # Wait for threads to finish
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=5)
        
        if self.export_thread and self.export_thread.is_alive():
            self.export_thread.join(timeout=5)
        
        # Close TraCI connection
        try:
            traci.close()
        except:
            pass
        
        self.logger.info("SUMO simulation stopped")
    
    def pause_simulation(self):
        """Pause simulation"""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
            self.logger.info("Simulation paused")
    
    def resume_simulation(self):
        """Resume simulation"""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
            self.logger.info("Simulation resumed")
    
    def _connect_to_sumo(self, scenario_file: str = None) -> bool:
        """Connect to SUMO using TraCI"""
        for attempt in range(self.max_retries):
            try:
                if scenario_file:
                    # Start SUMO with scenario file
                    sumo_cmd = [
                        "sumo",
                        "-c", scenario_file,
                        "--start", "--quit-on-end"
                    ]
                    traci.start(sumo_cmd)
                else:
                    # Connect to existing SUMO instance
                    traci.init()
                
                self.connection_retries = 0
                self.logger.info("Connected to SUMO successfully")
                return True
                
            except Exception as e:
                self.connection_retries += 1
                self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error("Failed to connect to SUMO after all retries")
                    return False
        
        return False
    
    def _initialize_simulation(self):
        """Initialize simulation data structures"""
        # Get traffic light IDs
        self.traffic_lights = traci.trafficlight.getIDList()
        self.logger.info(f"Found {len(self.traffic_lights)} traffic lights")
        
        # Initialize intersection data
        for tl_id in self.traffic_lights:
            self.intersections[tl_id] = IntersectionData(
                id=tl_id,
                position=traci.junction.getPosition(tl_id),
                current_phase=0,
                phase_duration=0.0,
                program_id="",
                controlled_lanes=[],
                vehicle_counts={},
                waiting_vehicles=0,
                timestamp=datetime.now()
            )
        
        # Initialize vehicle tracking
        self.vehicles = {}
        
        self.logger.info("Simulation initialized")
    
    def _simulation_loop(self):
        """Main simulation control loop"""
        while self.is_running:
            try:
                if self.state == SimulationState.RUNNING:
                    # Advance simulation by one step
                    traci.simulationStep()
                    self.simulation_time = traci.simulation.getTime()
                    
                    # Update data
                    self._update_vehicle_data()
                    self._update_intersection_data()
                    self._update_metrics()
                    
                    # Apply traffic light control
                    self._apply_traffic_light_control()
                
                time.sleep(self.step_size)
                
            except Exception as e:
                self.logger.error(f"Error in simulation loop: {e}")
                self.state = SimulationState.ERROR
                break
    
    def _update_vehicle_data(self):
        """Update vehicle data from simulation"""
        current_vehicles = traci.vehicle.getIDList()
        
        # Remove vehicles that left the simulation
        for vehicle_id in list(self.vehicles.keys()):
            if vehicle_id not in current_vehicles:
                del self.vehicles[vehicle_id]
        
        # Update existing vehicles and add new ones
        for vehicle_id in current_vehicles:
            try:
                position = traci.vehicle.getPosition(vehicle_id)
                speed = traci.vehicle.getSpeed(vehicle_id)
                lane = traci.vehicle.getLaneID(vehicle_id)
                route = traci.vehicle.getRoute(vehicle_id)
                waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
                co2_emission = traci.vehicle.getCO2Emission(vehicle_id)
                fuel_consumption = traci.vehicle.getFuelConsumption(vehicle_id)
                
                self.vehicles[vehicle_id] = VehicleData(
                    id=vehicle_id,
                    position=position,
                    speed=speed,
                    lane=lane,
                    route=route,
                    waiting_time=waiting_time,
                    co2_emission=co2_emission,
                    fuel_consumption=fuel_consumption,
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                self.logger.warning(f"Error updating vehicle {vehicle_id}: {e}")
    
    def _update_intersection_data(self):
        """Update intersection data from simulation"""
        for tl_id, intersection in self.intersections.items():
            try:
                # Get current phase information
                current_phase = traci.trafficlight.getPhase(tl_id)
                phase_duration = traci.trafficlight.getPhaseDuration(tl_id)
                program_id = traci.trafficlight.getProgram(tl_id)
                controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
                
                # Count vehicles on controlled lanes
                vehicle_counts = {}
                waiting_vehicles = 0
                
                for lane in controlled_lanes:
                    lane_vehicles = traci.lane.getLastStepVehicleNumber(lane)
                    vehicle_counts[lane] = lane_vehicles
                    waiting_vehicles += traci.lane.getLastStepHaltingNumber(lane)
                
                # Update intersection data
                intersection.current_phase = current_phase
                intersection.phase_duration = phase_duration
                intersection.program_id = program_id
                intersection.controlled_lanes = controlled_lanes
                intersection.vehicle_counts = vehicle_counts
                intersection.waiting_vehicles = waiting_vehicles
                intersection.timestamp = datetime.now()
                
            except Exception as e:
                self.logger.warning(f"Error updating intersection {tl_id}: {e}")
    
    def _update_metrics(self):
        """Update simulation performance metrics"""
        total_vehicles = len(self.vehicles)
        running_vehicles = sum(1 for v in self.vehicles.values() if v.speed > 0.1)
        waiting_vehicles = sum(1 for v in self.vehicles.values() if v.speed <= 0.1)
        total_waiting_time = sum(v.waiting_time for v in self.vehicles.values())
        total_co2_emission = sum(v.co2_emission for v in self.vehicles.values())
        total_fuel_consumption = sum(v.fuel_consumption for v in self.vehicles.values())
        average_speed = np.mean([v.speed for v in self.vehicles.values()]) if self.vehicles else 0.0
        
        metrics = SimulationMetrics(
            timestamp=datetime.now(),
            total_vehicles=total_vehicles,
            running_vehicles=running_vehicles,
            waiting_vehicles=waiting_vehicles,
            total_waiting_time=total_waiting_time,
            total_co2_emission=total_co2_emission,
            total_fuel_consumption=total_fuel_consumption,
            average_speed=average_speed,
            simulation_time=self.simulation_time
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only recent history
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
    
    def _apply_traffic_light_control(self):
        """Apply traffic light control based on ML optimizer output"""
        # This will be called by the ML optimizer
        # For now, we'll implement basic adaptive control
        pass
    
    def set_traffic_light_program(self, tl_id: str, program: str):
        """Set traffic light program"""
        try:
            traci.trafficlight.setProgram(tl_id, program)
            self.logger.debug(f"Set program {program} for traffic light {tl_id}")
        except Exception as e:
            self.logger.error(f"Error setting program for {tl_id}: {e}")
    
    def set_traffic_light_phase(self, tl_id: str, phase: int, duration: float = None):
        """Set traffic light phase"""
        try:
            if duration:
                traci.trafficlight.setPhaseDuration(tl_id, duration)
            traci.trafficlight.setPhase(tl_id, phase)
            self.logger.debug(f"Set phase {phase} for traffic light {tl_id}")
        except Exception as e:
            self.logger.error(f"Error setting phase for {tl_id}: {e}")
    
    def get_intersection_data(self, tl_id: str) -> Optional[IntersectionData]:
        """Get intersection data"""
        return self.intersections.get(tl_id)
    
    def get_all_intersections_data(self) -> Dict[str, IntersectionData]:
        """Get all intersections data"""
        return self.intersections.copy()
    
    def get_vehicle_data(self, vehicle_id: str) -> Optional[VehicleData]:
        """Get vehicle data"""
        return self.vehicles.get(vehicle_id)
    
    def get_all_vehicles_data(self) -> Dict[str, VehicleData]:
        """Get all vehicles data"""
        return self.vehicles.copy()
    
    def get_simulation_metrics(self) -> Optional[SimulationMetrics]:
        """Get current simulation metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, limit: int = 100) -> List[SimulationMetrics]:
        """Get simulation metrics history"""
        return self.metrics_history[-limit:] if self.metrics_history else []
    
    def _data_export_loop(self):
        """Data export loop for API integration"""
        while self.is_running:
            try:
                current_time = time.time()
                if current_time - self.last_export_time >= self.data_export_interval:
                    self._export_data_to_api()
                    self.last_export_time = current_time
                
                time.sleep(1.0)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Error in data export loop: {e}")
                time.sleep(5.0)
    
    def _export_data_to_api(self):
        """Export simulation data to backend API"""
        try:
            # Prepare data for API
            export_data = {
                'simulation_time': self.simulation_time,
                'timestamp': datetime.now().isoformat(),
                'intersections': {},
                'vehicles': {},
                'metrics': {}
            }
            
            # Export intersection data
            for tl_id, intersection in self.intersections.items():
                export_data['intersections'][tl_id] = {
                    'id': intersection.id,
                    'position': intersection.position,
                    'current_phase': intersection.current_phase,
                    'phase_duration': intersection.phase_duration,
                    'vehicle_counts': intersection.vehicle_counts,
                    'waiting_vehicles': intersection.waiting_vehicles,
                    'timestamp': intersection.timestamp.isoformat()
                }
            
            # Export vehicle data
            for vehicle_id, vehicle in self.vehicles.items():
                export_data['vehicles'][vehicle_id] = {
                    'id': vehicle.id,
                    'position': vehicle.position,
                    'speed': vehicle.speed,
                    'lane': vehicle.lane,
                    'waiting_time': vehicle.waiting_time,
                    'co2_emission': vehicle.co2_emission,
                    'fuel_consumption': vehicle.fuel_consumption,
                    'timestamp': vehicle.timestamp.isoformat()
                }
            
            # Export metrics
            if self.metrics_history:
                latest_metrics = self.metrics_history[-1]
                export_data['metrics'] = {
                    'total_vehicles': latest_metrics.total_vehicles,
                    'running_vehicles': latest_metrics.running_vehicles,
                    'waiting_vehicles': latest_metrics.waiting_vehicles,
                    'total_waiting_time': latest_metrics.total_waiting_time,
                    'total_co2_emission': latest_metrics.total_co2_emission,
                    'total_fuel_consumption': latest_metrics.total_fuel_consumption,
                    'average_speed': latest_metrics.average_speed,
                    'simulation_time': latest_metrics.simulation_time,
                    'timestamp': latest_metrics.timestamp.isoformat()
                }
            
            # Send to API (this would be implemented with actual API client)
            self.logger.debug(f"Exported data: {len(export_data['intersections'])} intersections, "
                            f"{len(export_data['vehicles'])} vehicles")
            
        except Exception as e:
            self.logger.error(f"Error exporting data to API: {e}")
    
    def get_simulation_status(self) -> Dict[str, Any]:
        """Get current simulation status"""
        return {
            'state': self.state.value,
            'simulation_time': self.simulation_time,
            'total_vehicles': len(self.vehicles),
            'total_intersections': len(self.intersections),
            'is_running': self.is_running,
            'connection_retries': self.connection_retries,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_simulation_data(self, filepath: str):
        """Export simulation data to file"""
        try:
            export_data = {
                'simulation_status': self.get_simulation_status(),
                'intersections': {k: {
                    'id': v.id,
                    'position': v.position,
                    'current_phase': v.current_phase,
                    'phase_duration': v.phase_duration,
                    'vehicle_counts': v.vehicle_counts,
                    'waiting_vehicles': v.waiting_vehicles,
                    'timestamp': v.timestamp.isoformat()
                } for k, v in self.intersections.items()},
                'vehicles': {k: {
                    'id': v.id,
                    'position': v.position,
                    'speed': v.speed,
                    'lane': v.lane,
                    'waiting_time': v.waiting_time,
                    'co2_emission': v.co2_emission,
                    'fuel_consumption': v.fuel_consumption,
                    'timestamp': v.timestamp.isoformat()
                } for k, v in self.vehicles.items()},
                'metrics_history': [{
                    'timestamp': m.timestamp.isoformat(),
                    'total_vehicles': m.total_vehicles,
                    'running_vehicles': m.running_vehicles,
                    'waiting_vehicles': m.waiting_vehicles,
                    'total_waiting_time': m.total_waiting_time,
                    'total_co2_emission': m.total_co2_emission,
                    'total_fuel_consumption': m.total_fuel_consumption,
                    'average_speed': m.average_speed,
                    'simulation_time': m.simulation_time
                } for m in self.metrics_history]
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Simulation data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting simulation data: {e}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create and start SUMO controller
    controller = SumoController()
    
    try:
        # Start simulation
        success = asyncio.run(controller.start_simulation("scenarios/basic_scenario.sumocfg"))
        
        if success:
            print("Simulation started successfully")
            
            # Run for some time
            time.sleep(60)
            
            # Get status
            status = controller.get_simulation_status()
            print(f"Simulation status: {status}")
            
            # Export data
            controller.export_simulation_data("simulation_data.json")
        
        else:
            print("Failed to start simulation")
    
    finally:
        # Stop simulation
        controller.stop_simulation()
