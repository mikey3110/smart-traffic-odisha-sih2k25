"""
Dynamic Traffic Light Controller for SUMO Integration
Implements ML-based traffic light control with real-time adaptation
"""

import traci
import traci.constants as tc
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np
import asyncio
import aiohttp

from config.sumo_config import SumoConfig, get_sumo_config


class PhaseType(Enum):
    """Traffic light phase types"""
    NORTH_SOUTH_GREEN = "north_south_green"
    EAST_WEST_GREEN = "east_west_green"
    YELLOW = "yellow"
    ALL_RED = "all_red"


@dataclass
class PhaseDefinition:
    """Traffic light phase definition"""
    phase_id: int
    phase_type: PhaseType
    duration: float
    min_duration: float
    max_duration: float
    state: str  # SUMO state string (e.g., "GGrrrrGGrrrr")
    description: str


@dataclass
class TrafficLightProgram:
    """Traffic light program definition"""
    program_id: str
    phases: List[PhaseDefinition]
    cycle_time: float
    offset: float = 0.0
    type: str = "adaptive"


@dataclass
class ControlSignal:
    """Control signal from ML optimizer"""
    intersection_id: str
    phase_id: int
    duration: float
    confidence: float
    timestamp: datetime
    algorithm_used: str


class TrafficLightController:
    """
    Dynamic Traffic Light Controller for SUMO Integration
    
    Features:
    - Real-time traffic light control
    - ML optimizer integration
    - Adaptive phase timing
    - Emergency vehicle priority
    - Pedestrian crossing control
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[SumoConfig] = None):
        self.config = config or get_sumo_config()
        self.logger = logging.getLogger(__name__)
        
        # Traffic light programs
        self.programs: Dict[str, TrafficLightProgram] = {}
        self.current_phases: Dict[str, int] = {}
        self.phase_start_times: Dict[str, float] = {}
        
        # Control signals from ML optimizer
        self.control_queue: List[ControlSignal] = []
        self.last_control_time: Dict[str, datetime] = {}
        
        # Performance tracking
        self.phase_switches: Dict[str, int] = {}
        self.total_waiting_time: Dict[str, float] = {}
        self.throughput: Dict[str, int] = {}
        
        # API integration
        self.api_client = None
        self.control_interval = 1.0  # seconds
        
        self.logger.info("Traffic Light Controller initialized")
    
    def initialize_traffic_lights(self, intersection_ids: List[str]):
        """Initialize traffic lights with default programs"""
        for intersection_id in intersection_ids:
            try:
                # Get current program
                current_program = traci.trafficlight.getProgram(intersection_id)
                self.logger.info(f"Initializing traffic light {intersection_id} with program {current_program}")
                
                # Create default adaptive program
                program = self._create_adaptive_program(intersection_id)
                self.programs[intersection_id] = program
                
                # Initialize tracking
                self.current_phases[intersection_id] = 0
                self.phase_start_times[intersection_id] = time.time()
                self.phase_switches[intersection_id] = 0
                self.total_waiting_time[intersection_id] = 0.0
                self.throughput[intersection_id] = 0
                
            except Exception as e:
                self.logger.error(f"Error initializing traffic light {intersection_id}: {e}")
    
    def _create_adaptive_program(self, intersection_id: str) -> TrafficLightProgram:
        """Create adaptive traffic light program"""
        # Get controlled lanes
        controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
        
        # Determine phase structure based on intersection type
        if len(controlled_lanes) >= 4:  # 4-way intersection
            phases = [
                PhaseDefinition(
                    phase_id=0,
                    phase_type=PhaseType.NORTH_SOUTH_GREEN,
                    duration=30.0,
                    min_duration=10.0,
                    max_duration=60.0,
                    state="GGrrrrGGrrrr",
                    description="North-South Green"
                ),
                PhaseDefinition(
                    phase_id=1,
                    phase_type=PhaseType.YELLOW,
                    duration=3.0,
                    min_duration=3.0,
                    max_duration=3.0,
                    state="yyrrrryyrrrr",
                    description="North-South Yellow"
                ),
                PhaseDefinition(
                    phase_id=2,
                    phase_type=PhaseType.ALL_RED,
                    duration=1.0,
                    min_duration=1.0,
                    max_duration=1.0,
                    state="rrrrrrrrrrrr",
                    description="All Red"
                ),
                PhaseDefinition(
                    phase_id=3,
                    phase_type=PhaseType.EAST_WEST_GREEN,
                    duration=30.0,
                    min_duration=10.0,
                    max_duration=60.0,
                    state="rrrrGGrrrrGG",
                    description="East-West Green"
                ),
                PhaseDefinition(
                    phase_id=4,
                    phase_type=PhaseType.YELLOW,
                    duration=3.0,
                    min_duration=3.0,
                    max_duration=3.0,
                    state="rrrryyrrrryy",
                    description="East-West Yellow"
                ),
                PhaseDefinition(
                    phase_id=5,
                    phase_type=PhaseType.ALL_RED,
                    duration=1.0,
                    min_duration=1.0,
                    max_duration=1.0,
                    state="rrrrrrrrrrrr",
                    description="All Red"
                )
            ]
        else:  # T-junction or other
            phases = [
                PhaseDefinition(
                    phase_id=0,
                    phase_type=PhaseType.NORTH_SOUTH_GREEN,
                    duration=30.0,
                    min_duration=10.0,
                    max_duration=60.0,
                    state="GGrrrr",
                    description="Main Green"
                ),
                PhaseDefinition(
                    phase_id=1,
                    phase_type=PhaseType.YELLOW,
                    duration=3.0,
                    min_duration=3.0,
                    max_duration=3.0,
                    state="yyrrrr",
                    description="Main Yellow"
                ),
                PhaseDefinition(
                    phase_id=2,
                    phase_type=PhaseType.EAST_WEST_GREEN,
                    duration=20.0,
                    min_duration=10.0,
                    max_duration=40.0,
                    state="rrrrGG",
                    description="Side Green"
                ),
                PhaseDefinition(
                    phase_id=3,
                    phase_type=PhaseType.YELLOW,
                    duration=3.0,
                    min_duration=3.0,
                    max_duration=3.0,
                    state="rrrryy",
                    description="Side Yellow"
                )
            ]
        
        return TrafficLightProgram(
            program_id=f"adaptive_{intersection_id}",
            phases=phases,
            cycle_time=sum(phase.duration for phase in phases),
            type="adaptive"
        )
    
    def apply_ml_control(self, control_signal: ControlSignal):
        """Apply control signal from ML optimizer"""
        try:
            intersection_id = control_signal.intersection_id
            
            # Validate control signal
            if not self._validate_control_signal(control_signal):
                self.logger.warning(f"Invalid control signal for {intersection_id}")
                return False
            
            # Check if intersection exists
            if intersection_id not in self.programs:
                self.logger.error(f"Traffic light {intersection_id} not initialized")
                return False
            
            program = self.programs[intersection_id]
            
            # Validate phase ID
            if control_signal.phase_id >= len(program.phases):
                self.logger.error(f"Invalid phase ID {control_signal.phase_id} for {intersection_id}")
                return False
            
            # Apply phase change
            self._apply_phase_change(intersection_id, control_signal.phase_id, control_signal.duration)
            
            # Update tracking
            self.last_control_time[intersection_id] = control_signal.timestamp
            self.phase_switches[intersection_id] += 1
            
            self.logger.info(f"Applied ML control to {intersection_id}: phase {control_signal.phase_id}, "
                           f"duration {control_signal.duration}s, confidence {control_signal.confidence}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying ML control: {e}")
            return False
    
    def _validate_control_signal(self, control_signal: ControlSignal) -> bool:
        """Validate control signal from ML optimizer"""
        # Check basic validity
        if not control_signal.intersection_id or control_signal.phase_id < 0:
            return False
        
        if control_signal.duration <= 0 or control_signal.confidence < 0 or control_signal.confidence > 1:
            return False
        
        # Check if enough time has passed since last control
        if control_signal.intersection_id in self.last_control_time:
            time_since_last = (control_signal.timestamp - self.last_control_time[control_signal.intersection_id]).total_seconds()
            if time_since_last < 1.0:  # Minimum 1 second between controls
                return False
        
        return True
    
    def _apply_phase_change(self, intersection_id: str, phase_id: int, duration: float):
        """Apply phase change to traffic light"""
        try:
            program = self.programs[intersection_id]
            phase = program.phases[phase_id]
            
            # Clamp duration to valid range
            duration = max(phase.min_duration, min(phase.max_duration, duration))
            
            # Set phase duration
            traci.trafficlight.setPhaseDuration(intersection_id, duration)
            
            # Set phase
            traci.trafficlight.setPhase(intersection_id, phase_id)
            
            # Update tracking
            self.current_phases[intersection_id] = phase_id
            self.phase_start_times[intersection_id] = time.time()
            
        except Exception as e:
            self.logger.error(f"Error applying phase change to {intersection_id}: {e}")
    
    def adaptive_control(self, intersection_id: str):
        """Apply adaptive control based on current traffic conditions"""
        try:
            if intersection_id not in self.programs:
                return
            
            program = self.programs[intersection_id]
            current_phase = self.current_phases[intersection_id]
            phase = program.phases[current_phase]
            
            # Get current traffic data
            traffic_data = self._get_intersection_traffic_data(intersection_id)
            
            # Determine if phase should be extended or shortened
            if phase.phase_type in [PhaseType.NORTH_SOUTH_GREEN, PhaseType.EAST_WEST_GREEN]:
                # Check if phase should be extended
                if self._should_extend_phase(intersection_id, traffic_data):
                    extension = min(10.0, phase.max_duration - phase.duration)
                    if extension > 0:
                        traci.trafficlight.setPhaseDuration(intersection_id, phase.duration + extension)
                        self.logger.debug(f"Extended phase {current_phase} for {intersection_id} by {extension}s")
                
                # Check if phase should be shortened
                elif self._should_shorten_phase(intersection_id, traffic_data):
                    reduction = min(5.0, phase.duration - phase.min_duration)
                    if reduction > 0:
                        traci.trafficlight.setPhaseDuration(intersection_id, phase.duration - reduction)
                        self.logger.debug(f"Shortened phase {current_phase} for {intersection_id} by {reduction}s")
            
        except Exception as e:
            self.logger.error(f"Error in adaptive control for {intersection_id}: {e}")
    
    def _get_intersection_traffic_data(self, intersection_id: str) -> Dict[str, Any]:
        """Get current traffic data for intersection"""
        try:
            controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
            
            traffic_data = {
                'intersection_id': intersection_id,
                'controlled_lanes': controlled_lanes,
                'lane_vehicles': {},
                'lane_waiting': {},
                'lane_speed': {},
                'total_vehicles': 0,
                'total_waiting': 0
            }
            
            for lane in controlled_lanes:
                vehicles = traci.lane.getLastStepVehicleNumber(lane)
                waiting = traci.lane.getLastStepHaltingNumber(lane)
                speed = traci.lane.getLastStepMeanSpeed(lane)
                
                traffic_data['lane_vehicles'][lane] = vehicles
                traffic_data['lane_waiting'][lane] = waiting
                traffic_data['lane_speed'][lane] = speed
                traffic_data['total_vehicles'] += vehicles
                traffic_data['total_waiting'] += waiting
            
            return traffic_data
            
        except Exception as e:
            self.logger.error(f"Error getting traffic data for {intersection_id}: {e}")
            return {}
    
    def _should_extend_phase(self, intersection_id: str, traffic_data: Dict[str, Any]) -> bool:
        """Determine if current phase should be extended"""
        if not traffic_data:
            return False
        
        # Extend if there are waiting vehicles on the current green phase
        current_phase = self.current_phases[intersection_id]
        program = self.programs[intersection_id]
        phase = program.phases[current_phase]
        
        if phase.phase_type == PhaseType.NORTH_SOUTH_GREEN:
            # Check north-south lanes
            ns_lanes = [lane for lane in traffic_data['controlled_lanes'] if 'north' in lane.lower() or 'south' in lane.lower()]
            ns_waiting = sum(traffic_data['lane_waiting'].get(lane, 0) for lane in ns_lanes)
            return ns_waiting > 2
        
        elif phase.phase_type == PhaseType.EAST_WEST_GREEN:
            # Check east-west lanes
            ew_lanes = [lane for lane in traffic_data['controlled_lanes'] if 'east' in lane.lower() or 'west' in lane.lower()]
            ew_waiting = sum(traffic_data['lane_waiting'].get(lane, 0) for lane in ew_lanes)
            return ew_waiting > 2
        
        return False
    
    def _should_shorten_phase(self, intersection_id: str, traffic_data: Dict[str, Any]) -> bool:
        """Determine if current phase should be shortened"""
        if not traffic_data:
            return False
        
        # Shorten if no waiting vehicles on current green phase
        current_phase = self.current_phases[intersection_id]
        program = self.programs[intersection_id]
        phase = program.phases[current_phase]
        
        if phase.phase_type == PhaseType.NORTH_SOUTH_GREEN:
            ns_lanes = [lane for lane in traffic_data['controlled_lanes'] if 'north' in lane.lower() or 'south' in lane.lower()]
            ns_waiting = sum(traffic_data['lane_waiting'].get(lane, 0) for lane in ns_lanes)
            return ns_waiting == 0
        
        elif phase.phase_type == PhaseType.EAST_WEST_GREEN:
            ew_lanes = [lane for lane in traffic_data['controlled_lanes'] if 'east' in lane.lower() or 'west' in lane.lower()]
            ew_waiting = sum(traffic_data['lane_waiting'].get(lane, 0) for lane in ew_lanes)
            return ew_waiting == 0
        
        return False
    
    def emergency_vehicle_priority(self, intersection_id: str, emergency_vehicle_id: str):
        """Handle emergency vehicle priority"""
        try:
            # Get emergency vehicle position and route
            position = traci.vehicle.getPosition(emergency_vehicle_id)
            route = traci.vehicle.getRoute(emergency_vehicle_id)
            
            # Determine which phase should be green for emergency vehicle
            target_phase = self._determine_emergency_phase(intersection_id, position, route)
            
            if target_phase is not None:
                # Immediately switch to emergency phase
                traci.trafficlight.setPhase(intersection_id, target_phase)
                traci.trafficlight.setPhaseDuration(intersection_id, 30.0)  # Extended green time
                
                self.logger.info(f"Emergency vehicle {emergency_vehicle_id} priority activated at {intersection_id}")
                
        except Exception as e:
            self.logger.error(f"Error handling emergency vehicle priority: {e}")
    
    def _determine_emergency_phase(self, intersection_id: str, position: Tuple[float, float], route: List[str]) -> Optional[int]:
        """Determine which phase should be green for emergency vehicle"""
        # This is a simplified implementation
        # In practice, you would use more sophisticated routing analysis
        
        program = self.programs[intersection_id]
        
        # Check if emergency vehicle is approaching from north-south
        if any('north' in lane.lower() or 'south' in lane.lower() for lane in route):
            for i, phase in enumerate(program.phases):
                if phase.phase_type == PhaseType.NORTH_SOUTH_GREEN:
                    return i
        
        # Check if emergency vehicle is approaching from east-west
        elif any('east' in lane.lower() or 'west' in lane.lower() for lane in route):
            for i, phase in enumerate(program.phases):
                if phase.phase_type == PhaseType.EAST_WEST_GREEN:
                    return i
        
        return None
    
    def get_intersection_status(self, intersection_id: str) -> Dict[str, Any]:
        """Get current status of intersection"""
        try:
            if intersection_id not in self.programs:
                return {}
            
            program = self.programs[intersection_id]
            current_phase = self.current_phases[intersection_id]
            phase = program.phases[current_phase]
            
            # Get traffic data
            traffic_data = self._get_intersection_traffic_data(intersection_id)
            
            return {
                'intersection_id': intersection_id,
                'current_phase': current_phase,
                'phase_type': phase.phase_type.value,
                'phase_duration': phase.duration,
                'phase_start_time': self.phase_start_times[intersection_id],
                'traffic_data': traffic_data,
                'phase_switches': self.phase_switches.get(intersection_id, 0),
                'total_waiting_time': self.total_waiting_time.get(intersection_id, 0.0),
                'throughput': self.throughput.get(intersection_id, 0),
                'last_control_time': self.last_control_time.get(intersection_id, datetime.min).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting intersection status for {intersection_id}: {e}")
            return {}
    
    def get_all_intersections_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all intersections"""
        return {intersection_id: self.get_intersection_status(intersection_id) 
                for intersection_id in self.programs.keys()}
    
    def export_control_data(self, filepath: str):
        """Export traffic light control data"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'programs': {k: {
                    'program_id': v.program_id,
                    'phases': [{
                        'phase_id': p.phase_id,
                        'phase_type': p.phase_type.value,
                        'duration': p.duration,
                        'min_duration': p.min_duration,
                        'max_duration': p.max_duration,
                        'state': p.state,
                        'description': p.description
                    } for p in v.phases],
                    'cycle_time': v.cycle_time,
                    'offset': v.offset,
                    'type': v.type
                } for k, v in self.programs.items()},
                'current_phases': self.current_phases,
                'phase_start_times': self.phase_start_times,
                'performance': {
                    'phase_switches': self.phase_switches,
                    'total_waiting_time': self.total_waiting_time,
                    'throughput': self.throughput
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Traffic light control data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting control data: {e}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create traffic light controller
    controller = TrafficLightController()
    
    # Initialize with sample intersections
    intersections = ["intersection_1", "intersection_2"]
    controller.initialize_traffic_lights(intersections)
    
    # Apply ML control
    control_signal = ControlSignal(
        intersection_id="intersection_1",
        phase_id=0,
        duration=45.0,
        confidence=0.85,
        timestamp=datetime.now(),
        algorithm_used="q_learning"
    )
    
    controller.apply_ml_control(control_signal)
    
    # Get status
    status = controller.get_intersection_status("intersection_1")
    print(f"Intersection status: {status}")
    
    # Export data
    controller.export_control_data("traffic_light_control.json")
