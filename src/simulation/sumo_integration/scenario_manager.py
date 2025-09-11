"""
Scenario Management System for SUMO Integration
Manages simulation scenarios including peak hours, incidents, and weather effects
"""

import traci
import traci.constants as tc
import logging
import time
import json
import random
import math
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
import threading
import xml.etree.ElementTree as ET


class ScenarioType(Enum):
    """Scenario types"""
    NORMAL = "normal"
    PEAK_HOUR = "peak_hour"
    INCIDENT = "incident"
    WEATHER = "weather"
    CONSTRUCTION = "construction"
    SPECIAL_EVENT = "special_event"
    EMERGENCY = "emergency"


class WeatherCondition(Enum):
    """Weather conditions"""
    CLEAR = "clear"
    RAIN = "rain"
    SNOW = "snow"
    FOG = "fog"
    STORM = "storm"


class IncidentType(Enum):
    """Incident types"""
    ACCIDENT = "accident"
    BREAKDOWN = "breakdown"
    ROAD_CLOSURE = "road_closure"
    CONSTRUCTION = "construction"
    EMERGENCY_VEHICLE = "emergency_vehicle"


@dataclass
class ScenarioConfig:
    """Scenario configuration"""
    name: str
    description: str
    scenario_type: ScenarioType
    duration: float  # seconds
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Traffic parameters
    base_flow_rate: float = 100.0  # vehicles per hour
    peak_multiplier: float = 1.5
    off_peak_multiplier: float = 0.3
    
    # Weather effects
    weather_condition: WeatherCondition = WeatherCondition.CLEAR
    visibility_reduction: float = 0.0  # 0.0 = no reduction, 1.0 = complete reduction
    speed_reduction: float = 0.0  # 0.0 = no reduction, 1.0 = complete reduction
    
    # Incident parameters
    incident_probability: float = 0.0
    incident_types: List[IncidentType] = field(default_factory=list)
    incident_duration: float = 300.0  # seconds
    
    # Special parameters
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveIncident:
    """Active incident data"""
    incident_id: str
    incident_type: IncidentType
    location: Tuple[float, float]
    affected_lanes: List[str]
    start_time: datetime
    duration: float
    severity: float  # 0.0 = minor, 1.0 = major
    description: str


@dataclass
class ScenarioState:
    """Current scenario state"""
    scenario_name: str
    start_time: datetime
    current_time: datetime
    elapsed_time: float
    progress: float  # 0.0 to 1.0
    active_incidents: List[ActiveIncident]
    weather_condition: WeatherCondition
    current_flow_rate: float
    performance_metrics: Dict[str, float]


class ScenarioManager:
    """
    Scenario Management System for SUMO Integration
    
    Features:
    - Multiple scenario types
    - Weather effects simulation
    - Incident management
    - Peak hour simulation
    - Real-time scenario switching
    - Performance monitoring
    - Custom scenario creation
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Scenario management
        self.scenarios: Dict[str, ScenarioConfig] = {}
        self.current_scenario: Optional[ScenarioConfig] = None
        self.scenario_state: Optional[ScenarioState] = None
        
        # Incident management
        self.active_incidents: Dict[str, ActiveIncident] = {}
        self.incident_counter = 0
        
        # Weather effects
        self.current_weather = WeatherCondition.CLEAR
        self.weather_effects_active = False
        
        # Control
        self.is_running = False
        self.control_thread = None
        self.update_interval = 1.0  # seconds
        
        # Performance tracking
        self.scenario_performance: Dict[str, Dict[str, float]] = {}
        
        # Initialize default scenarios
        self._create_default_scenarios()
        
        self.logger.info("Scenario Manager initialized")
    
    def _create_default_scenarios(self):
        """Create default scenario configurations"""
        # Normal traffic scenario
        self.scenarios["normal"] = ScenarioConfig(
            name="Normal Traffic",
            description="Normal traffic conditions",
            scenario_type=ScenarioType.NORMAL,
            duration=3600.0,  # 1 hour
            base_flow_rate=200.0,
            peak_multiplier=1.2,
            off_peak_multiplier=0.8
        )
        
        # Morning peak hour
        self.scenarios["morning_peak"] = ScenarioConfig(
            name="Morning Peak Hour",
            description="Heavy traffic during morning commute",
            scenario_type=ScenarioType.PEAK_HOUR,
            duration=7200.0,  # 2 hours
            base_flow_rate=400.0,
            peak_multiplier=2.5,
            off_peak_multiplier=0.3,
            start_time=datetime.now().replace(hour=7, minute=0, second=0),
            end_time=datetime.now().replace(hour=9, minute=0, second=0)
        )
        
        # Evening peak hour
        self.scenarios["evening_peak"] = ScenarioConfig(
            name="Evening Peak Hour",
            description="Heavy traffic during evening commute",
            scenario_type=ScenarioType.PEAK_HOUR,
            duration=7200.0,  # 2 hours
            base_flow_rate=400.0,
            peak_multiplier=2.5,
            off_peak_multiplier=0.3,
            start_time=datetime.now().replace(hour=17, minute=0, second=0),
            end_time=datetime.now().replace(hour=19, minute=0, second=0)
        )
        
        # Rain scenario
        self.scenarios["rain"] = ScenarioConfig(
            name="Rainy Weather",
            description="Traffic with rain effects",
            scenario_type=ScenarioType.WEATHER,
            duration=3600.0,  # 1 hour
            base_flow_rate=150.0,
            weather_condition=WeatherCondition.RAIN,
            visibility_reduction=0.3,
            speed_reduction=0.2
        )
        
        # Accident scenario
        self.scenarios["accident"] = ScenarioConfig(
            name="Traffic Accident",
            description="Traffic with accident incident",
            scenario_type=ScenarioType.INCIDENT,
            duration=1800.0,  # 30 minutes
            base_flow_rate=100.0,
            incident_probability=0.1,
            incident_types=[IncidentType.ACCIDENT],
            incident_duration=600.0  # 10 minutes
        )
        
        # Construction scenario
        self.scenarios["construction"] = ScenarioConfig(
            name="Road Construction",
            description="Traffic with construction work",
            scenario_type=ScenarioType.CONSTRUCTION,
            duration=14400.0,  # 4 hours
            base_flow_rate=80.0,
            incident_probability=0.05,
            incident_types=[IncidentType.CONSTRUCTION],
            incident_duration=3600.0  # 1 hour
        )
    
    def start_scenario(self, scenario_name: str) -> bool:
        """Start a scenario"""
        try:
            if scenario_name not in self.scenarios:
                self.logger.error(f"Scenario {scenario_name} not found")
                return False
            
            # Stop current scenario if running
            if self.is_running:
                self.stop_scenario()
            
            # Set current scenario
            self.current_scenario = self.scenarios[scenario_name]
            
            # Initialize scenario state
            self.scenario_state = ScenarioState(
                scenario_name=scenario_name,
                start_time=datetime.now(),
                current_time=datetime.now(),
                elapsed_time=0.0,
                progress=0.0,
                active_incidents=[],
                weather_condition=self.current_scenario.weather_condition,
                current_flow_rate=self.current_scenario.base_flow_rate,
                performance_metrics={}
            )
            
            # Start control thread
            self.is_running = True
            self.control_thread = threading.Thread(target=self._control_loop)
            self.control_thread.daemon = True
            self.control_thread.start()
            
            # Apply initial scenario effects
            self._apply_scenario_effects()
            
            self.logger.info(f"Started scenario: {scenario_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting scenario {scenario_name}: {e}")
            return False
    
    def stop_scenario(self):
        """Stop current scenario"""
        self.is_running = False
        
        if self.control_thread and self.control_thread.is_alive():
            self.control_thread.join(timeout=5)
        
        # Clear active incidents
        self.active_incidents.clear()
        
        # Reset weather effects
        self._reset_weather_effects()
        
        self.current_scenario = None
        self.scenario_state = None
        
        self.logger.info("Scenario stopped")
    
    def _control_loop(self):
        """Main scenario control loop"""
        while self.is_running and self.current_scenario and self.scenario_state:
            try:
                # Update scenario state
                self._update_scenario_state()
                
                # Check for scenario end
                if self.scenario_state.progress >= 1.0:
                    self.stop_scenario()
                    break
                
                # Update scenario effects
                self._update_scenario_effects()
                
                # Handle incidents
                self._handle_incidents()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                self.logger.error(f"Error in scenario control loop: {e}")
                time.sleep(1.0)
    
    def _update_scenario_state(self):
        """Update current scenario state"""
        if not self.scenario_state:
            return
        
        current_time = datetime.now()
        self.scenario_state.current_time = current_time
        self.scenario_state.elapsed_time = (current_time - self.scenario_state.start_time).total_seconds()
        self.scenario_state.progress = min(1.0, self.scenario_state.elapsed_time / self.current_scenario.duration)
        
        # Update active incidents
        self.scenario_state.active_incidents = list(self.active_incidents.values())
    
    def _update_scenario_effects(self):
        """Update scenario effects based on current state"""
        if not self.current_scenario or not self.scenario_state:
            return
        
        # Update flow rate based on time and scenario type
        current_flow_rate = self._calculate_current_flow_rate()
        self.scenario_state.current_flow_rate = current_flow_rate
        
        # Apply weather effects
        if self.current_scenario.weather_condition != WeatherCondition.CLEAR:
            self._apply_weather_effects()
        
        # Apply peak hour effects
        if self.current_scenario.scenario_type == ScenarioType.PEAK_HOUR:
            self._apply_peak_hour_effects()
    
    def _calculate_current_flow_rate(self) -> float:
        """Calculate current flow rate based on scenario and time"""
        if not self.current_scenario:
            return 100.0
        
        base_rate = self.current_scenario.base_flow_rate
        
        # Apply time-based multipliers
        if self.current_scenario.scenario_type == ScenarioType.PEAK_HOUR:
            current_hour = datetime.now().hour
            
            # Check if in peak hours
            if (self.current_scenario.start_time and 
                self.current_scenario.end_time and
                self.current_scenario.start_time.hour <= current_hour < self.current_scenario.end_time.hour):
                return base_rate * self.current_scenario.peak_multiplier
            else:
                return base_rate * self.current_scenario.off_peak_multiplier
        
        # Apply weather effects
        if self.current_scenario.weather_condition != WeatherCondition.CLEAR:
            weather_multiplier = 1.0 - (self.current_scenario.visibility_reduction * 0.5)
            return base_rate * weather_multiplier
        
        return base_rate
    
    def _apply_scenario_effects(self):
        """Apply initial scenario effects"""
        if not self.current_scenario:
            return
        
        # Apply weather effects
        if self.current_scenario.weather_condition != WeatherCondition.CLEAR:
            self._apply_weather_effects()
        
        # Apply incident effects
        if self.current_scenario.incident_probability > 0:
            self._create_initial_incidents()
    
    def _apply_weather_effects(self):
        """Apply weather effects to simulation"""
        try:
            if self.current_scenario.weather_condition == WeatherCondition.RAIN:
                # Reduce visibility and speed
                self._reduce_visibility(self.current_scenario.visibility_reduction)
                self._reduce_speed(self.current_scenario.speed_reduction)
            
            elif self.current_scenario.weather_condition == WeatherCondition.SNOW:
                # More severe effects
                self._reduce_visibility(self.current_scenario.visibility_reduction * 1.5)
                self._reduce_speed(self.current_scenario.speed_reduction * 1.5)
            
            elif self.current_scenario.weather_condition == WeatherCondition.FOG:
                # Visibility reduction only
                self._reduce_visibility(self.current_scenario.visibility_reduction)
            
            self.weather_effects_active = True
            
        except Exception as e:
            self.logger.error(f"Error applying weather effects: {e}")
    
    def _reduce_visibility(self, reduction_factor: float):
        """Reduce visibility in simulation"""
        try:
            # This would require SUMO configuration changes
            # For now, we'll simulate by reducing detection range
            self.logger.info(f"Reducing visibility by {reduction_factor * 100}%")
            
        except Exception as e:
            self.logger.error(f"Error reducing visibility: {e}")
    
    def _reduce_speed(self, reduction_factor: float):
        """Reduce speed limits in simulation"""
        try:
            # Get all lanes and reduce speed
            lane_ids = traci.lane.getIDList()
            
            for lane_id in lane_ids:
                try:
                    current_speed = traci.lane.getMaxSpeed(lane_id)
                    new_speed = current_speed * (1.0 - reduction_factor)
                    traci.lane.setMaxSpeed(lane_id, new_speed)
                except:
                    continue
            
            self.logger.info(f"Reduced speed limits by {reduction_factor * 100}%")
            
        except Exception as e:
            self.logger.error(f"Error reducing speed: {e}")
    
    def _apply_peak_hour_effects(self):
        """Apply peak hour effects"""
        try:
            # Increase vehicle generation rate
            # This would be handled by the traffic demand generator
            self.logger.debug("Applying peak hour effects")
            
        except Exception as e:
            self.logger.error(f"Error applying peak hour effects: {e}")
    
    def _handle_incidents(self):
        """Handle incident creation and management"""
        if not self.current_scenario or self.current_scenario.incident_probability <= 0:
            return
        
        try:
            # Check if new incident should be created
            if random.random() < self.current_scenario.incident_probability * self.update_interval:
                self._create_incident()
            
            # Update existing incidents
            self._update_incidents()
            
        except Exception as e:
            self.logger.error(f"Error handling incidents: {e}")
    
    def _create_incident(self):
        """Create a new incident"""
        try:
            # Select incident type
            incident_type = random.choice(self.current_scenario.incident_types)
            
            # Select location (random lane)
            lane_ids = traci.lane.getIDList()
            if not lane_ids:
                return
            
            affected_lane = random.choice(lane_ids)
            location = traci.lane.getPosition(affected_lane)
            
            # Create incident
            incident_id = f"incident_{self.incident_counter}"
            self.incident_counter += 1
            
            incident = ActiveIncident(
                incident_id=incident_id,
                incident_type=incident_type,
                location=location,
                affected_lanes=[affected_lane],
                start_time=datetime.now(),
                duration=self.current_scenario.incident_duration,
                severity=random.uniform(0.3, 1.0),
                description=f"{incident_type.value} on {affected_lane}"
            )
            
            self.active_incidents[incident_id] = incident
            
            # Apply incident effects
            self._apply_incident_effects(incident)
            
            self.logger.info(f"Created incident: {incident.description}")
            
        except Exception as e:
            self.logger.error(f"Error creating incident: {e}")
    
    def _update_incidents(self):
        """Update existing incidents"""
        incidents_to_remove = []
        
        for incident_id, incident in self.active_incidents.items():
            # Check if incident has expired
            elapsed = (datetime.now() - incident.start_time).total_seconds()
            if elapsed >= incident.duration:
                incidents_to_remove.append(incident_id)
                self._remove_incident_effects(incident)
            else:
                # Update incident effects
                self._update_incident_effects(incident)
        
        # Remove expired incidents
        for incident_id in incidents_to_remove:
            del self.active_incidents[incident_id]
            self.logger.info(f"Removed expired incident: {incident_id}")
    
    def _apply_incident_effects(self, incident: ActiveIncident):
        """Apply effects of an incident"""
        try:
            if incident.incident_type == IncidentType.ACCIDENT:
                # Block affected lanes
                for lane_id in incident.affected_lanes:
                    traci.lane.setMaxSpeed(lane_id, 0.0)
            
            elif incident.incident_type == IncidentType.BREAKDOWN:
                # Reduce speed on affected lanes
                for lane_id in incident.affected_lanes:
                    current_speed = traci.lane.getMaxSpeed(lane_id)
                    new_speed = current_speed * (1.0 - incident.severity * 0.5)
                    traci.lane.setMaxSpeed(lane_id, new_speed)
            
            elif incident.incident_type == IncidentType.ROAD_CLOSURE:
                # Close affected lanes
                for lane_id in incident.affected_lanes:
                    traci.lane.setMaxSpeed(lane_id, 0.0)
            
        except Exception as e:
            self.logger.error(f"Error applying incident effects: {e}")
    
    def _update_incident_effects(self, incident: ActiveIncident):
        """Update effects of an incident"""
        # This could include dynamic effects like traffic buildup
        pass
    
    def _remove_incident_effects(self, incident: ActiveIncident):
        """Remove effects of an incident"""
        try:
            # Restore normal conditions
            for lane_id in incident.affected_lanes:
                # Restore normal speed (this would need to be stored)
                traci.lane.setMaxSpeed(lane_id, 50.0)  # Default speed
            
        except Exception as e:
            self.logger.error(f"Error removing incident effects: {e}")
    
    def _create_initial_incidents(self):
        """Create initial incidents for scenario"""
        if not self.current_scenario:
            return
        
        # Create a few initial incidents
        num_incidents = random.randint(1, 3)
        for _ in range(num_incidents):
            self._create_incident()
    
    def _update_performance_metrics(self):
        """Update performance metrics for current scenario"""
        if not self.scenario_state:
            return
        
        try:
            # Collect performance data
            total_vehicles = len(traci.vehicle.getIDList())
            waiting_vehicles = len([v for v in traci.vehicle.getIDList() if traci.vehicle.getSpeed(v) <= 0.1])
            total_waiting_time = sum(traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList())
            average_speed = np.mean([traci.vehicle.getSpeed(v) for v in traci.vehicle.getIDList()]) if traci.vehicle.getIDList() else 0.0
            
            self.scenario_state.performance_metrics = {
                'total_vehicles': total_vehicles,
                'waiting_vehicles': waiting_vehicles,
                'total_waiting_time': total_waiting_time,
                'average_speed': average_speed,
                'active_incidents': len(self.active_incidents)
            }
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _reset_weather_effects(self):
        """Reset weather effects"""
        try:
            # Restore normal conditions
            lane_ids = traci.lane.getIDList()
            for lane_id in lane_ids:
                traci.lane.setMaxSpeed(lane_id, 50.0)  # Default speed
            
            self.weather_effects_active = False
            
        except Exception as e:
            self.logger.error(f"Error resetting weather effects: {e}")
    
    def get_scenario_status(self) -> Optional[Dict[str, Any]]:
        """Get current scenario status"""
        if not self.scenario_state:
            return None
        
        return {
            'scenario_name': self.scenario_state.scenario_name,
            'start_time': self.scenario_state.start_time.isoformat(),
            'current_time': self.scenario_state.current_time.isoformat(),
            'elapsed_time': self.scenario_state.elapsed_time,
            'progress': self.scenario_state.progress,
            'active_incidents': len(self.scenario_state.active_incidents),
            'weather_condition': self.scenario_state.weather_condition.value,
            'current_flow_rate': self.scenario_state.current_flow_rate,
            'performance_metrics': self.scenario_state.performance_metrics
        }
    
    def get_available_scenarios(self) -> List[str]:
        """Get list of available scenarios"""
        return list(self.scenarios.keys())
    
    def create_custom_scenario(self, config: ScenarioConfig) -> bool:
        """Create a custom scenario"""
        try:
            self.scenarios[config.name] = config
            self.logger.info(f"Created custom scenario: {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating custom scenario: {e}")
            return False
    
    def export_scenario_data(self, filepath: str):
        """Export scenario data"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'available_scenarios': list(self.scenarios.keys()),
                'current_scenario': self.get_scenario_status(),
                'scenario_configs': {name: {
                    'name': config.name,
                    'description': config.description,
                    'scenario_type': config.scenario_type.value,
                    'duration': config.duration,
                    'base_flow_rate': config.base_flow_rate,
                    'peak_multiplier': config.peak_multiplier,
                    'off_peak_multiplier': config.off_peak_multiplier,
                    'weather_condition': config.weather_condition.value,
                    'incident_probability': config.incident_probability
                } for name, config in self.scenarios.items()}
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Scenario data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting scenario data: {e}")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create scenario manager
    manager = ScenarioManager()
    
    # List available scenarios
    scenarios = manager.get_available_scenarios()
    print(f"Available scenarios: {scenarios}")
    
    # Start a scenario
    if scenarios:
        success = manager.start_scenario(scenarios[0])
        if success:
            print(f"Started scenario: {scenarios[0]}")
            
            # Run for some time
            time.sleep(30)
            
            # Get status
            status = manager.get_scenario_status()
            print(f"Scenario status: {status}")
            
            # Export data
            manager.export_scenario_data("scenario_data.json")
        
        # Stop scenario
        manager.stop_scenario()
