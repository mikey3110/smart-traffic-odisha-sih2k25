"""
Traffic Demand Pattern Generator for SUMO Integration
Generates realistic traffic demand patterns, routes, and scenarios
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
import xml.etree.ElementTree as ET


class DemandPattern(Enum):
    """Traffic demand patterns"""
    UNIFORM = "uniform"
    PEAK_HOUR = "peak_hour"
    RUSH_HOUR = "rush_hour"
    INCIDENT = "incident"
    WEATHER = "weather"
    WEEKEND = "weekend"
    CUSTOM = "custom"


class VehicleType(Enum):
    """Vehicle types for demand generation"""
    PASSENGER = "passenger"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    EMERGENCY = "emergency"


@dataclass
class DemandProfile:
    """Traffic demand profile"""
    pattern: DemandPattern
    base_flow_rate: float  # vehicles per hour
    peak_multiplier: float = 1.5
    off_peak_multiplier: float = 0.3
    peak_start_hour: int = 7
    peak_end_hour: int = 9
    evening_peak_start: int = 17
    evening_peak_end: int = 19
    vehicle_type_distribution: Dict[VehicleType, float] = field(default_factory=dict)
    speed_distribution: Tuple[float, float] = (30.0, 50.0)  # min, max speed in km/h


@dataclass
class RouteProfile:
    """Route generation profile"""
    origin_lanes: List[str]
    destination_lanes: List[str]
    route_probabilities: Dict[str, float]  # route_id -> probability
    turn_probabilities: Dict[str, float]  # turn_type -> probability
    detour_probability: float = 0.1


@dataclass
class ScenarioProfile:
    """Scenario generation profile"""
    name: str
    description: str
    duration: float  # seconds
    demand_profiles: Dict[str, DemandProfile]  # lane_id -> profile
    route_profiles: Dict[str, RouteProfile]  # intersection_id -> profile
    incident_probability: float = 0.01
    weather_effects: bool = False
    special_events: List[str] = field(default_factory=list)


class TrafficDemandGenerator:
    """
    Traffic Demand Pattern Generator for SUMO Integration
    
    Features:
    - Realistic traffic demand patterns
    - Dynamic route generation
    - Peak hour simulation
    - Incident and weather effects
    - Vehicle type distribution
    - Custom scenario creation
    - Real-time demand adjustment
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Demand profiles
        self.demand_profiles: Dict[str, DemandProfile] = {}
        self.route_profiles: Dict[str, RouteProfile] = {}
        self.scenario_profiles: Dict[str, ScenarioProfile] = {}
        
        # Current scenario
        self.current_scenario: Optional[ScenarioProfile] = None
        self.scenario_start_time: Optional[datetime] = None
        
        # Vehicle generation
        self.vehicle_counter = 0
        self.generation_interval = 1.0  # seconds
        self.is_generating = False
        self.generation_thread = None
        
        # Statistics
        self.generated_vehicles = 0
        self.completed_trips = 0
        self.failed_trips = 0
        
        # Load configuration
        if config_file:
            self.load_configuration(config_file)
        else:
            self._create_default_profiles()
        
        self.logger.info("Traffic Demand Generator initialized")
    
    def _create_default_profiles(self):
        """Create default demand profiles"""
        # Default vehicle type distribution
        vehicle_types = {
            VehicleType.PASSENGER: 0.7,
            VehicleType.TRUCK: 0.15,
            VehicleType.BUS: 0.1,
            VehicleType.MOTORCYCLE: 0.05
        }
        
        # Create demand profiles
        self.demand_profiles = {
            "low_traffic": DemandProfile(
                pattern=DemandPattern.UNIFORM,
                base_flow_rate=100.0,
                peak_multiplier=1.2,
                off_peak_multiplier=0.5,
                vehicle_type_distribution=vehicle_types
            ),
            "medium_traffic": DemandProfile(
                pattern=DemandPattern.PEAK_HOUR,
                base_flow_rate=300.0,
                peak_multiplier=2.0,
                off_peak_multiplier=0.4,
                vehicle_type_distribution=vehicle_types
            ),
            "high_traffic": DemandProfile(
                pattern=DemandPattern.RUSH_HOUR,
                base_flow_rate=500.0,
                peak_multiplier=3.0,
                off_peak_multiplier=0.3,
                vehicle_type_distribution=vehicle_types
            ),
            "incident_traffic": DemandProfile(
                pattern=DemandPattern.INCIDENT,
                base_flow_rate=200.0,
                peak_multiplier=0.5,
                off_peak_multiplier=0.2,
                vehicle_type_distribution=vehicle_types
            )
        }
        
        # Create scenario profiles
        self.scenario_profiles = {
            "morning_rush": ScenarioProfile(
                name="Morning Rush Hour",
                description="Heavy traffic during morning commute",
                duration=7200.0,  # 2 hours
                demand_profiles={"all": self.demand_profiles["high_traffic"]},
                route_profiles={},
                incident_probability=0.02
            ),
            "evening_rush": ScenarioProfile(
                name="Evening Rush Hour",
                description="Heavy traffic during evening commute",
                duration=7200.0,  # 2 hours
                demand_profiles={"all": self.demand_profiles["high_traffic"]},
                route_profiles={},
                incident_probability=0.02
            ),
            "weekend_traffic": ScenarioProfile(
                name="Weekend Traffic",
                description="Moderate traffic on weekends",
                duration=14400.0,  # 4 hours
                demand_profiles={"all": self.demand_profiles["medium_traffic"]},
                route_profiles={},
                incident_probability=0.005
            ),
            "incident_scenario": ScenarioProfile(
                name="Traffic Incident",
                description="Traffic with incident effects",
                duration=3600.0,  # 1 hour
                demand_profiles={"all": self.demand_profiles["incident_traffic"]},
                route_profiles={},
                incident_probability=0.1
            )
        }
    
    def load_configuration(self, config_file: str):
        """Load configuration from file"""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Load demand profiles
            for name, profile_data in config.get('demand_profiles', {}).items():
                self.demand_profiles[name] = DemandProfile(**profile_data)
            
            # Load scenario profiles
            for name, scenario_data in config.get('scenario_profiles', {}).items():
                self.scenario_profiles[name] = ScenarioProfile(**scenario_data)
            
            self.logger.info(f"Configuration loaded from {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            self._create_default_profiles()
    
    def save_configuration(self, config_file: str):
        """Save configuration to file"""
        try:
            config = {
                'demand_profiles': {name: profile.__dict__ for name, profile in self.demand_profiles.items()},
                'scenario_profiles': {name: scenario.__dict__ for name, scenario in self.scenario_profiles.items()}
            }
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Configuration saved to {config_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration: {e}")
    
    def start_scenario(self, scenario_name: str) -> bool:
        """Start traffic demand scenario"""
        try:
            if scenario_name not in self.scenario_profiles:
                self.logger.error(f"Scenario {scenario_name} not found")
                return False
            
            self.current_scenario = self.scenario_profiles[scenario_name]
            self.scenario_start_time = datetime.now()
            
            # Start vehicle generation
            self.is_generating = True
            self.generation_thread = threading.Thread(target=self._generation_loop)
            self.generation_thread.daemon = True
            self.generation_thread.start()
            
            self.logger.info(f"Started scenario: {scenario_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting scenario {scenario_name}: {e}")
            return False
    
    def stop_scenario(self):
        """Stop current scenario"""
        self.is_generating = False
        if self.generation_thread:
            self.generation_thread.join(timeout=5)
        
        self.current_scenario = None
        self.scenario_start_time = None
        
        self.logger.info("Scenario stopped")
    
    def _generation_loop(self):
        """Main vehicle generation loop"""
        while self.is_generating and self.current_scenario:
            try:
                # Check if scenario has ended
                if self.scenario_start_time:
                    elapsed = (datetime.now() - self.scenario_start_time).total_seconds()
                    if elapsed >= self.current_scenario.duration:
                        self.stop_scenario()
                        break
                
                # Generate vehicles for each demand profile
                for lane_id, profile in self.current_scenario.demand_profiles.items():
                    self._generate_vehicles_for_lane(lane_id, profile)
                
                time.sleep(self.generation_interval)
                
            except Exception as e:
                self.logger.error(f"Error in generation loop: {e}")
                time.sleep(1.0)
    
    def _generate_vehicles_for_lane(self, lane_id: str, profile: DemandProfile):
        """Generate vehicles for specific lane"""
        try:
            # Calculate current flow rate based on time of day
            current_flow_rate = self._calculate_current_flow_rate(profile)
            
            # Calculate probability of generating vehicle this step
            generation_probability = current_flow_rate / 3600.0  # Convert to per-second probability
            
            if random.random() < generation_probability:
                # Generate vehicle
                vehicle_id = f"vehicle_{self.vehicle_counter}"
                self.vehicle_counter += 1
                
                # Select vehicle type
                vehicle_type = self._select_vehicle_type(profile.vehicle_type_distribution)
                
                # Generate route
                route = self._generate_route(lane_id)
                
                if route:
                    # Add vehicle to simulation
                    self._add_vehicle_to_simulation(vehicle_id, vehicle_type, route)
                    self.generated_vehicles += 1
                
        except Exception as e:
            self.logger.error(f"Error generating vehicles for lane {lane_id}: {e}")
    
    def _calculate_current_flow_rate(self, profile: DemandProfile) -> float:
        """Calculate current flow rate based on time and profile"""
        if not self.scenario_start_time:
            return profile.base_flow_rate
        
        current_time = datetime.now()
        hour = current_time.hour
        
        # Apply time-based multipliers
        if profile.pattern == DemandPattern.PEAK_HOUR:
            if profile.peak_start_hour <= hour < profile.peak_end_hour:
                return profile.base_flow_rate * profile.peak_multiplier
            elif profile.evening_peak_start <= hour < profile.evening_peak_end:
                return profile.base_flow_rate * profile.peak_multiplier
            else:
                return profile.base_flow_rate * profile.off_peak_multiplier
        
        elif profile.pattern == DemandPattern.RUSH_HOUR:
            if 7 <= hour < 9 or 17 <= hour < 19:
                return profile.base_flow_rate * profile.peak_multiplier
            else:
                return profile.base_flow_rate * profile.off_peak_multiplier
        
        elif profile.pattern == DemandPattern.INCIDENT:
            # Reduced flow due to incident
            return profile.base_flow_rate * 0.5
        
        else:
            return profile.base_flow_rate
    
    def _select_vehicle_type(self, distribution: Dict[VehicleType, float]) -> VehicleType:
        """Select vehicle type based on distribution"""
        rand = random.random()
        cumulative = 0.0
        
        for vehicle_type, probability in distribution.items():
            cumulative += probability
            if rand <= cumulative:
                return vehicle_type
        
        return VehicleType.PASSENGER  # Default fallback
    
    def _generate_route(self, origin_lane: str) -> Optional[List[str]]:
        """Generate route for vehicle"""
        try:
            # Get all lanes in network
            all_lanes = traci.lane.getIDList()
            
            # Simple route generation - find destination lane
            destination_lanes = [lane for lane in all_lanes if lane != origin_lane]
            
            if not destination_lanes:
                return None
            
            # Select random destination
            destination = random.choice(destination_lanes)
            
            # Generate route using SUMO's routing
            try:
                route = traci.simulation.findRoute(origin_lane, destination)
                if route and len(route) > 0:
                    return route
            except:
                pass
            
            # Fallback: simple route
            return [origin_lane, destination]
            
        except Exception as e:
            self.logger.warning(f"Error generating route: {e}")
            return None
    
    def _add_vehicle_to_simulation(self, vehicle_id: str, vehicle_type: VehicleType, route: List[str]):
        """Add vehicle to SUMO simulation"""
        try:
            # Get vehicle type string
            vehicle_type_str = vehicle_type.value
            
            # Add vehicle
            traci.vehicle.add(vehicle_id, route, typeID=vehicle_type_str)
            
            # Set random speed
            speed = random.uniform(30.0, 50.0)  # km/h
            traci.vehicle.setSpeed(vehicle_id, speed)
            
            self.logger.debug(f"Added vehicle {vehicle_id} of type {vehicle_type_str}")
            
        except Exception as e:
            self.logger.warning(f"Error adding vehicle {vehicle_id}: {e}")
            self.failed_trips += 1
    
    def create_custom_scenario(self, name: str, description: str, duration: float,
                             demand_config: Dict[str, Any]) -> bool:
        """Create custom traffic scenario"""
        try:
            # Create demand profiles
            demand_profiles = {}
            for lane_id, config in demand_config.items():
                profile = DemandProfile(
                    pattern=DemandPattern(config.get('pattern', 'uniform')),
                    base_flow_rate=config.get('base_flow_rate', 100.0),
                    peak_multiplier=config.get('peak_multiplier', 1.5),
                    off_peak_multiplier=config.get('off_peak_multiplier', 0.3),
                    vehicle_type_distribution={
                        VehicleType(k): v for k, v in config.get('vehicle_types', {}).items()
                    }
                )
                demand_profiles[lane_id] = profile
            
            # Create scenario
            scenario = ScenarioProfile(
                name=name,
                description=description,
                duration=duration,
                demand_profiles=demand_profiles,
                route_profiles={},
                incident_probability=demand_config.get('incident_probability', 0.01),
                weather_effects=demand_config.get('weather_effects', False)
            )
            
            self.scenario_profiles[name] = scenario
            
            self.logger.info(f"Created custom scenario: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating custom scenario: {e}")
            return False
    
    def get_scenario_status(self) -> Dict[str, Any]:
        """Get current scenario status"""
        if not self.current_scenario:
            return {'active': False}
        
        elapsed = 0.0
        if self.scenario_start_time:
            elapsed = (datetime.now() - self.scenario_start_time).total_seconds()
        
        return {
            'active': True,
            'scenario_name': self.current_scenario.name,
            'elapsed_time': elapsed,
            'duration': self.current_scenario.duration,
            'progress': min(100.0, (elapsed / self.current_scenario.duration) * 100.0),
            'generated_vehicles': self.generated_vehicles,
            'completed_trips': self.completed_trips,
            'failed_trips': self.failed_trips
        }
    
    def get_available_scenarios(self) -> List[str]:
        """Get list of available scenarios"""
        return list(self.scenario_profiles.keys())
    
    def export_scenario_data(self, filepath: str):
        """Export scenario data"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'scenario_status': self.get_scenario_status(),
                'available_scenarios': self.get_available_scenarios(),
                'demand_profiles': {name: profile.__dict__ for name, profile in self.demand_profiles.items()},
                'scenario_profiles': {name: scenario.__dict__ for name, scenario in self.scenario_profiles.items()}
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
    
    # Create demand generator
    generator = TrafficDemandGenerator()
    
    # List available scenarios
    scenarios = generator.get_available_scenarios()
    print(f"Available scenarios: {scenarios}")
    
    # Start a scenario
    if scenarios:
        success = generator.start_scenario(scenarios[0])
        if success:
            print(f"Started scenario: {scenarios[0]}")
            
            # Run for some time
            time.sleep(30)
            
            # Get status
            status = generator.get_scenario_status()
            print(f"Scenario status: {status}")
            
            # Export data
            generator.export_scenario_data("scenario_data.json")
        
        # Stop scenario
        generator.stop_scenario()
