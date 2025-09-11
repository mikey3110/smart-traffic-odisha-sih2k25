"""
SUMO Configuration Management
Configuration settings for SUMO simulation integration
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import yaml
import json


class SimulationMode(Enum):
    """Simulation modes"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    REPLAY = "replay"


class TrafficLightProgram(Enum):
    """Traffic light program types"""
    STATIC = "static"
    ACTUATED = "actuated"
    ADAPTIVE = "adaptive"
    ML_CONTROLLED = "ml_controlled"


@dataclass
class SumoNetworkConfig:
    """SUMO network configuration"""
    net_file: str = "networks/intersection.net.xml"
    route_file: str = "routes/vehicles.rou.xml"
    additional_file: str = "additional/traffic_lights.add.xml"
    gui_file: str = "gui/sumo_gui.xml"
    output_dir: str = "output/"
    log_file: str = "logs/sumo.log"


@dataclass
class SimulationConfig:
    """Simulation configuration"""
    mode: SimulationMode = SimulationMode.REAL_TIME
    step_size: float = 1.0
    end_time: float = 3600.0  # 1 hour
    random_seed: int = 42
    warmup_time: float = 300.0  # 5 minutes
    cooldown_time: float = 300.0  # 5 minutes
    enable_gui: bool = False
    gui_delay: int = 100  # milliseconds


@dataclass
class TrafficLightConfig:
    """Traffic light configuration"""
    program_type: TrafficLightProgram = TrafficLightProgram.ML_CONTROLLED
    min_phase_duration: float = 5.0
    max_phase_duration: float = 60.0
    yellow_time: float = 3.0
    all_red_time: float = 1.0
    detection_range: float = 50.0
    update_interval: float = 1.0


@dataclass
class VehicleConfig:
    """Vehicle configuration"""
    vehicle_types: List[str] = field(default_factory=lambda: ["passenger", "truck", "bus"])
    max_speed: float = 50.0  # km/h
    acceleration: float = 2.6  # m/s²
    deceleration: float = 4.5  # m/s²
    length: float = 4.0  # meters
    width: float = 1.8  # meters
    emission_class: str = "HBEFA3/PC_G_EU4"


@dataclass
class DataCollectionConfig:
    """Data collection configuration"""
    collect_vehicles: bool = True
    collect_intersections: bool = True
    collect_emissions: bool = True
    collect_fuel: bool = True
    collect_waiting_time: bool = True
    collect_speed: bool = True
    output_interval: float = 10.0  # seconds
    output_file: str = "simulation_output.xml"


@dataclass
class APIIntegrationConfig:
    """API integration configuration"""
    enabled: bool = True
    base_url: str = "http://localhost:8000"
    endpoint: str = "/api/v1/sumo/data"
    timeout: float = 10.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100
    export_interval: float = 10.0  # seconds


@dataclass
class ScenarioConfig:
    """Scenario configuration"""
    name: str = "default"
    description: str = "Default simulation scenario"
    peak_hour_multiplier: float = 1.5
    incident_probability: float = 0.01
    weather_effects: bool = True
    road_conditions: str = "dry"  # dry, wet, icy
    visibility: float = 1000.0  # meters


@dataclass
class SumoConfig:
    """Main SUMO configuration container"""
    # Core configurations
    network: SumoNetworkConfig = field(default_factory=SumoNetworkConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    traffic_lights: TrafficLightConfig = field(default_factory=TrafficLightConfig)
    vehicles: VehicleConfig = field(default_factory=VehicleConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    api_integration: APIIntegrationConfig = field(default_factory=APIIntegrationConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    
    # General settings
    sumo_binary: str = "sumo"
    sumo_gui_binary: str = "sumo-gui"
    working_directory: str = "simulation/"
    log_level: str = "INFO"
    debug_mode: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "network": self.network.__dict__,
            "simulation": self.simulation.__dict__,
            "traffic_lights": self.traffic_lights.__dict__,
            "vehicles": self.vehicles.__dict__,
            "data_collection": self.data_collection.__dict__,
            "api_integration": self.api_integration.__dict__,
            "scenario": self.scenario.__dict__,
            "sumo_binary": self.sumo_binary,
            "sumo_gui_binary": self.sumo_gui_binary,
            "working_directory": self.working_directory,
            "log_level": self.log_level,
            "debug_mode": self.debug_mode
        }
    
    def save_to_file(self, filepath: str, format: str = "yaml"):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format.lower() == "yaml":
            with open(filepath, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        elif format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'SumoConfig':
        """Load configuration from file"""
        if not os.path.exists(filepath):
            return cls()
        
        with open(filepath, 'r') as f:
            if filepath.endswith('.yaml') or filepath.endswith('.yml'):
                data = yaml.safe_load(f)
            elif filepath.endswith('.json'):
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
        
        # Create config instance
        config = cls()
        
        # Update with loaded data
        for key, value in data.items():
            if hasattr(config, key):
                if key == 'simulation' and 'mode' in value:
                    value['mode'] = SimulationMode(value['mode'])
                elif key == 'traffic_lights' and 'program_type' in value:
                    value['program_type'] = TrafficLightProgram(value['program_type'])
                setattr(config, key, value)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate network files
        if not os.path.exists(self.network.net_file):
            issues.append(f"Network file not found: {self.network.net_file}")
        
        if not os.path.exists(self.network.route_file):
            issues.append(f"Route file not found: {self.network.route_file}")
        
        # Validate simulation settings
        if self.simulation.step_size <= 0:
            issues.append("Simulation step size must be positive")
        
        if self.simulation.end_time <= 0:
            issues.append("Simulation end time must be positive")
        
        # Validate traffic light settings
        if self.traffic_lights.min_phase_duration >= self.traffic_lights.max_phase_duration:
            issues.append("Min phase duration must be less than max phase duration")
        
        # Validate API settings
        if self.api_integration.enabled and not self.api_integration.base_url:
            issues.append("API base URL is required when API integration is enabled")
        
        return issues


# Global configuration instance
sumo_config = SumoConfig()


def get_sumo_config() -> SumoConfig:
    """Get the global SUMO configuration"""
    return sumo_config


def load_sumo_config(filepath: str) -> SumoConfig:
    """Load configuration from file and update global config"""
    global sumo_config
    sumo_config = SumoConfig.load_from_file(filepath)
    return sumo_config


def save_sumo_config(filepath: str, format: str = "yaml") -> None:
    """Save current configuration to file"""
    sumo_config.save_to_file(filepath, format)


# Example usage
if __name__ == "__main__":
    # Create default configuration
    config = SumoConfig()
    
    # Validate configuration
    issues = config.validate()
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid")
    
    # Save configuration
    config.save_to_file("config/sumo_config.yaml")
    print("Configuration saved to config/sumo_config.yaml")
