"""
SUMO Integration Manager
Main orchestrator for SUMO simulation integration with backend API
"""

import asyncio
import logging
import time
import json
import signal
import sys
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import threading
from pathlib import Path

# Import SUMO integration components
from sumo_controller import SumoController
from traffic_light_controller import TrafficLightController
from vehicle_detector import VehicleDetector
from traffic_demand_generator import TrafficDemandGenerator
from data_exporter import DataExporter
from scenario_manager import ScenarioManager
from visualization.sumo_visualizer import SumoVisualizer
from validation.validation_tools import SimulationValidator
from config.sumo_config import SumoConfig, get_sumo_config


class IntegrationState(Enum):
    """Integration states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class IntegrationConfig:
    """Integration configuration"""
    sumo_config: SumoConfig
    enable_visualization: bool = True
    enable_validation: bool = True
    enable_data_export: bool = True
    enable_scenario_management: bool = True
    log_level: str = "INFO"
    auto_restart: bool = True
    max_restart_attempts: int = 3


class SumoIntegrationManager:
    """
    SUMO Integration Manager
    
    Main orchestrator that coordinates all SUMO integration components:
    - SUMO simulation control
    - Traffic light management
    - Vehicle detection and counting
    - Traffic demand generation
    - Data export and API integration
    - Scenario management
    - Visualization
    - Validation and calibration
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        self.config = config or IntegrationConfig(sumo_config=get_sumo_config())
        self.logger = logging.getLogger(__name__)
        
        # Set up logging
        self._setup_logging()
        
        # Integration state
        self.state = IntegrationState.STOPPED
        self.restart_count = 0
        self.last_error = None
        
        # Component instances
        self.sumo_controller: Optional[SumoController] = None
        self.traffic_light_controller: Optional[TrafficLightController] = None
        self.vehicle_detector: Optional[VehicleDetector] = None
        self.demand_generator: Optional[TrafficDemandGenerator] = None
        self.data_exporter: Optional[DataExporter] = None
        self.scenario_manager: Optional[ScenarioManager] = None
        self.visualizer: Optional[SumoVisualizer] = None
        self.validator: Optional[SimulationValidator] = None
        
        # Control
        self.is_running = False
        self.control_thread = None
        self.shutdown_event = threading.Event()
        
        # Performance tracking
        self.start_time: Optional[datetime] = None
        self.performance_metrics: Dict[str, Any] = {}
        
        # Set up signal handlers
        self._setup_signal_handlers()
        
        self.logger.info("SUMO Integration Manager initialized")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('logs/sumo_integration.log')
            ]
        )
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
            self.stop_integration()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start_integration(self, scenario_file: str = None) -> bool:
        """Start SUMO integration"""
        try:
            if self.state == IntegrationState.RUNNING:
                self.logger.warning("Integration already running")
                return True
            
            self.state = IntegrationState.STARTING
            self.logger.info("Starting SUMO integration...")
            
            # Initialize components
            if not await self._initialize_components():
                self.state = IntegrationState.ERROR
                return False
            
            # Start components
            if not await self._start_components(scenario_file):
                self.state = IntegrationState.ERROR
                return False
            
            # Start control loop
            self.is_running = True
            self.control_thread = threading.Thread(target=self._control_loop)
            self.control_thread.daemon = True
            self.control_thread.start()
            
            self.state = IntegrationState.RUNNING
            self.start_time = datetime.now()
            self.restart_count = 0
            
            self.logger.info("SUMO integration started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting integration: {e}")
            self.state = IntegrationState.ERROR
            self.last_error = str(e)
            return False
    
    def stop_integration(self):
        """Stop SUMO integration"""
        try:
            if self.state == IntegrationState.STOPPED:
                return
            
            self.state = IntegrationState.STOPPING
            self.logger.info("Stopping SUMO integration...")
            
            # Stop control loop
            self.is_running = False
            self.shutdown_event.set()
            
            if self.control_thread and self.control_thread.is_alive():
                self.control_thread.join(timeout=10)
            
            # Stop components
            self._stop_components()
            
            self.state = IntegrationState.STOPPED
            self.logger.info("SUMO integration stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping integration: {e}")
            self.state = IntegrationState.ERROR
    
    async def _initialize_components(self) -> bool:
        """Initialize all components"""
        try:
            # Initialize SUMO controller
            self.sumo_controller = SumoController(self.config.sumo_config)
            
            # Initialize traffic light controller
            self.traffic_light_controller = TrafficLightController(self.config.sumo_config)
            
            # Initialize vehicle detector
            self.vehicle_detector = VehicleDetector(detection_range=100.0)
            
            # Initialize demand generator
            self.demand_generator = TrafficDemandGenerator()
            
            # Initialize data exporter
            if self.config.enable_data_export:
                from data_exporter import ExportConfig
                export_config = ExportConfig(
                    enabled=True,
                    api_endpoint=self.config.sumo_config.api_integration.api_endpoint
                )
                self.data_exporter = DataExporter(export_config)
            
            # Initialize scenario manager
            if self.config.enable_scenario_management:
                self.scenario_manager = ScenarioManager()
            
            # Initialize visualizer
            if self.config.enable_visualization:
                from visualization.sumo_visualizer import VisualizationConfig
                viz_config = VisualizationConfig(interactive=True)
                self.visualizer = SumoVisualizer(viz_config)
            
            # Initialize validator
            if self.config.enable_validation:
                from validation.validation_tools import ValidationConfig
                validation_config = ValidationConfig()
                self.validator = SimulationValidator(validation_config)
            
            self.logger.info("All components initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            return False
    
    async def _start_components(self, scenario_file: str = None) -> bool:
        """Start all components"""
        try:
            # Start SUMO simulation
            if not await self.sumo_controller.start_simulation(scenario_file):
                self.logger.error("Failed to start SUMO simulation")
                return False
            
            # Initialize traffic lights
            if self.sumo_controller.traffic_lights:
                self.traffic_light_controller.initialize_traffic_lights(
                    self.sumo_controller.traffic_lights
                )
            
            # Initialize detection zones
            if self.sumo_controller.traffic_lights:
                self.vehicle_detector.initialize_detection_zones(
                    self.sumo_controller.traffic_lights
                )
                self.vehicle_detector.start_detection()
            
            # Start data export
            if self.data_exporter:
                await self.data_exporter.start_export()
            
            # Start visualization
            if self.visualizer:
                self.visualizer.start_real_time_visualization()
            
            # Start validation
            if self.validator:
                self.validator.start_validation()
            
            # Start scenario if specified
            if self.scenario_manager and scenario_file:
                # Extract scenario name from file path
                scenario_name = Path(scenario_file).stem
                if scenario_name in self.scenario_manager.get_available_scenarios():
                    self.scenario_manager.start_scenario(scenario_name)
            
            self.logger.info("All components started")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting components: {e}")
            return False
    
    def _stop_components(self):
        """Stop all components"""
        try:
            # Stop validation
            if self.validator:
                self.validator.stop_validation()
            
            # Stop visualization
            if self.visualizer:
                self.visualizer.stop_visualization()
            
            # Stop data export
            if self.data_exporter:
                self.data_exporter.stop_export()
            
            # Stop vehicle detection
            if self.vehicle_detector:
                self.vehicle_detector.stop_detection()
            
            # Stop scenario
            if self.scenario_manager:
                self.scenario_manager.stop_scenario()
            
            # Stop SUMO simulation
            if self.sumo_controller:
                self.sumo_controller.stop_simulation()
            
            self.logger.info("All components stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping components: {e}")
    
    def _control_loop(self):
        """Main control loop"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Update performance metrics
                self._update_performance_metrics()
                
                # Check component health
                if not self._check_component_health():
                    self.logger.warning("Component health check failed")
                    if self.config.auto_restart:
                        self._restart_integration()
                
                # Apply ML control if available
                self._apply_ml_control()
                
                # Update visualization data
                if self.visualizer:
                    self._update_visualization_data()
                
                time.sleep(1.0)  # Control loop frequency
                
            except Exception as e:
                self.logger.error(f"Error in control loop: {e}")
                if self.config.auto_restart:
                    self._restart_integration()
                else:
                    self.state = IntegrationState.ERROR
                    break
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            if self.sumo_controller:
                status = self.sumo_controller.get_simulation_status()
                self.performance_metrics.update(status)
            
            if self.vehicle_detector:
                stats = self.vehicle_detector.get_detection_statistics()
                self.performance_metrics['detection'] = stats
            
            if self.data_exporter:
                stats = self.data_exporter.get_export_statistics()
                self.performance_metrics['export'] = stats
            
            if self.validator:
                stats = self.validator.get_validation_statistics()
                self.performance_metrics['validation'] = stats
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _check_component_health(self) -> bool:
        """Check health of all components"""
        try:
            # Check SUMO controller
            if self.sumo_controller and self.sumo_controller.state.value == "error":
                return False
            
            # Check other components
            # Add more health checks as needed
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking component health: {e}")
            return False
    
    def _restart_integration(self):
        """Restart integration"""
        if self.restart_count >= self.config.max_restart_attempts:
            self.logger.error("Maximum restart attempts reached")
            self.state = IntegrationState.ERROR
            return
        
        self.restart_count += 1
        self.logger.info(f"Restarting integration (attempt {self.restart_count})")
        
        # Stop current integration
        self.stop_integration()
        
        # Wait before restart
        time.sleep(5)
        
        # Start again
        asyncio.create_task(self.start_integration())
    
    def _apply_ml_control(self):
        """Apply ML control to traffic lights"""
        try:
            if not self.traffic_light_controller or not self.sumo_controller:
                return
            
            # Get current intersection data
            intersections = self.sumo_controller.get_all_intersections_data()
            
            for intersection_id, intersection_data in intersections.items():
                # Apply adaptive control
                self.traffic_light_controller.adaptive_control(intersection_id)
                
                # Here you would integrate with ML optimizer
                # For now, we'll use basic adaptive control
                
        except Exception as e:
            self.logger.error(f"Error applying ML control: {e}")
    
    def _update_visualization_data(self):
        """Update visualization data"""
        try:
            if not self.visualizer or not self.sumo_controller:
                return
            
            # Collect current simulation data
            data = {
                'vehicles': list(self.sumo_controller.get_all_vehicles_data().values()),
                'intersections': list(self.sumo_controller.get_all_intersections_data().values()),
                'performance_metrics': [self.sumo_controller.get_simulation_metrics()] if self.sumo_controller.get_simulation_metrics() else []
            }
            
            # Add to visualizer
            self.visualizer.add_data_point(data)
            
        except Exception as e:
            self.logger.error(f"Error updating visualization data: {e}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status"""
        return {
            'state': self.state.value,
            'is_running': self.is_running,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'restart_count': self.restart_count,
            'last_error': self.last_error,
            'performance_metrics': self.performance_metrics,
            'components': {
                'sumo_controller': self.sumo_controller is not None,
                'traffic_light_controller': self.traffic_light_controller is not None,
                'vehicle_detector': self.vehicle_detector is not None,
                'demand_generator': self.demand_generator is not None,
                'data_exporter': self.data_exporter is not None,
                'scenario_manager': self.scenario_manager is not None,
                'visualizer': self.visualizer is not None,
                'validator': self.validator is not None
            }
        }
    
    def export_integration_data(self, filepath: str):
        """Export integration data"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'integration_status': self.get_integration_status(),
                'configuration': {
                    'enable_visualization': self.config.enable_visualization,
                    'enable_validation': self.config.enable_validation,
                    'enable_data_export': self.config.enable_data_export,
                    'enable_scenario_management': self.config.enable_scenario_management,
                    'log_level': self.config.log_level,
                    'auto_restart': self.config.auto_restart
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Integration data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting integration data: {e}")


# Example usage
if __name__ == "__main__":
    # Create integration manager
    config = IntegrationConfig(
        sumo_config=get_sumo_config(),
        enable_visualization=True,
        enable_validation=True,
        enable_data_export=True,
        enable_scenario_management=True
    )
    
    manager = SumoIntegrationManager(config)
    
    # Start integration
    success = asyncio.run(manager.start_integration("scenarios/basic_scenario.sumocfg"))
    
    if success:
        print("SUMO integration started successfully")
        
        try:
            # Run for some time
            time.sleep(60)
            
            # Get status
            status = manager.get_integration_status()
            print(f"Integration status: {status}")
            
            # Export data
            manager.export_integration_data("integration_data.json")
        
        finally:
            # Stop integration
            manager.stop_integration()
    
    else:
        print("Failed to start SUMO integration")
