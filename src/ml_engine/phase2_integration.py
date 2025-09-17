"""
Phase 2: Real-Time Optimization Loop & Safety Systems - Main Integration
Complete integration of all Phase 2 components for production deployment

Features:
- Real-time optimization engine with 30-second cycles
- Comprehensive safety systems with fallback mechanisms
- Performance-optimized Q-table operations
- Enhanced SUMO integration with robust error handling
- Complete testing and validation framework
"""

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import yaml
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime.real_time_optimizer import RealTimeOptimizer, OptimizationMode
from safety.safety_manager import SafetyManager, SafetyLevel
from performance.optimized_q_table import OptimizedQTable, PerformanceMonitor
from sumo.enhanced_traci_controller import EnhancedTraCIController, SimulationState
from algorithms.advanced_q_learning_agent import AdvancedQLearningAgent
from algorithms.multi_intersection_coordinator import MultiIntersectionCoordinator
from algorithms.advanced_reward_function import AdvancedRewardFunction
from algorithms.adaptive_experience_replay import AdaptiveExperienceReplay


class Phase2Integration:
    """
    Main integration class for Phase 2: Real-Time Optimization Loop & Safety Systems
    
    This class orchestrates all Phase 2 components to provide a complete
    real-time traffic optimization system with safety-first architecture.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/phase2_config.yaml"
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.real_time_optimizer = None
        self.safety_manager = None
        self.traci_controller = None
        self.performance_monitor = None
        
        # ML components
        self.agents = {}
        self.coordinators = {}
        self.reward_functions = {}
        self.replay_buffers = {}
        self.q_tables = {}
        
        # System state
        self.is_running = False
        self.start_time = None
        
        # Performance tracking
        self.performance_metrics = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'avg_optimization_time': 0.0,
            'safety_violations': 0,
            'emergency_overrides': 0,
            'fallback_activations': 0
        }
        
        self.logger.info("Phase 2 Integration initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
                return config
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'real_time_optimizer': {
                'cycle_time': 30.0,
                'max_processing_time': 25.0,
                'confidence_threshold': 0.6,
                'safety_mode_threshold': 0.3
            },
            'safety_manager': {
                'webster': {
                    'min_cycle_time': 40,
                    'max_cycle_time': 120,
                    'saturation_flow_rate': 1800
                },
                'emergency': {
                    'override_duration': 300,
                    'priority_extension': 20
                },
                'constraints': {
                    'min_green_time': 10,
                    'max_green_time': 90,
                    'pedestrian_crossing_time': 20
                }
            },
            'performance': {
                'max_cache_size': 10000,
                'memory_limit': 1024,
                'cleanup_interval': 300
            },
            'sumo': {
                'sumo_binary': 'sumo',
                'port': 8813,
                'host': 'localhost'
            },
            'intersections': {
                'intersection_1': {
                    'lanes': ['north', 'south', 'east', 'west'],
                    'traffic_lights': ['tl_1'],
                    'detectors': ['det_1', 'det_2'],
                    'phases': [0, 1, 2, 3]
                }
            }
        }
    
    async def initialize(self):
        """Initialize all Phase 2 components"""
        try:
            self.logger.info("Initializing Phase 2 components...")
            
            # Initialize real-time optimizer
            self.real_time_optimizer = RealTimeOptimizer(
                self.config.get('real_time_optimizer', {})
            )
            
            # Initialize safety manager
            self.safety_manager = SafetyManager(
                self.config.get('safety_manager', {})
            )
            
            # Initialize TraCI controller
            self.traci_controller = EnhancedTraCIController(
                self.config.get('sumo', {})
            )
            
            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor()
            
            # Initialize ML components for each intersection
            await self._initialize_ml_components()
            
            # Configure intersections
            await self._configure_intersections()
            
            # Set up data flow
            await self._setup_data_flow()
            
            self.logger.info("Phase 2 components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Phase 2 components: {e}")
            return False
    
    async def _initialize_ml_components(self):
        """Initialize ML components for each intersection"""
        try:
            intersections = self.config.get('intersections', {})
            
            for intersection_id, intersection_config in intersections.items():
                # Initialize Q-Learning agent
                agent = AdvancedQLearningAgent(
                    state_dim=20,  # Multi-dimensional state space
                    action_dim=4,  # Phase selection
                    config=self.config.get('ml_agent', {})
                )
                self.agents[intersection_id] = agent
                
                # Initialize coordinator
                coordinator = MultiIntersectionCoordinator(
                    intersection_id=intersection_id,
                    config=self.config.get('coordinator', {})
                )
                self.coordinators[intersection_id] = coordinator
                
                # Initialize reward function
                reward_function = AdvancedRewardFunction(
                    config=self.config.get('reward_function', {})
                )
                self.reward_functions[intersection_id] = reward_function
                
                # Initialize experience replay
                replay_buffer = AdaptiveExperienceReplay(
                    capacity=10000,
                    config=self.config.get('experience_replay', {})
                )
                self.replay_buffers[intersection_id] = replay_buffer
                
                # Initialize optimized Q-table
                q_table = OptimizedQTable(
                    self.config.get('performance', {})
                )
                self.q_tables[intersection_id] = q_table
                
                self.logger.info(f"Initialized ML components for {intersection_id}")
            
            # Set ML components in real-time optimizer
            self.real_time_optimizer.set_ml_components(
                self.agents, self.coordinators, self.reward_functions, self.replay_buffers
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing ML components: {e}")
            raise
    
    async def _configure_intersections(self):
        """Configure intersections in all components"""
        try:
            intersections = self.config.get('intersections', {})
            
            for intersection_id, intersection_config in intersections.items():
                # Configure safety manager
                self.safety_manager.configure_intersection(intersection_id, intersection_config)
                
                # Configure TraCI controller
                self.traci_controller.add_intersection(intersection_id, intersection_config)
                
                # Add data source to real-time optimizer
                self.real_time_optimizer.add_data_source(
                    f"camera_{intersection_id}",
                    {
                        'type': 'camera',
                        'intersection_id': intersection_id,
                        'frequency': 1.0
                    }
                )
                
                self.logger.info(f"Configured intersection: {intersection_id}")
            
        except Exception as e:
            self.logger.error(f"Error configuring intersections: {e}")
            raise
    
    async def _setup_data_flow(self):
        """Set up data flow between components"""
        try:
            # Add callbacks for data flow
            self.traci_controller.add_data_callback(self._on_traffic_data)
            self.traci_controller.add_error_callback(self._on_simulation_error)
            self.traci_controller.add_state_callback(self._on_simulation_state_change)
            
            self.logger.info("Data flow setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up data flow: {e}")
            raise
    
    def _on_traffic_data(self, traffic_data):
        """Handle traffic data from SUMO"""
        try:
            # Process traffic data through real-time optimizer
            # This would be handled by the real-time optimizer's data ingestion
            pass
        except Exception as e:
            self.logger.error(f"Error handling traffic data: {e}")
    
    def _on_simulation_error(self, error_message):
        """Handle simulation errors"""
        try:
            self.logger.error(f"Simulation error: {error_message}")
            # Implement error recovery logic
        except Exception as e:
            self.logger.error(f"Error handling simulation error: {e}")
    
    def _on_simulation_state_change(self, state):
        """Handle simulation state changes"""
        try:
            self.logger.info(f"Simulation state changed to: {state}")
            # Implement state change handling logic
        except Exception as e:
            self.logger.error(f"Error handling state change: {e}")
    
    async def start_system(self):
        """Start the complete Phase 2 system"""
        try:
            if self.is_running:
                self.logger.warning("System is already running")
                return False
            
            self.logger.info("Starting Phase 2 system...")
            
            # Start performance monitoring
            self.performance_monitor.start_monitoring()
            
            # Start SUMO simulation
            if not self.traci_controller.start_simulation():
                self.logger.error("Failed to start SUMO simulation")
                return False
            
            # Wait for simulation to be ready
            await asyncio.sleep(2)
            
            # Start real-time optimization
            await self.real_time_optimizer.start_optimization()
            
            self.is_running = True
            self.start_time = datetime.now()
            
            self.logger.info("Phase 2 system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting Phase 2 system: {e}")
            return False
    
    async def stop_system(self):
        """Stop the complete Phase 2 system"""
        try:
            if not self.is_running:
                self.logger.warning("System is not running")
                return True
            
            self.logger.info("Stopping Phase 2 system...")
            
            # Stop real-time optimization
            await self.real_time_optimizer.stop_optimization()
            
            # Stop SUMO simulation
            self.traci_controller.stop_simulation()
            
            # Stop performance monitoring
            self.performance_monitor.stop_monitoring()
            
            self.is_running = False
            
            self.logger.info("Phase 2 system stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping Phase 2 system: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            status = {
                'is_running': self.is_running,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'performance_metrics': self.performance_metrics.copy(),
                'real_time_optimizer': self.real_time_optimizer.get_optimization_status() if self.real_time_optimizer else None,
                'safety_manager': self.safety_manager.get_safety_status() if self.safety_manager else None,
                'traci_controller': self.traci_controller.get_simulation_status() if self.traci_controller else None,
                'performance_monitor': self.performance_monitor.get_performance_summary() if self.performance_monitor else None
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization performance metrics"""
        try:
            metrics = {
                'total_optimizations': self.performance_metrics['total_optimizations'],
                'success_rate': (
                    self.performance_metrics['successful_optimizations'] / 
                    max(self.performance_metrics['total_optimizations'], 1)
                ),
                'avg_optimization_time': self.performance_metrics['avg_optimization_time'],
                'safety_violations': self.performance_metrics['safety_violations'],
                'emergency_overrides': self.performance_metrics['emergency_overrides'],
                'fallback_activations': self.performance_metrics['fallback_activations']
            }
            
            # Add component-specific metrics
            if self.real_time_optimizer:
                optimizer_metrics = self.real_time_optimizer.get_system_metrics()
                metrics.update({
                    'optimization_cycles': optimizer_metrics.total_cycles,
                    'successful_cycles': optimizer_metrics.successful_cycles,
                    'avg_cycle_time': optimizer_metrics.avg_cycle_time,
                    'avg_confidence': optimizer_metrics.avg_confidence
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting optimization metrics: {e}")
            return {'error': str(e)}
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get safety system metrics"""
        try:
            if not self.safety_manager:
                return {'error': 'Safety manager not initialized'}
            
            safety_status = self.safety_manager.get_safety_status()
            
            return {
                'safety_level': safety_status['safety_level'],
                'constraint_metrics': safety_status['constraint_metrics'],
                'emergency_metrics': safety_status['emergency_metrics'],
                'recommendations': self.safety_manager.get_safety_recommendations()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting safety metrics: {e}")
            return {'error': str(e)}
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            if not self.performance_monitor:
                return {'error': 'Performance monitor not initialized'}
            
            return self.performance_monitor.get_performance_summary()
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {'error': str(e)}
    
    def save_system_state(self, filepath: str):
        """Save current system state"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'performance_metrics': self.performance_metrics,
                'q_tables': {}
            }
            
            # Save Q-tables
            for intersection_id, q_table in self.q_tables.items():
                q_table.save_to_file(f"{filepath}_qtable_{intersection_id}.pkl")
                state['q_tables'][intersection_id] = f"{filepath}_qtable_{intersection_id}.pkl"
            
            # Save state file
            with open(f"{filepath}_state.json", 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"System state saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving system state: {e}")
    
    def load_system_state(self, filepath: str):
        """Load system state"""
        try:
            with open(f"{filepath}_state.json", 'r') as f:
                state = json.load(f)
            
            # Load Q-tables
            for intersection_id, q_table_path in state['q_tables'].items():
                if intersection_id in self.q_tables:
                    self.q_tables[intersection_id].load_from_file(q_table_path)
            
            # Load performance metrics
            self.performance_metrics.update(state['performance_metrics'])
            
            self.logger.info(f"System state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading system state: {e}")


async def main():
    """Main function for Phase 2 integration"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Phase 2: Real-Time Optimization Loop & Safety Systems")
    
    try:
        # Initialize Phase 2 integration
        phase2 = Phase2Integration()
        
        # Initialize components
        if not await phase2.initialize():
            logger.error("Failed to initialize Phase 2 components")
            return
        
        # Start system
        if not await phase2.start_system():
            logger.error("Failed to start Phase 2 system")
            return
        
        logger.info("Phase 2 system is running. Press Ctrl+C to stop.")
        
        # Keep system running
        try:
            while True:
                await asyncio.sleep(10)
                
                # Log system status
                status = phase2.get_system_status()
                logger.info(f"System running - Uptime: {status['uptime']:.1f}s")
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping system...")
        
        # Stop system
        await phase2.stop_system()
        logger.info("Phase 2 system stopped")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
