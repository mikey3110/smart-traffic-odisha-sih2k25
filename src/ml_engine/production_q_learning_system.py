"""
Production-Ready Multi-Intersection Q-Learning System
Integrates all components for real-time traffic signal optimization
"""

import asyncio
import logging
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import os
from pathlib import Path
import numpy as np
import torch

# Import all components
from algorithms.advanced_q_learning_agent import AdvancedQLearningAgent, MultiDimensionalState, SophisticatedAction
from algorithms.multi_intersection_coordinator import MultiIntersectionCoordinator
from algorithms.advanced_reward_function import AdvancedRewardFunction, RewardComponents
from algorithms.adaptive_experience_replay import AdaptiveExperienceReplay, ExperienceReplayAnalyzer
from training.advanced_training_pipeline import AdvancedTrainingPipeline, TrainingScenario
from config.ml_config import get_config


class ProductionQLearningSystem:
    """
    Production-ready Q-Learning system for multi-intersection traffic optimization
    
    Features:
    - Real-time optimization with 30-second cycles
    - Multi-intersection coordination
    - Advanced reward function
    - Adaptive experience replay
    - Automated training pipeline
    - Model checkpointing and versioning
    - Performance monitoring
    - Error handling and recovery
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = get_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # System state
        self.is_running = False
        self.is_training = False
        self.optimization_cycle = 0
        self.start_time = None
        
        # Performance monitoring
        self.performance_metrics = {
            'total_cycles': 0,
            'successful_cycles': 0,
            'failed_cycles': 0,
            'avg_cycle_time': 0.0,
            'avg_reward': 0.0,
            'total_reward': 0.0,
            'optimization_history': []
        }
        
        # Error handling
        self.error_count = 0
        self.max_errors = 10
        self.last_error_time = None
        
        self.logger.info("Production Q-Learning system initialized")
    
    def _initialize_components(self):
        """Initialize all system components"""
        # Get intersection configuration
        intersections = self.config.get('intersections', [])
        
        # Initialize agents and coordinators for each intersection
        self.agents = {}
        self.coordinators = {}
        self.reward_functions = {}
        self.replay_buffers = {}
        
        for intersection in intersections:
            intersection_id = intersection['id']
            
            # Create agent
            agent_config = self.config.get('q_learning', {}).copy()
            agent_config['intersection_id'] = intersection_id
            agent_config['adjacent_intersections'] = intersection.get('adjacent_intersections', [])
            
            self.agents[intersection_id] = AdvancedQLearningAgent(intersection_id, agent_config)
            
            # Create coordinator
            self.coordinators[intersection_id] = MultiIntersectionCoordinator(
                intersection_id, agent_config
            )
            
            # Create reward function
            reward_config = self.config.get('reward_function', {})
            self.reward_functions[intersection_id] = AdvancedRewardFunction(reward_config)
            
            # Create replay buffer
            replay_config = self.config.get('experience_replay', {})
            self.replay_buffers[intersection_id] = AdaptiveExperienceReplay(
                capacity=replay_config.get('capacity', 50000),
                alpha=replay_config.get('alpha', 0.6),
                beta=replay_config.get('beta', 0.4)
            )
            
            self.logger.info(f"Initialized components for {intersection_id}")
        
        # Initialize training pipeline
        self.training_pipeline = AdvancedTrainingPipeline(self.config)
        
        # Initialize system monitoring
        self.monitoring_thread = None
        self.optimization_thread = None
    
    async def start_system(self):
        """Start the production system"""
        if self.is_running:
            self.logger.warning("System is already running")
            return
        
        self.logger.info("Starting production Q-Learning system")
        self.is_running = True
        self.start_time = datetime.now()
        
        try:
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            
            # Start optimization loop
            self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
            self.optimization_thread.start()
            
            # Start online learning for all agents
            for agent in self.agents.values():
                agent.start_online_learning()
            
            self.logger.info("Production system started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            self.is_running = False
            raise
    
    async def stop_system(self):
        """Stop the production system"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping production Q-Learning system")
        self.is_running = False
        
        try:
            # Stop online learning
            for agent in self.agents.values():
                agent.stop_online_learning()
            
            # Wait for threads to finish
            if self.optimization_thread and self.optimization_thread.is_alive():
                self.optimization_thread.join(timeout=5)
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            # Save final models
            self._save_all_models()
            
            self.logger.info("Production system stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping system: {e}")
    
    def _optimization_loop(self):
        """Main optimization loop"""
        target_cycle_time = self.config.get('optimization', {}).get('cycle_time', 30.0)
        
        while self.is_running:
            try:
                cycle_start = time.time()
                
                # Run optimization cycle
                success = self._run_optimization_cycle()
                
                # Update performance metrics
                cycle_time = time.time() - cycle_start
                self._update_performance_metrics(success, cycle_time)
                
                # Calculate sleep time to maintain 30-second cycles
                sleep_time = max(0, target_cycle_time - cycle_time)
                time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                self._handle_error(e)
                time.sleep(5)  # Wait before retrying
    
    def _run_optimization_cycle(self) -> bool:
        """Run a single optimization cycle for all intersections"""
        try:
            self.optimization_cycle += 1
            cycle_start = datetime.now()
            
            # Collect traffic data for all intersections
            traffic_data = self._collect_traffic_data()
            
            # Run optimization for each intersection
            optimization_results = {}
            
            for intersection_id, agent in self.agents.items():
                try:
                    # Get current traffic data
                    current_data = traffic_data.get(intersection_id, {})
                    
                    # Create state
                    state = agent.create_state(
                        current_data, 
                        current_data.get('current_timings', {}),
                        current_data.get('historical_data', [])
                    )
                    
                    # Select action
                    action = agent.select_action(state, training=False)
                    
                    # Coordinate with other intersections
                    coordinator = self.coordinators[intersection_id]
                    coordinated_action = coordinator.coordinate_optimization(action.__dict__)
                    
                    # Convert back to SophisticatedAction
                    coordinated_action = SophisticatedAction(**coordinated_action)
                    
                    # Calculate reward
                    reward_function = self.reward_functions[intersection_id]
                    next_state = state  # Simplified - in real implementation, this would be the next state
                    
                    reward_components = reward_function.calculate_reward(
                        state, coordinated_action, next_state
                    )
                    
                    # Store results
                    optimization_results[intersection_id] = {
                        'action': coordinated_action,
                        'reward': reward_components.total_reward,
                        'reward_components': reward_components.to_dict(),
                        'state': state,
                        'success': True
                    }
                    
                    # Add experience to replay buffer
                    replay_buffer = self.replay_buffers[intersection_id]
                    replay_buffer.add_experience(
                        state, coordinated_action, reward_components.total_reward, 
                        next_state, False
                    )
                    
                    # Train agent
                    agent.train_step()
                    
                except Exception as e:
                    self.logger.error(f"Error optimizing {intersection_id}: {e}")
                    optimization_results[intersection_id] = {
                        'action': None,
                        'reward': 0.0,
                        'success': False,
                        'error': str(e)
                    }
            
            # Log cycle results
            successful_optimizations = sum(1 for result in optimization_results.values() if result['success'])
            total_reward = sum(result['reward'] for result in optimization_results.values())
            
            self.logger.info(
                f"Optimization cycle {self.optimization_cycle}: "
                f"{successful_optimizations}/{len(optimization_results)} successful, "
                f"Total reward: {total_reward:.3f}"
            )
            
            # Store optimization history
            self.performance_metrics['optimization_history'].append({
                'cycle': self.optimization_cycle,
                'timestamp': cycle_start.isoformat(),
                'successful_optimizations': successful_optimizations,
                'total_optimizations': len(optimization_results),
                'total_reward': total_reward,
                'results': optimization_results
            })
            
            return successful_optimizations > 0
            
        except Exception as e:
            self.logger.error(f"Error in optimization cycle: {e}")
            return False
    
    def _collect_traffic_data(self) -> Dict[str, Dict[str, Any]]:
        """Collect traffic data for all intersections"""
        # This is a mock implementation
        # In a real system, this would collect data from sensors, cameras, etc.
        
        traffic_data = {}
        
        for intersection_id in self.agents.keys():
            # Mock traffic data
            traffic_data[intersection_id] = {
                'intersection_id': intersection_id,
                'timestamp': datetime.now().isoformat(),
                'lane_counts': {
                    'north_lane': np.random.randint(5, 25),
                    'south_lane': np.random.randint(5, 25),
                    'east_lane': np.random.randint(5, 25),
                    'west_lane': np.random.randint(5, 25)
                },
                'avg_speed': np.random.uniform(20, 60),
                'weather_condition': 'clear',
                'temperature': np.random.uniform(15, 35),
                'visibility': np.random.uniform(5, 20),
                'emergency_vehicles': np.random.random() < 0.05,
                'current_timings': {
                    'north_lane': 30,
                    'south_lane': 30,
                    'east_lane': 30,
                    'west_lane': 30
                },
                'historical_data': []
            }
        
        return traffic_data
    
    def _update_performance_metrics(self, success: bool, cycle_time: float):
        """Update performance metrics"""
        self.performance_metrics['total_cycles'] += 1
        
        if success:
            self.performance_metrics['successful_cycles'] += 1
        else:
            self.performance_metrics['failed_cycles'] += 1
        
        # Update average cycle time
        total_cycles = self.performance_metrics['total_cycles']
        current_avg = self.performance_metrics['avg_cycle_time']
        self.performance_metrics['avg_cycle_time'] = (
            (current_avg * (total_cycles - 1) + cycle_time) / total_cycles
        )
    
    def _monitoring_loop(self):
        """System monitoring loop"""
        while self.is_running:
            try:
                # Monitor system health
                self._check_system_health()
                
                # Log performance metrics
                self._log_performance_metrics()
                
                # Check for errors
                self._check_error_conditions()
                
                # Sleep for monitoring interval
                time.sleep(60)  # Monitor every minute
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)
    
    def _check_system_health(self):
        """Check system health and performance"""
        # Check agent health
        for intersection_id, agent in self.agents.items():
            try:
                stats = agent.get_training_statistics()
                if stats['replay_buffer_size'] == 0:
                    self.logger.warning(f"Empty replay buffer for {intersection_id}")
            except Exception as e:
                self.logger.error(f"Health check failed for {intersection_id}: {e}")
        
        # Check coordinator health
        for intersection_id, coordinator in self.coordinators.items():
            try:
                metrics = coordinator.get_coordination_metrics()
                if metrics['pending_messages'] > 100:
                    self.logger.warning(f"High message queue for {intersection_id}")
            except Exception as e:
                self.logger.error(f"Coordinator health check failed for {intersection_id}: {e}")
    
    def _log_performance_metrics(self):
        """Log current performance metrics"""
        if self.performance_metrics['total_cycles'] % 10 == 0:  # Log every 10 cycles
            success_rate = (
                self.performance_metrics['successful_cycles'] / 
                self.performance_metrics['total_cycles']
            )
            
            self.logger.info(
                f"Performance: {self.performance_metrics['total_cycles']} cycles, "
                f"{success_rate:.2%} success rate, "
                f"{self.performance_metrics['avg_cycle_time']:.2f}s avg cycle time"
            )
    
    def _check_error_conditions(self):
        """Check for error conditions and handle them"""
        current_time = time.time()
        
        # Check for too many recent errors
        if self.error_count > self.max_errors:
            if (self.last_error_time is None or 
                current_time - self.last_error_time > 300):  # 5 minutes
                self.logger.error("Too many errors, restarting system components")
                self._restart_components()
                self.error_count = 0
                self.last_error_time = current_time
    
    def _handle_error(self, error: Exception):
        """Handle system errors"""
        self.error_count += 1
        self.last_error_time = time.time()
        
        self.logger.error(f"System error #{self.error_count}: {error}")
        
        # Log error details
        if self.error_count <= 3:  # Only log details for first few errors
            self.logger.error(f"Error details: {error}", exc_info=True)
    
    def _restart_components(self):
        """Restart system components"""
        try:
            self.logger.info("Restarting system components")
            
            # Restart agents
            for agent in self.agents.values():
                agent.stop_online_learning()
                agent.start_online_learning()
            
            self.logger.info("System components restarted successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to restart components: {e}")
    
    def _save_all_models(self):
        """Save all trained models"""
        try:
            models_dir = Path(self.config.get('model_save_dir', 'models'))
            models_dir.mkdir(parents=True, exist_ok=True)
            
            for intersection_id, agent in self.agents.items():
                model_path = models_dir / f"{intersection_id}_model.pth"
                agent.save_model(str(model_path))
                self.logger.info(f"Saved model for {intersection_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def start_training(self, num_episodes: int = 100):
        """Start training process"""
        if self.is_training:
            self.logger.warning("Training is already in progress")
            return
        
        self.logger.info(f"Starting training for {num_episodes} episodes")
        self.is_training = True
        
        try:
            # Start training pipeline
            self.training_pipeline.start_training(num_episodes)
            
            # Load best models after training
            for intersection_id in self.agents.keys():
                self.training_pipeline.load_best_model(intersection_id)
            
            self.logger.info("Training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        finally:
            self.is_training = False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'is_running': self.is_running,
            'is_training': self.is_training,
            'optimization_cycle': self.optimization_cycle,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'performance_metrics': self.performance_metrics.copy(),
            'error_count': self.error_count,
            'last_error_time': self.last_error_time
        }
        
        # Add agent status
        status['agents'] = {}
        for intersection_id, agent in self.agents.items():
            try:
                status['agents'][intersection_id] = agent.get_training_statistics()
            except Exception as e:
                status['agents'][intersection_id] = {'error': str(e)}
        
        # Add coordinator status
        status['coordinators'] = {}
        for intersection_id, coordinator in self.coordinators.items():
            try:
                status['coordinators'][intersection_id] = coordinator.get_coordination_metrics()
            except Exception as e:
                status['coordinators'][intersection_id] = {'error': str(e)}
        
        return status
    
    def get_optimization_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent optimization history"""
        return self.performance_metrics['optimization_history'][-limit:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics"""
        metrics = self.performance_metrics.copy()
        
        # Add reward function statistics
        metrics['reward_functions'] = {}
        for intersection_id, reward_function in self.reward_functions.items():
            try:
                metrics['reward_functions'][intersection_id] = reward_function.get_reward_statistics()
            except Exception as e:
                metrics['reward_functions'][intersection_id] = {'error': str(e)}
        
        # Add replay buffer statistics
        metrics['replay_buffers'] = {}
        for intersection_id, replay_buffer in self.replay_buffers.items():
            try:
                metrics['replay_buffers'][intersection_id] = replay_buffer.get_statistics()
            except Exception as e:
                metrics['replay_buffers'][intersection_id] = {'error': str(e)}
        
        return metrics


# Example usage and testing
async def main():
    """Example usage of the production Q-Learning system"""
    # Initialize system
    system = ProductionQLearningSystem()
    
    try:
        # Start system
        await system.start_system()
        
        # Run for some time
        await asyncio.sleep(300)  # Run for 5 minutes
        
        # Get status
        status = system.get_system_status()
        print(f"System status: {json.dumps(status, indent=2)}")
        
        # Get performance metrics
        metrics = system.get_performance_metrics()
        print(f"Performance metrics: {json.dumps(metrics, indent=2)}")
        
    except KeyboardInterrupt:
        print("Stopping system...")
    finally:
        # Stop system
        await system.stop_system()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run system
    asyncio.run(main())
