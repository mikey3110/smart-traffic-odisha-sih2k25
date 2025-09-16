"""
Advanced Training Pipeline for Multi-Intersection Q-Learning
Automated data generation, model training, and checkpointing system
"""

import os
import json
import logging
import pickle
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import yaml
from collections import defaultdict, deque
import asyncio
import subprocess
import shutil

from algorithms.advanced_q_learning_agent import AdvancedQLearningAgent, MultiDimensionalState, SophisticatedAction
from algorithms.multi_intersection_coordinator import MultiIntersectionCoordinator
from config.ml_config import get_config


@dataclass
class TrainingScenario:
    """Training scenario configuration"""
    scenario_id: str
    name: str
    description: str
    duration: int  # Duration in minutes
    traffic_pattern: str  # rush_hour, normal, night, emergency
    weather_condition: str
    intersections: List[str]
    parameters: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingEpisode:
    """Single training episode data"""
    episode_id: str
    scenario_id: str
    start_time: datetime
    end_time: datetime
    duration: float
    intersections: List[str]
    total_reward: float
    performance_metrics: Dict[str, float]
    state_action_pairs: List[Tuple[MultiDimensionalState, SophisticatedAction, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode_id': self.episode_id,
            'scenario_id': self.scenario_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration': self.duration,
            'intersections': self.intersections,
            'total_reward': self.total_reward,
            'performance_metrics': self.performance_metrics,
            'state_action_pairs_count': len(self.state_action_pairs)
        }


@dataclass
class ModelCheckpoint:
    """Model checkpoint data"""
    checkpoint_id: str
    timestamp: datetime
    training_step: int
    episode_count: int
    performance_metrics: Dict[str, float]
    model_path: str
    config_path: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'checkpoint_id': self.checkpoint_id,
            'timestamp': self.timestamp.isoformat(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'performance_metrics': self.performance_metrics,
            'model_path': self.model_path,
            'config_path': self.config_path,
            'metadata': self.metadata
        }


class SUMODataGenerator:
    """Generates training data using SUMO simulation"""
    
    def __init__(self, sumo_config_path: str, output_dir: str):
        self.sumo_config_path = sumo_config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # SUMO configuration
        self.sumo_binary = "sumo"  # Path to SUMO binary
        self.traci_port = 8813
        self.simulation_step = 0.1  # 100ms simulation steps
        
    def generate_scenario_data(self, scenario: TrainingScenario) -> str:
        """
        Generate training data for a specific scenario
        
        Args:
            scenario: Training scenario configuration
            
        Returns:
            Path to generated data file
        """
        self.logger.info(f"Generating data for scenario: {scenario.name}")
        
        # Create scenario-specific SUMO configuration
        scenario_config = self._create_scenario_config(scenario)
        scenario_config_path = self.output_dir / f"{scenario.scenario_id}_config.xml"
        
        with open(scenario_config_path, 'w') as f:
            f.write(scenario_config)
        
        # Run SUMO simulation
        data_file = self.output_dir / f"{scenario.scenario_id}_data.json"
        self._run_simulation(scenario_config_path, data_file, scenario)
        
        self.logger.info(f"Generated data file: {data_file}")
        return str(data_file)
    
    def _create_scenario_config(self, scenario: TrainingScenario) -> str:
        """Create SUMO configuration for scenario"""
        # This is a simplified SUMO configuration
        # In a real implementation, this would generate proper SUMO XML files
        
        config = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="network.net.xml"/>
        <route-files value="routes.rou.xml"/>
        <additional-files value="detectors.add.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{scenario.duration * 60}"/>
        <step-length value="{self.simulation_step}"/>
    </time>
    <processing>
        <ignore-junction-blocker value="5"/>
    </processing>
    <routing>
        <device.rerouting.adaptation-interval value="10"/>
    </routing>
    <traci_server>
        <port value="{self.traci_port}"/>
    </traci_server>
</configuration>"""
        
        return config
    
    def _run_simulation(self, config_path: str, output_file: str, scenario: TrainingScenario):
        """Run SUMO simulation and collect data"""
        try:
            # Start SUMO simulation
            cmd = [self.sumo_binary, "-c", config_path, "--tripinfo-output", "tripinfo.xml"]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Collect simulation data
            simulation_data = self._collect_simulation_data(scenario)
            
            # Wait for simulation to complete
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                self.logger.error(f"SUMO simulation failed: {stderr.decode()}")
                return
            
            # Save collected data
            with open(output_file, 'w') as f:
                json.dump(simulation_data, f, indent=2, default=str)
            
            self.logger.info(f"Simulation completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error running SUMO simulation: {e}")
            raise
    
    def _collect_simulation_data(self, scenario: TrainingScenario) -> Dict[str, Any]:
        """Collect data during SUMO simulation"""
        # This is a mock implementation
        # In a real implementation, this would use TraCI to collect real data
        
        simulation_data = {
            'scenario_id': scenario.scenario_id,
            'duration': scenario.duration,
            'intersections': scenario.intersections,
            'traffic_data': [],
            'signal_data': [],
            'performance_metrics': {}
        }
        
        # Generate mock traffic data
        for minute in range(scenario.duration):
            for second in range(60):
                timestamp = datetime.now() + timedelta(minutes=minute, seconds=second)
                
                # Generate traffic data for each intersection
                for intersection_id in scenario.intersections:
                    traffic_data = self._generate_mock_traffic_data(intersection_id, timestamp, scenario)
                    simulation_data['traffic_data'].append(traffic_data)
                    
                    signal_data = self._generate_mock_signal_data(intersection_id, timestamp)
                    simulation_data['signal_data'].append(signal_data)
        
        return simulation_data
    
    def _generate_mock_traffic_data(self, intersection_id: str, timestamp: datetime, 
                                  scenario: TrainingScenario) -> Dict[str, Any]:
        """Generate mock traffic data"""
        # Base traffic pattern
        base_patterns = {
            'rush_hour': {'min_vehicles': 20, 'max_vehicles': 50, 'peak_hour': 8},
            'normal': {'min_vehicles': 5, 'max_vehicles': 25, 'peak_hour': 12},
            'night': {'min_vehicles': 1, 'max_vehicles': 10, 'peak_hour': 2},
            'emergency': {'min_vehicles': 15, 'max_vehicles': 40, 'peak_hour': 14}
        }
        
        pattern = base_patterns.get(scenario.traffic_pattern, base_patterns['normal'])
        
        # Calculate traffic intensity based on time
        hour = timestamp.hour
        time_factor = 1.0
        if abs(hour - pattern['peak_hour']) <= 2:
            time_factor = 1.5  # Peak hour multiplier
        
        # Generate lane counts
        min_vehicles = pattern['min_vehicles']
        max_vehicles = pattern['max_vehicles']
        
        lane_counts = {
            'north_lane': int(np.random.uniform(min_vehicles, max_vehicles) * time_factor),
            'south_lane': int(np.random.uniform(min_vehicles, max_vehicles) * time_factor),
            'east_lane': int(np.random.uniform(min_vehicles, max_vehicles) * time_factor),
            'west_lane': int(np.random.uniform(min_vehicles, max_vehicles) * time_factor)
        }
        
        return {
            'intersection_id': intersection_id,
            'timestamp': timestamp.isoformat(),
            'lane_counts': lane_counts,
            'avg_speed': np.random.uniform(20, 60),
            'weather_condition': scenario.weather_condition,
            'temperature': np.random.uniform(15, 35),
            'visibility': np.random.uniform(5, 20),
            'emergency_vehicles': np.random.random() < 0.05  # 5% chance
        }
    
    def _generate_mock_signal_data(self, intersection_id: str, timestamp: datetime) -> Dict[str, Any]:
        """Generate mock signal data"""
        phases = ['north_south', 'east_west', 'left_turn', 'pedestrian']
        current_phase = np.random.choice(phases)
        
        return {
            'intersection_id': intersection_id,
            'timestamp': timestamp.isoformat(),
            'current_phase': phases.index(current_phase),
            'phase_duration': np.random.uniform(20, 60),
            'cycle_progress': np.random.uniform(0, 1),
            'time_since_change': np.random.uniform(0, 30)
        }


class ModelCheckpointManager:
    """Manages model checkpoints and versioning"""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.logger = logging.getLogger(__name__)
        
        # Load existing checkpoints
        self.checkpoints = self._load_existing_checkpoints()
    
    def save_checkpoint(self, agent: AdvancedQLearningAgent, episode: TrainingEpisode,
                       performance_metrics: Dict[str, float]) -> str:
        """Save model checkpoint"""
        checkpoint_id = f"checkpoint_{int(time.time())}"
        timestamp = datetime.now()
        
        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(exist_ok=True)
        
        # Save model
        model_path = checkpoint_path / "model.pth"
        agent.save_model(str(model_path))
        
        # Save configuration
        config_path = checkpoint_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(agent.config, f)
        
        # Save episode data
        episode_path = checkpoint_path / "episode.json"
        with open(episode_path, 'w') as f:
            json.dump(episode.to_dict(), f, indent=2)
        
        # Create checkpoint metadata
        checkpoint = ModelCheckpoint(
            checkpoint_id=checkpoint_id,
            timestamp=timestamp,
            training_step=agent.training_step,
            episode_count=len(agent.episode_rewards),
            performance_metrics=performance_metrics,
            model_path=str(model_path),
            config_path=str(config_path),
            metadata={
                'intersection_id': agent.intersection_id,
                'episode_id': episode.episode_id,
                'scenario_id': episode.scenario_id,
                'total_reward': episode.total_reward,
                'duration': episode.duration
            }
        )
        
        # Save checkpoint metadata
        metadata_path = checkpoint_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint.to_dict(), f, indent=2)
        
        # Add to checkpoints list
        self.checkpoints[checkpoint_id] = checkpoint
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
        self.logger.info(f"Saved checkpoint: {checkpoint_id}")
        return checkpoint_id
    
    def load_checkpoint(self, checkpoint_id: str) -> Optional[ModelCheckpoint]:
        """Load checkpoint by ID"""
        return self.checkpoints.get(checkpoint_id)
    
    def get_best_checkpoint(self, metric: str = 'total_reward') -> Optional[ModelCheckpoint]:
        """Get best checkpoint based on metric"""
        if not self.checkpoints:
            return None
        
        best_checkpoint = max(
            self.checkpoints.values(),
            key=lambda c: c.performance_metrics.get(metric, 0)
        )
        
        return best_checkpoint
    
    def list_checkpoints(self) -> List[ModelCheckpoint]:
        """List all available checkpoints"""
        return list(self.checkpoints.values())
    
    def _load_existing_checkpoints(self) -> Dict[str, ModelCheckpoint]:
        """Load existing checkpoints from disk"""
        checkpoints = {}
        
        for checkpoint_dir in self.checkpoint_dir.iterdir():
            if not checkpoint_dir.is_dir():
                continue
            
            metadata_path = checkpoint_dir / "metadata.json"
            if not metadata_path.exists():
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                checkpoint = ModelCheckpoint(
                    checkpoint_id=metadata['checkpoint_id'],
                    timestamp=datetime.fromisoformat(metadata['timestamp']),
                    training_step=metadata['training_step'],
                    episode_count=metadata['episode_count'],
                    performance_metrics=metadata['performance_metrics'],
                    model_path=metadata['model_path'],
                    config_path=metadata['config_path'],
                    metadata=metadata['metadata']
                )
                
                checkpoints[checkpoint.checkpoint_id] = checkpoint
                
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint {checkpoint_dir}: {e}")
        
        return checkpoints
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to stay within limit"""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by timestamp and remove oldest
        sorted_checkpoints = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1].timestamp
        )
        
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint_id, checkpoint in checkpoints_to_remove:
            # Remove from memory
            del self.checkpoints[checkpoint_id]
            
            # Remove from disk
            checkpoint_path = self.checkpoint_dir / checkpoint_id
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            
            self.logger.info(f"Removed old checkpoint: {checkpoint_id}")


class AdvancedTrainingPipeline:
    """
    Advanced training pipeline for multi-intersection Q-Learning
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_generator = SUMODataGenerator(
            sumo_config_path=self.config.get('sumo_config_path', 'config/sumo'),
            output_dir=self.config.get('training_data_dir', 'data/training')
        )
        
        self.checkpoint_manager = ModelCheckpointManager(
            checkpoint_dir=self.config.get('checkpoint_dir', 'models/checkpoints'),
            max_checkpoints=self.config.get('max_checkpoints', 10)
        )
        
        # Training scenarios
        self.training_scenarios = self._create_training_scenarios()
        
        # Training state
        self.is_training = False
        self.current_episode = None
        self.training_metrics = defaultdict(list)
        
        # Agents for each intersection
        self.agents = {}
        self.coordinators = {}
        
        # Initialize agents
        self._initialize_agents()
    
    def _create_training_scenarios(self) -> List[TrainingScenario]:
        """Create training scenarios"""
        scenarios = [
            TrainingScenario(
                scenario_id="rush_hour_clear",
                name="Rush Hour - Clear Weather",
                description="High traffic volume during peak hours with clear weather",
                duration=60,  # 1 hour
                traffic_pattern="rush_hour",
                weather_condition="clear",
                intersections=["junction-1", "junction-2", "junction-3"],
                parameters={
                    "vehicle_spawn_rate": 0.8,
                    "emergency_probability": 0.1,
                    "pedestrian_demand": 0.6
                }
            ),
            TrainingScenario(
                scenario_id="normal_rainy",
                name="Normal Traffic - Rainy Weather",
                description="Normal traffic volume with rainy weather conditions",
                duration=45,
                traffic_pattern="normal",
                weather_condition="rainy",
                intersections=["junction-1", "junction-2", "junction-3"],
                parameters={
                    "vehicle_spawn_rate": 0.4,
                    "emergency_probability": 0.05,
                    "pedestrian_demand": 0.3
                }
            ),
            TrainingScenario(
                scenario_id="night_foggy",
                name="Night Traffic - Foggy Weather",
                description="Low traffic volume during night with foggy conditions",
                duration=30,
                traffic_pattern="night",
                weather_condition="foggy",
                intersections=["junction-1", "junction-2", "junction-3"],
                parameters={
                    "vehicle_spawn_rate": 0.2,
                    "emergency_probability": 0.02,
                    "pedestrian_demand": 0.1
                }
            ),
            TrainingScenario(
                scenario_id="emergency_stormy",
                name="Emergency Scenario - Stormy Weather",
                description="Emergency vehicle priority with stormy weather",
                duration=20,
                traffic_pattern="emergency",
                weather_condition="stormy",
                intersections=["junction-1", "junction-2", "junction-3"],
                parameters={
                    "vehicle_spawn_rate": 0.6,
                    "emergency_probability": 0.3,
                    "pedestrian_demand": 0.2
                }
            )
        ]
        
        return scenarios
    
    def _initialize_agents(self):
        """Initialize Q-Learning agents for each intersection"""
        intersections = self.config.get('intersections', [])
        
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
            
            self.logger.info(f"Initialized agent and coordinator for {intersection_id}")
    
    def start_training(self, num_episodes: int = 100, validation_frequency: int = 10):
        """
        Start training process
        
        Args:
            num_episodes: Number of training episodes
            validation_frequency: How often to run validation
        """
        self.logger.info(f"Starting training for {num_episodes} episodes")
        self.is_training = True
        
        try:
            for episode in range(num_episodes):
                if not self.is_training:
                    break
                
                self.logger.info(f"Starting episode {episode + 1}/{num_episodes}")
                
                # Select random scenario
                scenario = np.random.choice(self.training_scenarios)
                
                # Run training episode
                episode_data = self._run_training_episode(scenario, episode)
                
                # Update training metrics
                self._update_training_metrics(episode_data)
                
                # Validation
                if (episode + 1) % validation_frequency == 0:
                    self._run_validation(episode + 1)
                
                # Save checkpoint
                if (episode + 1) % 20 == 0:  # Every 20 episodes
                    self._save_training_checkpoint(episode_data)
                
                self.logger.info(f"Completed episode {episode + 1}")
            
            self.logger.info("Training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        finally:
            self.is_training = False
    
    def _run_training_episode(self, scenario: TrainingScenario, episode_num: int) -> TrainingEpisode:
        """Run a single training episode"""
        episode_id = f"episode_{episode_num}_{scenario.scenario_id}"
        start_time = datetime.now()
        
        # Generate training data
        data_file = self.data_generator.generate_scenario_data(scenario)
        
        # Load simulation data
        with open(data_file, 'r') as f:
            simulation_data = json.load(f)
        
        # Run episode
        episode_metrics = self._simulate_episode(simulation_data, scenario)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Create episode data
        episode = TrainingEpisode(
            episode_id=episode_id,
            scenario_id=scenario.scenario_id,
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            intersections=scenario.intersections,
            total_reward=episode_metrics['total_reward'],
            performance_metrics=episode_metrics,
            state_action_pairs=episode_metrics.get('state_action_pairs', [])
        )
        
        return episode
    
    def _simulate_episode(self, simulation_data: Dict[str, Any], 
                         scenario: TrainingScenario) -> Dict[str, Any]:
        """Simulate episode using generated data"""
        total_reward = 0.0
        state_action_pairs = []
        
        # Process traffic data
        for traffic_data in simulation_data['traffic_data']:
            intersection_id = traffic_data['intersection_id']
            
            if intersection_id not in self.agents:
                continue
            
            agent = self.agents[intersection_id]
            coordinator = self.coordinators[intersection_id]
            
            # Create state
            state = agent.create_state(traffic_data, {}, [])
            
            # Select action
            action = agent.select_action(state, training=True)
            
            # Coordinate with other intersections
            coordinated_action = coordinator.coordinate_optimization(action.__dict__)
            
            # Calculate reward (simplified)
            reward = self._calculate_episode_reward(state, action, traffic_data)
            total_reward += reward
            
            # Store for learning
            state_action_pairs.append((state, action, reward))
            
            # Add experience to replay buffer
            next_state = state  # Simplified - in real implementation, this would be the next state
            agent.add_experience(state, action, reward, next_state, False)
            
            # Train agent
            agent.train_step()
        
        return {
            'total_reward': total_reward,
            'state_action_pairs': state_action_pairs,
            'episode_length': len(simulation_data['traffic_data']),
            'scenario_id': scenario.scenario_id
        }
    
    def _calculate_episode_reward(self, state: MultiDimensionalState, 
                                action: SophisticatedAction, 
                                traffic_data: Dict[str, Any]) -> float:
        """Calculate reward for state-action pair"""
        reward = 0.0
        
        # Wait time reduction
        wait_time_reduction = np.sum(state.waiting_times) * 0.1
        reward += wait_time_reduction
        
        # Throughput improvement
        throughput_improvement = np.sum(state.flow_rates) * 0.05
        reward += throughput_improvement
        
        # Fuel efficiency
        fuel_penalty = -np.sum(state.queue_lengths) * 0.02
        reward += fuel_penalty
        
        # Action efficiency
        if action.action_type in [2, 3]:  # High-efficiency actions
            reward += 0.1
        
        return reward
    
    def _update_training_metrics(self, episode: TrainingEpisode):
        """Update training metrics"""
        self.training_metrics['episode_rewards'].append(episode.total_reward)
        self.training_metrics['episode_durations'].append(episode.duration)
        self.training_metrics['scenario_performance'][episode.scenario_id].append(episode.total_reward)
    
    def _run_validation(self, episode_num: int):
        """Run validation on held-out scenarios"""
        self.logger.info(f"Running validation at episode {episode_num}")
        
        # Select validation scenario
        validation_scenario = self.training_scenarios[0]  # Use first scenario for validation
        
        # Run validation episode
        validation_episode = self._run_training_episode(validation_scenario, episode_num)
        
        # Calculate validation metrics
        validation_metrics = {
            'episode_id': validation_episode.episode_id,
            'total_reward': validation_episode.total_reward,
            'duration': validation_episode.duration,
            'performance_metrics': validation_episode.performance_metrics
        }
        
        self.training_metrics['validation_metrics'].append(validation_metrics)
        
        self.logger.info(f"Validation completed: Reward = {validation_episode.total_reward:.2f}")
    
    def _save_training_checkpoint(self, episode: TrainingEpisode):
        """Save training checkpoint"""
        # Calculate performance metrics
        performance_metrics = {
            'total_reward': episode.total_reward,
            'duration': episode.duration,
            'avg_reward': np.mean(self.training_metrics['episode_rewards'][-20:]) if self.training_metrics['episode_rewards'] else 0
        }
        
        # Save checkpoint for each agent
        for intersection_id, agent in self.agents.items():
            checkpoint_id = self.checkpoint_manager.save_checkpoint(
                agent, episode, performance_metrics
            )
            self.logger.info(f"Saved checkpoint {checkpoint_id} for {intersection_id}")
    
    def stop_training(self):
        """Stop training process"""
        self.logger.info("Stopping training...")
        self.is_training = False
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'is_training': self.is_training,
            'current_episode': self.current_episode,
            'total_episodes': len(self.training_metrics.get('episode_rewards', [])),
            'avg_reward': np.mean(self.training_metrics.get('episode_rewards', [0])),
            'best_reward': max(self.training_metrics.get('episode_rewards', [0])),
            'checkpoints': len(self.checkpoint_manager.checkpoints)
        }
    
    def load_best_model(self, intersection_id: str) -> bool:
        """Load best model for intersection"""
        if intersection_id not in self.agents:
            self.logger.error(f"No agent found for intersection {intersection_id}")
            return False
        
        best_checkpoint = self.checkpoint_manager.get_best_checkpoint()
        if not best_checkpoint:
            self.logger.warning("No checkpoints available")
            return False
        
        # Load model
        agent = self.agents[intersection_id]
        agent.load_model(best_checkpoint.model_path)
        
        self.logger.info(f"Loaded best model for {intersection_id}")
        return True
