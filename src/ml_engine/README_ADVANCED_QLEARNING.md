# Advanced Multi-Intersection Q-Learning System

## Overview

This is a production-ready, multi-intersection Q-Learning optimization engine for real-time traffic signal control. The system implements sophisticated state spaces, action spaces, reward functions, and coordination mechanisms to optimize traffic flow across multiple intersections.

## 🚀 Key Features

### 1. Enhanced Q-Learning Architecture
- **Multi-dimensional state space**: 45 features including traffic data, temporal features, environmental factors, and adjacent signal states
- **Sophisticated action space**: 8 different action types with dynamic phase durations and safety constraints
- **Advanced neural networks**: Dueling DQN with attention mechanisms and residual connections
- **Adaptive learning**: Dynamic learning rates and epsilon decay schedules

### 2. Multi-Intersection Coordination
- **Communication protocol**: Real-time message passing between adjacent intersections
- **Conflict resolution**: Automatic resolution of competing optimization goals
- **Network-level optimization**: Traffic flow propagation and green wave coordination
- **Performance balancing**: Load distribution across the network

### 3. Advanced Reward Function
- **Multi-objective optimization**: Wait time reduction, throughput increase, fuel efficiency, pedestrian safety
- **Environmental adaptation**: Weather and visibility-based reward adjustments
- **Coordination bonuses**: Rewards for synchronized actions with adjacent intersections
- **Stability penalties**: Prevents frequent signal changes

### 4. Adaptive Experience Replay
- **Prioritized sampling**: Focus on high-value experiences
- **Curriculum learning**: Start with easy experiences, gradually increase difficulty
- **Multiple sampling strategies**: Prioritized, uniform, curriculum, and balanced sampling
- **Dynamic parameter adjustment**: Adaptive alpha and beta values

### 5. Automated Training Pipeline
- **SUMO integration**: Automated data generation using traffic simulation
- **Model checkpointing**: Version control and performance tracking
- **Validation framework**: Held-out scenario testing
- **Performance monitoring**: Real-time metrics and optimization

## 📁 Project Structure

```
src/ml_engine/
├── algorithms/
│   ├── advanced_q_learning_agent.py      # Main Q-Learning agent
│   ├── multi_intersection_coordinator.py # Coordination system
│   ├── advanced_reward_function.py       # Multi-objective reward function
│   └── adaptive_experience_replay.py     # Advanced experience replay
├── training/
│   └── advanced_training_pipeline.py     # Training pipeline
├── config/
│   └── ml_config.py                      # Configuration management
├── production_q_learning_system.py       # Main production system
└── README_ADVANCED_QLEARNING.md          # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.9+
- NumPy 1.21+
- NetworkX 2.6+
- SUMO (for training data generation)

### Install Dependencies
```bash
pip install torch numpy networkx pyyaml asyncio
```

### Install SUMO (for training)
```bash
# Ubuntu/Debian
sudo apt-get install sumo sumo-tools

# macOS
brew install sumo

# Windows
# Download from https://sumo.dlr.de/docs/Downloads.php
```

## ⚙️ Configuration

### Basic Configuration
```yaml
# config/ml_config.yaml
q_learning:
  learning_rate: 0.001
  epsilon: 0.1
  epsilon_decay: 0.995
  epsilon_min: 0.01
  discount_factor: 0.95
  replay_buffer_size: 50000
  batch_size: 64
  target_update_frequency: 200
  hidden_layers: [256, 128, 64]

reward_function:
  wait_time_weight: 0.4
  throughput_weight: 0.2
  fuel_efficiency_weight: 0.15
  pedestrian_safety_weight: 0.1
  coordination_weight: 0.05
  stability_weight: 0.05
  emergency_weight: 0.03
  environmental_weight: 0.02

experience_replay:
  capacity: 50000
  alpha: 0.6
  beta: 0.4
  beta_increment: 0.001

intersections:
  - id: "junction-1"
    name: "Main Street & First Avenue"
    adjacent_intersections: ["junction-2", "junction-3"]
    priority: 1
  - id: "junction-2"
    name: "Second Street & Oak Avenue"
    adjacent_intersections: ["junction-1", "junction-3"]
    priority: 2
  - id: "junction-3"
    name: "Third Street & Pine Avenue"
    adjacent_intersections: ["junction-1", "junction-2"]
    priority: 3
```

## 🚀 Quick Start

### 1. Basic Usage
```python
from production_q_learning_system import ProductionQLearningSystem
import asyncio

async def main():
    # Initialize system
    system = ProductionQLearningSystem()
    
    # Start system
    await system.start_system()
    
    # Run for 5 minutes
    await asyncio.sleep(300)
    
    # Get status
    status = system.get_system_status()
    print(f"System running: {status['is_running']}")
    
    # Stop system
    await system.stop_system()

# Run system
asyncio.run(main())
```

### 2. Training Mode
```python
# Start training
system = ProductionQLearningSystem()
await system.start_system()

# Train for 100 episodes
system.start_training(num_episodes=100)

# System continues running with trained models
```

### 3. Custom Configuration
```python
# Load custom configuration
system = ProductionQLearningSystem(config_path="custom_config.yaml")
```

## 📊 State Space (45 dimensions)

### Traffic Data (20 dimensions)
- Lane counts: 4 lanes × normalized counts
- Average speed: Normalized speed (0-1)
- Queue lengths: 4 lanes × normalized lengths
- Waiting times: 4 lanes × normalized times
- Flow rates: 4 lanes × normalized rates

### Temporal Features (5 dimensions)
- Time of day: Normalized hour (0-1)
- Day of week: Normalized day (0-1)
- Weekend flag: Boolean
- Holiday flag: Boolean
- Season: Normalized season (0-1)

### Environmental Features (9 dimensions)
- Weather condition: One-hot encoded (6 conditions)
- Temperature: Normalized temperature
- Visibility: Normalized visibility
- Precipitation intensity: 0-1

### Signal State (4 dimensions)
- Current phase: Normalized phase (0-1)
- Phase duration: Normalized duration (0-1)
- Cycle progress: 0-1
- Time since change: Normalized time

### Adjacent Signals (12 dimensions)
- Upstream/downstream states: 4 intersections × 3 features each

### Historical Context (4 dimensions)
- Recent performance metrics
- Congestion trend
- Emergency vehicles flag
- Flow rate history

## 🎯 Action Space (8 actions)

1. **Maintain** (0): Keep current timing
2. **Reduce Congestion** (1): Reduce congestion in high-traffic lanes
3. **Increase Throughput** (2): Increase overall throughput
4. **Coordinate Green Wave** (3): Coordinate with adjacent intersections
5. **Emergency Priority** (4): Give priority to emergency vehicles
6. **Pedestrian Friendly** (5): Optimize for pedestrian safety
7. **Fuel Efficient** (6): Minimize fuel consumption
8. **Adaptive Cycle** (7): Adapt cycle time to traffic demand

## 🏆 Reward Function Components

### Primary Objectives
- **Wait Time Reduction** (40%): Primary optimization goal
- **Throughput Increase** (20%): Vehicle flow optimization
- **Fuel Efficiency** (15%): Reduce idling and fuel consumption

### Secondary Objectives
- **Pedestrian Safety** (10%): Safety-focused actions
- **Coordination** (5%): Synchronized actions with adjacent intersections
- **Stability** (5%): Prevent frequent signal changes
- **Emergency Priority** (3%): Emergency vehicle handling
- **Environmental Adaptation** (2%): Weather and visibility adaptation

## 🔧 Advanced Features

### 1. Multi-Intersection Coordination
```python
# Send coordination message
coordinator.send_message(
    receiver_id="junction-2",
    message_type=CoordinationMessageType.GREEN_WAVE_REQUEST,
    data={"timing": 30, "phase": 1}
)

# Coordinate optimization
coordinated_action = coordinator.coordinate_optimization(proposed_action)
```

### 2. Adaptive Experience Replay
```python
# Add experience with priority
replay_buffer.add_experience(
    state, action, reward, next_state, done, td_error
)

# Sample with adaptive strategy
experiences, weights, indices = replay_buffer.sample(batch_size=64)
```

### 3. Advanced Reward Function
```python
# Calculate multi-objective reward
reward_components = reward_function.calculate_reward(
    state, action, next_state, performance_metrics, coordination_data
)

# Get reward statistics
stats = reward_function.get_reward_statistics()
```

## 📈 Performance Monitoring

### Real-time Metrics
- Optimization cycle count and success rate
- Average cycle time and processing time
- Total and average rewards
- Agent training statistics
- Coordination performance

### Historical Data
- Optimization history with detailed results
- Reward component trends
- Training progress and convergence
- Error rates and recovery

### Example Monitoring
```python
# Get system status
status = system.get_system_status()
print(f"Success rate: {status['performance_metrics']['successful_cycles'] / status['performance_metrics']['total_cycles']:.2%}")

# Get performance metrics
metrics = system.get_performance_metrics()
print(f"Average reward: {metrics['avg_reward']:.3f}")

# Get optimization history
history = system.get_optimization_history(limit=10)
for cycle in history:
    print(f"Cycle {cycle['cycle']}: {cycle['successful_optimizations']} successful")
```

## 🧪 Training and Validation

### Training Scenarios
1. **Rush Hour - Clear Weather**: High traffic with optimal conditions
2. **Normal Traffic - Rainy Weather**: Moderate traffic with weather challenges
3. **Night Traffic - Foggy Weather**: Low traffic with visibility issues
4. **Emergency Scenario - Stormy Weather**: Emergency vehicles with severe weather

### Validation Process
- Held-out scenario testing
- Performance comparison with baseline algorithms
- A/B testing with different configurations
- Real-world deployment validation

### Training Commands
```bash
# Start training
python production_q_learning_system.py --mode train --episodes 100

# Validate models
python production_q_learning_system.py --mode validate

# Run production system
python production_q_learning_system.py --mode production
```

## 🔍 Troubleshooting

### Common Issues

1. **Empty Replay Buffer**
   - Check data collection and experience generation
   - Verify reward function is producing valid rewards
   - Increase exploration rate (epsilon)

2. **Poor Convergence**
   - Adjust learning rate and discount factor
   - Check reward function weights
   - Increase training episodes

3. **Coordination Failures**
   - Verify network connectivity
   - Check message queue sizes
   - Review conflict resolution logic

4. **Performance Degradation**
   - Monitor system resources
   - Check for memory leaks
   - Restart components if needed

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with debug information
system = ProductionQLearningSystem(debug=True)
```

## 📚 API Reference

### ProductionQLearningSystem
- `start_system()`: Start the production system
- `stop_system()`: Stop the production system
- `start_training(num_episodes)`: Start training process
- `get_system_status()`: Get current system status
- `get_performance_metrics()`: Get detailed performance metrics
- `get_optimization_history(limit)`: Get recent optimization history

### AdvancedQLearningAgent
- `create_state(traffic_data, current_timings, historical_data)`: Create state from data
- `select_action(state, training)`: Select action using epsilon-greedy policy
- `optimize_signal_timing(traffic_data, current_timings)`: Optimize signal timing
- `add_experience(state, action, reward, next_state, done)`: Add experience for learning
- `train_step()`: Perform one training step
- `save_model(filepath)`: Save trained model
- `load_model(filepath)`: Load trained model

### MultiIntersectionCoordinator
- `send_message(receiver_id, message_type, data)`: Send coordination message
- `receive_message(message)`: Receive coordination message
- `coordinate_optimization(proposed_action)`: Coordinate with other intersections
- `get_coordination_metrics()`: Get coordination performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- SUMO traffic simulation framework
- PyTorch deep learning library
- NetworkX graph analysis library
- Open source traffic management research

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation wiki

---

**Note**: This is a production-ready system designed for real-world deployment. Ensure proper testing and validation before deploying in live traffic environments.
