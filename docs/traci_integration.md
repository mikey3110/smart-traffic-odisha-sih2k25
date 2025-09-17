# TraCI Integration Documentation

## Overview

This document provides comprehensive documentation for the robust TraCI controller system that manages real-time traffic signal control through SUMO. The system provides fault-tolerant multi-intersection control with error handling, fallback mechanisms, and real-time monitoring.

## Architecture

### Core Components

1. **RobustTraCIController**: Main controller class managing all traffic signal operations
2. **WebsterFormula**: Traffic signal timing optimization using Webster's formula
3. **IntersectionConfig**: Configuration management for individual intersections
4. **TrafficData**: Real-time traffic data collection and storage
5. **ControlCommand**: Command queuing system for traffic signal control

### Key Features

- **Multi-intersection Control**: Synchronized control of multiple intersections
- **Fault Tolerance**: Automatic reconnection and error recovery
- **Real-time Monitoring**: Continuous traffic data collection
- **Fallback Mechanisms**: Webster's formula and emergency overrides
- **Command Queuing**: Priority-based command processing
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## Installation and Setup

### Prerequisites

```bash
# Install required packages
pip install traci
pip install numpy
pip install matplotlib  # For visualization (optional)
```

### Configuration

Create a configuration file `config/traci_config.json`:

```json
{
    "max_retries": 5,
    "retry_delay": 5.0,
    "port": 8813,
    "host": "localhost",
    "monitoring_interval": 1.0,
    "control_interval": 0.1,
    "log_level": "INFO",
    "intersections": [
        {
            "id": "center",
            "phases": ["GGrrGGrr", "yyrryyrr", "rrrrrrrr", "rrGGrrGG", "rryyrryy"],
            "min_green_time": 10.0,
            "max_green_time": 60.0,
            "yellow_time": 3.0,
            "all_red_time": 2.0,
            "approach_edges": ["north_approach", "south_approach", "east_approach", "west_approach"],
            "exit_edges": ["north_exit", "south_exit", "east_exit", "west_exit"],
            "detectors": []
        }
    ]
}
```

### Configuration Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `max_retries` | Maximum reconnection attempts | 5 | 1-10 |
| `retry_delay` | Delay between reconnection attempts (seconds) | 5.0 | 1.0-30.0 |
| `port` | TraCI server port | 8813 | 1024-65535 |
| `host` | TraCI server host | localhost | Any valid hostname |
| `monitoring_interval` | Traffic data collection interval (seconds) | 1.0 | 0.1-10.0 |
| `control_interval` | Control loop interval (seconds) | 0.1 | 0.01-1.0 |
| `log_level` | Logging level | INFO | DEBUG, INFO, WARNING, ERROR |

## Usage

### Basic Usage

```python
from src.simulation.traci_controller import RobustTraCIController

# Create controller
controller = RobustTraCIController('config/traci_config.json')

# Start controller
controller.start()

# Control traffic signals
controller.set_phase('center', 'GGrrGGrr')
controller.extend_phase('center', 10.0)

# Get traffic data
traffic_data = controller.get_traffic_data('center')
print(f"Vehicle count: {traffic_data.vehicle_counts}")

# Stop controller
controller.stop()
```

### Advanced Usage

```python
# Emergency override
controller.emergency_override('center')

# Webster optimization
optimization = controller.optimize_with_webster('center')
print(f"Optimal cycle time: {optimization['cycle_time']}s")

# Get statistics
stats = controller.get_statistics()
print(f"Commands sent: {stats['commands_sent']}")

# Health check
if controller.health_check():
    print("Controller is healthy")
else:
    print("Controller needs attention")
```

## API Reference

### RobustTraCIController

#### Constructor
```python
RobustTraCIController(config_file: str = "config/traci_config.json")
```

#### Methods

##### Connection Management
- `connect() -> bool`: Connect to SUMO via TraCI
- `disconnect()`: Disconnect from SUMO
- `reconnect() -> bool`: Reconnect with retry logic

##### Control Operations
- `start()`: Start the controller
- `stop()`: Stop the controller
- `set_phase(intersection_id: str, phase: str, priority: int = 0)`: Set signal phase
- `extend_phase(intersection_id: str, duration: float, priority: int = 0)`: Extend current phase
- `emergency_override(intersection_id: str)`: Activate emergency override
- `clear_emergency_override(intersection_id: str)`: Clear emergency override

##### Data Collection
- `get_traffic_data(intersection_id: str) -> Optional[TrafficData]`: Get traffic data
- `get_all_traffic_data() -> Dict[str, TrafficData]`: Get all traffic data

##### Optimization
- `optimize_with_webster(intersection_id: str) -> Dict[str, Any]`: Webster optimization

##### Monitoring
- `get_statistics() -> Dict[str, Any]`: Get controller statistics
- `health_check() -> bool`: Perform health check

### TrafficData

```python
@dataclass
class TrafficData:
    intersection_id: str
    timestamp: float
    vehicle_counts: Dict[str, int]
    lane_occupancy: Dict[str, float]
    queue_lengths: Dict[str, int]
    waiting_times: Dict[str, float]
    current_phase: str
    phase_remaining_time: float
```

### ControlCommand

```python
@dataclass
class ControlCommand:
    intersection_id: str
    action: str  # 'set_phase', 'extend_phase', 'emergency_override'
    phase: Optional[str] = None
    duration: Optional[float] = None
    priority: int = 0
```

## Error Handling

### Connection Errors

The controller automatically handles connection errors with retry logic:

```python
# Automatic reconnection on connection loss
controller.start()  # Will retry connection if it fails

# Manual reconnection
if not controller.connect():
    controller.reconnect()
```

### Command Errors

Commands are queued and processed asynchronously:

```python
# Commands are queued and processed in priority order
controller.set_phase('center', 'GGrrGGrr', priority=1)
controller.emergency_override('center')  # Higher priority (1000)
```

### Error Monitoring

```python
# Check for errors
stats = controller.get_statistics()
if stats['errors'] > 100:
    print("High error rate detected")

# Health check
if not controller.health_check():
    print("Controller is unhealthy")
```

## Fallback Mechanisms

### 1. Webster's Formula Fallback

When ML optimization fails, the controller falls back to Webster's formula:

```python
# Automatic Webster optimization
optimization = controller.optimize_with_webster('center')
cycle_time = optimization['cycle_time']
green_times = optimization['green_times']
```

### 2. Emergency Override

For emergency vehicles or critical situations:

```python
# Activate emergency override
controller.emergency_override('center')

# Clear emergency override
controller.clear_emergency_override('center')
```

### 3. Default Phase Timing

If all else fails, the controller uses configured default timing:

```python
# Default phase timing from configuration
min_green_time = 10.0
max_green_time = 60.0
yellow_time = 3.0
all_red_time = 2.0
```

## Real-time Monitoring

### Traffic Data Collection

The controller continuously collects traffic data:

```python
# Get real-time traffic data
traffic_data = controller.get_traffic_data('center')

print(f"Vehicle counts: {traffic_data.vehicle_counts}")
print(f"Lane occupancy: {traffic_data.lane_occupancy}")
print(f"Queue lengths: {traffic_data.queue_lengths}")
print(f"Waiting times: {traffic_data.waiting_times}")
print(f"Current phase: {traffic_data.current_phase}")
print(f"Phase remaining time: {traffic_data.phase_remaining_time}")
```

### Performance Metrics

```python
# Get controller statistics
stats = controller.get_statistics()

print(f"Uptime: {stats['uptime']:.1f} seconds")
print(f"Commands sent: {stats['commands_sent']}")
print(f"Commands failed: {stats['commands_failed']}")
print(f"Reconnections: {stats['reconnections']}")
print(f"Errors: {stats['errors']}")
print(f"Queue size: {stats['queue_size']}")
```

## Logging

### Log Configuration

Logs are written to `logs/traci_controller.log` and console:

```python
import logging

# Configure logging level
logging.getLogger('traci_controller').setLevel(logging.DEBUG)
```

### Log Messages

- **INFO**: Normal operations, phase changes, connections
- **WARNING**: Non-critical errors, retries, fallbacks
- **ERROR**: Critical errors, connection failures, command failures
- **DEBUG**: Detailed debugging information

### Log Analysis

```bash
# View recent logs
tail -f logs/traci_controller.log

# Filter for errors
grep "ERROR" logs/traci_controller.log

# Filter for specific intersection
grep "center" logs/traci_controller.log
```

## Testing

### Unit Tests

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/simulation/test_traci_controller.py -v

# Run specific test class
python -m pytest tests/simulation/test_traci_controller.py::TestTraCIController -v

# Run with coverage
python -m pytest tests/simulation/test_traci_controller.py --cov=src.simulation.traci_controller
```

### Test Categories

1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Full system testing with mocked SUMO
3. **Error Scenario Tests**: Error handling and edge cases
4. **Performance Tests**: Load testing and stress testing

### Manual Testing

```python
# Test connection
controller = RobustTraCIController()
assert controller.connect()
assert controller.health_check()

# Test phase control
controller.set_phase('center', 'GGrrGGrr')
time.sleep(1)
controller.set_phase('center', 'rrGGrrGG')

# Test emergency override
controller.emergency_override('center')
time.sleep(2)
controller.clear_emergency_override('center')

controller.stop()
```

## Performance Optimization

### Tuning Parameters

1. **Control Interval**: Lower values for more responsive control
2. **Monitoring Interval**: Balance between data freshness and performance
3. **Retry Delay**: Adjust based on network conditions
4. **Max Retries**: Set based on reliability requirements

### Memory Management

```python
# Monitor memory usage
import psutil
process = psutil.Process()
memory_usage = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory usage: {memory_usage:.1f} MB")
```

### CPU Optimization

```python
# Monitor CPU usage
cpu_percent = psutil.cpu_percent(interval=1)
print(f"CPU usage: {cpu_percent:.1f}%")
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if SUMO is running
   - Verify port and host settings
   - Check firewall settings

2. **High Error Rate**
   - Check SUMO stability
   - Verify network connection
   - Review log files for patterns

3. **Command Failures**
   - Check intersection IDs
   - Verify phase strings
   - Check command queue size

4. **Memory Leaks**
   - Monitor traffic data collection
   - Check for unclosed connections
   - Review log file size

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('traci_controller').setLevel(logging.DEBUG)

# Enable detailed error reporting
controller = RobustTraCIController()
controller.config['log_level'] = 'DEBUG'
```

### Health Monitoring

```python
# Continuous health monitoring
import time

while True:
    if not controller.health_check():
        print("Controller unhealthy, attempting restart...")
        controller.stop()
        time.sleep(5)
        controller.start()
    
    time.sleep(30)  # Check every 30 seconds
```

## Integration with ML System

### ML Optimization Integration

```python
# Get traffic data for ML processing
traffic_data = controller.get_all_traffic_data()

# Process with ML model
ml_predictions = ml_model.predict(traffic_data)

# Apply ML recommendations
for intersection_id, prediction in ml_predictions.items():
    if prediction['phase'] != traffic_data[intersection_id].current_phase:
        controller.set_phase(intersection_id, prediction['phase'])
```

### Real-time Feedback Loop

```python
# Continuous ML optimization
def ml_optimization_loop():
    while controller.running:
        # Get current traffic state
        traffic_data = controller.get_all_traffic_data()
        
        # Run ML optimization
        optimizations = ml_optimizer.optimize(traffic_data)
        
        # Apply optimizations
        for intersection_id, optimization in optimizations.items():
            controller.set_phase(intersection_id, optimization['phase'])
        
        time.sleep(1.0)  # Optimize every second

# Start ML optimization in separate thread
ml_thread = threading.Thread(target=ml_optimization_loop, daemon=True)
ml_thread.start()
```

## Best Practices

### 1. Error Handling
- Always check return values
- Implement proper error recovery
- Use try-catch blocks for critical operations

### 2. Performance
- Monitor resource usage
- Tune parameters for your use case
- Use appropriate logging levels

### 3. Reliability
- Implement health checks
- Use fallback mechanisms
- Monitor error rates

### 4. Maintenance
- Regular log analysis
- Performance monitoring
- Configuration updates

## Conclusion

The robust TraCI controller provides a comprehensive solution for real-time traffic signal control with SUMO. It offers fault tolerance, real-time monitoring, and seamless integration with ML optimization systems. The system is designed for production use with comprehensive error handling, logging, and monitoring capabilities.

For additional support or questions, refer to the test files and source code documentation.
