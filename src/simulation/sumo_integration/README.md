# SUMO Traffic Simulation Integration

A comprehensive SUMO traffic simulation integration system that connects with backend APIs and ML optimizers for real-time traffic management.

## Features

### Core Integration
- **Real-time SUMO Simulation Control** using TraCI
- **Dynamic Traffic Light Control** based on ML optimizer output
- **Vehicle Detection and Counting** at intersection approaches
- **Realistic Traffic Demand Patterns** and route generation
- **Real-time Data Export** to backend API
- **Scenario Management** (peak hours, incidents, weather effects)
- **Visualization Tools** for simulation results
- **Validation and Calibration** tools

### Advanced Features
- **Multi-threaded Architecture** for real-time performance
- **Error Handling and Recovery** with automatic restart
- **Performance Monitoring** and metrics collection
- **Interactive Dashboards** with Plotly
- **Statistical Validation** of simulation accuracy
- **Parameter Calibration** using optimization algorithms
- **Export Capabilities** in multiple formats

## Installation

### Prerequisites
- Python 3.8+
- SUMO 1.15.0+
- TraCI Python bindings

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd src/simulation/sumo_integration

# Install dependencies
pip install -r requirements.txt

# Install SUMO (Ubuntu/Debian)
sudo apt-get install sumo sumo-tools

# Install SUMO (macOS)
brew install sumo

# Install SUMO (Windows)
# Download from https://sumo.dlr.de/docs/Downloads.php
```

### Configuration
```bash
# Copy example configuration
cp config/sumo_config.yaml.example config/sumo_config.yaml

# Edit configuration
nano config/sumo_config.yaml
```

## Usage

### Basic Usage
```bash
# Run with default scenario
python run_sumo_integration.py

# Run with specific scenario
python run_sumo_integration.py --scenario scenarios/basic_scenario.sumocfg

# Run with GUI
python run_sumo_integration.py --enable-gui

# Run with custom configuration
python run_sumo_integration.py --config config/custom_config.yaml
```

### Advanced Usage
```bash
# Run without validation
python run_sumo_integration.py --disable-validation

# Run without data export
python run_sumo_integration.py --disable-export

# Run without visualization
python run_sumo_integration.py --disable-visualization

# Run with debug logging
python run_sumo_integration.py --log-level DEBUG
```

### Programmatic Usage
```python
import asyncio
from sumo_integration_manager import SumoIntegrationManager, IntegrationConfig
from config.sumo_config import get_sumo_config

async def main():
    # Create configuration
    config = IntegrationConfig(
        sumo_config=get_sumo_config(),
        enable_visualization=True,
        enable_validation=True,
        enable_data_export=True
    )
    
    # Create manager
    manager = SumoIntegrationManager(config)
    
    # Start integration
    success = await manager.start_integration("scenarios/basic_scenario.sumocfg")
    
    if success:
        print("Integration started successfully")
        
        # Run for some time
        await asyncio.sleep(60)
        
        # Get status
        status = manager.get_integration_status()
        print(f"Status: {status}")
        
        # Stop integration
        manager.stop_integration()

# Run
asyncio.run(main())
```

## Architecture

### Component Overview
```
┌─────────────────────────────────────────────────────────────┐
│                 SUMO Integration Manager                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   SUMO      │  │  Traffic    │  │  Vehicle    │        │
│  │ Controller  │  │   Light     │  │  Detector   │        │
│  │             │  │ Controller  │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Demand    │  │    Data     │  │  Scenario   │        │
│  │  Generator  │  │  Exporter   │  │  Manager    │        │
│  │             │  │             │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │Visualization│  │ Validation  │  │    API      │        │
│  │   Tools     │  │   Tools     │  │ Integration │        │
│  │             │  │             │  │             │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow
```
SUMO Simulation → TraCI → SUMO Controller → Data Exporter → Backend API
                     ↓
              Traffic Light Controller ← ML Optimizer
                     ↓
              Vehicle Detector → Performance Metrics
                     ↓
              Visualization Tools → Interactive Dashboard
```

## Configuration

### SUMO Configuration
```yaml
# config/sumo_config.yaml
network:
  net_file: "networks/intersection.net.xml"
  route_file: "routes/vehicles.rou.xml"
  additional_file: "additional/traffic_lights.add.xml"

simulation:
  mode: "real_time"
  step_size: 1.0
  end_time: 3600.0
  enable_gui: false

traffic_lights:
  program_type: "ml_controlled"
  min_phase_duration: 5.0
  max_phase_duration: 60.0
  update_interval: 1.0

api_integration:
  enabled: true
  base_url: "http://localhost:8000"
  endpoint: "/api/v1/sumo/data"
  timeout: 10.0
  retry_attempts: 3
```

### Integration Configuration
```python
from sumo_integration_manager import IntegrationConfig

config = IntegrationConfig(
    sumo_config=get_sumo_config(),
    enable_visualization=True,
    enable_validation=True,
    enable_data_export=True,
    enable_scenario_management=True,
    log_level="INFO",
    auto_restart=True
)
```

## API Integration

### Data Export Format
```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "simulation_time": 3600.0,
  "vehicles": [
    {
      "id": "vehicle_1",
      "position": [100.0, 200.0],
      "speed": 30.0,
      "lane": "north_approach_0",
      "waiting_time": 5.0,
      "co2_emission": 0.1
    }
  ],
  "intersections": [
    {
      "id": "center_junction",
      "current_phase": 0,
      "phase_duration": 30.0,
      "vehicle_counts": {
        "north_approach_0": 5,
        "south_approach_0": 3
      }
    }
  ],
  "performance_metrics": {
    "total_vehicles": 50,
    "waiting_vehicles": 10,
    "average_speed": 25.0,
    "total_co2_emission": 100.0
  }
}
```

### ML Optimizer Integration
```python
# Apply ML control to traffic lights
from traffic_light_controller import ControlSignal

control_signal = ControlSignal(
    intersection_id="center_junction",
    phase_id=0,
    duration=45.0,
    confidence=0.85,
    timestamp=datetime.now(),
    algorithm_used="q_learning"
)

traffic_light_controller.apply_ml_control(control_signal)
```

## Visualization

### Real-time Dashboard
```python
from visualization.sumo_visualizer import SumoVisualizer

# Create visualizer
visualizer = SumoVisualizer()

# Start real-time visualization
visualizer.start_real_time_visualization()

# Create dashboard
dashboard = visualizer.create_dashboard(simulation_data)

# Create interactive dashboard
html_file = visualizer.create_interactive_dashboard(simulation_data)
```

### Available Visualizations
- **Real-time Vehicle Tracking**
- **Traffic Flow Heatmaps**
- **Performance Metrics Charts**
- **Intersection Status Display**
- **Emissions Over Time**
- **Speed Distribution**
- **Waiting Time Analysis**

## Validation and Calibration

### Validation
```python
from validation.validation_tools import SimulationValidator, ValidationConfig

# Create validator
validator = SimulationValidator(ValidationConfig())

# Start validation
validator.start_validation()

# Get validation statistics
stats = validator.get_validation_statistics()
print(f"Validation accuracy: {stats['error_rate']:.2%}")
```

### Calibration
```python
from validation.validation_tools import CalibrationConfig, CalibrationMethod

# Create calibration configuration
calibration_config = CalibrationConfig(
    method=CalibrationMethod.LEAST_SQUARES,
    parameters=['max_speed', 'acceleration', 'deceleration'],
    bounds={
        'max_speed': (20.0, 50.0),
        'acceleration': (1.0, 4.0),
        'deceleration': (2.0, 6.0)
    }
)

# Run calibration
result = validator.calibrate_parameters(calibration_config)
print(f"Calibrated parameters: {result.parameters}")
```

## Scenario Management

### Available Scenarios
- **Normal Traffic**: Standard traffic conditions
- **Morning Peak Hour**: Heavy morning commute traffic
- **Evening Peak Hour**: Heavy evening commute traffic
- **Rainy Weather**: Traffic with weather effects
- **Traffic Accident**: Incident scenario
- **Road Construction**: Construction work scenario

### Custom Scenarios
```python
from scenario_manager import ScenarioConfig, ScenarioType, WeatherCondition

# Create custom scenario
custom_scenario = ScenarioConfig(
    name="Custom Peak Hour",
    description="Custom peak hour scenario",
    scenario_type=ScenarioType.PEAK_HOUR,
    duration=7200.0,
    base_flow_rate=500.0,
    peak_multiplier=2.5,
    weather_condition=WeatherCondition.RAIN,
    visibility_reduction=0.3
)

# Add to scenario manager
scenario_manager.create_custom_scenario(custom_scenario)
```

## Testing

### Unit Tests
```bash
# Run unit tests
pytest tests/test_*.py -v

# Run with coverage
pytest tests/test_*.py --cov=src --cov-report=html
```

### Integration Tests
```bash
# Run integration tests
pytest tests/integration/ -v

# Run specific test
pytest tests/integration/test_sumo_integration.py::test_basic_integration
```

## Troubleshooting

### Common Issues

#### SUMO Not Found
```bash
# Check SUMO installation
which sumo
sumo --version

# Add SUMO to PATH
export PATH=$PATH:/path/to/sumo/bin
```

#### TraCI Connection Failed
```bash
# Check if SUMO is running
ps aux | grep sumo

# Check port availability
netstat -an | grep 8813
```

#### API Connection Failed
```bash
# Check API endpoint
curl http://localhost:8000/api/v1/health

# Check network connectivity
ping localhost
```

### Debug Mode
```bash
# Run with debug logging
python run_sumo_integration.py --log-level DEBUG

# Check log files
tail -f logs/sumo_integration.log
tail -f logs/sumo.log
```

## Performance Optimization

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB+ recommended
- **Storage**: 1GB+ free space
- **Network**: Stable connection for API integration

### Optimization Tips
1. **Reduce Simulation Step Size** for better accuracy
2. **Enable Data Compression** for large datasets
3. **Use Batch Processing** for API calls
4. **Optimize Visualization** settings
5. **Monitor Memory Usage** during long runs

## Contributing

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatting
black src/
flake8 src/

# Run type checking
mypy src/
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write comprehensive docstrings
- Add unit tests for new features

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide
- Contact the development team

## Changelog

### Version 1.0.0
- Initial release
- Basic SUMO integration
- Traffic light control
- Vehicle detection
- Data export
- Visualization tools
- Validation and calibration
- Scenario management
