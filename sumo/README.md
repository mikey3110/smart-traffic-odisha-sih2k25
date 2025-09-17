# SUMO Traffic Scenarios

This directory contains realistic SUMO traffic simulation networks and scenarios for the Smart Traffic Management System prototype.

## Quick Start

### Prerequisites
- SUMO (Simulation of Urban MObility) installed
- Python 3.6+ (for launcher scripts)
- TraCI Python module (for ML integration)

### Installation
```bash
# Install SUMO (Ubuntu/Debian)
sudo apt-get install sumo sumo-tools

# Install SUMO (Windows)
# Download from: https://sumo.dlr.de/docs/Downloads.php

# Install TraCI Python module
pip install traci
```

### Running Scenarios

#### Using Python Launcher (Recommended)
```bash
# List available scenarios
python launch_scenarios.py --list

# Launch normal traffic scenario
python launch_scenarios.py normal

# Launch rush hour scenario
python launch_scenarios.py rush_hour

# Launch emergency vehicle scenario
python launch_scenarios.py emergency

# Run all scenarios in batch
python launch_scenarios.py --batch

# Validate all scenarios
python launch_scenarios.py --validate
```

#### Direct SUMO Commands
```bash
# Normal traffic
sumo-gui -c configs/normal_traffic.sumocfg --remote-port 8813

# Rush hour
sumo-gui -c configs/rush_hour.sumocfg --remote-port 8813

# Emergency vehicles
sumo-gui -c configs/emergency_vehicle.sumocfg --remote-port 8813
```

### Testing
```bash
# Run comprehensive tests
python test_scenarios.py

# Test specific scenario
python launch_scenarios.py --validate
```

## Scenarios

### 1. Normal Traffic (`normal_traffic`)
- **Duration**: 2 hours
- **Description**: Standard traffic conditions
- **Traffic Volume**: 300-280 vehicles per direction
- **Vehicle Types**: Cars, auto-rickshaws, motorcycles, buses, trucks
- **Use Case**: Baseline testing and ML model training

### 2. Rush Hour (`rush_hour`)
- **Duration**: 7 hours
- **Description**: High-density traffic during peak hours
- **Traffic Volume**: 500-600 vehicles per direction during rush
- **Phases**: Morning rush (7-9 AM), Evening rush (5-7 PM)
- **Use Case**: Stress testing and performance optimization

### 3. Emergency Vehicle (`emergency_vehicle`)
- **Duration**: 1 hour
- **Description**: Emergency vehicle priority testing
- **Emergency Types**: Ambulance, fire truck, police car
- **Traffic Volume**: 50% reduced normal traffic
- **Use Case**: Priority system testing and safety validation

## Network Architecture

### 4-Way Intersection
- **Approach lanes**: 3 lanes per direction
- **Speed limits**: 50 km/h (main), 30 km/h (left turn), 25 km/h (right turn)
- **Traffic lights**: 6-phase cycle with 45-second green phases
- **Pedestrian crossings**: North-South and East-West
- **Turning lanes**: Dedicated left and right turn lanes

### Vehicle Types
- **Cars**: 4.5m length, 50 km/h max speed
- **Auto-rickshaws**: 3.0m length, 40 km/h max speed
- **Motorcycles**: 2.0m length, 60 km/h max speed
- **Buses**: 12.0m length, 40 km/h max speed
- **Trucks**: 15.0m length, 35 km/h max speed
- **Emergency vehicles**: Various sizes, 70-90 km/h max speed

## File Structure

```
sumo/
├── networks/                    # Network definition files
│   ├── intersection_4way.nod.xml    # Nodes (intersections, endpoints)
│   ├── intersection_4way.edg.xml    # Edges (road segments)
│   ├── intersection_4way.con.xml    # Connections (lane assignments)
│   ├── intersection_4way.tll.xml    # Traffic light logic
│   └── intersection_4way.net.xml    # Complete network
├── routes/                      # Vehicle route definitions
│   ├── normal_traffic.rou.xml       # Normal traffic routes
│   ├── rush_hour.rou.xml            # Rush hour routes
│   └── emergency_vehicle.rou.xml    # Emergency vehicle routes
├── configs/                     # Scenario configurations
│   ├── normal_traffic.sumocfg       # Normal traffic config
│   ├── rush_hour.sumocfg            # Rush hour config
│   └── emergency_vehicle.sumocfg    # Emergency vehicle config
├── scenarios/                   # Simulation output files
│   └── (generated during simulation)
├── launch_scenarios.py          # Scenario launcher script
├── test_scenarios.py            # Test validation script
└── README.md                    # This file
```

## TraCI Integration

The scenarios are designed for TraCI integration with the ML optimization system:

```python
import traci

# Connect to SUMO
traci.init(8813)

# Get traffic light state
tl_state = traci.trafficlight.getRedYellowGreenState("center")

# Set traffic light state
traci.trafficlight.setRedYellowGreenState("center", "GGGrrrrrGGGrrrrr")

# Get vehicle counts
vehicle_count = traci.inductionloop.getLastStepVehicleNumber("detector_0")

# Get simulation time
sim_time = traci.simulation.getTime()

# Close connection
traci.close()
```

## Performance Metrics

### Key Performance Indicators
- **Average waiting time**: Time vehicles spend waiting at intersection
- **Throughput**: Number of vehicles processed per hour
- **Queue length**: Maximum queue length per approach
- **Fuel consumption**: Estimated fuel usage based on vehicle behavior
- **Emissions**: CO2 and other pollutant emissions
- **Pedestrian waiting time**: Time pedestrians wait to cross

### ML Optimization Targets
- **Wait time reduction**: 30-45% improvement over baseline
- **Throughput increase**: 20-30% improvement
- **Fuel efficiency**: 15-25% improvement
- **Pedestrian safety**: Reduced crossing wait times

## Troubleshooting

### Common Issues

1. **SUMO not found**
   ```bash
   # Check if SUMO is installed
   sumo --version
   
   # Install SUMO if missing
   sudo apt-get install sumo sumo-tools
   ```

2. **Configuration errors**
   ```bash
   # Validate configurations
   python launch_scenarios.py --validate
   
   # Check specific configuration
   sumo -c configs/normal_traffic.sumocfg --check-config
   ```

3. **Port conflicts**
   ```bash
   # Use different port
   python launch_scenarios.py normal --port 8814
   
   # Check port usage
   netstat -an | grep 8813
   ```

4. **File not found**
   ```bash
   # Check file structure
   ls -la networks/ routes/ configs/
   
   # Regenerate network if needed
   netconvert --node-files networks/intersection_4way.nod.xml \
              --edge-files networks/intersection_4way.edg.xml \
              --connection-files networks/intersection_4way.con.xml \
              --output-file networks/intersection_4way.net.xml
   ```

### Validation Commands

```bash
# Test all scenarios
python test_scenarios.py

# Validate network
netconvert --node-files networks/intersection_4way.nod.xml \
           --edge-files networks/intersection_4way.edg.xml \
           --connection-files networks/intersection_4way.con.xml \
           --output-file networks/intersection_4way.net.xml

# Validate routes
duarouter --net-file networks/intersection_4way.net.xml \
          --route-files routes/normal_traffic.rou.xml \
          --output-file routes/normal_traffic.rou.xml

# Check configuration
sumo -c configs/normal_traffic.sumocfg --check-config
```

## Advanced Usage

### Custom Scenarios
1. Create new route file in `routes/`
2. Create new configuration file in `configs/`
3. Add scenario to `launch_scenarios.py`
4. Test with `python test_scenarios.py`

### Network Modifications
1. Edit network files in `networks/`
2. Regenerate network with `netconvert`
3. Update route files if needed
4. Test with `python launch_scenarios.py --validate`

### ML Integration
1. Use TraCI to connect to running simulation
2. Implement ML algorithms for traffic optimization
3. Monitor performance metrics
4. Adjust traffic light timing based on ML predictions

## Documentation

For detailed documentation, see:
- [SUMO Scenarios Documentation](../docs/sumo_scenarios.md)
- [SUMO User Documentation](https://sumo.dlr.de/docs/)
- [TraCI Documentation](https://sumo.dlr.de/docs/TraCI.html)

## Support

For issues and questions:
1. Check troubleshooting section above
2. Run validation tests
3. Check SUMO documentation
4. Contact development team

## License

This project is part of the Smart Traffic Management System for the Smart India Hackathon 2025.
