# SUMO Traffic Scenarios Documentation

## Overview

This document describes the realistic SUMO traffic simulation networks and scenarios created for the Smart Traffic Management System prototype. The scenarios are designed to reflect real Indian urban traffic patterns and test various ML optimization strategies.

## Network Architecture

### 4-Way Intersection Network (`intersection_4way.net.xml`)

The base network consists of a realistic 4-way intersection with the following features:

#### **Nodes**
- **Approach nodes**: `north_in`, `south_in`, `east_in`, `west_in`
- **Exit nodes**: `north_out`, `south_out`, `east_out`, `west_out`
- **Intersection center**: `center` (traffic light controlled)
- **Turning lane nodes**: Dedicated nodes for left and right turns
- **Pedestrian crossings**: `ped_north`, `ped_south`, `ped_east`, `ped_west`

#### **Edges**
- **Main approach lanes**: 3 lanes per direction with 50 km/h speed limit
- **Left turn lanes**: Dedicated lanes with 30 km/h speed limit
- **Right turn lanes**: Dedicated lanes with 25 km/h speed limit
- **Pedestrian crossings**: 5 km/h speed limit for pedestrian safety

#### **Traffic Light Logic**
- **Phase 0**: North-South Green (45 seconds)
- **Phase 1**: North-South Yellow (3 seconds)
- **Phase 2**: All Red (2 seconds)
- **Phase 3**: East-West Green (45 seconds)
- **Phase 4**: East-West Yellow (3 seconds)
- **Phase 5**: All Red (2 seconds)

#### **Alternative Programs**
- **Program 1**: Rush hour (shorter 30-second phases)
- **Program 2**: Emergency vehicle priority (all green for emergency vehicles)

## Vehicle Types

### **Cars** (`car`)
- Acceleration: 2.5 m/s²
- Deceleration: 4.5 m/s²
- Length: 4.5m
- Max Speed: 50 km/h
- Color: Yellow

### **Auto-rickshaws** (`auto_rickshaw`)
- Acceleration: 2.0 m/s²
- Deceleration: 4.0 m/s²
- Length: 3.0m
- Max Speed: 40 km/h
- Color: Green

### **Motorcycles** (`motorcycle`)
- Acceleration: 3.0 m/s²
- Deceleration: 5.0 m/s²
- Length: 2.0m
- Max Speed: 60 km/h
- Color: Red

### **Buses** (`bus`)
- Acceleration: 1.5 m/s²
- Deceleration: 3.0 m/s²
- Length: 12.0m
- Max Speed: 40 km/h
- Color: Blue

### **Trucks** (`truck`)
- Acceleration: 1.0 m/s²
- Deceleration: 2.5 m/s²
- Length: 15.0m
- Max Speed: 35 km/h
- Color: Gray

### **Emergency Vehicles**
- **Ambulance** (`ambulance`): 5.0m length, 80 km/h max speed, White color
- **Fire Truck** (`fire_truck`): 8.0m length, 70 km/h max speed, Red color
- **Police** (`police`): 4.5m length, 90 km/h max speed, Blue color

## Scenarios

### 1. Normal Traffic Scenario (`normal_traffic`)

**Configuration**: `configs/normal_traffic.sumocfg`
**Duration**: 2 hours (7,200 seconds)
**Description**: Standard traffic conditions for baseline testing

#### **Traffic Flows**
- **Cars**: 300-280 vehicles per direction
- **Auto-rickshaws**: 150-140 vehicles per direction
- **Motorcycles**: 200-180 vehicles per direction
- **Buses**: 20-18 vehicles per direction
- **Trucks**: 15-12 vehicles per direction
- **Pedestrians**: 100-90 pedestrians per crossing

#### **Route Patterns**
- **Through traffic**: 40-45% of vehicles
- **Left turns**: 20-25% of vehicles
- **Right turns**: 15-20% of vehicles
- **U-turns**: 10-15% of vehicles
- **Cross traffic**: 5-10% of vehicles

### 2. Rush Hour Scenario (`rush_hour`)

**Configuration**: `configs/rush_hour.sumocfg`
**Duration**: 7 hours (25,200 seconds)
**Description**: High-density traffic during peak hours

#### **Traffic Phases**
- **Morning Rush (7-9 AM)**: 500-450 vehicles per direction
- **Evening Rush (5-7 PM)**: 600-550 vehicles per direction
- **Off-peak periods**: Reduced traffic density

#### **Traffic Characteristics**
- **Cars**: 500-600 vehicles per direction during rush
- **Auto-rickshaws**: 250-300 vehicles per direction during rush
- **Motorcycles**: 400-500 vehicles per direction during rush
- **Buses**: 35-40 vehicles per direction during rush
- **Trucks**: 20-25 vehicles per direction during rush
- **Pedestrians**: 150-180 pedestrians per crossing during rush

#### **Traffic Light Adjustments**
- **Shorter phases**: 30 seconds instead of 45 seconds
- **Faster adaptation**: 5-second intervals instead of 10 seconds
- **Reduced detector gap**: 1.5 seconds instead of 2.0 seconds

### 3. Emergency Vehicle Scenario (`emergency_vehicle`)

**Configuration**: `configs/emergency_vehicle.sumocfg`
**Duration**: 1 hour (3,600 seconds)
**Description**: Tests emergency vehicle priority and traffic management

#### **Emergency Vehicle Patterns**
- **Single emergency vehicles**: Ambulance, fire truck, police car
- **Emergency convoys**: Multiple emergency vehicles together
- **Simultaneous emergencies**: Multiple emergency vehicles from different directions
- **Priority testing**: Emergency vehicles get right-of-way

#### **Traffic Adjustments**
- **Reduced normal traffic**: 50% reduction during emergency scenarios
- **Emergency vehicle priority**: All traffic lights turn green for emergency vehicles
- **Traffic clearance**: Normal vehicles yield to emergency vehicles

## Configuration Parameters

### **Simulation Settings**
- **Step length**: 0.1 seconds
- **Random seed**: 42 (for reproducible results)
- **TraCI port**: 8813 (configurable)

### **Traffic Light Settings**
- **Minimum phase duration**: 3-5 seconds
- **Maximum phase duration**: 45-60 seconds
- **Detector gap**: 1.5-2.0 seconds
- **Adaptation interval**: 5-10 seconds

### **Vehicle Behavior**
- **Car-following model**: Krauss
- **Lane-changing model**: LC2013
- **Junction model**: Right-before-left
- **Keep-clear**: Enabled for all vehicle types

### **Output Files**
- **FCD output**: Vehicle position and speed data
- **Summary output**: Simulation statistics
- **Trip info**: Individual vehicle trip data
- **Vehicle routes**: Complete route information
- **Queue output**: Traffic queue lengths
- **Jam time output**: Traffic jam duration

## Usage Instructions

### **Launching Scenarios**

#### **Using Python Launcher**
```bash
# List available scenarios
python launch_scenarios.py --list

# Launch specific scenario
python launch_scenarios.py normal

# Launch with GUI
python launch_scenarios.py rush_hour

# Launch without GUI (headless)
python launch_scenarios.py emergency --no-gui

# Run all scenarios in batch
python launch_scenarios.py --batch

# Validate all scenarios
python launch_scenarios.py --validate
```

#### **Direct SUMO Commands**
```bash
# Normal traffic
sumo-gui -c configs/normal_traffic.sumocfg --remote-port 8813

# Rush hour
sumo-gui -c configs/rush_hour.sumocfg --remote-port 8813

# Emergency vehicles
sumo-gui -c configs/emergency_vehicle.sumocfg --remote-port 8813
```

### **TraCI Integration**

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
```

## Performance Metrics

### **Key Performance Indicators**
- **Average waiting time**: Time vehicles spend waiting at intersection
- **Throughput**: Number of vehicles processed per hour
- **Queue length**: Maximum queue length per approach
- **Fuel consumption**: Estimated fuel usage based on vehicle behavior
- **Emissions**: CO2 and other pollutant emissions
- **Pedestrian waiting time**: Time pedestrians wait to cross

### **ML Optimization Targets**
- **Wait time reduction**: 30-45% improvement over baseline
- **Throughput increase**: 20-30% improvement
- **Fuel efficiency**: 15-25% improvement
- **Pedestrian safety**: Reduced crossing wait times

## File Structure

```
sumo/
├── networks/
│   ├── intersection_4way.nod.xml      # Network nodes
│   ├── intersection_4way.edg.xml      # Network edges
│   ├── intersection_4way.con.xml      # Network connections
│   ├── intersection_4way.tll.xml      # Traffic light logic
│   └── intersection_4way.net.xml      # Complete network
├── routes/
│   ├── normal_traffic.rou.xml         # Normal traffic routes
│   ├── rush_hour.rou.xml              # Rush hour routes
│   └── emergency_vehicle.rou.xml      # Emergency vehicle routes
├── configs/
│   ├── normal_traffic.sumocfg         # Normal traffic config
│   ├── rush_hour.sumocfg              # Rush hour config
│   └── emergency_vehicle.sumocfg      # Emergency vehicle config
├── scenarios/
│   └── (output files generated during simulation)
└── launch_scenarios.py                # Scenario launcher script
```

## Troubleshooting

### **Common Issues**

1. **SUMO not found**: Install SUMO or specify path with `--sumo-path`
2. **Configuration errors**: Use `--validate` to check configurations
3. **Port conflicts**: Use different ports with `--port` parameter
4. **File not found**: Ensure all network and route files exist

### **Validation Commands**

```bash
# Check network validity
netconvert --node-files networks/intersection_4way.nod.xml \
           --edge-files networks/intersection_4way.edg.xml \
           --connection-files networks/intersection_4way.con.xml \
           --output-file networks/intersection_4way.net.xml

# Check route validity
duarouter --net-file networks/intersection_4way.net.xml \
          --route-files routes/normal_traffic.rou.xml \
          --output-file routes/normal_traffic.rou.xml

# Check configuration validity
sumo -c configs/normal_traffic.sumocfg --check-config
```

## Future Enhancements

### **Planned Scenarios**
- **Weather conditions**: Rain, fog, and extreme weather scenarios
- **Special events**: Festival traffic, sports events, political rallies
- **Construction zones**: Temporary lane closures and detours
- **Multi-intersection networks**: Connected intersection systems
- **Dynamic routing**: Real-time route optimization

### **Advanced Features**
- **Pedestrian behavior**: More realistic pedestrian crossing patterns
- **Public transport**: Bus stops and transit priority
- **Parking**: On-street parking and loading zones
- **Cycling**: Dedicated bicycle lanes and behavior
- **Mixed traffic**: More diverse vehicle types and behaviors

## Conclusion

These SUMO scenarios provide a comprehensive testing environment for the Smart Traffic Management System. The realistic Indian urban traffic patterns, combined with various traffic conditions, enable thorough testing of ML optimization algorithms and real-time traffic management strategies.

The scenarios are designed to be easily extensible and configurable, allowing for future enhancements and additional traffic patterns as needed for the competition demonstration.
