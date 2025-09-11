"""
Vehicle Detection and Counting System for SUMO Integration
Real-time vehicle detection, counting, and classification at intersection approaches
"""

import traci
import traci.constants as tc
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import threading


class VehicleType(Enum):
    """Vehicle type classification"""
    PASSENGER = "passenger"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    EMERGENCY = "emergency"
    UNKNOWN = "unknown"


class DetectionZone(Enum):
    """Detection zone types"""
    APPROACH = "approach"
    STOP_LINE = "stop_line"
    INTERSECTION = "intersection"
    EXIT = "exit"


@dataclass
class VehicleDetection:
    """Vehicle detection data"""
    vehicle_id: str
    vehicle_type: VehicleType
    position: Tuple[float, float]
    speed: float
    lane: str
    detection_zone: DetectionZone
    timestamp: datetime
    length: float
    width: float
    waiting_time: float
    acceleration: float
    route: List[str]


@dataclass
class LaneCounts:
    """Lane vehicle counts"""
    lane_id: str
    total_vehicles: int
    waiting_vehicles: int
    moving_vehicles: int
    vehicle_types: Dict[VehicleType, int]
    average_speed: float
    average_waiting_time: float
    timestamp: datetime


@dataclass
class IntersectionCounts:
    """Intersection vehicle counts"""
    intersection_id: str
    approach_lanes: Dict[str, LaneCounts]
    total_vehicles: int
    total_waiting: int
    total_moving: int
    timestamp: datetime


class VehicleDetector:
    """
    Vehicle Detection and Counting System for SUMO Integration
    
    Features:
    - Real-time vehicle detection
    - Vehicle type classification
    - Lane-based counting
    - Waiting time calculation
    - Speed and acceleration tracking
    - Detection zone management
    - Data export and API integration
    """
    
    def __init__(self, detection_range: float = 100.0):
        self.detection_range = detection_range
        self.logger = logging.getLogger(__name__)
        
        # Detection data
        self.detected_vehicles: Dict[str, VehicleDetection] = {}
        self.lane_counts: Dict[str, LaneCounts] = {}
        self.intersection_counts: Dict[str, IntersectionCounts] = {}
        
        # Detection zones
        self.detection_zones: Dict[str, List[str]] = {}  # intersection_id -> lane_ids
        self.stop_line_positions: Dict[str, float] = {}  # lane_id -> position
        
        # Tracking
        self.vehicle_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.detection_timestamps: Dict[str, datetime] = {}
        
        # Performance metrics
        self.detection_count = 0
        self.missed_detections = 0
        self.false_detections = 0
        
        # Threading
        self.detection_thread = None
        self.is_detecting = False
        self.detection_interval = 0.1  # seconds
        
        self.logger.info("Vehicle Detector initialized")
    
    def initialize_detection_zones(self, intersection_ids: List[str]):
        """Initialize detection zones for intersections"""
        for intersection_id in intersection_ids:
            try:
                # Get controlled lanes for intersection
                controlled_lanes = traci.trafficlight.getControlledLanes(intersection_id)
                
                # Create detection zones
                approach_lanes = []
                for lane in controlled_lanes:
                    # Get lane length and position
                    lane_length = traci.lane.getLength(lane)
                    stop_line_position = lane_length * 0.1  # 10% from intersection
                    
                    self.stop_line_positions[lane] = stop_line_position
                    approach_lanes.append(lane)
                
                self.detection_zones[intersection_id] = approach_lanes
                
                self.logger.info(f"Initialized detection zones for {intersection_id}: {len(approach_lanes)} lanes")
                
            except Exception as e:
                self.logger.error(f"Error initializing detection zones for {intersection_id}: {e}")
    
    def start_detection(self):
        """Start vehicle detection"""
        if self.is_detecting:
            return
        
        self.is_detecting = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        self.logger.info("Vehicle detection started")
    
    def stop_detection(self):
        """Stop vehicle detection"""
        self.is_detecting = False
        if self.detection_thread:
            self.detection_thread.join(timeout=5)
        
        self.logger.info("Vehicle detection stopped")
    
    def _detection_loop(self):
        """Main detection loop"""
        while self.is_detecting:
            try:
                # Detect vehicles in all zones
                for intersection_id, lanes in self.detection_zones.items():
                    self._detect_vehicles_in_zones(intersection_id, lanes)
                
                # Update counts
                self._update_lane_counts()
                self._update_intersection_counts()
                
                time.sleep(self.detection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in detection loop: {e}")
                time.sleep(1.0)
    
    def _detect_vehicles_in_zones(self, intersection_id: str, lanes: List[str]):
        """Detect vehicles in specific zones"""
        try:
            # Get all vehicles in simulation
            all_vehicles = traci.vehicle.getIDList()
            
            for vehicle_id in all_vehicles:
                try:
                    # Get vehicle position and lane
                    position = traci.vehicle.getPosition(vehicle_id)
                    lane = traci.vehicle.getLaneID(vehicle_id)
                    
                    # Check if vehicle is in detection zone
                    if lane in lanes and self._is_in_detection_zone(vehicle_id, lane, position):
                        # Detect vehicle
                        detection = self._create_vehicle_detection(vehicle_id, lane, position)
                        
                        if detection:
                            self.detected_vehicles[vehicle_id] = detection
                            self.detection_timestamps[vehicle_id] = datetime.now()
                            self.detection_count += 1
                            
                            # Add to history
                            self.vehicle_history[vehicle_id].append(detection)
                
                except Exception as e:
                    self.logger.warning(f"Error detecting vehicle {vehicle_id}: {e}")
                    continue
        
        except Exception as e:
            self.logger.error(f"Error detecting vehicles in zones for {intersection_id}: {e}")
    
    def _is_in_detection_zone(self, vehicle_id: str, lane: str, position: Tuple[float, float]) -> bool:
        """Check if vehicle is in detection zone"""
        try:
            # Get lane length and vehicle position along lane
            lane_length = traci.lane.getLength(lane)
            vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
            
            # Check if vehicle is within detection range of stop line
            stop_line_pos = self.stop_line_positions.get(lane, lane_length * 0.1)
            distance_to_stop_line = stop_line_pos - vehicle_position
            
            return 0 <= distance_to_stop_line <= self.detection_range
            
        except Exception as e:
            self.logger.warning(f"Error checking detection zone for vehicle {vehicle_id}: {e}")
            return False
    
    def _create_vehicle_detection(self, vehicle_id: str, lane: str, position: Tuple[float, float]) -> Optional[VehicleDetection]:
        """Create vehicle detection data"""
        try:
            # Get vehicle properties
            speed = traci.vehicle.getSpeed(vehicle_id)
            vehicle_type_str = traci.vehicle.getTypeID(vehicle_id)
            length = traci.vehicle.getLength(vehicle_id)
            width = traci.vehicle.getWidth(vehicle_id)
            waiting_time = traci.vehicle.getWaitingTime(vehicle_id)
            acceleration = traci.vehicle.getAcceleration(vehicle_id)
            route = traci.vehicle.getRoute(vehicle_id)
            
            # Classify vehicle type
            vehicle_type = self._classify_vehicle_type(vehicle_type_str)
            
            # Determine detection zone
            detection_zone = self._determine_detection_zone(vehicle_id, lane)
            
            return VehicleDetection(
                vehicle_id=vehicle_id,
                vehicle_type=vehicle_type,
                position=position,
                speed=speed,
                lane=lane,
                detection_zone=detection_zone,
                timestamp=datetime.now(),
                length=length,
                width=width,
                waiting_time=waiting_time,
                acceleration=acceleration,
                route=route
            )
            
        except Exception as e:
            self.logger.warning(f"Error creating detection for vehicle {vehicle_id}: {e}")
            return None
    
    def _classify_vehicle_type(self, vehicle_type_str: str) -> VehicleType:
        """Classify vehicle type from SUMO type ID"""
        vehicle_type_str = vehicle_type_str.lower()
        
        if 'passenger' in vehicle_type_str or 'car' in vehicle_type_str:
            return VehicleType.PASSENGER
        elif 'truck' in vehicle_type_str or 'heavy' in vehicle_type_str:
            return VehicleType.TRUCK
        elif 'bus' in vehicle_type_str or 'public' in vehicle_type_str:
            return VehicleType.BUS
        elif 'motorcycle' in vehicle_type_str or 'bike' in vehicle_type_str:
            return VehicleType.MOTORCYCLE
        elif 'emergency' in vehicle_type_str or 'ambulance' in vehicle_type_str:
            return VehicleType.EMERGENCY
        else:
            return VehicleType.UNKNOWN
    
    def _determine_detection_zone(self, vehicle_id: str, lane: str) -> DetectionZone:
        """Determine detection zone for vehicle"""
        try:
            # Get vehicle position along lane
            vehicle_position = traci.vehicle.getLanePosition(vehicle_id)
            lane_length = traci.lane.getLength(lane)
            stop_line_pos = self.stop_line_positions.get(lane, lane_length * 0.1)
            
            distance_to_stop_line = stop_line_pos - vehicle_position
            
            if distance_to_stop_line < 0:
                return DetectionZone.INTERSECTION
            elif distance_to_stop_line < 10:
                return DetectionZone.STOP_LINE
            elif distance_to_stop_line < 50:
                return DetectionZone.APPROACH
            else:
                return DetectionZone.APPROACH
                
        except Exception as e:
            self.logger.warning(f"Error determining detection zone for vehicle {vehicle_id}: {e}")
            return DetectionZone.APPROACH
    
    def _update_lane_counts(self):
        """Update lane vehicle counts"""
        for lane_id in self.stop_line_positions.keys():
            try:
                # Count vehicles in lane
                lane_vehicles = [v for v in self.detected_vehicles.values() if v.lane == lane_id]
                
                # Calculate counts by type
                vehicle_types = defaultdict(int)
                total_waiting = 0
                total_moving = 0
                speeds = []
                waiting_times = []
                
                for vehicle in lane_vehicles:
                    vehicle_types[vehicle.vehicle_type] += 1
                    
                    if vehicle.speed < 0.1:  # Consider stopped
                        total_waiting += 1
                        waiting_times.append(vehicle.waiting_time)
                    else:
                        total_moving += 1
                        speeds.append(vehicle.speed)
                
                # Create lane counts
                self.lane_counts[lane_id] = LaneCounts(
                    lane_id=lane_id,
                    total_vehicles=len(lane_vehicles),
                    waiting_vehicles=total_waiting,
                    moving_vehicles=total_moving,
                    vehicle_types=dict(vehicle_types),
                    average_speed=np.mean(speeds) if speeds else 0.0,
                    average_waiting_time=np.mean(waiting_times) if waiting_times else 0.0,
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                self.logger.error(f"Error updating lane counts for {lane_id}: {e}")
    
    def _update_intersection_counts(self):
        """Update intersection vehicle counts"""
        for intersection_id, lanes in self.detection_zones.items():
            try:
                # Get lane counts for intersection
                approach_lanes = {}
                total_vehicles = 0
                total_waiting = 0
                total_moving = 0
                
                for lane_id in lanes:
                    if lane_id in self.lane_counts:
                        approach_lanes[lane_id] = self.lane_counts[lane_id]
                        total_vehicles += self.lane_counts[lane_id].total_vehicles
                        total_waiting += self.lane_counts[lane_id].waiting_vehicles
                        total_moving += self.lane_counts[lane_id].moving_vehicles
                
                # Create intersection counts
                self.intersection_counts[intersection_id] = IntersectionCounts(
                    intersection_id=intersection_id,
                    approach_lanes=approach_lanes,
                    total_vehicles=total_vehicles,
                    total_waiting=total_waiting,
                    total_moving=total_moving,
                    timestamp=datetime.now()
                )
                
            except Exception as e:
                self.logger.error(f"Error updating intersection counts for {intersection_id}: {e}")
    
    def get_lane_counts(self, lane_id: str) -> Optional[LaneCounts]:
        """Get vehicle counts for specific lane"""
        return self.lane_counts.get(lane_id)
    
    def get_intersection_counts(self, intersection_id: str) -> Optional[IntersectionCounts]:
        """Get vehicle counts for specific intersection"""
        return self.intersection_counts.get(intersection_id)
    
    def get_all_intersection_counts(self) -> Dict[str, IntersectionCounts]:
        """Get vehicle counts for all intersections"""
        return self.intersection_counts.copy()
    
    def get_vehicle_detection(self, vehicle_id: str) -> Optional[VehicleDetection]:
        """Get detection data for specific vehicle"""
        return self.detected_vehicles.get(vehicle_id)
    
    def get_vehicles_in_lane(self, lane_id: str) -> List[VehicleDetection]:
        """Get all vehicles detected in specific lane"""
        return [v for v in self.detected_vehicles.values() if v.lane == lane_id]
    
    def get_vehicles_by_type(self, vehicle_type: VehicleType) -> List[VehicleDetection]:
        """Get all vehicles of specific type"""
        return [v for v in self.detected_vehicles.values() if v.vehicle_type == vehicle_type]
    
    def get_vehicles_in_zone(self, detection_zone: DetectionZone) -> List[VehicleDetection]:
        """Get all vehicles in specific detection zone"""
        return [v for v in self.detected_vehicles.values() if v.detection_zone == detection_zone]
    
    def get_detection_statistics(self) -> Dict[str, Any]:
        """Get detection system statistics"""
        return {
            'total_detections': self.detection_count,
            'missed_detections': self.missed_detections,
            'false_detections': self.false_detections,
            'detection_accuracy': self.detection_count / max(1, self.detection_count + self.missed_detections),
            'active_vehicles': len(self.detected_vehicles),
            'detection_zones': len(self.detection_zones),
            'monitored_lanes': len(self.stop_line_positions),
            'is_detecting': self.is_detecting
        }
    
    def export_detection_data(self, filepath: str):
        """Export vehicle detection data"""
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'detection_statistics': self.get_detection_statistics(),
                'lane_counts': {k: {
                    'lane_id': v.lane_id,
                    'total_vehicles': v.total_vehicles,
                    'waiting_vehicles': v.waiting_vehicles,
                    'moving_vehicles': v.moving_vehicles,
                    'vehicle_types': {t.value: count for t, count in v.vehicle_types.items()},
                    'average_speed': v.average_speed,
                    'average_waiting_time': v.average_waiting_time,
                    'timestamp': v.timestamp.isoformat()
                } for k, v in self.lane_counts.items()},
                'intersection_counts': {k: {
                    'intersection_id': v.intersection_id,
                    'total_vehicles': v.total_vehicles,
                    'total_waiting': v.total_waiting,
                    'total_moving': v.total_moving,
                    'approach_lanes': list(v.approach_lanes.keys()),
                    'timestamp': v.timestamp.isoformat()
                } for k, v in self.intersection_counts.items()},
                'detected_vehicles': {k: {
                    'vehicle_id': v.vehicle_id,
                    'vehicle_type': v.vehicle_type.value,
                    'position': v.position,
                    'speed': v.speed,
                    'lane': v.lane,
                    'detection_zone': v.detection_zone.value,
                    'length': v.length,
                    'width': v.width,
                    'waiting_time': v.waiting_time,
                    'acceleration': v.acceleration,
                    'timestamp': v.timestamp.isoformat()
                } for k, v in self.detected_vehicles.items()}
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Vehicle detection data exported to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error exporting detection data: {e}")
    
    def clear_old_detections(self, max_age_seconds: float = 60.0):
        """Clear old vehicle detections"""
        current_time = datetime.now()
        vehicles_to_remove = []
        
        for vehicle_id, detection in self.detected_vehicles.items():
            age = (current_time - detection.timestamp).total_seconds()
            if age > max_age_seconds:
                vehicles_to_remove.append(vehicle_id)
        
        for vehicle_id in vehicles_to_remove:
            del self.detected_vehicles[vehicle_id]
            if vehicle_id in self.detection_timestamps:
                del self.detection_timestamps[vehicle_id]
        
        if vehicles_to_remove:
            self.logger.info(f"Cleared {len(vehicles_to_remove)} old detections")


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Create vehicle detector
    detector = VehicleDetector(detection_range=50.0)
    
    # Initialize detection zones
    intersections = ["intersection_1", "intersection_2"]
    detector.initialize_detection_zones(intersections)
    
    # Start detection
    detector.start_detection()
    
    try:
        # Run detection for some time
        time.sleep(30)
        
        # Get counts
        counts = detector.get_intersection_counts("intersection_1")
        if counts:
            print(f"Intersection counts: {counts.total_vehicles} vehicles, {counts.total_waiting} waiting")
        
        # Get statistics
        stats = detector.get_detection_statistics()
        print(f"Detection statistics: {stats}")
        
        # Export data
        detector.export_detection_data("vehicle_detection.json")
    
    finally:
        # Stop detection
        detector.stop_detection()
