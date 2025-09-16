"""
Safety Manager Component with Comprehensive Fallback Systems
Phase 2: Real-Time Optimization Loop & Safety Systems

Features:
- Webster's formula as baseline fallback mechanism
- Safety constraints: minimum/maximum green times, pedestrian crossing requirements
- Emergency vehicle priority override system
- Fail-safe mechanisms for system failures or anomalies
- Real-time monitoring of traffic safety metrics
"""

import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import json
import uuid
from collections import defaultdict, deque
import math


class SafetyLevel(Enum):
    """Safety level for traffic operations"""
    NORMAL = "normal"
    CAUTION = "caution"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SafetyViolation(Enum):
    """Types of safety violations"""
    MIN_GREEN_TIME_VIOLATION = "min_green_time_violation"
    MAX_GREEN_TIME_VIOLATION = "max_green_time_violation"
    PEDESTRIAN_CROSSING_VIOLATION = "pedestrian_crossing_violation"
    EMERGENCY_VEHICLE_DELAY = "emergency_vehicle_delay"
    SYSTEM_FAILURE = "system_failure"
    DATA_STALE = "data_stale"
    HIGH_CONGESTION = "high_congestion"
    WEATHER_HAZARD = "weather_hazard"


@dataclass
class SafetyConstraint:
    """Safety constraint definition"""
    constraint_id: str
    name: str
    description: str
    min_value: float
    max_value: float
    current_value: float
    violation_threshold: float
    is_active: bool = True
    priority: int = 1  # 1 = highest, 5 = lowest
    
    def is_violated(self) -> bool:
        """Check if constraint is violated"""
        if not self.is_active:
            return False
        
        return (self.current_value < self.min_value or 
                self.current_value > self.max_value or
                abs(self.current_value - self.violation_threshold) < 0.1)
    
    def get_violation_severity(self) -> float:
        """Get violation severity (0-1)"""
        if not self.is_violated():
            return 0.0
        
        if self.current_value < self.min_value:
            return (self.min_value - self.current_value) / self.min_value
        elif self.current_value > self.max_value:
            return (self.current_value - self.max_value) / self.max_value
        else:
            return abs(self.current_value - self.violation_threshold) / self.violation_threshold


@dataclass
class SafetyViolationReport:
    """Safety violation report"""
    violation_id: str
    timestamp: datetime
    intersection_id: str
    violation_type: SafetyViolation
    severity: float
    description: str
    constraint_id: str
    current_value: float
    expected_value: float
    corrective_action: str
    is_resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class EmergencyVehicleAlert:
    """Emergency vehicle alert"""
    alert_id: str
    timestamp: datetime
    intersection_id: str
    vehicle_type: str  # ambulance, fire_truck, police
    priority: int  # 1-5, 1 being highest
    estimated_arrival: datetime
    current_location: Tuple[float, float]
    destination: str
    is_active: bool = True
    handled_by: Optional[str] = None


class WebsterFormulaFallback:
    """Webster's formula implementation for traffic signal optimization"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Webster's formula parameters
        self.saturation_flow_rate = self.config.get('saturation_flow_rate', 1800)  # veh/h
        self.min_cycle_time = self.config.get('min_cycle_time', 40)  # seconds
        self.max_cycle_time = self.config.get('max_cycle_time', 120)  # seconds
        self.yellow_time = self.config.get('yellow_time', 3)  # seconds
        self.all_red_time = self.config.get('all_red_time', 2)  # seconds
        
        # Intersection-specific parameters
        self.intersection_configs = {}
    
    def configure_intersection(self, intersection_id: str, config: Dict[str, Any]):
        """Configure Webster's formula for specific intersection"""
        self.intersection_configs[intersection_id] = {
            'lanes': config.get('lanes', ['north', 'south', 'east', 'west']),
            'approach_angles': config.get('approach_angles', [0, 180, 90, 270]),
            'lane_widths': config.get('lane_widths', [3.5, 3.5, 3.5, 3.5]),
            'pedestrian_volume': config.get('pedestrian_volume', 0.1),
            'turning_ratios': config.get('turning_ratios', [0.1, 0.1, 0.1, 0.1])
        }
    
    def optimize_intersection(self, intersection_id: str, traffic_data: Dict[str, Any]) -> Dict[str, int]:
        """Optimize intersection using Webster's formula"""
        try:
            # Get intersection configuration
            config = self.intersection_configs.get(intersection_id, {})
            lanes = config.get('lanes', ['north', 'south', 'east', 'west'])
            
            # Get traffic volumes
            lane_counts = traffic_data.get('lane_counts', {})
            if not lane_counts:
                # Use default volumes if no data available
                lane_counts = {lane: 10 for lane in lanes}
            
            # Calculate Webster's formula
            cycle_time, phase_timings = self._calculate_webster_formula(
                intersection_id, lane_counts, config
            )
            
            # Apply safety constraints
            phase_timings = self._apply_safety_constraints(phase_timings)
            
            self.logger.info(f"Webster optimization for {intersection_id}: cycle={cycle_time}s, timings={phase_timings}")
            
            return phase_timings
            
        except Exception as e:
            self.logger.error(f"Webster optimization failed for {intersection_id}: {e}")
            # Return safe default timings
            return {lane: 30 for lane in lanes}
    
    def _calculate_webster_formula(self, intersection_id: str, lane_counts: Dict[str, int], 
                                  config: Dict[str, Any]) -> Tuple[int, Dict[str, int]]:
        """Calculate Webster's formula for optimal cycle time and phase timings"""
        lanes = config.get('lanes', ['north', 'south', 'east', 'west'])
        
        # Calculate flow ratios for each approach
        flow_ratios = {}
        total_flow = sum(lane_counts.values()) if lane_counts else 1
        
        for lane in lanes:
            flow = lane_counts.get(lane, 0)
            flow_ratios[lane] = flow / total_flow if total_flow > 0 else 0.25
        
        # Calculate critical flow ratio
        critical_flow_ratio = max(flow_ratios.values()) if flow_ratios else 0.5
        
        # Calculate optimal cycle time using Webster's formula
        # C = (1.5 * L + 5) / (1 - Y)
        # where L = lost time, Y = critical flow ratio
        
        lost_time = len(lanes) * (self.yellow_time + self.all_red_time)
        cycle_time = int((1.5 * lost_time + 5) / (1 - critical_flow_ratio))
        
        # Apply cycle time constraints
        cycle_time = max(self.min_cycle_time, min(self.max_cycle_time, cycle_time))
        
        # Calculate phase timings
        phase_timings = {}
        for lane in lanes:
            flow_ratio = flow_ratios.get(lane, 0.25)
            green_time = int(cycle_time * flow_ratio)
            
            # Apply minimum green time
            green_time = max(10, green_time)
            
            phase_timings[lane] = green_time
        
        return cycle_time, phase_timings
    
    def _apply_safety_constraints(self, phase_timings: Dict[str, int]) -> Dict[str, int]:
        """Apply safety constraints to phase timings"""
        constrained_timings = phase_timings.copy()
        
        # Apply minimum green time constraint
        min_green_time = self.config.get('min_green_time', 10)
        for lane, timing in constrained_timings.items():
            constrained_timings[lane] = max(min_green_time, timing)
        
        # Apply maximum green time constraint
        max_green_time = self.config.get('max_green_time', 60)
        for lane, timing in constrained_timings.items():
            constrained_timings[lane] = min(max_green_time, timing)
        
        return constrained_timings


class EmergencyVehicleManager:
    """Emergency vehicle priority and override system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Emergency vehicle tracking
        self.active_alerts = {}
        self.alert_history = deque(maxlen=1000)
        
        # Priority levels
        self.priority_levels = {
            'ambulance': 1,
            'fire_truck': 1,
            'police': 2,
            'emergency_services': 3,
            'public_transport': 4,
            'other': 5
        }
        
        # Override parameters
        self.override_duration = self.config.get('override_duration', 300)  # 5 minutes
        self.priority_extension = self.config.get('priority_extension', 20)  # seconds
        
        # Thread safety
        self.lock = threading.RLock()
    
    def register_emergency_vehicle(self, intersection_id: str, vehicle_type: str, 
                                 priority: int, estimated_arrival: datetime,
                                 current_location: Tuple[float, float], 
                                 destination: str) -> str:
        """Register emergency vehicle alert"""
        with self.lock:
            alert_id = str(uuid.uuid4())
            
            alert = EmergencyVehicleAlert(
                alert_id=alert_id,
                timestamp=datetime.now(),
                intersection_id=intersection_id,
                vehicle_type=vehicle_type,
                priority=priority,
                estimated_arrival=estimated_arrival,
                current_location=current_location,
                destination=destination
            )
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
            
            self.logger.info(f"Emergency vehicle alert registered: {vehicle_type} at {intersection_id}")
            
            return alert_id
    
    def get_active_alerts(self, intersection_id: Optional[str] = None) -> List[EmergencyVehicleAlert]:
        """Get active emergency vehicle alerts"""
        with self.lock:
            current_time = datetime.now()
            active_alerts = []
            
            for alert in self.active_alerts.values():
                # Check if alert is still active
                if (alert.is_active and 
                    (current_time - alert.timestamp).total_seconds() < self.override_duration):
                    
                    if intersection_id is None or alert.intersection_id == intersection_id:
                        active_alerts.append(alert)
            
            return active_alerts
    
    def handle_emergency_override(self, intersection_id: str, 
                                current_timings: Dict[str, int]) -> Dict[str, int]:
        """Handle emergency vehicle priority override"""
        with self.lock:
            active_alerts = self.get_active_alerts(intersection_id)
            
            if not active_alerts:
                return current_timings
            
            # Get highest priority alert
            highest_priority_alert = min(active_alerts, key=lambda x: x.priority)
            
            # Apply emergency override
            override_timings = current_timings.copy()
            
            # Determine main approach based on emergency vehicle direction
            main_approach = self._determine_main_approach(highest_priority_alert)
            
            # Extend green time for main approach
            for lane in main_approach:
                if lane in override_timings:
                    override_timings[lane] = min(90, override_timings[lane] + self.priority_extension)
            
            # Reduce green time for other approaches
            other_lanes = [lane for lane in override_timings.keys() if lane not in main_approach]
            for lane in other_lanes:
                override_timings[lane] = max(10, override_timings[lane] - 10)
            
            self.logger.info(f"Emergency override applied for {intersection_id}: {highest_priority_alert.vehicle_type}")
            
            return override_timings
    
    def _determine_main_approach(self, alert: EmergencyVehicleAlert) -> List[str]:
        """Determine main approach for emergency vehicle"""
        # Simplified approach determination based on vehicle type and location
        # In a real system, this would use GPS coordinates and routing information
        
        if alert.vehicle_type in ['ambulance', 'fire_truck']:
            # Emergency vehicles typically use main roads
            return ['north_lane', 'south_lane']
        elif alert.vehicle_type == 'police':
            # Police vehicles may use any approach
            return ['north_lane', 'south_lane', 'east_lane', 'west_lane']
        else:
            # Default to main approach
            return ['north_lane', 'south_lane']
    
    def resolve_alert(self, alert_id: str, handled_by: str):
        """Resolve emergency vehicle alert"""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.is_active = False
                alert.handled_by = handled_by
                alert.resolution_time = datetime.now()
                
                self.logger.info(f"Emergency alert {alert_id} resolved by {handled_by}")
    
    def get_emergency_metrics(self) -> Dict[str, Any]:
        """Get emergency vehicle management metrics"""
        with self.lock:
            current_time = datetime.now()
            
            # Count active alerts by type
            active_by_type = defaultdict(int)
            for alert in self.active_alerts.values():
                if alert.is_active:
                    active_by_type[alert.vehicle_type] += 1
            
            # Calculate average response time
            resolved_alerts = [alert for alert in self.alert_history 
                             if not alert.is_active and alert.resolution_time]
            
            avg_response_time = 0.0
            if resolved_alerts:
                response_times = [(alert.resolution_time - alert.timestamp).total_seconds() 
                                for alert in resolved_alerts]
                avg_response_time = np.mean(response_times)
            
            return {
                'total_alerts': len(self.alert_history),
                'active_alerts': len([a for a in self.active_alerts.values() if a.is_active]),
                'active_by_type': dict(active_by_type),
                'avg_response_time': avg_response_time,
                'alerts_last_hour': len([a for a in self.alert_history 
                                       if (current_time - a.timestamp).total_seconds() < 3600])
            }


class SafetyConstraintManager:
    """Manages safety constraints and violations"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Safety constraints
        self.constraints = {}
        self.violation_history = deque(maxlen=1000)
        
        # Safety thresholds
        self.safety_thresholds = {
            'min_green_time': self.config.get('min_green_time', 10),
            'max_green_time': self.config.get('max_green_time', 90),
            'min_cycle_time': self.config.get('min_cycle_time', 40),
            'max_cycle_time': self.config.get('max_cycle_time', 120),
            'pedestrian_crossing_time': self.config.get('pedestrian_crossing_time', 20),
            'max_wait_time': self.config.get('max_wait_time', 120),
            'congestion_threshold': self.config.get('congestion_threshold', 0.8)
        }
        
        # Initialize default constraints
        self._initialize_default_constraints()
        
        # Thread safety
        self.lock = threading.RLock()
    
    def _initialize_default_constraints(self):
        """Initialize default safety constraints"""
        # Green time constraints
        self.constraints['min_green_time'] = SafetyConstraint(
            constraint_id='min_green_time',
            name='Minimum Green Time',
            description='Minimum green time for any phase',
            min_value=self.safety_thresholds['min_green_time'],
            max_value=float('inf'),
            current_value=self.safety_thresholds['min_green_time'],
            violation_threshold=self.safety_thresholds['min_green_time'],
            priority=1
        )
        
        self.constraints['max_green_time'] = SafetyConstraint(
            constraint_id='max_green_time',
            name='Maximum Green Time',
            description='Maximum green time for any phase',
            min_value=0,
            max_value=self.safety_thresholds['max_green_time'],
            current_value=self.safety_thresholds['max_green_time'],
            violation_threshold=self.safety_thresholds['max_green_time'],
            priority=1
        )
        
        # Cycle time constraints
        self.constraints['min_cycle_time'] = SafetyConstraint(
            constraint_id='min_cycle_time',
            name='Minimum Cycle Time',
            description='Minimum total cycle time',
            min_value=self.safety_thresholds['min_cycle_time'],
            max_value=float('inf'),
            current_value=self.safety_thresholds['min_cycle_time'],
            violation_threshold=self.safety_thresholds['min_cycle_time'],
            priority=2
        )
        
        self.constraints['max_cycle_time'] = SafetyConstraint(
            constraint_id='max_cycle_time',
            name='Maximum Cycle Time',
            description='Maximum total cycle time',
            min_value=0,
            max_value=self.safety_thresholds['max_cycle_time'],
            current_value=self.safety_thresholds['max_cycle_time'],
            violation_threshold=self.safety_thresholds['max_cycle_time'],
            priority=2
        )
        
        # Pedestrian safety constraints
        self.constraints['pedestrian_crossing'] = SafetyConstraint(
            constraint_id='pedestrian_crossing',
            name='Pedestrian Crossing Time',
            description='Minimum time for pedestrian crossing',
            min_value=self.safety_thresholds['pedestrian_crossing_time'],
            max_value=float('inf'),
            current_value=self.safety_thresholds['pedestrian_crossing_time'],
            violation_threshold=self.safety_thresholds['pedestrian_crossing_time'],
            priority=1
        )
    
    def check_constraints(self, intersection_id: str, timings: Dict[str, int], 
                         traffic_data: Dict[str, Any]) -> List[SafetyViolationReport]:
        """Check safety constraints and return violations"""
        violations = []
        
        with self.lock:
            # Check green time constraints
            for lane, timing in timings.items():
                # Update constraint values
                self.constraints['min_green_time'].current_value = timing
                self.constraints['max_green_time'].current_value = timing
                
                # Check violations
                if self.constraints['min_green_time'].is_violated():
                    violation = SafetyViolationReport(
                        violation_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        intersection_id=intersection_id,
                        violation_type=SafetyViolation.MIN_GREEN_TIME_VIOLATION,
                        severity=self.constraints['min_green_time'].get_violation_severity(),
                        description=f"Green time {timing}s is below minimum {self.safety_thresholds['min_green_time']}s for {lane}",
                        constraint_id='min_green_time',
                        current_value=timing,
                        expected_value=self.safety_thresholds['min_green_time'],
                        corrective_action=f"Increase green time for {lane} to at least {self.safety_thresholds['min_green_time']}s"
                    )
                    violations.append(violation)
                
                if self.constraints['max_green_time'].is_violated():
                    violation = SafetyViolationReport(
                        violation_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        intersection_id=intersection_id,
                        violation_type=SafetyViolation.MAX_GREEN_TIME_VIOLATION,
                        severity=self.constraints['max_green_time'].get_violation_severity(),
                        description=f"Green time {timing}s exceeds maximum {self.safety_thresholds['max_green_time']}s for {lane}",
                        constraint_id='max_green_time',
                        current_value=timing,
                        expected_value=self.safety_thresholds['max_green_time'],
                        corrective_action=f"Reduce green time for {lane} to at most {self.safety_thresholds['max_green_time']}s"
                    )
                    violations.append(violation)
            
            # Check cycle time constraints
            total_cycle_time = sum(timings.values())
            self.constraints['min_cycle_time'].current_value = total_cycle_time
            self.constraints['max_cycle_time'].current_value = total_cycle_time
            
            if self.constraints['min_cycle_time'].is_violated():
                violation = SafetyViolationReport(
                    violation_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    intersection_id=intersection_id,
                    violation_type=SafetyViolation.MIN_GREEN_TIME_VIOLATION,
                    severity=self.constraints['min_cycle_time'].get_violation_severity(),
                    description=f"Total cycle time {total_cycle_time}s is below minimum {self.safety_thresholds['min_cycle_time']}s",
                    constraint_id='min_cycle_time',
                    current_value=total_cycle_time,
                    expected_value=self.safety_thresholds['min_cycle_time'],
                    corrective_action=f"Increase total cycle time to at least {self.safety_thresholds['min_cycle_time']}s"
                )
                violations.append(violation)
            
            if self.constraints['max_cycle_time'].is_violated():
                violation = SafetyViolationReport(
                    violation_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    intersection_id=intersection_id,
                    violation_type=SafetyViolation.MAX_GREEN_TIME_VIOLATION,
                    severity=self.constraints['max_cycle_time'].get_violation_severity(),
                    description=f"Total cycle time {total_cycle_time}s exceeds maximum {self.safety_thresholds['max_cycle_time']}s",
                    constraint_id='max_cycle_time',
                    current_value=total_cycle_time,
                    expected_value=self.safety_thresholds['max_cycle_time'],
                    corrective_action=f"Reduce total cycle time to at most {self.safety_thresholds['max_cycle_time']}s"
                )
                violations.append(violation)
            
            # Check additional safety conditions
            violations.extend(self._check_additional_safety_conditions(intersection_id, timings, traffic_data))
            
            # Store violations in history
            for violation in violations:
                self.violation_history.append(violation)
        
        return violations
    
    def _check_additional_safety_conditions(self, intersection_id: str, timings: Dict[str, int], 
                                          traffic_data: Dict[str, Any]) -> List[SafetyViolationReport]:
        """Check additional safety conditions"""
        violations = []
        
        # Check for high congestion
        lane_counts = traffic_data.get('lane_counts', {})
        total_vehicles = sum(lane_counts.values()) if lane_counts else 0
        
        if total_vehicles > 100:  # High congestion threshold
            violation = SafetyViolationReport(
                violation_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                intersection_id=intersection_id,
                violation_type=SafetyViolation.HIGH_CONGESTION,
                severity=min(1.0, total_vehicles / 200.0),
                description=f"High congestion detected: {total_vehicles} vehicles",
                constraint_id='congestion_threshold',
                current_value=total_vehicles,
                expected_value=100,
                corrective_action="Implement congestion management strategies"
            )
            violations.append(violation)
        
        # Check for stale data
        timestamp_str = traffic_data.get('timestamp')
        if timestamp_str:
            try:
                data_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                age_seconds = (datetime.now() - data_time).total_seconds()
                
                if age_seconds > 60:  # Data older than 1 minute
                    violation = SafetyViolationReport(
                        violation_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        intersection_id=intersection_id,
                        violation_type=SafetyViolation.DATA_STALE,
                        severity=min(1.0, age_seconds / 300.0),  # Max severity at 5 minutes
                        description=f"Traffic data is stale: {age_seconds:.1f} seconds old",
                        constraint_id='data_freshness',
                        current_value=age_seconds,
                        expected_value=60,
                        corrective_action="Update traffic data or use fallback optimization"
                    )
                    violations.append(violation)
            except:
                pass
        
        # Check for weather hazards
        weather_condition = traffic_data.get('weather_condition', 'clear')
        if weather_condition in ['stormy', 'foggy']:
            violation = SafetyViolationReport(
                violation_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                intersection_id=intersection_id,
                violation_type=SafetyViolation.WEATHER_HAZARD,
                severity=0.7,
                description=f"Weather hazard detected: {weather_condition}",
                constraint_id='weather_safety',
                current_value=1.0,
                expected_value=0.0,
                corrective_action="Implement weather-appropriate safety measures"
            )
            violations.append(violation)
        
        return violations
    
    def apply_corrective_actions(self, intersection_id: str, timings: Dict[str, int], 
                               violations: List[SafetyViolationReport]) -> Dict[str, int]:
        """Apply corrective actions to fix safety violations"""
        corrected_timings = timings.copy()
        
        for violation in violations:
            if violation.is_resolved:
                continue
            
            if violation.violation_type == SafetyViolation.MIN_GREEN_TIME_VIOLATION:
                # Increase green time to minimum
                for lane in corrected_timings:
                    if corrected_timings[lane] < self.safety_thresholds['min_green_time']:
                        corrected_timings[lane] = self.safety_thresholds['min_green_time']
            
            elif violation.violation_type == SafetyViolation.MAX_GREEN_TIME_VIOLATION:
                # Reduce green time to maximum
                for lane in corrected_timings:
                    if corrected_timings[lane] > self.safety_thresholds['max_green_time']:
                        corrected_timings[lane] = self.safety_thresholds['max_green_time']
            
            elif violation.violation_type == SafetyViolation.HIGH_CONGESTION:
                # Implement congestion management
                total_cycle = sum(corrected_timings.values())
                if total_cycle < self.safety_thresholds['max_cycle_time']:
                    # Increase cycle time to improve throughput
                    scale_factor = min(1.2, self.safety_thresholds['max_cycle_time'] / total_cycle)
                    for lane in corrected_timings:
                        corrected_timings[lane] = int(corrected_timings[lane] * scale_factor)
            
            # Mark violation as resolved
            violation.is_resolved = True
            violation.resolution_time = datetime.now()
        
        return corrected_timings
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get safety metrics and statistics"""
        with self.lock:
            current_time = datetime.now()
            
            # Count violations by type
            violation_counts = defaultdict(int)
            for violation in self.violation_history:
                violation_counts[violation.violation_type.value] += 1
            
            # Count recent violations (last hour)
            recent_violations = [v for v in self.violation_history 
                               if (current_time - v.timestamp).total_seconds() < 3600]
            
            # Calculate violation severity
            avg_severity = 0.0
            if self.violation_history:
                severities = [v.severity for v in self.violation_history]
                avg_severity = np.mean(severities)
            
            return {
                'total_violations': len(self.violation_history),
                'recent_violations': len(recent_violations),
                'violation_counts': dict(violation_counts),
                'avg_severity': avg_severity,
                'constraints_active': len([c for c in self.constraints.values() if c.is_active]),
                'safety_level': self._determine_safety_level()
            }
    
    def _determine_safety_level(self) -> SafetyLevel:
        """Determine current safety level based on violations"""
        if not self.violation_history:
            return SafetyLevel.NORMAL
        
        # Check recent violations
        current_time = datetime.now()
        recent_violations = [v for v in self.violation_history 
                           if (current_time - v.timestamp).total_seconds() < 300]  # Last 5 minutes
        
        if not recent_violations:
            return SafetyLevel.NORMAL
        
        # Count high-severity violations
        high_severity_violations = [v for v in recent_violations if v.severity > 0.7]
        
        if len(high_severity_violations) > 3:
            return SafetyLevel.EMERGENCY
        elif len(high_severity_violations) > 1:
            return SafetyLevel.CRITICAL
        elif len(recent_violations) > 5:
            return SafetyLevel.WARNING
        else:
            return SafetyLevel.CAUTION


class SafetyManager:
    """
    Main Safety Manager with comprehensive fallback systems
    
    Features:
    - Webster's formula as baseline fallback mechanism
    - Safety constraints and violation monitoring
    - Emergency vehicle priority override system
    - Fail-safe mechanisms for system failures
    - Real-time safety metrics monitoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.webster_fallback = WebsterFormulaFallback(config.get('webster', {}))
        self.emergency_manager = EmergencyVehicleManager(config.get('emergency', {}))
        self.constraint_manager = SafetyConstraintManager(config.get('constraints', {}))
        
        # Safety state
        self.safety_level = SafetyLevel.NORMAL
        self.last_safety_check = None
        self.safety_check_interval = config.get('safety_check_interval', 5.0)  # seconds
        
        # Thread safety
        self.lock = threading.RLock()
        
        self.logger.info("Safety Manager initialized")
    
    def configure_intersection(self, intersection_id: str, config: Dict[str, Any]):
        """Configure safety parameters for specific intersection"""
        self.webster_fallback.configure_intersection(intersection_id, config)
        self.logger.info(f"Configured safety parameters for {intersection_id}")
    
    def check_safety_and_optimize(self, intersection_id: str, traffic_data: Dict[str, Any], 
                                 current_timings: Dict[str, int]) -> Dict[str, Any]:
        """Check safety constraints and optimize intersection"""
        try:
            # Check for emergency vehicles
            emergency_timings = self.emergency_manager.handle_emergency_override(
                intersection_id, current_timings
            )
            
            # Check safety constraints
            violations = self.constraint_manager.check_constraints(
                intersection_id, emergency_timings, traffic_data
            )
            
            # Apply corrective actions if violations exist
            if violations:
                corrected_timings = self.constraint_manager.apply_corrective_actions(
                    intersection_id, emergency_timings, violations
                )
                algorithm_used = 'safety_corrected'
            else:
                corrected_timings = emergency_timings
                algorithm_used = 'emergency_override' if emergency_timings != current_timings else 'ml_optimization'
            
            # Determine if fallback is needed
            if self._should_use_fallback(intersection_id, traffic_data, violations):
                fallback_timings = self.webster_fallback.optimize_intersection(
                    intersection_id, traffic_data
                )
                algorithm_used = 'webster_fallback'
            else:
                fallback_timings = corrected_timings
            
            # Update safety level
            self._update_safety_level(violations)
            
            return {
                'success': True,
                'optimized_timings': fallback_timings,
                'algorithm_used': algorithm_used,
                'confidence_score': 0.9 if algorithm_used == 'webster_fallback' else 0.7,
                'safety_violations': [v.description for v in violations],
                'warnings': [f"Safety level: {self.safety_level.value}"],
                'emergency_active': len(self.emergency_manager.get_active_alerts(intersection_id)) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Safety check failed for {intersection_id}: {e}")
            return {
                'success': False,
                'optimized_timings': current_timings,
                'algorithm_used': 'safety_error',
                'confidence_score': 0.0,
                'safety_violations': ['safety_check_error'],
                'warnings': [str(e)]
            }
    
    def _should_use_fallback(self, intersection_id: str, traffic_data: Dict[str, Any], 
                           violations: List[SafetyViolationReport]) -> bool:
        """Determine if fallback optimization should be used"""
        # Use fallback if there are critical safety violations
        critical_violations = [v for v in violations if v.severity > 0.7]
        if critical_violations:
            return True
        
        # Use fallback if data is stale
        timestamp_str = traffic_data.get('timestamp')
        if timestamp_str:
            try:
                data_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                age_seconds = (datetime.now() - data_time).total_seconds()
                if age_seconds > 120:  # Data older than 2 minutes
                    return True
            except:
                return True
        
        # Use fallback if system is in critical safety level
        if self.safety_level in [SafetyLevel.CRITICAL, SafetyLevel.EMERGENCY]:
            return True
        
        return False
    
    def _update_safety_level(self, violations: List[SafetyViolationReport]):
        """Update current safety level based on violations"""
        if not violations:
            self.safety_level = SafetyLevel.NORMAL
            return
        
        # Count high-severity violations
        high_severity_count = len([v for v in violations if v.severity > 0.7])
        
        if high_severity_count > 2:
            self.safety_level = SafetyLevel.EMERGENCY
        elif high_severity_count > 0:
            self.safety_level = SafetyLevel.CRITICAL
        elif len(violations) > 3:
            self.safety_level = SafetyLevel.WARNING
        else:
            self.safety_level = SafetyLevel.CAUTION
    
    def register_emergency_vehicle(self, intersection_id: str, vehicle_type: str, 
                                 priority: int, estimated_arrival: datetime,
                                 current_location: Tuple[float, float], 
                                 destination: str) -> str:
        """Register emergency vehicle alert"""
        return self.emergency_manager.register_emergency_vehicle(
            intersection_id, vehicle_type, priority, estimated_arrival,
            current_location, destination
        )
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status and metrics"""
        with self.lock:
            return {
                'safety_level': self.safety_level.value,
                'last_safety_check': self.last_safety_check.isoformat() if self.last_safety_check else None,
                'constraint_metrics': self.constraint_manager.get_safety_metrics(),
                'emergency_metrics': self.emergency_manager.get_emergency_metrics(),
                'webster_config': {
                    'min_cycle_time': self.webster_fallback.min_cycle_time,
                    'max_cycle_time': self.webster_fallback.max_cycle_time,
                    'saturation_flow_rate': self.webster_fallback.saturation_flow_rate
                }
            }
    
    def get_safety_recommendations(self) -> List[str]:
        """Get safety recommendations based on current status"""
        recommendations = []
        
        # Check safety level
        if self.safety_level == SafetyLevel.EMERGENCY:
            recommendations.append("CRITICAL: Implement emergency protocols immediately")
        elif self.safety_level == SafetyLevel.CRITICAL:
            recommendations.append("HIGH: Review and address safety violations")
        elif self.safety_level == SafetyLevel.WARNING:
            recommendations.append("MEDIUM: Monitor safety conditions closely")
        
        # Check emergency vehicles
        emergency_metrics = self.emergency_manager.get_emergency_metrics()
        if emergency_metrics['active_alerts'] > 0:
            recommendations.append(f"Emergency vehicles active: {emergency_metrics['active_alerts']} alerts")
        
        # Check constraint violations
        constraint_metrics = self.constraint_manager.get_safety_metrics()
        if constraint_metrics['recent_violations'] > 5:
            recommendations.append(f"High violation rate: {constraint_metrics['recent_violations']} violations in last hour")
        
        return recommendations
