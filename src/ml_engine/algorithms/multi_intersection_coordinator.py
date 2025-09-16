"""
Multi-Intersection Coordination System
Handles communication, conflict resolution, and network-level optimization
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Set
import numpy as np
import networkx as nx
from queue import Queue, Empty
import redis
import pickle

from config.ml_config import get_config


class CoordinationMessageType(Enum):
    """Types of coordination messages"""
    STATE_UPDATE = "state_update"
    ACTION_PROPOSAL = "action_proposal"
    CONFLICT_RESOLUTION = "conflict_resolution"
    GREEN_WAVE_REQUEST = "green_wave_request"
    EMERGENCY_OVERRIDE = "emergency_override"
    PERFORMANCE_SHARE = "performance_share"
    HEARTBEAT = "heartbeat"


@dataclass
class CoordinationMessage:
    """Coordination message between intersections"""
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: CoordinationMessageType
    timestamp: datetime
    data: Dict[str, Any]
    priority: int = 0  # Higher number = higher priority
    ttl: float = 30.0  # Time to live in seconds
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        return (datetime.now() - self.timestamp).total_seconds() > self.ttl
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'message_type': self.message_type.value,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'priority': self.priority,
            'ttl': self.ttl
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CoordinationMessage':
        """Create from dictionary"""
        return cls(
            message_id=data['message_id'],
            sender_id=data['sender_id'],
            receiver_id=data['receiver_id'],
            message_type=CoordinationMessageType(data['message_type']),
            timestamp=datetime.fromisoformat(data['timestamp']),
            data=data['data'],
            priority=data.get('priority', 0),
            ttl=data.get('ttl', 30.0)
        )


@dataclass
class IntersectionState:
    """State of an intersection for coordination"""
    intersection_id: str
    current_phase: int
    phase_duration: float
    cycle_progress: float
    congestion_level: float
    traffic_flow: Dict[str, float]  # Flow rates per direction
    queue_lengths: Dict[str, float]
    waiting_times: Dict[str, float]
    emergency_vehicles: bool
    last_update: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'intersection_id': self.intersection_id,
            'current_phase': self.current_phase,
            'phase_duration': self.phase_duration,
            'cycle_progress': self.cycle_progress,
            'congestion_level': self.congestion_level,
            'traffic_flow': self.traffic_flow,
            'queue_lengths': self.queue_lengths,
            'waiting_times': self.waiting_times,
            'emergency_vehicles': self.emergency_vehicles,
            'last_update': self.last_update.isoformat()
        }


class ConflictResolver:
    """Handles conflict resolution between competing optimization goals"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conflict_history = deque(maxlen=1000)
    
    def resolve_conflicts(self, intersection_actions: Dict[str, Any], 
                         intersection_states: Dict[str, IntersectionState]) -> Dict[str, Any]:
        """
        Resolve conflicts between intersection actions
        
        Args:
            intersection_actions: {intersection_id: action_data}
            intersection_states: {intersection_id: IntersectionState}
            
        Returns:
            Resolved actions with conflicts resolved
        """
        resolved_actions = intersection_actions.copy()
        
        # Check for green wave conflicts
        green_wave_conflicts = self._detect_green_wave_conflicts(intersection_actions, intersection_states)
        if green_wave_conflicts:
            resolved_actions = self._resolve_green_wave_conflicts(resolved_actions, green_wave_conflicts)
        
        # Check for timing conflicts
        timing_conflicts = self._detect_timing_conflicts(intersection_actions, intersection_states)
        if timing_conflicts:
            resolved_actions = self._resolve_timing_conflicts(resolved_actions, timing_conflicts)
        
        # Check for emergency vehicle conflicts
        emergency_conflicts = self._detect_emergency_conflicts(intersection_actions, intersection_states)
        if emergency_conflicts:
            resolved_actions = self._resolve_emergency_conflicts(resolved_actions, emergency_conflicts)
        
        # Log conflicts
        total_conflicts = len(green_wave_conflicts) + len(timing_conflicts) + len(emergency_conflicts)
        if total_conflicts > 0:
            self.conflict_history.append({
                'timestamp': datetime.now(),
                'conflicts': total_conflicts,
                'green_wave': len(green_wave_conflicts),
                'timing': len(timing_conflicts),
                'emergency': len(emergency_conflicts)
            })
            self.logger.info(f"Resolved {total_conflicts} conflicts")
        
        return resolved_actions
    
    def _detect_green_wave_conflicts(self, actions: Dict[str, Any], 
                                   states: Dict[str, IntersectionState]) -> List[Dict[str, Any]]:
        """Detect green wave coordination conflicts"""
        conflicts = []
        
        # Find intersections trying to coordinate green waves
        green_wave_actions = {
            iid: action for iid, action in actions.items()
            if action.get('action_type') == 3  # Green wave coordination
        }
        
        if len(green_wave_actions) < 2:
            return conflicts
        
        # Check for conflicting phase timings
        for iid1, action1 in green_wave_actions.items():
            for iid2, action2 in green_wave_actions.items():
                if iid1 >= iid2:  # Avoid duplicate checks
                    continue
                
                # Check if phases are compatible
                if not self._are_phases_compatible(action1, action2, states.get(iid1), states.get(iid2)):
                    conflicts.append({
                        'type': 'green_wave',
                        'intersections': [iid1, iid2],
                        'conflict': 'incompatible_phases'
                    })
        
        return conflicts
    
    def _detect_timing_conflicts(self, actions: Dict[str, Any], 
                               states: Dict[str, IntersectionState]) -> List[Dict[str, Any]]:
        """Detect timing conflicts between intersections"""
        conflicts = []
        
        # Check for overlapping cycle time changes
        cycle_changes = {
            iid: action.get('cycle_time_adjustment', 0)
            for iid, action in actions.items()
            if action.get('cycle_time_adjustment', 0) != 0
        }
        
        if len(cycle_changes) < 2:
            return conflicts
        
        # Check for conflicting cycle adjustments
        for iid1, change1 in cycle_changes.items():
            for iid2, change2 in cycle_changes.items():
                if iid1 >= iid2:
                    continue
                
                # Check if changes are in opposite directions
                if (change1 > 0 and change2 < 0) or (change1 < 0 and change2 > 0):
                    conflicts.append({
                        'type': 'timing',
                        'intersections': [iid1, iid2],
                        'conflict': 'opposing_cycle_changes'
                    })
        
        return conflicts
    
    def _detect_emergency_conflicts(self, actions: Dict[str, Any], 
                                  states: Dict[str, IntersectionState]) -> List[Dict[str, Any]]:
        """Detect emergency vehicle conflicts"""
        conflicts = []
        
        # Find intersections with emergency vehicles
        emergency_intersections = {
            iid: state for iid, state in states.items()
            if state.emergency_vehicles
        }
        
        if len(emergency_intersections) < 2:
            return conflicts
        
        # Check if multiple intersections are trying to give emergency priority
        emergency_actions = {
            iid: action for iid, action in actions.items()
            if action.get('action_type') == 4  # Emergency priority
        }
        
        if len(emergency_actions) > 1:
            conflicts.append({
                'type': 'emergency',
                'intersections': list(emergency_actions.keys()),
                'conflict': 'multiple_emergency_priorities'
            })
        
        return conflicts
    
    def _resolve_green_wave_conflicts(self, actions: Dict[str, Any], 
                                    conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve green wave conflicts"""
        resolved_actions = actions.copy()
        
        for conflict in conflicts:
            intersections = conflict['intersections']
            
            # Prioritize intersection with higher congestion
            congestion_levels = {
                iid: actions[iid].get('congestion_level', 0)
                for iid in intersections
            }
            
            # Keep action for intersection with highest congestion
            priority_intersection = max(congestion_levels, key=congestion_levels.get)
            
            for iid in intersections:
                if iid != priority_intersection:
                    # Modify action to be compatible
                    resolved_actions[iid]['action_type'] = 0  # Maintain current timing
                    resolved_actions[iid]['coordination_signal'] = 'maintain'
        
        return resolved_actions
    
    def _resolve_timing_conflicts(self, actions: Dict[str, Any], 
                                conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve timing conflicts"""
        resolved_actions = actions.copy()
        
        for conflict in conflicts:
            intersections = conflict['intersections']
            
            # Average the cycle time adjustments
            adjustments = [actions[iid].get('cycle_time_adjustment', 0) for iid in intersections]
            avg_adjustment = int(np.mean(adjustments))
            
            for iid in intersections:
                resolved_actions[iid]['cycle_time_adjustment'] = avg_adjustment
        
        return resolved_actions
    
    def _resolve_emergency_conflicts(self, actions: Dict[str, Any], 
                                   conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve emergency vehicle conflicts"""
        resolved_actions = actions.copy()
        
        for conflict in conflicts:
            intersections = conflict['intersections']
            
            # Prioritize based on emergency vehicle proximity and congestion
            priorities = {}
            for iid in intersections:
                action = actions[iid]
                priority = action.get('congestion_level', 0) * 0.7 + action.get('emergency_proximity', 0) * 0.3
                priorities[iid] = priority
            
            # Keep emergency priority for highest priority intersection
            priority_intersection = max(priorities, key=priorities.get)
            
            for iid in intersections:
                if iid != priority_intersection:
                    resolved_actions[iid]['action_type'] = 0  # Maintain current timing
                    resolved_actions[iid]['priority_boost'] = False
        
        return resolved_actions
    
    def _are_phases_compatible(self, action1: Dict[str, Any], action2: Dict[str, Any],
                             state1: Optional[IntersectionState], state2: Optional[IntersectionState]) -> bool:
        """Check if two actions are compatible for green wave coordination"""
        if not state1 or not state2:
            return True
        
        # Check if phases are in sync
        phase_diff = abs(state1.current_phase - state2.current_phase)
        if phase_diff > 1:  # Phases should be within 1 of each other
            return False
        
        # Check if cycle progress is compatible
        progress_diff = abs(state1.cycle_progress - state2.cycle_progress)
        if progress_diff > 0.3:  # Cycle progress should be within 30%
            return False
        
        return True


class NetworkOptimizer:
    """Handles network-level optimization considering traffic flow propagation"""
    
    def __init__(self, intersection_graph: nx.Graph):
        self.graph = intersection_graph
        self.logger = logging.getLogger(__name__)
        self.flow_propagation_model = self._initialize_flow_propagation_model()
    
    def _initialize_flow_propagation_model(self) -> Dict[str, Any]:
        """Initialize traffic flow propagation model"""
        return {
            'propagation_delay': 2.0,  # seconds per intersection
            'flow_decay_factor': 0.9,  # Flow reduction factor
            'congestion_threshold': 0.7,  # Congestion threshold
            'green_wave_speed': 50.0  # km/h for green wave coordination
        }
    
    def optimize_network_flow(self, intersection_states: Dict[str, IntersectionState],
                            proposed_actions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize network-level traffic flow
        
        Args:
            intersection_states: Current states of all intersections
            proposed_actions: Proposed actions for each intersection
            
        Returns:
            Network-optimized actions
        """
        # Calculate network-wide metrics
        network_metrics = self._calculate_network_metrics(intersection_states)
        
        # Identify critical paths
        critical_paths = self._identify_critical_paths(intersection_states)
        
        # Optimize green waves
        optimized_actions = self._optimize_green_waves(proposed_actions, intersection_states, critical_paths)
        
        # Balance network load
        optimized_actions = self._balance_network_load(optimized_actions, intersection_states, network_metrics)
        
        # Apply flow propagation constraints
        optimized_actions = self._apply_flow_propagation_constraints(optimized_actions, intersection_states)
        
        return optimized_actions
    
    def _calculate_network_metrics(self, states: Dict[str, IntersectionState]) -> Dict[str, Any]:
        """Calculate network-wide performance metrics"""
        total_vehicles = sum(state.congestion_level for state in states.values())
        avg_congestion = np.mean([state.congestion_level for state in states.values()])
        max_congestion = max(state.congestion_level for state in states.values())
        
        # Calculate network efficiency
        total_flow = sum(sum(state.traffic_flow.values()) for state in states.values())
        total_waiting = sum(sum(state.waiting_times.values()) for state in states.values())
        efficiency = total_flow / (total_waiting + 1e-6)  # Avoid division by zero
        
        return {
            'total_vehicles': total_vehicles,
            'avg_congestion': avg_congestion,
            'max_congestion': max_congestion,
            'total_flow': total_flow,
            'total_waiting': total_waiting,
            'efficiency': efficiency
        }
    
    def _identify_critical_paths(self, states: Dict[str, IntersectionState]) -> List[List[str]]:
        """Identify critical traffic paths through the network"""
        critical_paths = []
        
        # Find paths with high congestion
        for node in self.graph.nodes():
            if node in states and states[node].congestion_level > 0.7:
                # Find shortest paths through this node
                for other_node in self.graph.nodes():
                    if other_node != node and other_node in states:
                        try:
                            path = nx.shortest_path(self.graph, node, other_node)
                            if len(path) > 1:  # Valid path
                                path_congestion = np.mean([states[n].congestion_level for n in path if n in states])
                                if path_congestion > 0.6:
                                    critical_paths.append(path)
                        except nx.NetworkXNoPath:
                            continue
        
        return critical_paths
    
    def _optimize_green_waves(self, actions: Dict[str, Any], 
                            states: Dict[str, IntersectionState],
                            critical_paths: List[List[str]]) -> Dict[str, Any]:
        """Optimize green wave coordination along critical paths"""
        optimized_actions = actions.copy()
        
        for path in critical_paths:
            if len(path) < 2:
                continue
            
            # Calculate optimal timing for green wave
            green_wave_timing = self._calculate_green_wave_timing(path, states)
            
            # Apply green wave timing to intersections in path
            for i, intersection_id in enumerate(path):
                if intersection_id in optimized_actions:
                    action = optimized_actions[intersection_id]
                    
                    # Set green wave coordination
                    action['action_type'] = 3  # Green wave coordination
                    action['coordination_signal'] = 'green_wave'
                    action['green_wave_timing'] = green_wave_timing[i]
                    action['path_position'] = i
                    action['path_length'] = len(path)
        
        return optimized_actions
    
    def _calculate_green_wave_timing(self, path: List[str], 
                                   states: Dict[str, IntersectionState]) -> List[float]:
        """Calculate optimal timing for green wave along a path"""
        if len(path) < 2:
            return []
        
        # Calculate distances between intersections
        distances = []
        for i in range(len(path) - 1):
            try:
                distance = self.graph[path[i]][path[i+1]]['distance']
                distances.append(distance)
            except KeyError:
                distances.append(100.0)  # Default distance
        
        # Calculate timing based on green wave speed
        green_wave_speed = self.flow_propagation_model['green_wave_speed']  # km/h
        timing = [0.0]  # Start with first intersection
        
        for distance in distances:
            # Convert distance to time (distance in km, speed in km/h)
            time_delay = (distance / green_wave_speed) * 3600  # Convert to seconds
            timing.append(timing[-1] + time_delay)
        
        return timing
    
    def _balance_network_load(self, actions: Dict[str, Any], 
                            states: Dict[str, IntersectionState],
                            network_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Balance load across the network"""
        optimized_actions = actions.copy()
        
        # Find overloaded and underloaded intersections
        overloaded = [
            iid for iid, state in states.items()
            if state.congestion_level > network_metrics['avg_congestion'] * 1.2
        ]
        
        underloaded = [
            iid for iid, state in states.items()
            if state.congestion_level < network_metrics['avg_congestion'] * 0.8
        ]
        
        # Redistribute load
        for overloaded_id in overloaded:
            if overloaded_id in optimized_actions:
                action = optimized_actions[overloaded_id]
                # Increase throughput
                action['action_type'] = 2  # Increase throughput
                action['load_balancing'] = True
        
        for underloaded_id in underloaded:
            if underloaded_id in optimized_actions:
                action = optimized_actions[underloaded_id]
                # Maintain current timing to avoid over-optimization
                action['action_type'] = 0  # Maintain
                action['load_balancing'] = True
        
        return optimized_actions
    
    def _apply_flow_propagation_constraints(self, actions: Dict[str, Any],
                                         states: Dict[str, IntersectionState]) -> Dict[str, Any]:
        """Apply flow propagation constraints to actions"""
        optimized_actions = actions.copy()
        
        # Consider upstream flow impact
        for intersection_id, action in optimized_actions.items():
            if intersection_id not in states:
                continue
            
            state = states[intersection_id]
            
            # Check upstream intersections
            upstream_intersections = list(self.graph.predecessors(intersection_id))
            upstream_flow_impact = 0.0
            
            for upstream_id in upstream_intersections:
                if upstream_id in states:
                    upstream_state = states[upstream_id]
                    upstream_flow_impact += upstream_state.congestion_level * 0.3
            
            # Adjust action based on upstream flow
            if upstream_flow_impact > 0.5:  # High upstream flow
                action['upstream_flow_adjustment'] = min(10, int(upstream_flow_impact * 20))
                if action['action_type'] == 0:  # If maintaining, consider increasing throughput
                    action['action_type'] = 2
        
        return optimized_actions


class MultiIntersectionCoordinator:
    """
    Main coordinator for multi-intersection traffic optimization
    """
    
    def __init__(self, intersection_id: str, config: Optional[Dict] = None):
        self.intersection_id = intersection_id
        self.config = config or get_config()
        self.logger = logging.getLogger(f"coordinator_{intersection_id}")
        
        # Initialize components
        self.conflict_resolver = ConflictResolver()
        self.intersection_graph = self._build_intersection_graph()
        self.network_optimizer = NetworkOptimizer(self.intersection_graph)
        
        # Communication
        self.message_queue = Queue(maxsize=1000)
        self.received_messages = defaultdict(list)
        self.intersection_states = {}
        
        # Coordination state
        self.is_coordinating = False
        self.coordination_lock = threading.Lock()
        self.last_coordination_time = None
        
        # Performance tracking
        self.coordination_metrics = {
            'messages_sent': 0,
            'messages_received': 0,
            'conflicts_resolved': 0,
            'coordination_cycles': 0
        }
        
        # Start coordination thread
        self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordination_thread.start()
        
        self.logger.info(f"Multi-intersection coordinator initialized for {intersection_id}")
    
    def _build_intersection_graph(self) -> nx.Graph:
        """Build network graph of intersections"""
        graph = nx.DiGraph()
        
        # Add intersections from config
        intersections = self.config.get('intersections', [])
        for intersection in intersections:
            graph.add_node(intersection['id'], **intersection)
        
        # Add connections between adjacent intersections
        for intersection in intersections:
            intersection_id = intersection['id']
            adjacent = intersection.get('adjacent_intersections', [])
            
            for adj_id in adjacent:
                if adj_id in [i['id'] for i in intersections]:
                    # Add edge with distance (default 100m)
                    graph.add_edge(intersection_id, adj_id, distance=0.1)
        
        return graph
    
    def send_message(self, receiver_id: str, message_type: CoordinationMessageType,
                    data: Dict[str, Any], priority: int = 0) -> str:
        """Send coordination message to another intersection"""
        message = CoordinationMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.intersection_id,
            receiver_id=receiver_id,
            message_type=message_type,
            timestamp=datetime.now(),
            data=data,
            priority=priority
        )
        
        # In a real implementation, this would send over network
        # For now, we'll simulate by adding to our own queue
        self.message_queue.put(message)
        
        self.coordination_metrics['messages_sent'] += 1
        self.logger.debug(f"Sent {message_type.value} message to {receiver_id}")
        
        return message.message_id
    
    def receive_message(self, message: CoordinationMessage):
        """Receive coordination message from another intersection"""
        if message.is_expired():
            self.logger.warning(f"Received expired message: {message.message_id}")
            return
        
        with self.coordination_lock:
            self.received_messages[message.sender_id].append(message)
            self.coordination_metrics['messages_received'] += 1
        
        self.logger.debug(f"Received {message.message_type.value} message from {message.sender_id}")
    
    def update_intersection_state(self, state: IntersectionState):
        """Update local intersection state"""
        with self.coordination_lock:
            self.intersection_states[self.intersection_id] = state
        
        # Broadcast state update to adjacent intersections
        adjacent_intersections = list(self.intersection_graph.neighbors(self.intersection_id))
        for adj_id in adjacent_intersections:
            self.send_message(
                adj_id,
                CoordinationMessageType.STATE_UPDATE,
                state.to_dict()
            )
    
    def coordinate_optimization(self, proposed_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate optimization with other intersections
        
        Args:
            proposed_action: Proposed action for this intersection
            
        Returns:
            Coordinated action after conflict resolution and network optimization
        """
        with self.coordination_lock:
            self.is_coordinating = True
            self.last_coordination_time = datetime.now()
        
        try:
            # Collect states from all intersections
            all_states = self._collect_intersection_states()
            
            # Collect proposed actions from all intersections
            all_actions = self._collect_proposed_actions(proposed_action)
            
            # Resolve conflicts
            resolved_actions = self.conflict_resolver.resolve_conflicts(all_actions, all_states)
            
            # Apply network-level optimization
            optimized_actions = self.network_optimizer.optimize_network_flow(all_states, resolved_actions)
            
            # Get coordinated action for this intersection
            coordinated_action = optimized_actions.get(self.intersection_id, proposed_action)
            
            # Update coordination metrics
            self.coordination_metrics['coordination_cycles'] += 1
            conflicts_resolved = len(resolved_actions) - len(optimized_actions)
            self.coordination_metrics['conflicts_resolved'] += conflicts_resolved
            
            self.logger.info(f"Coordination complete: {conflicts_resolved} conflicts resolved")
            
            return coordinated_action
            
        except Exception as e:
            self.logger.error(f"Error in coordination: {e}")
            return proposed_action
        
        finally:
            with self.coordination_lock:
                self.is_coordinating = False
    
    def _coordination_loop(self):
        """Main coordination loop"""
        while True:
            try:
                # Process received messages
                self._process_received_messages()
                
                # Clean up expired messages
                self._cleanup_expired_messages()
                
                # Sleep briefly to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                time.sleep(1)
    
    def _process_received_messages(self):
        """Process received coordination messages"""
        with self.coordination_lock:
            for sender_id, messages in self.received_messages.items():
                for message in messages:
                    if message.is_expired():
                        continue
                    
                    # Process based on message type
                    if message.message_type == CoordinationMessageType.STATE_UPDATE:
                        self._handle_state_update(message)
                    elif message.message_type == CoordinationMessageType.ACTION_PROPOSAL:
                        self._handle_action_proposal(message)
                    elif message.message_type == CoordinationMessageType.GREEN_WAVE_REQUEST:
                        self._handle_green_wave_request(message)
                    elif message.message_type == CoordinationMessageType.EMERGENCY_OVERRIDE:
                        self._handle_emergency_override(message)
    
    def _handle_state_update(self, message: CoordinationMessage):
        """Handle state update message"""
        state_data = message.data
        intersection_id = state_data['intersection_id']
        
        # Convert to IntersectionState
        state = IntersectionState(
            intersection_id=intersection_id,
            current_phase=state_data['current_phase'],
            phase_duration=state_data['phase_duration'],
            cycle_progress=state_data['cycle_progress'],
            congestion_level=state_data['congestion_level'],
            traffic_flow=state_data['traffic_flow'],
            queue_lengths=state_data['queue_lengths'],
            waiting_times=state_data['waiting_times'],
            emergency_vehicles=state_data['emergency_vehicles'],
            last_update=datetime.fromisoformat(state_data['last_update'])
        )
        
        self.intersection_states[intersection_id] = state
    
    def _handle_action_proposal(self, message: CoordinationMessage):
        """Handle action proposal message"""
        # Store action proposal for coordination
        pass  # Implementation depends on specific coordination strategy
    
    def _handle_green_wave_request(self, message: CoordinationMessage):
        """Handle green wave coordination request"""
        # Process green wave coordination
        pass  # Implementation depends on specific coordination strategy
    
    def _handle_emergency_override(self, message: CoordinationMessage):
        """Handle emergency vehicle override"""
        # Process emergency vehicle priority
        pass  # Implementation depends on specific coordination strategy
    
    def _collect_intersection_states(self) -> Dict[str, IntersectionState]:
        """Collect states from all intersections"""
        return self.intersection_states.copy()
    
    def _collect_proposed_actions(self, local_action: Dict[str, Any]) -> Dict[str, Any]:
        """Collect proposed actions from all intersections"""
        actions = {self.intersection_id: local_action}
        
        # In a real implementation, this would collect from other intersections
        # For now, we'll use local state
        
        return actions
    
    def _cleanup_expired_messages(self):
        """Clean up expired messages"""
        with self.coordination_lock:
            for sender_id in list(self.received_messages.keys()):
                self.received_messages[sender_id] = [
                    msg for msg in self.received_messages[sender_id]
                    if not msg.is_expired()
                ]
                
                # Remove empty lists
                if not self.received_messages[sender_id]:
                    del self.received_messages[sender_id]
    
    def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination performance metrics"""
        return {
            'intersection_id': self.intersection_id,
            'is_coordinating': self.is_coordinating,
            'last_coordination_time': self.last_coordination_time.isoformat() if self.last_coordination_time else None,
            'metrics': self.coordination_metrics.copy(),
            'connected_intersections': len(self.intersection_states),
            'pending_messages': sum(len(msgs) for msgs in self.received_messages.values())
        }
    
    def get_network_topology(self) -> Dict[str, Any]:
        """Get network topology information"""
        return {
            'nodes': list(self.intersection_graph.nodes()),
            'edges': list(self.intersection_graph.edges()),
            'adjacent_intersections': list(self.intersection_graph.neighbors(self.intersection_id))
        }
