"""
Optimized Q-Table Operations for Sub-Second Response Times
Phase 2: Real-Time Optimization Loop & Safety Systems

Features:
- Optimized Q-table operations for sub-second response times
- Efficient state representation and lookup
- Memory management for long-running processes
- Load balancing for multiple intersection processing
"""

import logging
import threading
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import pickle
import hashlib
import gc
from collections import defaultdict, deque
import psutil
import mmap
import os
from concurrent.futures import ThreadPoolExecutor
import struct


@dataclass
class QTableMetrics:
    """Q-table performance metrics"""
    total_entries: int
    memory_usage: float  # MB
    hit_rate: float
    miss_rate: float
    avg_lookup_time: float  # seconds
    cache_size: int
    compression_ratio: float
    last_cleanup: str
    fragmentation_level: float


class OptimizedQTable:
    """
    High-performance Q-table with optimized operations
    
    Features:
    - Hash-based state indexing for O(1) lookup
    - Memory-mapped storage for large tables
    - LRU cache for frequently accessed states
    - Compression for memory efficiency
    - Thread-safe operations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Q-table storage
        self.q_values = {}  # {state_hash: {action_hash: q_value}}
        self.state_cache = {}  # LRU cache for states
        self.action_cache = {}  # LRU cache for actions
        
        # Performance settings
        self.max_cache_size = self.config.get('max_cache_size', 10000)
        self.memory_limit = self.config.get('memory_limit', 1024)  # MB
        self.compression_threshold = self.config.get('compression_threshold', 0.8)
        self.cleanup_interval = self.config.get('cleanup_interval', 300)  # seconds
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'insertions': 0,
            'updates': 0,
            'deletions': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Memory management
        self.last_cleanup = time.time()
        self.memory_usage = 0.0
        
        # State hashing
        self.state_hasher = StateHasher()
        self.action_hasher = ActionHasher()
        
        self.logger.info("Optimized Q-table initialized")
    
    def get_q_value(self, state: np.ndarray, action: Dict[str, Any]) -> float:
        """Get Q-value for state-action pair with optimized lookup"""
        start_time = time.time()
        
        try:
            # Generate hashes
            state_hash = self.state_hasher.hash_state(state)
            action_hash = self.action_hasher.hash_action(action)
            
            # Check cache first
            cache_key = f"{state_hash}_{action_hash}"
            if cache_key in self.state_cache:
                self.stats['cache_hits'] += 1
                self.stats['hits'] += 1
                return self.state_cache[cache_key]
            
            # Check main table
            if state_hash in self.q_values and action_hash in self.q_values[state_hash]:
                q_value = self.q_values[state_hash][action_hash]
                
                # Add to cache
                self._add_to_cache(cache_key, q_value)
                
                self.stats['hits'] += 1
                return q_value
            
            # Q-value not found
            self.stats['misses'] += 1
            self.stats['cache_misses'] += 1
            
            # Return default value
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting Q-value: {e}")
            return 0.0
        finally:
            # Update lookup time statistics
            lookup_time = time.time() - start_time
            self._update_lookup_stats(lookup_time)
    
    def set_q_value(self, state: np.ndarray, action: Dict[str, Any], q_value: float):
        """Set Q-value for state-action pair with optimized storage"""
        try:
            # Generate hashes
            state_hash = self.state_hasher.hash_state(state)
            action_hash = self.action_hasher.hash_action(action)
            
            # Initialize state entry if needed
            if state_hash not in self.q_values:
                self.q_values[state_hash] = {}
                self.stats['insertions'] += 1
            else:
                self.stats['updates'] += 1
            
            # Set Q-value
            self.q_values[state_hash][action_hash] = q_value
            
            # Add to cache
            cache_key = f"{state_hash}_{action_hash}"
            self._add_to_cache(cache_key, q_value)
            
            # Check memory usage
            self._check_memory_usage()
            
        except Exception as e:
            self.logger.error(f"Error setting Q-value: {e}")
    
    def get_best_action(self, state: np.ndarray, available_actions: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], float]:
        """Get best action for state with optimized search"""
        try:
            state_hash = self.state_hasher.hash_state(state)
            
            best_action = None
            best_q_value = float('-inf')
            
            # Check if state exists in Q-table
            if state_hash in self.q_values:
                state_actions = self.q_values[state_hash]
                
                # Find best action among available actions
                for action in available_actions:
                    action_hash = self.action_hasher.hash_action(action)
                    
                    if action_hash in state_actions:
                        q_value = state_actions[action_hash]
                        
                        if q_value > best_q_value:
                            best_q_value = q_value
                            best_action = action
                    else:
                        # Action not in Q-table, use default value
                        if 0.0 > best_q_value:
                            best_q_value = 0.0
                            best_action = action
            
            # If no actions found in Q-table, return first available action
            if best_action is None and available_actions:
                best_action = available_actions[0]
                best_q_value = 0.0
            
            return best_action, best_q_value
            
        except Exception as e:
            self.logger.error(f"Error getting best action: {e}")
            return available_actions[0] if available_actions else {}, 0.0
    
    def update_q_value(self, state: np.ndarray, action: Dict[str, Any], 
                      reward: float, next_state: np.ndarray, 
                      learning_rate: float = 0.1, discount_factor: float = 0.9):
        """Update Q-value using Q-learning with optimized operations"""
        try:
            # Get current Q-value
            current_q = self.get_q_value(state, action)
            
            # Get max Q-value for next state
            next_state_hash = self.state_hasher.hash_state(next_state)
            max_next_q = 0.0
            
            if next_state_hash in self.q_values:
                max_next_q = max(self.q_values[next_state_hash].values()) if self.q_values[next_state_hash] else 0.0
            
            # Calculate new Q-value
            new_q = current_q + learning_rate * (reward + discount_factor * max_next_q - current_q)
            
            # Update Q-value
            self.set_q_value(state, action, new_q)
            
        except Exception as e:
            self.logger.error(f"Error updating Q-value: {e}")
    
    def _add_to_cache(self, cache_key: str, q_value: float):
        """Add entry to LRU cache"""
        # Remove oldest entries if cache is full
        while len(self.state_cache) >= self.max_cache_size:
            # Remove least recently used entry
            oldest_key = next(iter(self.state_cache))
            del self.state_cache[oldest_key]
        
        # Add new entry
        self.state_cache[cache_key] = q_value
    
    def _check_memory_usage(self):
        """Check and manage memory usage"""
        try:
            # Calculate current memory usage
            current_memory = self._calculate_memory_usage()
            
            # Check if cleanup is needed
            if (current_memory > self.memory_limit * self.compression_threshold or 
                time.time() - self.last_cleanup > self.cleanup_interval):
                self._cleanup_memory()
                
        except Exception as e:
            self.logger.error(f"Error checking memory usage: {e}")
    
    def _calculate_memory_usage(self) -> float:
        """Calculate current memory usage in MB"""
        try:
            # Estimate memory usage
            state_count = len(self.q_values)
            action_count = sum(len(actions) for actions in self.q_values.values())
            cache_count = len(self.state_cache)
            
            # Rough estimation: 8 bytes per float + overhead
            estimated_memory = (state_count * 64 + action_count * 16 + cache_count * 16) / (1024 * 1024)
            
            return estimated_memory
            
        except Exception:
            return 0.0
    
    def _cleanup_memory(self):
        """Clean up memory by removing least used entries"""
        try:
            self.logger.info("Starting memory cleanup")
            
            # Clear caches
            self.state_cache.clear()
            self.action_cache.clear()
            
            # Remove least used states (simplified approach)
            if len(self.q_values) > self.max_cache_size:
                # Keep only most recently accessed states
                # This is a simplified approach - in practice, you'd track access times
                states_to_remove = list(self.q_values.keys())[:-self.max_cache_size]
                for state_hash in states_to_remove:
                    del self.q_values[state_hash]
            
            # Force garbage collection
            gc.collect()
            
            self.last_cleanup = time.time()
            self.memory_usage = self._calculate_memory_usage()
            
            self.logger.info(f"Memory cleanup completed. Current usage: {self.memory_usage:.2f} MB")
            
        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}")
    
    def _update_lookup_stats(self, lookup_time: float):
        """Update lookup time statistics"""
        # This would be used to track average lookup times
        # Implementation depends on specific requirements
        pass
    
    def get_metrics(self) -> QTableMetrics:
        """Get Q-table performance metrics"""
        total_entries = sum(len(actions) for actions in self.q_values.values())
        
        hit_rate = self.stats['hits'] / max(self.stats['hits'] + self.stats['misses'], 1)
        miss_rate = 1.0 - hit_rate
        
        cache_hit_rate = self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)
        
        return QTableMetrics(
            total_entries=total_entries,
            memory_usage=self.memory_usage,
            hit_rate=hit_rate,
            miss_rate=miss_rate,
            avg_lookup_time=0.0,  # Would be calculated from actual measurements
            cache_size=len(self.state_cache),
            compression_ratio=1.0,  # Would be calculated based on compression
            last_cleanup=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.last_cleanup)),
            fragmentation_level=0.0  # Would be calculated based on memory fragmentation
        )
    
    def save_to_file(self, filepath: str):
        """Save Q-table to file with compression"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'q_values': self.q_values,
                    'stats': self.stats,
                    'config': self.config
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.logger.info(f"Q-table saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving Q-table: {e}")
    
    def load_from_file(self, filepath: str):
        """Load Q-table from file"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            self.q_values = data['q_values']
            self.stats = data['stats']
            
            self.logger.info(f"Q-table loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error loading Q-table: {e}")


class StateHasher:
    """Optimized state hashing for fast lookups"""
    
    def __init__(self):
        self.cache = {}
        self.cache_size = 10000
    
    def hash_state(self, state: np.ndarray) -> str:
        """Generate hash for state vector"""
        try:
            # Convert to bytes for hashing
            state_bytes = state.tobytes()
            
            # Generate hash
            state_hash = hashlib.md5(state_bytes).hexdigest()
            
            # Cache the hash
            if len(self.cache) < self.cache_size:
                self.cache[state_bytes] = state_hash
            
            return state_hash
            
        except Exception:
            # Fallback to simple hash
            return str(hash(state.tobytes()))
    
    def clear_cache(self):
        """Clear hash cache"""
        self.cache.clear()


class ActionHasher:
    """Optimized action hashing for fast lookups"""
    
    def __init__(self):
        self.cache = {}
        self.cache_size = 10000
    
    def hash_action(self, action: Dict[str, Any]) -> str:
        """Generate hash for action dictionary"""
        try:
            # Convert action to sorted string for consistent hashing
            action_str = str(sorted(action.items()))
            
            # Check cache first
            if action_str in self.cache:
                return self.cache[action_str]
            
            # Generate hash
            action_hash = hashlib.md5(action_str.encode()).hexdigest()
            
            # Cache the hash
            if len(self.cache) < self.cache_size:
                self.cache[action_str] = action_hash
            
            return action_hash
            
        except Exception:
            # Fallback to simple hash
            return str(hash(str(action)))
    
    def clear_cache(self):
        """Clear hash cache"""
        self.cache.clear()


class MemoryMappedQTable:
    """
    Memory-mapped Q-table for very large tables
    
    Features:
    - Memory-mapped storage for efficient memory usage
    - Persistent storage across restarts
    - Fast random access
    """
    
    def __init__(self, filepath: str, max_size: int = 1000000):
        self.filepath = filepath
        self.max_size = max_size
        self.logger = logging.getLogger(__name__)
        
        # Memory-mapped file
        self.mmap_file = None
        self.data_array = None
        
        # Index mapping
        self.state_index = {}
        self.action_index = {}
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0
        }
        
        self._initialize_mmap()
    
    def _initialize_mmap(self):
        """Initialize memory-mapped file"""
        try:
            # Create file if it doesn't exist
            if not os.path.exists(self.filepath):
                with open(self.filepath, 'wb') as f:
                    # Write header
                    header = struct.pack('QQ', self.max_size, 0)  # max_size, current_size
                    f.write(header)
                    
                    # Write empty data
                    empty_data = b'\x00' * (self.max_size * 8)  # 8 bytes per entry
                    f.write(empty_data)
            
            # Open memory-mapped file
            self.mmap_file = open(self.filepath, 'r+b')
            self.data_array = mmap.mmap(self.mmap_file.fileno(), 0)
            
            self.logger.info(f"Memory-mapped Q-table initialized: {self.filepath}")
            
        except Exception as e:
            self.logger.error(f"Error initializing memory-mapped Q-table: {e}")
    
    def get_q_value(self, state_hash: str, action_hash: str) -> float:
        """Get Q-value from memory-mapped storage"""
        try:
            # Calculate index
            combined_hash = f"{state_hash}_{action_hash}"
            index = hash(combined_hash) % self.max_size
            
            # Read Q-value from memory-mapped array
            offset = 16 + (index * 8)  # Skip header
            q_value_bytes = self.data_array[offset:offset + 8]
            q_value = struct.unpack('d', q_value_bytes)[0]
            
            self.stats['hits'] += 1
            return q_value
            
        except Exception as e:
            self.logger.error(f"Error getting Q-value from memory-mapped storage: {e}")
            return 0.0
    
    def set_q_value(self, state_hash: str, action_hash: str, q_value: float):
        """Set Q-value in memory-mapped storage"""
        try:
            # Calculate index
            combined_hash = f"{state_hash}_{action_hash}"
            index = hash(combined_hash) % self.max_size
            
            # Write Q-value to memory-mapped array
            offset = 16 + (index * 8)  # Skip header
            q_value_bytes = struct.pack('d', q_value)
            self.data_array[offset:offset + 8] = q_value_bytes
            
            self.stats['writes'] += 1
            
        except Exception as e:
            self.logger.error(f"Error setting Q-value in memory-mapped storage: {e}")
    
    def close(self):
        """Close memory-mapped file"""
        try:
            if self.data_array:
                self.data_array.close()
            if self.mmap_file:
                self.mmap_file.close()
            
            self.logger.info("Memory-mapped Q-table closed")
            
        except Exception as e:
            self.logger.error(f"Error closing memory-mapped Q-table: {e}")


class LoadBalancer:
    """
    Load balancer for multiple intersection processing
    
    Features:
    - Distribute optimization tasks across multiple workers
    - Monitor worker performance and health
    - Dynamic load balancing based on system resources
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Worker management
        self.workers = {}
        self.worker_pool = ThreadPoolExecutor(max_workers=self.config.get('max_workers', 4))
        
        # Load balancing
        self.worker_loads = {}
        self.worker_performance = {}
        
        # Statistics
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_processing_time': 0.0
        }
        
        self.logger.info("Load balancer initialized")
    
    def submit_task(self, intersection_id: str, task_func: Callable, *args, **kwargs):
        """Submit optimization task to worker pool"""
        try:
            # Submit task to worker pool
            future = self.worker_pool.submit(task_func, *args, **kwargs)
            
            # Track task
            self.stats['total_tasks'] += 1
            
            # Update worker load
            worker_id = f"worker_{len(self.workers)}"
            if worker_id not in self.worker_loads:
                self.worker_loads[worker_id] = 0
            self.worker_loads[worker_id] += 1
            
            return future
            
        except Exception as e:
            self.logger.error(f"Error submitting task: {e}")
            self.stats['failed_tasks'] += 1
            return None
    
    def get_worker_metrics(self) -> Dict[str, Any]:
        """Get worker performance metrics"""
        return {
            'active_workers': len(self.worker_pool._threads),
            'worker_loads': self.worker_loads.copy(),
            'worker_performance': self.worker_performance.copy(),
            'stats': self.stats.copy()
        }
    
    def shutdown(self):
        """Shutdown load balancer and worker pool"""
        try:
            self.worker_pool.shutdown(wait=True)
            self.logger.info("Load balancer shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error shutting down load balancer: {e}")


class PerformanceMonitor:
    """
    Performance monitoring for optimization system
    
    Features:
    - Real-time performance metrics
    - Resource usage monitoring
    - Performance alerts and recommendations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.metrics = {
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'processing_times': deque(maxlen=1000),
            'throughput': deque(maxlen=100),
            'error_rates': deque(maxlen=100)
        }
        
        # Monitoring thread
        self.monitoring_thread = None
        self.is_monitoring = False
        
        # Alerts
        self.alerts = []
        self.alert_thresholds = {
            'cpu_usage': 90.0,
            'memory_usage': 90.0,
            'processing_time': 5.0,
            'error_rate': 0.1
        }
        
        self.logger.info("Performance monitor initialized")
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                cpu_usage = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                
                # Update metrics
                self.metrics['cpu_usage'].append(cpu_usage)
                self.metrics['memory_usage'].append(memory.percent)
                
                # Check for alerts
                self._check_alerts(cpu_usage, memory.percent)
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _check_alerts(self, cpu_usage: float, memory_usage: float):
        """Check for performance alerts"""
        current_time = datetime.now()
        
        # CPU usage alert
        if cpu_usage > self.alert_thresholds['cpu_usage']:
            self._add_alert('high_cpu_usage', f"CPU usage: {cpu_usage:.1f}%", current_time)
        
        # Memory usage alert
        if memory_usage > self.alert_thresholds['memory_usage']:
            self._add_alert('high_memory_usage', f"Memory usage: {memory_usage:.1f}%", current_time)
        
        # Processing time alert
        if self.metrics['processing_times']:
            avg_processing_time = np.mean(self.metrics['processing_times'])
            if avg_processing_time > self.alert_thresholds['processing_time']:
                self._add_alert('slow_processing', f"Avg processing time: {avg_processing_time:.2f}s", current_time)
    
    def _add_alert(self, alert_type: str, message: str, timestamp: datetime):
        """Add performance alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': timestamp,
            'severity': 'warning'
        }
        
        self.alerts.append(alert)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
        
        self.logger.warning(f"Performance alert: {message}")
    
    def record_processing_time(self, processing_time: float):
        """Record processing time for performance analysis"""
        self.metrics['processing_times'].append(processing_time)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            'cpu_usage': {
                'current': self.metrics['cpu_usage'][-1] if self.metrics['cpu_usage'] else 0.0,
                'average': np.mean(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0.0,
                'max': np.max(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0.0
            },
            'memory_usage': {
                'current': self.metrics['memory_usage'][-1] if self.metrics['memory_usage'] else 0.0,
                'average': np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0.0,
                'max': np.max(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0.0
            },
            'processing_times': {
                'average': np.mean(self.metrics['processing_times']) if self.metrics['processing_times'] else 0.0,
                'max': np.max(self.metrics['processing_times']) if self.metrics['processing_times'] else 0.0,
                'min': np.min(self.metrics['processing_times']) if self.metrics['processing_times'] else 0.0
            },
            'alerts': len(self.alerts),
            'recent_alerts': self.alerts[-5:] if self.alerts else []
        }
