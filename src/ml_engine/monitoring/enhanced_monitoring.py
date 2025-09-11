"""
Enhanced Monitoring and Logging System for ML Traffic Signal Optimization
Comprehensive monitoring, alerting, and performance tracking
"""

import logging
import time
import threading
import json
import os
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import psutil
import numpy as np
from pathlib import Path

from config.ml_config import LoggingConfig, get_config


class LogLevel(Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AlertType(Enum):
    """Alert types"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    HIGH_ERROR_RATE = "high_error_rate"
    RESOURCE_USAGE = "resource_usage"
    ALGORITHM_FAILURE = "algorithm_failure"
    DATA_QUALITY = "data_quality"
    SYSTEM_HEALTH = "system_health"


@dataclass
class Alert:
    """Alert definition"""
    alert_id: str
    alert_type: AlertType
    severity: LogLevel
    message: str
    timestamp: datetime
    intersection_id: Optional[str] = None
    algorithm_used: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    intersection_id: str
    algorithm_used: str
    wait_time: float
    throughput: float
    efficiency: float
    confidence: float
    processing_time: float
    system_metrics: Dict[str, Any]


@dataclass
class SystemHealth:
    """System health metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    active_optimizations: int
    error_rate: float
    avg_response_time: float


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alerts: Dict[str, Alert] = {}
        self.alert_handlers: List[Callable[[Alert], None]] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default alert rules
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default alert rules"""
        self.alert_rules = {
            'high_wait_time': {
                'metric': 'wait_time',
                'threshold': 60.0,
                'operator': '>',
                'alert_type': AlertType.PERFORMANCE_DEGRADATION,
                'severity': LogLevel.WARNING
            },
            'low_throughput': {
                'metric': 'throughput',
                'threshold': 200.0,
                'operator': '<',
                'alert_type': AlertType.PERFORMANCE_DEGRADATION,
                'severity': LogLevel.WARNING
            },
            'low_confidence': {
                'metric': 'confidence',
                'threshold': 0.3,
                'operator': '<',
                'alert_type': AlertType.ALGORITHM_FAILURE,
                'severity': LogLevel.ERROR
            },
            'high_processing_time': {
                'metric': 'processing_time',
                'threshold': 5.0,
                'operator': '>',
                'alert_type': AlertType.PERFORMANCE_DEGRADATION,
                'severity': LogLevel.WARNING
            },
            'high_cpu_usage': {
                'metric': 'cpu_usage',
                'threshold': 80.0,
                'operator': '>',
                'alert_type': AlertType.RESOURCE_USAGE,
                'severity': LogLevel.WARNING
            },
            'high_memory_usage': {
                'metric': 'memory_usage',
                'threshold': 85.0,
                'operator': '>',
                'alert_type': AlertType.RESOURCE_USAGE,
                'severity': LogLevel.WARNING
            }
        }
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert handler"""
        self.alert_handlers.append(handler)
    
    def check_metrics(self, metrics: Dict[str, Any], intersection_id: str = None):
        """Check metrics against alert rules"""
        for rule_name, rule in self.alert_rules.items():
            metric_name = rule['metric']
            threshold = rule['threshold']
            operator = rule['operator']
            
            if metric_name not in metrics:
                continue
            
            value = metrics[metric_name]
            should_alert = False
            
            if operator == '>':
                should_alert = value > threshold
            elif operator == '<':
                should_alert = value < threshold
            elif operator == '>=':
                should_alert = value >= threshold
            elif operator == '<=':
                should_alert = value <= threshold
            elif operator == '==':
                should_alert = value == threshold
            elif operator == '!=':
                should_alert = value != threshold
            
            if should_alert:
                self._create_alert(rule_name, rule, metrics, intersection_id)
    
    def _create_alert(self, rule_name: str, rule: Dict[str, Any], 
                     metrics: Dict[str, Any], intersection_id: str = None):
        """Create and send alert"""
        alert_id = f"{rule_name}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            alert_type=rule['alert_type'],
            severity=rule['severity'],
            message=f"{rule_name}: {rule['metric']} = {metrics[rule['metric']]} {rule['operator']} {rule['threshold']}",
            timestamp=datetime.now(),
            intersection_id=intersection_id,
            metrics=metrics
        )
        
        self.alerts[alert_id] = alert
        
        # Send to handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")
        
        self.logger.warning(f"Alert created: {alert.message}")
    
    def resolve_alert(self, alert_id: str):
        """Resolve an alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolution_time = datetime.now()
    
    def get_active_alerts(self) -> List[Alert]:
        """Get active (unresolved) alerts"""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alerts_by_type(self, alert_type: AlertType) -> List[Alert]:
        """Get alerts by type"""
        return [alert for alert in self.alerts.values() if alert.alert_type == alert_type]


class PerformanceTracker:
    """Tracks performance metrics over time"""
    
    def __init__(self, max_history: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.performance_history: deque = deque(maxlen=max_history)
        self.intersection_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.algorithm_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
    def record_performance(self, snapshot: PerformanceSnapshot):
        """Record performance snapshot"""
        self.performance_history.append(snapshot)
        
        # Record by intersection
        self.intersection_metrics[snapshot.intersection_id].append(snapshot)
        
        # Record by algorithm
        self.algorithm_metrics[snapshot.algorithm_used].append(snapshot)
    
    def get_performance_trends(self, intersection_id: str = None, 
                             algorithm: str = None, 
                             hours: int = 24) -> Dict[str, List[float]]:
        """Get performance trends"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        if intersection_id:
            data = [s for s in self.intersection_metrics[intersection_id] 
                   if s.timestamp >= cutoff_time]
        elif algorithm:
            data = [s for s in self.algorithm_metrics[algorithm] 
                   if s.timestamp >= cutoff_time]
        else:
            data = [s for s in self.performance_history 
                   if s.timestamp >= cutoff_time]
        
        if not data:
            return {}
        
        trends = {
            'wait_time': [s.wait_time for s in data],
            'throughput': [s.throughput for s in data],
            'efficiency': [s.efficiency for s in data],
            'confidence': [s.confidence for s in data],
            'processing_time': [s.processing_time for s in data]
        }
        
        return trends
    
    def calculate_performance_stats(self, intersection_id: str = None,
                                  algorithm: str = None,
                                  hours: int = 24) -> Dict[str, Any]:
        """Calculate performance statistics"""
        trends = self.get_performance_trends(intersection_id, algorithm, hours)
        
        if not trends:
            return {}
        
        stats = {}
        for metric, values in trends.items():
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values)
                }
        
        return stats


class SystemMonitor:
    """Monitors system resources and health"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.health_history: deque = deque(maxlen=100)
        self.monitoring_thread = None
        self.is_monitoring = False
        
    def start_monitoring(self, interval: int = 30):
        """Start system monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,)
        )
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self, interval: int):
        """System monitoring loop"""
        while self.is_monitoring:
            try:
                health = self.get_system_health()
                self.health_history.append(health)
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in system monitoring: {e}")
                time.sleep(interval)
    
    def get_system_health(self) -> SystemHealth:
        """Get current system health"""
        # CPU usage
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # Network I/O
        network = psutil.net_io_counters()
        network_io = {
            'bytes_sent': network.bytes_sent,
            'bytes_recv': network.bytes_recv,
            'packets_sent': network.packets_sent,
            'packets_recv': network.packets_recv
        }
        
        return SystemHealth(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            active_optimizations=0,  # Would be tracked by optimizer
            error_rate=0.0,  # Would be calculated from logs
            avg_response_time=0.0  # Would be calculated from performance data
        )
    
    def get_health_trends(self, hours: int = 24) -> Dict[str, List[float]]:
        """Get system health trends"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_health = [h for h in self.health_history if h.timestamp >= cutoff_time]
        
        if not recent_health:
            return {}
        
        return {
            'cpu_usage': [h.cpu_usage for h in recent_health],
            'memory_usage': [h.memory_usage for h in recent_health],
            'disk_usage': [h.disk_usage for h in recent_health],
            'error_rate': [h.error_rate for h in recent_health],
            'avg_response_time': [h.avg_response_time for h in recent_health]
        }


class EnhancedLogger:
    """Enhanced logging system with structured logging"""
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # Performance tracking
        self.log_counts: Dict[str, int] = defaultdict(int)
        self.error_counts: Dict[str, int] = defaultdict(int)
        
    def _setup_logging(self):
        """Setup logging configuration"""
        # Create logs directory
        log_dir = Path(self.config.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=self.config.log_format,
            handlers=[
                logging.StreamHandler(),
                logging.handlers.RotatingFileHandler(
                    self.config.log_file,
                    maxBytes=self.config.max_log_size,
                    backupCount=self.config.backup_count
                )
            ]
        )
        
        # Create specialized loggers
        self.traffic_logger = logging.getLogger('traffic_optimization')
        self.performance_logger = logging.getLogger('performance')
        self.error_logger = logging.getLogger('errors')
        self.system_logger = logging.getLogger('system')
    
    def log_optimization(self, intersection_id: str, algorithm: str, 
                        metrics: Dict[str, Any], processing_time: float):
        """Log optimization event"""
        log_data = {
            'event': 'optimization',
            'intersection_id': intersection_id,
            'algorithm': algorithm,
            'processing_time': processing_time,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.traffic_logger.info(json.dumps(log_data))
        self.log_counts['optimization'] += 1
    
    def log_performance(self, intersection_id: str, performance_metrics: Dict[str, Any]):
        """Log performance metrics"""
        log_data = {
            'event': 'performance',
            'intersection_id': intersection_id,
            'metrics': performance_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        self.performance_logger.info(json.dumps(log_data))
        self.log_counts['performance'] += 1
    
    def log_error(self, error_type: str, message: str, intersection_id: str = None,
                 algorithm: str = None, exception: Exception = None):
        """Log error event"""
        log_data = {
            'event': 'error',
            'error_type': error_type,
            'message': message,
            'intersection_id': intersection_id,
            'algorithm': algorithm,
            'exception': str(exception) if exception else None,
            'timestamp': datetime.now().isoformat()
        }
        
        self.error_logger.error(json.dumps(log_data))
        self.log_counts['error'] += 1
        self.error_counts[error_type] += 1
    
    def log_system_event(self, event_type: str, message: str, metrics: Dict[str, Any] = None):
        """Log system event"""
        log_data = {
            'event': 'system',
            'event_type': event_type,
            'message': message,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.system_logger.info(json.dumps(log_data))
        self.log_counts['system'] += 1
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        return {
            'log_counts': dict(self.log_counts),
            'error_counts': dict(self.error_counts),
            'total_logs': sum(self.log_counts.values()),
            'total_errors': sum(self.error_counts.values())
        }


class EnhancedMonitoring:
    """
    Enhanced monitoring system for ML traffic signal optimization
    
    Features:
    - Real-time performance tracking
    - System health monitoring
    - Alert management and notification
    - Structured logging with JSON format
    - Performance trend analysis
    - Resource usage monitoring
    - Error tracking and analysis
    """
    
    def __init__(self, config: Optional[LoggingConfig] = None):
        self.config = config or get_config().logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.alert_manager = AlertManager()
        self.performance_tracker = PerformanceTracker()
        self.system_monitor = SystemMonitor()
        self.enhanced_logger = EnhancedLogger(self.config)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        
        self.logger.info("Enhanced monitoring system initialized")
    
    def start_monitoring(self):
        """Start all monitoring systems"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        
        # Start system monitoring
        self.system_monitor.start_monitoring()
        
        # Start performance monitoring
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        self.logger.info("Enhanced monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring systems"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        
        # Stop system monitoring
        self.system_monitor.stop_monitoring()
        
        # Stop performance monitoring
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Enhanced monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Check system health
                health = self.system_monitor.get_system_health()
                
                # Check for alerts
                health_metrics = {
                    'cpu_usage': health.cpu_usage,
                    'memory_usage': health.memory_usage,
                    'disk_usage': health.disk_usage,
                    'error_rate': health.error_rate,
                    'avg_response_time': health.avg_response_time
                }
                self.alert_manager.check_metrics(health_metrics)
                
                time.sleep(30)  # Check every 30 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)
    
    def record_optimization(self, intersection_id: str, algorithm: str,
                          metrics: Dict[str, Any], processing_time: float):
        """Record optimization event"""
        # Log optimization
        self.enhanced_logger.log_optimization(
            intersection_id, algorithm, metrics, processing_time
        )
        
        # Record performance snapshot
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            intersection_id=intersection_id,
            algorithm_used=algorithm,
            wait_time=metrics.get('wait_time', 0),
            throughput=metrics.get('throughput', 0),
            efficiency=metrics.get('efficiency', 0),
            confidence=metrics.get('confidence', 0),
            processing_time=processing_time,
            system_metrics={}
        )
        self.performance_tracker.record_performance(snapshot)
        
        # Check for performance alerts
        self.alert_manager.check_metrics(metrics, intersection_id)
    
    def record_error(self, error_type: str, message: str, intersection_id: str = None,
                    algorithm: str = None, exception: Exception = None):
        """Record error event"""
        self.enhanced_logger.log_error(
            error_type, message, intersection_id, algorithm, exception
        )
    
    def record_system_event(self, event_type: str, message: str, metrics: Dict[str, Any] = None):
        """Record system event"""
        self.enhanced_logger.log_system_event(event_type, message, metrics)
    
    def get_performance_summary(self, intersection_id: str = None,
                              algorithm: str = None,
                              hours: int = 24) -> Dict[str, Any]:
        """Get performance summary"""
        stats = self.performance_tracker.calculate_performance_stats(
            intersection_id, algorithm, hours
        )
        
        trends = self.performance_tracker.get_performance_trends(
            intersection_id, algorithm, hours
        )
        
        return {
            'statistics': stats,
            'trends': trends,
            'time_range_hours': hours
        }
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get system health summary"""
        current_health = self.system_monitor.get_system_health()
        trends = self.system_monitor.get_health_trends(24)
        
        return {
            'current_health': {
                'cpu_usage': current_health.cpu_usage,
                'memory_usage': current_health.memory_usage,
                'disk_usage': current_health.disk_usage,
                'error_rate': current_health.error_rate,
                'avg_response_time': current_health.avg_response_time
            },
            'trends_24h': trends,
            'timestamp': current_health.timestamp.isoformat()
        }
    
    def get_alerts_summary(self) -> Dict[str, Any]:
        """Get alerts summary"""
        active_alerts = self.alert_manager.get_active_alerts()
        
        alert_summary = {
            'total_active': len(active_alerts),
            'by_type': {},
            'by_severity': {},
            'recent_alerts': []
        }
        
        for alert in active_alerts:
            # Count by type
            alert_type = alert.alert_type.value
            alert_summary['by_type'][alert_type] = alert_summary['by_type'].get(alert_type, 0) + 1
            
            # Count by severity
            severity = alert.severity.value
            alert_summary['by_severity'][severity] = alert_summary['by_severity'].get(severity, 0) + 1
            
            # Recent alerts (last 10)
            if len(alert_summary['recent_alerts']) < 10:
                alert_summary['recent_alerts'].append({
                    'alert_id': alert.alert_id,
                    'type': alert_type,
                    'severity': severity,
                    'message': alert.message,
                    'timestamp': alert.timestamp.isoformat(),
                    'intersection_id': alert.intersection_id
                })
        
        return alert_summary
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            'performance': self.get_performance_summary(),
            'system_health': self.get_system_health_summary(),
            'alerts': self.get_alerts_summary(),
            'logging': self.enhanced_logger.get_log_statistics(),
            'timestamp': datetime.now().isoformat()
        }
    
    def export_monitoring_data(self, filepath: str, hours: int = 24):
        """Export monitoring data to file"""
        export_data = {
            'performance_trends': self.performance_tracker.get_performance_trends(hours=hours),
            'system_health_trends': self.system_monitor.get_health_trends(hours=hours),
            'alerts': [alert.__dict__ for alert in self.alert_manager.alerts.values()],
            'log_statistics': self.enhanced_logger.get_log_statistics(),
            'export_timestamp': datetime.now().isoformat(),
            'export_hours': hours
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring data exported to {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the enhanced monitoring system
    monitoring = EnhancedMonitoring()
    
    # Start monitoring
    monitoring.start_monitoring()
    
    # Test performance recording
    print("Testing performance recording...")
    monitoring.record_optimization(
        "junction-1", "q_learning", 
        {"wait_time": 25.0, "throughput": 600.0, "efficiency": 0.8, "confidence": 0.9},
        0.5
    )
    
    # Test error recording
    print("Testing error recording...")
    monitoring.record_error(
        "optimization_failure", "Failed to optimize signals", 
        "junction-1", "q_learning", Exception("Test error")
    )
    
    # Test system event recording
    print("Testing system event recording...")
    monitoring.record_system_event(
        "startup", "Monitoring system started", 
        {"version": "1.0", "components": 4}
    )
    
    # Get summaries
    print("Performance summary:", monitoring.get_performance_summary())
    print("System health:", monitoring.get_system_health_summary())
    print("Alerts summary:", monitoring.get_alerts_summary())
    
    # Stop monitoring
    monitoring.stop_monitoring()
