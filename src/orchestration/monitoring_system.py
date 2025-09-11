#!/usr/bin/env python3
"""
Monitoring and Observability System for Smart Traffic Management
Provides system health monitoring, metrics collection, and alerting
"""

import asyncio
import json
import logging
import time
import psutil
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import redis
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import threading

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ComponentHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DOWN = "down"

@dataclass
class HealthCheck:
    name: str
    url: str
    timeout: int = 5
    expected_status: int = 200
    check_interval: int = 30
    last_check: Optional[datetime] = None
    last_status: Optional[ComponentHealth] = None
    consecutive_failures: int = 0
    enabled: bool = True

@dataclass
class Alert:
    id: str
    level: AlertLevel
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricThreshold:
    name: str
    warning_threshold: float
    critical_threshold: float
    unit: str = ""
    enabled: bool = True

class MonitoringSystem:
    """Comprehensive monitoring and observability system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.running = False
        
        # Health checks
        self.health_checks: List[HealthCheck] = []
        self.component_status: Dict[str, ComponentHealth] = {}
        
        # Alerts
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Metrics
        self.metrics = {
            "system": {},
            "components": {},
            "performance": {},
            "custom": {}
        }
        
        # Thresholds
        self.thresholds: List[MetricThreshold] = []
        
        # Prometheus metrics
        self.prometheus_metrics = self._setup_prometheus_metrics()
        
        # Redis for metrics storage
        self.redis_client = None
        self._initialize_redis()
        
        # Alert handlers
        self.alert_handlers: List[Callable] = []
        
        self._setup_health_checks()
        self._setup_thresholds()
    
    def _initialize_redis(self):
        """Initialize Redis connection for metrics storage"""
        try:
            redis_config = self.config.get("monitoring", {}).get("redis", {})
            self.redis_client = redis.Redis(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                password=redis_config.get("password", ""),
                decode_responses=True
            )
            self.logger.info("Redis connection established for monitoring")
        except Exception as e:
            self.logger.error(f"Failed to initialize Redis: {e}")
    
    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        return {
            "http_requests_total": Counter(
                "http_requests_total", 
                "Total HTTP requests", 
                ["method", "endpoint", "status"]
            ),
            "http_request_duration": Histogram(
                "http_request_duration_seconds",
                "HTTP request duration",
                ["method", "endpoint"]
            ),
            "system_cpu_usage": Gauge(
                "system_cpu_usage_percent",
                "System CPU usage percentage"
            ),
            "system_memory_usage": Gauge(
                "system_memory_usage_percent",
                "System memory usage percentage"
            ),
            "system_disk_usage": Gauge(
                "system_disk_usage_percent",
                "System disk usage percentage"
            ),
            "component_health": Gauge(
                "component_health_status",
                "Component health status (1=healthy, 0=unhealthy)",
                ["component"]
            ),
            "active_alerts": Gauge(
                "active_alerts_total",
                "Total number of active alerts",
                ["level"]
            ),
            "data_flow_messages": Counter(
                "data_flow_messages_total",
                "Total data flow messages processed",
                ["source", "target", "type"]
            )
        }
    
    def _setup_health_checks(self):
        """Setup health checks for all components"""
        components = self.config.get("components", {})
        
        for name, config in components.items():
            health_check = HealthCheck(
                name=name,
                url=config.get("health_check_url", f"http://localhost:{config.get('port', 8000)}/health"),
                timeout=config.get("timeout", 5),
                check_interval=config.get("check_interval", 30)
            )
            self.health_checks.append(health_check)
            self.component_status[name] = ComponentHealth.UNHEALTHY
    
    def _setup_thresholds(self):
        """Setup metric thresholds for alerting"""
        thresholds_config = self.config.get("monitoring", {}).get("alert_thresholds", {})
        
        self.thresholds = [
            MetricThreshold(
                name="cpu_usage",
                warning_threshold=thresholds_config.get("cpu_usage", 80),
                critical_threshold=thresholds_config.get("cpu_usage", 90),
                unit="%"
            ),
            MetricThreshold(
                name="memory_usage",
                warning_threshold=thresholds_config.get("memory_usage", 85),
                critical_threshold=thresholds_config.get("memory_usage", 95),
                unit="%"
            ),
            MetricThreshold(
                name="disk_usage",
                warning_threshold=thresholds_config.get("disk_usage", 80),
                critical_threshold=thresholds_config.get("disk_usage", 90),
                unit="%"
            ),
            MetricThreshold(
                name="response_time",
                warning_threshold=thresholds_config.get("response_time", 2.0),
                critical_threshold=thresholds_config.get("response_time", 5.0),
                unit="s"
            )
        ]
    
    async def start(self):
        """Start the monitoring system"""
        self.logger.info("Starting Monitoring System...")
        self.running = True
        
        # Start Prometheus metrics server
        prometheus_port = self.config.get("monitoring", {}).get("prometheus", {}).get("port", 9090)
        start_http_server(prometheus_port)
        self.logger.info(f"Prometheus metrics server started on port {prometheus_port}")
        
        # Start monitoring tasks
        tasks = [
            asyncio.create_task(self._health_check_loop()),
            asyncio.create_task(self._system_metrics_loop()),
            asyncio.create_task(self._threshold_monitoring_loop()),
            asyncio.create_task(self._alert_processing_loop()),
            asyncio.create_task(self._metrics_cleanup_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Monitoring system error: {e}")
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the monitoring system"""
        self.logger.info("Stopping Monitoring System...")
        self.running = False
    
    async def _health_check_loop(self):
        """Continuous health checking of all components"""
        while self.running:
            for health_check in self.health_checks:
                if health_check.enabled:
                    await self._perform_health_check(health_check)
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    async def _perform_health_check(self, health_check: HealthCheck):
        """Perform health check for a single component"""
        try:
            start_time = time.time()
            response = requests.get(
                health_check.url,
                timeout=health_check.timeout
            )
            response_time = time.time() - start_time
            
            health_check.last_check = datetime.now()
            
            if response.status_code == health_check.expected_status:
                health_check.last_status = ComponentHealth.HEALTHY
                health_check.consecutive_failures = 0
                self.component_status[health_check.name] = ComponentHealth.HEALTHY
                
                # Update Prometheus metrics
                self.prometheus_metrics["component_health"].labels(
                    component=health_check.name
                ).set(1)
                
                # Resolve any active alerts for this component
                await self._resolve_component_alerts(health_check.name)
                
            else:
                health_check.last_status = ComponentHealth.UNHEALTHY
                health_check.consecutive_failures += 1
                self.component_status[health_check.name] = ComponentHealth.UNHEALTHY
                
                # Update Prometheus metrics
                self.prometheus_metrics["component_health"].labels(
                    component=health_check.name
                ).set(0)
                
                # Create alert
                await self._create_alert(
                    level=AlertLevel.ERROR,
                    component=health_check.name,
                    message=f"Health check failed: HTTP {response.status_code}",
                    metadata={"response_time": response_time, "status_code": response.status_code}
                )
        
        except requests.exceptions.Timeout:
            health_check.last_status = ComponentHealth.UNHEALTHY
            health_check.consecutive_failures += 1
            self.component_status[health_check.name] = ComponentHealth.UNHEALTHY
            
            await self._create_alert(
                level=AlertLevel.ERROR,
                component=health_check.name,
                message="Health check timeout",
                metadata={"timeout": health_check.timeout}
            )
        
        except Exception as e:
            health_check.last_status = ComponentHealth.UNHEALTHY
            health_check.consecutive_failures += 1
            self.component_status[health_check.name] = ComponentHealth.UNHEALTHY
            
            await self._create_alert(
                level=AlertLevel.ERROR,
                component=health_check.name,
                message=f"Health check error: {str(e)}",
                metadata={"error": str(e)}
            )
    
    async def _system_metrics_loop(self):
        """Collect system metrics"""
        while self.running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.metrics["system"]["cpu_usage"] = cpu_percent
                self.prometheus_metrics["system_cpu_usage"].set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                self.metrics["system"]["memory_usage"] = memory_percent
                self.metrics["system"]["memory_available"] = memory.available
                self.prometheus_metrics["system_memory_usage"].set(memory_percent)
                
                # Disk usage
                disk = psutil.disk_usage('/')
                disk_percent = (disk.used / disk.total) * 100
                self.metrics["system"]["disk_usage"] = disk_percent
                self.metrics["system"]["disk_free"] = disk.free
                self.prometheus_metrics["system_disk_usage"].set(disk_percent)
                
                # Network I/O
                network = psutil.net_io_counters()
                self.metrics["system"]["network_bytes_sent"] = network.bytes_sent
                self.metrics["system"]["network_bytes_recv"] = network.bytes_recv
                
                # Process count
                self.metrics["system"]["process_count"] = len(psutil.pids())
                
                # Store metrics in Redis
                if self.redis_client:
                    self.redis_client.setex(
                        "monitoring:system_metrics",
                        300,  # 5 minutes TTL
                        json.dumps(self.metrics["system"])
                    )
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                self.logger.error(f"System metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _threshold_monitoring_loop(self):
        """Monitor metrics against thresholds"""
        while self.running:
            try:
                for threshold in self.thresholds:
                    if not threshold.enabled:
                        continue
                    
                    value = self.metrics["system"].get(threshold.name, 0)
                    
                    if value >= threshold.critical_threshold:
                        await self._create_alert(
                            level=AlertLevel.CRITICAL,
                            component="system",
                            message=f"{threshold.name} is critical: {value}{threshold.unit} (threshold: {threshold.critical_threshold}{threshold.unit})",
                            metadata={"value": value, "threshold": threshold.critical_threshold}
                        )
                    elif value >= threshold.warning_threshold:
                        await self._create_alert(
                            level=AlertLevel.WARNING,
                            component="system",
                            message=f"{threshold.name} is high: {value}{threshold.unit} (threshold: {threshold.warning_threshold}{threshold.unit})",
                            metadata={"value": value, "threshold": threshold.warning_threshold}
                        )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Threshold monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _create_alert(self, level: AlertLevel, component: str, message: str, metadata: Dict[str, Any] = None):
        """Create a new alert"""
        alert_id = f"{component}_{level.value}_{int(time.time())}"
        
        # Check if similar alert already exists
        existing_alert = None
        for alert in self.active_alerts.values():
            if (alert.component == component and 
                alert.level == level and 
                alert.message == message and 
                not alert.resolved):
                existing_alert = alert
                break
        
        if existing_alert:
            # Update existing alert timestamp
            existing_alert.timestamp = datetime.now()
            return
        
        alert = Alert(
            id=alert_id,
            level=level,
            component=component,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # Update Prometheus metrics
        self.prometheus_metrics["active_alerts"].labels(level=level.value).inc()
        
        # Trigger alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler error: {e}")
        
        self.logger.warning(f"Alert created: {alert_id} - {message}")
    
    async def _resolve_component_alerts(self, component: str):
        """Resolve all active alerts for a component"""
        resolved_count = 0
        for alert in self.active_alerts.values():
            if alert.component == component and not alert.resolved:
                alert.resolved = True
                alert.resolved_at = datetime.now()
                resolved_count += 1
                
                # Update Prometheus metrics
                self.prometheus_metrics["active_alerts"].labels(level=alert.level.value).dec()
        
        if resolved_count > 0:
            self.logger.info(f"Resolved {resolved_count} alerts for component {component}")
    
    async def _alert_processing_loop(self):
        """Process and send alerts"""
        while self.running:
            try:
                # Send alerts via email, Slack, etc.
                await self._send_alerts()
                
                # Clean up old resolved alerts
                await self._cleanup_old_alerts()
                
                await asyncio.sleep(60)  # Process every minute
                
            except Exception as e:
                self.logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(60)
    
    async def _send_alerts(self):
        """Send alerts via configured channels"""
        for alert in self.active_alerts.values():
            if not alert.resolved:
                # Send email alert
                await self._send_email_alert(alert)
                
                # Send Slack alert (if configured)
                await self._send_slack_alert(alert)
    
    async def _send_email_alert(self, alert: Alert):
        """Send email alert"""
        try:
            email_config = self.config.get("monitoring", {}).get("email", {})
            if not email_config.get("enabled", False):
                return
            
            msg = MIMEMultipart()
            msg['From'] = email_config.get("from_email")
            msg['To'] = email_config.get("to_email")
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.component} - {alert.message}"
            
            body = f"""
            Alert Details:
            Component: {alert.component}
            Level: {alert.level.value}
            Message: {alert.message}
            Timestamp: {alert.timestamp}
            Metadata: {json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(email_config.get("smtp_host"), email_config.get("smtp_port"))
            server.starttls()
            server.login(email_config.get("username"), email_config.get("password"))
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")
    
    async def _send_slack_alert(self, alert: Alert):
        """Send Slack alert"""
        try:
            slack_config = self.config.get("monitoring", {}).get("slack", {})
            if not slack_config.get("enabled", False):
                return
            
            webhook_url = slack_config.get("webhook_url")
            if not webhook_url:
                return
            
            color_map = {
                AlertLevel.INFO: "good",
                AlertLevel.WARNING: "warning",
                AlertLevel.ERROR: "danger",
                AlertLevel.CRITICAL: "danger"
            }
            
            payload = {
                "attachments": [{
                    "color": color_map.get(alert.level, "good"),
                    "title": f"Alert: {alert.component}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Level", "value": alert.level.value, "short": True},
                        {"title": "Component", "value": alert.component, "short": True},
                        {"title": "Timestamp", "value": alert.timestamp.isoformat(), "short": False}
                    ],
                    "footer": "Smart Traffic Management System",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code != 200:
                self.logger.warning(f"Slack alert failed: {response.status_code}")
        
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {e}")
    
    async def _cleanup_old_alerts(self):
        """Clean up old resolved alerts"""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # Remove old resolved alerts from active alerts
        to_remove = []
        for alert_id, alert in self.active_alerts.items():
            if alert.resolved and alert.resolved_at and alert.resolved_at < cutoff_time:
                to_remove.append(alert_id)
        
        for alert_id in to_remove:
            del self.active_alerts[alert_id]
        
        # Keep only last 1000 alerts in history
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]
    
    async def _metrics_cleanup_loop(self):
        """Clean up old metrics data"""
        while self.running:
            try:
                if self.redis_client:
                    # Clean up old metrics keys
                    keys = self.redis_client.keys("monitoring:*")
                    for key in keys:
                        ttl = self.redis_client.ttl(key)
                        if ttl == -1:  # No expiration set
                            self.redis_client.expire(key, 3600)  # Set 1 hour expiration
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Metrics cleanup error: {e}")
                await asyncio.sleep(3600)
    
    def register_alert_handler(self, handler: Callable):
        """Register an alert handler"""
        self.alert_handlers.append(handler)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": self.metrics["system"],
            "component_status": {name: status.value for name, status in self.component_status.items()},
            "active_alerts": len(self.active_alerts),
            "alert_breakdown": {
                level.value: len([a for a in self.active_alerts.values() if a.level == level])
                for level in AlertLevel
            },
            "health_checks": [
                {
                    "name": hc.name,
                    "last_check": hc.last_check.isoformat() if hc.last_check else None,
                    "last_status": hc.last_status.value if hc.last_status else None,
                    "consecutive_failures": hc.consecutive_failures
                }
                for hc in self.health_checks
            ]
        }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            "system": self.metrics["system"],
            "components": self.metrics["components"],
            "performance": self.metrics["performance"],
            "custom": self.metrics["custom"]
        }

# Example usage
async def main():
    config = {
        "monitoring": {
            "prometheus": {"port": 9090},
            "alert_thresholds": {
                "cpu_usage": 80,
                "memory_usage": 85,
                "disk_usage": 90
            }
        },
        "components": {
            "backend": {"port": 8000, "health_check_url": "http://localhost:8000/health"},
            "frontend": {"port": 3000, "health_check_url": "http://localhost:3000"}
        }
    }
    
    monitoring = MonitoringSystem(config)
    await monitoring.start()

if __name__ == "__main__":
    asyncio.run(main())
