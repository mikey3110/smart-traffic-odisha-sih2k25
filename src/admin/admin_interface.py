#!/usr/bin/env python3
"""
Admin Interface for Smart Traffic Management System
Provides system management, monitoring, and configuration capabilities
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import psutil
from pathlib import Path
import yaml

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn

class SystemStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class ComponentType(Enum):
    BACKEND = "backend"
    ML_OPTIMIZER = "ml_optimizer"
    SUMO_SIMULATION = "sumo_simulation"
    FRONTEND = "frontend"
    DATABASE = "database"
    REDIS = "redis"

@dataclass
class ComponentInfo:
    name: str
    type: ComponentType
    status: SystemStatus
    port: int
    pid: Optional[int] = None
    uptime: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    last_health_check: Optional[datetime] = None
    restart_count: int = 0
    error_message: Optional[str] = None

@dataclass
class SystemMetrics:
    timestamp: datetime
    total_requests: int
    active_connections: int
    error_rate: float
    response_time: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]

class AdminInterface:
    """Admin interface for system management"""
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
        # System components
        self.components: Dict[str, ComponentInfo] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.alerts: List[Dict[str, Any]] = []
        
        # API client
        self.api_base_url = "http://localhost:8000"
        
        self._initialize_components()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {self.config_path} not found, using defaults")
            return {}
    
    def _initialize_components(self):
        """Initialize system components"""
        components_config = self.config.get("components", {})
        
        for name, config in components_config.items():
            component_type = ComponentType(name) if name in [e.value for e in ComponentType] else ComponentType.BACKEND
            self.components[name] = ComponentInfo(
                name=name,
                type=component_type,
                status=SystemStatus.STOPPED,
                port=config.get("port", 8000)
            )
    
    async def start(self):
        """Start the admin interface"""
        self.logger.info("Starting Admin Interface...")
        
        # Start monitoring tasks
        asyncio.create_task(self._monitor_components())
        asyncio.create_task(self._collect_metrics())
        asyncio.create_task(self._check_alerts())
        
        self.logger.info("Admin Interface started")
    
    async def _monitor_components(self):
        """Monitor system components"""
        while True:
            try:
                for name, component in self.components.items():
                    await self._check_component_health(component)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Component monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _check_component_health(self, component: ComponentInfo):
        """Check health of a single component"""
        try:
            # Check if process is running
            if component.pid and psutil.pid_exists(component.pid):
                process = psutil.Process(component.pid)
                component.uptime = time.time() - process.create_time()
                component.memory_usage = process.memory_percent()
                component.cpu_usage = process.cpu_percent()
                component.status = SystemStatus.RUNNING
            else:
                component.status = SystemStatus.STOPPED
                component.uptime = None
                component.memory_usage = None
                component.cpu_usage = None
            
            # Check API health if available
            if component.type in [ComponentType.BACKEND, ComponentType.ML_OPTIMIZER, ComponentType.SUMO_SIMULATION]:
                try:
                    response = requests.get(f"http://localhost:{component.port}/health", timeout=5)
                    if response.status_code == 200:
                        component.status = SystemStatus.RUNNING
                    else:
                        component.status = SystemStatus.ERROR
                        component.error_message = f"HTTP {response.status_code}"
                except Exception as e:
                    component.status = SystemStatus.ERROR
                    component.error_message = str(e)
            
            component.last_health_check = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Health check failed for {component.name}: {e}")
            component.status = SystemStatus.ERROR
            component.error_message = str(e)
    
    async def _collect_metrics(self):
        """Collect system metrics"""
        while True:
            try:
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                # API metrics (if available)
                total_requests = 0
                active_connections = 0
                error_rate = 0.0
                response_time = 0.0
                
                try:
                    response = requests.get(f"{self.api_base_url}/metrics", timeout=5)
                    if response.status_code == 200:
                        metrics_data = response.json()
                        total_requests = metrics_data.get("total_requests", 0)
                        active_connections = metrics_data.get("active_connections", 0)
                        error_rate = metrics_data.get("error_rate", 0.0)
                        response_time = metrics_data.get("response_time", 0.0)
                except Exception:
                    pass
                
                metrics = SystemMetrics(
                    timestamp=datetime.now(),
                    total_requests=total_requests,
                    active_connections=active_connections,
                    error_rate=error_rate,
                    response_time=response_time,
                    cpu_usage=cpu_percent,
                    memory_usage=memory.percent,
                    disk_usage=(disk.used / disk.total) * 100,
                    network_io={
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv
                    }
                )
                
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                await asyncio.sleep(60)  # Collect every minute
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(60)
    
    async def _check_alerts(self):
        """Check for system alerts"""
        while True:
            try:
                # Check component status
                for component in self.components.values():
                    if component.status == SystemStatus.ERROR:
                        await self._create_alert(
                            level="error",
                            component=component.name,
                            message=f"Component {component.name} is in error state",
                            details=component.error_message
                        )
                    elif component.status == SystemStatus.STOPPED:
                        await self._create_alert(
                            level="warning",
                            component=component.name,
                            message=f"Component {component.name} is stopped",
                            details="Component is not running"
                        )
                
                # Check system resources
                if self.metrics_history:
                    latest_metrics = self.metrics_history[-1]
                    
                    if latest_metrics.cpu_usage > 90:
                        await self._create_alert(
                            level="critical",
                            component="system",
                            message="High CPU usage",
                            details=f"CPU usage is {latest_metrics.cpu_usage:.1f}%"
                        )
                    
                    if latest_metrics.memory_usage > 90:
                        await self._create_alert(
                            level="critical",
                            component="system",
                            message="High memory usage",
                            details=f"Memory usage is {latest_metrics.memory_usage:.1f}%"
                        )
                    
                    if latest_metrics.disk_usage > 90:
                        await self._create_alert(
                            level="critical",
                            component="system",
                            message="High disk usage",
                            details=f"Disk usage is {latest_metrics.disk_usage:.1f}%"
                        )
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Alert checking error: {e}")
                await asyncio.sleep(60)
    
    async def _create_alert(self, level: str, component: str, message: str, details: str = None):
        """Create a new alert"""
        alert = {
            "id": f"{component}_{level}_{int(time.time())}",
            "level": level,
            "component": component,
            "message": message,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "resolved": False
        }
        
        # Check if similar alert already exists
        existing_alert = None
        for existing in self.alerts:
            if (existing["component"] == component and 
                existing["level"] == level and 
                existing["message"] == message and 
                not existing["resolved"]):
                existing_alert = existing
                break
        
        if not existing_alert:
            self.alerts.append(alert)
            self.logger.warning(f"Alert created: {alert['id']} - {message}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "timestamp": datetime.now().isoformat(),
            "components": {name: asdict(comp) for name, comp in self.components.items()},
            "system_metrics": asdict(self.metrics_history[-1]) if self.metrics_history else None,
            "active_alerts": len([a for a in self.alerts if not a["resolved"]]),
            "total_alerts": len(self.alerts)
        }
    
    def get_component_status(self, component_name: str) -> Dict[str, Any]:
        """Get status of a specific component"""
        if component_name not in self.components:
            raise HTTPException(status_code=404, detail="Component not found")
        
        return asdict(self.components[component_name])
    
    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get metrics history for the specified number of hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [asdict(m) for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_alerts(self, resolved: bool = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by resolved status"""
        if resolved is None:
            return self.alerts
        return [a for a in self.alerts if a["resolved"] == resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        for alert in self.alerts:
            if alert["id"] == alert_id:
                alert["resolved"] = True
                alert["resolved_at"] = datetime.now().isoformat()
                return True
        return False
    
    def restart_component(self, component_name: str) -> bool:
        """Restart a component"""
        if component_name not in self.components:
            return False
        
        component = self.components[component_name]
        
        try:
            # Kill existing process
            if component.pid and psutil.pid_exists(component.pid):
                process = psutil.Process(component.pid)
                process.terminate()
                process.wait(timeout=10)
            
            # Start new process (implementation depends on your setup)
            # This is a simplified example
            component.status = SystemStatus.RUNNING
            component.restart_count += 1
            component.error_message = None
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restart {component_name}: {e}")
            component.status = SystemStatus.ERROR
            component.error_message = str(e)
            return False
    
    def update_config(self, component_name: str, config: Dict[str, Any]) -> bool:
        """Update component configuration"""
        try:
            # Update configuration file
            with open(self.config_path, 'r') as f:
                full_config = yaml.safe_load(f)
            
            if "components" not in full_config:
                full_config["components"] = {}
            
            if component_name not in full_config["components"]:
                full_config["components"][component_name] = {}
            
            full_config["components"][component_name].update(config)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(full_config, f, default_flow_style=False)
            
            # Reload configuration
            self.config = full_config
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update config for {component_name}: {e}")
            return False

# FastAPI application
app = FastAPI(
    title="Smart Traffic Management Admin Interface",
    description="Admin interface for system management and monitoring",
    version="2.1.0"
)

# Global admin interface instance
admin_interface = AdminInterface()

# Security
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication (implement proper auth in production)"""
    if credentials.credentials != "admin-token":
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    return {"username": "admin"}

# API Routes
@app.get("/api/status")
async def get_system_status(current_user: dict = Depends(get_current_user)):
    """Get system status"""
    return admin_interface.get_system_status()

@app.get("/api/components/{component_name}")
async def get_component_status(component_name: str, current_user: dict = Depends(get_current_user)):
    """Get component status"""
    try:
        return admin_interface.get_component_status(component_name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics")
async def get_metrics(hours: int = 24, current_user: dict = Depends(get_current_user)):
    """Get metrics history"""
    return admin_interface.get_metrics_history(hours)

@app.get("/api/alerts")
async def get_alerts(resolved: bool = None, current_user: dict = Depends(get_current_user)):
    """Get alerts"""
    return admin_interface.get_alerts(resolved)

@app.post("/api/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str, current_user: dict = Depends(get_current_user)):
    """Resolve an alert"""
    if admin_interface.resolve_alert(alert_id):
        return {"message": "Alert resolved"}
    else:
        raise HTTPException(status_code=404, detail="Alert not found")

@app.post("/api/components/{component_name}/restart")
async def restart_component(component_name: str, current_user: dict = Depends(get_current_user)):
    """Restart a component"""
    if admin_interface.restart_component(component_name):
        return {"message": f"Component {component_name} restarted"}
    else:
        raise HTTPException(status_code=500, detail="Failed to restart component")

@app.put("/api/components/{component_name}/config")
async def update_component_config(
    component_name: str, 
    config: Dict[str, Any], 
    current_user: dict = Depends(get_current_user)
):
    """Update component configuration"""
    if admin_interface.update_config(component_name, config):
        return {"message": f"Configuration updated for {component_name}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to update configuration")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Static files and templates
app.mount("/static", StaticFiles(directory="admin/static"), name="static")
templates = Jinja2Templates(directory="admin/templates")

@app.get("/", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    """Admin dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/components", response_class=HTMLResponse)
async def components_page(request: Request):
    """Components management page"""
    return templates.TemplateResponse("components.html", {"request": request})

@app.get("/metrics", response_class=HTMLResponse)
async def metrics_page(request: Request):
    """Metrics page"""
    return templates.TemplateResponse("metrics.html", {"request": request})

@app.get("/alerts", response_class=HTMLResponse)
async def alerts_page(request: Request):
    """Alerts page"""
    return templates.TemplateResponse("alerts.html", {"request": request})

# Startup event
@app.on_event("startup")
async def startup_event():
    """Startup event"""
    await admin_interface.start()

# Main entry point
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
