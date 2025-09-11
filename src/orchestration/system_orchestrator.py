#!/usr/bin/env python3
"""
Smart Traffic Management System Orchestrator
Coordinates all system components for end-to-end functionality
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
import psutil
import requests
from pathlib import Path

# System component imports
sys.path.append(str(Path(__file__).parent.parent))
from backend.main import create_app as create_backend_app
from ml_engine.signal_optimizer import SignalOptimizer
from simulation.sumo_integration.sumo_integration_manager import SumoIntegrationManager
from frontend.smart-traffic-ui.src.main import create_app as create_frontend_app

class ComponentStatus(Enum):
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"

@dataclass
class ComponentInfo:
    name: str
    status: ComponentStatus
    port: int
    pid: Optional[int] = None
    health_check_url: Optional[str] = None
    last_health_check: Optional[datetime] = None
    restart_count: int = 0
    max_restarts: int = 5
    dependencies: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

class SystemOrchestrator:
    """Main orchestrator for the Smart Traffic Management System"""
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.components: Dict[str, ComponentInfo] = {}
        self.running = False
        self.logger = self._setup_logging()
        self.event_loop = None
        self.health_check_interval = 30
        self.monitoring_data = {
            "system_start_time": datetime.now(),
            "total_requests": 0,
            "errors": 0,
            "component_restarts": 0
        }
        
        # Initialize components
        self._initialize_components()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load system configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default system configuration"""
        return {
            "system": {
                "name": "Smart Traffic Management System",
                "version": "2.1.0",
                "environment": "development",
                "log_level": "INFO"
            },
            "components": {
                "backend": {
                    "port": 8000,
                    "health_check_url": "http://localhost:8000/health",
                    "max_restarts": 5,
                    "dependencies": []
                },
                "ml_optimizer": {
                    "port": 8001,
                    "health_check_url": "http://localhost:8001/health",
                    "max_restarts": 3,
                    "dependencies": ["backend"]
                },
                "sumo_simulation": {
                    "port": 8002,
                    "health_check_url": "http://localhost:8002/health",
                    "max_restarts": 5,
                    "dependencies": ["backend"]
                },
                "frontend": {
                    "port": 3000,
                    "health_check_url": "http://localhost:3000",
                    "max_restarts": 3,
                    "dependencies": ["backend"]
                }
            },
            "monitoring": {
                "health_check_interval": 30,
                "metrics_collection_interval": 60,
                "alert_thresholds": {
                    "cpu_usage": 80,
                    "memory_usage": 85,
                    "response_time": 5.0
                }
            },
            "data_flow": {
                "sumo_to_backend_interval": 1,
                "ml_optimization_interval": 30,
                "frontend_refresh_interval": 5
            }
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup system logging"""
        logging.basicConfig(
            level=getattr(logging, self.config.get("system", {}).get("log_level", "INFO")),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/system_orchestrator.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    def _initialize_components(self):
        """Initialize all system components"""
        components_config = self.config.get("components", {})
        
        for name, config in components_config.items():
            self.components[name] = ComponentInfo(
                name=name,
                status=ComponentStatus.STOPPED,
                port=config.get("port", 8000),
                health_check_url=config.get("health_check_url"),
                max_restarts=config.get("max_restarts", 5),
                dependencies=config.get("dependencies", []),
                config=config
            )
    
    async def start_system(self):
        """Start the entire system"""
        self.logger.info("Starting Smart Traffic Management System...")
        self.running = True
        
        try:
            # Start components in dependency order
            await self._start_components_in_order()
            
            # Start monitoring tasks
            await self._start_monitoring_tasks()
            
            # Start data flow coordination
            await self._start_data_flow_coordination()
            
            self.logger.info("System started successfully")
            
            # Keep running until shutdown
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            await self.stop_system()
            raise
    
    async def _start_components_in_order(self):
        """Start components in dependency order"""
        started_components = set()
        
        while len(started_components) < len(self.components):
            progress_made = False
            
            for name, component in self.components.items():
                if name in started_components:
                    continue
                
                # Check if all dependencies are started
                if all(dep in started_components for dep in component.dependencies):
                    try:
                        await self._start_component(component)
                        started_components.add(name)
                        progress_made = True
                        self.logger.info(f"Started component: {name}")
                    except Exception as e:
                        self.logger.error(f"Failed to start component {name}: {e}")
                        component.status = ComponentStatus.ERROR
            
            if not progress_made:
                # Circular dependency or startup failure
                failed_components = [name for name, comp in self.components.items() 
                                   if comp.status == ComponentStatus.ERROR]
                raise Exception(f"Failed to start components: {failed_components}")
    
    async def _start_component(self, component: ComponentInfo):
        """Start a single component"""
        component.status = ComponentStatus.STARTING
        
        try:
            if component.name == "backend":
                await self._start_backend(component)
            elif component.name == "ml_optimizer":
                await self._start_ml_optimizer(component)
            elif component.name == "sumo_simulation":
                await self._start_sumo_simulation(component)
            elif component.name == "frontend":
                await self._start_frontend(component)
            else:
                raise Exception(f"Unknown component: {component.name}")
            
            component.status = ComponentStatus.RUNNING
            self.logger.info(f"Component {component.name} started on port {component.port}")
            
        except Exception as e:
            component.status = ComponentStatus.ERROR
            raise Exception(f"Failed to start {component.name}: {e}")
    
    async def _start_backend(self, component: ComponentInfo):
        """Start the backend API service"""
        import subprocess
        import os
        
        # Start backend in a subprocess
        env = os.environ.copy()
        env["PORT"] = str(component.port)
        
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.main:app", 
             "--host", "0.0.0.0", "--port", str(component.port)],
            env=env,
            cwd=Path(__file__).parent.parent
        )
        
        component.pid = process.pid
        
        # Wait for health check to pass
        await self._wait_for_health_check(component)
    
    async def _start_ml_optimizer(self, component: ComponentInfo):
        """Start the ML optimizer service"""
        import subprocess
        import os
        
        env = os.environ.copy()
        env["PORT"] = str(component.port)
        
        process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "ml_engine.main:app", 
             "--host", "0.0.0.0", "--port", str(component.port)],
            env=env,
            cwd=Path(__file__).parent.parent
        )
        
        component.pid = process.pid
        await self._wait_for_health_check(component)
    
    async def _start_sumo_simulation(self, component: ComponentInfo):
        """Start the SUMO simulation service"""
        import subprocess
        import os
        
        env = os.environ.copy()
        env["PORT"] = str(component.port)
        
        process = subprocess.Popen(
            [sys.executable, "simulation/sumo_integration/run_sumo_integration.py"],
            env=env,
            cwd=Path(__file__).parent.parent
        )
        
        component.pid = process.pid
        await self._wait_for_health_check(component)
    
    async def _start_frontend(self, component: ComponentInfo):
        """Start the frontend service"""
        import subprocess
        import os
        
        env = os.environ.copy()
        env["PORT"] = str(component.port)
        
        process = subprocess.Popen(
            ["npm", "run", "dev", "--", "--port", str(component.port)],
            env=env,
            cwd=Path(__file__).parent.parent / "frontend" / "smart-traffic-ui"
        )
        
        component.pid = process.pid
        await self._wait_for_health_check(component)
    
    async def _wait_for_health_check(self, component: ComponentInfo, timeout: int = 60):
        """Wait for component health check to pass"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if component.health_check_url:
                    response = requests.get(component.health_check_url, timeout=5)
                    if response.status_code == 200:
                        component.last_health_check = datetime.now()
                        return
            except Exception:
                pass
            
            await asyncio.sleep(2)
        
        raise Exception(f"Health check timeout for {component.name}")
    
    async def _start_monitoring_tasks(self):
        """Start system monitoring tasks"""
        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._metrics_collection_loop())
        asyncio.create_task(self._alert_monitoring_loop())
    
    async def _health_check_loop(self):
        """Continuous health checking of all components"""
        while self.running:
            for component in self.components.values():
                if component.status == ComponentStatus.RUNNING:
                    await self._check_component_health(component)
            
            await asyncio.sleep(self.health_check_interval)
    
    async def _check_component_health(self, component: ComponentInfo):
        """Check health of a single component"""
        try:
            if component.health_check_url:
                response = requests.get(component.health_check_url, timeout=5)
                if response.status_code == 200:
                    component.status = ComponentStatus.HEALTHY
                    component.last_health_check = datetime.now()
                else:
                    component.status = ComponentStatus.UNHEALTHY
            else:
                # Check if process is still running
                if component.pid and psutil.pid_exists(component.pid):
                    component.status = ComponentStatus.HEALTHY
                else:
                    component.status = ComponentStatus.UNHEALTHY
                    
        except Exception as e:
            self.logger.warning(f"Health check failed for {component.name}: {e}")
            component.status = ComponentStatus.UNHEALTHY
        
        # Restart unhealthy components
        if component.status == ComponentStatus.UNHEALTHY:
            await self._restart_component(component)
    
    async def _restart_component(self, component: ComponentInfo):
        """Restart an unhealthy component"""
        if component.restart_count >= component.max_restarts:
            self.logger.error(f"Max restarts exceeded for {component.name}")
            component.status = ComponentStatus.ERROR
            return
        
        self.logger.warning(f"Restarting component: {component.name}")
        component.restart_count += 1
        self.monitoring_data["component_restarts"] += 1
        
        try:
            await self._stop_component(component)
            await asyncio.sleep(5)  # Wait before restart
            await self._start_component(component)
        except Exception as e:
            self.logger.error(f"Failed to restart {component.name}: {e}")
            component.status = ComponentStatus.ERROR
    
    async def _metrics_collection_loop(self):
        """Collect system metrics"""
        while self.running:
            try:
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "system_uptime": (datetime.now() - self.monitoring_data["system_start_time"]).total_seconds(),
                    "components": {}
                }
                
                for name, component in self.components.items():
                    metrics["components"][name] = {
                        "status": component.status.value,
                        "restart_count": component.restart_count,
                        "last_health_check": component.last_health_check.isoformat() if component.last_health_check else None
                    }
                
                # Save metrics
                with open("logs/system_metrics.json", "a") as f:
                    f.write(json.dumps(metrics) + "\n")
                
            except Exception as e:
                self.logger.error(f"Metrics collection failed: {e}")
            
            await asyncio.sleep(self.config.get("monitoring", {}).get("metrics_collection_interval", 60))
    
    async def _alert_monitoring_loop(self):
        """Monitor system for alerts"""
        while self.running:
            try:
                # Check system resources
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                thresholds = self.config.get("monitoring", {}).get("alert_thresholds", {})
                
                if cpu_percent > thresholds.get("cpu_usage", 80):
                    self.logger.warning(f"High CPU usage: {cpu_percent}%")
                
                if memory_percent > thresholds.get("memory_usage", 85):
                    self.logger.warning(f"High memory usage: {memory_percent}%")
                
                # Check component health
                unhealthy_components = [name for name, comp in self.components.items() 
                                      if comp.status == ComponentStatus.UNHEALTHY]
                if unhealthy_components:
                    self.logger.warning(f"Unhealthy components: {unhealthy_components}")
                
            except Exception as e:
                self.logger.error(f"Alert monitoring failed: {e}")
            
            await asyncio.sleep(30)
    
    async def _start_data_flow_coordination(self):
        """Start data flow coordination between components"""
        asyncio.create_task(self._sumo_to_backend_data_flow())
        asyncio.create_task(self._ml_optimization_loop())
        asyncio.create_task(self._frontend_data_refresh())
    
    async def _sumo_to_backend_data_flow(self):
        """Coordinate data flow from SUMO to backend"""
        while self.running:
            try:
                # This would coordinate real-time data flow
                # Implementation depends on your specific data flow requirements
                pass
            except Exception as e:
                self.logger.error(f"SUMO data flow failed: {e}")
            
            await asyncio.sleep(self.config.get("data_flow", {}).get("sumo_to_backend_interval", 1))
    
    async def _ml_optimization_loop(self):
        """Run ML optimization cycles"""
        while self.running:
            try:
                # Trigger ML optimization
                # Implementation depends on your ML optimizer
                pass
            except Exception as e:
                self.logger.error(f"ML optimization failed: {e}")
            
            await asyncio.sleep(self.config.get("data_flow", {}).get("ml_optimization_interval", 30))
    
    async def _frontend_data_refresh(self):
        """Coordinate frontend data refresh"""
        while self.running:
            try:
                # Trigger frontend data refresh
                # Implementation depends on your frontend requirements
                pass
            except Exception as e:
                self.logger.error(f"Frontend refresh failed: {e}")
            
            await asyncio.sleep(self.config.get("data_flow", {}).get("frontend_refresh_interval", 5))
    
    async def stop_system(self):
        """Stop the entire system"""
        self.logger.info("Stopping Smart Traffic Management System...")
        self.running = False
        
        # Stop all components
        for component in self.components.values():
            await self._stop_component(component)
        
        self.logger.info("System stopped")
    
    async def _stop_component(self, component: ComponentInfo):
        """Stop a single component"""
        if component.pid and psutil.pid_exists(component.pid):
            try:
                process = psutil.Process(component.pid)
                process.terminate()
                process.wait(timeout=10)
            except Exception as e:
                self.logger.warning(f"Failed to stop {component.name}: {e}")
        
        component.status = ComponentStatus.STOPPED
        component.pid = None
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "system": {
                "name": self.config.get("system", {}).get("name"),
                "version": self.config.get("system", {}).get("version"),
                "uptime": (datetime.now() - self.monitoring_data["system_start_time"]).total_seconds(),
                "running": self.running
            },
            "components": {
                name: {
                    "status": comp.status.value,
                    "port": comp.port,
                    "pid": comp.pid,
                    "restart_count": comp.restart_count,
                    "last_health_check": comp.last_health_check.isoformat() if comp.last_health_check else None
                }
                for name, comp in self.components.items()
            },
            "monitoring": self.monitoring_data
        }

async def main():
    """Main entry point"""
    orchestrator = SystemOrchestrator()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        asyncio.create_task(orchestrator.stop_system())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await orchestrator.start_system()
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        print(f"System error: {e}")
        sys.exit(1)
    finally:
        await orchestrator.stop_system()

if __name__ == "__main__":
    asyncio.run(main())
