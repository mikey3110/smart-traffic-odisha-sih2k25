#!/usr/bin/env python3
"""
Smart Traffic Management System Startup Script
Main entry point for starting the entire system
"""

import asyncio
import argparse
import logging
import sys
import signal
from pathlib import Path
from typing import Optional

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.orchestration.system_orchestrator import SystemOrchestrator
from src.orchestration.data_flow_manager import DataFlowManager
from src.orchestration.monitoring_system import MonitoringSystem
from src.admin.admin_interface import AdminInterface

class SmartTrafficSystem:
    """Main system coordinator"""
    
    def __init__(self, config_path: str = "config/system_config.yaml"):
        self.config_path = config_path
        self.logger = self._setup_logging()
        self.running = False
        
        # Initialize system components
        self.orchestrator = SystemOrchestrator(config_path)
        self.data_flow_manager = None
        self.monitoring_system = None
        self.admin_interface = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup system logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/system_startup.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
    
    async def start(self, enable_monitoring: bool = True, enable_admin: bool = True):
        """Start the entire system"""
        self.logger.info("Starting Smart Traffic Management System...")
        self.running = True
        
        try:
            # Start orchestrator
            self.logger.info("Starting System Orchestrator...")
            await self.orchestrator.start_system()
            
            # Start data flow manager
            self.logger.info("Starting Data Flow Manager...")
            self.data_flow_manager = DataFlowManager(self.orchestrator.config)
            asyncio.create_task(self.data_flow_manager.start())
            
            # Start monitoring system
            if enable_monitoring:
                self.logger.info("Starting Monitoring System...")
                self.monitoring_system = MonitoringSystem(self.orchestrator.config)
                asyncio.create_task(self.monitoring_system.start())
            
            # Start admin interface
            if enable_admin:
                self.logger.info("Starting Admin Interface...")
                self.admin_interface = AdminInterface(self.config_path)
                asyncio.create_task(self.admin_interface.start())
            
            self.logger.info("Smart Traffic Management System started successfully!")
            self.logger.info("System components:")
            self.logger.info("- Backend API: http://localhost:8000")
            self.logger.info("- ML Optimizer: http://localhost:8001")
            self.logger.info("- SUMO Simulation: http://localhost:8002")
            self.logger.info("- Frontend Dashboard: http://localhost:3000")
            self.logger.info("- Admin Interface: http://localhost:8003")
            self.logger.info("- Prometheus: http://localhost:9090")
            self.logger.info("- Grafana: http://localhost:3001")
            
            # Keep running until shutdown
            while self.running:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Shutdown requested by user")
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the entire system"""
        self.logger.info("Stopping Smart Traffic Management System...")
        self.running = False
        
        # Stop components in reverse order
        if self.admin_interface:
            self.logger.info("Stopping Admin Interface...")
            # Admin interface will stop automatically
        
        if self.monitoring_system:
            self.logger.info("Stopping Monitoring System...")
            await self.monitoring_system.stop()
        
        if self.data_flow_manager:
            self.logger.info("Stopping Data Flow Manager...")
            await self.data_flow_manager.stop()
        
        if self.orchestrator:
            self.logger.info("Stopping System Orchestrator...")
            await self.orchestrator.stop_system()
        
        self.logger.info("Smart Traffic Management System stopped")
    
    def get_system_status(self) -> dict:
        """Get current system status"""
        status = {
            "running": self.running,
            "components": {}
        }
        
        if self.orchestrator:
            status["components"]["orchestrator"] = self.orchestrator.get_system_status()
        
        if self.data_flow_manager:
            status["components"]["data_flow_manager"] = self.data_flow_manager.get_metrics()
        
        if self.monitoring_system:
            status["components"]["monitoring_system"] = self.monitoring_system.get_system_status()
        
        if self.admin_interface:
            status["components"]["admin_interface"] = self.admin_interface.get_system_status()
        
        return status

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print(f"\nReceived signal {signum}, shutting down...")
    sys.exit(0)

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Smart Traffic Management System")
    parser.add_argument(
        "--config", 
        default="config/system_config.yaml",
        help="Path to system configuration file"
    )
    parser.add_argument(
        "--no-monitoring",
        action="store_true",
        help="Disable monitoring system"
    )
    parser.add_argument(
        "--no-admin",
        action="store_true",
        help="Disable admin interface"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode"
    )
    
    args = parser.parse_args()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create necessary directories
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("backups").mkdir(exist_ok=True)
    
    # Initialize and start system
    system = SmartTrafficSystem(args.config)
    
    try:
        await system.start(
            enable_monitoring=not args.no_monitoring,
            enable_admin=not args.no_admin
        )
    except Exception as e:
        print(f"System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())