#!/usr/bin/env python3
"""
SUMO Integration Main Execution Script
Run SUMO simulation with full integration to backend API
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from sumo_integration_manager import SumoIntegrationManager, IntegrationConfig
from config.sumo_config import get_sumo_config


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/sumo_integration.log')
        ]
    )


def create_directories():
    """Create necessary directories"""
    directories = [
        'logs',
        'output',
        'data',
        'config',
        'networks',
        'routes',
        'additional',
        'scenarios'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='SUMO Integration System')
    parser.add_argument('--scenario', '-s', type=str, default='scenarios/basic_scenario.sumocfg',
                       help='SUMO scenario configuration file')
    parser.add_argument('--config', '-c', type=str, default='config/sumo_config.yaml',
                       help='Integration configuration file')
    parser.add_argument('--log-level', '-l', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--enable-gui', action='store_true',
                       help='Enable SUMO GUI')
    parser.add_argument('--disable-validation', action='store_true',
                       help='Disable validation')
    parser.add_argument('--disable-export', action='store_true',
                       help='Disable data export')
    parser.add_argument('--disable-visualization', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create directories
    create_directories()
    
    logger.info("Starting SUMO Integration System")
    logger.info(f"Scenario: {args.scenario}")
    logger.info(f"Config: {args.config}")
    logger.info(f"Log Level: {args.log_level}")
    
    try:
        # Load configuration
        sumo_config = get_sumo_config()
        if os.path.exists(args.config):
            from config.sumo_config import load_sumo_config
            sumo_config = load_sumo_config(args.config)
        
        # Create integration configuration
        integration_config = IntegrationConfig(
            sumo_config=sumo_config,
            enable_visualization=not args.disable_visualization,
            enable_validation=not args.disable_validation,
            enable_data_export=not args.disable_export,
            enable_scenario_management=True,
            log_level=args.log_level,
            auto_restart=True
        )
        
        # Create integration manager
        manager = SumoIntegrationManager(integration_config)
        
        # Start integration
        success = await manager.start_integration(args.scenario)
        
        if success:
            logger.info("SUMO integration started successfully")
            
            try:
                # Keep running until interrupted
                while True:
                    await asyncio.sleep(1)
                    
                    # Check status
                    status = manager.get_integration_status()
                    if status['state'] == 'error':
                        logger.error("Integration error detected")
                        break
                    
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
            
            finally:
                # Stop integration
                manager.stop_integration()
                logger.info("SUMO integration stopped")
        
        else:
            logger.error("Failed to start SUMO integration")
            return 1
    
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nReceived interrupt signal")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
