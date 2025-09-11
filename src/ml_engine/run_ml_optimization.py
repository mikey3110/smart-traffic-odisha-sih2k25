"""
Main Execution Script for ML Traffic Signal Optimization
Comprehensive script to run the complete ML optimization system
"""

import asyncio
import logging
import argparse
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
import json

# Import ML components
from enhanced_signal_optimizer import EnhancedSignalOptimizer, OptimizationRequest, OptimizationMode
from data.enhanced_data_integration import EnhancedDataIntegration
from prediction.enhanced_traffic_predictor import EnhancedTrafficPredictor
from metrics.enhanced_performance_metrics import EnhancedPerformanceMetrics
from ab_testing.ab_testing_framework import ABTestingFramework, ABTestConfig, TestVariant, StatisticalTest
from monitoring.enhanced_monitoring import EnhancedMonitoring
from visualization.performance_visualizer import PerformanceVisualizer
from config.ml_config import get_config, load_config


class MLTrafficOptimizationSystem:
    """
    Main system class for ML Traffic Signal Optimization
    
    Features:
    - Complete ML optimization pipeline
    - Real-time monitoring and alerting
    - A/B testing capabilities
    - Performance visualization
    - Comprehensive logging
    - Graceful shutdown handling
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the ML optimization system"""
        # Load configuration
        if config_path:
            self.config = load_config(config_path)
        else:
            self.config = get_config()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.signal_optimizer = EnhancedSignalOptimizer(self.config)
        self.monitoring = EnhancedMonitoring(self.config.logging)
        self.visualizer = PerformanceVisualizer()
        
        # System state
        self.is_running = False
        self.intersection_ids = []
        self.optimization_interval = self.config.optimization_interval
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger.info("ML Traffic Optimization System initialized")
    
    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_config = self.config.logging
        
        # Create logs directory
        log_dir = Path(log_config.log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.log_level),
            format=log_config.log_format,
            handlers=[
                logging.StreamHandler(),
                logging.handlers.RotatingFileHandler(
                    log_config.log_file,
                    maxBytes=log_config.max_log_size,
                    backupCount=log_config.backup_count
                )
            ]
        )
        
        # Create specialized loggers
        self.traffic_logger = logging.getLogger('traffic_optimization')
        self.performance_logger = logging.getLogger('performance')
        self.system_logger = logging.getLogger('system')
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.is_running = False
    
    async def start(self, intersection_ids: list = None):
        """Start the ML optimization system"""
        if self.is_running:
            self.logger.warning("System is already running")
            return
        
        self.is_running = True
        self.intersection_ids = intersection_ids or ["junction-1", "junction-2", "junction-3"]
        
        self.logger.info("Starting ML Traffic Optimization System...")
        self.logger.info(f"Monitoring intersections: {self.intersection_ids}")
        self.logger.info(f"Optimization interval: {self.optimization_interval} seconds")
        
        try:
            # Start signal optimizer
            await self.signal_optimizer.start()
            
            # Start monitoring
            self.monitoring.start_monitoring()
            
            # Start optimization loop
            await self._optimization_loop()
            
        except Exception as e:
            self.logger.error(f"Error in optimization system: {e}")
            raise
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the ML optimization system"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping ML Traffic Optimization System...")
        
        # Stop signal optimizer
        await self.signal_optimizer.stop()
        
        # Stop monitoring
        self.monitoring.stop_monitoring()
        
        self.is_running = False
        self.logger.info("ML Traffic Optimization System stopped")
    
    async def _optimization_loop(self):
        """Main optimization loop"""
        self.logger.info("Starting optimization loop...")
        
        while self.is_running:
            try:
                # Optimize each intersection
                for intersection_id in self.intersection_ids:
                    await self._optimize_intersection(intersection_id)
                
                # Wait for next optimization cycle
                await asyncio.sleep(self.optimization_interval)
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _optimize_intersection(self, intersection_id: str):
        """Optimize a single intersection"""
        try:
            # Create optimization request
            request = OptimizationRequest(
                intersection_id=intersection_id,
                current_timings={'north_lane': 30, 'south_lane': 30, 'east_lane': 30, 'west_lane': 30},
                optimization_mode=OptimizationMode.ADAPTIVE
            )
            
            # Perform optimization
            response = await self.signal_optimizer.optimize_intersection(request)
            
            # Log optimization result
            self.traffic_logger.info(f"Optimized {intersection_id}: {response.algorithm_used} "
                                   f"(confidence: {response.confidence:.3f}, "
                                   f"processing time: {response.processing_time:.3f}s)")
            
            # Record performance metrics
            self.monitoring.record_optimization(
                intersection_id, response.algorithm_used,
                {
                    'wait_time': response.improvement_prediction.get('wait_time', 0),
                    'throughput': response.improvement_prediction.get('throughput', 0),
                    'efficiency': response.improvement_prediction.get('efficiency', 0),
                    'confidence': response.confidence
                },
                response.processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Error optimizing {intersection_id}: {e}")
            self.monitoring.record_error(
                "optimization_failure", f"Failed to optimize {intersection_id}",
                intersection_id, None, e
            )
    
    def create_ab_test(self, test_name: str, control_algorithm: str, 
                      treatment_algorithm: str, intersection_ids: list) -> str:
        """Create an A/B test"""
        self.logger.info(f"Creating A/B test: {test_name}")
        
        variants = [
            TestVariant(
                name="control",
                algorithm=control_algorithm,
                parameters={},
                traffic_split=0.5,
                is_control=True,
                description=f"Control group using {control_algorithm}"
            ),
            TestVariant(
                name="treatment",
                algorithm=treatment_algorithm,
                parameters={},
                traffic_split=0.5,
                description=f"Treatment group using {treatment_algorithm}"
            )
        ]
        
        test_config = ABTestConfig(
            test_id=f"test_{int(time.time())}",
            name=test_name,
            description=f"Compare {control_algorithm} vs {treatment_algorithm}",
            variants=variants,
            target_metrics=["wait_time", "throughput", "efficiency", "confidence"],
            statistical_test=StatisticalTest(test_type="t_test", alpha=0.05),
            duration_hours=24,
            min_sample_size=100
        )
        
        # Create test
        test_id = self.signal_optimizer.create_ab_test(test_config)
        
        # Start test for specified intersections
        self.signal_optimizer.start_ab_test(test_id, intersection_ids)
        
        self.logger.info(f"A/B test created: {test_id}")
        return test_id
    
    def get_system_status(self) -> dict:
        """Get comprehensive system status"""
        return {
            'is_running': self.is_running,
            'intersection_ids': self.intersection_ids,
            'optimization_interval': self.optimization_interval,
            'optimization_stats': self.signal_optimizer.get_optimization_statistics(),
            'performance_summary': self.monitoring.get_performance_summary(),
            'system_health': self.monitoring.get_system_health_summary(),
            'alerts': self.monitoring.get_alerts_summary(),
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_performance_report(self, output_dir: str = None) -> str:
        """Generate comprehensive performance report"""
        self.logger.info("Generating performance report...")
        
        # Get system data
        performance_data = self.monitoring.get_performance_summary()
        system_health = self.monitoring.get_system_health_summary()
        alerts = self.monitoring.get_alerts_summary()
        
        # Generate report
        report_path = self.visualizer.create_performance_report(
            performance_data, system_health, alerts, output_dir
        )
        
        self.logger.info(f"Performance report generated: {report_path}")
        return report_path
    
    def export_system_data(self, output_dir: str = None):
        """Export all system data"""
        if output_dir is None:
            output_dir = f"exports/export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting system data to {output_dir}")
        
        # Export optimization data
        self.signal_optimizer.export_optimization_data(
            str(output_path / "optimization_data.json")
        )
        
        # Export monitoring data
        self.monitoring.export_monitoring_data(
            str(output_path / "monitoring_data.json")
        )
        
        # Export system status
        status = self.get_system_status()
        with open(output_path / "system_status.json", 'w') as f:
            json.dump(status, f, indent=2, default=str)
        
        self.logger.info(f"System data exported to {output_dir}")


async def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='ML Traffic Signal Optimization System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--intersections', nargs='+', default=['junction-1', 'junction-2'],
                       help='Intersection IDs to monitor')
    parser.add_argument('--interval', type=int, default=10,
                       help='Optimization interval in seconds')
    parser.add_argument('--ab-test', action='store_true',
                       help='Create A/B test')
    parser.add_argument('--control-algorithm', default='websters_formula',
                       help='Control algorithm for A/B test')
    parser.add_argument('--treatment-algorithm', default='q_learning',
                       help='Treatment algorithm for A/B test')
    parser.add_argument('--report', action='store_true',
                       help='Generate performance report')
    parser.add_argument('--export', action='store_true',
                       help='Export system data')
    parser.add_argument('--status', action='store_true',
                       help='Show system status')
    
    args = parser.parse_args()
    
    # Initialize system
    system = MLTrafficOptimizationSystem(args.config)
    
    try:
        if args.status:
            # Show system status
            status = system.get_system_status()
            print(json.dumps(status, indent=2, default=str))
            return
        
        if args.report:
            # Generate performance report
            report_path = system.generate_performance_report()
            print(f"Performance report generated: {report_path}")
            return
        
        if args.export:
            # Export system data
            system.export_system_data()
            print("System data exported")
            return
        
        # Set optimization interval
        system.optimization_interval = args.interval
        
        # Create A/B test if requested
        if args.ab_test:
            test_id = system.create_ab_test(
                f"{args.control_algorithm}_vs_{args.treatment_algorithm}",
                args.control_algorithm,
                args.treatment_algorithm,
                args.intersections
            )
            print(f"A/B test created: {test_id}")
        
        # Start the system
        await system.start(args.intersections)
        
    except KeyboardInterrupt:
        print("\nReceived interrupt signal, shutting down...")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
