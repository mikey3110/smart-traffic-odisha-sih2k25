"""
Phase 3: ML Model Validation & Performance Analytics - Main Integration
Complete integration of all Phase 3 components for comprehensive validation and analytics

Features:
- Comprehensive model validation with A/B testing
- Real-time performance analytics and visualization
- ML monitoring with drift detection and explainable AI
- Advanced ML features (transfer learning, meta-learning, ensemble methods, RLHF)
- Detailed validation reports with statistical analysis
"""

import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import yaml
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.model_validator import ModelValidator, ValidationScenario, A/BTestResult
from analytics.performance_analytics import PerformanceAnalytics, MetricType
from monitoring.ml_monitoring import MLMonitoring, DriftType, AlertLevel
from advanced.advanced_ml_features import AdvancedMLFeatures, LearningType
from reports.validation_reports import ValidationReports, ValidationReport, ReportType


class Phase3Integration:
    """
    Main integration class for Phase 3: ML Model Validation & Performance Analytics
    
    This class orchestrates all Phase 3 components to provide comprehensive
    validation, analytics, and monitoring for ML traffic optimization systems.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/phase3_config.yaml"
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.model_validator = None
        self.performance_analytics = None
        self.ml_monitoring = None
        self.advanced_features = None
        self.validation_reports = None
        
        # Integration state
        self.is_running = False
        self.start_time = None
        
        # Validation results
        self.validation_results = {}
        self.ab_test_results = {}
        self.performance_metrics = {}
        
        self.logger.info("Phase 3 Integration initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                self.logger.info(f"Loaded configuration from {self.config_path}")
                return config
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                return self._get_default_config()
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'model_validator': {
                'test_duration': 3600,
                'num_iterations': 10,
                'cross_validation_folds': 5,
                'alpha': 0.05,
                'confidence_level': 0.95
            },
            'performance_analytics': {
                'collection_interval': 1.0,
                'buffer_size': 10000,
                'forecast_horizon': 60,
                'training_window': 1440
            },
            'ml_monitoring': {
                'drift_threshold': 0.1,
                'window_size': 1000,
                'min_samples': 100,
                'contamination': 0.1
            },
            'advanced_features': {
                'transfer_learning': {
                    'freeze_layers': 0.7,
                    'learning_rate_multiplier': 0.1,
                    'fine_tuning_epochs': 50
                },
                'meta_learning': {
                    'meta_learning_rate': 0.001,
                    'inner_learning_rate': 0.01,
                    'meta_epochs': 100,
                    'few_shot_samples': 10
                },
                'ensemble_methods': {
                    'ensemble_types': ['voting', 'stacking'],
                    'diversity_threshold': 0.1,
                    'performance_threshold': 0.7
                },
                'rlhf': {
                    'feedback_weight': 0.1,
                    'learning_rate_adjustment': 0.01,
                    'confidence_threshold': 0.7
                }
            },
            'validation_reports': {
                'output_dir': 'reports',
                'template_dir': 'templates',
                'statistical_analyzer': {
                    'alpha': 0.05,
                    'confidence_level': 0.95,
                    'min_sample_size': 30
                },
                'visualization': {
                    'output_dir': 'reports/visualizations',
                    'figure_size': [12, 8],
                    'dpi': 300
                }
            }
        }
    
    async def initialize(self):
        """Initialize all Phase 3 components"""
        try:
            self.logger.info("Initializing Phase 3 components...")
            
            # Initialize model validator
            self.model_validator = ModelValidator(
                self.config.get('model_validator', {})
            )
            
            # Initialize performance analytics
            self.performance_analytics = PerformanceAnalytics(
                self.config.get('performance_analytics', {})
            )
            
            # Initialize ML monitoring
            self.ml_monitoring = MLMonitoring(
                self.config.get('ml_monitoring', {})
            )
            
            # Initialize advanced features
            self.advanced_features = AdvancedMLFeatures(
                self.config.get('advanced_features', {})
            )
            
            # Initialize validation reports
            self.validation_reports = ValidationReports(
                self.config.get('validation_reports', {})
            )
            
            self.logger.info("Phase 3 components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Phase 3 components: {e}")
            return False
    
    async def start_validation_system(self):
        """Start the complete Phase 3 validation system"""
        try:
            if self.is_running:
                self.logger.warning("Validation system is already running")
                return False
            
            self.logger.info("Starting Phase 3 validation system...")
            
            # Start performance analytics
            self.performance_analytics.start_analytics()
            
            # Start ML monitoring
            self.ml_monitoring.start_monitoring()
            
            # Activate advanced features
            self.advanced_features.activate_features()
            
            self.is_running = True
            self.start_time = datetime.now()
            
            self.logger.info("Phase 3 validation system started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting validation system: {e}")
            return False
    
    async def stop_validation_system(self):
        """Stop the Phase 3 validation system"""
        try:
            if not self.is_running:
                self.logger.warning("Validation system is not running")
                return True
            
            self.logger.info("Stopping Phase 3 validation system...")
            
            # Stop performance analytics
            self.performance_analytics.stop_analytics()
            
            # Stop ML monitoring
            self.ml_monitoring.stop_monitoring()
            
            # Deactivate advanced features
            self.advanced_features.deactivate_features()
            
            self.is_running = False
            
            self.logger.info("Phase 3 validation system stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping validation system: {e}")
            return False
    
    async def run_comprehensive_validation(self, intersection_ids: List[str], 
                                         scenarios: List[ValidationScenario] = None) -> Dict[str, Any]:
        """Run comprehensive validation across all intersections and scenarios"""
        try:
            if not scenarios:
                scenarios = list(ValidationScenario)
            
            self.logger.info(f"Starting comprehensive validation for {len(intersection_ids)} intersections")
            
            validation_results = {}
            
            # Run A/B tests for each intersection and scenario
            for intersection_id in intersection_ids:
                intersection_results = {}
                
                for scenario in scenarios:
                    try:
                        # Run A/B test
                        ab_result = await self.model_validator.run_ab_test(
                            scenario, intersection_id, num_runs=10
                        )
                        
                        if ab_result:
                            intersection_results[scenario.value] = ab_result
                            
                            # Store in global results
                            key = f"{intersection_id}_{scenario.value}"
                            self.ab_test_results[key] = ab_result
                            
                    except Exception as e:
                        self.logger.error(f"Error in A/B test for {intersection_id} {scenario.value}: {e}")
                
                validation_results[intersection_id] = intersection_results
            
            # Generate validation reports
            await self._generate_validation_reports(validation_results)
            
            self.logger.info("Comprehensive validation completed")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive validation: {e}")
            return {}
    
    async def _generate_validation_reports(self, validation_results: Dict[str, Any]):
        """Generate validation reports from results"""
        try:
            # Prepare data for report generation
            report_data = {}
            
            for intersection_id, results in validation_results.items():
                baseline_data = {}
                optimized_data = {}
                
                for scenario, ab_result in results.items():
                    if ab_result.ml_results and ab_result.baseline_results:
                        # Extract metrics from results
                        for metric_name in ['wait_time_reduction', 'throughput_increase', 
                                          'fuel_consumption_reduction', 'emission_reduction']:
                            
                            ml_values = [r.metrics.get(metric_name, 0) for r in ab_result.ml_results]
                            baseline_values = [r.metrics.get(metric_name, 0) for r in ab_result.baseline_results]
                            
                            if ml_values and baseline_values:
                                if metric_name not in baseline_data:
                                    baseline_data[metric_name] = []
                                    optimized_data[metric_name] = []
                                
                                baseline_data[metric_name].extend(baseline_values)
                                optimized_data[metric_name].extend(ml_values)
                
                if baseline_data and optimized_data:
                    report_data[intersection_id] = {
                        'baseline_data': baseline_data,
                        'optimized_data': optimized_data,
                        'validation_period': (datetime.now() - timedelta(hours=1), datetime.now())
                    }
            
            # Generate reports
            if report_data:
                reports = self.validation_reports.generate_comprehensive_reports(report_data)
                self.validation_results.update(reports)
                
                self.logger.info(f"Generated validation reports for {len(reports)} intersections")
            
        except Exception as e:
            self.logger.error(f"Error generating validation reports: {e}")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary"""
        try:
            summary = {
                'is_running': self.is_running,
                'start_time': self.start_time.isoformat() if self.start_time else None,
                'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
                'total_ab_tests': len(self.ab_test_results),
                'total_validation_reports': len(self.validation_results),
                'performance_analytics': self.performance_analytics.get_analytics_summary() if self.performance_analytics else {},
                'ml_monitoring': self.ml_monitoring.get_monitoring_summary() if self.ml_monitoring else {},
                'advanced_features': self.advanced_features.get_feature_summary() if self.advanced_features else {},
                'validation_reports': self.validation_reports.get_report_summary() if self.validation_reports else {}
            }
            
            # Calculate overall improvements
            if self.ab_test_results:
                all_improvements = []
                for ab_result in self.ab_test_results.values():
                    for metric, improvement in ab_result.improvement_percentage.items():
                        all_improvements.append(improvement)
                
                if all_improvements:
                    summary['overall_improvements'] = {
                        'average': sum(all_improvements) / len(all_improvements),
                        'median': sorted(all_improvements)[len(all_improvements) // 2],
                        'max': max(all_improvements),
                        'min': min(all_improvements),
                        'above_target_30_percent': sum(1 for imp in all_improvements if imp > 30),
                        'meeting_target_15_30_percent': sum(1 for imp in all_improvements if 15 <= imp <= 30),
                        'below_target_15_percent': sum(1 for imp in all_improvements if imp < 15)
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting validation summary: {e}")
            return {'error': str(e)}
    
    def get_intersection_validation_report(self, intersection_id: str) -> Optional[ValidationReport]:
        """Get validation report for specific intersection"""
        return self.validation_reports.get_intersection_report(intersection_id) if self.validation_reports else None
    
    def get_performance_metrics(self, intersection_id: str) -> Dict[str, Any]:
        """Get performance metrics for intersection"""
        if not self.performance_analytics:
            return {}
        
        return self.performance_analytics.get_performance_report(intersection_id)
    
    def get_ml_monitoring_status(self, intersection_id: str) -> Dict[str, Any]:
        """Get ML monitoring status for intersection"""
        if not self.ml_monitoring:
            return {}
        
        return self.ml_monitoring.get_intersection_monitoring_report(intersection_id)
    
    def get_advanced_features_status(self, intersection_id: str) -> Dict[str, Any]:
        """Get advanced features status for intersection"""
        if not self.advanced_features:
            return {}
        
        return self.advanced_features.get_intersection_advanced_features(intersection_id)
    
    def export_validation_results(self, output_format: str = 'json') -> Dict[str, str]:
        """Export validation results"""
        try:
            if not self.validation_reports:
                return {}
            
            return self.validation_reports.export_reports(output_format)
            
        except Exception as e:
            self.logger.error(f"Error exporting validation results: {e}")
            return {}
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary of validation results"""
        try:
            summary = self.get_validation_summary()
            
            # Create executive summary content
            content = f"""
# Executive Summary - ML Traffic Optimization Validation

## Validation Overview
- **System Status**: {'Running' if summary['is_running'] else 'Stopped'}
- **Total A/B Tests**: {summary.get('total_ab_tests', 0)}
- **Validation Reports**: {summary.get('total_validation_reports', 0)}
- **Uptime**: {summary.get('uptime', 0):.1f} seconds

## Performance Improvements
"""
            
            if 'overall_improvements' in summary:
                improvements = summary['overall_improvements']
                content += f"""
- **Average Improvement**: {improvements['average']:.1f}%
- **Median Improvement**: {improvements['median']:.1f}%
- **Maximum Improvement**: {improvements['max']:.1f}%
- **Minimum Improvement**: {improvements['min']:.1f}%

## Target Achievement
- **Above Target (30%+)**: {improvements['above_target_30_percent']} metrics
- **Meeting Target (15-30%)**: {improvements['meeting_target_15_30_percent']} metrics
- **Below Target (<15%)**: {improvements['below_target_15_percent']} metrics
"""
            
            content += f"""
## Recommendations
- {'✅ ML optimization shows excellent performance and is ready for production deployment' if summary.get('overall_improvements', {}).get('average', 0) > 25 else '⚠️ ML optimization shows moderate performance - consider additional tuning' if summary.get('overall_improvements', {}).get('average', 0) > 15 else '❌ ML optimization shows limited performance - requires significant improvement'}
- Monitor system performance continuously
- Collect additional validation data for statistical significance
- Consider advanced ML features for further optimization

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            
            # Save executive summary
            summary_file = Path(self.config.get('validation_reports', {}).get('output_dir', 'reports')) / f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(summary_file, 'w') as f:
                f.write(content)
            
            return str(summary_file)
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            return ""


async def main():
    """Main function for Phase 3 integration"""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Phase 3: ML Model Validation & Performance Analytics")
    
    try:
        # Initialize Phase 3 integration
        phase3 = Phase3Integration()
        
        # Initialize components
        if not await phase3.initialize():
            logger.error("Failed to initialize Phase 3 components")
            return
        
        # Start validation system
        if not await phase3.start_validation_system():
            logger.error("Failed to start Phase 3 validation system")
            return
        
        logger.info("Phase 3 validation system is running. Press Ctrl+C to stop.")
        
        # Run comprehensive validation
        intersection_ids = ['intersection_1', 'intersection_2', 'intersection_3']
        validation_results = await phase3.run_comprehensive_validation(intersection_ids)
        
        logger.info(f"Validation completed for {len(validation_results)} intersections")
        
        # Generate executive summary
        summary_file = phase3.generate_executive_summary()
        logger.info(f"Executive summary generated: {summary_file}")
        
        # Keep system running
        try:
            while True:
                await asyncio.sleep(60)
                
                # Log system status
                summary = phase3.get_validation_summary()
                logger.info(f"System running - Uptime: {summary['uptime']:.1f}s, A/B Tests: {summary['total_ab_tests']}")
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping system...")
        
        # Stop system
        await phase3.stop_validation_system()
        logger.info("Phase 3 validation system stopped")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
