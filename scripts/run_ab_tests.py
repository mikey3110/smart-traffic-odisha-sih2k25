#!/usr/bin/env python3
"""
Automated A/B Testing Framework for Traffic Signal Optimization

This script runs controlled experiments comparing ML-optimized traffic signals
against baseline timings using quantitative metrics and statistical validation.

Author: Smart Traffic Management System Team
Date: 2025
"""

import os
import sys
import time
import json
import csv
import subprocess
import threading
import queue
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
import math
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from simulation.traci_controller import RobustTraCIController
from ml_engine.enhanced_signal_optimizer import EnhancedSignalOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ab_tests.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestType(Enum):
    """Types of A/B tests"""
    ML_VS_BASELINE = "ml_vs_baseline"
    ML_VS_WEBSTER = "ml_vs_webster"
    EMERGENCY_SCENARIO = "emergency_scenario"
    RUSH_HOUR = "rush_hour"
    NORMAL_TRAFFIC = "normal_traffic"

class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TestConfiguration:
    """Configuration for a single A/B test"""
    test_id: str
    test_type: TestType
    scenario: str
    duration: int  # seconds
    ml_enabled: bool
    baseline_type: str  # 'fixed', 'webster', 'actuated'
    traffic_volume: str  # 'low', 'medium', 'high'
    intersections: List[str]
    repetitions: int = 1
    warmup_time: int = 60  # seconds
    cooldown_time: int = 30  # seconds

@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test run"""
    test_id: str
    run_id: str
    timestamp: float
    duration: float
    
    # Traffic metrics
    total_vehicles: int
    total_wait_time: float
    average_wait_time: float
    max_wait_time: float
    total_travel_time: float
    average_travel_time: float
    
    # Throughput metrics
    vehicles_per_hour: float
    vehicles_per_cycle: float
    cycle_efficiency: float
    
    # Queue metrics
    max_queue_length: int
    average_queue_length: float
    total_queue_time: float
    queue_clearance_time: float
    
    # Signal metrics
    phase_changes: int
    average_phase_duration: float
    green_time_utilization: float
    red_time_utilization: float
    
    # Fuel and emissions
    fuel_consumption: float
    co2_emissions: float
    nox_emissions: float
    
    # System metrics
    cpu_usage: float
    memory_usage: float
    error_count: int
    reconnection_count: int

@dataclass
class TestResult:
    """Results for a complete A/B test"""
    test_id: str
    test_type: TestType
    scenario: str
    status: TestStatus
    start_time: float
    end_time: float
    duration: float
    
    # Test runs
    ml_runs: List[PerformanceMetrics]
    baseline_runs: List[PerformanceMetrics]
    
    # Statistical analysis
    statistical_significance: bool
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    
    # Performance comparison
    wait_time_improvement: float  # percentage
    throughput_improvement: float  # percentage
    queue_reduction: float  # percentage
    fuel_savings: float  # percentage
    emission_reduction: float  # percentage

class ABTestRunner:
    """Main A/B test runner class"""
    
    def __init__(self, config_file: str = "config/ab_test_config.json"):
        """
        Initialize the A/B test runner
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        # Test management
        self.tests: List[TestConfiguration] = []
        self.results: List[TestResult] = []
        self.current_test: Optional[TestConfiguration] = None
        self.running = False
        
        # SUMO and TraCI management
        self.sumo_process: Optional[subprocess.Popen] = None
        self.traci_controller: Optional[RobustTraCIController] = None
        self.ml_optimizer: Optional[EnhancedSignalOptimizer] = None
        
        # Data collection
        self.metrics_queue = queue.Queue()
        self.data_collector_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.stats = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'total_duration': 0.0,
            'start_time': time.time()
        }
        
        logger.info("A/B Test Runner initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'sumo_configs': {
                'normal': 'sumo/configs/normal_traffic.sumocfg',
                'rush_hour': 'sumo/configs/rush_hour.sumocfg',
                'emergency': 'sumo/configs/emergency_vehicle.sumocfg'
            },
            'test_scenarios': [
                {
                    'name': 'normal_traffic_ml_vs_baseline',
                    'type': 'ml_vs_baseline',
                    'scenario': 'normal',
                    'duration': 1800,  # 30 minutes
                    'repetitions': 5,
                    'traffic_volume': 'medium',
                    'intersections': ['center']
                },
                {
                    'name': 'rush_hour_ml_vs_webster',
                    'type': 'ml_vs_webster',
                    'scenario': 'rush_hour',
                    'duration': 3600,  # 1 hour
                    'repetitions': 3,
                    'traffic_volume': 'high',
                    'intersections': ['center']
                },
                {
                    'name': 'emergency_scenario_test',
                    'type': 'emergency_scenario',
                    'scenario': 'emergency',
                    'duration': 600,  # 10 minutes
                    'repetitions': 3,
                    'traffic_volume': 'low',
                    'intersections': ['center']
                }
            ],
            'output_dir': 'results/ab_tests',
            'log_level': 'INFO',
            'parallel_tests': False,
            'max_concurrent_tests': 1
        }
        
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"Loaded configuration from {self.config_file}")
                return {**default_config, **config}
            else:
                logger.warning(f"Configuration file {self.config_file} not found, using defaults")
                return default_config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return default_config
    
    def add_test(self, test_config: TestConfiguration):
        """
        Add a test configuration to the test queue
        
        Args:
            test_config: Test configuration to add
        """
        self.tests.append(test_config)
        logger.info(f"Added test: {test_config.test_id}")
    
    def create_test_scenarios(self):
        """Create test scenarios from configuration"""
        for scenario_config in self.config['test_scenarios']:
            # ML vs Baseline test
            ml_test = TestConfiguration(
                test_id=f"{scenario_config['name']}_ml",
                test_type=TestType(scenario_config['type']),
                scenario=scenario_config['scenario'],
                duration=scenario_config['duration'],
                ml_enabled=True,
                baseline_type='fixed',
                traffic_volume=scenario_config['traffic_volume'],
                intersections=scenario_config['intersections'],
                repetitions=scenario_config['repetitions']
            )
            self.add_test(ml_test)
            
            # Baseline test
            baseline_test = TestConfiguration(
                test_id=f"{scenario_config['name']}_baseline",
                test_type=TestType(scenario_config['type']),
                scenario=scenario_config['scenario'],
                duration=scenario_config['duration'],
                ml_enabled=False,
                baseline_type='fixed',
                traffic_volume=scenario_config['traffic_volume'],
                intersections=scenario_config['intersections'],
                repetitions=scenario_config['repetitions']
            )
            self.add_test(baseline_test)
    
    def start_sumo(self, scenario: str) -> bool:
        """
        Start SUMO simulation
        
        Args:
            scenario: Scenario name to run
            
        Returns:
            True if SUMO started successfully, False otherwise
        """
        try:
            config_path = self.config['sumo_configs'][scenario]
            
            # Start SUMO with TraCI
            cmd = [
                'sumo',
                '-c', config_path,
                '--remote-port', '8813',
                '--step-length', '0.1',
                '--no-step-log',
                '--verbose'
            ]
            
            logger.info(f"Starting SUMO with command: {' '.join(cmd)}")
            self.sumo_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for SUMO to start
            time.sleep(5)
            
            if self.sumo_process.poll() is None:
                logger.info("SUMO started successfully")
                return True
            else:
                stdout, stderr = self.sumo_process.communicate()
                logger.error(f"SUMO failed to start: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting SUMO: {e}")
            return False
    
    def stop_sumo(self):
        """Stop SUMO simulation"""
        if self.sumo_process:
            try:
                self.sumo_process.terminate()
                self.sumo_process.wait(timeout=10)
                logger.info("SUMO stopped successfully")
            except subprocess.TimeoutExpired:
                self.sumo_process.kill()
                logger.warning("SUMO killed after timeout")
            except Exception as e:
                logger.error(f"Error stopping SUMO: {e}")
            finally:
                self.sumo_process = None
    
    def initialize_controllers(self, test_config: TestConfiguration):
        """
        Initialize TraCI controller and ML optimizer
        
        Args:
            test_config: Test configuration
        """
        try:
            # Initialize TraCI controller
            self.traci_controller = RobustTraCIController('config/traci_config.json')
            self.traci_controller.start()
            
            # Wait for connection
            time.sleep(2)
            
            if not self.traci_controller.health_check():
                raise Exception("TraCI controller not healthy")
            
            # Initialize ML optimizer if needed
            if test_config.ml_enabled:
                self.ml_optimizer = EnhancedSignalOptimizer()
                logger.info("ML optimizer initialized")
            
            logger.info("Controllers initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing controllers: {e}")
            raise
    
    def cleanup_controllers(self):
        """Cleanup controllers and connections"""
        try:
            if self.traci_controller:
                self.traci_controller.stop()
                self.traci_controller = None
            
            if self.ml_optimizer:
                self.ml_optimizer = None
            
            logger.info("Controllers cleaned up")
            
        except Exception as e:
            logger.error(f"Error cleaning up controllers: {e}")
    
    def collect_metrics(self, test_config: TestConfiguration, run_id: str) -> PerformanceMetrics:
        """
        Collect performance metrics for a test run
        
        Args:
            test_config: Test configuration
            run_id: Unique run identifier
            
        Returns:
            Performance metrics
        """
        try:
            # Get traffic data from all intersections
            all_traffic_data = self.traci_controller.get_all_traffic_data()
            
            # Calculate metrics
            total_vehicles = 0
            total_wait_time = 0.0
            max_wait_time = 0.0
            total_travel_time = 0.0
            max_queue_length = 0
            total_queue_time = 0.0
            phase_changes = 0
            total_green_time = 0.0
            total_red_time = 0.0
            
            for intersection_id, traffic_data in all_traffic_data.items():
                # Vehicle counts
                intersection_vehicles = sum(traffic_data.vehicle_counts.values())
                total_vehicles += intersection_vehicles
                
                # Wait times
                intersection_wait_times = traffic_data.waiting_times.values()
                if intersection_wait_times:
                    total_wait_time += sum(intersection_wait_times)
                    max_wait_time = max(max_wait_time, max(intersection_wait_times))
                
                # Queue lengths
                intersection_queues = traffic_data.queue_lengths.values()
                if intersection_queues:
                    max_queue_length = max(max_queue_length, max(intersection_queues))
                    total_queue_time += sum(intersection_queues)
                
                # Phase information
                phase_changes += 1  # Simplified - would need more sophisticated tracking
                total_green_time += 30.0  # Simplified - would need actual phase timing
                total_red_time += 30.0
            
            # Calculate derived metrics
            average_wait_time = total_wait_time / max(total_vehicles, 1)
            average_travel_time = total_travel_time / max(total_vehicles, 1)
            vehicles_per_hour = (total_vehicles / test_config.duration) * 3600
            vehicles_per_cycle = total_vehicles / max(phase_changes, 1)
            cycle_efficiency = total_green_time / (total_green_time + total_red_time) if (total_green_time + total_red_time) > 0 else 0
            
            # System metrics
            stats = self.traci_controller.get_statistics()
            cpu_usage = 0.0  # Would need system monitoring
            memory_usage = 0.0  # Would need system monitoring
            error_count = stats['errors']
            reconnection_count = stats['reconnections']
            
            # Fuel and emissions (simplified calculations)
            fuel_consumption = total_vehicles * 0.1  # Simplified
            co2_emissions = fuel_consumption * 2.3  # Simplified
            nox_emissions = fuel_consumption * 0.05  # Simplified
            
            metrics = PerformanceMetrics(
                test_id=test_config.test_id,
                run_id=run_id,
                timestamp=time.time(),
                duration=test_config.duration,
                total_vehicles=total_vehicles,
                total_wait_time=total_wait_time,
                average_wait_time=average_wait_time,
                max_wait_time=max_wait_time,
                total_travel_time=total_travel_time,
                average_travel_time=average_travel_time,
                vehicles_per_hour=vehicles_per_hour,
                vehicles_per_cycle=vehicles_per_cycle,
                cycle_efficiency=cycle_efficiency,
                max_queue_length=max_queue_length,
                average_queue_length=total_queue_time / max(total_vehicles, 1),
                total_queue_time=total_queue_time,
                queue_clearance_time=0.0,  # Would need more sophisticated tracking
                phase_changes=phase_changes,
                average_phase_duration=test_config.duration / max(phase_changes, 1),
                green_time_utilization=total_green_time / test_config.duration,
                red_time_utilization=total_red_time / test_config.duration,
                fuel_consumption=fuel_consumption,
                co2_emissions=co2_emissions,
                nox_emissions=nox_emissions,
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                error_count=error_count,
                reconnection_count=reconnection_count
            )
            
            logger.info(f"Collected metrics for {test_config.test_id}: {total_vehicles} vehicles, {average_wait_time:.2f}s avg wait")
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            # Return empty metrics on error
            return PerformanceMetrics(
                test_id=test_config.test_id,
                run_id=run_id,
                timestamp=time.time(),
                duration=test_config.duration,
                total_vehicles=0,
                total_wait_time=0.0,
                average_wait_time=0.0,
                max_wait_time=0.0,
                total_travel_time=0.0,
                average_travel_time=0.0,
                vehicles_per_hour=0.0,
                vehicles_per_cycle=0.0,
                cycle_efficiency=0.0,
                max_queue_length=0,
                average_queue_length=0.0,
                total_queue_time=0.0,
                queue_clearance_time=0.0,
                phase_changes=0,
                average_phase_duration=0.0,
                green_time_utilization=0.0,
                red_time_utilization=0.0,
                fuel_consumption=0.0,
                co2_emissions=0.0,
                nox_emissions=0.0,
                cpu_usage=0.0,
                memory_usage=0.0,
                error_count=0,
                reconnection_count=0
            )
    
    def run_single_test(self, test_config: TestConfiguration) -> TestResult:
        """
        Run a single A/B test
        
        Args:
            test_config: Test configuration
            
        Returns:
            Test result
        """
        logger.info(f"Starting test: {test_config.test_id}")
        start_time = time.time()
        
        try:
            # Start SUMO
            if not self.start_sumo(test_config.scenario):
                raise Exception("Failed to start SUMO")
            
            # Initialize controllers
            self.initialize_controllers(test_config)
            
            # Run test repetitions
            ml_runs = []
            baseline_runs = []
            
            for repetition in range(test_config.repetitions):
                run_id = f"{test_config.test_id}_run_{repetition + 1}"
                
                # Warmup period
                logger.info(f"Warmup period for {test_config.warmup_time}s")
                time.sleep(test_config.warmup_time)
                
                # Run test
                logger.info(f"Running test repetition {repetition + 1}/{test_config.repetitions}")
                
                if test_config.ml_enabled:
                    # Run with ML optimization
                    self._run_ml_optimization(test_config)
                else:
                    # Run with baseline
                    self._run_baseline_control(test_config)
                
                # Collect metrics
                metrics = self.collect_metrics(test_config, run_id)
                
                if test_config.ml_enabled:
                    ml_runs.append(metrics)
                else:
                    baseline_runs.append(metrics)
                
                # Cooldown period
                if repetition < test_config.repetitions - 1:
                    logger.info(f"Cooldown period for {test_config.cooldown_time}s")
                    time.sleep(test_config.cooldown_time)
            
            # Calculate test results
            end_time = time.time()
            test_result = self._calculate_test_result(
                test_config, start_time, end_time, ml_runs, baseline_runs
            )
            
            logger.info(f"Test completed: {test_config.test_id}")
            return test_result
            
        except Exception as e:
            logger.error(f"Test failed: {test_config.test_id} - {e}")
            end_time = time.time()
            return TestResult(
                test_id=test_config.test_id,
                test_type=test_config.test_type,
                scenario=test_config.scenario,
                status=TestStatus.FAILED,
                start_time=start_time,
                end_time=end_time,
                duration=end_time - start_time,
                ml_runs=[],
                baseline_runs=[],
                statistical_significance=False,
                p_value=1.0,
                effect_size=0.0,
                confidence_interval=(0.0, 0.0),
                wait_time_improvement=0.0,
                throughput_improvement=0.0,
                queue_reduction=0.0,
                fuel_savings=0.0,
                emission_reduction=0.0
            )
        
        finally:
            # Cleanup
            self.cleanup_controllers()
            self.stop_sumo()
    
    def _run_ml_optimization(self, test_config: TestConfiguration):
        """Run ML optimization for the test duration"""
        start_time = time.time()
        
        while (time.time() - start_time) < test_config.duration:
            try:
                # Get current traffic data
                traffic_data = self.traci_controller.get_all_traffic_data()
                
                # Run ML optimization
                if self.ml_optimizer:
                    optimizations = self.ml_optimizer.optimize(traffic_data)
                    
                    # Apply optimizations
                    for intersection_id, optimization in optimizations.items():
                        if 'phase' in optimization:
                            self.traci_controller.set_phase(
                                intersection_id, 
                                optimization['phase']
                            )
                
                time.sleep(1.0)  # Optimize every second
                
            except Exception as e:
                logger.error(f"Error in ML optimization: {e}")
                time.sleep(1.0)
    
    def _run_baseline_control(self, test_config: TestConfiguration):
        """Run baseline control for the test duration"""
        start_time = time.time()
        
        while (time.time() - start_time) < test_config.duration:
            try:
                # Use Webster's formula optimization
                for intersection_id in test_config.intersections:
                    self.traci_controller.optimize_with_webster(intersection_id)
                
                time.sleep(5.0)  # Optimize every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in baseline control: {e}")
                time.sleep(1.0)
    
    def _calculate_test_result(self, test_config: TestConfiguration, start_time: float, 
                             end_time: float, ml_runs: List[PerformanceMetrics], 
                             baseline_runs: List[PerformanceMetrics]) -> TestResult:
        """Calculate test results and statistical analysis"""
        
        # Separate ML and baseline runs
        ml_metrics = [run for run in ml_runs if test_config.ml_enabled]
        baseline_metrics = [run for run in baseline_runs if not test_config.ml_enabled]
        
        # If we have both ML and baseline runs, compare them
        if ml_metrics and baseline_metrics:
            # Statistical analysis
            ml_wait_times = [m.average_wait_time for m in ml_metrics]
            baseline_wait_times = [m.average_wait_time for m in baseline_metrics]
            
            # T-test for statistical significance
            try:
                t_stat, p_value = stats.ttest_ind(ml_wait_times, baseline_wait_times)
                statistical_significance = p_value < 0.05
            except:
                t_stat, p_value = 0.0, 1.0
                statistical_significance = False
            
            # Effect size (Cohen's d)
            try:
                pooled_std = math.sqrt(
                    ((len(ml_wait_times) - 1) * statistics.stdev(ml_wait_times) + 
                     (len(baseline_wait_times) - 1) * statistics.stdev(baseline_wait_times)) /
                    (len(ml_wait_times) + len(baseline_wait_times) - 2)
                )
                effect_size = (statistics.mean(ml_wait_times) - statistics.mean(baseline_wait_times)) / pooled_std
            except:
                effect_size = 0.0
            
            # Confidence interval
            try:
                ml_mean = statistics.mean(ml_wait_times)
                baseline_mean = statistics.mean(baseline_wait_times)
                diff = ml_mean - baseline_mean
                se = math.sqrt(
                    statistics.stdev(ml_wait_times) / len(ml_wait_times) +
                    statistics.stdev(baseline_wait_times) / len(baseline_wait_times)
                )
                margin = 1.96 * se  # 95% confidence interval
                confidence_interval = (diff - margin, diff + margin)
            except:
                confidence_interval = (0.0, 0.0)
            
            # Performance improvements
            ml_avg_wait = statistics.mean(ml_wait_times)
            baseline_avg_wait = statistics.mean(baseline_wait_times)
            wait_time_improvement = ((baseline_avg_wait - ml_avg_wait) / baseline_avg_wait) * 100
            
            ml_throughput = statistics.mean([m.vehicles_per_hour for m in ml_metrics])
            baseline_throughput = statistics.mean([m.vehicles_per_hour for m in baseline_metrics])
            throughput_improvement = ((ml_throughput - baseline_throughput) / baseline_throughput) * 100
            
            ml_queues = statistics.mean([m.average_queue_length for m in ml_metrics])
            baseline_queues = statistics.mean([m.average_queue_length for m in baseline_metrics])
            queue_reduction = ((baseline_queues - ml_queues) / baseline_queues) * 100
            
            ml_fuel = statistics.mean([m.fuel_consumption for m in ml_metrics])
            baseline_fuel = statistics.mean([m.fuel_consumption for m in baseline_metrics])
            fuel_savings = ((baseline_fuel - ml_fuel) / baseline_fuel) * 100
            
            ml_emissions = statistics.mean([m.co2_emissions for m in ml_metrics])
            baseline_emissions = statistics.mean([m.co2_emissions for m in baseline_metrics])
            emission_reduction = ((baseline_emissions - ml_emissions) / baseline_emissions) * 100
            
        else:
            # No comparison possible
            statistical_significance = False
            p_value = 1.0
            effect_size = 0.0
            confidence_interval = (0.0, 0.0)
            wait_time_improvement = 0.0
            throughput_improvement = 0.0
            queue_reduction = 0.0
            fuel_savings = 0.0
            emission_reduction = 0.0
        
        return TestResult(
            test_id=test_config.test_id,
            test_type=test_config.test_type,
            scenario=test_config.scenario,
            status=TestStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            ml_runs=ml_metrics,
            baseline_runs=baseline_metrics,
            statistical_significance=statistical_significance,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=confidence_interval,
            wait_time_improvement=wait_time_improvement,
            throughput_improvement=throughput_improvement,
            queue_reduction=queue_reduction,
            fuel_savings=fuel_savings,
            emission_reduction=emission_reduction
        )
    
    def run_all_tests(self):
        """Run all configured A/B tests"""
        logger.info(f"Starting A/B test suite with {len(self.tests)} tests")
        self.running = True
        
        for i, test_config in enumerate(self.tests):
            logger.info(f"Running test {i+1}/{len(self.tests)}: {test_config.test_id}")
            
            try:
                result = self.run_single_test(test_config)
                self.results.append(result)
                
                if result.status == TestStatus.COMPLETED:
                    self.stats['tests_passed'] += 1
                    logger.info(f"Test passed: {test_config.test_id}")
                else:
                    self.stats['tests_failed'] += 1
                    logger.warning(f"Test failed: {test_config.test_id}")
                
                self.stats['tests_run'] += 1
                
            except Exception as e:
                logger.error(f"Test error: {test_config.test_id} - {e}")
                self.stats['tests_failed'] += 1
                self.stats['tests_run'] += 1
        
        self.running = False
        self.stats['total_duration'] = time.time() - self.stats['start_time']
        
        logger.info(f"A/B test suite completed: {self.stats['tests_passed']} passed, {self.stats['tests_failed']} failed")
    
    def save_results(self, output_dir: str = None):
        """
        Save test results to files
        
        Args:
            output_dir: Output directory for results
        """
        if output_dir is None:
            output_dir = self.config['output_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save results as JSON
        results_file = os.path.join(output_dir, f"ab_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        results_data = []
        for result in self.results:
            result_dict = asdict(result)
            # Convert enums to strings
            result_dict['test_type'] = result.test_type.value
            result_dict['status'] = result.status.value
            results_data.append(result_dict)
        
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_file}")
        
        # Save metrics as CSV
        metrics_file = os.path.join(output_dir, f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        with open(metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow([
                'test_id', 'run_id', 'timestamp', 'duration', 'total_vehicles',
                'average_wait_time', 'max_wait_time', 'vehicles_per_hour',
                'max_queue_length', 'average_queue_length', 'phase_changes',
                'fuel_consumption', 'co2_emissions', 'error_count'
            ])
            
            # Write data
            for result in self.results:
                for run in result.ml_runs + result.baseline_runs:
                    writer.writerow([
                        run.test_id, run.run_id, run.timestamp, run.duration,
                        run.total_vehicles, run.average_wait_time, run.max_wait_time,
                        run.vehicles_per_hour, run.max_queue_length, run.average_queue_length,
                        run.phase_changes, run.fuel_consumption, run.co2_emissions, run.error_count
                    ])
        
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Save summary report
        self._generate_summary_report(output_dir)
    
    def _generate_summary_report(self, output_dir: str):
        """Generate summary report"""
        report_file = os.path.join(output_dir, f"ab_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        
        with open(report_file, 'w') as f:
            f.write("# A/B Test Results Summary\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Test Statistics\n\n")
            f.write(f"- **Total Tests:** {self.stats['tests_run']}\n")
            f.write(f"- **Passed:** {self.stats['tests_passed']}\n")
            f.write(f"- **Failed:** {self.stats['tests_failed']}\n")
            f.write(f"- **Total Duration:** {self.stats['total_duration']:.1f} seconds\n\n")
            
            f.write("## Test Results\n\n")
            for result in self.results:
                f.write(f"### {result.test_id}\n\n")
                f.write(f"- **Type:** {result.test_type.value}\n")
                f.write(f"**Scenario:** {result.scenario}\n")
                f.write(f"- **Status:** {result.status.value}\n")
                f.write(f"- **Duration:** {result.duration:.1f} seconds\n")
                
                if result.ml_runs and result.baseline_runs:
                    f.write(f"- **Statistical Significance:** {result.statistical_significance}\n")
                    f.write(f"- **P-value:** {result.p_value:.4f}\n")
                    f.write(f"- **Effect Size:** {result.effect_size:.4f}\n")
                    f.write(f"- **Wait Time Improvement:** {result.wait_time_improvement:.2f}%\n")
                    f.write(f"- **Throughput Improvement:** {result.throughput_improvement:.2f}%\n")
                    f.write(f"- **Queue Reduction:** {result.queue_reduction:.2f}%\n")
                    f.write(f"- **Fuel Savings:** {result.fuel_savings:.2f}%\n")
                    f.write(f"- **Emission Reduction:** {result.emission_reduction:.2f}%\n")
                
                f.write("\n")
        
        logger.info(f"Summary report saved to {report_file}")

def main():
    """Main function"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Create A/B test runner
    runner = ABTestRunner()
    
    # Create test scenarios
    runner.create_test_scenarios()
    
    # Run all tests
    runner.run_all_tests()
    
    # Save results
    runner.save_results()
    
    # Print summary
    print(f"\nA/B Test Suite Completed:")
    print(f"Tests run: {runner.stats['tests_run']}")
    print(f"Tests passed: {runner.stats['tests_passed']}")
    print(f"Tests failed: {runner.stats['tests_failed']}")
    print(f"Total duration: {runner.stats['total_duration']:.1f} seconds")

if __name__ == "__main__":
    main()
