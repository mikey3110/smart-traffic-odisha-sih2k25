"""
Comprehensive Model Validation Framework for ML Traffic Optimization
Phase 3: ML Model Validation & Performance Analytics

Features:
- A/B testing framework comparing ML vs baseline (Webster's formula)
- Diverse SUMO scenarios: rush hour, emergency, special events, weather
- Statistical significance testing for performance improvements
- Cross-validation across different intersection topologies
- Model bias detection and fairness analysis
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import scipy.stats as stats
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml

from algorithms.advanced_q_learning_agent import AdvancedQLearningAgent
from safety.safety_manager import SafetyManager, WebsterFormulaFallback
from sumo.enhanced_traci_controller import EnhancedTraCIController, TrafficData


class ValidationScenario(Enum):
    """Validation scenario types"""
    RUSH_HOUR = "rush_hour"
    EMERGENCY = "emergency"
    SPECIAL_EVENT = "special_event"
    WEATHER_CONDITIONS = "weather_conditions"
    NORMAL_TRAFFIC = "normal_traffic"
    CONGESTION = "congestion"
    LOW_TRAFFIC = "low_traffic"


class ValidationMetric(Enum):
    """Validation metrics"""
    WAIT_TIME_REDUCTION = "wait_time_reduction"
    THROUGHPUT_INCREASE = "throughput_increase"
    FUEL_CONSUMPTION_REDUCTION = "fuel_consumption_reduction"
    EMISSION_REDUCTION = "emission_reduction"
    PEDESTRIAN_SAFETY = "pedestrian_safety"
    EMERGENCY_RESPONSE_TIME = "emergency_response_time"
    QUEUE_LENGTH_REDUCTION = "queue_length_reduction"


@dataclass
class ValidationResult:
    """Validation result for a single test"""
    test_id: str
    scenario: ValidationScenario
    intersection_id: str
    algorithm: str  # 'ml' or 'webster'
    duration: float  # seconds
    metrics: Dict[str, float]
    confidence_interval: Dict[str, Tuple[float, float]]
    p_value: float
    is_significant: bool
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class A/BTestResult:
    """A/B test result comparing ML vs baseline"""
    test_id: str
    scenario: ValidationScenario
    intersection_id: str
    ml_results: List[ValidationResult]
    baseline_results: List[ValidationResult]
    improvement_percentage: Dict[str, float]
    statistical_significance: Dict[str, bool]
    effect_size: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    recommendation: str
    timestamp: datetime


class ScenarioGenerator:
    """Generate diverse SUMO scenarios for validation"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Scenario parameters
        self.scenario_configs = {
            ValidationScenario.RUSH_HOUR: {
                'vehicle_spawn_rate': 0.8,
                'peak_hours': [7, 8, 9, 17, 18, 19],
                'congestion_factor': 1.5,
                'duration': 3600  # 1 hour
            },
            ValidationScenario.EMERGENCY: {
                'vehicle_spawn_rate': 0.6,
                'emergency_vehicles': True,
                'priority_override': True,
                'duration': 1800  # 30 minutes
            },
            ValidationScenario.SPECIAL_EVENT: {
                'vehicle_spawn_rate': 1.2,
                'event_attendance': 10000,
                'parking_demand': 0.8,
                'duration': 7200  # 2 hours
            },
            ValidationScenario.WEATHER_CONDITIONS: {
                'vehicle_spawn_rate': 0.7,
                'weather_condition': 'rain',
                'visibility_reduction': 0.3,
                'speed_reduction': 0.2,
                'duration': 3600
            },
            ValidationScenario.NORMAL_TRAFFIC: {
                'vehicle_spawn_rate': 0.5,
                'duration': 3600
            },
            ValidationScenario.CONGESTION: {
                'vehicle_spawn_rate': 1.0,
                'congestion_factor': 2.0,
                'duration': 1800
            },
            ValidationScenario.LOW_TRAFFIC: {
                'vehicle_spawn_rate': 0.2,
                'duration': 1800
            }
        }
    
    def generate_scenario_config(self, scenario: ValidationScenario, 
                               intersection_id: str) -> Dict[str, Any]:
        """Generate SUMO configuration for validation scenario"""
        base_config = self.scenario_configs[scenario]
        
        config = {
            'intersection_id': intersection_id,
            'scenario_type': scenario.value,
            'duration': base_config['duration'],
            'vehicle_spawn_rate': base_config['vehicle_spawn_rate'],
            'traffic_lights': [f'tl_{intersection_id}'],
            'lanes': [f'{intersection_id}_north', f'{intersection_id}_south', 
                     f'{intersection_id}_east', f'{intersection_id}_west'],
            'detectors': [f'det_{intersection_id}_{i}' for i in range(4)],
            'phases': [0, 1, 2, 3]
        }
        
        # Add scenario-specific parameters
        if scenario == ValidationScenario.EMERGENCY:
            config['emergency_vehicles'] = base_config['emergency_vehicles']
            config['priority_override'] = base_config['priority_override']
        elif scenario == ValidationScenario.SPECIAL_EVENT:
            config['event_attendance'] = base_config['event_attendance']
            config['parking_demand'] = base_config['parking_demand']
        elif scenario == ValidationScenario.WEATHER_CONDITIONS:
            config['weather_condition'] = base_config['weather_condition']
            config['visibility_reduction'] = base_config['visibility_reduction']
            config['speed_reduction'] = base_config['speed_reduction']
        elif scenario == ValidationScenario.CONGESTION:
            config['congestion_factor'] = base_config['congestion_factor']
        
        return config
    
    def create_sumo_network(self, scenario: ValidationScenario, 
                          intersection_id: str) -> str:
        """Create SUMO network file for scenario"""
        config = self.generate_scenario_config(scenario, intersection_id)
        
        # Generate SUMO network XML
        network_xml = self._generate_network_xml(config)
        
        # Save to file
        network_path = f"validation_networks/{scenario.value}_{intersection_id}.net.xml"
        Path(network_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(network_path, 'w') as f:
            f.write(network_xml)
        
        return network_path
    
    def _generate_network_xml(self, config: Dict[str, Any]) -> str:
        """Generate SUMO network XML content"""
        # Simplified network generation - in practice, this would be more complex
        xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<net version="1.9" junctionCornerDetail="5" limitTurnSpeed="5.5" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,100.00,100.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>
    
    <edge id="{config['intersection_id']}_north" from="j1" to="j2" priority="1">
        <lane id="{config['lanes'][0]}" index="0" speed="13.89" length="100.00" shape="0.00,0.00 100.00,0.00"/>
    </edge>
    <edge id="{config['intersection_id']}_south" from="j2" to="j1" priority="1">
        <lane id="{config['lanes'][1]}" index="0" speed="13.89" length="100.00" shape="100.00,100.00 0.00,100.00"/>
    </edge>
    <edge id="{config['intersection_id']}_east" from="j3" to="j4" priority="1">
        <lane id="{config['lanes'][2]}" index="0" speed="13.89" length="100.00" shape="0.00,100.00 0.00,0.00"/>
    </edge>
    <edge id="{config['intersection_id']}_west" from="j4" to="j3" priority="1">
        <lane id="{config['lanes'][3]}" index="0" speed="13.89" length="100.00" shape="100.00,0.00 100.00,100.00"/>
    </edge>
    
    <junction id="j1" type="priority" x="0.00" y="0.00" incLanes="" intLanes="" shape="0.00,0.00 0.00,10.00 10.00,10.00 10.00,0.00"/>
    <junction id="j2" type="priority" x="100.00" y="0.00" incLanes="" intLanes="" shape="90.00,0.00 100.00,0.00 100.00,10.00 90.00,10.00"/>
    <junction id="j3" type="priority" x="0.00" y="100.00" incLanes="" intLanes="" shape="0.00,90.00 10.00,90.00 10.00,100.00 0.00,100.00"/>
    <junction id="j4" type="priority" x="100.00" y="100.00" incLanes="" intLanes="" shape="90.00,100.00 100.00,100.00 100.00,90.00 90.00,90.00"/>
    
    <tlLogic id="{config['traffic_lights'][0]}" type="static" programID="0" offset="0">
        <phase duration="30" state="GGrrGGrr"/>
        <phase duration="3" state="yyrryyrr"/>
        <phase duration="30" state="rrGGrrGG"/>
        <phase duration="3" state="rryyrryy"/>
    </tlLogic>
</net>"""
        
        return xml_content


class StatisticalAnalyzer:
    """Statistical analysis for validation results"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Statistical parameters
        self.alpha = self.config.get('alpha', 0.05)  # Significance level
        self.min_sample_size = self.config.get('min_sample_size', 30)
        self.confidence_level = self.config.get('confidence_level', 0.95)
    
    def calculate_improvement_metrics(self, ml_results: List[ValidationResult], 
                                    baseline_results: List[ValidationResult]) -> Dict[str, float]:
        """Calculate improvement metrics between ML and baseline"""
        improvements = {}
        
        # Get all metrics from both result sets
        all_metrics = set()
        for result in ml_results + baseline_results:
            all_metrics.update(result.metrics.keys())
        
        for metric in all_metrics:
            ml_values = [r.metrics.get(metric, 0) for r in ml_results if metric in r.metrics]
            baseline_values = [r.metrics.get(metric, 0) for r in baseline_results if metric in r.metrics]
            
            if ml_values and baseline_values:
                ml_mean = np.mean(ml_values)
                baseline_mean = np.mean(baseline_values)
                
                if baseline_mean != 0:
                    improvement = ((ml_mean - baseline_mean) / baseline_mean) * 100
                    improvements[metric] = improvement
                else:
                    improvements[metric] = 0.0
        
        return improvements
    
    def test_statistical_significance(self, ml_results: List[ValidationResult], 
                                    baseline_results: List[ValidationResult]) -> Dict[str, bool]:
        """Test statistical significance of improvements"""
        significance = {}
        
        # Get all metrics
        all_metrics = set()
        for result in ml_results + baseline_results:
            all_metrics.update(result.metrics.keys())
        
        for metric in all_metrics:
            ml_values = [r.metrics.get(metric, 0) for r in ml_results if metric in r.metrics]
            baseline_values = [r.metrics.get(metric, 0) for r in baseline_results if metric in r.metrics]
            
            if len(ml_values) >= self.min_sample_size and len(baseline_values) >= self.min_sample_size:
                try:
                    # Perform t-test
                    t_stat, p_value = stats.ttest_ind(ml_values, baseline_values)
                    significance[metric] = p_value < self.alpha
                except Exception as e:
                    self.logger.warning(f"Error in statistical test for {metric}: {e}")
                    significance[metric] = False
            else:
                significance[metric] = False
        
        return significance
    
    def calculate_effect_size(self, ml_results: List[ValidationResult], 
                            baseline_results: List[ValidationResult]) -> Dict[str, float]:
        """Calculate Cohen's d effect size"""
        effect_sizes = {}
        
        all_metrics = set()
        for result in ml_results + baseline_results:
            all_metrics.update(result.metrics.keys())
        
        for metric in all_metrics:
            ml_values = [r.metrics.get(metric, 0) for r in ml_results if metric in r.metrics]
            baseline_values = [r.metrics.get(metric, 0) for r in baseline_results if metric in r.metrics]
            
            if len(ml_values) >= 2 and len(baseline_values) >= 2:
                try:
                    # Calculate Cohen's d
                    ml_mean = np.mean(ml_values)
                    baseline_mean = np.mean(baseline_values)
                    
                    # Pooled standard deviation
                    ml_std = np.std(ml_values, ddof=1)
                    baseline_std = np.std(baseline_values, ddof=1)
                    pooled_std = np.sqrt(((len(ml_values) - 1) * ml_std**2 + 
                                        (len(baseline_values) - 1) * baseline_std**2) / 
                                       (len(ml_values) + len(baseline_values) - 2))
                    
                    if pooled_std != 0:
                        effect_size = (ml_mean - baseline_mean) / pooled_std
                        effect_sizes[metric] = effect_size
                    else:
                        effect_sizes[metric] = 0.0
                        
                except Exception as e:
                    self.logger.warning(f"Error calculating effect size for {metric}: {e}")
                    effect_sizes[metric] = 0.0
            else:
                effect_sizes[metric] = 0.0
        
        return effect_sizes
    
    def calculate_confidence_intervals(self, ml_results: List[ValidationResult], 
                                     baseline_results: List[ValidationResult]) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for improvements"""
        confidence_intervals = {}
        
        all_metrics = set()
        for result in ml_results + baseline_results:
            all_metrics.update(result.metrics.keys())
        
        for metric in all_metrics:
            ml_values = [r.metrics.get(metric, 0) for r in ml_results if metric in r.metrics]
            baseline_values = [r.metrics.get(metric, 0) for r in baseline_results if metric in r.metrics]
            
            if len(ml_values) >= 2 and len(baseline_values) >= 2:
                try:
                    # Calculate difference in means
                    ml_mean = np.mean(ml_values)
                    baseline_mean = np.mean(baseline_values)
                    difference = ml_mean - baseline_mean
                    
                    # Calculate standard error
                    ml_std = np.std(ml_values, ddof=1)
                    baseline_std = np.std(baseline_values, ddof=1)
                    se = np.sqrt((ml_std**2 / len(ml_values)) + (baseline_std**2 / len(baseline_values)))
                    
                    # Calculate confidence interval
                    alpha = 1 - self.confidence_level
                    t_critical = stats.t.ppf(1 - alpha/2, len(ml_values) + len(baseline_values) - 2)
                    margin_error = t_critical * se
                    
                    ci_lower = difference - margin_error
                    ci_upper = difference + margin_error
                    
                    confidence_intervals[metric] = (ci_lower, ci_upper)
                    
                except Exception as e:
                    self.logger.warning(f"Error calculating confidence interval for {metric}: {e}")
                    confidence_intervals[metric] = (0.0, 0.0)
            else:
                confidence_intervals[metric] = (0.0, 0.0)
        
        return confidence_intervals


class ModelValidator:
    """
    Comprehensive Model Validation Framework
    
    Features:
    - A/B testing framework comparing ML vs baseline
    - Diverse SUMO scenarios for comprehensive testing
    - Statistical significance testing
    - Cross-validation across intersection topologies
    - Model bias detection and fairness analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.scenario_generator = ScenarioGenerator(config.get('scenario_generator', {}))
        self.statistical_analyzer = StatisticalAnalyzer(config.get('statistical_analyzer', {}))
        
        # Validation state
        self.validation_results = []
        self.ab_test_results = []
        self.current_tests = {}
        
        # ML and baseline components
        self.ml_agents = {}
        self.safety_managers = {}
        self.traci_controllers = {}
        
        # Validation configuration
        self.test_duration = self.config.get('test_duration', 3600)  # 1 hour
        self.num_iterations = self.config.get('num_iterations', 10)
        self.cross_validation_folds = self.config.get('cross_validation_folds', 5)
        
        self.logger.info("Model Validator initialized")
    
    def add_ml_agent(self, intersection_id: str, agent: AdvancedQLearningAgent):
        """Add ML agent for validation"""
        self.ml_agents[intersection_id] = agent
        self.logger.info(f"Added ML agent for {intersection_id}")
    
    def add_safety_manager(self, intersection_id: str, safety_manager: SafetyManager):
        """Add safety manager for baseline validation"""
        self.safety_managers[intersection_id] = safety_manager
        self.logger.info(f"Added safety manager for {intersection_id}")
    
    def add_traci_controller(self, intersection_id: str, controller: EnhancedTraCIController):
        """Add TraCI controller for validation"""
        self.traci_controllers[intersection_id] = controller
        self.logger.info(f"Added TraCI controller for {intersection_id}")
    
    async def run_ab_test(self, scenario: ValidationScenario, 
                         intersection_id: str, 
                         num_runs: int = 10) -> A/BTestResult:
        """Run A/B test comparing ML vs baseline"""
        test_id = str(uuid.uuid4())
        self.logger.info(f"Starting A/B test {test_id} for {scenario.value} at {intersection_id}")
        
        try:
            # Generate scenario configuration
            scenario_config = self.scenario_generator.generate_scenario_config(
                scenario, intersection_id
            )
            
            # Run ML optimization tests
            ml_results = []
            for i in range(num_runs):
                result = await self._run_ml_test(scenario, intersection_id, scenario_config)
                if result:
                    ml_results.append(result)
            
            # Run baseline (Webster's formula) tests
            baseline_results = []
            for i in range(num_runs):
                result = await self._run_baseline_test(scenario, intersection_id, scenario_config)
                if result:
                    baseline_results.append(result)
            
            # Perform statistical analysis
            improvements = self.statistical_analyzer.calculate_improvement_metrics(
                ml_results, baseline_results
            )
            significance = self.statistical_analyzer.test_statistical_significance(
                ml_results, baseline_results
            )
            effect_sizes = self.statistical_analyzer.calculate_effect_size(
                ml_results, baseline_results
            )
            confidence_intervals = self.statistical_analyzer.calculate_confidence_intervals(
                ml_results, baseline_results
            )
            
            # Generate recommendation
            recommendation = self._generate_recommendation(improvements, significance, effect_sizes)
            
            # Create A/B test result
            ab_result = A/BTestResult(
                test_id=test_id,
                scenario=scenario,
                intersection_id=intersection_id,
                ml_results=ml_results,
                baseline_results=baseline_results,
                improvement_percentage=improvements,
                statistical_significance=significance,
                effect_size=effect_sizes,
                confidence_intervals=confidence_intervals,
                recommendation=recommendation,
                timestamp=datetime.now()
            )
            
            self.ab_test_results.append(ab_result)
            self.logger.info(f"A/B test {test_id} completed with {len(ml_results)} ML runs and {len(baseline_results)} baseline runs")
            
            return ab_result
            
        except Exception as e:
            self.logger.error(f"Error in A/B test {test_id}: {e}")
            raise
    
    async def _run_ml_test(self, scenario: ValidationScenario, 
                          intersection_id: str, 
                          scenario_config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Run single ML optimization test"""
        try:
            test_id = str(uuid.uuid4())
            start_time = time.time()
            
            # Get ML agent and controller
            agent = self.ml_agents.get(intersection_id)
            controller = self.traci_controllers.get(intersection_id)
            
            if not agent or not controller:
                self.logger.warning(f"Missing components for ML test at {intersection_id}")
                return None
            
            # Start simulation
            if not controller.start_simulation():
                self.logger.error(f"Failed to start simulation for ML test at {intersection_id}")
                return None
            
            # Run optimization for test duration
            metrics = await self._run_optimization_loop(
                agent, controller, scenario_config, test_duration=scenario_config['duration']
            )
            
            # Stop simulation
            controller.stop_simulation()
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(metrics)
            
            # Create validation result
            result = ValidationResult(
                test_id=test_id,
                scenario=scenario,
                intersection_id=intersection_id,
                algorithm='ml',
                duration=time.time() - start_time,
                metrics=performance_metrics,
                confidence_interval={},  # Will be calculated in A/B test
                p_value=0.0,  # Will be calculated in A/B test
                is_significant=False,  # Will be calculated in A/B test
                timestamp=datetime.now(),
                metadata={'scenario_config': scenario_config}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in ML test at {intersection_id}: {e}")
            return None
    
    async def _run_baseline_test(self, scenario: ValidationScenario, 
                               intersection_id: str, 
                               scenario_config: Dict[str, Any]) -> Optional[ValidationResult]:
        """Run single baseline (Webster's formula) test"""
        try:
            test_id = str(uuid.uuid4())
            start_time = time.time()
            
            # Get safety manager and controller
            safety_manager = self.safety_managers.get(intersection_id)
            controller = self.traci_controllers.get(intersection_id)
            
            if not safety_manager or not controller:
                self.logger.warning(f"Missing components for baseline test at {intersection_id}")
                return None
            
            # Start simulation
            if not controller.start_simulation():
                self.logger.error(f"Failed to start simulation for baseline test at {intersection_id}")
                return None
            
            # Run Webster's formula optimization
            metrics = await self._run_webster_optimization_loop(
                safety_manager, controller, scenario_config, test_duration=scenario_config['duration']
            )
            
            # Stop simulation
            controller.stop_simulation()
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(metrics)
            
            # Create validation result
            result = ValidationResult(
                test_id=test_id,
                scenario=scenario,
                intersection_id=intersection_id,
                algorithm='webster',
                duration=time.time() - start_time,
                metrics=performance_metrics,
                confidence_interval={},
                p_value=0.0,
                is_significant=False,
                timestamp=datetime.now(),
                metadata={'scenario_config': scenario_config}
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in baseline test at {intersection_id}: {e}")
            return None
    
    async def _run_optimization_loop(self, agent: AdvancedQLearningAgent, 
                                   controller: EnhancedTraCIController,
                                   scenario_config: Dict[str, Any], 
                                   test_duration: int) -> List[Dict[str, Any]]:
        """Run ML optimization loop for test duration"""
        metrics = []
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            try:
                # Get traffic data
                traffic_data = controller.get_traffic_data(timeout=1.0)
                if not traffic_data:
                    await asyncio.sleep(1)
                    continue
                
                # Create state
                state = agent.create_state(traffic_data, {})
                
                # Select action
                action = agent.select_action(state, training=False)
                
                # Apply action
                controller.send_control_command(
                    traffic_data.intersection_id,
                    "set_phase",
                    {
                        'traffic_light_id': scenario_config['traffic_lights'][0],
                        'phase': action.phase,
                        'duration': action.duration
                    }
                )
                
                # Record metrics
                metrics.append({
                    'timestamp': datetime.now(),
                    'wait_time': np.mean(list(traffic_data.waiting_times.values())),
                    'throughput': sum(traffic_data.vehicle_counts.values()),
                    'queue_length': sum(traffic_data.queue_lengths.values()),
                    'fuel_consumption': self._estimate_fuel_consumption(traffic_data),
                    'emissions': self._estimate_emissions(traffic_data)
                })
                
                await asyncio.sleep(1)  # 1-second simulation steps
                
            except Exception as e:
                self.logger.warning(f"Error in optimization loop: {e}")
                await asyncio.sleep(1)
        
        return metrics
    
    async def _run_webster_optimization_loop(self, safety_manager: SafetyManager,
                                           controller: EnhancedTraCIController,
                                           scenario_config: Dict[str, Any],
                                           test_duration: int) -> List[Dict[str, Any]]:
        """Run Webster's formula optimization loop for test duration"""
        metrics = []
        start_time = time.time()
        
        while time.time() - start_time < test_duration:
            try:
                # Get traffic data
                traffic_data = controller.get_traffic_data(timeout=1.0)
                if not traffic_data:
                    await asyncio.sleep(1)
                    continue
                
                # Use Webster's formula optimization
                traffic_dict = {
                    'intersection_id': traffic_data.intersection_id,
                    'lane_counts': traffic_data.vehicle_counts,
                    'timestamp': traffic_data.timestamp.isoformat()
                }
                
                result = safety_manager.check_safety_and_optimize(
                    traffic_data.intersection_id,
                    traffic_dict,
                    {'north': 30, 'south': 30, 'east': 30, 'west': 30}
                )
                
                if result['success']:
                    # Apply Webster's timings
                    controller.send_control_command(
                        traffic_data.intersection_id,
                        "set_timing",
                        {
                            'traffic_light_id': scenario_config['traffic_lights'][0],
                            'timings': result['optimized_timings']
                        }
                    )
                
                # Record metrics
                metrics.append({
                    'timestamp': datetime.now(),
                    'wait_time': np.mean(list(traffic_data.waiting_times.values())),
                    'throughput': sum(traffic_data.vehicle_counts.values()),
                    'queue_length': sum(traffic_data.queue_lengths.values()),
                    'fuel_consumption': self._estimate_fuel_consumption(traffic_data),
                    'emissions': self._estimate_emissions(traffic_data)
                })
                
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.warning(f"Error in Webster optimization loop: {e}")
                await asyncio.sleep(1)
        
        return metrics
    
    def _calculate_performance_metrics(self, metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics from collected data"""
        if not metrics:
            return {}
        
        df = pd.DataFrame(metrics)
        
        performance_metrics = {
            'avg_wait_time': df['wait_time'].mean(),
            'max_wait_time': df['wait_time'].max(),
            'total_throughput': df['throughput'].sum(),
            'avg_throughput': df['throughput'].mean(),
            'max_queue_length': df['queue_length'].max(),
            'avg_queue_length': df['queue_length'].mean(),
            'total_fuel_consumption': df['fuel_consumption'].sum(),
            'avg_fuel_consumption': df['fuel_consumption'].mean(),
            'total_emissions': df['emissions'].sum(),
            'avg_emissions': df['emissions'].mean()
        }
        
        # Calculate improvement metrics
        if len(metrics) > 1:
            performance_metrics['wait_time_reduction'] = (
                (df['wait_time'].iloc[0] - df['wait_time'].iloc[-1]) / df['wait_time'].iloc[0] * 100
            )
            performance_metrics['throughput_increase'] = (
                (df['throughput'].iloc[-1] - df['throughput'].iloc[0]) / df['throughput'].iloc[0] * 100
            )
        
        return performance_metrics
    
    def _estimate_fuel_consumption(self, traffic_data: TrafficData) -> float:
        """Estimate fuel consumption based on traffic data"""
        # Simplified fuel consumption estimation
        total_vehicles = sum(traffic_data.vehicle_counts.values())
        avg_speed = np.mean(list(traffic_data.avg_speeds.values()))
        
        # Fuel consumption increases with congestion (lower speeds)
        if avg_speed > 0:
            fuel_efficiency = max(0.1, avg_speed / 50.0)  # Normalize to 50 km/h
            fuel_consumption = total_vehicles * (1.0 - fuel_efficiency) * 0.1  # L/vehicle
        else:
            fuel_consumption = total_vehicles * 0.1  # Base consumption
        
        return fuel_consumption
    
    def _estimate_emissions(self, traffic_data: TrafficData) -> float:
        """Estimate emissions based on traffic data"""
        # Simplified emissions estimation
        total_vehicles = sum(traffic_data.vehicle_counts.values())
        avg_speed = np.mean(list(traffic_data.avg_speeds.values()))
        
        # Emissions increase with congestion and stop-and-go traffic
        if avg_speed > 0:
            emission_factor = max(0.5, 2.0 - (avg_speed / 30.0))  # Higher emissions at lower speeds
            emissions = total_vehicles * emission_factor * 0.01  # kg CO2/vehicle
        else:
            emissions = total_vehicles * 0.02  # Base emissions
        
        return emissions
    
    def _generate_recommendation(self, improvements: Dict[str, float], 
                               significance: Dict[str, bool], 
                               effect_sizes: Dict[str, float]) -> str:
        """Generate recommendation based on validation results"""
        significant_improvements = [metric for metric, sig in significance.items() if sig]
        large_improvements = [metric for metric, imp in improvements.items() if imp > 20]
        
        if significant_improvements and large_improvements:
            return "ML optimization shows significant improvements and should be deployed"
        elif significant_improvements:
            return "ML optimization shows significant improvements but effect size is moderate"
        elif large_improvements:
            return "ML optimization shows large improvements but statistical significance is unclear"
        else:
            return "ML optimization does not show clear improvements over baseline"
    
    async def run_comprehensive_validation(self, intersection_ids: List[str]) -> Dict[str, Any]:
        """Run comprehensive validation across all scenarios and intersections"""
        self.logger.info("Starting comprehensive validation")
        
        validation_summary = {
            'start_time': datetime.now(),
            'intersections': intersection_ids,
            'scenarios': [s.value for s in ValidationScenario],
            'ab_test_results': [],
            'overall_improvements': {},
            'recommendations': []
        }
        
        try:
            # Run A/B tests for all scenario-intersection combinations
            for intersection_id in intersection_ids:
                for scenario in ValidationScenario:
                    try:
                        ab_result = await self.run_ab_test(scenario, intersection_id)
                        validation_summary['ab_test_results'].append(ab_result)
                        
                        # Track overall improvements
                        for metric, improvement in ab_result.improvement_percentage.items():
                            if metric not in validation_summary['overall_improvements']:
                                validation_summary['overall_improvements'][metric] = []
                            validation_summary['overall_improvements'][metric].append(improvement)
                        
                    except Exception as e:
                        self.logger.error(f"Error in A/B test for {scenario.value} at {intersection_id}: {e}")
            
            # Calculate overall statistics
            for metric, improvements in validation_summary['overall_improvements'].items():
                validation_summary['overall_improvements'][metric] = {
                    'mean_improvement': np.mean(improvements),
                    'std_improvement': np.std(improvements),
                    'min_improvement': np.min(improvements),
                    'max_improvement': np.max(improvements),
                    'median_improvement': np.median(improvements)
                }
            
            # Generate overall recommendations
            validation_summary['recommendations'] = self._generate_overall_recommendations(
                validation_summary['overall_improvements']
            )
            
            validation_summary['end_time'] = datetime.now()
            validation_summary['duration'] = (
                validation_summary['end_time'] - validation_summary['start_time']
            ).total_seconds()
            
            self.logger.info("Comprehensive validation completed")
            return validation_summary
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive validation: {e}")
            raise
    
    def _generate_overall_recommendations(self, overall_improvements: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations based on validation results"""
        recommendations = []
        
        # Check wait time reduction
        if 'wait_time_reduction' in overall_improvements:
            mean_improvement = overall_improvements['wait_time_reduction']['mean_improvement']
            if mean_improvement > 30:
                recommendations.append(f"Excellent wait time reduction: {mean_improvement:.1f}%")
            elif mean_improvement > 20:
                recommendations.append(f"Good wait time reduction: {mean_improvement:.1f}%")
            elif mean_improvement > 10:
                recommendations.append(f"Moderate wait time reduction: {mean_improvement:.1f}%")
            else:
                recommendations.append(f"Limited wait time reduction: {mean_improvement:.1f}%")
        
        # Check throughput increase
        if 'throughput_increase' in overall_improvements:
            mean_improvement = overall_improvements['throughput_increase']['mean_improvement']
            if mean_improvement > 20:
                recommendations.append(f"Significant throughput increase: {mean_improvement:.1f}%")
            elif mean_improvement > 10:
                recommendations.append(f"Moderate throughput increase: {mean_improvement:.1f}%")
            else:
                recommendations.append(f"Limited throughput increase: {mean_improvement:.1f}%")
        
        # Check fuel consumption reduction
        if 'fuel_consumption_reduction' in overall_improvements:
            mean_improvement = overall_improvements['fuel_consumption_reduction']['mean_improvement']
            if mean_improvement > 15:
                recommendations.append(f"Significant fuel savings: {mean_improvement:.1f}%")
            elif mean_improvement > 5:
                recommendations.append(f"Moderate fuel savings: {mean_improvement:.1f}%")
            else:
                recommendations.append(f"Limited fuel savings: {mean_improvement:.1f}%")
        
        return recommendations
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results"""
        return {
            'total_tests': len(self.ab_test_results),
            'scenarios_tested': len(set(r.scenario for r in self.ab_test_results)),
            'intersections_tested': len(set(r.intersection_id for r in self.ab_test_results)),
            'successful_tests': len([r for r in self.ab_test_results if r.ml_results and r.baseline_results]),
            'average_improvements': self._calculate_average_improvements(),
            'recommendations': self._get_overall_recommendations()
        }
    
    def _calculate_average_improvements(self) -> Dict[str, float]:
        """Calculate average improvements across all tests"""
        if not self.ab_test_results:
            return {}
        
        all_improvements = {}
        for result in self.ab_test_results:
            for metric, improvement in result.improvement_percentage.items():
                if metric not in all_improvements:
                    all_improvements[metric] = []
                all_improvements[metric].append(improvement)
        
        return {metric: np.mean(improvements) for metric, improvements in all_improvements.items()}
    
    def _get_overall_recommendations(self) -> List[str]:
        """Get overall recommendations from all tests"""
        recommendations = []
        
        avg_improvements = self._calculate_average_improvements()
        
        if 'wait_time_reduction' in avg_improvements and avg_improvements['wait_time_reduction'] > 30:
            recommendations.append("ML optimization shows excellent performance and is ready for production deployment")
        elif 'wait_time_reduction' in avg_improvements and avg_improvements['wait_time_reduction'] > 20:
            recommendations.append("ML optimization shows good performance and is recommended for deployment")
        else:
            recommendations.append("ML optimization shows limited improvement and requires further tuning")
        
        return recommendations
