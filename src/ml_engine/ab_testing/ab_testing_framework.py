"""
A/B Testing Framework for ML Traffic Signal Optimization
Comprehensive framework for comparing optimization strategies and algorithms
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import random
from collections import defaultdict
import statistics
from scipy import stats
import threading
import queue

from config.ml_config import ABTestingConfig, get_config
from metrics.enhanced_performance_metrics import EnhancedPerformanceMetrics, TrafficMetrics


class TestStatus(Enum):
    """A/B Test status"""
    PLANNING = "planning"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class StatisticalSignificance(Enum):
    """Statistical significance levels"""
    NOT_SIGNIFICANT = "not_significant"
    MARGINAL = "marginal"
    SIGNIFICANT = "significant"
    HIGHLY_SIGNIFICANT = "highly_significant"


@dataclass
class TestVariant:
    """A/B Test variant configuration"""
    name: str
    algorithm: str
    parameters: Dict[str, Any]
    traffic_split: float = 0.5
    description: str = ""
    is_control: bool = False


@dataclass
class TestResult:
    """Individual test result"""
    timestamp: datetime
    intersection_id: str
    variant: str
    metrics: TrafficMetrics
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class StatisticalTest:
    """Statistical test configuration"""
    test_type: str  # 't_test', 'mann_whitney', 'chi_square'
    alpha: float = 0.05
    power: float = 0.8
    effect_size: float = 0.2
    min_sample_size: int = 30


@dataclass
class ABTestConfig:
    """A/B Test configuration"""
    test_id: str
    name: str
    description: str
    variants: List[TestVariant]
    target_metrics: List[str]
    statistical_test: StatisticalTest
    duration_hours: int = 24
    min_sample_size: int = 100
    max_duration_hours: int = 168  # 1 week
    success_criteria: Dict[str, float] = field(default_factory=dict)
    stop_criteria: Dict[str, float] = field(default_factory=dict)


@dataclass
class TestAnalysis:
    """A/B Test analysis results"""
    test_id: str
    status: TestStatus
    start_time: datetime
    end_time: Optional[datetime]
    total_samples: int
    variant_results: Dict[str, Dict[str, Any]]
    statistical_significance: Dict[str, StatisticalSignificance]
    confidence_intervals: Dict[str, Tuple[float, float]]
    effect_sizes: Dict[str, float]
    recommendations: List[str]
    p_values: Dict[str, float]
    power_analysis: Dict[str, float]


class TrafficSplitter:
    """Traffic splitting logic for A/B testing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.assignment_cache: Dict[str, str] = {}
        self.assignment_lock = threading.Lock()
    
    def assign_variant(self, intersection_id: str, user_id: Optional[str] = None,
                      session_id: Optional[str] = None, variants: List[TestVariant]) -> str:
        """Assign a variant to a user/intersection"""
        
        # Create a unique key for assignment
        assignment_key = f"{intersection_id}_{user_id}_{session_id}"
        
        with self.assignment_lock:
            # Check if already assigned
            if assignment_key in self.assignment_cache:
                return self.assignment_cache[assignment_key]
            
            # Assign based on traffic split
            total_split = sum(v.traffic_split for v in variants)
            if total_split == 0:
                # Equal split if no splits specified
                variant = random.choice(variants)
            else:
                # Weighted random selection
                rand = random.random() * total_split
                cumulative = 0
                for variant in variants:
                    cumulative += variant.traffic_split
                    if rand <= cumulative:
                        break
                else:
                    variant = variants[-1]  # Fallback to last variant
            
            # Cache assignment
            self.assignment_cache[assignment_key] = variant.name
            return variant.name
    
    def clear_cache(self):
        """Clear assignment cache"""
        with self.assignment_lock:
            self.assignment_cache.clear()


class StatisticalAnalyzer:
    """Statistical analysis for A/B testing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_sample_size(self, effect_size: float, alpha: float = 0.05, 
                            power: float = 0.8) -> int:
        """Calculate required sample size for statistical power"""
        # Simplified sample size calculation
        # In practice, would use more sophisticated methods
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = (2 * (z_alpha + z_beta)**2) / (effect_size**2)
        return int(np.ceil(n))
    
    def perform_t_test(self, control_data: List[float], 
                      treatment_data: List[float]) -> Dict[str, Any]:
        """Perform two-sample t-test"""
        try:
            t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
            
            # Calculate effect size (Cohen's d)
            pooled_std = np.sqrt(((len(control_data) - 1) * np.var(control_data, ddof=1) +
                                (len(treatment_data) - 1) * np.var(treatment_data, ddof=1)) /
                               (len(control_data) + len(treatment_data) - 2))
            effect_size = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std
            
            # Calculate confidence interval
            diff = np.mean(treatment_data) - np.mean(control_data)
            se_diff = pooled_std * np.sqrt(1/len(control_data) + 1/len(treatment_data))
            ci_lower = diff - 1.96 * se_diff
            ci_upper = diff + 1.96 * se_diff
            
            return {
                'test_type': 't_test',
                't_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'confidence_interval': (ci_lower, ci_upper),
                'mean_control': np.mean(control_data),
                'mean_treatment': np.mean(treatment_data),
                'std_control': np.std(control_data),
                'std_treatment': np.std(treatment_data)
            }
        except Exception as e:
            self.logger.error(f"Error in t-test: {e}")
            return {'error': str(e)}
    
    def perform_mann_whitney(self, control_data: List[float], 
                           treatment_data: List[float]) -> Dict[str, Any]:
        """Perform Mann-Whitney U test"""
        try:
            u_stat, p_value = stats.mannwhitneyu(treatment_data, control_data, 
                                               alternative='two-sided')
            
            # Calculate effect size (r)
            n1, n2 = len(treatment_data), len(control_data)
            effect_size = 1 - (2 * u_stat) / (n1 * n2)
            
            return {
                'test_type': 'mann_whitney',
                'u_statistic': u_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'median_control': np.median(control_data),
                'median_treatment': np.median(treatment_data)
            }
        except Exception as e:
            self.logger.error(f"Error in Mann-Whitney test: {e}")
            return {'error': str(e)}
    
    def determine_significance(self, p_value: float, alpha: float = 0.05) -> StatisticalSignificance:
        """Determine statistical significance level"""
        if p_value < alpha / 10:
            return StatisticalSignificance.HIGHLY_SIGNIFICANT
        elif p_value < alpha:
            return StatisticalSignificance.SIGNIFICANT
        elif p_value < alpha * 2:
            return StatisticalSignificance.MARGINAL
        else:
            return StatisticalSignificance.NOT_SIGNIFICANT
    
    def calculate_power(self, effect_size: float, sample_size: int, alpha: float = 0.05) -> float:
        """Calculate statistical power"""
        # Simplified power calculation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = effect_size * np.sqrt(sample_size / 2) - z_alpha
        power = stats.norm.cdf(z_beta)
        return max(0, min(1, power))


class ABTestingFramework:
    """
    Comprehensive A/B Testing Framework for Traffic Signal Optimization
    
    Features:
    - Multi-variant testing support
    - Statistical significance testing
    - Real-time monitoring and analysis
    - Automatic test stopping criteria
    - Traffic splitting and assignment
    - Performance metrics tracking
    - Comprehensive reporting
    """
    
    def __init__(self, config: Optional[ABTestingConfig] = None):
        self.config = config or get_config().ab_testing
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.traffic_splitter = TrafficSplitter()
        self.statistical_analyzer = StatisticalAnalyzer()
        self.metrics_calculator = EnhancedPerformanceMetrics()
        
        # Test management
        self.active_tests: Dict[str, ABTestConfig] = {}
        self.test_results: Dict[str, List[TestResult]] = defaultdict(list)
        self.test_analyses: Dict[str, TestAnalysis] = {}
        
        # Monitoring
        self.monitoring_thread = None
        self.stop_monitoring = False
        
        self.logger.info("A/B Testing framework initialized")
    
    def create_test(self, test_config: ABTestConfig) -> str:
        """Create a new A/B test"""
        # Validate test configuration
        if not self._validate_test_config(test_config):
            raise ValueError("Invalid test configuration")
        
        # Normalize traffic splits
        self._normalize_traffic_splits(test_config.variants)
        
        # Store test configuration
        self.active_tests[test_config.test_id] = test_config
        self.test_results[test_config.test_id] = []
        
        self.logger.info(f"Created A/B test: {test_config.test_id}")
        return test_config.test_id
    
    def _validate_test_config(self, test_config: ABTestConfig) -> bool:
        """Validate test configuration"""
        # Check required fields
        if not test_config.test_id or not test_config.name:
            return False
        
        if len(test_config.variants) < 2:
            return False
        
        # Check traffic splits sum to 1.0
        total_split = sum(v.traffic_split for v in test_config.variants)
        if abs(total_split - 1.0) > 0.01:
            self.logger.warning("Traffic splits don't sum to 1.0, will normalize")
        
        return True
    
    def _normalize_traffic_splits(self, variants: List[TestVariant]):
        """Normalize traffic splits to sum to 1.0"""
        total_split = sum(v.traffic_split for v in variants)
        if total_split > 0:
            for variant in variants:
                variant.traffic_split /= total_split
        else:
            # Equal split if no splits specified
            equal_split = 1.0 / len(variants)
            for variant in variants:
                variant.traffic_split = equal_split
    
    def start_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        if test_id not in self.active_tests:
            self.logger.error(f"Test {test_id} not found")
            return False
        
        test_config = self.active_tests[test_id]
        
        # Start monitoring if not already running
        if not self.monitoring_thread or not self.monitoring_thread.is_alive():
            self.start_monitoring()
        
        self.logger.info(f"Started A/B test: {test_id}")
        return True
    
    def stop_test(self, test_id: str, reason: str = "manual") -> bool:
        """Stop an A/B test"""
        if test_id not in self.active_tests:
            self.logger.error(f"Test {test_id} not found")
            return False
        
        # Perform final analysis
        analysis = self.analyze_test(test_id)
        analysis.status = TestStatus.COMPLETED
        analysis.end_time = datetime.now()
        self.test_analyses[test_id] = analysis
        
        # Remove from active tests
        del self.active_tests[test_id]
        
        self.logger.info(f"Stopped A/B test: {test_id} - {reason}")
        return True
    
    def record_result(self, test_id: str, intersection_id: str, 
                     metrics: TrafficMetrics, user_id: Optional[str] = None,
                     session_id: Optional[str] = None) -> bool:
        """Record a test result"""
        if test_id not in self.active_tests:
            return False
        
        test_config = self.active_tests[test_id]
        
        # Assign variant
        variant_name = self.traffic_splitter.assign_variant(
            intersection_id, user_id, session_id, test_config.variants
        )
        
        # Create test result
        result = TestResult(
            timestamp=datetime.now(),
            intersection_id=intersection_id,
            variant=variant_name,
            metrics=metrics,
            user_id=user_id,
            session_id=session_id
        )
        
        # Store result
        self.test_results[test_id].append(result)
        
        return True
    
    def analyze_test(self, test_id: str) -> TestAnalysis:
        """Analyze A/B test results"""
        if test_id not in self.test_results:
            raise ValueError(f"No results found for test {test_id}")
        
        test_config = self.active_tests.get(test_id)
        results = self.test_results[test_id]
        
        if not results:
            return TestAnalysis(
                test_id=test_id,
                status=TestStatus.PLANNING,
                start_time=datetime.now(),
                end_time=None,
                total_samples=0,
                variant_results={},
                statistical_significance={},
                confidence_intervals={},
                effect_sizes={},
                recommendations=[],
                p_values={},
                power_analysis={}
            )
        
        # Group results by variant
        variant_data = defaultdict(list)
        for result in results:
            variant_data[result.variant].append(result)
        
        # Calculate variant statistics
        variant_results = {}
        for variant_name, variant_results_list in variant_data.items():
            if not variant_results_list:
                continue
            
            # Extract metrics
            metrics_data = {}
            for metric_name in test_config.target_metrics:
                values = [getattr(r.metrics, metric_name, 0) for r in variant_results_list]
                metrics_data[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'count': len(values),
                    'values': values
                }
            
            variant_results[variant_name] = metrics_data
        
        # Perform statistical tests
        statistical_significance = {}
        confidence_intervals = {}
        effect_sizes = {}
        p_values = {}
        
        # Find control variant
        control_variant = None
        for variant in test_config.variants:
            if variant.is_control:
                control_variant = variant.name
                break
        
        if not control_variant and variant_results:
            control_variant = list(variant_results.keys())[0]
        
        # Compare each variant to control
        for variant_name, variant_data in variant_results.items():
            if variant_name == control_variant:
                continue
            
            for metric_name in test_config.target_metrics:
                if metric_name not in variant_data or metric_name not in variant_results[control_variant]:
                    continue
                
                control_values = variant_results[control_variant][metric_name]['values']
                treatment_values = variant_data[metric_name]['values']
                
                if len(control_values) < 2 or len(treatment_values) < 2:
                    continue
                
                # Perform statistical test
                if test_config.statistical_test.test_type == 't_test':
                    test_result = self.statistical_analyzer.perform_t_test(
                        control_values, treatment_values
                    )
                elif test_config.statistical_test.test_type == 'mann_whitney':
                    test_result = self.statistical_analyzer.perform_mann_whitney(
                        control_values, treatment_values
                    )
                else:
                    continue
                
                if 'error' not in test_result:
                    key = f"{variant_name}_{metric_name}"
                    p_values[key] = test_result['p_value']
                    statistical_significance[key] = self.statistical_analyzer.determine_significance(
                        test_result['p_value'], test_config.statistical_test.alpha
                    )
                    confidence_intervals[key] = test_result.get('confidence_interval', (0, 0))
                    effect_sizes[key] = test_result.get('effect_size', 0)
        
        # Calculate power analysis
        power_analysis = {}
        for key, effect_size in effect_sizes.items():
            sample_size = len(results)
            power = self.statistical_analyzer.calculate_power(
                effect_size, sample_size, test_config.statistical_test.alpha
            )
            power_analysis[key] = power
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            test_config, variant_results, statistical_significance, effect_sizes
        )
        
        # Determine test status
        status = TestStatus.RUNNING
        if test_id not in self.active_tests:
            status = TestStatus.COMPLETED
        
        # Calculate test duration
        start_time = min(r.timestamp for r in results) if results else datetime.now()
        end_time = max(r.timestamp for r in results) if results else None
        
        return TestAnalysis(
            test_id=test_id,
            status=status,
            start_time=start_time,
            end_time=end_time,
            total_samples=len(results),
            variant_results=variant_results,
            statistical_significance=statistical_significance,
            confidence_intervals=confidence_intervals,
            effect_sizes=effect_sizes,
            recommendations=recommendations,
            p_values=p_values,
            power_analysis=power_analysis
        )
    
    def _generate_recommendations(self, test_config: ABTestConfig, 
                                variant_results: Dict[str, Dict[str, Any]],
                                statistical_significance: Dict[str, StatisticalSignificance],
                                effect_sizes: Dict[str, float]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check for significant improvements
        significant_improvements = []
        for key, significance in statistical_significance.items():
            if significance in [StatisticalSignificance.SIGNIFICANT, 
                              StatisticalSignificance.HIGHLY_SIGNIFICANT]:
                variant_name, metric_name = key.split('_', 1)
                effect_size = effect_sizes.get(key, 0)
                significant_improvements.append((variant_name, metric_name, effect_size))
        
        if significant_improvements:
            best_variant = max(significant_improvements, key=lambda x: x[2])
            recommendations.append(
                f"Variant '{best_variant[0]}' shows significant improvement in {best_variant[1]} "
                f"(effect size: {best_variant[2]:.3f})"
            )
        
        # Check sample size adequacy
        total_samples = sum(len(variant_data.get('wait_time', {}).get('values', [])) 
                          for variant_data in variant_results.values())
        if total_samples < test_config.min_sample_size:
            recommendations.append(
                f"Insufficient sample size ({total_samples}/{test_config.min_sample_size}). "
                "Consider running test longer."
            )
        
        # Check power analysis
        low_power_tests = [key for key, power in self.test_analyses.get(test_config.test_id, 
                                                                       TestAnalysis('', TestStatus.PLANNING, 
                                                                                   datetime.now(), None, 0, {}, {}, {}, {}, {}, [], {}, {})).power_analysis.items() 
                          if power < 0.8]
        if low_power_tests:
            recommendations.append(
                f"Low statistical power detected for {len(low_power_tests)} comparisons. "
                "Consider increasing sample size or effect size."
            )
        
        return recommendations
    
    def start_monitoring(self):
        """Start monitoring thread for active tests"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        self.logger.info("Started A/B testing monitoring")
    
    def stop_monitoring(self):
        """Stop monitoring thread"""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Stopped A/B testing monitoring")
    
    def _monitoring_loop(self):
        """Monitoring loop for active tests"""
        while not self.stop_monitoring:
            try:
                for test_id in list(self.active_tests.keys()):
                    self._check_test_stopping_criteria(test_id)
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _check_test_stopping_criteria(self, test_id: str):
        """Check if test should be stopped based on criteria"""
        test_config = self.active_tests[test_id]
        results = self.test_results[test_id]
        
        if not results:
            return
        
        # Check duration
        test_duration = datetime.now() - min(r.timestamp for r in results)
        if test_duration.total_seconds() > test_config.max_duration_hours * 3600:
            self.stop_test(test_id, "Maximum duration reached")
            return
        
        # Check sample size
        if len(results) >= test_config.min_sample_size:
            # Perform analysis
            analysis = self.analyze_test(test_id)
            
            # Check stopping criteria
            for criterion, threshold in test_config.stop_criteria.items():
                if criterion in analysis.effect_sizes:
                    if abs(analysis.effect_sizes[criterion]) > threshold:
                        self.stop_test(test_id, f"Stopping criterion met: {criterion}")
                        return
    
    def get_test_summary(self, test_id: str) -> Dict[str, Any]:
        """Get summary of test results"""
        if test_id not in self.test_results:
            return {}
        
        analysis = self.analyze_test(test_id)
        results = self.test_results[test_id]
        
        return {
            'test_id': test_id,
            'status': analysis.status.value,
            'total_samples': analysis.total_samples,
            'duration_hours': (analysis.end_time or datetime.now() - analysis.start_time).total_seconds() / 3600,
            'variants': list(analysis.variant_results.keys()),
            'significant_differences': len([s for s in analysis.statistical_significance.values() 
                                         if s in [StatisticalSignificance.SIGNIFICANT, 
                                                StatisticalSignificance.HIGHLY_SIGNIFICANT]]),
            'recommendations': analysis.recommendations
        }
    
    def export_test_results(self, test_id: str, filepath: str):
        """Export test results to file"""
        if test_id not in self.test_results:
            return
        
        analysis = self.analyze_test(test_id)
        results = self.test_results[test_id]
        
        export_data = {
            'test_config': {
                'test_id': test_id,
                'name': self.active_tests.get(test_id, {}).get('name', ''),
                'target_metrics': self.active_tests.get(test_id, {}).get('target_metrics', [])
            },
            'analysis': {
                'status': analysis.status.value,
                'total_samples': analysis.total_samples,
                'variant_results': analysis.variant_results,
                'statistical_significance': {k: v.value for k, v in analysis.statistical_significance.items()},
                'effect_sizes': analysis.effect_sizes,
                'recommendations': analysis.recommendations
            },
            'raw_results': [
                {
                    'timestamp': r.timestamp.isoformat(),
                    'intersection_id': r.intersection_id,
                    'variant': r.variant,
                    'metrics': {
                        'wait_time': r.metrics.wait_time,
                        'throughput': r.metrics.throughput,
                        'efficiency': r.metrics.efficiency
                    }
                }
                for r in results
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Test results exported to {filepath}")


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the A/B testing framework
    framework = ABTestingFramework()
    
    # Create test variants
    variants = [
        TestVariant(
            name="control",
            algorithm="websters_formula",
            parameters={},
            traffic_split=0.5,
            is_control=True,
            description="Baseline Webster's formula"
        ),
        TestVariant(
            name="q_learning",
            algorithm="q_learning",
            parameters={"learning_rate": 0.1, "epsilon": 0.1},
            traffic_split=0.5,
            description="Q-Learning optimization"
        )
    ]
    
    # Create test configuration
    test_config = ABTestConfig(
        test_id="test_001",
        name="Q-Learning vs Webster's Formula",
        description="Compare Q-Learning optimization with Webster's formula",
        variants=variants,
        target_metrics=["wait_time", "throughput", "efficiency"],
        statistical_test=StatisticalTest(test_type="t_test", alpha=0.05),
        duration_hours=24,
        min_sample_size=50
    )
    
    # Create and start test
    print("Creating A/B test...")
    test_id = framework.create_test(test_config)
    framework.start_test(test_id)
    
    # Simulate test results
    print("Simulating test results...")
    for i in range(100):
        # Generate mock metrics
        control_metrics = TrafficMetrics(
            timestamp=datetime.now(),
            intersection_id=f"junction_{i % 5}",
            wait_time=30 + np.random.normal(0, 5),
            throughput=500 + np.random.normal(0, 50),
            efficiency=0.7 + np.random.normal(0, 0.1)
        )
        
        q_learning_metrics = TrafficMetrics(
            timestamp=datetime.now(),
            intersection_id=f"junction_{i % 5}",
            wait_time=25 + np.random.normal(0, 4),  # Better performance
            throughput=600 + np.random.normal(0, 40),  # Better performance
            efficiency=0.8 + np.random.normal(0, 0.08)  # Better performance
        )
        
        # Record results
        framework.record_result(test_id, f"junction_{i % 5}", control_metrics)
        framework.record_result(test_id, f"junction_{i % 5}", q_learning_metrics)
    
    # Analyze test
    print("Analyzing test results...")
    analysis = framework.analyze_test(test_id)
    print(f"Analysis: {analysis}")
    
    # Get test summary
    summary = framework.get_test_summary(test_id)
    print(f"Test summary: {summary}")
    
    # Stop test
    framework.stop_test(test_id, "Test completed")
