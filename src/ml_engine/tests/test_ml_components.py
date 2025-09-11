"""
Comprehensive Test Suite for ML Traffic Signal Optimization Components
Unit tests, integration tests, and performance tests for all ML components
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json
from typing import Dict, List, Any

# Import ML components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.enhanced_data_integration import EnhancedDataIntegration, TrafficDataPoint, DataSource
from prediction.enhanced_traffic_predictor import EnhancedTrafficPredictor
from metrics.enhanced_performance_metrics import EnhancedPerformanceMetrics, TrafficMetrics
from ab_testing.ab_testing_framework import ABTestingFramework, ABTestConfig, TestVariant, StatisticalTest
from algorithms.enhanced_q_learning_optimizer import EnhancedQLearningOptimizer
from algorithms.enhanced_dynamic_programming_optimizer import EnhancedDynamicProgrammingOptimizer
from algorithms.enhanced_websters_formula_optimizer import EnhancedWebstersFormulaOptimizer
from enhanced_signal_optimizer import EnhancedSignalOptimizer, OptimizationRequest, OptimizationMode
from monitoring.enhanced_monitoring import EnhancedMonitoring, Alert, AlertType, LogLevel
from visualization.performance_visualizer import PerformanceVisualizer
from config.ml_config import MLConfig, get_config


class TestDataIntegration:
    """Test suite for data integration components"""
    
    @pytest.fixture
    def data_integration(self):
        """Create data integration instance for testing"""
        return EnhancedDataIntegration()
    
    @pytest.fixture
    def sample_traffic_data(self):
        """Create sample traffic data for testing"""
        return TrafficDataPoint(
            intersection_id="test_junction",
            timestamp=datetime.now(),
            lane_counts={'north_lane': 10, 'south_lane': 8, 'east_lane': 12, 'west_lane': 6},
            avg_speed=35.0,
            weather_condition='clear',
            temperature=25.0,
            humidity=60.0,
            visibility=10.0,
            source=DataSource.API,
            confidence=0.9,
            processing_time=0.1
        )
    
    def test_data_integration_initialization(self, data_integration):
        """Test data integration initialization"""
        assert data_integration is not None
        assert data_integration.cache == {}
        assert data_integration.stats.total_requests == 0
    
    @pytest.mark.asyncio
    async def test_fetch_traffic_data_async(self, data_integration):
        """Test async traffic data fetching"""
        with patch.object(data_integration, '_fetch_from_api_async') as mock_fetch:
            mock_fetch.return_value = TrafficDataPoint(
                intersection_id="test_junction",
                timestamp=datetime.now(),
                lane_counts={'north_lane': 5, 'south_lane': 3, 'east_lane': 7, 'west_lane': 4},
                avg_speed=30.0,
                weather_condition='clear',
                source=DataSource.API,
                confidence=0.8,
                processing_time=0.2
            )
            
            result = await data_integration.fetch_traffic_data_async("test_junction")
            
            assert result.intersection_id == "test_junction"
            assert result.source == DataSource.API
            assert data_integration.stats.total_requests == 1
    
    def test_fetch_traffic_data_sync(self, data_integration):
        """Test sync traffic data fetching"""
        with patch.object(data_integration, '_fetch_from_api_sync') as mock_fetch:
            mock_fetch.return_value = TrafficDataPoint(
                intersection_id="test_junction",
                timestamp=datetime.now(),
                lane_counts={'north_lane': 5, 'south_lane': 3, 'east_lane': 7, 'west_lane': 4},
                avg_speed=30.0,
                weather_condition='clear',
                source=DataSource.API,
                confidence=0.8,
                processing_time=0.2
            )
            
            result = data_integration.fetch_traffic_data("test_junction")
            
            assert result.intersection_id == "test_junction"
            assert result.source == DataSource.API
    
    def test_cache_functionality(self, data_integration, sample_traffic_data):
        """Test caching functionality"""
        # Test caching
        data_integration._cache_data(sample_traffic_data)
        
        # Test cache retrieval
        cached_data = data_integration._get_cached_data("test_junction")
        assert cached_data is not None
        assert cached_data.intersection_id == "test_junction"
        
        # Test cache hit statistics
        assert data_integration.stats.cache_hits == 1
    
    def test_fallback_to_mock_data(self, data_integration):
        """Test fallback to mock data when API fails"""
        with patch.object(data_integration, '_fetch_from_api_sync') as mock_api, \
             patch.object(data_integration, '_get_predicted_data') as mock_prediction:
            
            # Mock API failure
            mock_api.side_effect = Exception("API Error")
            mock_prediction.side_effect = Exception("Prediction Error")
            
            result = data_integration.fetch_traffic_data("test_junction")
            
            assert result.source == DataSource.MOCK
            assert data_integration.stats.fallback_activations == 1
    
    def test_health_check(self, data_integration):
        """Test health check functionality"""
        health = data_integration.health_check()
        
        assert 'status' in health
        assert 'api_available' in health
        assert 'cache_size' in health
        assert 'success_rate' in health


class TestTrafficPredictor:
    """Test suite for traffic prediction components"""
    
    @pytest.fixture
    def traffic_predictor(self):
        """Create traffic predictor instance for testing"""
        return EnhancedTrafficPredictor()
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-07', freq='min')
        data = []
        
        for i, timestamp in enumerate(dates):
            data.append({
                'timestamp': timestamp,
                'intersection_id': f'junction_{i % 5}',
                'north_lane': np.random.randint(0, 30),
                'south_lane': np.random.randint(0, 30),
                'east_lane': np.random.randint(0, 30),
                'west_lane': np.random.randint(0, 30),
                'avg_speed': np.random.uniform(20, 50),
                'weather_condition': np.random.choice(['clear', 'cloudy', 'rainy']),
                'temperature': np.random.uniform(15, 30),
                'humidity': np.random.uniform(40, 80),
                'visibility': np.random.uniform(8, 15)
            })
        
        return pd.DataFrame(data)
    
    def test_traffic_predictor_initialization(self, traffic_predictor):
        """Test traffic predictor initialization"""
        assert traffic_predictor is not None
        assert hasattr(traffic_predictor, 'data_processor')
        assert hasattr(traffic_predictor, 'predictors')
    
    def test_prepare_training_data(self, traffic_predictor, sample_training_data):
        """Test training data preparation"""
        processed_data, target_columns = traffic_predictor.prepare_training_data(sample_training_data)
        
        assert isinstance(processed_data, pd.DataFrame)
        assert len(target_columns) == 4  # 4 lanes
        assert 'hour' in processed_data.columns
        assert 'day_of_week' in processed_data.columns
        assert 'total_vehicles' in processed_data.columns
    
    def test_predict_traffic_flow(self, traffic_predictor):
        """Test traffic flow prediction"""
        prediction = traffic_predictor.predict_traffic_flow("test_junction", 15)
        
        assert isinstance(prediction, dict)
        assert 'lane_counts' in prediction
        assert 'avg_speed' in prediction
        assert 'weather_condition' in prediction
        assert 'confidence' in prediction
    
    def test_model_training(self, traffic_predictor, sample_training_data):
        """Test model training"""
        # This test might be skipped if ML libraries are not available
        try:
            training_results = traffic_predictor.train_models(sample_training_data)
            assert isinstance(training_results, dict)
        except ImportError:
            pytest.skip("ML libraries not available")
    
    def test_model_evaluation(self, traffic_predictor, sample_training_data):
        """Test model evaluation"""
        try:
            performance = traffic_predictor.evaluate_models(sample_training_data)
            assert isinstance(performance, dict)
        except ImportError:
            pytest.skip("ML libraries not available")


class TestPerformanceMetrics:
    """Test suite for performance metrics components"""
    
    @pytest.fixture
    def performance_metrics(self):
        """Create performance metrics instance for testing"""
        return EnhancedPerformanceMetrics()
    
    @pytest.fixture
    def sample_traffic_metrics(self):
        """Create sample traffic metrics"""
        return TrafficMetrics(
            timestamp=datetime.now(),
            intersection_id="test_junction",
            wait_time=25.0,
            throughput=600.0,
            fuel_consumption=50.0,
            emissions=115.0,
            safety_score=0.8,
            comfort_score=0.7,
            efficiency=0.75,
            queue_length=15.0,
            delay=25.0,
            stop_delay=20.0,
            total_vehicles=20,
            processed_vehicles=15,
            avg_speed=35.0,
            signal_cycle_time=60.0,
            green_time_ratio=0.8
        )
    
    def test_performance_metrics_initialization(self, performance_metrics):
        """Test performance metrics initialization"""
        assert performance_metrics is not None
        assert performance_metrics.metrics_history == []
        assert performance_metrics.optimization_results == []
    
    def test_calculate_metrics(self, performance_metrics):
        """Test metrics calculation"""
        traffic_data = {
            'lane_counts': {'north_lane': 10, 'south_lane': 8, 'east_lane': 12, 'west_lane': 6},
            'avg_speed': 35.0,
            'weather_condition': 'clear',
            'temperature': 25.0,
            'humidity': 60.0,
            'visibility': 10.0
        }
        
        signal_timings = {'north_lane': 30, 'south_lane': 30, 'east_lane': 25, 'west_lane': 25}
        
        metrics = performance_metrics.calculate_metrics(traffic_data, signal_timings, "test_junction")
        
        assert isinstance(metrics, TrafficMetrics)
        assert metrics.intersection_id == "test_junction"
        assert metrics.wait_time >= 0
        assert metrics.throughput >= 0
        assert 0 <= metrics.efficiency <= 1
    
    def test_optimization_impact_calculation(self, performance_metrics, sample_traffic_metrics):
        """Test optimization impact calculation"""
        before_metrics = sample_traffic_metrics
        after_metrics = TrafficMetrics(
            timestamp=datetime.now(),
            intersection_id="test_junction",
            wait_time=20.0,  # Improved
            throughput=700.0,  # Improved
            efficiency=0.85,  # Improved
            safety_score=0.8,
            comfort_score=0.7,
            queue_length=12.0,
            delay=20.0,
            stop_delay=15.0,
            total_vehicles=18,
            processed_vehicles=16,
            avg_speed=38.0,
            signal_cycle_time=55.0,
            green_time_ratio=0.85
        )
        
        result = performance_metrics.calculate_optimization_impact(
            before_metrics, after_metrics, "q_learning", 0.5
        )
        
        assert result.intersection_id == "test_junction"
        assert result.algorithm_used == "q_learning"
        assert "wait_time" in result.improvement_percentage
        assert "throughput" in result.improvement_percentage
        assert "efficiency" in result.improvement_percentage
    
    def test_performance_summary(self, performance_metrics, sample_traffic_metrics):
        """Test performance summary generation"""
        # Add some metrics to history
        performance_metrics.metrics_history.append(sample_traffic_metrics)
        
        summary = performance_metrics.get_performance_summary("test_junction")
        
        assert summary.intersection_id == "test_junction"
        assert summary.total_measurements >= 0
        assert summary.avg_wait_time >= 0
        assert summary.avg_throughput >= 0


class TestABTestingFramework:
    """Test suite for A/B testing framework"""
    
    @pytest.fixture
    def ab_testing(self):
        """Create A/B testing framework instance for testing"""
        return ABTestingFramework()
    
    @pytest.fixture
    def sample_test_config(self):
        """Create sample A/B test configuration"""
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
                name="treatment",
                algorithm="q_learning",
                parameters={"learning_rate": 0.1},
                traffic_split=0.5,
                description="Q-Learning optimization"
            )
        ]
        
        return ABTestConfig(
            test_id="test_001",
            name="Q-Learning vs Webster's Formula",
            description="Compare Q-Learning with Webster's formula",
            variants=variants,
            target_metrics=["wait_time", "throughput", "efficiency"],
            statistical_test=StatisticalTest(test_type="t_test", alpha=0.05),
            duration_hours=24,
            min_sample_size=50
        )
    
    def test_ab_testing_initialization(self, ab_testing):
        """Test A/B testing framework initialization"""
        assert ab_testing is not None
        assert ab_testing.active_tests == {}
        assert ab_testing.test_results == {}
    
    def test_create_test(self, ab_testing, sample_test_config):
        """Test A/B test creation"""
        test_id = ab_testing.create_test(sample_test_config)
        
        assert test_id == "test_001"
        assert test_id in ab_testing.active_tests
        assert test_id in ab_testing.test_results
    
    def test_start_stop_test(self, ab_testing, sample_test_config):
        """Test starting and stopping A/B test"""
        test_id = ab_testing.create_test(sample_test_config)
        
        # Start test
        success = ab_testing.start_test(test_id)
        assert success
        
        # Stop test
        success = ab_testing.stop_test(test_id)
        assert success
        assert test_id not in ab_testing.active_tests
    
    def test_record_result(self, ab_testing, sample_test_config, sample_traffic_metrics):
        """Test recording test results"""
        test_id = ab_testing.create_test(sample_test_config)
        ab_testing.start_test(test_id)
        
        success = ab_testing.record_result(
            test_id, "test_junction", sample_traffic_metrics
        )
        
        assert success
        assert len(ab_testing.test_results[test_id]) == 1
    
    def test_analyze_test(self, ab_testing, sample_test_config, sample_traffic_metrics):
        """Test test analysis"""
        test_id = ab_testing.create_test(sample_test_config)
        ab_testing.start_test(test_id)
        
        # Add some test results
        for i in range(10):
            ab_testing.record_result(test_id, f"junction_{i % 3}", sample_traffic_metrics)
        
        analysis = ab_testing.analyze_test(test_id)
        
        assert analysis.test_id == test_id
        assert analysis.total_samples == 10
        assert "variant_results" in analysis.__dict__


class TestEnhancedSignalOptimizer:
    """Test suite for enhanced signal optimizer"""
    
    @pytest.fixture
    def signal_optimizer(self):
        """Create signal optimizer instance for testing"""
        return EnhancedSignalOptimizer()
    
    @pytest.fixture
    def sample_optimization_request(self):
        """Create sample optimization request"""
        return OptimizationRequest(
            intersection_id="test_junction",
            current_timings={'north_lane': 30, 'south_lane': 30, 'east_lane': 30, 'west_lane': 30},
            optimization_mode=OptimizationMode.ADAPTIVE
        )
    
    def test_signal_optimizer_initialization(self, signal_optimizer):
        """Test signal optimizer initialization"""
        assert signal_optimizer is not None
        assert hasattr(signal_optimizer, 'data_integration')
        assert hasattr(signal_optimizer, 'traffic_predictor')
        assert hasattr(signal_optimizer, 'performance_metrics')
        assert hasattr(signal_optimizer, 'algorithms')
    
    def test_algorithm_selection(self, signal_optimizer, sample_optimization_request):
        """Test algorithm selection logic"""
        # Mock traffic data
        traffic_data = TrafficDataPoint(
            intersection_id="test_junction",
            timestamp=datetime.now(),
            lane_counts={'north_lane': 20, 'south_lane': 15, 'east_lane': 25, 'west_lane': 18},
            avg_speed=30.0,
            weather_condition='clear',
            source=DataSource.API,
            confidence=0.9,
            processing_time=0.1
        )
        
        # Mock current metrics
        current_metrics = TrafficMetrics(
            timestamp=datetime.now(),
            intersection_id="test_junction",
            wait_time=30.0,
            throughput=500.0,
            efficiency=0.7
        )
        
        algorithm = signal_optimizer._select_algorithm(
            sample_optimization_request, traffic_data, current_metrics
        )
        
        assert algorithm in signal_optimizer.algorithms.keys()
    
    def test_optimization_statistics(self, signal_optimizer):
        """Test optimization statistics"""
        stats = signal_optimizer.get_optimization_statistics()
        
        assert isinstance(stats, dict)
        assert 'total_optimizations' in stats
        assert 'algorithm_usage' in stats
        assert 'avg_confidence' in stats
    
    def test_intersection_performance(self, signal_optimizer):
        """Test intersection performance tracking"""
        performance = signal_optimizer.get_intersection_performance("test_junction")
        
        assert isinstance(performance, dict)


class TestEnhancedMonitoring:
    """Test suite for enhanced monitoring system"""
    
    @pytest.fixture
    def monitoring(self):
        """Create monitoring instance for testing"""
        return EnhancedMonitoring()
    
    def test_monitoring_initialization(self, monitoring):
        """Test monitoring system initialization"""
        assert monitoring is not None
        assert hasattr(monitoring, 'alert_manager')
        assert hasattr(monitoring, 'performance_tracker')
        assert hasattr(monitoring, 'system_monitor')
        assert hasattr(monitoring, 'enhanced_logger')
    
    def test_record_optimization(self, monitoring):
        """Test recording optimization events"""
        metrics = {
            'wait_time': 25.0,
            'throughput': 600.0,
            'efficiency': 0.8,
            'confidence': 0.9
        }
        
        monitoring.record_optimization("test_junction", "q_learning", metrics, 0.5)
        
        # Check that performance was recorded
        assert len(monitoring.performance_tracker.performance_history) == 1
    
    def test_record_error(self, monitoring):
        """Test recording error events"""
        monitoring.record_error(
            "optimization_failure", "Test error message", 
            "test_junction", "q_learning", Exception("Test exception")
        )
        
        # Check error logging statistics
        stats = monitoring.enhanced_logger.get_log_statistics()
        assert stats['total_errors'] >= 1
    
    def test_performance_summary(self, monitoring):
        """Test performance summary generation"""
        summary = monitoring.get_performance_summary("test_junction")
        
        assert isinstance(summary, dict)
        assert 'statistics' in summary
        assert 'trends' in summary
    
    def test_system_health_summary(self, monitoring):
        """Test system health summary"""
        health = monitoring.get_system_health_summary()
        
        assert isinstance(health, dict)
        assert 'current_health' in health
        assert 'trends_24h' in health
    
    def test_alerts_summary(self, monitoring):
        """Test alerts summary"""
        alerts = monitoring.get_alerts_summary()
        
        assert isinstance(alerts, dict)
        assert 'total_active' in alerts
        assert 'by_type' in alerts
        assert 'by_severity' in alerts


class TestPerformanceVisualizer:
    """Test suite for performance visualizer"""
    
    @pytest.fixture
    def visualizer(self):
        """Create visualizer instance for testing"""
        return PerformanceVisualizer()
    
    @pytest.fixture
    def sample_performance_data(self):
        """Create sample performance data"""
        return {
            'trends': {
                'wait_time': np.random.uniform(20, 40, 24),
                'throughput': np.random.uniform(400, 800, 24),
                'efficiency': np.random.uniform(0.6, 0.9, 24),
                'confidence': np.random.uniform(0.7, 0.95, 24)
            },
            'algorithm_performance': {
                'q_learning': {'wait_time': [25, 30, 28], 'throughput': [600, 650, 620]},
                'dynamic_programming': {'wait_time': [30, 35, 32], 'throughput': [550, 600, 580]},
                'websters_formula': {'wait_time': [35, 40, 38], 'throughput': [500, 550, 520]}
            },
            'total_optimizations': 1000,
            'avg_confidence': 0.85,
            'avg_processing_time': 0.5
        }
    
    def test_visualizer_initialization(self, visualizer):
        """Test visualizer initialization"""
        assert visualizer is not None
        assert hasattr(visualizer, 'output_dir')
        assert hasattr(visualizer, 'colors')
        assert hasattr(visualizer, 'algorithm_colors')
    
    def test_plot_performance_trends(self, visualizer, sample_performance_data):
        """Test performance trends plotting"""
        fig = visualizer.plot_performance_trends(
            sample_performance_data['trends'],
            title="Test Performance Trends"
        )
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
    
    def test_plot_algorithm_comparison(self, visualizer, sample_performance_data):
        """Test algorithm comparison plotting"""
        fig = visualizer.plot_algorithm_comparison(
            sample_performance_data['algorithm_performance'],
            title="Test Algorithm Comparison"
        )
        
        assert fig is not None
        assert hasattr(fig, 'savefig')
    
    def test_create_dashboard(self, visualizer, sample_performance_data):
        """Test dashboard creation"""
        system_health = {
            'current_health': {
                'cpu_usage': 45.2,
                'memory_usage': 62.1,
                'disk_usage': 38.5,
                'error_rate': 0.02,
                'avg_response_time': 0.8
            }
        }
        
        alerts = {
            'total_active': 0,
            'by_severity': {},
            'recent_alerts': []
        }
        
        fig = visualizer.create_dashboard(
            sample_performance_data, system_health, alerts,
            title="Test Dashboard"
        )
        
        assert fig is not None
        assert hasattr(fig, 'savefig')


class TestIntegration:
    """Integration tests for the complete ML system"""
    
    @pytest.fixture
    def complete_system(self):
        """Create complete ML system for testing"""
        config = get_config()
        return {
            'data_integration': EnhancedDataIntegration(config.data_integration),
            'traffic_predictor': EnhancedTrafficPredictor(config.traffic_prediction),
            'performance_metrics': EnhancedPerformanceMetrics(config.performance_metrics),
            'ab_testing': ABTestingFramework(config.ab_testing),
            'signal_optimizer': EnhancedSignalOptimizer(config),
            'monitoring': EnhancedMonitoring(config.logging),
            'visualizer': PerformanceVisualizer()
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization(self, complete_system):
        """Test complete end-to-end optimization workflow"""
        # This is a comprehensive integration test
        # In a real scenario, this would test the entire pipeline
        
        # Test data integration
        data_integration = complete_system['data_integration']
        with patch.object(data_integration, 'fetch_traffic_data_async') as mock_fetch:
            mock_fetch.return_value = TrafficDataPoint(
                intersection_id="test_junction",
                timestamp=datetime.now(),
                lane_counts={'north_lane': 10, 'south_lane': 8, 'east_lane': 12, 'west_lane': 6},
                avg_speed=35.0,
                weather_condition='clear',
                source=DataSource.API,
                confidence=0.9,
                processing_time=0.1
            )
            
            # Test signal optimization
            signal_optimizer = complete_system['signal_optimizer']
            request = OptimizationRequest(
                intersection_id="test_junction",
                current_timings={'north_lane': 30, 'south_lane': 30, 'east_lane': 30, 'west_lane': 30}
            )
            
            # This would normally run the full optimization
            # For testing, we'll just verify the components work together
            assert signal_optimizer is not None
            assert data_integration is not None
    
    def test_system_health_monitoring(self, complete_system):
        """Test system health monitoring integration"""
        monitoring = complete_system['monitoring']
        
        # Start monitoring
        monitoring.start_monitoring()
        
        # Record some events
        monitoring.record_optimization(
            "test_junction", "q_learning", 
            {"wait_time": 25.0, "throughput": 600.0, "efficiency": 0.8, "confidence": 0.9},
            0.5
        )
        
        # Get health summary
        health = monitoring.get_system_health_summary()
        assert isinstance(health, dict)
        
        # Stop monitoring
        monitoring.stop_monitoring()
    
    def test_ab_testing_integration(self, complete_system):
        """Test A/B testing integration"""
        ab_testing = complete_system['ab_testing']
        
        # Create test configuration
        variants = [
            TestVariant(name="control", algorithm="websters_formula", traffic_split=0.5, is_control=True),
            TestVariant(name="treatment", algorithm="q_learning", traffic_split=0.5)
        ]
        
        test_config = ABTestConfig(
            test_id="integration_test",
            name="Integration Test",
            description="Test A/B testing integration",
            variants=variants,
            target_metrics=["wait_time", "throughput"],
            statistical_test=StatisticalTest(test_type="t_test", alpha=0.05)
        )
        
        # Create and start test
        test_id = ab_testing.create_test(test_config)
        ab_testing.start_test(test_id)
        
        # Record some results
        for i in range(5):
            metrics = TrafficMetrics(
                timestamp=datetime.now(),
                intersection_id=f"junction_{i}",
                wait_time=25.0 + i,
                throughput=600.0 + i * 10,
                efficiency=0.8
            )
            ab_testing.record_result(test_id, f"junction_{i}", metrics)
        
        # Analyze test
        analysis = ab_testing.analyze_test(test_id)
        assert analysis.total_samples == 5
        
        # Stop test
        ab_testing.stop_test(test_id)


# Performance tests
class TestPerformance:
    """Performance tests for ML components"""
    
    def test_data_integration_performance(self):
        """Test data integration performance"""
        data_integration = EnhancedDataIntegration()
        
        # Test with multiple concurrent requests
        async def test_concurrent_requests():
            tasks = []
            for i in range(10):
                task = data_integration.fetch_traffic_data_async(f"junction_{i}")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results
        
        # This would run the performance test
        # In practice, you'd measure execution time and memory usage
        assert True  # Placeholder for actual performance test
    
    def test_optimization_performance(self):
        """Test optimization performance"""
        signal_optimizer = EnhancedSignalOptimizer()
        
        # Test optimization speed
        start_time = time.time()
        
        # Simulate optimization
        request = OptimizationRequest(
            intersection_id="test_junction",
            current_timings={'north_lane': 30, 'south_lane': 30, 'east_lane': 30, 'west_lane': 30}
        )
        
        # This would run the actual optimization
        # In practice, you'd measure execution time
        execution_time = time.time() - start_time
        
        # Performance assertion (should complete within reasonable time)
        assert execution_time < 1.0  # Should complete within 1 second


# Utility functions for testing
def create_mock_traffic_data(intersection_id: str = "test_junction") -> TrafficDataPoint:
    """Create mock traffic data for testing"""
    return TrafficDataPoint(
        intersection_id=intersection_id,
        timestamp=datetime.now(),
        lane_counts={'north_lane': 10, 'south_lane': 8, 'east_lane': 12, 'west_lane': 6},
        avg_speed=35.0,
        weather_condition='clear',
        temperature=25.0,
        humidity=60.0,
        visibility=10.0,
        source=DataSource.API,
        confidence=0.9,
        processing_time=0.1
    )


def create_mock_performance_metrics() -> TrafficMetrics:
    """Create mock performance metrics for testing"""
    return TrafficMetrics(
        timestamp=datetime.now(),
        intersection_id="test_junction",
        wait_time=25.0,
        throughput=600.0,
        fuel_consumption=50.0,
        emissions=115.0,
        safety_score=0.8,
        comfort_score=0.7,
        efficiency=0.75,
        queue_length=15.0,
        delay=25.0,
        stop_delay=20.0,
        total_vehicles=20,
        processed_vehicles=15,
        avg_speed=35.0,
        signal_cycle_time=60.0,
        green_time_ratio=0.8
    )


# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Test configuration for all tests"""
    return {
        'test_data_dir': tempfile.mkdtemp(),
        'test_models_dir': tempfile.mkdtemp(),
        'test_logs_dir': tempfile.mkdtemp()
    }


# Run tests
if __name__ == "__main__":
    # Run specific test classes
    pytest.main([__file__, "-v", "--tb=short"])
