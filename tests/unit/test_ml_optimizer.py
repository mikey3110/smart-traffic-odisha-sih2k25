"""
Unit tests for ML Optimizer components
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.ml_engine.signal_optimizer import SignalOptimizer
from src.ml_engine.algorithms.q_learning_optimizer import QLearningOptimizer
from src.ml_engine.algorithms.dynamic_programming_optimizer import DynamicProgrammingOptimizer
from src.ml_engine.algorithms.websters_formula_optimizer import WebstersFormulaOptimizer
from src.ml_engine.prediction.traffic_predictor import TrafficPredictor
from src.ml_engine.metrics.performance_metrics import PerformanceMetrics

class TestSignalOptimizer:
    """Test SignalOptimizer class"""
    
    def test_signal_optimizer_initialization(self):
        """Test SignalOptimizer initialization"""
        optimizer = SignalOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, 'q_learning_optimizer')
        assert hasattr(optimizer, 'dp_optimizer')
        assert hasattr(optimizer, 'websters_optimizer')
    
    def test_optimize_with_q_learning(self, sample_traffic_data):
        """Test optimization using Q-learning algorithm"""
        optimizer = SignalOptimizer()
        
        with patch.object(optimizer.q_learning_optimizer, 'optimize') as mock_optimize:
            mock_optimize.return_value = {
                'optimized_phases': [30, 25, 35, 20],
                'efficiency_improvement': 0.15,
                'waiting_time_reduction': 12.5
            }
            
            result = optimizer.optimize(sample_traffic_data, algorithm='q_learning')
            
            assert result['optimized_phases'] == [30, 25, 35, 20]
            assert result['efficiency_improvement'] == 0.15
            assert result['waiting_time_reduction'] == 12.5
            mock_optimize.assert_called_once_with(sample_traffic_data)
    
    def test_optimize_with_dynamic_programming(self, sample_traffic_data):
        """Test optimization using dynamic programming algorithm"""
        optimizer = SignalOptimizer()
        
        with patch.object(optimizer.dp_optimizer, 'optimize') as mock_optimize:
            mock_optimize.return_value = {
                'optimized_phases': [35, 30, 40, 25],
                'efficiency_improvement': 0.12,
                'waiting_time_reduction': 10.0
            }
            
            result = optimizer.optimize(sample_traffic_data, algorithm='dynamic_programming')
            
            assert result['optimized_phases'] == [35, 30, 40, 25]
            assert result['efficiency_improvement'] == 0.12
            assert result['waiting_time_reduction'] == 10.0
            mock_optimize.assert_called_once_with(sample_traffic_data)
    
    def test_optimize_with_websters_formula(self, sample_traffic_data):
        """Test optimization using Webster's formula algorithm"""
        optimizer = SignalOptimizer()
        
        with patch.object(optimizer.websters_optimizer, 'optimize') as mock_optimize:
            mock_optimize.return_value = {
                'optimized_phases': [28, 22, 32, 18],
                'efficiency_improvement': 0.08,
                'waiting_time_reduction': 8.0
            }
            
            result = optimizer.optimize(sample_traffic_data, algorithm='websters_formula')
            
            assert result['optimized_phases'] == [28, 22, 32, 18]
            assert result['efficiency_improvement'] == 0.08
            assert result['waiting_time_reduction'] == 8.0
            mock_optimize.assert_called_once_with(sample_traffic_data)
    
    def test_optimize_invalid_algorithm(self, sample_traffic_data):
        """Test optimization with invalid algorithm"""
        optimizer = SignalOptimizer()
        
        with pytest.raises(ValueError, match="Invalid algorithm"):
            optimizer.optimize(sample_traffic_data, algorithm='invalid_algorithm')
    
    def test_optimize_empty_data(self):
        """Test optimization with empty data"""
        optimizer = SignalOptimizer()
        
        result = optimizer.optimize([])
        assert result == {
            'optimized_phases': [30, 30, 30, 30],  # Default phases
            'efficiency_improvement': 0.0,
            'waiting_time_reduction': 0.0
        }

class TestQLearningOptimizer:
    """Test QLearningOptimizer class"""
    
    def test_q_learning_initialization(self):
        """Test QLearningOptimizer initialization"""
        optimizer = QLearningOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, 'q_table')
        assert hasattr(optimizer, 'learning_rate')
        assert hasattr(optimizer, 'discount_factor')
        assert hasattr(optimizer, 'epsilon')
    
    def test_q_learning_optimize(self, sample_traffic_data):
        """Test Q-learning optimization"""
        optimizer = QLearningOptimizer()
        
        result = optimizer.optimize(sample_traffic_data)
        
        assert 'optimized_phases' in result
        assert 'efficiency_improvement' in result
        assert 'waiting_time_reduction' in result
        assert isinstance(result['optimized_phases'], list)
        assert len(result['optimized_phases']) == 4  # 4 phases
        assert all(phase > 0 for phase in result['optimized_phases'])
    
    def test_q_learning_learning_process(self, sample_traffic_data):
        """Test Q-learning learning process"""
        optimizer = QLearningOptimizer()
        
        # Run optimization multiple times to test learning
        for _ in range(10):
            result = optimizer.optimize(sample_traffic_data)
            assert result['efficiency_improvement'] >= 0
    
    def test_q_learning_epsilon_decay(self):
        """Test epsilon decay in Q-learning"""
        optimizer = QLearningOptimizer(initial_epsilon=1.0, epsilon_decay=0.9)
        
        initial_epsilon = optimizer.epsilon
        optimizer._decay_epsilon()
        assert optimizer.epsilon < initial_epsilon
    
    def test_q_learning_state_encoding(self, sample_traffic_data):
        """Test state encoding in Q-learning"""
        optimizer = QLearningOptimizer()
        
        state = optimizer._encode_state(sample_traffic_data)
        assert isinstance(state, tuple)
        assert len(state) > 0
    
    def test_q_learning_action_selection(self):
        """Test action selection in Q-learning"""
        optimizer = QLearningOptimizer()
        
        # Test greedy action selection
        action = optimizer._select_action((1, 2, 3, 4))
        assert isinstance(action, int)
        assert 0 <= action < 4  # 4 possible actions
    
    def test_q_learning_reward_calculation(self, sample_traffic_data):
        """Test reward calculation in Q-learning"""
        optimizer = QLearningOptimizer()
        
        reward = optimizer._calculate_reward(sample_traffic_data, 0)
        assert isinstance(reward, float)
        assert -1 <= reward <= 1  # Normalized reward

class TestDynamicProgrammingOptimizer:
    """Test DynamicProgrammingOptimizer class"""
    
    def test_dp_optimizer_initialization(self):
        """Test DynamicProgrammingOptimizer initialization"""
        optimizer = DynamicProgrammingOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, 'max_phases')
        assert hasattr(optimizer, 'min_phases')
    
    def test_dp_optimize(self, sample_traffic_data):
        """Test dynamic programming optimization"""
        optimizer = DynamicProgrammingOptimizer()
        
        result = optimizer.optimize(sample_traffic_data)
        
        assert 'optimized_phases' in result
        assert 'efficiency_improvement' in result
        assert 'waiting_time_reduction' in result
        assert isinstance(result['optimized_phases'], list)
        assert len(result['optimized_phases']) == 4
        assert all(phase > 0 for phase in result['optimized_phases'])
    
    def test_dp_cost_calculation(self, sample_traffic_data):
        """Test cost calculation in dynamic programming"""
        optimizer = DynamicProgrammingOptimizer()
        
        cost = optimizer._calculate_cost(sample_traffic_data, [30, 25, 35, 20])
        assert isinstance(cost, float)
        assert cost >= 0
    
    def test_dp_phase_validation(self):
        """Test phase validation in dynamic programming"""
        optimizer = DynamicProgrammingOptimizer()
        
        # Valid phases
        assert optimizer._validate_phases([30, 25, 35, 20]) is True
        
        # Invalid phases (negative)
        assert optimizer._validate_phases([-10, 25, 35, 20]) is False
        
        # Invalid phases (too short)
        assert optimizer._validate_phases([5, 5, 5, 5]) is False
    
    def test_dp_optimization_algorithm(self, sample_traffic_data):
        """Test the core optimization algorithm"""
        optimizer = DynamicProgrammingOptimizer()
        
        phases = optimizer._optimize_phases(sample_traffic_data)
        assert isinstance(phases, list)
        assert len(phases) == 4
        assert all(phase >= optimizer.min_phases for phase in phases)
        assert all(phase <= optimizer.max_phases for phase in phases)

class TestWebstersFormulaOptimizer:
    """Test WebstersFormulaOptimizer class"""
    
    def test_websters_initialization(self):
        """Test WebstersFormulaOptimizer initialization"""
        optimizer = WebstersFormulaOptimizer()
        assert optimizer is not None
        assert hasattr(optimizer, 'lost_time')
        assert hasattr(optimizer, 'saturation_flow_rate')
    
    def test_websters_optimize(self, sample_traffic_data):
        """Test Webster's formula optimization"""
        optimizer = WebstersFormulaOptimizer()
        
        result = optimizer.optimize(sample_traffic_data)
        
        assert 'optimized_phases' in result
        assert 'efficiency_improvement' in result
        assert 'waiting_time_reduction' in result
        assert isinstance(result['optimized_phases'], list)
        assert len(result['optimized_phases']) == 4
        assert all(phase > 0 for phase in result['optimized_phases'])
    
    def test_websters_cycle_time_calculation(self, sample_traffic_data):
        """Test cycle time calculation in Webster's formula"""
        optimizer = WebstersFormulaOptimizer()
        
        cycle_time = optimizer._calculate_cycle_time(sample_traffic_data)
        assert isinstance(cycle_time, float)
        assert cycle_time > 0
    
    def test_websters_phase_time_calculation(self, sample_traffic_data):
        """Test phase time calculation in Webster's formula"""
        optimizer = WebstersFormulaOptimizer()
        
        phase_times = optimizer._calculate_phase_times(sample_traffic_data)
        assert isinstance(phase_times, list)
        assert len(phase_times) == 4
        assert all(time > 0 for time in phase_times)
    
    def test_websters_saturation_flow_calculation(self, sample_traffic_data):
        """Test saturation flow calculation in Webster's formula"""
        optimizer = WebstersFormulaOptimizer()
        
        saturation_flow = optimizer._calculate_saturation_flow(sample_traffic_data)
        assert isinstance(saturation_flow, float)
        assert saturation_flow > 0

class TestTrafficPredictor:
    """Test TrafficPredictor class"""
    
    def test_traffic_predictor_initialization(self):
        """Test TrafficPredictor initialization"""
        predictor = TrafficPredictor()
        assert predictor is not None
        assert hasattr(predictor, 'model')
        assert hasattr(predictor, 'scaler')
    
    def test_predict_traffic_flow(self, sample_traffic_data):
        """Test traffic flow prediction"""
        predictor = TrafficPredictor()
        
        with patch.object(predictor, 'model') as mock_model:
            mock_model.predict.return_value = np.array([25.5, 30.2, 28.1, 32.0])
            
            prediction = predictor.predict_traffic_flow(sample_traffic_data)
            
            assert isinstance(prediction, dict)
            assert 'predicted_flow' in prediction
            assert 'confidence' in prediction
            assert isinstance(prediction['predicted_flow'], list)
            assert len(prediction['predicted_flow']) == 4
    
    def test_predict_waiting_time(self, sample_traffic_data):
        """Test waiting time prediction"""
        predictor = TrafficPredictor()
        
        with patch.object(predictor, 'model') as mock_model:
            mock_model.predict.return_value = np.array([15.5, 20.2, 18.1, 22.0])
            
            prediction = predictor.predict_waiting_time(sample_traffic_data)
            
            assert isinstance(prediction, dict)
            assert 'predicted_waiting_time' in prediction
            assert 'confidence' in prediction
            assert isinstance(prediction['predicted_waiting_time'], list)
            assert len(prediction['predicted_waiting_time']) == 4
    
    def test_predict_optimization_impact(self, sample_traffic_data):
        """Test optimization impact prediction"""
        predictor = TrafficPredictor()
        
        with patch.object(predictor, 'model') as mock_model:
            mock_model.predict.return_value = np.array([0.15, 0.12, 0.18, 0.10])
            
            prediction = predictor.predict_optimization_impact(sample_traffic_data, [30, 25, 35, 20])
            
            assert isinstance(prediction, dict)
            assert 'efficiency_improvement' in prediction
            assert 'waiting_time_reduction' in prediction
            assert 'throughput_increase' in prediction
    
    def test_data_preprocessing(self, sample_traffic_data):
        """Test data preprocessing for prediction"""
        predictor = TrafficPredictor()
        
        processed_data = predictor._preprocess_data(sample_traffic_data)
        assert isinstance(processed_data, np.ndarray)
        assert processed_data.shape[1] > 0  # Should have features
    
    def test_model_training(self, sample_traffic_data):
        """Test model training"""
        predictor = TrafficPredictor()
        
        # Create training data
        training_data = [sample_traffic_data] * 100
        training_labels = [25.5] * 100
        
        with patch.object(predictor, 'model') as mock_model:
            predictor.train(training_data, training_labels)
            mock_model.fit.assert_called_once()

class TestPerformanceMetrics:
    """Test PerformanceMetrics class"""
    
    def test_performance_metrics_initialization(self):
        """Test PerformanceMetrics initialization"""
        metrics = PerformanceMetrics()
        assert metrics is not None
        assert hasattr(metrics, 'metrics_history')
    
    def test_calculate_efficiency(self, sample_traffic_data):
        """Test efficiency calculation"""
        metrics = PerformanceMetrics()
        
        efficiency = metrics.calculate_efficiency(sample_traffic_data)
        assert isinstance(efficiency, float)
        assert 0 <= efficiency <= 1
    
    def test_calculate_throughput(self, sample_traffic_data):
        """Test throughput calculation"""
        metrics = PerformanceMetrics()
        
        throughput = metrics.calculate_throughput(sample_traffic_data)
        assert isinstance(throughput, float)
        assert throughput >= 0
    
    def test_calculate_waiting_time(self, sample_traffic_data):
        """Test waiting time calculation"""
        metrics = PerformanceMetrics()
        
        waiting_time = metrics.calculate_waiting_time(sample_traffic_data)
        assert isinstance(waiting_time, float)
        assert waiting_time >= 0
    
    def test_calculate_co2_emission(self, sample_traffic_data):
        """Test CO2 emission calculation"""
        metrics = PerformanceMetrics()
        
        co2_emission = metrics.calculate_co2_emission(sample_traffic_data)
        assert isinstance(co2_emission, float)
        assert co2_emission >= 0
    
    def test_update_metrics_history(self, sample_traffic_data):
        """Test metrics history update"""
        metrics = PerformanceMetrics()
        
        initial_count = len(metrics.metrics_history)
        metrics.update_metrics_history(sample_traffic_data)
        
        assert len(metrics.metrics_history) == initial_count + 1
        assert metrics.metrics_history[-1]['timestamp'] is not None
    
    def test_get_metrics_summary(self, sample_traffic_data):
        """Test metrics summary generation"""
        metrics = PerformanceMetrics()
        
        # Add some data to history
        for _ in range(10):
            metrics.update_metrics_history(sample_traffic_data)
        
        summary = metrics.get_metrics_summary()
        assert isinstance(summary, dict)
        assert 'average_efficiency' in summary
        assert 'average_throughput' in summary
        assert 'average_waiting_time' in summary
        assert 'total_vehicles' in summary

class TestMLModelIntegration:
    """Test ML model integration and training"""
    
    def test_model_training_pipeline(self, sample_traffic_data):
        """Test complete model training pipeline"""
        optimizer = SignalOptimizer()
        
        # Mock training data
        training_data = [sample_traffic_data] * 1000
        training_labels = [0.15] * 1000
        
        with patch.object(optimizer.q_learning_optimizer, 'train') as mock_train:
            optimizer.train_models(training_data, training_labels)
            mock_train.assert_called_once()
    
    def test_model_evaluation(self, sample_traffic_data):
        """Test model evaluation"""
        optimizer = SignalOptimizer()
        
        # Mock test data
        test_data = [sample_traffic_data] * 100
        test_labels = [0.12] * 100
        
        with patch.object(optimizer.q_learning_optimizer, 'evaluate') as mock_evaluate:
            mock_evaluate.return_value = {'accuracy': 0.85, 'mse': 0.05}
            
            evaluation = optimizer.evaluate_models(test_data, test_labels)
            assert 'accuracy' in evaluation
            assert 'mse' in evaluation
            mock_evaluate.assert_called_once()
    
    def test_model_persistence(self, sample_traffic_data):
        """Test model saving and loading"""
        optimizer = SignalOptimizer()
        
        with patch.object(optimizer.q_learning_optimizer, 'save_model') as mock_save:
            optimizer.save_models('models/')
            mock_save.assert_called_once_with('models/')
        
        with patch.object(optimizer.q_learning_optimizer, 'load_model') as mock_load:
            optimizer.load_models('models/')
            mock_load.assert_called_once_with('models/')

@pytest.fixture
def sample_traffic_data():
    """Sample traffic data for testing"""
    return {
        'intersection_id': 'test_intersection_1',
        'traffic_lights': [
            {
                'id': 'light_1',
                'phase': 0,
                'duration': 30,
                'vehicle_count': 15,
                'waiting_time': 45.5
            },
            {
                'id': 'light_2',
                'phase': 2,
                'duration': 25,
                'vehicle_count': 8,
                'waiting_time': 20.0
            }
        ],
        'vehicles': [
            {
                'id': 'vehicle_1',
                'type': 'passenger',
                'speed': 25.5,
                'lane': 'north_approach_0',
                'waiting_time': 5.2
            },
            {
                'id': 'vehicle_2',
                'type': 'truck',
                'speed': 18.0,
                'lane': 'east_approach_0',
                'waiting_time': 12.8
            }
        ],
        'timestamp': datetime.now().isoformat()
    }
