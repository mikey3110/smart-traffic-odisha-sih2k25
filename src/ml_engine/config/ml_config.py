"""
ML Configuration Management for Smart Traffic Signal Optimization
Handles hyperparameters, algorithm settings, and system configuration
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import yaml


class OptimizationAlgorithm(Enum):
    """Available optimization algorithms"""
    Q_LEARNING = "q_learning"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    WEBSTERS_FORMULA = "websters_formula"
    HYBRID = "hybrid"
    RANDOM = "random"


class TrafficPredictionModel(Enum):
    """Available traffic prediction models"""
    LSTM = "lstm"
    ARIMA = "arima"
    PROPHET = "prophet"
    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"


@dataclass
class QLearningConfig:
    """Q-Learning algorithm configuration"""
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    epsilon: float = 0.1
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    memory_size: int = 10000
    batch_size: int = 32
    target_update_frequency: int = 100
    state_size: int = 8  # [lane_counts, avg_speed, time_of_day, day_of_week, weather, etc.]
    action_size: int = 5  # [reduce_green, maintain, increase_green_small, increase_green_medium, increase_green_large]
    hidden_layers: List[int] = field(default_factory=lambda: [64, 32])
    activation: str = "relu"
    optimizer: str = "adam"
    loss_function: str = "mse"


@dataclass
class DynamicProgrammingConfig:
    """Dynamic Programming algorithm configuration"""
    time_horizon: int = 60  # seconds
    time_step: int = 5  # seconds
    max_cycle_time: int = 120  # seconds
    min_cycle_time: int = 60  # seconds
    min_green_time: int = 15  # seconds
    max_green_time: int = 60  # seconds
    yellow_time: int = 3  # seconds
    all_red_time: int = 2  # seconds
    cost_weights: Dict[str, float] = field(default_factory=lambda: {
        "wait_time": 1.0,
        "queue_length": 0.5,
        "fuel_consumption": 0.3,
        "emissions": 0.2
    })


@dataclass
class WebstersFormulaConfig:
    """Webster's Formula configuration"""
    base_cycle_time: int = 60  # seconds
    min_cycle_time: int = 40  # seconds
    max_cycle_time: int = 120  # seconds
    lost_time: int = 4  # seconds per phase
    saturation_flow_rate: float = 1800  # vehicles per hour per lane
    critical_flow_ratio: float = 0.9
    ml_enhancement: bool = True
    prediction_horizon: int = 15  # minutes


@dataclass
class TrafficPredictionConfig:
    """Traffic prediction model configuration"""
    model_type: TrafficPredictionModel = TrafficPredictionModel.LSTM
    sequence_length: int = 60  # minutes of historical data
    prediction_horizon: int = 15  # minutes ahead
    features: List[str] = field(default_factory=lambda: [
        "lane_counts", "avg_speed", "weather_condition", "time_of_day", 
        "day_of_week", "temperature", "humidity", "visibility"
    ])
    lstm_units: int = 50
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2


@dataclass
class PerformanceMetricsConfig:
    """Performance metrics calculation configuration"""
    wait_time_weight: float = 1.0
    throughput_weight: float = 0.8
    fuel_consumption_weight: float = 0.3
    emission_weight: float = 0.2
    safety_weight: float = 0.5
    comfort_weight: float = 0.3
    calculation_window: int = 300  # seconds
    update_frequency: int = 30  # seconds


@dataclass
class ABTestingConfig:
    """A/B Testing framework configuration"""
    enabled: bool = True
    test_duration: int = 3600  # seconds
    traffic_split: float = 0.5  # 50-50 split
    confidence_level: float = 0.95
    minimum_sample_size: int = 100
    statistical_power: float = 0.8
    metrics_to_track: List[str] = field(default_factory=lambda: [
        "wait_time", "throughput", "fuel_consumption", "emissions"
    ])


@dataclass
class DataIntegrationConfig:
    """Real-time data integration configuration"""
    api_base_url: str = "http://localhost:8000"
    api_timeout: int = 10  # seconds
    retry_attempts: int = 3
    retry_delay: float = 1.0  # seconds
    retry_backoff: float = 2.0
    data_fetch_interval: int = 10  # seconds
    data_validation: bool = True
    fallback_to_mock: bool = True
    cache_duration: int = 30  # seconds


@dataclass
class LoggingConfig:
    """Logging and monitoring configuration"""
    log_level: str = "INFO"
    log_file: str = "logs/ml_optimization.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    enable_metrics_logging: bool = True
    metrics_log_interval: int = 60  # seconds
    enable_performance_profiling: bool = False


@dataclass
class MLConfig:
    """Main ML configuration container"""
    # Algorithm configurations
    q_learning: QLearningConfig = field(default_factory=QLearningConfig)
    dynamic_programming: DynamicProgrammingConfig = field(default_factory=DynamicProgrammingConfig)
    websters_formula: WebstersFormulaConfig = field(default_factory=WebstersFormulaConfig)
    traffic_prediction: TrafficPredictionConfig = field(default_factory=TrafficPredictionConfig)
    
    # System configurations
    performance_metrics: PerformanceMetricsConfig = field(default_factory=PerformanceMetricsConfig)
    ab_testing: ABTestingConfig = field(default_factory=ABTestingConfig)
    data_integration: DataIntegrationConfig = field(default_factory=DataIntegrationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # General settings
    primary_algorithm: OptimizationAlgorithm = OptimizationAlgorithm.HYBRID
    fallback_algorithm: OptimizationAlgorithm = OptimizationAlgorithm.WEBSTERS_FORMULA
    optimization_interval: int = 10  # seconds
    model_save_path: str = "models/"
    data_save_path: str = "data/"
    enable_gpu: bool = False
    random_seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "q_learning": self.q_learning.__dict__,
            "dynamic_programming": self.dynamic_programming.__dict__,
            "websters_formula": self.websters_formula.__dict__,
            "traffic_prediction": self.traffic_prediction.__dict__,
            "performance_metrics": self.performance_metrics.__dict__,
            "ab_testing": self.ab_testing.__dict__,
            "data_integration": self.data_integration.__dict__,
            "logging": self.logging.__dict__,
            "primary_algorithm": self.primary_algorithm.value,
            "fallback_algorithm": self.fallback_algorithm.value,
            "optimization_interval": self.optimization_interval,
            "model_save_path": self.model_save_path,
            "data_save_path": self.data_save_path,
            "enable_gpu": self.enable_gpu,
            "random_seed": self.random_seed
        }
    
    def save_to_file(self, filepath: str, format: str = "json") -> None:
        """Save configuration to file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        elif format.lower() == "yaml":
            with open(filepath, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'MLConfig':
        """Load configuration from file"""
        if not os.path.exists(filepath):
            return cls()
        
        with open(filepath, 'r') as f:
            if filepath.endswith('.json'):
                data = json.load(f)
            elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {filepath}")
        
        # Create config instance
        config = cls()
        
        # Update with loaded data
        for key, value in data.items():
            if hasattr(config, key):
                if key in ['primary_algorithm', 'fallback_algorithm']:
                    setattr(config, key, OptimizationAlgorithm(value))
                elif key == 'traffic_prediction' and 'model_type' in value:
                    value['model_type'] = TrafficPredictionModel(value['model_type'])
                setattr(config, key, value)
        
        return config
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate Q-Learning config
        if self.q_learning.learning_rate <= 0 or self.q_learning.learning_rate > 1:
            issues.append("Q-Learning learning rate must be between 0 and 1")
        
        if self.q_learning.discount_factor <= 0 or self.q_learning.discount_factor >= 1:
            issues.append("Q-Learning discount factor must be between 0 and 1")
        
        # Validate Dynamic Programming config
        if self.dynamic_programming.min_cycle_time >= self.dynamic_programming.max_cycle_time:
            issues.append("Min cycle time must be less than max cycle time")
        
        # Validate Webster's Formula config
        if self.websters_formula.min_cycle_time >= self.websters_formula.max_cycle_time:
            issues.append("Webster's min cycle time must be less than max cycle time")
        
        # Validate data integration config
        if self.data_integration.retry_attempts < 0:
            issues.append("Retry attempts must be non-negative")
        
        if self.data_integration.api_timeout <= 0:
            issues.append("API timeout must be positive")
        
        return issues


# Global configuration instance
ml_config = MLConfig()


def get_config() -> MLConfig:
    """Get the global ML configuration"""
    return ml_config


def load_config(filepath: str) -> MLConfig:
    """Load configuration from file and update global config"""
    global ml_config
    ml_config = MLConfig.load_from_file(filepath)
    return ml_config


def save_config(filepath: str, format: str = "json") -> None:
    """Save current configuration to file"""
    ml_config.save_to_file(filepath, format)




