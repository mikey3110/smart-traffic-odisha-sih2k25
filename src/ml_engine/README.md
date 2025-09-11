# ML-based Traffic Signal Optimization System

A comprehensive machine learning system for intelligent traffic signal optimization with real-time data integration, multiple optimization algorithms, and advanced monitoring capabilities.

## 🚀 Features

### Core ML Components
- **Real-time Data Integration**: Robust API communication with fallback strategies
- **Traffic Prediction**: LSTM, ARIMA, Prophet, and ensemble models
- **Advanced Optimization Algorithms**:
  - Q-Learning with Deep Q-Networks (DQN)
  - Dynamic Programming for optimal control
  - Enhanced Webster's Formula with ML predictions
- **Performance Metrics**: Comprehensive traffic flow analysis
- **A/B Testing Framework**: Statistical comparison of optimization strategies
- **Monitoring & Alerting**: Real-time system health and performance tracking
- **Visualization**: Interactive dashboards and performance reports

### Key Capabilities
- **Adaptive Algorithm Selection**: Automatically chooses the best algorithm based on traffic conditions
- **Multi-intersection Support**: Simultaneous optimization of multiple intersections
- **Fallback Strategies**: Graceful degradation when components fail
- **Comprehensive Logging**: Structured logging with JSON format
- **Performance Visualization**: Real-time dashboards and historical analysis
- **Statistical Analysis**: A/B testing with significance testing
- **Model Training**: Automated training pipeline with evaluation metrics

## 📁 Project Structure

```
src/ml_engine/
├── algorithms/                    # Optimization algorithms
│   ├── enhanced_q_learning_optimizer.py
│   ├── enhanced_dynamic_programming_optimizer.py
│   └── enhanced_websters_formula_optimizer.py
├── data/                         # Data integration
│   └── enhanced_data_integration.py
├── prediction/                   # Traffic prediction models
│   └── enhanced_traffic_predictor.py
├── metrics/                      # Performance metrics
│   └── enhanced_performance_metrics.py
├── ab_testing/                   # A/B testing framework
│   └── ab_testing_framework.py
├── monitoring/                   # Monitoring and alerting
│   └── enhanced_monitoring.py
├── visualization/                # Performance visualization
│   └── performance_visualizer.py
├── training/                     # Model training scripts
│   └── model_training_script.py
├── tests/                        # Comprehensive test suite
│   └── test_ml_components.py
├── config/                       # Configuration management
│   └── ml_config.py
├── enhanced_signal_optimizer.py  # Main optimizer
└── run_ml_optimization.py       # Execution script
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Dependencies
```bash
# Core ML libraries
pip install numpy pandas scikit-learn
pip install tensorflow torch  # For deep learning
pip install statsmodels prophet  # For time series
pip install aiohttp requests  # For API communication
pip install matplotlib seaborn  # For visualization
pip install psutil  # For system monitoring
pip install pytest  # For testing

# Optional: GPU support
pip install tensorflow-gpu  # For GPU acceleration
```

### Installation Steps
1. Clone the repository
2. Navigate to the ML engine directory
3. Install dependencies
4. Configure the system

```bash
cd src/ml_engine
pip install -r requirements.txt
python -m pytest tests/  # Run tests
```

## ⚙️ Configuration

### Environment Variables
```bash
# API Configuration
ML_API_BASE_URL=http://localhost:8000
ML_API_TIMEOUT=10
ML_RETRY_ATTEMPTS=3

# Database Configuration
ML_DB_URL=postgresql://user:pass@localhost/traffic_db
ML_REDIS_URL=redis://localhost:6379

# Model Configuration
ML_MODEL_SAVE_PATH=models/
ML_DATA_SAVE_PATH=data/
ML_ENABLE_GPU=false

# Logging Configuration
ML_LOG_LEVEL=INFO
ML_LOG_FILE=logs/ml_optimization.log
```

### Configuration File
Create `config/ml_config.yaml`:
```yaml
primary_algorithm: "hybrid"
optimization_interval: 10
model_save_path: "models/"
data_save_path: "data/"

q_learning:
  learning_rate: 0.1
  epsilon: 0.1
  memory_size: 10000
  batch_size: 32

traffic_prediction:
  model_type: "lstm"
  sequence_length: 60
  prediction_horizon: 15

ab_testing:
  enabled: true
  test_duration: 3600
  traffic_split: 0.5
  confidence_level: 0.95
```

## 🚀 Quick Start

### Basic Usage
```python
from enhanced_signal_optimizer import EnhancedSignalOptimizer, OptimizationRequest, OptimizationMode

# Initialize optimizer
optimizer = EnhancedSignalOptimizer()

# Start the system
await optimizer.start()

# Create optimization request
request = OptimizationRequest(
    intersection_id="junction-1",
    current_timings={'north_lane': 30, 'south_lane': 30, 'east_lane': 30, 'west_lane': 30},
    optimization_mode=OptimizationMode.ADAPTIVE
)

# Optimize intersection
response = await optimizer.optimize_intersection(request)
print(f"Optimized timings: {response.optimized_timings}")
print(f"Algorithm used: {response.algorithm_used}")
print(f"Confidence: {response.confidence}")
```

### Command Line Interface
```bash
# Start optimization system
python run_ml_optimization.py --intersections junction-1 junction-2 --interval 10

# Create A/B test
python run_ml_optimization.py --ab-test --control-algorithm websters_formula --treatment-algorithm q_learning

# Generate performance report
python run_ml_optimization.py --report

# Export system data
python run_ml_optimization.py --export

# Show system status
python run_ml_optimization.py --status
```

### Training Models
```bash
# Train all models
python training/model_training_script.py --samples 10000 --days 30

# Train with custom data
python training/model_training_script.py --data-file custom_data.csv

# Train with specific configuration
python training/model_training_script.py --config config/custom_config.yaml
```

## 🔧 Advanced Usage

### Custom Algorithm Implementation
```python
from algorithms.base_optimizer import BaseOptimizer

class CustomOptimizer(BaseOptimizer):
    def optimize_signal_timing(self, traffic_data, current_timings):
        # Implement your custom optimization logic
        optimized_timings = {}
        # ... your optimization code ...
        return optimized_timings

# Register custom algorithm
optimizer.algorithms['custom'] = CustomOptimizer()
```

### A/B Testing
```python
from ab_testing.ab_testing_framework import ABTestConfig, TestVariant, StatisticalTest

# Create A/B test
variants = [
    TestVariant(name="control", algorithm="websters_formula", traffic_split=0.5, is_control=True),
    TestVariant(name="treatment", algorithm="q_learning", traffic_split=0.5)
]

test_config = ABTestConfig(
    test_id="custom_test",
    name="Custom Algorithm Test",
    variants=variants,
    target_metrics=["wait_time", "throughput", "efficiency"],
    statistical_test=StatisticalTest(test_type="t_test", alpha=0.05)
)

# Create and start test
test_id = optimizer.create_ab_test(test_config)
optimizer.start_ab_test(test_id, ["junction-1", "junction-2"])
```

### Custom Monitoring
```python
from monitoring.enhanced_monitoring import EnhancedMonitoring, Alert, AlertType

# Create custom alert handler
def custom_alert_handler(alert):
    if alert.alert_type == AlertType.PERFORMANCE_DEGRADATION:
        # Send notification, log to external system, etc.
        print(f"Performance alert: {alert.message}")

# Add alert handler
monitoring = EnhancedMonitoring()
monitoring.alert_manager.add_alert_handler(custom_alert_handler)
```

## 📊 Performance Metrics

### Traffic Metrics
- **Wait Time**: Average time vehicles wait at intersection
- **Throughput**: Vehicles processed per hour
- **Efficiency**: Overall intersection efficiency score
- **Safety Score**: Safety assessment based on speed and conditions
- **Comfort Score**: Driver comfort based on acceleration patterns
- **Fuel Consumption**: Estimated fuel usage
- **Emissions**: CO2 emissions estimation

### System Metrics
- **CPU Usage**: System CPU utilization
- **Memory Usage**: RAM consumption
- **Response Time**: API response times
- **Error Rate**: System error frequency
- **Confidence**: Algorithm confidence scores

### Optimization Metrics
- **Improvement Percentage**: Performance improvement over baseline
- **Algorithm Performance**: Comparative algorithm effectiveness
- **Processing Time**: Optimization computation time
- **Success Rate**: Successful optimization percentage

## 🧪 Testing

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests
python -m pytest tests/test_ml_components.py::TestDataIntegration -v

# Integration tests
python -m pytest tests/test_ml_components.py::TestIntegration -v

# Performance tests
python -m pytest tests/test_ml_components.py::TestPerformance -v
```

### Test Coverage
```bash
python -m pytest tests/ --cov=src/ml_engine --cov-report=html
```

## 📈 Monitoring and Visualization

### Real-time Dashboard
The system provides a comprehensive real-time dashboard showing:
- Performance trends over time
- Algorithm comparison charts
- System health metrics
- Active alerts and warnings
- Optimization statistics

### Performance Reports
Generate detailed performance reports with:
- Historical trend analysis
- Algorithm effectiveness comparison
- A/B test results
- System health monitoring
- Exportable visualizations

### Alert System
Configurable alerts for:
- Performance degradation
- High error rates
- Resource usage thresholds
- Algorithm failures
- Data quality issues

## 🔧 Troubleshooting

### Common Issues

#### 1. API Connection Errors
```bash
# Check API availability
curl http://localhost:8000/api/v1/health

# Verify configuration
python -c "from config.ml_config import get_config; print(get_config().data_integration.api_base_url)"
```

#### 2. Model Training Failures
```bash
# Check dependencies
python -c "import tensorflow; print('TensorFlow available')"
python -c "import statsmodels; print('Statsmodels available')"

# Verify data format
python training/model_training_script.py --samples 1000 --days 7
```

#### 3. Performance Issues
```bash
# Monitor system resources
python -c "from monitoring.enhanced_monitoring import EnhancedMonitoring; m = EnhancedMonitoring(); print(m.get_system_health_summary())"

# Check optimization statistics
python run_ml_optimization.py --status
```

### Debug Mode
```bash
# Enable debug logging
export ML_LOG_LEVEL=DEBUG
python run_ml_optimization.py --intersections junction-1
```

### Log Analysis
```bash
# View recent logs
tail -f logs/ml_optimization.log

# Search for errors
grep "ERROR" logs/ml_optimization.log

# Analyze performance
grep "optimization" logs/ml_optimization.log | tail -100
```

## 📚 API Reference

### EnhancedSignalOptimizer
Main optimization class providing:
- `optimize_intersection(request)`: Optimize single intersection
- `create_ab_test(config)`: Create A/B test
- `get_optimization_statistics()`: Get performance statistics
- `export_optimization_data(path)`: Export data

### DataIntegration
Real-time data fetching with:
- `fetch_traffic_data(intersection_id)`: Get traffic data
- `fetch_multiple_intersections(ids)`: Batch data fetching
- `health_check()`: Check system health

### TrafficPredictor
ML-based traffic prediction:
- `predict_traffic_flow(intersection_id, horizon)`: Predict traffic
- `train_models(data)`: Train prediction models
- `evaluate_models(data)`: Evaluate model performance

### PerformanceMetrics
Comprehensive metrics calculation:
- `calculate_metrics(traffic_data, timings, intersection_id)`: Calculate metrics
- `get_performance_summary(intersection_id)`: Get summary
- `export_metrics(filepath)`: Export metrics data

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Install development dependencies
4. Run tests
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints
- Write comprehensive docstrings
- Include unit tests for new features

### Testing Guidelines
- Write tests for all new functionality
- Maintain test coverage above 80%
- Include integration tests for complex features
- Test error conditions and edge cases

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Traffic engineering research community
- Open source ML libraries (TensorFlow, scikit-learn, etc.)
- Smart city initiatives worldwide
- Contributors and maintainers

## 📞 Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation wiki
- Join the community discussions

---

**Note**: This system is designed for research and development purposes. For production deployment, ensure proper testing, monitoring, and security measures are in place.
