# A/B Testing Framework for Traffic Signal Optimization

## Overview

This A/B testing framework provides comprehensive evaluation of machine learning-based traffic signal optimization against traditional baseline methods. The framework includes automated test execution, statistical analysis, visualization, and reporting capabilities.

## Features

### 🧪 Automated Testing
- **Controlled Experiments**: Side-by-side comparison of ML vs baseline methods
- **Multiple Scenarios**: Normal traffic, rush hour, emergency, and varying volumes
- **Repetitive Testing**: Multiple runs per scenario for statistical reliability
- **Real-time Monitoring**: Continuous data collection during test execution

### 📊 Statistical Analysis
- **T-tests**: Independent samples t-tests for significance testing
- **Effect Sizes**: Cohen's d calculation for practical significance
- **Confidence Intervals**: 95% confidence intervals for performance estimates
- **Correlation Analysis**: Performance metric relationships and dependencies

### 📈 Visualization
- **Performance Charts**: Box plots, bar charts, and trend analysis
- **Improvement Metrics**: Percentage improvement visualizations
- **Correlation Matrices**: Heat maps showing metric relationships
- **Scenario Analysis**: Performance comparison across different traffic conditions

### 📋 Reporting
- **Automated Reports**: JSON, CSV, and Markdown output formats
- **Statistical Summaries**: Comprehensive performance analysis
- **Business Impact**: ROI calculations and economic benefits
- **Recommendations**: Actionable insights for deployment

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/smart-traffic-odisha-sih2k25.git
cd smart-traffic-odisha-sih2k25

# Install dependencies
pip install -r requirements_ab_tests.txt

# Setup directories
mkdir -p results/{ab_tests,analysis,visualizations,reports}
mkdir -p logs config notebooks
```

### 2. Configuration

Create a configuration file `config/ab_test_config.json`:

```json
{
    "sumo_configs": {
        "normal": "sumo/configs/normal_traffic.sumocfg",
        "rush_hour": "sumo/configs/rush_hour.sumocfg",
        "emergency": "sumo/configs/emergency_vehicle.sumocfg"
    },
    "test_scenarios": [
        {
            "name": "normal_traffic_ml_vs_baseline",
            "type": "ml_vs_baseline",
            "scenario": "normal",
            "duration": 1800,
            "repetitions": 5,
            "traffic_volume": "medium",
            "intersections": ["center"]
        }
    ],
    "output_dir": "results/ab_tests",
    "log_level": "INFO"
}
```

### 3. Run Complete Pipeline

```bash
# Run the complete A/B testing pipeline
python scripts/run_complete_ab_tests.py
```

### 4. Run Individual Components

```bash
# Run A/B tests only
python scripts/run_ab_tests.py

# Analyze results only
python scripts/analyze_ab_results.py

# Interactive analysis
jupyter notebook notebooks/ab_test_analysis.ipynb
```

## Architecture

### Test Execution Engine

The `ABTestRunner` class manages test execution:

```python
class ABTestRunner:
    def __init__(self, config_file):
        self.config = self._load_config()
        self.tests = []
        self.results = []
    
    def run_all_tests(self):
        for test_config in self.tests:
            result = self.run_single_test(test_config)
            self.results.append(result)
```

### Data Collection System

Real-time metrics collection during test execution:

- **Traffic Metrics**: Wait times, vehicle counts, queue lengths
- **Performance Indicators**: Throughput, fuel consumption, emissions
- **System Metrics**: CPU usage, memory usage, error rates
- **Statistical Data**: Confidence intervals, p-values, effect sizes

### Analysis Pipeline

The `ABTestAnalyzer` class provides comprehensive analysis:

```python
class ABTestAnalyzer:
    def analyze_performance(self):
        # Statistical analysis of performance metrics
        pass
    
    def generate_visualizations(self):
        # Create performance charts and graphs
        pass
    
    def generate_summary_report(self):
        # Generate comprehensive reports
        pass
```

## Test Scenarios

### 1. Normal Traffic Conditions
- **Duration**: 30 minutes
- **Repetitions**: 5 runs
- **Traffic Volume**: Medium
- **Comparison**: ML vs Fixed Timing

### 2. Rush Hour Traffic
- **Duration**: 60 minutes
- **Repetitions**: 3 runs
- **Traffic Volume**: High
- **Comparison**: ML vs Webster's Formula

### 3. Emergency Vehicle Priority
- **Duration**: 10 minutes
- **Repetitions**: 3 runs
- **Traffic Volume**: Low
- **Emergency Vehicles**: 2 per run

### 4. Low Traffic Conditions
- **Duration**: 20 minutes
- **Repetitions**: 5 runs
- **Traffic Volume**: Low
- **Comparison**: ML vs Fixed Timing

### 5. High Traffic Conditions
- **Duration**: 40 minutes
- **Repetitions**: 3 runs
- **Traffic Volume**: High
- **Comparison**: ML vs Webster's Formula

## Performance Metrics

### Primary Metrics
- **Average Wait Time**: Mean waiting time per vehicle
- **Vehicles per Hour**: Throughput capacity
- **Max Queue Length**: Maximum queue length observed
- **Fuel Consumption**: Total fuel used during test
- **CO2 Emissions**: Total CO2 emissions generated

### Secondary Metrics
- **Phase Changes**: Number of signal phase changes
- **Cycle Efficiency**: Green time utilization
- **Queue Clearance Time**: Time to clear queues
- **System Uptime**: System availability percentage
- **Error Rate**: System error frequency

## Statistical Analysis

### Significance Testing
- **T-tests**: Independent samples t-tests
- **P-values**: Statistical significance levels
- **Effect Sizes**: Cohen's d for practical significance
- **Confidence Intervals**: 95% confidence intervals

### Effect Size Interpretation
- **Small Effect**: d = 0.2
- **Medium Effect**: d = 0.5
- **Large Effect**: d = 0.8
- **Very Large Effect**: d > 1.0

### Sample Size Requirements
- **Minimum**: 3 runs per scenario
- **Recommended**: 5 runs per scenario
- **Optimal**: 10+ runs for high confidence

## Output Files

### Test Results
- `ab_test_results_YYYYMMDD_HHMMSS.json`: Complete test results
- `performance_metrics_YYYYMMDD_HHMMSS.csv`: Performance metrics data
- `ab_test_summary_YYYYMMDD_HHMMSS.md`: Test summary report

### Analysis Results
- `analysis_results.json`: Statistical analysis results
- `scenario_analysis.csv`: Scenario-specific analysis
- `summary_statistics.json`: Summary statistics

### Visualizations
- `performance_comparison.png`: Performance comparison charts
- `improvement_chart.png`: Improvement percentage chart
- `correlation_matrix.png`: Correlation matrix heatmap
- `scenario_analysis.png`: Scenario-specific analysis

### Reports
- `ab_test_summary_report.md`: Detailed analysis report
- `final_ab_test_report.md`: Complete pipeline report
- `simulation_report.md`: Comprehensive simulation report

## Configuration Options

### Test Configuration
```json
{
    "test_scenarios": [
        {
            "name": "test_name",
            "type": "ml_vs_baseline",
            "scenario": "normal",
            "duration": 1800,
            "repetitions": 5,
            "traffic_volume": "medium",
            "intersections": ["center"]
        }
    ]
}
```

### Analysis Configuration
```json
{
    "statistical_analysis": {
        "significance_level": 0.05,
        "confidence_interval": 0.95,
        "min_sample_size": 3
    },
    "performance_thresholds": {
        "min_wait_time_improvement": 5.0,
        "min_throughput_improvement": 3.0,
        "min_queue_reduction": 5.0
    }
}
```

## Usage Examples

### Basic A/B Test
```python
from scripts.run_ab_tests import ABTestRunner

# Create test runner
runner = ABTestRunner('config/ab_test_config.json')

# Add test scenarios
runner.create_test_scenarios()

# Run all tests
runner.run_all_tests()

# Save results
runner.save_results()
```

### Custom Analysis
```python
from scripts.analyze_ab_results import ABTestAnalyzer

# Create analyzer
analyzer = ABTestAnalyzer('results/ab_tests')

# Load data
analyzer.load_data()

# Run analysis
analyzer.analyze_performance()
analyzer.generate_visualizations()
analyzer.generate_summary_report()
```

### Interactive Analysis
```python
# Launch Jupyter notebook
import subprocess
subprocess.run(['jupyter', 'notebook', 'notebooks/ab_test_analysis.ipynb'])
```

## Troubleshooting

### Common Issues

#### 1. SUMO Not Found
```
Error: SUMO executable not found
Solution: Set SUMO_HOME environment variable
export SUMO_HOME=/path/to/sumo
```

#### 2. Missing Dependencies
```
Error: ModuleNotFoundError: No module named 'pandas'
Solution: Install required packages
pip install -r requirements_ab_tests.txt
```

#### 3. Permission Denied
```
Error: Permission denied when creating directories
Solution: Check write permissions
chmod 755 results/
```

#### 4. Test Execution Failed
```
Error: A/B test execution failed
Solution: Check SUMO configuration and network files
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### Test Execution
- **Parallel Testing**: Enable parallel test execution for faster results
- **Resource Management**: Monitor CPU and memory usage
- **Data Compression**: Compress large result files

### Analysis Pipeline
- **Caching**: Cache analysis results for repeated runs
- **Batch Processing**: Process multiple scenarios simultaneously
- **Memory Management**: Optimize memory usage for large datasets

## Best Practices

### Test Design
1. **Randomization**: Randomize test order to avoid bias
2. **Control Groups**: Use appropriate baseline methods
3. **Sample Size**: Ensure sufficient sample sizes for statistical power
4. **Duration**: Run tests long enough to capture traffic patterns

### Data Collection
1. **Real-time Monitoring**: Collect data continuously during tests
2. **Data Validation**: Check data quality and remove outliers
3. **Backup**: Store raw data for reproducibility
4. **Documentation**: Document test conditions and parameters

### Analysis
1. **Statistical Rigor**: Use appropriate statistical tests
2. **Effect Sizes**: Report both statistical and practical significance
3. **Confidence Intervals**: Provide uncertainty estimates
4. **Reproducibility**: Make analysis reproducible and transparent

## Contributing

### Adding New Test Scenarios
1. Define scenario in configuration file
2. Implement test logic in `ABTestRunner`
3. Add analysis methods in `ABTestAnalyzer`
4. Update documentation

### Adding New Metrics
1. Define metric in `PerformanceMetrics` class
2. Implement collection in `collect_metrics` method
3. Add analysis in `analyze_performance` method
4. Update visualization methods

### Adding New Visualizations
1. Create visualization method in `ABTestAnalyzer`
2. Add to `generate_visualizations` method
3. Update documentation and examples
4. Test with sample data

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue on GitHub
- Contact the development team
- Check the documentation and examples

## Changelog

### Version 1.0.0
- Initial release of A/B testing framework
- Basic test execution and analysis
- Statistical analysis and visualization
- Comprehensive reporting

---

**Smart Traffic Management System Team**  
**Smart India Hackathon 2025**
