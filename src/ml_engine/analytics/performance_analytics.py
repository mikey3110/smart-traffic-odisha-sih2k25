"""
Performance Analytics Engine for ML Traffic Optimization
Phase 3: ML Model Validation & Performance Analytics

Features:
- Real-time metrics collection: wait times, throughput, fuel consumption, emissions
- Learning curve visualization and convergence analysis
- Comparative dashboards showing before/after optimization
- Predictive analytics for traffic pattern forecasting
- Anomaly detection for unusual traffic behaviors
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
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class MetricType(Enum):
    """Types of performance metrics"""
    WAIT_TIME = "wait_time"
    THROUGHPUT = "throughput"
    FUEL_CONSUMPTION = "fuel_consumption"
    EMISSIONS = "emissions"
    QUEUE_LENGTH = "queue_length"
    TRAVEL_TIME = "travel_time"
    DELAY = "delay"
    STOP_COUNT = "stop_count"


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    metric_id: str
    timestamp: datetime
    intersection_id: str
    metric_type: MetricType
    value: float
    unit: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LearningCurve:
    """Learning curve data"""
    intersection_id: str
    algorithm: str
    timestamps: List[datetime]
    rewards: List[float]
    losses: List[float]
    epsilon_values: List[float]
    q_values: List[float]
    convergence_epoch: Optional[int] = None


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    anomaly_id: str
    timestamp: datetime
    intersection_id: str
    metric_type: MetricType
    anomaly_score: float
    is_anomaly: bool
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'


class RealTimeMetricsCollector:
    """Real-time metrics collection system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Metrics storage
        self.metrics_buffer = deque(maxlen=10000)
        self.intersection_metrics = {}
        
        # Collection settings
        self.collection_interval = self.config.get('collection_interval', 1.0)  # seconds
        self.buffer_size = self.config.get('buffer_size', 10000)
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Collection thread
        self.collection_thread = None
        self.is_collecting = False
        
        self.logger.info("Real-time metrics collector initialized")
    
    def start_collection(self):
        """Start real-time metrics collection"""
        if self.is_collecting:
            return
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        self.logger.info("Real-time metrics collection started")
    
    def stop_collection(self):
        """Stop real-time metrics collection"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        self.logger.info("Real-time metrics collection stopped")
    
    def _collection_loop(self):
        """Main metrics collection loop"""
        while self.is_collecting:
            try:
                # Collect metrics from all intersections
                self._collect_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                self.logger.error(f"Error in metrics collection loop: {e}")
                time.sleep(5)
    
    def _collect_metrics(self):
        """Collect metrics from all intersections"""
        # This would integrate with the real-time optimizer
        # For now, we'll simulate metric collection
        pass
    
    def add_metric(self, intersection_id: str, metric_type: MetricType, 
                  value: float, unit: str, metadata: Optional[Dict] = None):
        """Add a performance metric"""
        with self.lock:
            metric = PerformanceMetric(
                metric_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                intersection_id=intersection_id,
                metric_type=metric_type,
                value=value,
                unit=unit,
                metadata=metadata or {}
            )
            
            self.metrics_buffer.append(metric)
            
            # Update intersection-specific metrics
            if intersection_id not in self.intersection_metrics:
                self.intersection_metrics[intersection_id] = deque(maxlen=1000)
            
            self.intersection_metrics[intersection_id].append(metric)
    
    def get_metrics(self, intersection_id: Optional[str] = None, 
                   metric_type: Optional[MetricType] = None,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None) -> List[PerformanceMetric]:
        """Get metrics with optional filtering"""
        with self.lock:
            if intersection_id:
                metrics = list(self.intersection_metrics.get(intersection_id, []))
            else:
                metrics = list(self.metrics_buffer)
            
            # Filter by metric type
            if metric_type:
                metrics = [m for m in metrics if m.metric_type == metric_type]
            
            # Filter by time range
            if start_time:
                metrics = [m for m in metrics if m.timestamp >= start_time]
            if end_time:
                metrics = [m for m in metrics if m.timestamp <= end_time]
            
            return metrics
    
    def get_metric_summary(self, intersection_id: str, 
                          metric_type: MetricType,
                          time_window: int = 3600) -> Dict[str, float]:
        """Get metric summary for time window"""
        end_time = datetime.now()
        start_time = end_time - timedelta(seconds=time_window)
        
        metrics = self.get_metrics(intersection_id, metric_type, start_time, end_time)
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        }


class LearningCurveAnalyzer:
    """Learning curve analysis and visualization"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Learning curves storage
        self.learning_curves = {}
        
        # Analysis parameters
        self.convergence_threshold = self.config.get('convergence_threshold', 0.01)
        self.min_epochs = self.config.get('min_epochs', 100)
    
    def add_learning_curve(self, intersection_id: str, algorithm: str, 
                          learning_curve: LearningCurve):
        """Add learning curve data"""
        key = f"{intersection_id}_{algorithm}"
        self.learning_curves[key] = learning_curve
        
        # Analyze convergence
        self._analyze_convergence(learning_curve)
    
    def _analyze_convergence(self, learning_curve: LearningCurve):
        """Analyze learning curve convergence"""
        if len(learning_curve.rewards) < self.min_epochs:
            return
        
        # Calculate moving average of rewards
        window_size = min(50, len(learning_curve.rewards) // 10)
        moving_avg = pd.Series(learning_curve.rewards).rolling(window=window_size).mean()
        
        # Find convergence point
        convergence_epoch = None
        for i in range(window_size, len(moving_avg)):
            if i + window_size < len(moving_avg):
                recent_std = moving_avg.iloc[i:i+window_size].std()
                if recent_std < self.convergence_threshold:
                    convergence_epoch = i
                    break
        
        learning_curve.convergence_epoch = convergence_epoch
    
    def get_convergence_analysis(self, intersection_id: str, 
                               algorithm: str) -> Dict[str, Any]:
        """Get convergence analysis for learning curve"""
        key = f"{intersection_id}_{algorithm}"
        learning_curve = self.learning_curves.get(key)
        
        if not learning_curve:
            return {}
        
        analysis = {
            'total_epochs': len(learning_curve.rewards),
            'convergence_epoch': learning_curve.convergence_epoch,
            'converged': learning_curve.convergence_epoch is not None,
            'final_reward': learning_curve.rewards[-1] if learning_curve.rewards else 0,
            'max_reward': max(learning_curve.rewards) if learning_curve.rewards else 0,
            'min_reward': min(learning_curve.rewards) if learning_curve.rewards else 0,
            'reward_improvement': 0
        }
        
        if len(learning_curve.rewards) > 1:
            analysis['reward_improvement'] = (
                learning_curve.rewards[-1] - learning_curve.rewards[0]
            ) / abs(learning_curve.rewards[0]) * 100
        
        return analysis
    
    def create_learning_curve_plot(self, intersection_id: str, 
                                 algorithm: str) -> go.Figure:
        """Create learning curve visualization"""
        key = f"{intersection_id}_{algorithm}"
        learning_curve = self.learning_curves.get(key)
        
        if not learning_curve:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rewards', 'Losses', 'Epsilon Decay', 'Q-Values'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot rewards
        fig.add_trace(
            go.Scatter(
                x=list(range(len(learning_curve.rewards))),
                y=learning_curve.rewards,
                mode='lines',
                name='Rewards',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Plot losses
        if learning_curve.losses:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(learning_curve.losses))),
                    y=learning_curve.losses,
                    mode='lines',
                    name='Losses',
                    line=dict(color='red')
                ),
                row=1, col=2
            )
        
        # Plot epsilon decay
        if learning_curve.epsilon_values:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(learning_curve.epsilon_values))),
                    y=learning_curve.epsilon_values,
                    mode='lines',
                    name='Epsilon',
                    line=dict(color='green')
                ),
                row=2, col=1
            )
        
        # Plot Q-values
        if learning_curve.q_values:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(learning_curve.q_values))),
                    y=learning_curve.q_values,
                    mode='lines',
                    name='Q-Values',
                    line=dict(color='purple')
                ),
                row=2, col=2
            )
        
        # Add convergence line if available
        if learning_curve.convergence_epoch:
            fig.add_vline(
                x=learning_curve.convergence_epoch,
                line_dash="dash",
                line_color="red",
                annotation_text="Convergence"
            )
        
        fig.update_layout(
            title=f"Learning Curve - {intersection_id} ({algorithm})",
            showlegend=True,
            height=600
        )
        
        return fig


class ComparativeDashboard:
    """Comparative dashboard for before/after optimization"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Dashboard data
        self.baseline_metrics = {}
        self.optimized_metrics = {}
        self.comparison_data = {}
    
    def add_baseline_metrics(self, intersection_id: str, 
                           metrics: Dict[str, float], timestamp: datetime):
        """Add baseline metrics for comparison"""
        if intersection_id not in self.baseline_metrics:
            self.baseline_metrics[intersection_id] = []
        
        self.baseline_metrics[intersection_id].append({
            'timestamp': timestamp,
            'metrics': metrics
        })
    
    def add_optimized_metrics(self, intersection_id: str, 
                            metrics: Dict[str, float], timestamp: datetime):
        """Add optimized metrics for comparison"""
        if intersection_id not in self.optimized_metrics:
            self.optimized_metrics[intersection_id] = []
        
        self.optimized_metrics[intersection_id].append({
            'timestamp': timestamp,
            'metrics': metrics
        })
    
    def calculate_improvements(self, intersection_id: str) -> Dict[str, float]:
        """Calculate improvement percentages"""
        baseline_data = self.baseline_metrics.get(intersection_id, [])
        optimized_data = self.optimized_metrics.get(intersection_id, [])
        
        if not baseline_data or not optimized_data:
            return {}
        
        # Calculate average metrics
        baseline_avg = self._calculate_average_metrics(baseline_data)
        optimized_avg = self._calculate_average_metrics(optimized_data)
        
        improvements = {}
        for metric in baseline_avg:
            if metric in optimized_avg and baseline_avg[metric] != 0:
                improvement = (
                    (optimized_avg[metric] - baseline_avg[metric]) / 
                    baseline_avg[metric] * 100
                )
                improvements[metric] = improvement
        
        return improvements
    
    def _calculate_average_metrics(self, data: List[Dict]) -> Dict[str, float]:
        """Calculate average metrics from data"""
        if not data:
            return {}
        
        all_metrics = set()
        for entry in data:
            all_metrics.update(entry['metrics'].keys())
        
        averages = {}
        for metric in all_metrics:
            values = [entry['metrics'].get(metric, 0) for entry in data]
            averages[metric] = np.mean(values)
        
        return averages
    
    def create_comparison_plot(self, intersection_id: str, 
                             metric_type: str) -> go.Figure:
        """Create comparison plot for specific metric"""
        baseline_data = self.baseline_metrics.get(intersection_id, [])
        optimized_data = self.optimized_metrics.get(intersection_id, [])
        
        if not baseline_data or not optimized_data:
            return go.Figure()
        
        # Extract metric values
        baseline_values = [entry['metrics'].get(metric_type, 0) for entry in baseline_data]
        optimized_values = [entry['metrics'].get(metric_type, 0) for entry in optimized_data]
        
        baseline_timestamps = [entry['timestamp'] for entry in baseline_data]
        optimized_timestamps = [entry['timestamp'] for entry in optimized_data]
        
        fig = go.Figure()
        
        # Add baseline line
        fig.add_trace(go.Scatter(
            x=baseline_timestamps,
            y=baseline_values,
            mode='lines',
            name='Baseline (Webster)',
            line=dict(color='red', dash='dash')
        ))
        
        # Add optimized line
        fig.add_trace(go.Scatter(
            x=optimized_timestamps,
            y=optimized_values,
            mode='lines',
            name='ML Optimized',
            line=dict(color='blue', dash='solid')
        ))
        
        # Calculate improvement
        improvement = self.calculate_improvements(intersection_id).get(metric_type, 0)
        
        fig.update_layout(
            title=f"{metric_type.title()} Comparison - {intersection_id}<br>Improvement: {improvement:.1f}%",
            xaxis_title="Time",
            yaxis_title=metric_type.title(),
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_summary_dashboard(self, intersection_id: str) -> go.Figure:
        """Create comprehensive summary dashboard"""
        improvements = self.calculate_improvements(intersection_id)
        
        if not improvements:
            return go.Figure()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Improvements', 'Metric Comparison', 
                          'Improvement Distribution', 'Key Metrics'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "table"}]]
        )
        
        # Bar chart of improvements
        metrics = list(improvements.keys())
        values = list(improvements.values())
        
        fig.add_trace(
            go.Bar(x=metrics, y=values, name='Improvement %'),
            row=1, col=1
        )
        
        # Scatter plot of baseline vs optimized
        baseline_avg = self._calculate_average_metrics(self.baseline_metrics.get(intersection_id, []))
        optimized_avg = self._calculate_average_metrics(self.optimized_metrics.get(intersection_id, []))
        
        fig.add_trace(
            go.Scatter(
                x=list(baseline_avg.values()),
                y=list(optimized_avg.values()),
                mode='markers+text',
                text=list(baseline_avg.keys()),
                name='Baseline vs Optimized'
            ),
            row=1, col=2
        )
        
        # Pie chart of improvement distribution
        positive_improvements = [v for v in values if v > 0]
        negative_improvements = [v for v in values if v < 0]
        
        fig.add_trace(
            go.Pie(
                labels=['Positive Improvements', 'Negative Changes'],
                values=[len(positive_improvements), len(negative_improvements)],
                name="Improvement Distribution"
            ),
            row=2, col=1
        )
        
        # Table of key metrics
        table_data = []
        for metric, improvement in improvements.items():
            table_data.append([metric, f"{improvement:.1f}%"])
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Improvement %']),
                cells=dict(values=list(zip(*table_data)) if table_data else [[], []])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Performance Dashboard - {intersection_id}",
            height=800,
            showlegend=True
        )
        
        return fig


class PredictiveAnalytics:
    """Predictive analytics for traffic pattern forecasting"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Prediction models
        self.forecasting_models = {}
        self.scalers = {}
        
        # Prediction parameters
        self.forecast_horizon = self.config.get('forecast_horizon', 60)  # minutes
        self.training_window = self.config.get('training_window', 1440)  # 24 hours in minutes
    
    def train_forecasting_model(self, intersection_id: str, 
                              metric_type: MetricType,
                              historical_data: List[PerformanceMetric]):
        """Train forecasting model for specific metric"""
        if not historical_data:
            return
        
        # Prepare training data
        df = pd.DataFrame([{
            'timestamp': m.timestamp,
            'value': m.value
        } for m in historical_data])
        
        df = df.set_index('timestamp').sort_index()
        
        # Create features
        features = self._create_time_features(df)
        
        if len(features) < 10:  # Need minimum data points
            return
        
        # Train model
        X = features[:-self.forecast_horizon]
        y = df['value'].iloc[self.forecast_horizon:].values
        
        if len(X) != len(y):
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train linear regression model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Store model and scaler
        key = f"{intersection_id}_{metric_type.value}"
        self.forecasting_models[key] = model
        self.scalers[key] = scaler
        
        self.logger.info(f"Trained forecasting model for {key}")
    
    def _create_time_features(self, df: pd.DataFrame) -> np.ndarray:
        """Create time-based features for forecasting"""
        features = []
        
        for i in range(len(df)):
            row_features = []
            
            # Hour of day
            row_features.append(df.index[i].hour)
            
            # Day of week
            row_features.append(df.index[i].weekday())
            
            # Month
            row_features.append(df.index[i].month)
            
            # Lag features (previous values)
            for lag in [1, 5, 15, 30, 60]:  # 1, 5, 15, 30, 60 minutes ago
                if i >= lag:
                    row_features.append(df['value'].iloc[i - lag])
                else:
                    row_features.append(0)
            
            # Rolling statistics
            if i >= 10:
                rolling_mean = df['value'].iloc[i-10:i].mean()
                rolling_std = df['value'].iloc[i-10:i].std()
                row_features.extend([rolling_mean, rolling_std])
            else:
                row_features.extend([0, 0])
            
            features.append(row_features)
        
        return np.array(features)
    
    def forecast(self, intersection_id: str, metric_type: MetricType,
                current_data: List[PerformanceMetric]) -> List[Dict[str, Any]]:
        """Generate forecast for specific metric"""
        key = f"{intersection_id}_{metric_type.value}"
        
        if key not in self.forecasting_models:
            return []
        
        model = self.forecasting_models[key]
        scaler = self.scalers[key]
        
        # Prepare current data
        df = pd.DataFrame([{
            'timestamp': m.timestamp,
            'value': m.value
        } for m in current_data])
        
        df = df.set_index('timestamp').sort_index()
        
        # Create features for last data point
        features = self._create_time_features(df)
        
        if len(features) == 0:
            return []
        
        # Use last data point for forecasting
        last_features = features[-1].reshape(1, -1)
        last_features_scaled = scaler.transform(last_features)
        
        # Generate forecast
        forecast_values = []
        current_features = last_features_scaled.copy()
        
        for i in range(self.forecast_horizon):
            # Predict next value
            pred_value = model.predict(current_features)[0]
            forecast_values.append(pred_value)
            
            # Update features for next prediction (simplified)
            # In practice, this would be more sophisticated
            current_features[0, 0] = (df.index[-1].hour + i // 60) % 24  # Update hour
            current_features[0, 1] = (df.index[-1].weekday() + i // (24 * 60)) % 7  # Update day
            
            # Update lag features
            if i > 0:
                current_features[0, 3] = forecast_values[-1]  # Update 1-minute lag
        
        # Create forecast results
        forecast_results = []
        base_time = df.index[-1]
        
        for i, value in enumerate(forecast_values):
            forecast_time = base_time + timedelta(minutes=i+1)
            forecast_results.append({
                'timestamp': forecast_time,
                'value': value,
                'confidence_interval': self._calculate_confidence_interval(value, i+1)
            })
        
        return forecast_results
    
    def _calculate_confidence_interval(self, value: float, horizon: int) -> Tuple[float, float]:
        """Calculate confidence interval for forecast"""
        # Simplified confidence interval calculation
        # In practice, this would use more sophisticated methods
        uncertainty = 0.1 * horizon  # Increase uncertainty with horizon
        lower = value * (1 - uncertainty)
        upper = value * (1 + uncertainty)
        return (lower, upper)


class AnomalyDetector:
    """Anomaly detection for unusual traffic behaviors"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Anomaly detection models
        self.isolation_forests = {}
        self.scalers = {}
        
        # Detection parameters
        self.contamination = self.config.get('contamination', 0.1)
        self.min_samples = self.config.get('min_samples', 100)
        self.anomaly_threshold = self.config.get('anomaly_threshold', 0.5)
    
    def train_anomaly_model(self, intersection_id: str, 
                          metric_type: MetricType,
                          historical_data: List[PerformanceMetric]):
        """Train anomaly detection model"""
        if len(historical_data) < self.min_samples:
            return
        
        # Prepare training data
        values = np.array([m.value for m in historical_data]).reshape(-1, 1)
        
        # Scale data
        scaler = StandardScaler()
        values_scaled = scaler.fit_transform(values)
        
        # Train Isolation Forest
        model = IsolationForest(
            contamination=self.contamination,
            random_state=42
        )
        model.fit(values_scaled)
        
        # Store model and scaler
        key = f"{intersection_id}_{metric_type.value}"
        self.isolation_forests[key] = model
        self.scalers[key] = scaler
        
        self.logger.info(f"Trained anomaly detection model for {key}")
    
    def detect_anomalies(self, intersection_id: str, 
                        metric_type: MetricType,
                        current_data: List[PerformanceMetric]) -> List[AnomalyDetection]:
        """Detect anomalies in current data"""
        key = f"{intersection_id}_{metric_type.value}"
        
        if key not in self.isolation_forests:
            return []
        
        model = self.isolation_forests[key]
        scaler = self.scalers[key]
        
        anomalies = []
        
        for metric in current_data:
            # Scale value
            value_scaled = scaler.transform([[metric.value]])
            
            # Predict anomaly
            anomaly_score = model.decision_function(value_scaled)[0]
            is_anomaly = model.predict(value_scaled)[0] == -1
            
            if is_anomaly or abs(anomaly_score) > self.anomaly_threshold:
                # Determine severity
                severity = self._determine_severity(anomaly_score)
                
                # Create anomaly detection result
                anomaly = AnomalyDetection(
                    anomaly_id=str(uuid.uuid4()),
                    timestamp=metric.timestamp,
                    intersection_id=intersection_id,
                    metric_type=metric_type,
                    anomaly_score=anomaly_score,
                    is_anomaly=is_anomaly,
                    description=self._generate_anomaly_description(metric, anomaly_score),
                    severity=severity
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _determine_severity(self, anomaly_score: float) -> str:
        """Determine anomaly severity based on score"""
        abs_score = abs(anomaly_score)
        
        if abs_score > 2.0:
            return 'critical'
        elif abs_score > 1.5:
            return 'high'
        elif abs_score > 1.0:
            return 'medium'
        else:
            return 'low'
    
    def _generate_anomaly_description(self, metric: PerformanceMetric, 
                                    anomaly_score: float) -> str:
        """Generate human-readable anomaly description"""
        if anomaly_score < -1.0:
            return f"Unusually low {metric.metric_type.value}: {metric.value:.2f} {metric.unit}"
        elif anomaly_score > 1.0:
            return f"Unusually high {metric.metric_type.value}: {metric.value:.2f} {metric.unit}"
        else:
            return f"Anomalous {metric.metric_type.value}: {metric.value:.2f} {metric.unit}"


class PerformanceAnalytics:
    """
    Main Performance Analytics Engine
    
    Features:
    - Real-time metrics collection and analysis
    - Learning curve visualization and convergence analysis
    - Comparative dashboards for before/after optimization
    - Predictive analytics for traffic pattern forecasting
    - Anomaly detection for unusual traffic behaviors
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.metrics_collector = RealTimeMetricsCollector(config.get('metrics_collector', {}))
        self.learning_analyzer = LearningCurveAnalyzer(config.get('learning_analyzer', {}))
        self.comparative_dashboard = ComparativeDashboard(config.get('dashboard', {}))
        self.predictive_analytics = PredictiveAnalytics(config.get('predictive', {}))
        self.anomaly_detector = AnomalyDetector(config.get('anomaly_detector', {}))
        
        # Analytics state
        self.is_running = False
        self.analytics_thread = None
        
        self.logger.info("Performance Analytics Engine initialized")
    
    def start_analytics(self):
        """Start performance analytics"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start metrics collection
        self.metrics_collector.start_collection()
        
        # Start analytics thread
        self.analytics_thread = threading.Thread(target=self._analytics_loop, daemon=True)
        self.analytics_thread.start()
        
        self.logger.info("Performance analytics started")
    
    def stop_analytics(self):
        """Stop performance analytics"""
        self.is_running = False
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        # Wait for analytics thread
        if self.analytics_thread:
            self.analytics_thread.join(timeout=5)
        
        self.logger.info("Performance analytics stopped")
    
    def _analytics_loop(self):
        """Main analytics processing loop"""
        while self.is_running:
            try:
                # Process analytics
                self._process_analytics()
                time.sleep(60)  # Process every minute
            except Exception as e:
                self.logger.error(f"Error in analytics loop: {e}")
                time.sleep(30)
    
    def _process_analytics(self):
        """Process analytics for all intersections"""
        # Get all intersection IDs
        intersection_ids = set()
        for metric in self.metrics_collector.metrics_buffer:
            intersection_ids.add(metric.intersection_id)
        
        # Process each intersection
        for intersection_id in intersection_ids:
            try:
                # Train forecasting models
                self._train_forecasting_models(intersection_id)
                
                # Train anomaly detection models
                self._train_anomaly_models(intersection_id)
                
                # Detect anomalies
                self._detect_current_anomalies(intersection_id)
                
            except Exception as e:
                self.logger.error(f"Error processing analytics for {intersection_id}: {e}")
    
    def _train_forecasting_models(self, intersection_id: str):
        """Train forecasting models for intersection"""
        for metric_type in MetricType:
            historical_data = self.metrics_collector.get_metrics(
                intersection_id, metric_type
            )
            
            if len(historical_data) >= 100:  # Need sufficient data
                self.predictive_analytics.train_forecasting_model(
                    intersection_id, metric_type, historical_data
                )
    
    def _train_anomaly_models(self, intersection_id: str):
        """Train anomaly detection models for intersection"""
        for metric_type in MetricType:
            historical_data = self.metrics_collector.get_metrics(
                intersection_id, metric_type
            )
            
            if len(historical_data) >= 100:  # Need sufficient data
                self.anomaly_detector.train_anomaly_model(
                    intersection_id, metric_type, historical_data
                )
    
    def _detect_current_anomalies(self, intersection_id: str):
        """Detect anomalies in current data"""
        for metric_type in MetricType:
            recent_data = self.metrics_collector.get_metrics(
                intersection_id, metric_type,
                start_time=datetime.now() - timedelta(minutes=10)
            )
            
            if recent_data:
                anomalies = self.anomaly_detector.detect_anomalies(
                    intersection_id, metric_type, recent_data
                )
                
                # Log critical anomalies
                for anomaly in anomalies:
                    if anomaly.severity in ['high', 'critical']:
                        self.logger.warning(f"Anomaly detected: {anomaly.description}")
    
    def add_learning_curve(self, intersection_id: str, algorithm: str,
                          timestamps: List[datetime], rewards: List[float],
                          losses: List[float] = None, epsilon_values: List[float] = None,
                          q_values: List[float] = None):
        """Add learning curve data"""
        learning_curve = LearningCurve(
            intersection_id=intersection_id,
            algorithm=algorithm,
            timestamps=timestamps,
            rewards=rewards,
            losses=losses or [],
            epsilon_values=epsilon_values or [],
            q_values=q_values or []
        )
        
        self.learning_analyzer.add_learning_curve(intersection_id, algorithm, learning_curve)
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        return {
            'metrics_collected': len(self.metrics_collector.metrics_buffer),
            'learning_curves': len(self.learning_analyzer.learning_curves),
            'forecasting_models': len(self.predictive_analytics.forecasting_models),
            'anomaly_models': len(self.anomaly_detector.isolation_forests),
            'intersections_monitored': len(self.metrics_collector.intersection_metrics),
            'is_running': self.is_running
        }
    
    def get_performance_report(self, intersection_id: str) -> Dict[str, Any]:
        """Get detailed performance report for intersection"""
        report = {
            'intersection_id': intersection_id,
            'timestamp': datetime.now(),
            'metrics_summary': {},
            'learning_curves': {},
            'forecasts': {},
            'anomalies': []
        }
        
        # Get metrics summary
        for metric_type in MetricType:
            summary = self.metrics_collector.get_metric_summary(intersection_id, metric_type)
            if summary:
                report['metrics_summary'][metric_type.value] = summary
        
        # Get learning curve analysis
        for algorithm in ['ml', 'webster']:
            analysis = self.learning_analyzer.get_convergence_analysis(intersection_id, algorithm)
            if analysis:
                report['learning_curves'][algorithm] = analysis
        
        # Get forecasts
        for metric_type in MetricType:
            recent_data = self.metrics_collector.get_metrics(
                intersection_id, metric_type,
                start_time=datetime.now() - timedelta(hours=1)
            )
            
            if recent_data:
                forecast = self.predictive_analytics.forecast(
                    intersection_id, metric_type, recent_data
                )
                if forecast:
                    report['forecasts'][metric_type.value] = forecast
        
        # Get recent anomalies
        for metric_type in MetricType:
            recent_data = self.metrics_collector.get_metrics(
                intersection_id, metric_type,
                start_time=datetime.now() - timedelta(minutes=30)
            )
            
            if recent_data:
                anomalies = self.anomaly_detector.detect_anomalies(
                    intersection_id, metric_type, recent_data
                )
                report['anomalies'].extend([asdict(anomaly) for anomaly in anomalies])
        
        return report
