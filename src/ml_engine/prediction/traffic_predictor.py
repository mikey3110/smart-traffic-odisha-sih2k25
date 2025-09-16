"""
Traffic Prediction Models for ML Traffic Signal Optimization
Implements various ML models for predicting traffic patterns and flow
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
import pickle
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from config import TrafficPredictionConfig, get_config
from data import TrafficData


@dataclass
class PredictionResult:
    """Traffic prediction result"""
    timestamp: datetime
    intersection_id: str
    predicted_flows: Dict[str, float]  # Predicted flow rates for each approach
    predicted_counts: Dict[str, int]   # Predicted vehicle counts for each approach
    confidence: float                  # Prediction confidence (0-1)
    model_used: str                   # Model that made the prediction
    features_used: List[str]          # Features used for prediction


class TrafficPredictor:
    """
    Base class for traffic prediction models
    
    Implements common functionality for all prediction models including
    data preprocessing, feature engineering, and model evaluation.
    """
    
    def __init__(self, config: Optional[TrafficPredictionConfig] = None):
        self.config = config or get_config().traffic_prediction
        self.logger = logging.getLogger(__name__)
        
        # Model parameters
        self.model_type = self.config.model_type
        self.sequence_length = self.config.sequence_length
        self.prediction_horizon = self.config.prediction_horizon
        self.features = self.config.features
        
        # Data storage
        self.historical_data: List[TrafficData] = []
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # Model and training state
        self.model = None
        self.is_trained = False
        self.training_history = []
        
        # Performance metrics
        self.metrics = {
            'mse': [],
            'mae': [],
            'r2': [],
            'accuracy': []
        }
    
    def add_historical_data(self, traffic_data: TrafficData):
        """Add historical traffic data for training"""
        self.historical_data.append(traffic_data)
        
        # Keep only recent data (last 1000 records)
        if len(self.historical_data) > 1000:
            self.historical_data = self.historical_data[-1000:]
    
    def _extract_features(self, traffic_data: TrafficData) -> np.ndarray:
        """Extract features from traffic data"""
        features = []
        
        # Lane counts
        for lane in ['north_lane', 'south_lane', 'east_lane', 'west_lane']:
            features.append(traffic_data.lane_counts.get(lane, 0))
        
        # Average speed
        features.append(traffic_data.avg_speed or 0.0)
        
        # Time features
        hour = traffic_data.timestamp.hour
        day_of_week = traffic_data.timestamp.weekday()
        features.extend([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7)
        ])
        
        # Weather features
        weather_conditions = ['clear', 'cloudy', 'rainy', 'foggy', 'stormy', 'snowy']
        weather_vector = [0.0] * len(weather_conditions)
        if traffic_data.weather_condition in weather_conditions:
            weather_vector[weather_conditions.index(traffic_data.weather_condition)] = 1.0
        features.extend(weather_vector)
        
        # Environmental features
        features.extend([
            traffic_data.temperature or 20.0,
            traffic_data.humidity or 50.0,
            traffic_data.visibility or 10.0
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical records"""
        if len(self.historical_data) < self.sequence_length + 1:
            raise ValueError(f"Not enough historical data. Need at least {self.sequence_length + 1} records")
        
        X, y = [], []
        
        for i in range(len(self.historical_data) - self.sequence_length):
            # Input sequence
            sequence = []
            for j in range(i, i + self.sequence_length):
                features = self._extract_features(self.historical_data[j])
                sequence.append(features)
            X.append(np.array(sequence))
            
            # Target (next time step)
            target_features = self._extract_features(self.historical_data[i + self.sequence_length])
            y.append(target_features[:4])  # Only predict lane counts
        
        return np.array(X), np.array(y)
    
    def train(self) -> Dict[str, float]:
        """Train the prediction model"""
        if len(self.historical_data) < 10:
            raise ValueError("Not enough historical data for training")
        
        self.logger.info(f"Training {self.model_type.value} model with {len(self.historical_data)} records")
        
        # Prepare training data
        X, y = self._prepare_training_data()
        
        if len(X) == 0:
            raise ValueError("No valid training sequences found")
        
        # Reshape data for different model types
        if self.model_type.value == "lstm":
            # LSTM expects 3D data: (samples, timesteps, features)
            X_reshaped = X
        else:
            # Other models expect 2D data: (samples, features)
            X_reshaped = X.reshape(X.shape[0], -1)
        
        # Split data
        split_idx = int(len(X_reshaped) * (1 - self.config.validation_split))
        X_train, X_val = X_reshaped[:split_idx], X_reshaped[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        if X_train.ndim == 3:  # LSTM data
            X_train_scaled = X_train.reshape(-1, X_train.shape[-1])
            X_train_scaled = self.feature_scaler.fit_transform(X_train_scaled)
            X_train_scaled = X_train_scaled.reshape(X_train.shape)
            
            X_val_scaled = X_val.reshape(-1, X_val.shape[-1])
            X_val_scaled = self.feature_scaler.transform(X_val_scaled)
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
        else:
            X_train_scaled = self.feature_scaler.fit_transform(X_train)
            X_val_scaled = self.feature_scaler.transform(X_val)
        
        # Scale targets
        y_train_scaled = self.target_scaler.fit_transform(y_train)
        y_val_scaled = self.target_scaler.transform(y_val)
        
        # Train model based on type
        if self.model_type.value == "linear_regression":
            self.model = LinearRegression()
            self.model.fit(X_train_scaled, y_train_scaled)
        
        elif self.model_type.value == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.model.fit(X_train_scaled, y_train_scaled)
        
        elif self.model_type.value == "lstm":
            self.model = self._create_lstm_model(X_train_scaled.shape[1:])
            self._train_lstm_model(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
        
        else:
            raise ValueError(f"Unsupported model type: {self.model_type.value}")
        
        # Evaluate model
        train_pred = self._predict_internal(X_train_scaled)
        val_pred = self._predict_internal(X_val_scaled)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(y_train_scaled, train_pred)
        val_metrics = self._calculate_metrics(y_val_scaled, val_pred)
        
        self.is_trained = True
        self.training_history.append({
            'timestamp': datetime.now(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'data_size': len(self.historical_data)
        })
        
        self.logger.info(f"Training completed. Validation R²: {val_metrics['r2']:.3f}")
        
        return val_metrics
    
    def _create_lstm_model(self, input_shape: Tuple[int, ...]) -> Any:
        """Create LSTM model (simplified implementation)"""
        # This is a simplified LSTM implementation
        # In practice, you would use TensorFlow/Keras or PyTorch
        from sklearn.neural_network import MLPRegressor
        
        # Use MLP as LSTM substitute for this implementation
        model = MLPRegressor(
            hidden_layer_sizes=(self.config.lstm_units, self.config.lstm_units // 2),
            activation='relu',
            learning_rate_init=self.config.learning_rate,
            max_iter=self.config.epochs,
            random_state=42
        )
        
        return model
    
    def _train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray):
        """Train LSTM model"""
        # Reshape for MLP (LSTM substitute)
        X_train_reshaped = X_train.reshape(X_train.shape[0], -1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], -1)
        
        self.model.fit(X_train_reshaped, y_train)
    
    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """Internal prediction method"""
        if self.model_type.value == "lstm":
            X_reshaped = X.reshape(X.shape[0], -1)
            predictions = self.model.predict(X_reshaped)
        else:
            predictions = self.model.predict(X)
        
        return predictions
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate prediction metrics"""
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Calculate accuracy (simplified)
        accuracy = max(0, 1 - mse / np.var(y_true)) if np.var(y_true) > 0 else 0
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'accuracy': accuracy
        }
    
    def predict(self, traffic_data: TrafficData) -> PredictionResult:
        """Predict future traffic conditions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Extract features from current data
        current_features = self._extract_features(traffic_data)
        
        # Create input sequence (use recent historical data)
        if len(self.historical_data) >= self.sequence_length:
            sequence = []
            for i in range(max(0, len(self.historical_data) - self.sequence_length + 1), len(self.historical_data)):
                features = self._extract_features(self.historical_data[i])
                sequence.append(features)
            sequence.append(current_features)
            X = np.array(sequence[-self.sequence_length:])
        else:
            # Pad with current features if not enough history
            X = np.array([current_features] * self.sequence_length)
        
        # Reshape for model
        if self.model_type.value == "lstm":
            X_reshaped = X.reshape(1, X.shape[0], X.shape[1])
        else:
            X_reshaped = X.reshape(1, -1)
        
        # Scale features
        if X_reshaped.ndim == 3:  # LSTM data
            X_scaled = X_reshaped.reshape(-1, X_reshaped.shape[-1])
            X_scaled = self.feature_scaler.transform(X_scaled)
            X_scaled = X_scaled.reshape(X_reshaped.shape)
        else:
            X_scaled = self.feature_scaler.transform(X_reshaped)
        
        # Make prediction
        prediction_scaled = self._predict_internal(X_scaled)
        prediction = self.target_scaler.inverse_transform(prediction_scaled)
        
        # Convert to traffic counts and flows
        predicted_counts = {
            'north_lane': max(0, int(round(prediction[0][0]))),
            'south_lane': max(0, int(round(prediction[0][1]))),
            'east_lane': max(0, int(round(prediction[0][2]))),
            'west_lane': max(0, int(round(prediction[0][3])))
        }
        
        # Convert counts to flow rates (vehicles per hour)
        predicted_flows = {
            approach: count * 60  # Assume counts are per minute
            for approach, count in predicted_counts.items()
        }
        
        # Calculate confidence based on model performance
        confidence = self._calculate_prediction_confidence(prediction_scaled)
        
        return PredictionResult(
            timestamp=datetime.now() + timedelta(minutes=self.prediction_horizon),
            intersection_id=traffic_data.intersection_id,
            predicted_flows=predicted_flows,
            predicted_counts=predicted_counts,
            confidence=confidence,
            model_used=self.model_type.value,
            features_used=self.features
        )
    
    def _calculate_prediction_confidence(self, prediction_scaled: np.ndarray) -> float:
        """Calculate prediction confidence"""
        if not self.training_history:
            return 0.5
        
        # Use recent validation R² as confidence indicator
        recent_r2 = self.training_history[-1]['val_metrics']['r2']
        confidence = max(0.0, min(1.0, recent_r2))
        
        return confidence
    
    def evaluate_model(self, test_data: List[TrafficData]) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        if len(test_data) < self.sequence_length + 1:
            raise ValueError("Not enough test data for evaluation")
        
        # Prepare test data
        X_test, y_test = [], []
        
        for i in range(len(test_data) - self.sequence_length):
            sequence = []
            for j in range(i, i + self.sequence_length):
                features = self._extract_features(test_data[j])
                sequence.append(features)
            X_test.append(np.array(sequence))
            
            target_features = self._extract_features(test_data[i + self.sequence_length])
            y_test.append(target_features[:4])
        
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # Reshape and scale
        if self.model_type.value == "lstm":
            X_test_reshaped = X_test
        else:
            X_test_reshaped = X_test.reshape(X_test.shape[0], -1)
        
        if X_test_reshaped.ndim == 3:
            X_test_scaled = X_test_reshaped.reshape(-1, X_test_reshaped.shape[-1])
            X_test_scaled = self.feature_scaler.transform(X_test_scaled)
            X_test_scaled = X_test_scaled.reshape(X_test_reshaped.shape)
        else:
            X_test_scaled = self.feature_scaler.transform(X_test_reshaped)
        
        y_test_scaled = self.target_scaler.transform(y_test)
        
        # Make predictions
        predictions_scaled = self._predict_internal(X_test_scaled)
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test_scaled, predictions_scaled)
        
        self.logger.info(f"Model evaluation completed. R²: {metrics['r2']:.3f}")
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler,
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_scaler = model_data['feature_scaler']
        self.target_scaler = model_data['target_scaler']
        self.training_history = model_data['training_history']
        self.is_trained = model_data['is_trained']
        
        self.logger.info(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        return {
            'model_type': self.model_type.value,
            'is_trained': self.is_trained,
            'training_samples': len(self.historical_data),
            'sequence_length': self.sequence_length,
            'prediction_horizon': self.prediction_horizon,
            'features': self.features,
            'training_history': self.training_history,
            'recent_performance': self.training_history[-1] if self.training_history else None
        }


class TrafficPredictionEnsemble:
    """
    Ensemble of multiple traffic prediction models
    
    Combines predictions from different models to improve accuracy and robustness.
    """
    
    def __init__(self, models: List[TrafficPredictor]):
        self.models = models
        self.logger = logging.getLogger(__name__)
        self.weights = [1.0 / len(models)] * len(models)  # Equal weights initially
    
    def predict(self, traffic_data: TrafficData) -> PredictionResult:
        """Make ensemble prediction"""
        if not self.models:
            raise ValueError("No models available for ensemble prediction")
        
        predictions = []
        confidences = []
        
        for model in self.models:
            if model.is_trained:
                try:
                    pred = model.predict(traffic_data)
                    predictions.append(pred)
                    confidences.append(pred.confidence)
                except Exception as e:
                    self.logger.warning(f"Model {model.model_type.value} prediction failed: {e}")
        
        if not predictions:
            raise ValueError("No successful predictions from ensemble models")
        
        # Weighted average of predictions
        total_weight = sum(self.weights[:len(predictions)])
        weighted_flows = {}
        weighted_counts = {}
        
        for approach in ['north_lane', 'south_lane', 'east_lane', 'west_lane']:
            weighted_flows[approach] = sum(
                pred.predicted_flows[approach] * self.weights[i] 
                for i, pred in enumerate(predictions)
            ) / total_weight
            
            weighted_counts[approach] = int(round(sum(
                pred.predicted_counts[approach] * self.weights[i] 
                for i, pred in enumerate(predictions)
            ) / total_weight))
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences)
        
        return PredictionResult(
            timestamp=predictions[0].timestamp,
            intersection_id=traffic_data.intersection_id,
            predicted_flows=weighted_flows,
            predicted_counts=weighted_counts,
            confidence=avg_confidence,
            model_used=f"ensemble_{len(predictions)}_models",
            features_used=predictions[0].features_used
        )
    
    def update_weights(self, performance_scores: List[float]):
        """Update model weights based on performance"""
        if len(performance_scores) != len(self.models):
            raise ValueError("Performance scores must match number of models")
        
        # Normalize weights
        total_score = sum(performance_scores)
        if total_score > 0:
            self.weights = [score / total_score for score in performance_scores]
        else:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        self.logger.info(f"Updated ensemble weights: {self.weights}")



