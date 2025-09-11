"""
Enhanced Traffic Prediction Model for ML-based Signal Optimization
Implements multiple prediction algorithms: LSTM, ARIMA, Prophet, and ensemble methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import pickle
import os
from enum import Enum

# ML Libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Deep Learning (optional)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logging.warning("TensorFlow not available. LSTM predictions will be disabled.")

# Time Series Libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    logging.warning("Statsmodels not available. ARIMA predictions will be disabled.")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet not available. Prophet predictions will be disabled.")

from config.ml_config import TrafficPredictionConfig, get_config


class PredictionModel(Enum):
    """Available prediction models"""
    LSTM = "lstm"
    ARIMA = "arima"
    PROPHET = "prophet"
    RANDOM_FOREST = "random_forest"
    LINEAR_REGRESSION = "linear_regression"
    ENSEMBLE = "ensemble"


@dataclass
class PredictionResult:
    """Traffic prediction result"""
    intersection_id: str
    timestamp: datetime
    predictions: Dict[str, Any]
    confidence: float
    model_used: str
    prediction_horizon: int
    features_used: List[str]
    processing_time: float


@dataclass
class ModelPerformance:
    """Model performance metrics"""
    model_name: str
    mae: float
    mse: float
    rmse: float
    r2_score: float
    mape: float
    training_time: float
    prediction_time: float


class TrafficDataProcessor:
    """Data preprocessing and feature engineering for traffic prediction"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scalers = {}
        self.feature_columns = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features from raw traffic data"""
        df = df.copy()
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                             (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Traffic flow features
        df['total_vehicles'] = df[['north_lane', 'south_lane', 'east_lane', 'west_lane']].sum(axis=1)
        df['north_south_ratio'] = (df['north_lane'] + df['south_lane']) / (df['total_vehicles'] + 1)
        df['east_west_ratio'] = (df['east_lane'] + df['west_lane']) / (df['total_vehicles'] + 1)
        
        # Rolling statistics
        for window in [5, 15, 30]:
            for lane in ['north_lane', 'south_lane', 'east_lane', 'west_lane']:
                df[f'{lane}_ma_{window}'] = df[lane].rolling(window=window, min_periods=1).mean()
                df[f'{lane}_std_{window}'] = df[lane].rolling(window=window, min_periods=1).std()
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            for lane in ['north_lane', 'south_lane', 'east_lane', 'west_lane']:
                df[f'{lane}_lag_{lag}'] = df[lane].shift(lag)
        
        # Weather encoding
        weather_mapping = {'clear': 1, 'cloudy': 2, 'rainy': 3, 'foggy': 4, 'stormy': 5, 'snowy': 6}
        df['weather_encoded'] = df['weather_condition'].map(weather_mapping).fillna(1)
        
        # Environmental features normalization
        if 'temperature' in df.columns:
            df['temperature_norm'] = (df['temperature'] - df['temperature'].mean()) / df['temperature'].std()
        if 'humidity' in df.columns:
            df['humidity_norm'] = (df['humidity'] - df['humidity'].mean()) / df['humidity'].std()
        if 'visibility' in df.columns:
            df['visibility_norm'] = (df['visibility'] - df['visibility'].mean()) / df['visibility'].std()
        
        return df
    
    def prepare_sequences(self, df: pd.DataFrame, sequence_length: int, target_columns: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        # Select feature columns
        feature_columns = [col for col in df.columns if col not in ['timestamp', 'intersection_id']]
        X = df[feature_columns].values
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(df)):
            X_sequences.append(X[i-sequence_length:i])
            y_sequences.append(df[target_columns].iloc[i].values)
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def scale_features(self, X: np.ndarray, fit: bool = False, scaler_name: str = "default") -> np.ndarray:
        """Scale features using StandardScaler"""
        if fit:
            self.scalers[scaler_name] = StandardScaler()
            return self.scalers[scaler_name].fit_transform(X)
        else:
            if scaler_name in self.scalers:
                return self.scalers[scaler_name].transform(X)
            else:
                self.logger.warning(f"Scaler {scaler_name} not found, returning original data")
                return X


class LSTMPredictor:
    """LSTM-based traffic flow predictor"""
    
    def __init__(self, config: TrafficPredictionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def build_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """Build LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        model = Sequential([
            LSTM(self.config.lstm_units, return_sequences=True, input_shape=input_shape),
            Dropout(self.config.dropout_rate),
            BatchNormalization(),
            
            LSTM(self.config.lstm_units // 2, return_sequences=False),
            Dropout(self.config.dropout_rate),
            BatchNormalization(),
            
            Dense(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(4, activation='linear')  # 4 lanes
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available")
        
        self.logger.info("Training LSTM model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Build model
        self.model = self.build_model((X.shape[1], X.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train model
        start_time = time.time()
        history = self.model.fit(
            X_scaled, y,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        training_time = time.time() - start_time
        
        self.is_trained = True
        
        return {
            'history': history.history,
            'training_time': training_time,
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1]
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained LSTM model"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model and self.is_trained:
            self.model.save(filepath)
            # Save scaler
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            self.logger.info(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        if TENSORFLOW_AVAILABLE:
            self.model = load_model(filepath)
            # Load scaler
            scaler_path = filepath.replace('.h5', '_scaler.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
            self.is_trained = True
            self.logger.info(f"LSTM model loaded from {filepath}")


class ARIMAPredictor:
    """ARIMA-based traffic flow predictor"""
    
    def __init__(self, config: TrafficPredictionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.is_trained = False
        
    def _check_stationarity(self, series: pd.Series) -> bool:
        """Check if time series is stationary"""
        if not ARIMA_AVAILABLE:
            return False
        
        result = adfuller(series.dropna())
        return result[1] <= 0.05  # p-value <= 0.05 means stationary
    
    def _make_stationary(self, series: pd.Series) -> Tuple[pd.Series, int]:
        """Make time series stationary using differencing"""
        diff_order = 0
        current_series = series.copy()
        
        while not self._check_stationarity(current_series) and diff_order < 3:
            current_series = current_series.diff().dropna()
            diff_order += 1
        
        return current_series, diff_order
    
    def train(self, df: pd.DataFrame, target_columns: List[str]) -> Dict[str, Any]:
        """Train ARIMA models for each target column"""
        if not ARIMA_AVAILABLE:
            raise ImportError("Statsmodels not available")
        
        self.logger.info("Training ARIMA models...")
        training_results = {}
        
        for column in target_columns:
            if column not in df.columns:
                continue
            
            self.logger.info(f"Training ARIMA for {column}")
            
            # Prepare time series
            ts = df.set_index('timestamp')[column].dropna()
            
            # Make stationary
            stationary_ts, diff_order = self._make_stationary(ts)
            
            # Fit ARIMA model
            try:
                model = ARIMA(ts, order=(1, diff_order, 1))
                fitted_model = model.fit()
                self.models[column] = fitted_model
                training_results[column] = {
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'diff_order': diff_order
                }
            except Exception as e:
                self.logger.error(f"Error training ARIMA for {column}: {e}")
                training_results[column] = {'error': str(e)}
        
        self.is_trained = True
        return training_results
    
    def predict(self, steps: int, target_columns: List[str]) -> Dict[str, np.ndarray]:
        """Make predictions using trained ARIMA models"""
        if not self.is_trained:
            raise ValueError("Models not trained")
        
        predictions = {}
        
        for column in target_columns:
            if column in self.models:
                try:
                    forecast = self.models[column].forecast(steps=steps)
                    predictions[column] = forecast
                except Exception as e:
                    self.logger.error(f"Error predicting with ARIMA for {column}: {e}")
                    predictions[column] = np.zeros(steps)
            else:
                predictions[column] = np.zeros(steps)
        
        return predictions


class ProphetPredictor:
    """Prophet-based traffic flow predictor"""
    
    def __init__(self, config: TrafficPredictionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.is_trained = False
        
    def train(self, df: pd.DataFrame, target_columns: List[str]) -> Dict[str, Any]:
        """Train Prophet model"""
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available")
        
        self.logger.info("Training Prophet model...")
        
        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        prophet_df = df[['timestamp']].copy()
        prophet_df.columns = ['ds']
        
        # Use total vehicles as target
        prophet_df['y'] = df[target_columns].sum(axis=1)
        
        # Create and fit model
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='multiplicative'
        )
        
        # Add weather as additional regressor
        if 'weather_encoded' in df.columns:
            self.model.add_regressor('weather')
            prophet_df['weather'] = df['weather_encoded']
        
        self.model.fit(prophet_df)
        self.is_trained = True
        
        return {'status': 'trained'}
    
    def predict(self, steps: int, future_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Make predictions using Prophet model"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(periods=steps, freq='min')
        
        # Add regressors if available
        if future_df is not None and 'weather_encoded' in future_df.columns:
            future['weather'] = future_df['weather_encoded'].iloc[:len(future)]
        
        # Make prediction
        forecast = self.model.predict(future)
        
        # Return only the future predictions
        future_forecast = forecast.tail(steps)
        
        return {
            'predictions': future_forecast['yhat'].values,
            'lower_bound': future_forecast['yhat_lower'].values,
            'upper_bound': future_forecast['yhat_upper'].values
        }


class EnsemblePredictor:
    """Ensemble predictor combining multiple models"""
    
    def __init__(self, config: TrafficPredictionConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.weights = {}
        self.is_trained = False
        
    def add_model(self, name: str, model: Any, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] /= total_weight
    
    def predict(self, X: np.ndarray, steps: int = 1) -> Dict[str, np.ndarray]:
        """Make ensemble predictions"""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict'):
                    pred = model.predict(X)
                elif hasattr(model, 'forecast'):
                    pred = model.forecast(steps)
                else:
                    continue
                
                predictions[name] = pred
            except Exception as e:
                self.logger.error(f"Error with model {name}: {e}")
        
        # Weighted average of predictions
        if predictions:
            ensemble_pred = np.zeros_like(list(predictions.values())[0])
            for name, pred in predictions.items():
                ensemble_pred += self.weights[name] * pred
            predictions['ensemble'] = ensemble_pred
        
        return predictions


class EnhancedTrafficPredictor:
    """
    Enhanced traffic predictor with multiple algorithms and ensemble methods
    
    Features:
    - LSTM for sequence modeling
    - ARIMA for time series analysis
    - Prophet for seasonal patterns
    - Random Forest for feature-based prediction
    - Ensemble methods for improved accuracy
    - Comprehensive feature engineering
    - Model performance evaluation
    """
    
    def __init__(self, config: Optional[TrafficPredictionConfig] = None):
        self.config = config or get_config().traffic_prediction
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_processor = TrafficDataProcessor()
        self.predictors = {}
        self.ensemble = EnsemblePredictor(self.config)
        self.performance_metrics = {}
        
        # Initialize individual predictors
        if TENSORFLOW_AVAILABLE:
            self.predictors['lstm'] = LSTMPredictor(self.config)
        
        if ARIMA_AVAILABLE:
            self.predictors['arima'] = ARIMAPredictor(self.config)
        
        if PROPHET_AVAILABLE:
            self.predictors['prophet'] = ProphetPredictor(self.config)
        
        # Traditional ML models
        self.predictors['random_forest'] = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.predictors['linear_regression'] = LinearRegression()
        
        self.logger.info("Enhanced traffic predictor initialized")
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare training data with feature engineering"""
        self.logger.info("Preparing training data...")
        
        # Create features
        df_processed = self.data_processor.create_features(df)
        
        # Define target columns
        target_columns = ['north_lane', 'south_lane', 'east_lane', 'west_lane']
        
        # Remove rows with NaN values
        df_processed = df_processed.dropna()
        
        self.logger.info(f"Prepared data shape: {df_processed.shape}")
        return df_processed, target_columns
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train all available models"""
        self.logger.info("Training all models...")
        
        # Prepare data
        df_processed, target_columns = self.prepare_training_data(df)
        
        training_results = {}
        
        # Train LSTM
        if 'lstm' in self.predictors:
            try:
                X_seq, y_seq = self.data_processor.prepare_sequences(
                    df_processed, self.config.sequence_length, target_columns
                )
                lstm_result = self.predictors['lstm'].train(X_seq, y_seq)
                training_results['lstm'] = lstm_result
                self.ensemble.add_model('lstm', self.predictors['lstm'], weight=0.3)
            except Exception as e:
                self.logger.error(f"LSTM training failed: {e}")
        
        # Train ARIMA
        if 'arima' in self.predictors:
            try:
                arima_result = self.predictors['arima'].train(df_processed, target_columns)
                training_results['arima'] = arima_result
                self.ensemble.add_model('arima', self.predictors['arima'], weight=0.2)
            except Exception as e:
                self.logger.error(f"ARIMA training failed: {e}")
        
        # Train Prophet
        if 'prophet' in self.predictors:
            try:
                prophet_result = self.predictors['prophet'].train(df_processed, target_columns)
                training_results['prophet'] = prophet_result
                self.ensemble.add_model('prophet', self.predictors['prophet'], weight=0.2)
            except Exception as e:
                self.logger.error(f"Prophet training failed: {e}")
        
        # Train traditional ML models
        feature_columns = [col for col in df_processed.columns 
                          if col not in ['timestamp', 'intersection_id'] + target_columns]
        X = df_processed[feature_columns].values
        y = df_processed[target_columns].values
        
        # Scale features
        X_scaled = self.data_processor.scale_features(X, fit=True, scaler_name="ml_models")
        
        for model_name in ['random_forest', 'linear_regression']:
            try:
                self.predictors[model_name].fit(X_scaled, y)
                training_results[model_name] = {'status': 'trained'}
                self.ensemble.add_model(model_name, self.predictors[model_name], weight=0.15)
            except Exception as e:
                self.logger.error(f"{model_name} training failed: {e}")
        
        self.logger.info("All models trained successfully")
        return training_results
    
    def evaluate_models(self, df: pd.DataFrame) -> Dict[str, ModelPerformance]:
        """Evaluate model performance"""
        self.logger.info("Evaluating model performance...")
        
        # Prepare data
        df_processed, target_columns = self.prepare_training_data(df)
        
        # Split data for evaluation
        split_idx = int(len(df_processed) * 0.8)
        train_df = df_processed.iloc[:split_idx]
        test_df = df_processed.iloc[split_idx:]
        
        performance_results = {}
        
        # Evaluate each model
        for model_name, model in self.predictors.items():
            try:
                if model_name == 'lstm':
                    X_seq, y_seq = self.data_processor.prepare_sequences(
                        test_df, self.config.sequence_length, target_columns
                    )
                    predictions = model.predict(X_seq)
                    y_true = y_seq
                
                elif model_name in ['arima', 'prophet']:
                    predictions = model.predict(self.config.prediction_horizon, target_columns)
                    y_true = test_df[target_columns].values[:self.config.prediction_horizon]
                
                else:  # Traditional ML models
                    feature_columns = [col for col in test_df.columns 
                                      if col not in ['timestamp', 'intersection_id'] + target_columns]
                    X_test = test_df[feature_columns].values
                    X_test_scaled = self.data_processor.scale_features(X_test, scaler_name="ml_models")
                    predictions = model.predict(X_test_scaled)
                    y_true = test_df[target_columns].values
                
                # Calculate metrics
                mae = mean_absolute_error(y_true, predictions)
                mse = mean_squared_error(y_true, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, predictions)
                mape = np.mean(np.abs((y_true - predictions) / (y_true + 1e-8))) * 100
                
                performance_results[model_name] = ModelPerformance(
                    model_name=model_name,
                    mae=mae,
                    mse=mse,
                    rmse=rmse,
                    r2_score=r2,
                    mape=mape,
                    training_time=0.0,  # Would need to track this during training
                    prediction_time=0.0  # Would need to measure this
                )
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {e}")
        
        self.performance_metrics = performance_results
        return performance_results
    
    def predict_traffic_flow(self, intersection_id: str, prediction_horizon: int = 15) -> Dict[str, Any]:
        """
        Predict traffic flow for an intersection
        
        Args:
            intersection_id: ID of the intersection
            prediction_horizon: Minutes ahead to predict
            
        Returns:
            Dictionary containing predictions and metadata
        """
        self.logger.info(f"Predicting traffic flow for {intersection_id}")
        
        # This would typically load historical data for the intersection
        # For now, we'll generate a mock prediction
        predictions = {
            'lane_counts': {
                'north_lane': np.random.randint(5, 25),
                'south_lane': np.random.randint(5, 25),
                'east_lane': np.random.randint(5, 25),
                'west_lane': np.random.randint(5, 25)
            },
            'avg_speed': np.random.uniform(20, 50),
            'weather_condition': 'clear',
            'temperature': np.random.uniform(15, 30),
            'humidity': np.random.uniform(40, 80),
            'visibility': np.random.uniform(8, 15),
            'confidence': 0.8,
            'prediction_horizon': prediction_horizon,
            'model_used': 'ensemble'
        }
        
        return predictions
    
    def save_models(self, base_path: str):
        """Save all trained models"""
        os.makedirs(base_path, exist_ok=True)
        
        for model_name, model in self.predictors.items():
            try:
                if hasattr(model, 'save_model'):
                    model.save_model(os.path.join(base_path, f"{model_name}.h5"))
                elif hasattr(model, 'save'):
                    model.save(os.path.join(base_path, f"{model_name}.pkl"))
                else:
                    # Save using pickle
                    with open(os.path.join(base_path, f"{model_name}.pkl"), 'wb') as f:
                        pickle.dump(model, f)
                
                self.logger.info(f"Saved {model_name} model")
            except Exception as e:
                self.logger.error(f"Error saving {model_name}: {e}")
    
    def load_models(self, base_path: str):
        """Load all trained models"""
        for model_name, model in self.predictors.items():
            try:
                if hasattr(model, 'load_model'):
                    model.load_model(os.path.join(base_path, f"{model_name}.h5"))
                elif hasattr(model, 'load'):
                    model.load(os.path.join(base_path, f"{model_name}.pkl"))
                else:
                    # Load using pickle
                    with open(os.path.join(base_path, f"{model_name}.pkl"), 'rb') as f:
                        loaded_model = pickle.load(f)
                        self.predictors[model_name] = loaded_model
                
                self.logger.info(f"Loaded {model_name} model")
            except Exception as e:
                self.logger.error(f"Error loading {model_name}: {e}")
    
    def get_model_performance(self) -> Dict[str, ModelPerformance]:
        """Get performance metrics for all models"""
        return self.performance_metrics


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Test the traffic predictor
    predictor = EnhancedTrafficPredictor()
    
    # Generate mock training data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='min')
    mock_data = pd.DataFrame({
        'timestamp': dates,
        'intersection_id': 'junction-1',
        'north_lane': np.random.randint(0, 30, len(dates)),
        'south_lane': np.random.randint(0, 30, len(dates)),
        'east_lane': np.random.randint(0, 30, len(dates)),
        'west_lane': np.random.randint(0, 30, len(dates)),
        'avg_speed': np.random.uniform(20, 50, len(dates)),
        'weather_condition': np.random.choice(['clear', 'cloudy', 'rainy'], len(dates)),
        'temperature': np.random.uniform(15, 30, len(dates)),
        'humidity': np.random.uniform(40, 80, len(dates)),
        'visibility': np.random.uniform(8, 15, len(dates))
    })
    
    print("Training models...")
    training_results = predictor.train_models(mock_data)
    print(f"Training results: {training_results}")
    
    print("Evaluating models...")
    performance = predictor.evaluate_models(mock_data)
    for model_name, metrics in performance.items():
        print(f"{model_name}: MAE={metrics.mae:.2f}, R²={metrics.r2_score:.2f}")
    
    print("Making prediction...")
    prediction = predictor.predict_traffic_flow("junction-1", 15)
    print(f"Prediction: {prediction}")
