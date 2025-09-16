"""
Advanced ML Features for Traffic Optimization
Phase 3: ML Model Validation & Performance Analytics

Features:
- Transfer learning for new intersection deployment
- Meta-learning for rapid adaptation to seasonal traffic changes
- Ensemble methods combining multiple ML approaches
- Reinforcement learning with human feedback (RLHF)
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class LearningType(Enum):
    """Types of learning approaches"""
    TRANSFER_LEARNING = "transfer_learning"
    META_LEARNING = "meta_learning"
    ENSEMBLE = "ensemble"
    RLHF = "rlhf"


@dataclass
class TransferLearningResult:
    """Transfer learning result"""
    source_intersection: str
    target_intersection: str
    transfer_accuracy: float
    fine_tuning_epochs: int
    performance_improvement: float
    transferred_layers: List[str]
    adaptation_time: float


@dataclass
class MetaLearningResult:
    """Meta-learning result"""
    task_id: str
    adaptation_time: float
    few_shot_accuracy: float
    meta_parameters: Dict[str, float]
    task_similarity: float
    learning_curve: List[float]


@dataclass
class EnsembleResult:
    """Ensemble method result"""
    ensemble_id: str
    base_models: List[str]
    ensemble_accuracy: float
    individual_accuracies: Dict[str, float]
    model_weights: Dict[str, float]
    diversity_score: float


@dataclass
class RLHFResult:
    """RLHF result"""
    feedback_id: str
    human_feedback: float
    model_update: Dict[str, float]
    confidence_adjustment: float
    learning_rate_adjustment: float
    feedback_quality: float


class TransferLearning:
    """Transfer learning for new intersection deployment"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Transfer learning parameters
        self.freeze_layers = self.config.get('freeze_layers', 0.7)
        self.learning_rate_multiplier = self.config.get('learning_rate_multiplier', 0.1)
        self.fine_tuning_epochs = self.config.get('fine_tuning_epochs', 50)
        
        # Model storage
        self.source_models = {}
        self.transfer_history = []
        
        self.logger.info("Transfer learning system initialized")
    
    def transfer_model(self, source_intersection: str, target_intersection: str,
                     source_model: Any, target_data: pd.DataFrame) -> TransferLearningResult:
        """Transfer model from source to target intersection"""
        try:
            start_time = time.time()
            
            # Create target model based on source model
            target_model = self._create_target_model(source_model, target_data)
            
            # Fine-tune on target data
            fine_tuned_model = self._fine_tune_model(
                target_model, target_data, target_intersection
            )
            
            # Evaluate transfer performance
            transfer_accuracy = self._evaluate_transfer_performance(
                fine_tuned_model, target_data
            )
            
            # Calculate performance improvement
            baseline_accuracy = self._get_baseline_accuracy(target_data)
            performance_improvement = (
                (transfer_accuracy - baseline_accuracy) / baseline_accuracy * 100
            )
            
            # Create transfer result
            result = TransferLearningResult(
                source_intersection=source_intersection,
                target_intersection=target_intersection,
                transfer_accuracy=transfer_accuracy,
                fine_tuning_epochs=self.fine_tuning_epochs,
                performance_improvement=performance_improvement,
                transferred_layers=self._get_transferred_layers(source_model),
                adaptation_time=time.time() - start_time
            )
            
            self.transfer_history.append(result)
            self.logger.info(f"Transfer learning completed: {source_intersection} -> {target_intersection}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in transfer learning: {e}")
            return None
    
    def _create_target_model(self, source_model: Any, target_data: pd.DataFrame) -> Any:
        """Create target model based on source model"""
        # This is a simplified implementation
        # In practice, this would involve more sophisticated model architecture adaptation
        
        if hasattr(source_model, 'copy'):
            target_model = source_model.copy()
        else:
            # Create new model with same architecture
            target_model = type(source_model)()
        
        # Adjust model for target data characteristics
        self._adapt_model_architecture(target_model, target_data)
        
        return target_model
    
    def _adapt_model_architecture(self, model: Any, target_data: pd.DataFrame):
        """Adapt model architecture for target data"""
        # Adjust input dimensions if needed
        if hasattr(model, 'n_features_in_'):
            expected_features = target_data.shape[1]
            if model.n_features_in_ != expected_features:
                # This would require more sophisticated architecture adaptation
                self.logger.warning(f"Feature dimension mismatch: {model.n_features_in_} vs {expected_features}")
        
        # Adjust output dimensions if needed
        # This would depend on the specific model type
    
    def _fine_tune_model(self, model: Any, target_data: pd.DataFrame, 
                        target_intersection: str) -> Any:
        """Fine-tune model on target data"""
        try:
            # Prepare data
            X = target_data.drop('target', axis=1) if 'target' in target_data.columns else target_data
            y = target_data['target'] if 'target' in target_data.columns else None
            
            if y is None:
                # Use unsupervised fine-tuning
                model.fit(X)
            else:
                # Use supervised fine-tuning
                model.fit(X, y)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Error in fine-tuning: {e}")
            return model
    
    def _evaluate_transfer_performance(self, model: Any, target_data: pd.DataFrame) -> float:
        """Evaluate transfer learning performance"""
        try:
            X = target_data.drop('target', axis=1) if 'target' in target_data.columns else target_data
            y = target_data['target'] if 'target' in target_data.columns else None
            
            if y is None:
                return 0.0
            
            # Make predictions
            predictions = model.predict(X)
            
            # Calculate accuracy
            mse = mean_squared_error(y, predictions)
            accuracy = 1.0 / (1.0 + mse)  # Convert MSE to accuracy-like metric
            
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Error evaluating transfer performance: {e}")
            return 0.0
    
    def _get_baseline_accuracy(self, target_data: pd.DataFrame) -> float:
        """Get baseline accuracy for comparison"""
        try:
            if 'target' not in target_data.columns:
                return 0.0
            
            y = target_data['target']
            baseline_prediction = np.mean(y)
            
            mse = mean_squared_error(y, [baseline_prediction] * len(y))
            accuracy = 1.0 / (1.0 + mse)
            
            return accuracy
            
        except Exception:
            return 0.0
    
    def _get_transferred_layers(self, source_model: Any) -> List[str]:
        """Get list of transferred layers"""
        # This would depend on the specific model architecture
        # For now, return a generic list
        return ['input_layer', 'hidden_layers', 'output_layer']


class MetaLearning:
    """Meta-learning for rapid adaptation to seasonal traffic changes"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Meta-learning parameters
        self.meta_learning_rate = self.config.get('meta_learning_rate', 0.001)
        self.inner_learning_rate = self.config.get('inner_learning_rate', 0.01)
        self.meta_epochs = self.config.get('meta_epochs', 100)
        self.few_shot_samples = self.config.get('few_shot_samples', 10)
        
        # Meta-parameters storage
        self.meta_parameters = {}
        self.task_embeddings = {}
        self.adaptation_history = []
        
        self.logger.info("Meta-learning system initialized")
    
    def train_meta_learner(self, training_tasks: List[Dict[str, Any]]):
        """Train meta-learner on multiple tasks"""
        try:
            self.logger.info(f"Training meta-learner on {len(training_tasks)} tasks")
            
            # Initialize meta-parameters
            self.meta_parameters = self._initialize_meta_parameters()
            
            # Train on multiple tasks
            for epoch in range(self.meta_epochs):
                task_losses = []
                
                for task in training_tasks:
                    # Inner loop: adapt to specific task
                    adapted_params = self._adapt_to_task(task, self.meta_parameters)
                    
                    # Calculate task loss
                    task_loss = self._calculate_task_loss(task, adapted_params)
                    task_losses.append(task_loss)
                
                # Outer loop: update meta-parameters
                self._update_meta_parameters(task_losses)
                
                if epoch % 10 == 0:
                    avg_loss = np.mean(task_losses)
                    self.logger.info(f"Meta-epoch {epoch}: Average task loss = {avg_loss:.4f}")
            
            self.logger.info("Meta-learning training completed")
            
        except Exception as e:
            self.logger.error(f"Error in meta-learning training: {e}")
    
    def adapt_to_new_task(self, new_task: Dict[str, Any]) -> MetaLearningResult:
        """Adapt to new task using meta-learning"""
        try:
            start_time = time.time()
            
            # Calculate task similarity
            task_similarity = self._calculate_task_similarity(new_task)
            
            # Adapt meta-parameters to new task
            adapted_params = self._adapt_to_task(new_task, self.meta_parameters)
            
            # Evaluate few-shot performance
            few_shot_accuracy = self._evaluate_few_shot_performance(new_task, adapted_params)
            
            # Create meta-learning result
            result = MetaLearningResult(
                task_id=str(uuid.uuid4()),
                adaptation_time=time.time() - start_time,
                few_shot_accuracy=few_shot_accuracy,
                meta_parameters=adapted_params,
                task_similarity=task_similarity,
                learning_curve=self._get_learning_curve(new_task, adapted_params)
            )
            
            self.adaptation_history.append(result)
            self.logger.info(f"Meta-learning adaptation completed for task {result.task_id}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in meta-learning adaptation: {e}")
            return None
    
    def _initialize_meta_parameters(self) -> Dict[str, float]:
        """Initialize meta-parameters"""
        return {
            'learning_rate': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0001,
            'dropout_rate': 0.1,
            'hidden_size': 64,
            'num_layers': 2
        }
    
    def _adapt_to_task(self, task: Dict[str, Any], meta_params: Dict[str, float]) -> Dict[str, float]:
        """Adapt meta-parameters to specific task"""
        # Simplified adaptation - in practice, this would be more sophisticated
        adapted_params = meta_params.copy()
        
        # Adjust parameters based on task characteristics
        if 'traffic_volume' in task:
            if task['traffic_volume'] > 1000:  # High volume
                adapted_params['learning_rate'] *= 1.2
                adapted_params['hidden_size'] = int(adapted_params['hidden_size'] * 1.5)
            else:  # Low volume
                adapted_params['learning_rate'] *= 0.8
                adapted_params['hidden_size'] = int(adapted_params['hidden_size'] * 0.8)
        
        if 'time_of_day' in task:
            if task['time_of_day'] in ['morning_rush', 'evening_rush']:
                adapted_params['momentum'] = 0.95  # Higher momentum for rush hours
            else:
                adapted_params['momentum'] = 0.85  # Lower momentum for off-peak
        
        return adapted_params
    
    def _calculate_task_loss(self, task: Dict[str, Any], adapted_params: Dict[str, float]) -> float:
        """Calculate loss for specific task"""
        # Simplified loss calculation
        # In practice, this would involve actual model training and evaluation
        
        base_loss = 0.1
        
        # Adjust loss based on task complexity
        if 'traffic_volume' in task and task['traffic_volume'] > 1000:
            base_loss *= 1.2  # Higher loss for complex tasks
        
        if 'weather_condition' in task and task['weather_condition'] != 'clear':
            base_loss *= 1.1  # Higher loss for adverse weather
        
        return base_loss
    
    def _update_meta_parameters(self, task_losses: List[float]):
        """Update meta-parameters based on task losses"""
        # Simplified meta-parameter update
        # In practice, this would use gradient-based meta-learning algorithms
        
        avg_loss = np.mean(task_losses)
        
        # Adjust meta-parameters based on average loss
        if avg_loss > 0.2:  # High loss
            self.meta_parameters['learning_rate'] *= 1.1
        elif avg_loss < 0.05:  # Low loss
            self.meta_parameters['learning_rate'] *= 0.9
        
        # Ensure parameters stay within bounds
        self.meta_parameters['learning_rate'] = max(0.001, min(0.1, self.meta_parameters['learning_rate']))
    
    def _calculate_task_similarity(self, new_task: Dict[str, Any]) -> float:
        """Calculate similarity between new task and training tasks"""
        # Simplified similarity calculation
        # In practice, this would use more sophisticated similarity metrics
        
        if not self.adaptation_history:
            return 0.5  # Default similarity
        
        # Calculate similarity based on task characteristics
        similarities = []
        
        for historical_task in self.adaptation_history[-10:]:  # Last 10 tasks
            similarity = 0.0
            
            # Compare traffic volume
            if 'traffic_volume' in new_task and 'traffic_volume' in historical_task.meta_parameters:
                vol_sim = 1.0 - abs(new_task['traffic_volume'] - historical_task.meta_parameters.get('traffic_volume', 0)) / 1000
                similarity += vol_sim * 0.3
            
            # Compare time patterns
            if 'time_of_day' in new_task and 'time_of_day' in historical_task.meta_parameters:
                if new_task['time_of_day'] == historical_task.meta_parameters.get('time_of_day', ''):
                    similarity += 0.4
            
            # Compare weather conditions
            if 'weather_condition' in new_task and 'weather_condition' in historical_task.meta_parameters:
                if new_task['weather_condition'] == historical_task.meta_parameters.get('weather_condition', ''):
                    similarity += 0.3
            
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.5
    
    def _evaluate_few_shot_performance(self, task: Dict[str, Any], adapted_params: Dict[str, float]) -> float:
        """Evaluate few-shot learning performance"""
        # Simplified performance evaluation
        # In practice, this would involve actual model training and testing
        
        base_performance = 0.7
        
        # Adjust performance based on task similarity
        task_similarity = self._calculate_task_similarity(task)
        performance = base_performance + (task_similarity * 0.3)
        
        return min(1.0, performance)
    
    def _get_learning_curve(self, task: Dict[str, Any], adapted_params: Dict[str, float]) -> List[float]:
        """Get learning curve for task adaptation"""
        # Simplified learning curve generation
        # In practice, this would track actual learning progress
        
        curve_length = 20
        base_performance = 0.5
        final_performance = self._evaluate_few_shot_performance(task, adapted_params)
        
        # Generate exponential learning curve
        learning_curve = []
        for i in range(curve_length):
            progress = i / (curve_length - 1)
            performance = base_performance + (final_performance - base_performance) * (1 - np.exp(-3 * progress))
            learning_curve.append(performance)
        
        return learning_curve


class EnsembleMethods:
    """Ensemble methods combining multiple ML approaches"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Ensemble parameters
        self.ensemble_types = self.config.get('ensemble_types', ['voting', 'stacking'])
        self.diversity_threshold = self.config.get('diversity_threshold', 0.1)
        self.performance_threshold = self.config.get('performance_threshold', 0.7)
        
        # Ensemble models
        self.ensemble_models = {}
        self.base_models = {}
        self.ensemble_history = []
        
        self.logger.info("Ensemble methods system initialized")
    
    def create_ensemble(self, intersection_id: str, 
                       base_models: Dict[str, Any],
                       training_data: pd.DataFrame) -> EnsembleResult:
        """Create ensemble of base models"""
        try:
            # Prepare data
            X = training_data.drop('target', axis=1) if 'target' in training_data.columns else training_data
            y = training_data['target'] if 'target' in training_data.columns else None
            
            if y is None:
                self.logger.warning("No target variable found for ensemble training")
                return None
            
            # Train individual models
            individual_accuracies = {}
            trained_models = {}
            
            for model_name, model in base_models.items():
                try:
                    # Train model
                    model.fit(X, y)
                    trained_models[model_name] = model
                    
                    # Evaluate individual performance
                    predictions = model.predict(X)
                    mse = mean_squared_error(y, predictions)
                    accuracy = 1.0 / (1.0 + mse)
                    individual_accuracies[model_name] = accuracy
                    
                except Exception as e:
                    self.logger.warning(f"Error training {model_name}: {e}")
                    continue
            
            if not trained_models:
                self.logger.error("No models could be trained successfully")
                return None
            
            # Create ensemble models
            ensemble_models = {}
            ensemble_accuracies = {}
            
            # Voting ensemble
            if 'voting' in self.ensemble_types:
                voting_ensemble = VotingRegressor(
                    [(name, model) for name, model in trained_models.items()]
                )
                voting_ensemble.fit(X, y)
                ensemble_models['voting'] = voting_ensemble
                
                # Evaluate voting ensemble
                voting_predictions = voting_ensemble.predict(X)
                voting_mse = mean_squared_error(y, voting_predictions)
                ensemble_accuracies['voting'] = 1.0 / (1.0 + voting_mse)
            
            # Stacking ensemble
            if 'stacking' in self.ensemble_types:
                stacking_ensemble = StackingRegressor(
                    [(name, model) for name, model in trained_models.items()],
                    final_estimator=LinearRegression()
                )
                stacking_ensemble.fit(X, y)
                ensemble_models['stacking'] = stacking_ensemble
                
                # Evaluate stacking ensemble
                stacking_predictions = stacking_ensemble.predict(X)
                stacking_mse = mean_squared_error(y, stacking_predictions)
                ensemble_accuracies['stacking'] = 1.0 / (1.0 + stacking_mse)
            
            # Select best ensemble
            best_ensemble_type = max(ensemble_accuracies.keys(), key=lambda k: ensemble_accuracies[k])
            best_ensemble = ensemble_models[best_ensemble_type]
            best_accuracy = ensemble_accuracies[best_ensemble_type]
            
            # Calculate diversity score
            diversity_score = self._calculate_diversity_score(trained_models, X, y)
            
            # Calculate model weights
            model_weights = self._calculate_model_weights(individual_accuracies)
            
            # Create ensemble result
            result = EnsembleResult(
                ensemble_id=str(uuid.uuid4()),
                base_models=list(trained_models.keys()),
                ensemble_accuracy=best_accuracy,
                individual_accuracies=individual_accuracies,
                model_weights=model_weights,
                diversity_score=diversity_score
            )
            
            # Store ensemble model
            self.ensemble_models[intersection_id] = best_ensemble
            self.base_models[intersection_id] = trained_models
            self.ensemble_history.append(result)
            
            self.logger.info(f"Ensemble created for {intersection_id}: {best_ensemble_type} with accuracy {best_accuracy:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating ensemble: {e}")
            return None
    
    def _calculate_diversity_score(self, models: Dict[str, Any], 
                                 X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate diversity score of ensemble models"""
        try:
            predictions = {}
            
            # Get predictions from all models
            for name, model in models.items():
                try:
                    pred = model.predict(X)
                    predictions[name] = pred
                except:
                    continue
            
            if len(predictions) < 2:
                return 0.0
            
            # Calculate pairwise diversity
            diversities = []
            model_names = list(predictions.keys())
            
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    pred1 = predictions[model_names[i]]
                    pred2 = predictions[model_names[j]]
                    
                    # Calculate correlation diversity
                    correlation = np.corrcoef(pred1, pred2)[0, 1]
                    diversity = 1.0 - abs(correlation)
                    diversities.append(diversity)
            
            return np.mean(diversities) if diversities else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error calculating diversity score: {e}")
            return 0.0
    
    def _calculate_model_weights(self, individual_accuracies: Dict[str, float]) -> Dict[str, float]:
        """Calculate model weights based on individual accuracies"""
        if not individual_accuracies:
            return {}
        
        # Normalize accuracies to get weights
        total_accuracy = sum(individual_accuracies.values())
        
        if total_accuracy == 0:
            # Equal weights if no accuracy
            equal_weight = 1.0 / len(individual_accuracies)
            return {name: equal_weight for name in individual_accuracies.keys()}
        
        weights = {}
        for name, accuracy in individual_accuracies.items():
            weights[name] = accuracy / total_accuracy
        
        return weights
    
    def predict_with_ensemble(self, intersection_id: str, X: pd.DataFrame) -> Dict[str, Any]:
        """Make prediction using ensemble model"""
        if intersection_id not in self.ensemble_models:
            return {'error': 'No ensemble model found for intersection'}
        
        try:
            ensemble_model = self.ensemble_models[intersection_id]
            base_models = self.base_models[intersection_id]
            
            # Get ensemble prediction
            ensemble_prediction = ensemble_model.predict(X)
            
            # Get individual model predictions
            individual_predictions = {}
            for name, model in base_models.items():
                try:
                    pred = model.predict(X)
                    individual_predictions[name] = pred.tolist() if hasattr(pred, 'tolist') else pred
                except:
                    individual_predictions[name] = None
            
            return {
                'ensemble_prediction': ensemble_prediction.tolist() if hasattr(ensemble_prediction, 'tolist') else ensemble_prediction,
                'individual_predictions': individual_predictions,
                'prediction_confidence': self._calculate_prediction_confidence(individual_predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Error making ensemble prediction: {e}")
            return {'error': str(e)}
    
    def _calculate_prediction_confidence(self, individual_predictions: Dict[str, Any]) -> float:
        """Calculate prediction confidence based on individual model agreement"""
        try:
            valid_predictions = [pred for pred in individual_predictions.values() if pred is not None]
            
            if len(valid_predictions) < 2:
                return 0.5  # Default confidence
            
            # Calculate variance of predictions
            predictions_array = np.array(valid_predictions)
            variance = np.var(predictions_array, axis=0).mean()
            
            # Convert variance to confidence (lower variance = higher confidence)
            confidence = 1.0 / (1.0 + variance)
            
            return min(1.0, confidence)
            
        except Exception:
            return 0.5


class RLHF:
    """Reinforcement Learning with Human Feedback"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # RLHF parameters
        self.feedback_weight = self.config.get('feedback_weight', 0.1)
        self.learning_rate_adjustment = self.config.get('learning_rate_adjustment', 0.01)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.feedback_history_size = self.config.get('feedback_history_size', 1000)
        
        # Feedback storage
        self.feedback_history = deque(maxlen=self.feedback_history_size)
        self.model_updates = {}
        self.feedback_quality_scores = {}
        
        self.logger.info("RLHF system initialized")
    
    def process_human_feedback(self, intersection_id: str, 
                             model_prediction: float,
                             human_feedback: float,
                             context: Dict[str, Any]) -> RLHFResult:
        """Process human feedback and update model"""
        try:
            # Calculate feedback quality
            feedback_quality = self._calculate_feedback_quality(
                model_prediction, human_feedback, context
            )
            
            # Calculate model update
            model_update = self._calculate_model_update(
                model_prediction, human_feedback, feedback_quality
            )
            
            # Adjust learning parameters
            confidence_adjustment = self._calculate_confidence_adjustment(feedback_quality)
            learning_rate_adjustment = self._calculate_learning_rate_adjustment(feedback_quality)
            
            # Create RLHF result
            result = RLHFResult(
                feedback_id=str(uuid.uuid4()),
                human_feedback=human_feedback,
                model_update=model_update,
                confidence_adjustment=confidence_adjustment,
                learning_rate_adjustment=learning_rate_adjustment,
                feedback_quality=feedback_quality
            )
            
            # Store feedback
            self.feedback_history.append({
                'intersection_id': intersection_id,
                'timestamp': datetime.now(),
                'model_prediction': model_prediction,
                'human_feedback': human_feedback,
                'context': context,
                'result': result
            })
            
            # Update model parameters
            self._update_model_parameters(intersection_id, model_update)
            
            self.logger.info(f"Processed human feedback for {intersection_id}: quality={feedback_quality:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing human feedback: {e}")
            return None
    
    def _calculate_feedback_quality(self, model_prediction: float, 
                                  human_feedback: float, 
                                  context: Dict[str, Any]) -> float:
        """Calculate quality of human feedback"""
        try:
            # Base quality on prediction-feedback agreement
            prediction_error = abs(model_prediction - human_feedback)
            max_error = max(model_prediction, human_feedback, 1.0)
            agreement_score = 1.0 - (prediction_error / max_error)
            
            # Adjust based on context
            context_factor = 1.0
            
            # Check for expert feedback
            if context.get('expert_feedback', False):
                context_factor *= 1.2
            
            # Check for feedback consistency
            if 'previous_feedback' in context:
                prev_feedback = context['previous_feedback']
                consistency = 1.0 - abs(human_feedback - prev_feedback) / max(human_feedback, prev_feedback, 1.0)
                context_factor *= (0.8 + 0.4 * consistency)
            
            # Check for feedback timing
            if 'feedback_delay' in context:
                delay = context['feedback_delay']
                if delay < 60:  # Less than 1 minute
                    context_factor *= 1.1
                elif delay > 3600:  # More than 1 hour
                    context_factor *= 0.9
            
            quality = min(1.0, agreement_score * context_factor)
            
            return quality
            
        except Exception:
            return 0.5  # Default quality
    
    def _calculate_model_update(self, model_prediction: float, 
                              human_feedback: float, 
                              feedback_quality: float) -> Dict[str, float]:
        """Calculate model update based on human feedback"""
        try:
            # Calculate prediction error
            error = human_feedback - model_prediction
            
            # Weight update by feedback quality
            weighted_error = error * feedback_quality * self.feedback_weight
            
            # Calculate parameter updates
            updates = {
                'bias_adjustment': weighted_error,
                'learning_rate_multiplier': 1.0 + (weighted_error * 0.1),
                'confidence_adjustment': feedback_quality - 0.5,
                'exploration_adjustment': -weighted_error * 0.05
            }
            
            return updates
            
        except Exception:
            return {'bias_adjustment': 0.0, 'learning_rate_multiplier': 1.0, 
                   'confidence_adjustment': 0.0, 'exploration_adjustment': 0.0}
    
    def _calculate_confidence_adjustment(self, feedback_quality: float) -> float:
        """Calculate confidence adjustment based on feedback quality"""
        # Higher quality feedback increases confidence
        return (feedback_quality - 0.5) * 0.2
    
    def _calculate_learning_rate_adjustment(self, feedback_quality: float) -> float:
        """Calculate learning rate adjustment based on feedback quality"""
        # Higher quality feedback allows for larger learning rate adjustments
        return (feedback_quality - 0.5) * self.learning_rate_adjustment
    
    def _update_model_parameters(self, intersection_id: str, model_update: Dict[str, float]):
        """Update model parameters based on feedback"""
        if intersection_id not in self.model_updates:
            self.model_updates[intersection_id] = {
                'bias_adjustment': 0.0,
                'learning_rate_multiplier': 1.0,
                'confidence_adjustment': 0.0,
                'exploration_adjustment': 0.0
            }
        
        # Accumulate updates
        for param, adjustment in model_update.items():
            if param in self.model_updates[intersection_id]:
                self.model_updates[intersection_id][param] += adjustment
        
        # Apply bounds
        self.model_updates[intersection_id]['learning_rate_multiplier'] = max(0.1, min(2.0, 
            self.model_updates[intersection_id]['learning_rate_multiplier']))
        self.model_updates[intersection_id]['confidence_adjustment'] = max(-0.5, min(0.5,
            self.model_updates[intersection_id]['confidence_adjustment']))
    
    def get_feedback_summary(self, intersection_id: str) -> Dict[str, Any]:
        """Get feedback summary for intersection"""
        intersection_feedback = [
            f for f in self.feedback_history 
            if f['intersection_id'] == intersection_id
        ]
        
        if not intersection_feedback:
            return {'total_feedback': 0}
        
        # Calculate statistics
        feedback_qualities = [f['result'].feedback_quality for f in intersection_feedback]
        human_feedbacks = [f['human_feedback'] for f in intersection_feedback]
        model_predictions = [f['model_prediction'] for f in intersection_feedback]
        
        return {
            'total_feedback': len(intersection_feedback),
            'average_feedback_quality': np.mean(feedback_qualities),
            'feedback_consistency': 1.0 - np.std(human_feedbacks) / (np.mean(human_feedbacks) + 1e-8),
            'prediction_accuracy': 1.0 - np.mean(np.abs(np.array(human_feedbacks) - np.array(model_predictions))) / (np.mean(human_feedbacks) + 1e-8),
            'recent_trend': self._calculate_recent_trend(intersection_feedback[-10:]) if len(intersection_feedback) >= 10 else 'insufficient_data'
        }
    
    def _calculate_recent_trend(self, recent_feedback: List[Dict]) -> str:
        """Calculate recent feedback trend"""
        if len(recent_feedback) < 3:
            return 'insufficient_data'
        
        qualities = [f['result'].feedback_quality for f in recent_feedback]
        
        # Simple trend calculation
        if len(qualities) >= 3:
            early_avg = np.mean(qualities[:len(qualities)//2])
            late_avg = np.mean(qualities[len(qualities)//2:])
            
            if late_avg > early_avg + 0.1:
                return 'improving'
            elif late_avg < early_avg - 0.1:
                return 'declining'
            else:
                return 'stable'
        
        return 'stable'


class AdvancedMLFeatures:
    """
    Main Advanced ML Features System
    
    Features:
    - Transfer learning for new intersection deployment
    - Meta-learning for rapid adaptation to seasonal traffic changes
    - Ensemble methods combining multiple ML approaches
    - Reinforcement learning with human feedback (RLHF)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.transfer_learning = TransferLearning(config.get('transfer_learning', {}))
        self.meta_learning = MetaLearning(config.get('meta_learning', {}))
        self.ensemble_methods = EnsembleMethods(config.get('ensemble_methods', {}))
        self.rlhf = RLHF(config.get('rlhf', {}))
        
        # Feature state
        self.is_active = False
        self.feature_history = []
        
        self.logger.info("Advanced ML Features system initialized")
    
    def activate_features(self):
        """Activate all advanced ML features"""
        self.is_active = True
        self.logger.info("Advanced ML features activated")
    
    def deactivate_features(self):
        """Deactivate all advanced ML features"""
        self.is_active = False
        self.logger.info("Advanced ML features deactivated")
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get summary of all advanced ML features"""
        return {
            'is_active': self.is_active,
            'transfer_learning': {
                'completed_transfers': len(self.transfer_learning.transfer_history),
                'available_source_models': len(self.transfer_learning.source_models)
            },
            'meta_learning': {
                'completed_adaptations': len(self.meta_learning.adaptation_history),
                'meta_parameters_trained': len(self.meta_learning.meta_parameters) > 0
            },
            'ensemble_methods': {
                'active_ensembles': len(self.ensemble_methods.ensemble_models),
                'ensemble_history': len(self.ensemble_methods.ensemble_history)
            },
            'rlhf': {
                'total_feedback': len(self.rlhf.feedback_history),
                'active_intersections': len(set(f['intersection_id'] for f in self.rlhf.feedback_history))
            }
        }
    
    def get_intersection_advanced_features(self, intersection_id: str) -> Dict[str, Any]:
        """Get advanced features status for specific intersection"""
        return {
            'intersection_id': intersection_id,
            'has_ensemble': intersection_id in self.ensemble_methods.ensemble_models,
            'has_transfer_learning': any(t.target_intersection == intersection_id for t in self.transfer_learning.transfer_history),
            'has_meta_learning': any(a.task_id for a in self.meta_learning.adaptation_history),
            'feedback_summary': self.rlhf.get_feedback_summary(intersection_id),
            'recommendations': self._generate_feature_recommendations(intersection_id)
        }
    
    def _generate_feature_recommendations(self, intersection_id: str) -> List[str]:
        """Generate recommendations for advanced features"""
        recommendations = []
        
        # Check ensemble availability
        if intersection_id not in self.ensemble_methods.ensemble_models:
            recommendations.append("Consider creating ensemble model for improved accuracy")
        
        # Check transfer learning opportunities
        if not any(t.target_intersection == intersection_id for t in self.transfer_learning.transfer_history):
            recommendations.append("Consider transfer learning from similar intersections")
        
        # Check meta-learning opportunities
        if not any(a.task_id for a in self.meta_learning.adaptation_history):
            recommendations.append("Consider meta-learning for seasonal adaptation")
        
        # Check RLHF opportunities
        feedback_summary = self.rlhf.get_feedback_summary(intersection_id)
        if feedback_summary['total_feedback'] < 10:
            recommendations.append("Collect more human feedback for RLHF improvement")
        
        return recommendations
