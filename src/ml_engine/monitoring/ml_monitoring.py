"""
ML Monitoring & Observability System
Phase 3: ML Model Validation & Performance Analytics

Features:
- Model drift detection for changing traffic patterns
- Feature importance analysis and interpretability
- Performance degradation alerts and automatic retraining triggers
- Detailed logging for ML decision audit trails
- Explainable AI components for stakeholder presentations
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import shap
import lime
import lime.lime_tabular
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


class DriftType(Enum):
    """Types of model drift"""
    CONCEPT_DRIFT = "concept_drift"
    DATA_DRIFT = "data_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    FEATURE_DRIFT = "feature_drift"


class AlertLevel(Enum):
    """Alert levels for monitoring"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class DriftDetection:
    """Drift detection result"""
    drift_id: str
    timestamp: datetime
    intersection_id: str
    drift_type: DriftType
    drift_score: float
    severity: AlertLevel
    description: str
    affected_features: List[str]
    recommended_action: str
    confidence: float


@dataclass
class FeatureImportance:
    """Feature importance analysis result"""
    feature_name: str
    importance_score: float
    importance_rank: int
    permutation_importance: float
    shap_importance: float
    stability_score: float
    description: str


@dataclass
class ModelExplanation:
    """Model explanation for a specific prediction"""
    explanation_id: str
    timestamp: datetime
    intersection_id: str
    prediction: float
    feature_contributions: Dict[str, float]
    shap_values: Dict[str, float]
    lime_explanation: Dict[str, Any]
    confidence: float
    reasoning: str


@dataclass
class PerformanceAlert:
    """Performance degradation alert"""
    alert_id: str
    timestamp: datetime
    intersection_id: str
    alert_level: AlertLevel
    metric_name: str
    current_value: float
    threshold_value: float
    degradation_percentage: float
    description: str
    recommended_action: str


class ModelDriftDetector:
    """Model drift detection system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Drift detection parameters
        self.drift_threshold = self.config.get('drift_threshold', 0.1)
        self.window_size = self.config.get('window_size', 1000)
        self.min_samples = self.config.get('min_samples', 100)
        
        # Reference data storage
        self.reference_data = {}
        self.reference_models = {}
        
        # Drift detection history
        self.drift_history = deque(maxlen=1000)
        
        # Statistical tests
        self.ks_test_alpha = 0.05
        self.psi_threshold = 0.2
        
        self.logger.info("Model drift detector initialized")
    
    def set_reference_data(self, intersection_id: str, 
                          reference_data: List[Dict[str, Any]]):
        """Set reference data for drift detection"""
        self.reference_data[intersection_id] = reference_data
        self.logger.info(f"Set reference data for {intersection_id}")
    
    def set_reference_model(self, intersection_id: str, model: Any):
        """Set reference model for drift detection"""
        self.reference_models[intersection_id] = model
        self.logger.info(f"Set reference model for {intersection_id}")
    
    def detect_drift(self, intersection_id: str, 
                    current_data: List[Dict[str, Any]]) -> List[DriftDetection]:
        """Detect model drift for intersection"""
        if intersection_id not in self.reference_data:
            return []
        
        reference_data = self.reference_data[intersection_id]
        drift_detections = []
        
        try:
            # Convert to DataFrames
            ref_df = pd.DataFrame(reference_data)
            curr_df = pd.DataFrame(current_data)
            
            # Detect different types of drift
            drift_detections.extend(
                self._detect_data_drift(intersection_id, ref_df, curr_df)
            )
            drift_detections.extend(
                self._detect_concept_drift(intersection_id, ref_df, curr_df)
            )
            drift_detections.extend(
                self._detect_performance_drift(intersection_id, ref_df, curr_df)
            )
            drift_detections.extend(
                self._detect_feature_drift(intersection_id, ref_df, curr_df)
            )
            
            # Store drift detections
            for drift in drift_detections:
                self.drift_history.append(drift)
            
            return drift_detections
            
        except Exception as e:
            self.logger.error(f"Error detecting drift for {intersection_id}: {e}")
            return []
    
    def _detect_data_drift(self, intersection_id: str, 
                          ref_df: pd.DataFrame, 
                          curr_df: pd.DataFrame) -> List[DriftDetection]:
        """Detect data drift using statistical tests"""
        drift_detections = []
        
        # Get common numeric columns
        numeric_cols = ref_df.select_dtypes(include=[np.number]).columns
        common_cols = [col for col in numeric_cols if col in curr_df.columns]
        
        for col in common_cols:
            try:
                ref_values = ref_df[col].dropna()
                curr_values = curr_df[col].dropna()
                
                if len(ref_values) < 10 or len(curr_values) < 10:
                    continue
                
                # Kolmogorov-Smirnov test
                ks_statistic, ks_p_value = stats.ks_2samp(ref_values, curr_values)
                
                # Population Stability Index (PSI)
                psi_score = self._calculate_psi(ref_values, curr_values)
                
                # Determine drift severity
                drift_score = max(ks_statistic, psi_score)
                severity = self._determine_drift_severity(drift_score)
                
                if drift_score > self.drift_threshold:
                    drift = DriftDetection(
                        drift_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        intersection_id=intersection_id,
                        drift_type=DriftType.DATA_DRIFT,
                        drift_score=drift_score,
                        severity=severity,
                        description=f"Data drift detected in feature {col}",
                        affected_features=[col],
                        recommended_action=self._get_drift_recommendation(DriftType.DATA_DRIFT),
                        confidence=1 - ks_p_value
                    )
                    drift_detections.append(drift)
                    
            except Exception as e:
                self.logger.warning(f"Error detecting data drift for {col}: {e}")
        
        return drift_detections
    
    def _detect_concept_drift(self, intersection_id: str, 
                             ref_df: pd.DataFrame, 
                             curr_df: pd.DataFrame) -> List[DriftDetection]:
        """Detect concept drift using model performance"""
        drift_detections = []
        
        if intersection_id not in self.reference_models:
            return drift_detections
        
        try:
            model = self.reference_models[intersection_id]
            
            # Prepare features (assuming 'target' column exists)
            if 'target' in ref_df.columns and 'target' in curr_df.columns:
                ref_X = ref_df.drop('target', axis=1).select_dtypes(include=[np.number])
                ref_y = ref_df['target']
                curr_X = curr_df.drop('target', axis=1).select_dtypes(include=[np.number])
                curr_y = curr_df['target']
                
                # Calculate performance on reference and current data
                ref_pred = model.predict(ref_X)
                curr_pred = model.predict(curr_X)
                
                ref_mse = mean_squared_error(ref_y, ref_pred)
                curr_mse = mean_squared_error(curr_y, curr_pred)
                
                # Calculate performance degradation
                performance_degradation = (curr_mse - ref_mse) / ref_mse
                
                if performance_degradation > self.drift_threshold:
                    severity = self._determine_drift_severity(performance_degradation)
                    
                    drift = DriftDetection(
                        drift_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        intersection_id=intersection_id,
                        drift_type=DriftType.CONCEPT_DRIFT,
                        drift_score=performance_degradation,
                        severity=severity,
                        description=f"Concept drift detected: {performance_degradation:.2%} performance degradation",
                        affected_features=[],
                        recommended_action=self._get_drift_recommendation(DriftType.CONCEPT_DRIFT),
                        confidence=min(1.0, performance_degradation * 2)
                    )
                    drift_detections.append(drift)
                    
        except Exception as e:
            self.logger.warning(f"Error detecting concept drift: {e}")
        
        return drift_detections
    
    def _detect_performance_drift(self, intersection_id: str, 
                                 ref_df: pd.DataFrame, 
                                 curr_df: pd.DataFrame) -> List[DriftDetection]:
        """Detect performance drift using key metrics"""
        drift_detections = []
        
        # Key performance metrics to monitor
        performance_metrics = ['wait_time', 'throughput', 'queue_length', 'delay']
        
        for metric in performance_metrics:
            if metric in ref_df.columns and metric in curr_df.columns:
                try:
                    ref_values = ref_df[metric].dropna()
                    curr_values = curr_df[metric].dropna()
                    
                    if len(ref_values) < 10 or len(curr_values) < 10:
                        continue
                    
                    # Calculate performance change
                    ref_mean = ref_values.mean()
                    curr_mean = curr_values.mean()
                    
                    if ref_mean != 0:
                        performance_change = abs(curr_mean - ref_mean) / ref_mean
                        
                        if performance_change > self.drift_threshold:
                            severity = self._determine_drift_severity(performance_change)
                            
                            drift = DriftDetection(
                                drift_id=str(uuid.uuid4()),
                                timestamp=datetime.now(),
                                intersection_id=intersection_id,
                                drift_type=DriftType.PERFORMANCE_DRIFT,
                                drift_score=performance_change,
                                severity=severity,
                                description=f"Performance drift in {metric}: {performance_change:.2%} change",
                                affected_features=[metric],
                                recommended_action=self._get_drift_recommendation(DriftType.PERFORMANCE_DRIFT),
                                confidence=min(1.0, performance_change)
                            )
                            drift_detections.append(drift)
                            
                except Exception as e:
                    self.logger.warning(f"Error detecting performance drift for {metric}: {e}")
        
        return drift_detections
    
    def _detect_feature_drift(self, intersection_id: str, 
                             ref_df: pd.DataFrame, 
                             curr_df: pd.DataFrame) -> List[DriftDetection]:
        """Detect feature drift using feature distributions"""
        drift_detections = []
        
        # Get common features
        common_features = set(ref_df.columns) & set(curr_df.columns)
        
        for feature in common_features:
            try:
                ref_values = ref_df[feature].dropna()
                curr_values = curr_df[feature].dropna()
                
                if len(ref_values) < 10 or len(curr_values) < 10:
                    continue
                
                # Calculate distribution statistics
                ref_mean = ref_values.mean()
                ref_std = ref_values.std()
                curr_mean = curr_values.mean()
                curr_std = curr_values.std()
                
                # Calculate drift score based on mean and std changes
                mean_change = abs(curr_mean - ref_mean) / (ref_std + 1e-8)
                std_change = abs(curr_std - ref_std) / (ref_std + 1e-8)
                
                drift_score = max(mean_change, std_change)
                
                if drift_score > self.drift_threshold:
                    severity = self._determine_drift_severity(drift_score)
                    
                    drift = DriftDetection(
                        drift_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        intersection_id=intersection_id,
                        drift_type=DriftType.FEATURE_DRIFT,
                        drift_score=drift_score,
                        severity=severity,
                        description=f"Feature drift in {feature}: {drift_score:.2f} drift score",
                        affected_features=[feature],
                        recommended_action=self._get_drift_recommendation(DriftType.FEATURE_DRIFT),
                        confidence=min(1.0, drift_score)
                    )
                    drift_detections.append(drift)
                    
            except Exception as e:
                self.logger.warning(f"Error detecting feature drift for {feature}: {e}")
        
        return drift_detections
    
    def _calculate_psi(self, ref_values: pd.Series, curr_values: pd.Series) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins based on reference data
            bins = np.percentile(ref_values, [0, 20, 40, 60, 80, 100])
            bins[0] = -np.inf
            bins[-1] = np.inf
            
            # Calculate distributions
            ref_dist = np.histogram(ref_values, bins=bins)[0] / len(ref_values)
            curr_dist = np.histogram(curr_values, bins=bins)[0] / len(curr_values)
            
            # Calculate PSI
            psi = 0
            for i in range(len(ref_dist)):
                if ref_dist[i] > 0 and curr_dist[i] > 0:
                    psi += (curr_dist[i] - ref_dist[i]) * np.log(curr_dist[i] / ref_dist[i])
            
            return psi
            
        except Exception:
            return 0.0
    
    def _determine_drift_severity(self, drift_score: float) -> AlertLevel:
        """Determine drift severity based on score"""
        if drift_score > 0.5:
            return AlertLevel.EMERGENCY
        elif drift_score > 0.3:
            return AlertLevel.CRITICAL
        elif drift_score > 0.1:
            return AlertLevel.WARNING
        else:
            return AlertLevel.INFO
    
    def _get_drift_recommendation(self, drift_type: DriftType) -> str:
        """Get recommendation for drift type"""
        recommendations = {
            DriftType.DATA_DRIFT: "Retrain model with new data distribution",
            DriftType.CONCEPT_DRIFT: "Update model architecture or retrain completely",
            DriftType.PERFORMANCE_DRIFT: "Investigate performance degradation causes",
            DriftType.FEATURE_DRIFT: "Review feature engineering and data preprocessing"
        }
        return recommendations.get(drift_type, "Investigate and retrain model")


class FeatureImportanceAnalyzer:
    """Feature importance analysis and interpretability"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Feature importance storage
        self.feature_importances = {}
        self.importance_history = {}
        
        # Analysis parameters
        self.stability_window = self.config.get('stability_window', 100)
        self.min_importance_threshold = self.config.get('min_importance_threshold', 0.01)
    
    def analyze_feature_importance(self, intersection_id: str, 
                                 model: Any, 
                                 X: pd.DataFrame, 
                                 y: pd.Series) -> List[FeatureImportance]:
        """Analyze feature importance for model"""
        try:
            # Get model-specific feature importance
            if hasattr(model, 'feature_importances_'):
                model_importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                model_importance = np.abs(model.coef_)
            else:
                # Use permutation importance as fallback
                perm_importance = permutation_importance(
                    model, X, y, n_repeats=10, random_state=42
                )
                model_importance = perm_importance.importances_mean
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model, X, y, n_repeats=10, random_state=42
            )
            
            # Calculate SHAP values
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = np.array(shap_values).mean(axis=0)
                shap_importance = np.abs(shap_values).mean(axis=0)
            except:
                shap_importance = np.zeros(len(X.columns))
            
            # Create feature importance results
            feature_importances = []
            for i, feature_name in enumerate(X.columns):
                if i < len(model_importance):
                    importance_score = model_importance[i]
                    perm_score = perm_importance.importances_mean[i]
                    shap_score = shap_importance[i] if i < len(shap_importance) else 0
                    
                    # Calculate stability score
                    stability_score = self._calculate_stability_score(
                        intersection_id, feature_name, importance_score
                    )
                    
                    # Create feature importance result
                    feature_importance = FeatureImportance(
                        feature_name=feature_name,
                        importance_score=importance_score,
                        importance_rank=0,  # Will be set after sorting
                        permutation_importance=perm_score,
                        shap_importance=shap_score,
                        stability_score=stability_score,
                        description=self._generate_feature_description(
                            feature_name, importance_score, stability_score
                        )
                    )
                    
                    feature_importances.append(feature_importance)
            
            # Sort by importance and set ranks
            feature_importances.sort(key=lambda x: x.importance_score, reverse=True)
            for i, fi in enumerate(feature_importances):
                fi.importance_rank = i + 1
            
            # Store importance history
            self._store_importance_history(intersection_id, feature_importances)
            
            return feature_importances
            
        except Exception as e:
            self.logger.error(f"Error analyzing feature importance for {intersection_id}: {e}")
            return []
    
    def _calculate_stability_score(self, intersection_id: str, 
                                 feature_name: str, 
                                 current_importance: float) -> float:
        """Calculate stability score for feature importance"""
        if intersection_id not in self.importance_history:
            return 1.0
        
        feature_history = self.importance_history[intersection_id].get(feature_name, [])
        
        if len(feature_history) < 2:
            return 1.0
        
        # Calculate coefficient of variation
        recent_importances = feature_history[-self.stability_window:]
        if len(recent_importances) < 2:
            return 1.0
        
        mean_importance = np.mean(recent_importances)
        std_importance = np.std(recent_importances)
        
        if mean_importance == 0:
            return 1.0
        
        cv = std_importance / mean_importance
        stability_score = max(0.0, 1.0 - cv)
        
        return stability_score
    
    def _store_importance_history(self, intersection_id: str, 
                                 feature_importances: List[FeatureImportance]):
        """Store feature importance history"""
        if intersection_id not in self.importance_history:
            self.importance_history[intersection_id] = {}
        
        for fi in feature_importances:
            if fi.feature_name not in self.importance_history[intersection_id]:
                self.importance_history[intersection_id][fi.feature_name] = []
            
            self.importance_history[intersection_id][fi.feature_name].append(fi.importance_score)
            
            # Keep only recent history
            if len(self.importance_history[intersection_id][fi.feature_name]) > self.stability_window:
                self.importance_history[intersection_id][fi.feature_name] = \
                    self.importance_history[intersection_id][fi.feature_name][-self.stability_window:]
    
    def _generate_feature_description(self, feature_name: str, 
                                    importance_score: float, 
                                    stability_score: float) -> str:
        """Generate human-readable feature description"""
        importance_level = "high" if importance_score > 0.1 else "medium" if importance_score > 0.05 else "low"
        stability_level = "stable" if stability_score > 0.8 else "moderate" if stability_score > 0.5 else "unstable"
        
        return f"{feature_name} has {importance_level} importance ({importance_score:.3f}) and is {stability_level} ({stability_score:.3f})"
    
    def get_feature_importance_summary(self, intersection_id: str) -> Dict[str, Any]:
        """Get feature importance summary for intersection"""
        if intersection_id not in self.feature_importances:
            return {}
        
        importances = self.feature_importances[intersection_id]
        
        return {
            'total_features': len(importances),
            'high_importance_features': len([f for f in importances if f.importance_score > 0.1]),
            'stable_features': len([f for f in importances if f.stability_score > 0.8]),
            'top_features': [f.feature_name for f in importances[:5]],
            'least_important_features': [f.feature_name for f in importances[-5:]]
        }


class ExplainableAI:
    """Explainable AI components for model interpretability"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Explanation storage
        self.explanations = {}
        self.explanation_history = deque(maxlen=1000)
        
        # SHAP and LIME explainers
        self.shap_explainers = {}
        self.lime_explainers = {}
    
    def create_model_explanation(self, intersection_id: str, 
                               model: Any, 
                               X: pd.DataFrame, 
                               prediction: float,
                               instance_idx: int = 0) -> ModelExplanation:
        """Create comprehensive model explanation"""
        try:
            # Get SHAP values
            shap_values = self._get_shap_values(intersection_id, model, X, instance_idx)
            
            # Get LIME explanation
            lime_explanation = self._get_lime_explanation(intersection_id, model, X, instance_idx)
            
            # Calculate feature contributions
            feature_contributions = self._calculate_feature_contributions(
                X.iloc[instance_idx], shap_values
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(feature_contributions, prediction)
            
            # Create explanation
            explanation = ModelExplanation(
                explanation_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                intersection_id=intersection_id,
                prediction=prediction,
                feature_contributions=feature_contributions,
                shap_values=shap_values,
                lime_explanation=lime_explanation,
                confidence=self._calculate_explanation_confidence(shap_values),
                reasoning=reasoning
            )
            
            # Store explanation
            self.explanation_history.append(explanation)
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error creating model explanation: {e}")
            return None
    
    def _get_shap_values(self, intersection_id: str, model: Any, 
                        X: pd.DataFrame, instance_idx: int) -> Dict[str, float]:
        """Get SHAP values for model explanation"""
        try:
            if intersection_id not in self.shap_explainers:
                # Create SHAP explainer
                if hasattr(model, 'predict_proba'):
                    explainer = shap.TreeExplainer(model)
                else:
                    explainer = shap.Explainer(model)
                self.shap_explainers[intersection_id] = explainer
            
            explainer = self.shap_explainers[intersection_id]
            shap_values = explainer.shap_values(X.iloc[[instance_idx]])
            
            # Convert to dictionary
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            if len(shap_values.shape) > 1:
                shap_values = shap_values[0]
            
            return dict(zip(X.columns, shap_values))
            
        except Exception as e:
            self.logger.warning(f"Error getting SHAP values: {e}")
            return {col: 0.0 for col in X.columns}
    
    def _get_lime_explanation(self, intersection_id: str, model: Any, 
                             X: pd.DataFrame, instance_idx: int) -> Dict[str, Any]:
        """Get LIME explanation for model prediction"""
        try:
            if intersection_id not in self.lime_explainers:
                # Create LIME explainer
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X.values,
                    feature_names=X.columns.tolist(),
                    class_names=['prediction'],
                    mode='regression'
                )
                self.lime_explainers[intersection_id] = explainer
            
            explainer = self.lime_explainers[intersection_id]
            explanation = explainer.explain_instance(
                X.iloc[instance_idx].values,
                model.predict,
                num_features=len(X.columns)
            )
            
            # Convert to dictionary
            lime_dict = {}
            for feature, weight in explanation.as_list():
                lime_dict[feature] = weight
            
            return lime_dict
            
        except Exception as e:
            self.logger.warning(f"Error getting LIME explanation: {e}")
            return {}
    
    def _calculate_feature_contributions(self, instance: pd.Series, 
                                       shap_values: Dict[str, float]) -> Dict[str, float]:
        """Calculate feature contributions to prediction"""
        contributions = {}
        
        for feature in instance.index:
            if feature in shap_values:
                contributions[feature] = shap_values[feature]
            else:
                contributions[feature] = 0.0
        
        return contributions
    
    def _generate_reasoning(self, feature_contributions: Dict[str, float], 
                          prediction: float) -> str:
        """Generate human-readable reasoning for prediction"""
        # Sort features by absolute contribution
        sorted_features = sorted(
            feature_contributions.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        reasoning_parts = []
        
        # Add prediction context
        if prediction > 0.7:
            reasoning_parts.append("The model predicts high optimization potential")
        elif prediction > 0.4:
            reasoning_parts.append("The model predicts moderate optimization potential")
        else:
            reasoning_parts.append("The model predicts low optimization potential")
        
        # Add top contributing features
        top_features = sorted_features[:3]
        if top_features:
            feature_descriptions = []
            for feature, contribution in top_features:
                if abs(contribution) > 0.1:
                    direction = "increases" if contribution > 0 else "decreases"
                    feature_descriptions.append(f"{feature} {direction} the prediction")
            
            if feature_descriptions:
                reasoning_parts.append(f"Key factors: {', '.join(feature_descriptions)}")
        
        return ". ".join(reasoning_parts) + "."
    
    def _calculate_explanation_confidence(self, shap_values: Dict[str, float]) -> float:
        """Calculate confidence in explanation based on SHAP values"""
        if not shap_values:
            return 0.0
        
        # Calculate confidence based on magnitude and consistency of SHAP values
        values = list(shap_values.values())
        mean_abs_value = np.mean(np.abs(values))
        std_value = np.std(values)
        
        # Higher confidence for larger, more consistent contributions
        confidence = min(1.0, mean_abs_value * (1 - std_value / (mean_abs_value + 1e-8)))
        
        return confidence
    
    def create_explanation_visualization(self, explanation: ModelExplanation) -> go.Figure:
        """Create visualization for model explanation"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Feature Contributions', 'SHAP Values', 
                          'LIME Explanation', 'Prediction Summary'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # Feature contributions
        features = list(explanation.feature_contributions.keys())
        contributions = list(explanation.feature_contributions.values())
        
        fig.add_trace(
            go.Bar(
                x=features,
                y=contributions,
                name='Contributions',
                marker_color=['red' if x < 0 else 'blue' for x in contributions]
            ),
            row=1, col=1
        )
        
        # SHAP values
        shap_features = list(explanation.shap_values.keys())
        shap_vals = list(explanation.shap_values.values())
        
        fig.add_trace(
            go.Bar(
                x=shap_features,
                y=shap_vals,
                name='SHAP Values',
                marker_color=['red' if x < 0 else 'green' for x in shap_vals]
            ),
            row=1, col=2
        )
        
        # LIME explanation
        lime_features = list(explanation.lime_explanation.keys())
        lime_vals = list(explanation.lime_explanation.values())
        
        fig.add_trace(
            go.Bar(
                x=lime_features,
                y=lime_vals,
                name='LIME Values',
                marker_color=['red' if x < 0 else 'orange' for x in lime_vals]
            ),
            row=2, col=1
        )
        
        # Prediction summary
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=explanation.prediction,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Prediction"},
                gauge={'axis': {'range': [0, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, 0.3], 'color': "lightgray"},
                                {'range': [0.3, 0.7], 'color': "gray"},
                                {'range': [0.7, 1], 'color': "darkgray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': 0.8}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=f"Model Explanation - {explanation.intersection_id}",
            height=600,
            showlegend=True
        )
        
        return fig


class PerformanceAlertSystem:
    """Performance degradation alerts and retraining triggers"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Alert thresholds
        self.alert_thresholds = {
            'wait_time': {'warning': 0.1, 'critical': 0.2, 'emergency': 0.5},
            'throughput': {'warning': -0.1, 'critical': -0.2, 'emergency': -0.5},
            'queue_length': {'warning': 0.15, 'critical': 0.3, 'emergency': 0.6},
            'delay': {'warning': 0.1, 'critical': 0.25, 'emergency': 0.5}
        }
        
        # Performance history
        self.performance_history = {}
        self.alert_history = deque(maxlen=1000)
        
        # Retraining triggers
        self.retraining_threshold = self.config.get('retraining_threshold', 0.3)
        self.retraining_cooldown = self.config.get('retraining_cooldown', 3600)  # 1 hour
        self.last_retraining = {}
    
    def check_performance_alerts(self, intersection_id: str, 
                               current_metrics: Dict[str, float]) -> List[PerformanceAlert]:
        """Check for performance degradation alerts"""
        alerts = []
        
        # Get historical performance
        if intersection_id not in self.performance_history:
            self.performance_history[intersection_id] = {}
        
        for metric_name, current_value in current_metrics.items():
            if metric_name not in self.alert_thresholds:
                continue
            
            # Calculate performance change
            historical_values = self.performance_history[intersection_id].get(metric_name, [])
            
            if len(historical_values) < 10:  # Need sufficient history
                continue
            
            historical_mean = np.mean(historical_values[-100:])  # Last 100 values
            performance_change = (current_value - historical_mean) / historical_mean
            
            # Check against thresholds
            thresholds = self.alert_thresholds[metric_name]
            alert_level = None
            
            if abs(performance_change) >= thresholds['emergency']:
                alert_level = AlertLevel.EMERGENCY
            elif abs(performance_change) >= thresholds['critical']:
                alert_level = AlertLevel.CRITICAL
            elif abs(performance_change) >= thresholds['warning']:
                alert_level = AlertLevel.WARNING
            
            if alert_level:
                alert = PerformanceAlert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    intersection_id=intersection_id,
                    alert_level=alert_level,
                    metric_name=metric_name,
                    current_value=current_value,
                    threshold_value=historical_mean * (1 + thresholds[alert_level.value]),
                    degradation_percentage=performance_change * 100,
                    description=f"{metric_name} degraded by {performance_change:.1%}",
                    recommended_action=self._get_alert_recommendation(metric_name, alert_level)
                )
                
                alerts.append(alert)
                self.alert_history.append(alert)
        
        # Update performance history
        for metric_name, value in current_metrics.items():
            if metric_name not in self.performance_history[intersection_id]:
                self.performance_history[intersection_id][metric_name] = deque(maxlen=1000)
            
            self.performance_history[intersection_id][metric_name].append(value)
        
        return alerts
    
    def _get_alert_recommendation(self, metric_name: str, alert_level: AlertLevel) -> str:
        """Get recommendation for performance alert"""
        recommendations = {
            'wait_time': {
                AlertLevel.WARNING: "Monitor traffic patterns and adjust timing",
                AlertLevel.CRITICAL: "Immediate timing adjustment required",
                AlertLevel.EMERGENCY: "Emergency traffic management protocols"
            },
            'throughput': {
                AlertLevel.WARNING: "Review signal timing optimization",
                AlertLevel.CRITICAL: "Urgent signal timing review",
                AlertLevel.EMERGENCY: "Emergency signal timing override"
            },
            'queue_length': {
                AlertLevel.WARNING: "Increase green time for congested approaches",
                AlertLevel.CRITICAL: "Immediate queue management required",
                AlertLevel.EMERGENCY: "Emergency queue clearance protocols"
            },
            'delay': {
                AlertLevel.WARNING: "Review and optimize signal coordination",
                AlertLevel.CRITICAL: "Urgent signal coordination review",
                AlertLevel.EMERGENCY: "Emergency signal coordination override"
            }
        }
        
        return recommendations.get(metric_name, {}).get(alert_level, "Investigate and take corrective action")
    
    def should_retrain_model(self, intersection_id: str, 
                           recent_alerts: List[PerformanceAlert]) -> bool:
        """Determine if model should be retrained"""
        # Check cooldown period
        if intersection_id in self.last_retraining:
            time_since_retraining = (
                datetime.now() - self.last_retraining[intersection_id]
            ).total_seconds()
            
            if time_since_retraining < self.retraining_cooldown:
                return False
        
        # Check for critical alerts
        critical_alerts = [a for a in recent_alerts if a.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]]
        
        if len(critical_alerts) >= 3:  # 3 or more critical alerts
            self.last_retraining[intersection_id] = datetime.now()
            return True
        
        # Check for sustained performance degradation
        if intersection_id in self.performance_history:
            for metric_name, values in self.performance_history[intersection_id].items():
                if len(values) >= 50:  # Need sufficient history
                    recent_values = list(values)[-50:]
                    older_values = list(values)[-100:-50] if len(values) >= 100 else list(values)[:-50]
                    
                    if older_values:
                        recent_mean = np.mean(recent_values)
                        older_mean = np.mean(older_values)
                        
                        if older_mean != 0:
                            degradation = (recent_mean - older_mean) / older_mean
                            
                            if abs(degradation) >= self.retraining_threshold:
                                self.last_retraining[intersection_id] = datetime.now()
                                return True
        
        return False


class MLMonitoring:
    """
    Main ML Monitoring & Observability System
    
    Features:
    - Model drift detection for changing traffic patterns
    - Feature importance analysis and interpretability
    - Performance degradation alerts and automatic retraining triggers
    - Detailed logging for ML decision audit trails
    - Explainable AI components for stakeholder presentations
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.drift_detector = ModelDriftDetector(config.get('drift_detector', {}))
        self.feature_analyzer = FeatureImportanceAnalyzer(config.get('feature_analyzer', {}))
        self.explainable_ai = ExplainableAI(config.get('explainable_ai', {}))
        self.alert_system = PerformanceAlertSystem(config.get('alert_system', {}))
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Audit trail
        self.audit_trail = deque(maxlen=10000)
        
        self.logger.info("ML Monitoring system initialized")
    
    def start_monitoring(self):
        """Start ML monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("ML monitoring started")
    
    def stop_monitoring(self):
        """Stop ML monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("ML monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Process monitoring tasks
                self._process_monitoring_tasks()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)
    
    def _process_monitoring_tasks(self):
        """Process all monitoring tasks"""
        # This would integrate with the real-time system
        # For now, we'll implement the monitoring framework
        pass
    
    def log_ml_decision(self, intersection_id: str, 
                       decision: Dict[str, Any], 
                       model_input: Dict[str, Any],
                       model_output: float,
                       confidence: float):
        """Log ML decision for audit trail"""
        audit_entry = {
            'timestamp': datetime.now(),
            'intersection_id': intersection_id,
            'decision': decision,
            'model_input': model_input,
            'model_output': model_output,
            'confidence': confidence,
            'audit_id': str(uuid.uuid4())
        }
        
        self.audit_trail.append(audit_entry)
        
        # Log to file if configured
        if self.config.get('log_decisions', False):
            self.logger.info(f"ML Decision: {audit_entry}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        return {
            'is_monitoring': self.is_monitoring,
            'drift_detections': len(self.drift_detector.drift_history),
            'performance_alerts': len(self.alert_system.alert_history),
            'audit_entries': len(self.audit_trail),
            'feature_analyses': len(self.feature_analyzer.feature_importances),
            'explanations_generated': len(self.explainable_ai.explanation_history)
        }
    
    def get_intersection_monitoring_report(self, intersection_id: str) -> Dict[str, Any]:
        """Get detailed monitoring report for intersection"""
        report = {
            'intersection_id': intersection_id,
            'timestamp': datetime.now(),
            'drift_detections': [],
            'feature_importance': {},
            'performance_alerts': [],
            'recent_decisions': [],
            'recommendations': []
        }
        
        # Get drift detections
        recent_drifts = [
            drift for drift in self.drift_detector.drift_history
            if drift.intersection_id == intersection_id
        ]
        report['drift_detections'] = [asdict(drift) for drift in recent_drifts[-10:]]
        
        # Get feature importance
        if intersection_id in self.feature_analyzer.feature_importances:
            report['feature_importance'] = self.feature_analyzer.get_feature_importance_summary(intersection_id)
        
        # Get performance alerts
        recent_alerts = [
            alert for alert in self.alert_system.alert_history
            if alert.intersection_id == intersection_id
        ]
        report['performance_alerts'] = [asdict(alert) for alert in recent_alerts[-10:]]
        
        # Get recent decisions
        recent_decisions = [
            entry for entry in self.audit_trail
            if entry['intersection_id'] == intersection_id
        ]
        report['recent_decisions'] = recent_decisions[-10:]
        
        # Generate recommendations
        report['recommendations'] = self._generate_monitoring_recommendations(
            intersection_id, recent_drifts, recent_alerts
        )
        
        return report
    
    def _generate_monitoring_recommendations(self, intersection_id: str, 
                                           drifts: List[DriftDetection],
                                           alerts: List[PerformanceAlert]) -> List[str]:
        """Generate monitoring recommendations"""
        recommendations = []
        
        # Check for drift issues
        critical_drifts = [d for d in drifts if d.severity in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]]
        if critical_drifts:
            recommendations.append(f"Critical drift detected: {len(critical_drifts)} issues require immediate attention")
        
        # Check for performance issues
        critical_alerts = [a for a in alerts if a.alert_level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]]
        if critical_alerts:
            recommendations.append(f"Performance degradation: {len(critical_alerts)} critical alerts")
        
        # Check for retraining needs
        if self.alert_system.should_retrain_model(intersection_id, alerts):
            recommendations.append("Model retraining recommended due to performance degradation")
        
        # General recommendations
        if not drifts and not alerts:
            recommendations.append("System performing normally - continue monitoring")
        
        return recommendations
