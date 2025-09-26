"""
Machine Learning Integration for AI Slop Detection.

This module provides:
- Anomaly detection using One-Class SVM
- Learned combiners for feature integration
- Ensemble methods for improved accuracy
- Domain-specific model adaptation
"""

import logging
import pickle
import os
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from sklearn.svm import OneClassSVM
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    OneClassSVM = None
    IsolationForest = None
    RandomForestClassifier = None
    LogisticRegression = None
    StandardScaler = None

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Anomaly detection for identifying novel AI generation patterns."""

    def __init__(self, model_type: str = "one_class_svm", contamination: float = 0.1):
        """Initialize anomaly detector."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for anomaly detection")
        
        self.model_type = model_type
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Initialize model based on type
        if model_type == "one_class_svm":
            self.model = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
        elif model_type == "isolation_forest":
            self.model = IsolationForest(contamination=contamination, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X: np.ndarray) -> None:
        """Fit the anomaly detection model."""
        if X is None or len(X) == 0:
            raise ValueError("Training data cannot be empty")
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        logger.info(f"Anomaly detector ({self.model_type}) fitted on {len(X)} samples")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies in the data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        if self.model_type == "one_class_svm":
            # OneClassSVM returns 1 for normal, -1 for anomaly
            predictions = self.model.predict(X_scaled)
            # Convert to 0 for normal, 1 for anomaly
            return (predictions == -1).astype(int)
        else:
            # IsolationForest returns -1 for anomaly, 1 for normal
            predictions = self.model.predict(X_scaled)
            # Convert to 0 for normal, 1 for anomaly
            return (predictions == -1).astype(int)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'contamination': self.contamination,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Anomaly detector saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.contamination = model_data['contamination']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Anomaly detector loaded from {filepath}")


class LearnedCombiner:
    """Learned combiner for integrating multiple slop detection features."""

    def __init__(self, model_type: str = "logistic_regression", domain: str = "general"):
        """Initialize learned combiner."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for learned combiners")
        
        self.model_type = model_type
        self.domain = domain
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = []
        
        # Initialize model based on type
        if model_type == "logistic_regression":
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _extract_features(self, feature_dict: Dict[str, Any]) -> np.ndarray:
        """Extract numerical features from feature dictionary."""
        features = []
        feature_names = []
        
        # Define feature extraction mapping
        feature_mapping = {
            'density': ['perplexity', 'idea_density', 'semantic_density', 'conceptual_density'],
            'repetition': ['compression_ratio', 'ngram_repetition'],
            'coherence': ['entity_continuity', 'embedding_drift'],
            'templated': ['pos_diversity', 'boilerplate_hits'],
            'tone': ['hedging_score', 'sycophancy_score'],
            'verbosity': ['words_per_sentence', 'listiness'],
            'complexity': ['flesch_kincaid', 'gunning_fog'],
            'subjectivity': ['subjectivity_score'],
            'fluency': ['grammar_errors', 'perplexity_spikes'],
            'relevance': ['mean_similarity', 'low_similarity_fraction'],
        }
        
        for category, subfeatures in feature_mapping.items():
            for subfeature in subfeatures:
                value = feature_dict.get(subfeature, 0.0)
                if isinstance(value, (int, float)):
                    features.append(float(value))
                    feature_names.append(f"{category}_{subfeature}")
                else:
                    features.append(0.0)
                    feature_names.append(f"{category}_{subfeature}")
        
        self.feature_names = feature_names
        return np.array(features).reshape(1, -1)

    def fit(self, feature_dicts: List[Dict[str, Any]], labels: List[int]) -> None:
        """Fit the learned combiner."""
        if len(feature_dicts) != len(labels):
            raise ValueError("Number of feature dictionaries must match number of labels")
        
        # Extract features
        X = np.vstack([self._extract_features(fd) for fd in feature_dicts])
        y = np.array(labels)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1] if hasattr(self.model, 'predict_proba') else None
        
        # Log performance
        logger.info(f"Learned combiner ({self.model_type}) fitted on {len(X_train)} samples")
        logger.info(f"Validation accuracy: {self.model.score(X_val, y_val):.3f}")
        
        if y_pred_proba is not None:
            try:
                auc_score = roc_auc_score(y_val, y_pred_proba)
                logger.info(f"Validation AUC: {auc_score:.3f}")
            except ValueError:
                logger.warning("Could not calculate AUC score")
        
        self.is_fitted = True

    def predict(self, feature_dict: Dict[str, Any]) -> Tuple[int, float]:
        """Predict slop score and probability."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._extract_features(feature_dict)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(X_scaled)[0, 1]
        else:
            probability = float(prediction)
        
        return prediction, probability

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return {}
        
        return dict(zip(self.feature_names, importances))

    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'domain': self.domain,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Learned combiner saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.domain = model_data['domain']
        self.feature_names = model_data['feature_names']
        self.is_fitted = model_data['is_fitted']
        
        logger.info(f"Learned combiner loaded from {filepath}")


class EnsembleDetector:
    """Ensemble detector combining multiple approaches."""

    def __init__(self, domain: str = "general"):
        """Initialize ensemble detector."""
        self.domain = domain
        self.anomaly_detector = AnomalyDetector()
        self.learned_combiner = LearnedCombiner(domain=domain)
        self.is_fitted = False

    def fit(
        self,
        feature_dicts: List[Dict[str, Any]],
        labels: List[int],
        anomaly_data: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Fit the ensemble detector."""
        # Fit learned combiner
        self.learned_combiner.fit(feature_dicts, labels)
        
        # Fit anomaly detector if data provided
        if anomaly_data:
            anomaly_features = np.vstack([
                self.learned_combiner._extract_features(fd) for fd in anomaly_data
            ])
            self.anomaly_detector.fit(anomaly_features)
        
        self.is_fitted = True
        logger.info(f"Ensemble detector fitted for domain: {self.domain}")

    def predict(self, feature_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Predict using ensemble approach."""
        if not self.is_fitted:
            raise ValueError("Ensemble detector must be fitted before prediction")
        
        # Get learned combiner prediction
        combiner_pred, combiner_prob = self.learned_combiner.predict(feature_dict)
        
        # Get anomaly detection result
        anomaly_features = self.learned_combiner._extract_features(feature_dict)
        anomaly_pred = self.anomaly_detector.predict(anomaly_features)[0]
        anomaly_score = self.anomaly_detector.decision_function(anomaly_features)[0]
        
        # Combine predictions
        ensemble_score = (combiner_prob + (1 - anomaly_score / 2)) / 2
        
        return {
            'ensemble_score': ensemble_score,
            'combiner_prediction': combiner_pred,
            'combiner_probability': combiner_prob,
            'anomaly_prediction': anomaly_pred,
            'anomaly_score': anomaly_score,
            'confidence': abs(combiner_prob - 0.5) * 2,  # Higher confidence when prediction is more certain
        }

    def save_models(self, base_path: str) -> None:
        """Save all models."""
        self.learned_combiner.save_model(f"{base_path}_combiner.pkl")
        self.anomaly_detector.save_model(f"{base_path}_anomaly.pkl")
        logger.info(f"Ensemble models saved to {base_path}")

    def load_models(self, base_path: str) -> None:
        """Load all models."""
        self.learned_combiner.load_model(f"{base_path}_combiner.pkl")
        self.anomaly_detector.load_model(f"{base_path}_anomaly.pkl")
        self.is_fitted = True
        logger.info(f"Ensemble models loaded from {base_path}")
