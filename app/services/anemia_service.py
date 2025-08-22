import warnings
import numpy as np
from typing import Any, Optional
from sklearn.preprocessing import StandardScaler

from app.models.anemia_model import AnemiaRequest, AnemiaResponse
from app.core.model_loader import ModelLoader
from app.core.config import settings

warnings.filterwarnings('ignore')


class AnemiaClassificationService:
    """Main service for anemia classification with multiple model support"""

    def __init__(self):
        self._model: Optional[Any] = None
        self._scaler: Optional[StandardScaler] = None
        self._model_name: str = settings.model_name
        self._load_model()

    def _load_model(self) -> None:
        """Load the ML model and scaler based on configuration"""
        try:
            self._model, self._scaler = ModelLoader.load_model_and_scaler(self._model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize classification service: {str(e)}")

    def classify(self, request: AnemiaRequest) -> AnemiaResponse:
        """
        Classify anemia based on blood test parameters.

        Args:
            request: AnemiaRequest with blood test values

        Returns:
            AnemiaResponse with classification results
        """
        if self._model is None:
            raise RuntimeError("Model not loaded")

        features = self._prepare_features(request)

        classification = self._model.predict([features])[0]
        probability = self._model.predict_proba([features])[0][1]

        confidence = self._get_confidence_level(probability)

        return AnemiaResponse(
            prediction=int(classification),
            probability=float(probability),
            confidence=confidence,
            model_used=self._model_name
        )

    def _prepare_features(self, request: AnemiaRequest) -> np.ndarray:
        """
        Prepare features for classification, applying scaling if needed.

        Args:
            request: AnemiaRequest with input values

        Returns:
            Processed feature array
        """
        features = np.array([
            request.gender,
            request.hemoglobin,
            request.mch,
            request.mchc,
            request.mcv
        ])

        if self._model_name in ModelLoader.models_need_scaling() and self._scaler is not None:
            features = self._scaler.transform([features])[0]

        return features

    def _get_confidence_level(self, probability: float) -> str:
        """
        Determine confidence level based on classification probability.

        Args:
            probability: Classification probability

        Returns:
            Confidence level string
        """
        if probability < 0.3 or probability > 0.7:
            return "High"
        elif probability < 0.4 or probability > 0.6:
            return "Medium"
        else:
            return "Low"

    def get_feature_importance(self) -> Optional[dict]:
        """
        Get feature importance if available for the current model.

        Returns:
            Dictionary with feature names and their importance values
        """
        if not hasattr(self._model, 'feature_importances_'):
            return None

        feature_names = ['gender', 'hemoglobin', 'mch', 'mchc', 'mcv']
        importance_values = self._model.feature_importances_

        return dict(zip(feature_names, importance_values.tolist()))

    def get_model_info(self) -> dict:
        """
        Get information about the currently loaded model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self._model_name,
            "model_type": type(self._model).__name__ if self._model else None,
            "has_scaler": self._scaler is not None,
            "available_models": settings.available_models
        }
