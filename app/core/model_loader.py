import joblib
import os
from typing import Any, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from app.core.config import settings


class ModelLoader:
    """Responsible for loading ML models and scalers from disk"""

    @staticmethod
    def load_model_and_scaler(model_name: str) -> Tuple[Any, Optional[StandardScaler]]:
        """
        Load model and scaler for the specified model name.

        Args:
            model_name: Name of the model to load

        Returns:
            Tuple of (model, scaler) where scaler can be None for models that don't need scaling

        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        model_path = os.path.join(settings.model_path, f"{model_name}_model.joblib")
        scaler_path = os.path.join(settings.model_path, f"{model_name}_scaler.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        try:
            model = joblib.load(model_path)

            scaler = None
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)

            return model, scaler

        except Exception as e:
            raise Exception(f"Failed to load model {model_name}: {str(e)}")

    @staticmethod
    def models_need_scaling() -> list:
        """Return list of model names that require feature scaling"""
        return ["logistic", "svm", "knn"]
