from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    app_name: str = "Detect Anemia Service API"
    app_version: str = "1.0.0"
    debug: bool = False

    # Model configuration
    model_name: str = "random_forest"  # Default to random forest
    model_path: str = "models"

    # Available models
    available_models: list = ["logistic", "random_forest", "gradient_boosting", "svm", "knn", "decision_tree", "naive_bayes"]

    @property
    def model_file_path(self) -> str:
        return os.path.join(self.model_path, f"{self.model_name}_model.joblib")

    @property
    def scaler_file_path(self) -> str:
        return os.path.join(self.model_path, f"{self.model_name}_scaler.joblib")

settings = Settings()