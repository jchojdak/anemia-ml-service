from fastapi import APIRouter, HTTPException
from app.services.anemia_service import AnemiaClassificationService
from app.models.anemia_model import AnemiaRequest, AnemiaResponse


router = APIRouter()

try:
    anemia_service = AnemiaClassificationService()
except Exception as e:
    anemia_service = None
    print(f"Warning: Failed to initialize anemia service: {e}")

@router.post("/anemia", response_model=AnemiaResponse)
def classify_anemia(request: AnemiaRequest):
    """
    Classify anemia based on blood test parameters.

    Parameters:
    - gender: 0 for male, 1 for female
    - hemoglobin: Hemoglobin level in g/dL
    - mch: Mean Corpuscular Hemoglobin in pg
    - mchc: Mean Corpuscular Hemoglobin Concentration in g/dL
    - mcv: Mean Corpuscular Volume in fL

    Returns:
    - prediction: 0 (No Anemia) or 1 (Anemia) - Classification result
    - probability: Probability of having anemia (0.0-1.0)
    - confidence: Confidence level (Low/Medium/High)
    - model_used: Name of the ML model used
    """
    if anemia_service is None:
        raise HTTPException(status_code=503, detail="Anemia classification service unavailable")

    try:
        return anemia_service.classify(request)
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(ex)}")

@router.get("/anemia/feature-importance")
def get_feature_importance():
    """
    Get feature importance from the anemia classification model.
    """
    if anemia_service is None:
        raise HTTPException(status_code=503, detail="Anemia classification service unavailable")

    try:
        importance = anemia_service.get_feature_importance()
        if importance is None:
            raise HTTPException(status_code=404, detail="Feature importance not available for this model type")
        return {"feature_importance": importance}
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error retrieving feature importance: {str(ex)}")

@router.get("/anemia/model-info")
def get_model_info():
    """
    Get information about the currently loaded anemia classification model.
    """
    if anemia_service is None:
        raise HTTPException(status_code=503, detail="Anemia classification service unavailable")

    try:
        return anemia_service.get_model_info()
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(ex)}")

@router.get("/health")
def health_check():
    """
    Health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "anemia-classification",
        "models_available": anemia_service is not None
    }
