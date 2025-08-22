from pydantic import BaseModel, Field


class AnemiaRequest(BaseModel):
    gender: int = Field(..., description="Gender (0 for male, 1 for female)")
    hemoglobin: float = Field(..., description="Hemoglobin level (g/dL)", gt=0)
    mch: float = Field(..., description="Mean Corpuscular Hemoglobin (pg)", gt=0)
    mchc: float = Field(..., description="Mean Corpuscular Hemoglobin Concentration (g/dL)", gt=0)
    mcv: float = Field(..., description="Mean Corpuscular Volume (fL)", gt=0)

class AnemiaResponse(BaseModel):
    prediction: int = Field(..., description="Classification result (0: No Anemia, 1: Anemia)")
    probability: float = Field(..., description="Probability of having anemia (0.0-1.0)")
    confidence: str = Field(..., description="Confidence level (Low/Medium/High)")
    model_used: str = Field(..., description="Name of the ML model used for classification")
