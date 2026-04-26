from pydantic import BaseModel
from typing import Dict, Optional


class PredictionResponse(BaseModel):
    death_risk: float


class ExplanationResponse(BaseModel):
    death_risk: float
    explanation: Dict[str, float]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None