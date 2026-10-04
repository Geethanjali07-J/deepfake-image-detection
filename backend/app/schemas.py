from pydantic import BaseModel


class PredictionResponse(BaseModel):
    prediction: str
    score: float
    confidence: float
