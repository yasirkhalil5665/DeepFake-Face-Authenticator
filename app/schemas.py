from pydantic import BaseModel


class PredictionResponse(BaseModel):
    label: str            # "real" or "fake"
    confidence: float     # probability of the predicted class, 0-1
    probabilities: dict   # {"fake": 0.03, "real": 0.97}


