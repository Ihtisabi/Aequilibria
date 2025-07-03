from pydantic import BaseModel, Field
from typing import Optional

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to classify")
    
    class Config:
        schema_extra = {
            "example": {
                "text": "This is a sample text that needs to be classified."
            }
        }

class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Predicted class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")
    text_length: int = Field(..., description="Number of characters in input text")
    num_sentences: int = Field(..., description="Number of sentences in input text")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": "true",
                "confidence": 0.85,
                "text_length": 45,
                "num_sentences": 1
            }
        }

class HealthResponse(BaseModel):
    status: str = Field(..., description="API health status")
    message: str = Field(..., description="Health check message")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "message": "API is running successfully"
            }
        }