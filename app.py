from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from datetime import datetime

from schemas.request_response import TextRequest, PredictionResponse, HealthResponse
from src.predictor import classifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Text Classification API",
    description="API for text classification using XGBoost model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("🚀 Starting Text Classification API...")
    try:
        # Models are already loaded in the classifier instance
        if classifier.health_check():
            logger.info("✅ Models loaded successfully!")
        else:
            logger.error("❌ Models failed to load!")
            raise Exception("Models not loaded properly")
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise e

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Text Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        is_healthy = classifier.health_check()
        if is_healthy:
            return HealthResponse(
                status="healthy",
                message="API is running successfully"
            )
        else:
            raise HTTPException(status_code=503, detail="Service unhealthy")
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_text(request: TextRequest):
    """
    Classify input text
    
    - **text**: The text to classify (required)
    
    Returns the prediction with confidence score and text statistics.
    """
    try:
        logger.info(f"📥 Received prediction request for text length: {len(request.text)}")
        
        # Make prediction
        result = classifier.predict(request.text)
        
        # Create response
        response = PredictionResponse(
            prediction=result['prediction'],
            confidence=result['confidence'],
            text_length=result['text_length'],
            num_sentences=result['num_sentences']
        )
        
        logger.info(f"📤 Prediction completed: {result['prediction']} (confidence: {result['confidence']:.3f})")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )