import joblib
import os
from pathlib import Path

class ModelLoader:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        
    def load_models(self):
        """Load all required models and components"""
        try:
            # Load XGBoost model
            model_path = self.model_dir / "xgb_model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model = joblib.load(model_path)
            
            # Load TF-IDF vectorizer
            vectorizer_path = self.model_dir / "tfidf_vectorizer.pkl"
            if not vectorizer_path.exists():
                raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
            self.vectorizer = joblib.load(vectorizer_path)
            
            # Load label encoder
            encoder_path = self.model_dir / "label_encoder.pkl"
            if not encoder_path.exists():
                raise FileNotFoundError(f"Label encoder file not found: {encoder_path}")
            self.label_encoder = joblib.load(encoder_path)
            
            print("✅ All models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {str(e)}")
            return False
    
    def get_models(self):
        """Return loaded models"""
        if self.model is None or self.vectorizer is None or self.label_encoder is None:
            raise ValueError("Models not loaded. Call load_models() first.")
        
        return {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_encoder': self.label_encoder
        }

# Global instance
model_loader = ModelLoader()