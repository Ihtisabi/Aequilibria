import numpy as np
from src.model_loader import model_loader
from src.preprocessor import preprocessor

class TextClassifier:
    def __init__(self):
        self.models = None
        self.load_models()
    
    def load_models(self):
        """Load all required models"""
        if not model_loader.load_models():
            raise Exception("Failed to load models")
        self.models = model_loader.get_models()
    
    def predict(self, text: str) -> dict:
        """Make prediction on input text"""
        if self.models is None:
            raise ValueError("Models not loaded")
        
        try:
            # Prepare features
            features, num_features = preprocessor.prepare_features(
                text, self.models['vectorizer']
            )
            
            # Make prediction
            prediction_encoded = self.models['model'].predict(features)[0]
            prediction_proba = self.models['model'].predict_proba(features)[0]
            
            # Decode prediction
            prediction_label = self.models['label_encoder'].inverse_transform([prediction_encoded])[0]
            
            # Get confidence score
            confidence = float(np.max(prediction_proba))
            
            return {
                'prediction': prediction_label,
                'confidence': confidence,
                'text_length': num_features['num_of_characters'],
                'num_sentences': num_features['num_of_sentences']
            }
            
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")
    
    def health_check(self) -> bool:
        """Check if all models are loaded and working"""
        try:
            if self.models is None:
                return False
            
            # Test with sample text
            sample_text = "This is a test message."
            result = self.predict(sample_text)
            
            return True
        except:
            return False

# Global instance
classifier = TextClassifier()