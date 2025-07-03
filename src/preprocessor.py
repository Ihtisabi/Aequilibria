import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from scipy.sparse import hstack
import numpy as np
import os

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self._download_nltk_data()
    
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            # Set NLTK data path for Railway
            if os.path.exists('/app'):  # Railway environment
                nltk.data.path.append('/app/nltk_data')
            
            # Try to find punkt data
            nltk.data.find('tokenizers/punkt')
            print("✅ NLTK punkt data found")
        except LookupError:
            try:
                print("📥 Downloading NLTK punkt data...")
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
                print("✅ NLTK data downloaded")
            except Exception as e:
                print(f"⚠️ NLTK download failed: {e}")
    
    def remove_patterns(self, text: str) -> str:
        """Remove URLs, markdown links, handles, and special characters"""
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        # Remove markdown-style links
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)
        # Remove handles (that start with '@')
        text = re.sub(r'@\w+', '', text)
        # Remove punctuation and other special characters
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def stem_tokens(self, tokens: list) -> str:
        """Stem tokens and convert them to string"""
        return ' '.join(self.stemmer.stem(str(token)) for token in tokens)
    
    def preprocess_text(self, text: str) -> str:
        """Complete text preprocessing pipeline"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove patterns
        text = self.remove_patterns(text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Stem tokens
        stemmed_text = self.stem_tokens(tokens)
        
        return stemmed_text
    
    def extract_numerical_features(self, text: str) -> dict:
        """Extract numerical features from text"""
        return {
            'num_of_characters': len(text),
            'num_of_sentences': len(sent_tokenize(text))
        }
    
    def prepare_features(self, text: str, vectorizer):
        """Prepare features for model prediction"""
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Extract numerical features
        num_features = self.extract_numerical_features(text)
        
        # Vectorize text
        tfidf_features = vectorizer.transform([processed_text])
        
        # Combine features
        numerical_array = np.array([[num_features['num_of_characters'], 
                                   num_features['num_of_sentences']]])
        
        combined_features = hstack([tfidf_features, numerical_array])
        
        return combined_features, num_features

# Global instance
preprocessor = TextPreprocessor()
