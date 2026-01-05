# Fake News Detection System - ML Backend for Pakistan News
# Requirements: pip install flask flask-cors scikit-learn pandas numpy

from flask import Flask, request, jsonify
from flask_cors import CORS
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import pickle
import os
import traceback

app = Flask(__name__)
CORS(app)

class PakistanNewsDetector:
    """Machine Learning based fake news detector for Pakistan news"""
    
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1
        )
        self.model = MultinomialNB()
        self.is_trained = False
        
        # Pakistan-specific keywords
        self.pakistan_keywords: List[str] = [
            'pakistan', 'karachi', 'lahore', 'islamabad', 'punjab', 
            'sindh', 'balochistan', 'kpk', 'peshawar', 'quetta'
        ]
        
        self.credible_sources: List[str] = [
            'dawn', 'express tribune', 'state bank', 'supreme court', 
            'high court', 'university', 'research', 'study', 
            'according to', 'ministry', 'official'
        ]
        
        self.fake_indicators: List[str] = [
            'shocking', 'you won\'t believe', 'miracle', 'secret',
            'exposed', 'bombshell', 'conspiracy', 'share before',
            'government deletes', 'act now', 'urgent'
        ]
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def train_model(self, csv_path: str = 'pakistan_news_dataset.csv') -> Dict[str, float]:
        """Train the model on Pakistan news dataset"""
        try:
            print(f"📂 Loading dataset from: {csv_path}")
            
            # Check if file exists
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"Dataset file not found: {csv_path}")
            
            # Load dataset
            df = pd.read_csv(csv_path)
            print(f"✅ Loaded {len(df)} samples")
            
            # Validate dataset
            if 'text' not in df.columns or 'label' not in df.columns:
                raise ValueError("Dataset must have 'text' and 'label' columns")
            
            # Remove any NaN values
            df = df.dropna()
            print(f"✅ After cleaning: {len(df)} samples")
            
            if len(df) < 10:
                raise ValueError("Dataset too small. Need at least 10 samples.")
            
            # Preprocess text
            df['processed_text'] = df['text'].apply(self.preprocess_text)
            
            # Remove empty texts
            df = df[df['processed_text'].str.len() > 0]
            
            print(f"✅ Processed {len(df)} samples")
            
            # Split data (smaller test size for small datasets)
            test_size = min(0.2, 5 / len(df))  # At least 80% training
            X_train, X_test, y_train, y_test = train_test_split(
                df['processed_text'], 
                df['label'], 
                test_size=test_size, 
                random_state=42,
                stratify=df['label'] if len(df) > 20 else None
            )
            
            print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # Vectorize text
            X_train_vectorized = self.vectorizer.fit_transform(X_train)
            X_test_vectorized = self.vectorizer.transform(X_test)
            
            print("✅ Text vectorization complete")
            
            # Train model
            self.model.fit(X_train_vectorized, y_train)
            print("✅ Model training complete")
            
            # Evaluate
            train_predictions = self.model.predict(X_train_vectorized)
            train_accuracy = accuracy_score(y_train, train_predictions)
            
            test_accuracy = 0.0
            if len(X_test) > 0:
                test_predictions = self.model.predict(X_test_vectorized)
                test_accuracy = accuracy_score(y_test, test_predictions)
            
            self.is_trained = True
            
            # Save model
            self.save_model()
            
            print(f"\n{'='*50}")
            print(f"MODEL TRAINING SUCCESSFUL!")
            print(f"{'='*50}")
            print(f"Training Accuracy: {train_accuracy:.2%}")
            print(f"Testing Accuracy: {test_accuracy:.2%}")
            print(f"Dataset size: {len(df)} samples")
            print(f"{'='*50}\n")
            
            return {
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'dataset_size': len(df)
            }
            
        except Exception as e:
            print(f"\n❌ ERROR TRAINING MODEL:")
            print(f"Error: {str(e)}")
            traceback.print_exc()
            raise
    
    def save_model(self, model_path: str = 'model.pkl', 
                   vectorizer_path: str = 'vectorizer.pkl') -> None:
        """Save trained model and vectorizer"""
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(vectorizer_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            print("💾 Model saved successfully!")
        except Exception as e:
            print(f"⚠️ Could not save model: {str(e)}")
    
    def load_model(self, model_path: str = 'model.pkl', 
                   vectorizer_path: str = 'vectorizer.pkl') -> bool:
        """Load trained model and vectorizer"""
        try:
            if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
                return False
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            self.is_trained = True
            print("✅ Model loaded successfully from disk!")
            return True
        except Exception as e:
            print(f"⚠️ Could not load model: {str(e)}")
            return False
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze text for fake news detection"""
        try:
            if not self.is_trained:
                raise ValueError("Model not trained. Please train the model first.")
            
            if not text or len(text.strip()) == 0:
                raise ValueError("Empty text provided")
            
            # Preprocess
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                raise ValueError("Text preprocessing resulted in empty string")
            
            # Vectorize
            text_vectorized = self.vectorizer.transform([processed_text])
            
            # Predict
            prediction = int(self.model.predict(text_vectorized)[0])
            prediction_proba = self.model.predict_proba(text_vectorized)[0]
            
            # Get confidence - handle array indexing safely
            confidence = float(prediction_proba[prediction] * 100)
            
            # Additional rule-based analysis
            indicators = self._get_indicators(text.lower())
            
            # Adjust confidence based on indicators
            final_confidence = self._adjust_confidence(confidence, indicators)
            
            # Classification
            if prediction == 1:
                classification = "Likely Reliable"
                score = final_confidence
            else:
                classification = "Likely Fake/Misleading"
                score = 100 - final_confidence
            
            return {
                'classification': classification,
                'confidence': round(final_confidence, 2),
                'score': round(score, 2),
                'indicators': indicators,
                'ml_prediction': prediction,
                'model_confidence': round(confidence, 2)
            }
            
        except Exception as e:
            print(f"❌ Error in analyze_text: {str(e)}")
            traceback.print_exc()
            raise
    
    def _get_indicators(self, text: str) -> List[Dict[str, str]]:
        """Get specific indicators from text"""
        indicators = []
        
        # Pakistan context
        has_pakistan = any(kw in text for kw in self.pakistan_keywords)
        if has_pakistan:
            indicators.append({
                'type': 'info',
                'text': 'Content related to Pakistan'
            })
        
        # Credible sources
        credible = [s for s in self.credible_sources if s in text]
        if credible:
            indicators.append({
                'type': 'positive',
                'text': f'References credible sources'
            })
        
        # Fake indicators
        fake = [f for f in self.fake_indicators if f in text]
        if fake:
            indicators.append({
                'type': 'negative',
                'text': f'Contains suspicious language'
            })
        
        # Punctuation
        exclamations = text.count('!')
        if exclamations > 3:
            indicators.append({
                'type': 'negative',
                'text': f'Excessive exclamation marks ({exclamations})'
            })
        
        # Caps
        if len(text) > 20:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.3:
                indicators.append({
                    'type': 'negative',
                    'text': 'Excessive use of capital letters'
                })
        
        return indicators
    
    def _adjust_confidence(self, confidence: float, indicators: List[Dict]) -> float:
        """Adjust ML confidence based on indicators"""
        adjustment = 0.0
        
        for indicator in indicators:
            if indicator['type'] == 'positive':
                adjustment += 5
            elif indicator['type'] == 'negative':
                adjustment -= 5
        
        return max(0.0, min(100.0, confidence + adjustment))

# Initialize detector
detector = PakistanNewsDetector()

# Try to load existing model or train new one
print("\n" + "="*50)
print("🇵🇰 PAKISTAN FAKE NEWS DETECTOR - INITIALIZING")
print("="*50 + "\n")

if detector.load_model():
    print("✅ Using existing trained model\n")
else:
    print("⚠️ No existing model found. Training new model...\n")
    try:
        metrics = detector.train_model('pakistan_news_dataset.csv')
    except Exception as e:
        print(f"\n❌ CRITICAL: Could not train model")
        print(f"Error: {str(e)}")
        print("\n💡 SOLUTION:")
        print("1. Make sure 'pakistan_news_dataset.csv' exists in the same folder")
        print("2. Check that the CSV has 'text' and 'label' columns")
        print("3. Verify the CSV format is correct\n")

@app.route('/api/analyze', methods=['POST'])
def analyze_news() -> tuple:
    """API endpoint to analyze news text"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if not detector.is_trained:
            return jsonify({
                'error': 'Model not trained. Check server console for details.'
            }), 500
        
        # Analyze the text
        result = detector.analyze_text(text)
        
        return jsonify(result), 200
    
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ API Error: {error_msg}")
        traceback.print_exc()
        return jsonify({'error': error_msg}), 500

@app.route('/api/train', methods=['POST'])
def train_model() -> tuple:
    """API endpoint to retrain the model"""
    try:
        metrics = detector.train_model('pakistan_news_dataset.csv')
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'metrics': metrics
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check() -> tuple:
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'pakistan-fake-news-detector',
        'model_trained': detector.is_trained
    }), 200

@app.route('/api/stats', methods=['GET'])
def get_stats() -> tuple:
    """Get model statistics"""
    return jsonify({
        'model_trained': detector.is_trained,
        'model_type': 'Naive Bayes with TF-IDF',
        'features': 'Pakistan-focused news analysis'
    }), 200

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 STARTING API SERVER")
    print("="*50)
    print(f"📍 Server: http://localhost:5000")
    print(f"🔧 Status: {'✅ Ready' if detector.is_trained else '❌ Model Not Trained'}")
    print("\n📋 Available Endpoints:")
    print("   POST /api/analyze - Analyze news text")
    print("   POST /api/train   - Retrain model")
    print("   GET  /api/health  - Health check")
    print("   GET  /api/stats   - Model statistics")
    print("="*50 + "\n")
    
    app.run(debug=True, port=5000, host='127.0.0.1')