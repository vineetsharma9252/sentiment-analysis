from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List
import os
import re

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for classifying text sentiment - Trained on Twitter Dataset",
    version="1.0.0"
)

# Global variables for loaded model
model = None
vectorizer = None
lemmatizer = None
stop_words = None
label_encoder = None  # Add this

class SentimentRequest(BaseModel):
    text: str

class BatchSentimentRequest(BaseModel):
    texts: List[str]

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool

    class Config:
        protected_namespaces = ()  # Fix the warning

def load_model():
    """Load the trained model and vectorizer"""
    global model, vectorizer, lemmatizer, stop_words, label_encoder
    
    try:
        model_path = 'models/model.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run the training script first.")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        vectorizer = model_data['vectorizer']
        lemmatizer = model_data.get('lemmatizer')
        stop_words = model_data.get('stop_words')
        label_encoder = model_data.get('label_encoder')  # Load label encoder
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model classes: {model.classes_}")
        if label_encoder:
            print(f"🎯 Label encoder classes: {label_encoder.classes_}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

def clean_text(text):
    """Clean and preprocess text"""
    if lemmatizer is None or stop_words is None:
        # Simple cleaning if advanced preprocessing not available
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # Advanced cleaning with lemmatization
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize and lemmatize
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words 
            if word not in stop_words and len(word) > 2]
    
    return ' '.join(words)

def predict_sentiment(text):
    """Predict sentiment for a single text"""
    try:
        if not text.strip():
            return "invalid", 0.0
        
        # Clean text
        cleaned_text = clean_text(text)
        
        # Vectorize text
        text_vectorized = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(text_vectorized)[0]
        
        # Convert numeric prediction to string label
        if label_encoder:
            sentiment = label_encoder.inverse_transform([prediction])[0]
        else:
            # Fallback: assume model returns string labels
            sentiment = str(prediction)
        
        confidence = np.max(model.predict_proba(text_vectorized))
        
        return sentiment, round(confidence, 4)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()

@app.get("/", summary="Root endpoint")
async def root():
    if model is not None and label_encoder is not None:
        supported_sentiments = label_encoder.classes_.tolist()
    else:
        supported_sentiments = []
    
    return {
        "message": "Sentiment Analysis API (Twitter Dataset)", 
        "status": "running",
        "model_loaded": model is not None,
        "supported_sentiments": supported_sentiments
    }

@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None and vectorizer is not None
    return HealthResponse(
        status="success" if model_loaded else "error",
        message="API is running smoothly" if model_loaded else "Model not loaded",
        model_loaded=model_loaded
    )

@app.post("/predict", response_model=SentimentResponse, summary="Predict sentiment")
async def predict_sentiment_endpoint(request: SentimentRequest):
    """Predict sentiment for a single text"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check if model.pkl exists.")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    sentiment, confidence = predict_sentiment(request.text)
    
    return SentimentResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=confidence
    )

@app.post("/batch_predict", response_model=List[SentimentResponse], summary="Batch predict sentiment")
async def batch_predict_sentiment(request: BatchSentimentRequest):
    """Predict sentiment for multiple texts"""
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check if model.pkl exists.")
    
    if not request.texts:
        raise HTTPException(status_code=400, detail="Texts list cannot be empty")
    
    results = []
    for text in request.texts:
        sentiment, confidence = predict_sentiment(text)
        results.append(SentimentResponse(
            text=text,
            sentiment=sentiment,
            confidence=confidence
        ))
    
    return results

@app.get("/model_info", summary="Get model information")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if label_encoder:
        classes = label_encoder.classes_.tolist()
    else:
        classes = model.classes_.tolist() if hasattr(model, 'classes_') else []
    
    return {
        "model_type": type(model).__name__,
        "classes": classes,
        "n_features": vectorizer.get_feature_names_out().shape[0] if vectorizer else 0,
        "vectorizer_type": type(vectorizer).__name__,
        "label_encoder_loaded": label_encoder is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)