import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

def create_model():
    """Create and save a simple sentiment analysis model"""
    
    # Create a balanced dataset
    data = {
        'text': [
            # Positive reviews (20)
            'I love this product it is amazing', 'Excellent quality and fast delivery',
            'Outstanding service great support', 'Wonderful experience fantastic',
            'Perfect solution works flawlessly', 'Amazing features user friendly',
            'Brilliant product excellent performance', 'Very happy with purchase',
            'Superb quality great value', 'Fantastic from start to finish',
            'Exceeded expectations completely', 'Delivers on all promises',
            'Excellent craftsmanship attention to detail', 'Love everything perfect',
            'Outstanding performance reliable', 'Great product highly recommend',
            'Awesome service quick response', 'Best purchase ever made',
            'Impressive quality and design', 'Very satisfied would buy again',
            
            # Negative reviews (20)
            'Terrible product hate it', 'Poor quality awful service',
            'Worst experience very disappointed', 'Bad value useless product',
            'Horrible performance garbage', 'Useless product waste money',
            'Awful features terrible', 'Disappointing solution bad',
            'Garbage product horrible', 'Useless service worst',
            'Broken immediately regret buying', 'Poor construction cheap materials',
            'Frustrating experience avoid', 'Not working properly defective',
            'Complete scam fake product', 'Very poor quality disappointed',
            'Waste of time money', 'Terrible customer service rude',
            'Defective item broken', 'Low quality not worth it',
            
            # Neutral reviews (20)
            'Okay product average', 'Normal quality standard service',
            'Average experience regular', 'Medium value acceptable',
            'Standard performance normal', 'Regular product okay',
            'Average features normal', 'Acceptable solution standard',
            'Normal product average', 'Standard service okay',
            'Nothing special ordinary', 'Basic product functional',
            'Decent quality average', 'Moderate performance standard',
            'Fair price acceptable', 'Usable product adequate',
            'Standard features normal', 'Average performance expected',
            'Basic functionality works', 'Regular quality standard'
        ],
        'sentiment': ['positive']*20 + ['negative']*20 + ['neutral']*20
    }
    
    df = pd.DataFrame(data)
    
    # Create vectorizer and model
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # Train the model
    X = vectorizer.fit_transform(df['text'])
    y = df['sentiment']
    
    model.fit(X, y)
    
    # Calculate accuracy
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"Model trained with accuracy: {accuracy:.4f}")
    
    # Save the model
    model_data = {
        'model': model,
        'vectorizer': vectorizer
    }
    
    # Create models directory if it doesn't exist
    os.makedirs('../models', exist_ok=True)
    
    with open('../models/model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print("Model saved as 'models/model.pkl'")
    print(f"Dataset size: {len(df)} samples")
    print("Sentiment distribution:")
    print(df['sentiment'].value_counts())

if __name__ == "__main__":
    create_model()