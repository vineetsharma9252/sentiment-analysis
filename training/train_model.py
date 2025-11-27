import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests
import io

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

class SentimentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words='english')
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if isinstance(text, float):  # Handle NaN values
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Tokenize and lemmatize
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words and len(word) > 2]
        return ' '.join(words)
    
    def load_imdb_dataset(self):
        """Load IMDB dataset from Hugging Face"""
        try:
            # Using a smaller, more manageable dataset
            url = "https://raw.githubusercontent.com/clairett/pytorch-sentiment-classification/master/data/SST2/train.tsv"
            response = requests.get(url)
            df = pd.read_csv(io.StringIO(response.text), sep='\t')
            
            # Convert binary sentiment to multi-class for demonstration
            # 0 = negative, 1 = positive - we'll add some neutral samples
            df = df.rename(columns={'sentence': 'text', 'label': 'sentiment'})
            df['sentiment'] = df['sentiment'].map({0: 'negative', 1: 'positive'})
            
            # Add some neutral samples (simulated)
            neutral_samples = [
                "The movie was okay, nothing special.",
                "It was an average film with some good moments.",
                "Not great but not terrible either.",
                "The acting was decent but the plot was weak.",
                "I have mixed feelings about this movie.",
                "It was neither good nor bad.",
                "The film was mediocre at best.",
                "Some parts were interesting, others were boring.",
                "An acceptable movie but not memorable.",
                "The storyline was predictable but entertaining."
            ]
            
            neutral_df = pd.DataFrame({
                'text': neutral_samples,
                'sentiment': ['neutral'] * len(neutral_samples)
            })
            
            # Combine datasets
            df = pd.concat([df.head(200), neutral_df], ignore_index=True)
            
            return df
            
        except Exception as e:
            print(f"Error loading IMDB dataset: {e}")
            print("Creating a custom dataset as fallback...")
            return self.create_custom_dataset()
    
    def create_custom_dataset(self):
        """Create a custom dataset with sufficient samples"""
        data = {
            'text': [
                # Positive samples
                'I love this product! It is absolutely amazing and fantastic.',
                'Excellent quality and fast delivery, very satisfied!',
                'Outstanding service and great customer support.',
                'This is wonderful and I would highly recommend it.',
                'Perfect solution for my needs, works flawlessly.',
                'Amazing features and very user-friendly interface.',
                'Brilliant product with excellent performance.',
                'I am extremely happy with this purchase.',
                'Superb quality and great value for money.',
                'Fantastic experience from start to finish.',
                'This exceeded my expectations completely.',
                'Wonderful product that delivers on all promises.',
                'Excellent craftsmanship and attention to detail.',
                'I love everything about this, perfect!',
                'Outstanding performance and reliability.',
                
                # Negative samples
                'This is terrible and I hate it, very disappointed.',
                'Poor quality and terrible customer service.',
                'Absolutely awful product, do not buy.',
                'Worst purchase I have ever made, complete waste.',
                'Horrible experience and poor quality.',
                'This product is garbage and broke immediately.',
                'Very dissatisfied with this terrible product.',
                'Awful performance and unreliable.',
                'I regret buying this, completely useless.',
                'Terrible quality and not worth the money.',
                'This is the worst product ever.',
                'Horrible customer service experience.',
                'Completely disappointed and frustrated.',
                'Poor construction and cheap materials.',
                'Absolutely terrible, avoid at all costs.',
                
                # Neutral samples
                'It is okay, nothing special about it.',
                'The product is average, neither good nor bad.',
                'Not bad but could be better in many ways.',
                'Mediocre product with some okay features.',
                'It works but nothing extraordinary.',
                'Average quality and standard performance.',
                'Neither impressive nor disappointing.',
                'It serves its purpose but nothing more.',
                'Standard product with no special features.',
                'Acceptable but not remarkable in any way.',
                'Fair quality for the price paid.',
                'It is what it is, nothing special.',
                'Average performance as expected.',
                'Does the job but nothing exceptional.',
                'Moderate quality with basic functionality.'
            ],
            'sentiment': ['positive'] * 15 + ['negative'] * 15 + ['neutral'] * 15
        }
        
        return pd.DataFrame(data)
    
    def train(self, df):
        """Train the sentiment classifier"""
        # Clean the text
        print("Cleaning text data...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Remove empty texts after cleaning
        df = df[df['cleaned_text'].str.len() > 0]
        
        print(f"Dataset size: {len(df)}")
        print("Sentiment distribution:")
        print(df['sentiment'].value_counts())
        
        # Prepare features and labels
        X = df['cleaned_text']
        y = df['sentiment']
        
        # Split the data - using larger test size for small dataset
        test_size = 0.15 if len(df) > 100 else 0.1
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Vectorize text
        print("Vectorizing text...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_tfidf.shape}")
        
        # Train model
        print("Training model...")
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Show some predictions
        print("\nSample Predictions:")
        for i in range(min(5, len(X_test))):
            print(f"Text: {X_test.iloc[i][:50]}...")
            print(f"True: {y_test.iloc[i]}, Predicted: {y_pred[i]}")
            print()
        
        return accuracy
    
    def save_model(self, file_path):
        """Save the trained model and vectorizer"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'lemmatizer': self.lemmatizer,
            'stop_words': self.stop_words
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {file_path}")

def main():
    """Main training function"""
    classifier = SentimentClassifier()
    
    # Load and preprocess data
    print("Loading dataset...")
    df = classifier.load_imdb_dataset()
    
    print(f"Dataset loaded with {len(df)} samples")
    
    # Train model
    accuracy = classifier.train(df)
    
    # Save model
    print("Saving model...")
    classifier.save_model('../models/model.pkl')
    
    print("Training completed successfully!")
    print(f"Final model accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()