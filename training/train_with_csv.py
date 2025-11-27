import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
import seaborn as sns


nltk.download('stopwords')
nltk.download('wordnet')

class CSVSentimentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=3000, 
            stop_words='english', 
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9
        )
        self.model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            C=1.0,
            solver='liblinear'
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Enhanced text cleaning and preprocessing"""
        if pd.isna(text) or text == "":
            return ""
        
        text = str(text).lower()
        
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def load_and_analyze_dataset(self, csv_path):
        """Load and analyze the CSV dataset"""
        print("📊 Loading dataset...")
        df = pd.read_csv(csv_path)
        
        print(f"📁 Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
        print("📋 Columns:", df.columns.tolist())
        
    
        if 'text' not in df.columns or 'sentiment' not in df.columns:
            raise ValueError("Dataset must contain 'text' and 'sentiment' columns")
        
    
        print(f"\n📈 Dataset Analysis:")
        print(f"   - Total samples: {len(df)}")
        print(f"   - Sentiment distribution:")
        sentiment_counts = df['sentiment'].value_counts()
        print(sentiment_counts)
        
    
        missing_text = df['text'].isna().sum()
        missing_sentiment = df['sentiment'].isna().sum()
        print(f"   - Missing text values: {missing_text}")
        print(f"   - Missing sentiment values: {missing_sentiment}")
        
        
        df = df.dropna(subset=['text', 'sentiment'])
        
        
        df['sentiment'] = df['sentiment'].str.strip().str.lower()
        
        unique_sentiments = df['sentiment'].unique()
        print(f"   - Unique sentiments: {unique_sentiments}")
        
        return df
    
    def explore_text_lengths(self, df):
        """Explore text length distribution"""
        df['text_length'] = df['text'].str.len()
        
        print(f"\n📏 Text Length Analysis:")
        print(f"   - Average length: {df['text_length'].mean():.1f} characters")
        print(f"   - Minimum length: {df['text_length'].min()} characters")
        print(f"   - Maximum length: {df['text_length'].max()} characters")
        
        
        print(f"\n📏 Text Length by Sentiment:")
        for sentiment in df['sentiment'].unique():
            sentiment_df = df[df['sentiment'] == sentiment]
            avg_len = sentiment_df['text_length'].mean()
            print(f"   - {sentiment}: {avg_len:.1f} chars (n={len(sentiment_df)})")
    
    def train(self, csv_path):
        """Train the sentiment classifier using the CSV dataset"""
        
        df = self.load_and_analyze_dataset(csv_path)
        
        self.explore_text_lengths(df)
        
        print("\n🧹 Cleaning text data...")
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        df = df[df['cleaned_text'].str.len() > 0]
        
        print(f"\n✅ Final dataset size: {len(df)} samples")
        print("🎯 Sentiment distribution:")
        final_distribution = df['sentiment'].value_counts()
        print(final_distribution)
        
        X = df['cleaned_text']
        y = df['sentiment']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Data Split:")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Test samples: {len(X_test)}")
        
        # Vectorize text
        print("🔤 Vectorizing text...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"   - Feature matrix shape: {X_train_tfidf.shape}")
        
        # Train model
        print("🤖 Training model...")
        self.model.fit(X_train_tfidf, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🎯 Model Test Accuracy: {accuracy:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        print("\n📈 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Show some predictions with confidence
        print("\n🔍 Sample Predictions (with confidence scores):")
        sample_indices = np.random.choice(len(X_test), min(10, len(X_test)), replace=False)
        
        for idx in sample_indices:
            actual_text = X_test.iloc[idx]
            true_label = y_test.iloc[idx]
            
            # Get prediction and confidence
            text_vectorized = self.vectorizer.transform([actual_text])
            pred_label = self.model.predict(text_vectorized)[0]
            confidence = np.max(self.model.predict_proba(text_vectorized))
            
            print(f"Text: {actual_text[:70]}...")
            print(f"True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.3f}")
            print("✓ Correct" if true_label == pred_label else "✗ Wrong")
            print()
        
        return accuracy, X_test, y_test, y_pred
    
    def save_model(self, file_path):
        """Save the trained model and all required components"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'lemmatizer': self.lemmatizer,
            'stop_words': self.stop_words
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {file_path}")

def main():
    """Main training function"""
    print("🚀 Starting Sentiment Analysis Model Training with CSV Dataset...")
    
    classifier = CSVSentimentClassifier()
    
    # Train model - update the path to your CSV file
    csv_path = 'C:/Users/maste/Downloads/Python_Assignment/sentiment-analysis-api/training/sentiment_analysis.csv'  # Update this path if needed
    
    try:
        accuracy, X_test, y_test, y_pred = classifier.train(csv_path)
        
        # Save model
        classifier.save_model('../models/model.pkl')
        
        print("✅ Training completed successfully!")
        print(f"📊 Final model test accuracy: {accuracy:.4f}")
        
        # Print model info
        print(f"\n📋 Model Information:")
        print(f"   - Dataset: Your custom sentiment analysis CSV")
        print(f"   - Features: 3000 TF-IDF features with n-grams (1,2)")
        print(f"   - Algorithm: Logistic Regression")
        print(f"   - Preprocessing: Advanced text cleaning & lemmatization")
        
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at '{csv_path}'")
        print("💡 Please make sure the CSV file is in the correct location.")
        print("💡 You may need to update the 'csv_path' variable in the script.")
    except Exception as e:
        print(f"❌ Error during training: {e}")

if __name__ == "__main__":
    main()