import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import emoji
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class TwitterSentimentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            stop_words='english', 
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            sublinear_tf=True
        )
        self.model = LogisticRegression(
            random_state=42, 
            max_iter=2000,
            C=1.0,
            solver='liblinear',
            class_weight='balanced'
        )
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.label_encoder = LabelEncoder()
        
        # Twitter-specific stop words
        self.stop_words.update(['http', 'https', 'com', 'www', 'rt', 'amp', 'co', 'twitter'])
    
    def parse_twitter_dataset(self, csv_path):
        """Parse the Twitter dataset format"""
        print("📊 Parsing Twitter dataset...")
        
        # Try different possible column structures
        try:
            # Common Twitter dataset formats
            df = pd.read_csv(csv_path, encoding='latin-1')
            
            print(f"📋 Dataset columns: {df.columns.tolist()}")
            print(f"📊 Dataset shape: {df.shape}")
            
            # Check common Twitter dataset column names
            text_column = None
            sentiment_column = None
            
            # Look for common text column names
            possible_text_columns = ['text', 'tweet', 'Tweet', 'Text', 'tweet_text', 'message']
            for col in possible_text_columns:
                if col in df.columns:
                    text_column = col
                    break
            
            # Look for common sentiment column names  
            possible_sentiment_columns = ['sentiment', 'Sentiment', 'label', 'Label', 'target', 'class', 'airline_sentiment']
            for col in possible_sentiment_columns:
                if col in df.columns:
                    sentiment_column = col
                    break
            
            if text_column is None:
                # If no standard text column found, try to infer
                # Look for columns that contain text data
                for col in df.columns:
                    if df[col].dtype == 'object' and len(str(df[col].iloc[0])) > 20:
                        text_column = col
                        break
            
            if sentiment_column is None:
                # If no sentiment column, we'll need to create it
                print("❌ No sentiment column found. Creating synthetic labels...")
                df = self.create_synthetic_sentiment(df, text_column)
                sentiment_column = 'sentiment'
            
            print(f"✅ Using text column: '{text_column}'")
            print(f"✅ Using sentiment column: '{sentiment_column}'")
            
            # Basic cleaning
            df = df.dropna(subset=[text_column, sentiment_column])
            df = df.rename(columns={text_column: 'text', sentiment_column: 'sentiment'})
            
            # Clean sentiment labels
            df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
            
            # Map common sentiment encodings
            sentiment_mapping = {
                '0': 'negative', '1': 'positive', '2': 'neutral',
                'negative': 'negative', 'positive': 'positive', 'neutral': 'neutral',
                'neg': 'negative', 'pos': 'positive', 'neu': 'neutral'
            }
            
            df['sentiment'] = df['sentiment'].map(sentiment_mapping).fillna(df['sentiment'])
            
            return df
            
        except Exception as e:
            print(f"❌ Error parsing dataset: {e}")
            raise
    
    def create_synthetic_sentiment(self, df, text_column):
        """Create synthetic sentiment labels using TextBlob"""
        print("🤖 Creating synthetic sentiment labels...")
        
        def get_sentiment_label(text):
            try:
                blob = TextBlob(str(text))
                polarity = blob.sentiment.polarity
                
                if polarity > 0.1:
                    return 'positive'
                elif polarity < -0.1:
                    return 'negative'
                else:
                    return 'neutral'
            except:
                return 'neutral'
        
        df['sentiment'] = df[text_column].apply(get_sentiment_label)
        return df
    
    def advanced_clean_twitter_text(self, text):
        """Advanced cleaning for Twitter text"""
        if pd.isna(text) or text == "":
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Handle emojis
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions but keep the text
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtag symbol but keep the word
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Handle common Twitter patterns
        text = re.sub(r'rt\s+', '', text)  # Remove RT
        text = re.sub(r'&amp;', 'and', text)  # Replace &amp;
        
        # Handle contractions
        contractions = {
            "don't": "do not", "doesn't": "does not", "didn't": "did not",
            "can't": "cannot", "couldn't": "could not", "won't": "will not",
            "wouldn't": "would not", "isn't": "is not", "aren't": "are not",
            "wasn't": "was not", "weren't": "were not", "haven't": "have not",
            "hasn't": "has not", "hadn't": "had not", "i'm": "i am",
            "you're": "you are", "he's": "he is", "she's": "she is",
            "it's": "it is", "we're": "we are", "they're": "they are",
            "i've": "i have", "you've": "you have", "we've": "we have",
            "they've": "they have", "i'll": "i will", "you'll": "you will",
            "he'll": "he will", "she'll": "she will", "we'll": "we will",
            "they'll": "they will", "i'd": "i would", "you'd": "you would"
        }
        
        for cont, expanded in contractions.items():
            text = re.sub(cont, expanded, text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z\s!?]', ' ', text)
        
        # Handle repeated characters
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and lemmatize
        words = text.split()
        cleaned_words = []
        
        for word in words:
            if word not in self.stop_words and len(word) > 1:
                # Handle negation
                if word in ['not', 'no', 'never']:
                    cleaned_words.append('not')
                else:
                    lemma = self.lemmatizer.lemmatize(word)
                    cleaned_words.append(lemma)
        
        return ' '.join(cleaned_words)
    
    def analyze_twitter_dataset(self, df):
        """Analyze the Twitter dataset"""
        print(f"\n📈 Twitter Dataset Analysis:")
        print(f"   - Total tweets: {len(df)}")
        
        # Sentiment distribution
        sentiment_counts = df['sentiment'].value_counts()
        print(f"   - Sentiment distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(df)) * 100
            print(f"     {sentiment}: {count} ({percentage:.1f}%)")
        
        # Text length analysis
        df['text_length'] = df['text'].str.len()
        df['cleaned_length'] = df['cleaned_text'].str.len()
        
        print(f"   - Original text length: {df['text_length'].mean():.1f} chars")
        print(f"   - Cleaned text length: {df['cleaned_length'].mean():.1f} chars")
        
        # Show sample tweets
        print(f"\n🔍 Sample Tweets:")
        for sentiment in df['sentiment'].unique():
            sample = df[df['sentiment'] == sentiment].head(1)
            if len(sample) > 0:
                text = sample['text'].iloc[0]
                cleaned = sample['cleaned_text'].iloc[0]
                print(f"   {sentiment}: '{text[:60]}...'")
                print(f"     → '{cleaned}'")
    
    def train_multiple_models(self, X_train_tfidf, X_test_tfidf, y_train, y_test):
        """Train and compare multiple models"""
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000, class_weight='balanced'),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'),
            # 'SVM': SVC(random_state=42, probability=True, class_weight='balanced'),
        }
        
        best_model = None
        best_accuracy = 0
        best_model_name = ""
        
        print("\n🤖 Training Multiple Models:")
        print("-" * 50)
        
        for name, model in models.items():
            try:
                model.fit(X_train_tfidf, y_train)
                y_pred = model.predict(X_test_tfidf)
                accuracy = accuracy_score(y_test, y_pred)
                
                print(f"{name:20} Accuracy: {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_model_name = name
                    
            except Exception as e:
                print(f"{name:20} Failed: {str(e)}")
        
        print("-" * 50)
        print(f"🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return best_model, best_accuracy
    
    def train(self, csv_path):
        """Main training function for Twitter dataset"""
        # Parse dataset
        df = self.parse_twitter_dataset(csv_path)
        
        # Clean text
        print("🧹 Cleaning Twitter text...")
        df['cleaned_text'] = df['text'].apply(self.advanced_clean_twitter_text)
        
        # Remove very short texts
        df = df[df['cleaned_text'].str.len() > 5]
        
        # Analyze dataset
        self.analyze_twitter_dataset(df)
        
        # Prepare features and labels
        X = df['cleaned_text']
        y = df['sentiment']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data - keep original X for display, but use encoded y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"\n📊 Data Split:")
        print(f"   - Training samples: {len(X_train)}")
        print(f"   - Test samples: {len(X_test)}")
        
        # Vectorize
        print("🔤 Vectorizing text...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        print(f"   - Feature matrix shape: {X_train_tfidf.shape}")
        
        # Train multiple models
        best_model, best_accuracy = self.train_multiple_models(
            X_train_tfidf, X_test_tfidf, y_train, y_test
        )
        
        self.model = best_model
        
        # Final evaluation
        y_pred = self.model.predict(X_test_tfidf)
        final_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🎯 Final Model Accuracy: {final_accuracy:.4f}")
        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Show some predictions - FIXED VERSION
        print("\n🔍 Sample Predictions:")
        sample_indices = np.random.choice(len(X_test), min(8, len(X_test)), replace=False)
        
        for idx in sample_indices:
            # FIX: Use X_test.iloc[idx] for pandas Series, but y_test[idx] for numpy array
            original_text = X_test.iloc[idx]  # This is pandas Series, so use .iloc
            true_label_encoded = y_test[idx]  # This is numpy array, use direct indexing
            pred_label_encoded = y_pred[idx]  # This is numpy array, use direct indexing
            
            true_label = self.label_encoder.inverse_transform([true_label_encoded])[0]
            pred_label = self.label_encoder.inverse_transform([pred_label_encoded])[0]
            
            # Get confidence
            text_vectorized = self.vectorizer.transform([original_text])
            confidence = np.max(self.model.predict_proba(text_vectorized))
            
            print(f"Text: {original_text[:70]}...")
            print(f"True: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.3f}")
            print("✓ Correct" if true_label == pred_label else "✗ Wrong")
            print()
        
        return final_accuracy
    
    def save_model(self, file_path):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'lemmatizer': self.lemmatizer,
            'stop_words': self.stop_words,
            'label_encoder': self.label_encoder
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to {file_path}")

def main():
    """Main training function"""
    print("🚀 Starting Twitter Sentiment Analysis Training...")
    
    classifier = TwitterSentimentClassifier()
    
    # Train model
    csv_path = 'C:/Users/maste/Downloads/Python_Assignment/twitter_training.csv/twitter_training.csv'
    
    try:
        accuracy = classifier.train(csv_path)
        
        # Save model
        classifier.save_model('../models/model.pkl')
        
        print("✅ Training completed successfully!")
        print(f"📊 Final model accuracy: {accuracy:.4f}")
        
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at '{csv_path}'")
        print("💡 Please make sure your Twitter dataset file exists and update the csv_path variable.")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Install required packages
    try:
        import emoji
        from textblob import TextBlob
    except ImportError:
        print("📦 Installing required packages...")
        import subprocess
        subprocess.check_call(["pip", "install", "emoji", "textblob"])
        import emoji
        from textblob import TextBlob
    
    main()