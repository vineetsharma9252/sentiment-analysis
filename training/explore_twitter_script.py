import pandas as pd
import numpy as np

def explore_twitter_dataset(csv_path):
    """Explore the structure of the Twitter dataset"""
    print("🔍 Exploring Twitter dataset structure...")
    
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding, nrows=5)
                print(f"✅ Successfully read with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        else:
            print("❌ Could not read with any standard encoding")
            return
        
        print(f"\n📊 Dataset Info:")
        print(f"   - Shape: {df.shape}")
        print(f"   - Columns: {df.columns.tolist()}")
        
        print(f"\n📋 First 5 rows:")
        for i, row in df.iterrows():
            print(f"\nRow {i}:")
            for col in df.columns:
                value = str(row[col])
                if len(value) > 100:
                    value = value[:100] + "..."
                print(f"   {col}: {value}")
        
        print(f"\n🔍 Column Analysis:")
        for col in df.columns:
            print(f"\n{col}:")
            print(f"   - Data type: {df[col].dtype}")
            print(f"   - Non-null count: {df[col].notna().sum()}")
            print(f"   - Unique values: {df[col].nunique()}")
            
            # Show sample values
            sample_values = df[col].dropna().head(3).tolist()
            print(f"   - Sample values: {sample_values}")
        
        # Check file size
        import os
        file_size = os.path.getsize(csv_path)
        print(f"\n💾 File size: {file_size / (1024*1024):.2f} MB")
        
        # Estimate total rows
        with open(csv_path, 'r', encoding=encoding) as f:
            line_count = sum(1 for line in f)
        print(f"📈 Estimated total rows: {line_count - 1}")  # Subtract header
        
    except Exception as e:
        print(f"❌ Error exploring dataset: {e}")

if __name__ == "__main__":
    csv_path = 'C:/Users/maste/Downloads/Python_Assignment/training.1600000.processed.noemoticon.csv/training.1600000.processed.noemoticon.csv'  # Update this to your file name
    explore_twitter_dataset(csv_path)