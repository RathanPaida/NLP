import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

def get_preprocessor():
    stop_words = set(stopwords.words('english'))
    def preprocess(text):
        if not isinstance(text, str):
            return "", []
        # 1. Lowercase
        text_lower = text.lower()
        # 2. Remove punctuation and numbers
        text_clean = re.sub(r'[^a-z\s]', ' ', text_lower)
        # 3. Tokenize
        tokens = text_clean.split()
        # 4. Remove stopwords
        tokens_no_stop = [w for w in tokens if w not in stop_words and len(w) > 1]
        cleaned_text = " ".join(tokens_no_stop)
        return cleaned_text, tokens_no_stop
    return preprocess

def sample_and_preprocess():
    raw_train_path = r"C:\Users\paida\.cache\kagglehub\datasets\kritanjalijain\amazon-reviews\versions\2\train.csv"
    raw_test_path = r"C:\Users\paida\.cache\kagglehub\datasets\kritanjalijain\amazon-reviews\versions\2\test.csv"
    
    print("Reading candidate rows from train.csv...")
    # Read first 50,000 rows to stratify 10,000 from it cleanly
    df_train_raw = pd.read_csv(raw_train_path, nrows=50000, header=None, names=['label', 'title', 'review'])
    print(f"Read {len(df_train_raw)} train candidate rows. Class distribution:\n{df_train_raw['label'].value_counts()}")
    
    # Stratified sampling of 10,000 samples
    train_sampled, _ = train_test_split(
        df_train_raw, 
        train_size=10000, 
        stratify=df_train_raw['label'], 
        random_state=42
    )
    train_sampled = train_sampled.copy().reset_index(drop=True)
    print(f"Sampled train set shape: {train_sampled.shape}")
    print(f"Train class distribution:\n{train_sampled['label'].value_counts()}")
    
    print("\nReading candidate rows from test.csv...")
    df_test_raw = pd.read_csv(raw_test_path, nrows=10000, header=None, names=['label', 'title', 'review'])
    print(f"Read {len(df_test_raw)} test candidate rows. Class distribution:\n{df_test_raw['label'].value_counts()}")
    
    # Stratified sampling of 1,000 samples
    test_sampled, _ = train_test_split(
        df_test_raw, 
        train_size=1000, 
        stratify=df_test_raw['label'], 
        random_state=42
    )
    test_sampled = test_sampled.copy().reset_index(drop=True)
    print(f"Sampled test set shape: {test_sampled.shape}")
    print(f"Test class distribution:\n{test_sampled['label'].value_counts()}")
    
    # Preprocessing
    print("\nApplying preprocessing: lowercase, remove punctuation/numbers, tokenize, remove stopwords...")
    preprocessor = get_preprocessor()
    
    train_cleaned = [preprocessor(r) for r in train_sampled['review']]
    train_sampled['cleaned_review'] = [item[0] for item in train_cleaned]
    train_sampled['tokens'] = [",".join(item[1]) for item in train_cleaned]
    train_sampled['token_count'] = [len(item[1]) for item in train_cleaned]
    
    test_cleaned = [preprocessor(r) for r in test_sampled['review']]
    test_sampled['cleaned_review'] = [item[0] for item in test_cleaned]
    test_sampled['tokens'] = [",".join(item[1]) for item in test_cleaned]
    test_sampled['token_count'] = [len(item[1]) for item in test_cleaned]
    
    # Save sampled datasets
    os.makedirs("Assignment_10/data", exist_ok=True)
    train_out = "Assignment_10/data/train_sampled.csv"
    test_out = "Assignment_10/data/test_sampled.csv"
    train_sampled.to_csv(train_out, index=False)
    test_sampled.to_csv(test_out, index=False)
    print(f"\nSaved stratified sampled train data to {train_out}")
    print(f"Saved stratified sampled test data to {test_out}")
    print(f"Sample train row 0:\n{train_sampled.iloc[0].to_dict()}")

if __name__ == '__main__':
    sample_and_preprocess()

