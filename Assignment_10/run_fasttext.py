import os
import gc
import json
import time
import zipfile
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors

from model_utils import (
    compute_corpus_embeddings, 
    evaluate_custom_analogy, 
    evaluate_analogy_dataset, 
    train_and_evaluate_nn
)

def run_fasttext():
    print("=" * 60)
    print("Running Pipeline for FastText Subword 300d Embeddings")
    print("=" * 60)
    
    t0 = time.time()
    
    # 1. Unzip if needed
    vec_path = "Assignment_10/data/wiki-news-300d-1M-subword.vec"
    zip_path = "Assignment_10/data/wiki-news-300d-1M-subword.vec.zip"
    if not os.path.exists(vec_path):
        print(f"Extracting {zip_path}...")
        unzip_t0 = time.time()
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall("Assignment_10/data")
        print(f"Extracted in {time.time() - unzip_t0:.2f}s")
        
    # 2. Load Data
    train_df = pd.read_csv("Assignment_10/data/train_sampled.csv")
    test_df = pd.read_csv("Assignment_10/data/test_sampled.csv")
    print(f"Loaded Train: {len(train_df)}, Test: {len(test_df)}")
    
    # 3. Fit TF-IDF Vectorizer
    print("Fitting TF-IDF Vectorizer on train reviews...")
    tfidf_vec = TfidfVectorizer(max_features=40000, lowercase=False)
    tfidf_vec.fit(train_df['cleaned_review'].fillna(''))
    print(f"TF-IDF Vocabulary size: {len(tfidf_vec.vocabulary_)}")
    
    # 4. Load FastText Embeddings
    print(f"Loading FastText embeddings from {vec_path}...")
    load_t0 = time.time()
    # FastText .vec has header line: 999994 300
    fasttext_model = KeyedVectors.load_word2vec_format(vec_path, binary=False)
    print(f"FastText loaded in {time.time() - load_t0:.2f}s. Vocabulary size: {len(fasttext_model)}")
    
    # 5. Q1: Compute Sentence Representations
    print("\n--- Q1: Computing Sentence Representations (Mean & TF-IDF) ---")
    os.makedirs("Assignment_10/data/embeddings", exist_ok=True)
    train_mean, train_tfidf = compute_corpus_embeddings(train_df, fasttext_model, tfidf_vectorizer=tfidf_vec, dim=300)
    test_mean, test_tfidf = compute_corpus_embeddings(test_df, fasttext_model, tfidf_vectorizer=tfidf_vec, dim=300)
    
    np.save("Assignment_10/data/embeddings/fasttext_mean_train.npy", train_mean)
    np.save("Assignment_10/data/embeddings/fasttext_mean_test.npy", test_mean)
    np.save("Assignment_10/data/embeddings/fasttext_tfidf_train.npy", train_tfidf)
    np.save("Assignment_10/data/embeddings/fasttext_tfidf_test.npy", test_tfidf)
    print(f"Saved FastText embeddings to Assignment_10/data/embeddings/")
    print(f"Train Mean shape: {train_mean.shape}, Train TF-IDF shape: {train_tfidf.shape}")
    
    # 6. Q2: Word Analogies
    print("\n--- Q2: Target Word Analogies ---")
    # a) good - bad + excellent =
    analogy_a = evaluate_custom_analogy(fasttext_model, positive=['good', 'excellent'], negative=['bad'], topn=5)
    print(f"  a) good - bad + excellent = {analogy_a}")
    
    # b) delivery - late + fast =
    analogy_b = evaluate_custom_analogy(fasttext_model, positive=['delivery', 'fast'], negative=['late'], topn=5)
    print(f"  b) delivery - late + fast = {analogy_b}")
    
    print("\nEvaluating questions-words.txt benchmark on FastText...")
    analogy_eval = evaluate_analogy_dataset(fasttext_model, "Assignment_10/questions-words.txt", max_per_section=150)
    print(f"FastText Analogy Overall Accuracy: {analogy_eval['overall_accuracy']:.2f}% ({analogy_eval['total_correct']}/{analogy_eval['total_tested']})")
    
    # 7. Q3: Neural Network Classification
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    print("\n--- Q3: Neural Network Classification ---")
    print("Training SimpleNN with FastText Mean Pooling...")
    mean_eval = train_and_evaluate_nn(train_mean, y_train, test_mean, y_test, hidden_dim=128, epochs=15, batch_size=64, lr=1e-3)
    print(f"FastText Mean Pooling -> Test Accuracy: {mean_eval['accuracy']:.2f}%, Macro F1: {mean_eval['macro_f1']:.2f}%")
    print(mean_eval['report'])
    
    print("Training SimpleNN with FastText TF-IDF Weighted Pooling...")
    tfidf_eval = train_and_evaluate_nn(train_tfidf, y_train, test_tfidf, y_test, hidden_dim=128, epochs=15, batch_size=64, lr=1e-3)
    print(f"FastText TF-IDF Pooling -> Test Accuracy: {tfidf_eval['accuracy']:.2f}%, Macro F1: {tfidf_eval['macro_f1']:.2f}%")
    print(tfidf_eval['report'])
    
    results = {
        'model_name': 'FastText-Subword-300d',
        'vocab_size': len(fasttext_model),
        'elapsed_seconds': time.time() - t0,
        'analogy_a': analogy_a,
        'analogy_b': analogy_b,
        'analogy_benchmark': analogy_eval,
        'nn_mean': {
            'accuracy': mean_eval['accuracy'],
            'macro_precision': mean_eval['macro_precision'],
            'macro_recall': mean_eval['macro_recall'],
            'macro_f1': mean_eval['macro_f1'],
            'confusion_matrix': mean_eval['confusion_matrix'],
            'history': mean_eval['history']
        },
        'nn_tfidf': {
            'accuracy': tfidf_eval['accuracy'],
            'macro_precision': tfidf_eval['macro_precision'],
            'macro_recall': tfidf_eval['macro_recall'],
            'macro_f1': tfidf_eval['macro_f1'],
            'confusion_matrix': tfidf_eval['confusion_matrix'],
            'history': tfidf_eval['history']
        }
    }
    
    os.makedirs("Assignment_10/results", exist_ok=True)
    with open("Assignment_10/results/fasttext_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved FastText results to Assignment_10/results/fasttext_results.json")
    
    # Release memory
    del fasttext_model
    gc.collect()
    print("FastText model freed from memory.\n")

if __name__ == '__main__':
    run_fasttext()

