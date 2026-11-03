import os
import gc
import json
import time
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

def run_word2vec():
    print("=" * 60)
    print("Running Pipeline for Google News Word2Vec 300d Embeddings")
    print("=" * 60)
    
    t0 = time.time()
    
    # 1. Load Data
    train_df = pd.read_csv("Assignment_10/data/train_sampled.csv")
    test_df = pd.read_csv("Assignment_10/data/test_sampled.csv")
    print(f"Loaded Train: {len(train_df)}, Test: {len(test_df)}")
    
    # 2. Fit TF-IDF Vectorizer
    print("Fitting TF-IDF Vectorizer on train reviews...")
    tfidf_vec = TfidfVectorizer(max_features=40000, lowercase=False)
    tfidf_vec.fit(train_df['cleaned_review'].fillna(''))
    print(f"TF-IDF Vocabulary size: {len(tfidf_vec.vocabulary_)}")
    
    # 3. Load Word2Vec Embeddings
    w2v_path = "Assignment_10/data/word2vec-google-news-300.gz"
    print(f"Loading Word2Vec embeddings from {w2v_path}...")
    load_t0 = time.time()
    # GoogleNews is binary format, gzipped
    word2vec_model = KeyedVectors.load_word2vec_format(w2v_path, binary=True)
    print(f"Word2Vec loaded in {time.time() - load_t0:.2f}s. Vocabulary size: {len(word2vec_model)}")
    
    # 4. Q1: Compute Sentence Representations
    print("\n--- Q1: Computing Sentence Representations (Mean & TF-IDF) ---")
    os.makedirs("Assignment_10/data/embeddings", exist_ok=True)
    train_mean, train_tfidf = compute_corpus_embeddings(train_df, word2vec_model, tfidf_vectorizer=tfidf_vec, dim=300)
    test_mean, test_tfidf = compute_corpus_embeddings(test_df, word2vec_model, tfidf_vectorizer=tfidf_vec, dim=300)
    
    np.save("Assignment_10/data/embeddings/word2vec_mean_train.npy", train_mean)
    np.save("Assignment_10/data/embeddings/word2vec_mean_test.npy", test_mean)
    np.save("Assignment_10/data/embeddings/word2vec_tfidf_train.npy", train_tfidf)
    np.save("Assignment_10/data/embeddings/word2vec_tfidf_test.npy", test_tfidf)
    print(f"Saved Word2Vec embeddings to Assignment_10/data/embeddings/")
    print(f"Train Mean shape: {train_mean.shape}, Train TF-IDF shape: {train_tfidf.shape}")
    
    # 5. Q2: Word Analogies
    print("\n--- Q2: Target Word Analogies ---")
    # a) good - bad + excellent =
    analogy_a = evaluate_custom_analogy(word2vec_model, positive=['good', 'excellent'], negative=['bad'], topn=5)
    print(f"  a) good - bad + excellent = {analogy_a}")
    
    # b) delivery - late + fast =
    analogy_b = evaluate_custom_analogy(word2vec_model, positive=['delivery', 'fast'], negative=['late'], topn=5)
    print(f"  b) delivery - late + fast = {analogy_b}")
    
    print("\nEvaluating questions-words.txt benchmark on Word2Vec...")
    analogy_eval = evaluate_analogy_dataset(word2vec_model, "Assignment_10/questions-words.txt", max_per_section=150)
    print(f"Word2Vec Analogy Overall Accuracy: {analogy_eval['overall_accuracy']:.2f}% ({analogy_eval['total_correct']}/{analogy_eval['total_tested']})")
    
    # 6. Q3: Neural Network Classification
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    print("\n--- Q3: Neural Network Classification ---")
    print("Training SimpleNN with Word2Vec Mean Pooling...")
    mean_eval = train_and_evaluate_nn(train_mean, y_train, test_mean, y_test, hidden_dim=128, epochs=15, batch_size=64, lr=1e-3)
    print(f"Word2Vec Mean Pooling -> Test Accuracy: {mean_eval['accuracy']:.2f}%, Macro F1: {mean_eval['macro_f1']:.2f}%")
    print(mean_eval['report'])
    
    print("Training SimpleNN with Word2Vec TF-IDF Weighted Pooling...")
    tfidf_eval = train_and_evaluate_nn(train_tfidf, y_train, test_tfidf, y_test, hidden_dim=128, epochs=15, batch_size=64, lr=1e-3)
    print(f"Word2Vec TF-IDF Pooling -> Test Accuracy: {tfidf_eval['accuracy']:.2f}%, Macro F1: {tfidf_eval['macro_f1']:.2f}%")
    print(tfidf_eval['report'])
    
    results = {
        'model_name': 'Word2Vec-GoogleNews-300d',
        'vocab_size': len(word2vec_model),
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
    with open("Assignment_10/results/word2vec_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved Word2Vec results to Assignment_10/results/word2vec_results.json")
    
    # Release memory
    del word2vec_model
    gc.collect()
    print("Word2Vec model freed from memory.\n")

if __name__ == '__main__':
    run_word2vec()

