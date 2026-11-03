import os
import gc
import json
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from gensim.models import KeyedVectors

from model_utils import (
    compute_corpus_embeddings, 
    evaluate_custom_analogy, 
    evaluate_analogy_dataset, 
    train_and_evaluate_nn
)

def run_experiment_for_model(model_name, model, train_df, test_df, tfidf_vec, analogy_path):
    print(f"\n{'='*30} Processing {model_name} {'='*30}")
    results = {}
    
    # Q1: Compute sentence representations
    print(f"[{model_name}] Computing sentence embeddings (Mean Pooling & TF-IDF Weighted Pooling)...")
    train_mean, train_tfidf = compute_corpus_embeddings(train_df, model, tfidf_vectorizer=tfidf_vec, dim=300)
    test_mean, test_tfidf = compute_corpus_embeddings(test_df, model, tfidf_vectorizer=tfidf_vec, dim=300)
    
    # Save embeddings to disk
    os.makedirs("Assignment_10/data/embeddings", exist_ok=True)
    np.save(f"Assignment_10/data/embeddings/{model_name.lower()}_mean_train.npy", train_mean)
    np.save(f"Assignment_10/data/embeddings/{model_name.lower()}_mean_test.npy", test_mean)
    np.save(f"Assignment_10/data/embeddings/{model_name.lower()}_tfidf_train.npy", train_tfidf)
    np.save(f"Assignment_10/data/embeddings/{model_name.lower()}_tfidf_test.npy", test_tfidf)
    print(f"[{model_name}] Sentence embeddings saved successfully.")
    
    # Q2: Word Analogies
    print(f"\n[{model_name}] Evaluating Target Word Analogies...")
    # a) good - bad + excellent =
    analogy_a = evaluate_custom_analogy(model, positive=['good', 'excellent'], negative=['bad'], topn=5)
    print(f"  a) good - bad + excellent = {analogy_a}")
    
    # b) delivery - late + fast =
    analogy_b = evaluate_custom_analogy(model, positive=['delivery', 'fast'], negative=['late'], topn=5)
    print(f"  b) delivery - late + fast = {analogy_b}")
    
    results['analogies'] = {
        'analogy_a': str(analogy_a),
        'analogy_b': str(analogy_b)
    }
    
    print(f"\n[{model_name}] Evaluating questions-words.txt benchmark...")
    analogy_eval = evaluate_analogy_dataset(model, analogy_path, max_per_section=150)
    print(f"[{model_name}] Analogy Benchmark Accuracy: {analogy_eval['overall_accuracy']:.2f}% (Tested: {analogy_eval['total_tested']}, Correct: {analogy_eval['total_correct']})")
    results['analogy_benchmark'] = analogy_eval
    
    # Q3: Neural Network Classification
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    print(f"\n[{model_name}] Training Neural Network with Mean Pooling representation...")
    mean_nn_eval = train_and_evaluate_nn(train_mean, y_train, test_mean, y_test, hidden_dim=128, epochs=15, batch_size=64, lr=1e-3)
    print(f"[{model_name} + Mean] Test Accuracy: {mean_nn_eval['accuracy']:.2f}%, Macro F1: {mean_nn_eval['macro_f1']:.2f}%")
    
    print(f"\n[{model_name}] Training Neural Network with TF-IDF Weighted Pooling representation...")
    tfidf_nn_eval = train_and_evaluate_nn(train_tfidf, y_train, test_tfidf, y_test, hidden_dim=128, epochs=15, batch_size=64, lr=1e-3)
    print(f"[{model_name} + TF-IDF] Test Accuracy: {tfidf_nn_eval['accuracy']:.2f}%, Macro F1: {tfidf_nn_eval['macro_f1']:.2f}%")
    
    # Strip un-serializable PyTorch model object before saving JSON
    results['nn_mean'] = {
        'accuracy': mean_nn_eval['accuracy'],
        'macro_precision': mean_nn_eval['macro_precision'],
        'macro_recall': mean_nn_eval['macro_recall'],
        'macro_f1': mean_nn_eval['macro_f1'],
        'confusion_matrix': mean_nn_eval['confusion_matrix'],
        'history': mean_nn_eval['history'],
        'report': mean_nn_eval['report']
    }
    results['nn_tfidf'] = {
        'accuracy': tfidf_nn_eval['accuracy'],
        'macro_precision': tfidf_nn_eval['macro_precision'],
        'macro_recall': tfidf_nn_eval['macro_recall'],
        'macro_f1': tfidf_nn_eval['macro_f1'],
        'confusion_matrix': tfidf_nn_eval['confusion_matrix'],
        'history': tfidf_nn_eval['history'],
        'report': tfidf_nn_eval['report']
    }
    
    return results

if __name__ == '__main__':
    print("Run experiments script initialized.")

