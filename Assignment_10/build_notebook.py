import os
import json
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        'kernelspec': {
            'display_name': 'Python 3',
            'language': 'python',
            'name': 'python3'
        },
        'language_info': {
            'name': 'python',
            'version': '3.10.20'
        }
    }
    
    cells = []
    
    # -------------------------------------------------------------------------
    # Cell 0: Header & Mathematical Foundations
    # -------------------------------------------------------------------------
    c0 = """# NLP Lab Assignment 10: Comparative Analysis of Word Embeddings (Word2Vec, GloVe, FastText) on Semantic Analogies & Neural Text Classification

**Course:** AI357 - Natural Language Processing  
**Institution:** Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat  
**Department:** Department of Artificial Intelligence  
**Semester:** 5th Semester  

---

## Executive Abstract & Theoretical Framework

This assignment presents a rigorous comparative evaluation of three landmark distributed word representation models:
1. **Google Word2Vec (Mikolov et al., 2013):** Continuous Skip-Gram with Negative Sampling trained on Google News (~100B tokens, 300d).
2. **GloVe (Pennington et al., 2014):** Global Vectors for Word Representation trained on Wikipedia 2014 + Gigaword 5 (6B tokens, 300d).
3. **FastText (Bojanowski et al., 2017):** Subword-enriched Skip-gram trained on Wikipedia 2017 + UMBC Webbase + Statmt.org News (16B tokens, 300d).

The investigation systematically evaluates three core aspects:
- **Q1: Data Preprocessing & Sentence Representations:** Clean the Amazon Reviews polarity corpus and compute sentence representations using:
  - Simple Average (Mean Pooling): $\\vec{s}_{\\text{mean}} = \\frac{1}{|W|} \\sum_{w \\in W} \\vec{v}_w$
  - TF-IDF Weighted Average: $\\vec{s}_{\\text{tfidf}} = \\frac{\\sum_{w \\in W} \\text{TF-IDF}(w, S) \\cdot \\vec{v}_w}{\\sum_{w \\in W} \\text{TF-IDF}(w, S)}$
- **Q2: Vector Arithmetic & Analogy Benchmarking:** Solve specific target analogies:
  - $\\text{good} - \\text{bad} + \\text{excellent} = ?$
  - $\\text{delivery} - \\text{late} + \\text{fast} = ?$
  And benchmark all models on the full Google Analogy Dataset (`questions-words.txt`, 14 syntactic and semantic categories).
- **Q3: Neural Network Sentiment Classification:** Design a single-hidden-layer PyTorch neural network ($300 \\to 128 \\to 2$) and benchmark performance across all 6 model-pooling combinations using **Macro F1-score**.
"""
    cells.append(nbf.v4.new_markdown_cell(c0))
    
    # -------------------------------------------------------------------------
    # Cell 1: Imports & Setup
    # -------------------------------------------------------------------------
    c1 = """# =========================================================================
# 1. System Configuration, Reproducibility Seed & Library Imports
# =========================================================================
import os
import re
import gc
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from gensim.models import KeyedVectors

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"PyTorch Version: {torch.__version__}")
print(f"Computation Device: {device}")
"""
    cells.append(nbf.v4.new_code_cell(c1))
    
    # -------------------------------------------------------------------------
    # Cell 2: Q1 Preprocessing Pipeline
    # -------------------------------------------------------------------------
    c2 = """---
## 2. Question 1: Dataset Preprocessing & Stratified Sampling

### Mathematical Preprocessing Specification:
1. **Case Normalization:** $c \\mapsto \\text{lower}(c)$ to unify orthographic variants.
2. **Punctuation & Digit Stripping:** Filtering out non-alphabetic characters $r'[^a-z\\s]'$.
3. **Lexical Tokenization:** Decomposing text sequences into whitespace-delimited tokens $S = (w_1, w_2, \\dots, w_N)$.
4. **Stopword Elimination:** Removing non-informative grammatical particles ($w_i \\notin \\mathcal{V}_{\\text{stopwords}}$).
5. **Stratified Sampling:** Sampling 10,000 instances from `train.csv` and 1,000 instances from `test.csv` while maintaining class balance ($51.01\\% / 48.99\\%$ positive/negative).
"""
    cells.append(nbf.v4.new_markdown_cell(c2))
    
    c3 = """# =========================================================================
# 2. Load Stratified Sampled Data & Inspect Tokenization Statistics
# =========================================================================
train_df = pd.read_csv("data/train_sampled.csv")
test_df = pd.read_csv("data/test_sampled.csv")

print(f"Sampled Training Set Shape: {train_df.shape}")
print(f"Sampled Testing Set Shape:  {test_df.shape}")

print("\\n--- Training Class Distribution ---")
print(train_df['label'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))

print("\\n--- Testing Class Distribution ---")
print(test_df['label'].value_counts(normalize=True).apply(lambda x: f"{x*100:.2f}%"))

# Tokenization & Vocabulary Statistics
train_tokens = [tok for t in train_df['tokens'].dropna() for tok in t.split(',')]
vocab_counter = Counter(train_tokens)
print(f"\\nTotal Tokens in Training Sample: {len(train_tokens):,}")
print(f"Unique Vocabulary Size: {len(vocab_counter):,}")
print(f"Average Tokens per Review: {train_df['token_count'].mean():.2f} (std: {train_df['token_count'].std():.2f})")

print("\\nTop 15 Most Frequent Content Words:")
for w, c in vocab_counter.most_common(15):
    print(f"  {w:12s} : {c:,}")
"""
    cells.append(nbf.v4.new_code_cell(c3))
    
    # -------------------------------------------------------------------------
    # Cell 4: Sentence Representation Formulation
    # -------------------------------------------------------------------------
    c4 = """---
## 3. Sentence Vector Representation Schemes

For a given review $S = (w_1, w_2, \\dots, w_k)$ and pre-trained embedding model $\\mathcal{M}$:

### A. Mean Pooling (Simple Average)
$$\\vec{s}_{\\text{mean}} = \\frac{1}{|W_{\\text{vocab}}|} \\sum_{w \\in W_{\\text{vocab}}} \\vec{v}_w$$
where $W_{\\text{vocab}} = \\{w \\in S \\mid w \\in \\text{Vocab}(\\mathcal{M})\\}$. If $|W_{\\text{vocab}}| = 0$, $\\vec{s} = \\mathbf{0}_{300}$.

### B. TF-IDF Weighted Pooling
Let $\\text{TF-IDF}(w, S) = \\text{TF}(w, S) \\times \\text{IDF}(w)$ be the term weight fitted over the training corpus:
$$\\vec{s}_{\\text{tfidf}} = \\frac{\\sum_{w \\in W_{\\text{vocab}}} \\text{TF-IDF}(w, S) \\cdot \\vec{v}_w}{\\sum_{w \\in W_{\\text{vocab}}} \\text{TF-IDF}(w, S)}$$
TF-IDF down-weights frequent domain terms (e.g., `'book'`, `'movie'`, `'one'`) while amplifying salient evaluative descriptors (e.g., `'phenomenal'`, `'defective'`, `'riveting'`).
"""
    cells.append(nbf.v4.new_markdown_cell(c4))
    
    c5 = """# =========================================================================
# 3. Verify Saved Sentence Representation Arrays
# =========================================================================
embedding_files = {
    'GloVe Mean Train': 'data/embeddings/glove_mean_train.npy',
    'GloVe TF-IDF Train': 'data/embeddings/glove_tfidf_train.npy',
    'FastText Mean Train': 'data/embeddings/fasttext_mean_train.npy',
    'FastText TF-IDF Train': 'data/embeddings/fasttext_tfidf_train.npy',
    'Word2Vec Mean Train': 'data/embeddings/word2vec_mean_train.npy',
    'Word2Vec TF-IDF Train': 'data/embeddings/word2vec_tfidf_train.npy',
}

for name, path in embedding_files.items():
    if os.path.exists(path):
        arr = np.load(path)
        print(f"{name:25s} -> Shape: {arr.shape}, Mean Norm: {np.linalg.norm(arr, axis=1).mean():.3f}, Non-Zero Rate: {np.mean(np.any(arr != 0, axis=1))*100:.2f}%")
"""
    cells.append(nbf.v4.new_code_cell(c5))
    
    # -------------------------------------------------------------------------
    # Cell 6: Q2 Word Analogies
    # -------------------------------------------------------------------------
    c6 = """---
## 4. Question 2: Word Analogies & Vector Arithmetic

### Vector Arithmetic (3CosAdd Formulation)
For analogy $a : b :: c : d$, the predicted word $\\hat{d}$ maximizes cosine similarity:
$$\\hat{d} = \\arg\\max_{x \\in \\mathcal{V} \\setminus \\{a, b, c\\}} \\cos(\\vec{v}_x, \\vec{v}_b - \\vec{v}_a + \\vec{v}_c)$$

We evaluate:
1. $\\text{good} - \\text{bad} + \\text{excellent} = ?$  
   Expected: Semantic superlative / high positive attribute (e.g., `superb`, `terrific`, `fantastic`).
2. $\\text{delivery} - \\text{late} + \\text{fast} = ?$  
   Expected: Speed / logistics concept (e.g., `faster`, `super-fast`, `deliveries`, `prompt`).
3. Google Analogy Benchmark (`questions-words.txt`): 14 syntactic and semantic categories tested across all 3 models.
"""
    cells.append(nbf.v4.new_markdown_cell(c6))
    
    c7 = """# =========================================================================
# 4. Display Analogy Results: Target Queries & Benchmark Accuracies
# =========================================================================
with open("results/word2vec_results.json", "r") as f:
    w2v_res = json.load(f)
with open("results/fasttext_results.json", "r") as f:
    ft_res = json.load(f)
with open("results/glove_results.json", "r") as f:
    glove_res = json.load(f)

print("=" * 80)
print("TARGET ANALOGY A: good - bad + excellent = ?")
print("=" * 80)
print(f"Word2Vec : {w2v_res['analogy_a']}")
print(f"GloVe    : {glove_res['analogy_a']}")
print(f"FastText : {ft_res['analogy_a']}")

print("\\n" + "=" * 80)
print("TARGET ANALOGY B: delivery - late + fast = ?")
print("=" * 80)
print(f"Word2Vec : {w2v_res['analogy_b']}")
print(f"GloVe    : {glove_res['analogy_b']}")
print(f"FastText : {ft_res['analogy_b']}")

print("\\n" + "=" * 80)
print("GOOGLE ANALOGY DATASET BENCHMARK (questions-words.txt)")
print("=" * 80)
print(f"{'Model Name':<28} | {'Vocabulary Size':<15} | {'Tested':<8} | {'Correct':<8} | {'Overall Accuracy':<16}")
print("-" * 85)
for res in [w2v_res, ft_res, glove_res]:
    bm = res['analogy_benchmark']
    print(f"{res['model_name']:<28} | {res['vocab_size']:<15,d} | {bm['total_tested']:<8} | {bm['total_correct']:<8} | {bm['overall_accuracy']:.2f}%")
"""
    cells.append(nbf.v4.new_code_cell(c7))
    
    # -------------------------------------------------------------------------
    # Cell 8: Category-wise Breakdown Table
    # -------------------------------------------------------------------------
    c8 = """# =========================================================================
# 4.1 Detailed Category-wise Analogy Benchmark Breakdown
# =========================================================================
sections = list(glove_res['analogy_benchmark']['section_metrics'].keys())

rows = []
for s in sections:
    g_acc = glove_res['analogy_benchmark']['section_metrics'][s]['accuracy']
    ft_acc = ft_res['analogy_benchmark']['section_metrics'][s]['accuracy']
    w_acc = w2v_res['analogy_benchmark']['section_metrics'][s]['accuracy']
    cat_type = "Semantic" if not s.startswith("gram") else "Syntactic"
    rows.append({
        'Section': s,
        'Type': cat_type,
        'Word2Vec (%)': w_acc,
        'FastText (%)': ft_acc,
        'GloVe (%)': g_acc,
        'Winner': 'FastText' if ft_acc >= max(w_acc, g_acc) else ('Word2Vec' if w_acc >= g_acc else 'GloVe')
    })

df_analogy_cat = pd.DataFrame(rows)
print(df_analogy_cat.to_string(index=False))
"""
    cells.append(nbf.v4.new_code_cell(c8))
    
    # -------------------------------------------------------------------------
    # Cell 9: Q3 Neural Network Architecture & Training
    # -------------------------------------------------------------------------
    c9 = """---
## 5. Question 3: Single-Hidden-Layer Neural Network Architecture

### Architectural Specification:
- **Input Dimension:** $D = 300$ (matching the embedding dimension of Word2Vec, GloVe, and FastText).
- **Hidden Layer:** Linear projection $\\mathbb{R}^{300} \\to \\mathbb{R}^{128}$, followed by non-linear Rectified Linear Unit (ReLU) activation:
  $$h = \\text{ReLU}(W_1 \\vec{s} + b_1), \\quad W_1 \\in \\mathbb{R}^{128 \\times 300}, \\; b_1 \\in \\mathbb{R}^{128}$$
- **Regularization:** Inverted Dropout with $p=0.2$ during training to prevent co-adaptation of hidden features.
- **Output Layer:** Linear projection $\\mathbb{R}^{128} \\to \\mathbb{R}^{2}$ generating unnormalized class logits $z$:
  $$z = W_2 h + b_2, \\quad W_2 \\in \\mathbb{R}^{2 \\times 128}, \\; b_2 \\in \\mathbb{R}^{2}$$
- **Objective Function:** Categorical Cross-Entropy Loss:
  $$\\mathcal{L} = -\\frac{1}{B} \\sum_{i=1}^B \\log \\left( \\frac{e^{z_{i, y_i}}}{\\sum_{k=1}^2 e^{z_{i, k}}} \\right)$$
- **Optimization:** Adam optimizer ($\\eta = 10^{-3}$, weight decay $= 10^{-4}$), batch size 64, trained for 15 epochs.
"""
    cells.append(nbf.v4.new_markdown_cell(c9))
    
    c10 = """# =========================================================================
# 5. Comparative Evaluation Table Across All 6 Neural Network Models
# =========================================================================
bench_data = []

configurations = [
    ('Word2Vec (Google News 300d)', 'Mean Pooling', w2v_res['nn_mean']),
    ('Word2Vec (Google News 300d)', 'TF-IDF Weighted', w2v_res['nn_tfidf']),
    ('FastText (Subwords 300d)', 'Mean Pooling', ft_res['nn_mean']),
    ('FastText (Subwords 300d)', 'TF-IDF Weighted', ft_res['nn_tfidf']),
    ('GloVe (6B 300d)', 'Mean Pooling', glove_res['nn_mean']),
    ('GloVe (6B 300d)', 'TF-IDF Weighted', glove_res['nn_tfidf'])
]

for model_name, pooling, m_eval in configurations:
    bench_data.append({
        'Model': model_name,
        'Pooling Strategy': pooling,
        'Test Accuracy (%)': f"{m_eval['accuracy']:.2f}%",
        'Macro Precision (%)': f"{m_eval['macro_precision']:.2f}%",
        'Macro Recall (%)': f"{m_eval['macro_recall']:.2f}%",
        'Macro F1-Score (%)': f"{m_eval['macro_f1']:.2f}%"
    })

bench_df = pd.DataFrame(bench_data)
print(bench_df.to_string(index=False))
"""
    cells.append(nbf.v4.new_code_cell(c10))
    
    # -------------------------------------------------------------------------
    # Cell 11: Display Visualizations
    # -------------------------------------------------------------------------
    c11 = """# =========================================================================
# 6. Publication-Quality Comparative Visualizations
# =========================================================================
from IPython.display import Image, display

print("--- Plot 1: Macro F1-Score Comparison Across Models & Pooling Strategies ---")
display(Image(filename="plots/macro_f1_comparison.png"))

print("\\n--- Plot 2: Google Analogy Benchmark Accuracy ---")
display(Image(filename="plots/analogy_benchmark_comparison.png"))

print("\\n--- Plot 3: Neural Network Training Dynamics (Loss & Accuracy) ---")
display(Image(filename="plots/training_curves.png"))

print("\\n--- Plot 4: Confusion Matrices Across All Models ---")
display(Image(filename="plots/confusion_matrices.png"))
"""
    cells.append(nbf.v4.new_code_cell(c11))
    
    # -------------------------------------------------------------------------
    # Cell 12: Discussion & Theoretical Conclusions
    # -------------------------------------------------------------------------
    c12 = """---
## 7. Comprehensive Discussion & Theoretical Insights

### 1. Word Analogy Performance & Subword Dynamics (Q2)
- **FastText's Dominance on Syntactic Analogies (84.00% overall):**
  FastText outperforms Word2Vec (70.86%) and GloVe (66.48%) by a substantial margin. This superiority is primarily driven by its **character $n$-gram subword embeddings** ($n \\in [3, 6]$). In categories like `gram1-adjective-to-adverb` (`quick` $\\to$ `quickly`, 70.7% vs 21.3% GloVe), `gram4-superlative` (`great` $\\to$ `greatest`, 98.7% vs 73.3%), and `gram5-present-participle` (`sing` $\\to$ `singing`, 96.7% vs 70.7%), morphological suffixes are encoded directly in the constituent $n$-grams.
- **Semantic Word Analogies:**
  For high-level semantic categories such as `capital-world` (GloVe 96.7%, FastText 94.7%, Word2Vec 80.0%) and `family` (GloVe 88.0%, Word2Vec 84.7%, FastText 84.0%), GloVe and FastText both demonstrate strong geometric alignment, while Word2Vec produces rich semantic synonyms (`good - bad + excellent` $\\to$ `terrific`, `superb`, `fantastic`).

### 2. Sentence Representations: Mean Pooling vs TF-IDF Weighted Pooling (Q1 & Q3)
- In all three embedding architectures, **Mean Pooling consistently outperformed TF-IDF Weighted Pooling** by $+1.70\\%$ (Word2Vec: $81.10\\%$ vs $79.38\\%$) and $+0.80\\%$ (FastText: $80.09\\%$ vs $79.29\\%$).
- *Linguistic Explanation:* In Amazon sentiment reviews, evaluative sentiment is distributed across multiple clause constituents (e.g., adverbs, auxiliary verbs, common adjectives like "good", "great", "well"). Because TF-IDF heavily penalizes words with high document frequency, it suppresses frequent sentiment carriers in favor of rare product-specific nouns (e.g., specific camera or book names), introducing variance and attenuating sentiment signal.

### 3. Embedding Architecture Comparison for Downstream Classification (Q3)
- **Word2Vec (Google News 300d)** achieved the highest downstream Macro F1-score (**81.10%**), followed closely by **FastText** (**80.09%**) and **GloVe** (**79.40%**).
- Word2Vec's training on a colossal 100-billion-word Google News corpus with negative sampling yields tightly clustered semantic neighborhoods for sentiment adjectives and adverbs.
- FastText achieves near-identical classification performance ($80.09\\%$) while guaranteeing zero Out-Of-Vocabulary failures through its subword architecture.
"""
    cells.append(nbf.v4.new_markdown_cell(c12))
    
    nb.cells = cells
    
    out_nb_path = "Assignment_10/NLP_Lab_Assignment_10.ipynb"
    with open(out_nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook at {out_nb_path} with {len(cells)} cells.")

if __name__ == '__main__':
    create_notebook()

