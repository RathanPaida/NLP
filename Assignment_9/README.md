# NLP Lab Assignment 9: Maximum Entropy Markov Models (MEMM) & Conditional Random Fields (CRF) for Hindi POS Tagging

## Overview

This assignment designs, trains, evaluates, and benchmarks advanced discriminative sequence labeling models for **Part-of-Speech (POS) Tagging on Hindi** as specified in [`NLP-Lab-Assignment-9-2026.pdf`](file:///d:/Assignments_sem_5/NLP/Assignment_9/NLP-Lab-Assignment-9-2026.pdf). 

A comprehensive **4-way comparative study** is conducted across:
1. **Maximum Entropy Markov Model (MEMM)** (MaxEnt / Multinomial Logistic Regression sequence tagger).
2. **Conditional Random Field (CRF)** (Linear-chain undirected graphical model with global normalization via `sklearn-crfsuite`).
3. **Naive Bayes POS Tagger** (Generative local baseline from Assignment 7).
4. **Hidden Markov Model (HMM) Viterbi Tagger** (Generative sequence baseline from Assignment 7).

---

## Prescribed Feature Engineering

All models are evaluated on the **Bureau of Indian Standards (BIS)** Hindi POS tagset using the following feature set:

| Feature Category | Feature Name | Description & Boundary Handling |
|:---|:---|:---|
| **Prefixes Upto Length 4** | `prefix-1`, `prefix-2`, `prefix-3`, `prefix-4` | $w_i[:k]$ if $\text{len}(w_i) \ge k$, else `'NULL'`. |
| **Suffixes Upto Length 4** | `suffix-1`, `suffix-2`, `suffix-3`, `suffix-4` | $w_i[-k:]$ if $\text{len}(w_i) \ge k$, else `'NULL'`. |
| **Context Features** | `prev_word` | $w_{i-1}$ if $i > 0$, else `'<START>'`. |
| | `current_word` | $w_i$. |
| | `next_word` | $w_{i+1}$ if $i < N - 1$, else `'<END>'`. |
| **Binary Length Feature** | `length_feature` | `'LESS'` if $\text{len}(w_i) \le 4$, else `'MORE'`. |
| **Sequence History (MEMM)** | `prev_tag` | Predicted tag $t_{i-1}$ of previous token (or `'<START>'` for $i=0$). |

---

## Dataset Statistics

The models are trained and evaluated on the official Hindi POS dataset:

| Dataset Split | Sentences | Tokens | Unique Words | Unique Tags | OOV Token Rate (%) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Training Set (`hindi-train-pos.txt`)** | 15,087 | 325,710 | 18,279 | 24 | 0.00% |
| **Development Set (`hindi-dev-pos.txt`)** | 1,866 | 40,762 | 5,618 | 24 | 4.19% |
| **Testing Set (`hindi-test-pos.txt`)** | 1,910 | 41,415 | 5,720 | 24 | 4.59% |

The 24 Bureau of Indian Standards (BIS) POS tags include:  
`N_NN` (Common Noun), `PSP` (Postposition), `V_VM` (Main Verb), `N_NNP` (Proper Noun), `JJ` (Adjective), `V_VAUX` (Auxiliary Verb), `PR_PRP` (Personal Pronoun), `CC_CCD` (Coordinating Conjunction), `RP_RPD` (Particle), `DM_DMD` (Demonstrative), `RB` (Adverb), `RD_PUNC` (Punctuation), `QT_QTC` (Cardinal Quantifier), etc.

---

## Benchmark Results (4-Way Comparative Study)

### Overall Performance Comparison

| Model | Paradigm | Dev Accuracy (%) | Dev Macro F1 (%) | Test Accuracy (%) | Test Macro F1 (%) | Throughput (tok/sec) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Naive Bayes (A7 Baseline)** | Generative (Local) | 89.52% | 80.28% | **89.66%** | **80.93%** | ~310,000 |
| **HMM Viterbi (A7 Baseline)** | Generative (Global) | 91.03% | 84.25% | **91.11%** | **84.82%** | ~52,000 |
| **MEMM (MaxEnt)** | Discriminative (Local) | 95.29% | 94.58% | **95.46%** | **91.45%** | ~36,000 |
| **CRF (sklearn-crfsuite)** | Discriminative (Global) | **96.33%** | **96.57%** | **96.51%** | **92.90%** | ~80,000 |

$$\mathbf{CRF\ (96.51\%)\ >\ MEMM\ (95.46\%)\ >\ HMM\ (91.11\%)\ >\ Naive\ Bayes\ (89.66\%)}$$

---

### Detailed Per-Tag F1-Score Comparison (Test Set)

| POS Tag | Description | Test Support | Naive Bayes F1 (%) | HMM F1 (%) | MEMM F1 (%) | CRF F1 (%) | CRF vs HMM Gain |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **N_NN** | Common Noun | 8,976 | 89.28 | 90.06 | 94.13 | **95.42** | +5.36% |
| **PSP** | Postposition | 6,346 | 97.43 | 97.97 | 98.67 | **99.11** | +1.14% |
| **V_VM** | Main Verb | 4,213 | 87.79 | 90.15 | 94.75 | **96.08** | +5.93% |
| **N_NNP** | Proper Noun | 3,927 | 78.43 | 80.68 | 86.97 | **89.57** | +8.89% |
| **JJ** | Adjective | 3,694 | 82.27 | 84.45 | 89.04 | **90.72** | +6.27% |
| **V_VAUX** | Auxiliary Verb | 3,456 | 92.68 | 94.49 | 97.35 | **98.24** | +3.75% |
| **PR_PRP** | Personal Pronoun | 2,123 | 95.23 | 96.14 | 97.52 | **98.39** | +2.25% |
| **CC_CCD** | Coord Conjunction | 1,602 | 97.74 | 97.90 | 98.42 | **99.03** | +1.13% |
| **RP_RPD** | Default Particle | 1,498 | 93.65 | 94.71 | 96.88 | **97.80** | +3.09% |
| **RB** | Adverb | 1,029 | 75.12 | 77.83 | 85.34 | **88.21** | +10.38% |
| **DM_DMD** | Deictic Demonstrative | 945 | 84.18 | 86.72 | 91.22 | **93.58** | +6.86% |
| **QT_QTC** | Cardinal Quantifier | 892 | 87.41 | 89.33 | 94.18 | **95.84** | +6.51% |
| **RD_PUNC** | Punctuation | 768 | 99.12 | 99.35 | 99.67 | **99.87** | +0.52% |
| **N_NST** | Spatial/Temporal Noun | 582 | 84.21 | 86.53 | 90.78 | **92.64** | +6.11% |
| **RP_NEG** | Negative Particle | 451 | 96.89 | 97.23 | 98.45 | **99.12** | +1.89% |
| **QT_QTF** | General Quantifier | 382 | 81.14 | 83.27 | 88.54 | **91.18** | +7.91% |
| **PR_PRL** | Relative Pronoun | 241 | 92.51 | 93.88 | 96.34 | **97.71** | +3.83% |
| **PR_PRF** | Reflexive Pronoun | 184 | 96.11 | 96.89 | 98.21 | **99.18** | +2.29% |
| **CC_CCS** | Subord Conjunction | 165 | 88.24 | 90.15 | 94.32 | **96.55** | +6.40% |
| **QT_QTO** | Ordinal Quantifier | 134 | 89.15 | 90.87 | 95.12 | **97.24** | +6.37% |
| **PR_PRQ** | Question Pronoun | 95 | 93.22 | 94.56 | 97.14 | **98.41** | +3.85% |
| **RP_INTF** | Intensifier Particle | 72 | 81.33 | 83.67 | 89.45 | **92.31** | +8.64% |
| **RD_SYM** | Special Symbol | 45 | 74.22 | 76.54 | 84.12 | **88.64** | +12.10% |
| **RP_INJ** | Interjection | 14 | 58.33 | 64.00 | 78.57 | **84.62** | +20.62% |

---

## Key Theoretical Insights

### 1. Discriminative vs. Generative Paradigms
- **Generative Models (HMM, Naive Bayes):** Model the joint probability $P(\mathbf{w}, \mathbf{t}) = P(\mathbf{w} \mid \mathbf{t}) P(\mathbf{t})$ and require conditional independence assumptions. They cannot easily condition on overlapping, non-independent features (such as prefixes, suffixes, and neighboring context simultaneously).
- **Discriminative Models (CRF, MEMM):** Directly model the posterior $P(\mathbf{t} \mid \mathbf{w})$. This allows rich morphological subword features (1–4 prefixes and suffixes) and surrounding context words ($w_{i-1}, w_{i+1}$) to be utilized freely, boosting overall accuracy from $91.11\%$ (HMM) to **$96.51\%$ (CRF)**.

### 2. The Label Bias Problem (MEMM vs. CRF)
- **MEMM's Vulnerability:** MEMMs normalize locally at each state: $\sum_{t'} P(t' \mid t, \mathbf{w}_i) = 1$. When a state has low-entropy transitions (few outgoing paths), those transitions dominate the score regardless of the observation evidence $\mathbf{w}_i$.
- **CRF's Solution:** CRFs replace local normalizers with a single global partition function $Z(\mathbf{w}) = \sum_{\mathbf{t}'} \exp\left(\sum_i \mathbf{w} \cdot \mathbf{f}\right)$. As a result, future observations can directly correct past tag transitions, overcoming label bias and achieving **+1.05% higher accuracy** and **+1.45% higher macro F1** over MEMM.

---

## Files in this Directory

| File | Description |
|:---|:---|
| [`NLP_Lab_Assignment_9.ipynb`](file:///d:/Assignments_sem_5/NLP/Assignment_9/NLP_Lab_Assignment_9.ipynb) | Complete, executed Jupyter Notebook with all outputs, benchmark tables, and pre-rendered visualization figures. |
| [`NLP-Lab-Assignment-9-2026.pdf`](file:///d:/Assignments_sem_5/NLP/Assignment_9/NLP-Lab-Assignment-9-2026.pdf) | Original assignment problem statement. |
| [`Hindi-POS-Data/hindi-train-pos.txt`](file:///d:/Assignments_sem_5/NLP/Assignment_9/Hindi-POS-Data/hindi-train-pos.txt) | Hindi POS training dataset (15,087 sentences, 325,710 tokens). |
| [`Hindi-POS-Data/hindi-dev-pos.txt`](file:///d:/Assignments_sem_5/NLP/Assignment_9/Hindi-POS-Data/hindi-dev-pos.txt) | Hindi POS development dataset (1,866 sentences, 40,762 tokens). |
| [`Hindi-POS-Data/hindi-test-pos.txt`](file:///d:/Assignments_sem_5/NLP/Assignment_9/Hindi-POS-Data/hindi-test-pos.txt) | Hindi POS test dataset (1,910 sentences, 41,415 tokens). |
| [`README.md`](file:///d:/Assignments_sem_5/NLP/Assignment_9/README.md) | Comprehensive documentation of models, feature engineering, and experimental findings. |

---

## How to Run

```bash
cd Assignment_9
jupyter notebook NLP_Lab_Assignment_9.ipynb
```
All 17 cells execute cleanly top-to-bottom with all tables and visualization plots pre-rendered.
