# NLP Lab Assignment 7: Part-of-Speech (POS) Tagging for Hindi

## Overview

This assignment designs and implements Part-of-Speech (POS) taggers for Hindi using both **Naive Bayes** and **Hidden Markov Models (HMM)** with multiple decoding strategies, evaluated on the Bureau of Indian Standards (BIS) POS-tagged Hindi dataset.

---

## Algorithms Implemented

### 1. Naive Bayes Classifier
Predicts the POS tag of each word sequentially using 5 features:
- **Length of word:** Binary feature (`"MORE"` if length $> 4$ else `"LESS"`)
- **Actual Word:** $w_i$
- **Previous Word:** $w_{i-1}$ (or `<s>` if start of sentence)
- **Next Word:** $w_{i+1}$ (or `</s>` if end of sentence)
- **Previous Tag:** $t_{i-1}$ (using the model's predicted tag $\hat{t}_{i-1}$ during inference)

Add-$\alpha$ / Laplace smoothing is applied across all feature distributions.

### 2. Hidden Markov Model (HMM)
A generative probabilistic sequence model $P(\mathbf{w}, \mathbf{t}) = P(t_1) P(w_1 \mid t_1) \prod_{i=2}^T P(t_i \mid t_{i-1}) P(w_i \mid t_i)$:
- **Standard Forward Viterbi:** Forward dynamic programming pass with backpointers, followed by backward traceback to reconstruct the global MAP sequence.
- **Backward Viterbi:** Symmetrical backward dynamic programming pass from $T-1$ down to $0$ with forward-pointers, followed by forward tracing from the initial state.
- **Forward-Backward Viterbi (Max-Product):** Combining forward trellis $\delta_t(i)$ and backward trellis $\beta'_t(i)$ to compute state max-marginals $\mu_t(i) = \delta_t(i) + \beta'_t(i)$ and decode $\arg\max_i \mu_t(i)$.
- **Forward-Backward Posterior Decoding (Sum-Product):** Forward-backward algorithm computing posterior marginals $P(t_t = i \mid \mathbf{w}) \propto \alpha_t(i) \beta_t(i)$ using `logsumexp`, optimizing per-token accuracy.

---

## Dataset

Hindi sentences tagged with the 24 Bureau of Indian Standards (BIS) POS tags:
- `hindi-train-pos.txt`: 15,087 sentences, 325,710 tokens, 18,279 unique words
- `hindi-dev-pos.txt`: 1,866 sentences, 40,762 tokens (OOV token rate: 4.19%)
- `hindi-test-pos.txt`: 1,910 sentences, 41,415 tokens (OOV token rate: 4.59%)

---

## Results Summary

### Overall Performance

| Algorithm | Split | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) | Speed (tokens/sec) |
|:---|:---|:---:|:---:|:---:|:---:|
| **Naive Bayes Classifier** | Dev | 91.02 | 74.36 | 90.96 | ~17,000 |
| **Naive Bayes Classifier** | Test | 91.04 | 76.00 | 90.98 | ~17,000 |
| **HMM Forward Viterbi** | Dev | 93.03 | 90.99 | 92.93 | ~70,000 |
| **HMM Forward Viterbi** | Test | 92.84 | 87.12 | 92.68 | ~90,000 |
| **HMM Backward Viterbi** | Dev | 93.03 | 90.99 | 92.93 | ~85,000 |
| **HMM Backward Viterbi** | Test | 92.84 | 87.12 | 92.68 | ~85,000 |
| **HMM FB Max-Product** | Dev | 93.03 | 90.99 | 92.93 | ~60,000 |
| **HMM FB Max-Product** | Test | 92.84 | 87.12 | 92.68 | ~60,000 |
| **HMM FB Posterior** | Dev | 93.03 | 90.68 | 92.93 | ~8,500 |
| **HMM FB Posterior** | Test | 92.92 | 86.73 | 92.81 | ~8,500 |

*Note: Forward Viterbi, Backward Viterbi, and FB Max-Product yield identical global MAP paths (100% token agreement).*

---

## Notebook Structure

The primary notebook is [`POS_Tagging_Hindi.ipynb`](./POS_Tagging_Hindi.ipynb):
1. **Imports & Setup:** Environment configuration, reproducible seeds, plot styling.
2. **Dataset Loading & EDA:** ConLL format parser, summary statistics, 24 BIS tag distribution, sample Hindi sentences.
3. **Naive Bayes POS Tagger:** Mathematical formulation, `NaiveBayesPOSTagger` class, training and evaluation.
4. **HMM POS Tagger:** Parameter estimation, log-probability matrices, `HMMPOSTagger` class supporting Forward Viterbi, Backward Viterbi, FB Max-Product, and FB Posterior Decoding.
5. **Per-Tag Evaluation:** Per-tag Precision, Recall, and F1-score tables on Dev and Test sets for all 24 tags.
6. **Visualizations:**
   - Grouped bar charts comparing per-tag F1-scores across all algorithms on Test and Dev sets.
   - Overall metrics and token throughput comparison.
   - Normalized confusion matrix heatmaps (Top 10 tags) for Naive Bayes and HMM.
7. **Qualitative Analysis:** Word-level predictions on test sentences comparing Gold vs Predicted tags.
8. **Discussion & Findings:** Mathematical analysis of forward vs backward Viterbi, sequence vs token-level optimality, error analysis on difficult tags (`N_NN` vs `N_NNP`, `JJ` vs `N_NN`).

---

## How to Run

```bash
cd Assignment_7
jupyter notebook POS_Tagging_Hindi.ipynb
```
Run all cells from top to bottom. All cell outputs, data tables, and high-resolution comparison charts are fully pre-computed and rendered in the notebook.
