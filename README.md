# Natural Language Processing (AI357) — Lab Assignments

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)]()

This repository contains a comprehensive suite of programming assignments, mid-semester examination solutions, and research implementations developed for the **Natural Language Processing (AI357)** course at **Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat** (Department of Artificial Intelligence, 5th Semester).

The repository spans statistical language modeling, subword tokenization, finite state transducers, noisy channel spelling correction, and discriminative sequence labeling for Indian languages (Telugu, Hindi) and English.

---

## Assignments Index

| # | Assignment Title | Key Concepts & Algorithms | Dataset / Language | Accuracy / Metric | Link |
|:---:|:---|:---|:---|:---:|:---:|
| **1** | **Tokenizer Statistics** | Corpus analysis, Zipf's law, Heaps' law, Vocabulary distributions | IndicCorpV2 (Telugu, ~15 GB) | Corpus EDA | [View](./Assignment_1/) |
| **2** | **Text Segmentation** | Greedy Longest-Match, Dynamic Programming (Viterbi MAP) | Brown Corpus (English) | Exact match / WER | [View](./Assignment_2/) |
| **3** | **Finite Automata for NLP** | Deterministic Finite Automaton (DFA), Finite State Transducers (FST) | English Morphology | Morphological FST | [View](./Assignment_3/) |
| **4** | **N-Gram Language Models** | Unigram, Bigram, Trigram, Quadrigram, Laplace (Add-1) Smoothing | IndicCorpV2 Telugu (1M sents) | Perplexity | [View](./Assignment_4/) |
| **5** | **Add-K Smoothing** | Generalized Add-K Smoothing ($K=0.3$), Grid Search Optimization | IndicCorpV2 Telugu (1M sents) | Perplexity | [View](./Assignment_5/) |
| **6** | **Advanced N-Gram Smoothing** | Good-Turing, Jelinek-Mercer Interpolation, Katz Back-off, Kneser-Ney | IndicCorpV2 Telugu (1M sents) | Perplexity comparison | [View](./Assignment_6/) |
| **7** | **Hindi POS Tagging (NB & HMM)** | Naive Bayes, Bigram HMM (Forward, Backward, FB Max-Product, FB Posterior) | BIS Tagset (Hindi, 15k sents) | **92.84%** Test Acc | [View](./Assignment_7/) |
| **8** | **Midsem Solutions & Spell Correction** | Preprocessing, WordPiece, Monsoon Midsem Q1–Q5, Noisy Channel Model | Web 1T Unigrams, Peter Norvig | **77.32%** Top-1 (1-edit) | [View](./Assignment_8/) |
| **9** | **MEMM & CRF for Hindi POS Tagging** | Subword features, MEMM, Linear-Chain CRF (`sklearn-crfsuite`), 4-Way Study | BIS Tagset (Hindi, 15k sents) | **96.51%** Test Acc | [View](./Assignment_9/) |

---

## Detailed Assignment Summaries

### [Assignment 1 — Tokenizer Statistics on Telugu Corpus](./Assignment_1/)
- **Objective:** Linguistic and statistical profiling of the agglutinative Telugu language using IndicCorpV2 (~15 GB parquet shards).
- **Key Deliverables:**
  - Tokenization statistics across 83 partition shards.
  - Vocabulary frequency distributions and empirical validation of Zipf's Law ($f \propto 1/r$).
  - Subword vocabulary coverage and Heaps' law vocabulary growth modeling.
- **Notebooks:** [`Final_Tokenizer_Stats.ipynb`](./Assignment_1/Final_Tokenizer_Stats.ipynb), [`download_data.py`](./Assignment_1/download_data.py).

---

### [Assignment 2 — Text Segmentation via Dynamic Programming](./Assignment_2/)
- **Objective:** Recovering true word boundaries from concatenated (space-free) text strings using lexical unigram language models.
- **Algorithms:**
  - **Greedy Segmentation:** Longest-prefix matching heuristic.
  - **Dynamic Programming Segmentation:** Global MAP path decoding minimizing negative log-likelihood $\sum -\log P(w_i)$.
  - Quantitative evaluation on Brown Corpus subsets measuring exact match rate and word error rate (WER).
- **Notebooks:** [`greedy_segmentation.ipynb`](./Assignment_2/Lab-2_/Lab-2/greedy_segmentation.ipynb), [`dp_segmentation.ipynb`](./Assignment_2/Lab-2_/Lab-2/dp_segmentation.ipynb).

---

### [Assignment 3 — Finite Automata for Morphological Analysis](./Assignment_3/)
- **Objective:** Application of formal language theory and state transducers to natural language morphology.
- **Key Deliverables:**
  - **DFA:** Deterministic Finite Automaton parsing and accepting regular word syntax patterns.
  - **FST:** Finite State Transducer mapping between lexical surface forms and morphological glosses (`cat+N+PL` $\leftrightarrow$ `cats`).
  - Graphviz and Automathon state diagram visualizations.
- **Notebooks:** [`Question_1_DFA.ipynb`](./Assignment_3/Question_1_DFA.ipynb), [`Question_2_FST.ipynb`](./Assignment_3/Question_2_FST.ipynb).

---

### [Assignment 4 — N-Gram Language Models & Laplace Smoothing](./Assignment_4/)
- **Objective:** Building probabilistic n-gram language models from scratch on 1,000,000 Telugu sentences (998,000 train / 1,000 dev / 1,000 test).
- **Models Implemented:**
  - Unigram ($N=1$), Bigram ($N=2$), Trigram ($N=3$), and Quadrigram ($N=4$).
  - Laplace (Add-1) smoothing resolving zero-frequency conditional probabilities:
    $$P_{\text{Laplace}}(w_n \mid w_{1}^{n-1}) = \frac{C(w_{1}^n) + 1}{C(w_{1}^{n-1}) + V}$$
  - Perplexity evaluation across train, dev, and test splits.
- **Notebooks:** [`Question_1_NGram_Models.ipynb`](./Assignment_4/Question_1_NGram_Models.ipynb), [`Question_2_Laplace_Smoothing.ipynb`](./Assignment_4/Question_2_Laplace_Smoothing.ipynb).

---

### [Assignment 5 — Add-K Smoothing for Language Models](./Assignment_5/)
- **Objective:** Generalizing Add-1 smoothing to arbitrary fractional pseudo-counts ($K = 0.3$) to avoid over-allocating probability mass to unseen transitions.
- **Key Deliverables:**
  - Add-K smoothed n-gram formulations across all four orders.
  - Direct 3-way perplexity comparison: **MLE (Unsmoothed) vs. Laplace ($K=1.0$) vs. Add-K ($K=0.3$)**.
- **Notebook:** [`Add_K_Smoothing.ipynb`](./Assignment_5/Add_K_Smoothing.ipynb).

---

### [Assignment 6 — Advanced N-Gram Smoothing Techniques](./Assignment_6/)
- **Objective:** Comparative analysis of state-of-the-art discount-based and back-off language model smoothing algorithms on the Telugu corpus.
- **Smoothing Algorithms Implemented:**
  1. **Linear Interpolation (Jelinek-Mercer):** Convex combination of unigram, bigram, and trigram MLE probabilities.
  2. **Good-Turing Smoothing:** Re-estimating zero-count and low-count frequencies using frequency-of-frequencies $N_r$.
  3. **Katz Back-off:** Redistributing discounted mass using Good-Turing coefficients for seen n-grams and proportional lower-order backoff.
  4. **Stupid Back-off:** Scaled score heuristic ($\alpha = 0.4$) optimized for large-scale web corpora.
  5. **Interpolated Kneser-Ney:** Absolute discounting ($d = 0.75$) combined with continuation probabilities $P_{\text{cont}}(w)$.
- **Notebooks:** [`N-GRAM_SMOOTHING.ipynb`](./Assignment_6/N-GRAM_SMOOTHING.ipynb), [`code.ipynb`](./Assignment_6/code.ipynb).

---

### [Assignment 7 — Part-of-Speech Tagging for Hindi (Naive Bayes & HMM)](./Assignment_7/)
- **Objective:** Developing generative sequence taggers for Hindi evaluated on the 24 Bureau of Indian Standards (BIS) POS tags (15,087 train, 1,866 dev, 1,910 test sentences).
- **Key Implementations:**
  - **Naive Bayes POS Tagger:** Sequential classifier utilizing binary word length, current word, context window ($w_{i-1}, w_{i+1}$), and previous tag with Laplace smoothing (Test Acc: **91.04%**).
  - **Hidden Markov Model (HMM):** Full bigram transition and emission modeling with 4 dynamic programming decoders:
    - *Forward Viterbi:* Global MAP path decoding with backward backpointers (Test Acc: **92.84%**).
    - *Backward Viterbi:* Symmetrical backward dynamic programming with forward pointers.
    - *Forward-Backward Max-Product:* Trellis combination $\mu_t(i) = \delta_t(i) + \beta'_t(i)$.
    - *Forward-Backward Posterior Decoding:* Sum-product marginals optimizing per-token accuracy.
- **Notebook:** [`POS_Tagging_Hindi.ipynb`](./Assignment_7/POS_Tagging_Hindi.ipynb).

---

### [Assignment 8 — Midsem Exam Programming Solutions & Noisy Channel Spell Correction](./Assignment_8/)
- **Objective:** Providing end-to-end programming solutions to SVNIT Midsem Examination papers and implementing Peter Norvig's Bayesian Noisy Channel Model.
- **Section Overview:**
  - **Part 1A (Midsem Paper 1):**
    - Text preprocessing pipeline with regex special-token replacements (`URL`, `NUMBER`, `PUNCT`).
    - Log-normalized Term Frequency and Inverse Document Frequency engine (`compute_tf_idf_scores`).
    - WordPiece subword tokenization (20 merges) with exact likelihood-ratio scoring: $\text{score}(A, B) = \frac{\text{count}(A, B)}{\text{count}(A) \times \text{count}(B)}$.
    - Naive Bayes sentence classification (`Inform`, `Promo`, `Reminder`) with Add-K bigram language models.
  - **Part 1B (SVNIT Monsoon 2026 Midsem Paper):**
    - **Q1:** FST Morphological Analyzer for English regular noun plurals and E-insertion rule for sibilants ($s, x, z, ch, sh$).
    - **Q2:** WordPiece subword tokenization (2 merges) on Monsoon corpus & subword tokenization of `newest`.
    - **Q3:** TF-IDF on toy dataset & sentence closeness via pairwise cosine similarity (identifying S1 and S3 as closest: $0.8719$).
    - **Q4:** Naive Bayes spam classifier predicting email $E8 = (0, 0, 0)$ as `NotSpam` ($P = 93.43\%$).
    - **Q5:** Kneser-Ney language model smoothing with $d=0.5$ ($P_{KN}(\text{AI} \mid \text{love}) = 0.3750$, $P_{KN}(\text{NLP} \mid \text{like}) = 0.5833$).
  - **Part 2 (Statistical Noisy Channel Spell Correction):**
    - Bayesian formulation: $\hat{c} = \arg\max_c P(w \mid c) P(c)$.
    - Unigram prior $P(c)$ estimated from Google Web 1T corpus (`count_unigram.txt`, $N \approx 1.025 \times 10^{12}$ tokens).
    - Single-edit channel likelihood $P(w \mid c)$ estimated from empirical confusion counts (`count_1edit.txt`).
    - Evaluated on 1,000 real-world spelling error pairs (`spell-errors.txt`): **77.32% Top-1 accuracy** on 1-edit errors.
- **Notebook:** [`NLP_Lab_Assignment_8.ipynb`](./Assignment_8/NLP_Lab_Assignment_8.ipynb).

---

### [Assignment 9 — Maximum Entropy Markov Models (MEMM) & Conditional Random Fields (CRF)](./Assignment_9/)
- **Objective:** Advanced discriminative sequence labeling for Hindi POS tagging with prescribed morphological features and a rigorous 4-way comparative benchmark.
- **Prescribed Feature Set:**
  - Prefix lengths 1 to 4 (`NULL` if shorter).
  - Suffix lengths 1 to 4 (`NULL` if shorter).
  - Context words: $w_{i-1}$ (`<START>`), $w_i$, $w_{i+1}$ (`<END>`).
  - Binary length feature: `length_feature = 'LESS'` if $\text{len}(w_i) \le 4$, else `'MORE'`.
  - Sequential transition feature: $t_{i-1}$ (`<START>`).
- **Models Evaluated:**
  1. **Linear-Chain CRF (`sklearn-crfsuite`):** Global normalization avoiding label bias, optimized via L-BFGS with elastic-net penalties (**96.51% Test Accuracy, 92.90% Macro F1**).
  2. **MEMM (MaxEnt):** Decoupled vectorized logit precomputation and autoregressive sequential decoding (**95.46% Test Accuracy, 91.45% Macro F1**).
  3. **HMM Viterbi (Assignment 7 Baseline):** Generative sequence model (**91.11% Test Accuracy, 84.82% Macro F1**).
  4. **Naive Bayes (Assignment 7 Baseline):** Generative local model (**89.66% Test Accuracy, 80.93% Macro F1**).
- **Core Findings:**
  $$\mathbf{CRF\ (96.51\%)\ >\ MEMM\ (95.46\%)\ >\ HMM\ (91.11\%)\ >\ Naive\ Bayes\ (89.66\%)}$$
  - *Discriminative Advantage:* Direct modeling of $P(\mathbf{t} \mid \mathbf{w})$ allows subword prefix/suffix features to generalize seamlessly to Out-Of-Vocabulary Hindi tokens (+5.4% over HMM).
  - *Label Bias Resolution:* CRF's global partition function $Z(\mathbf{w})$ prevents states with low-entropy transitions from ignoring observation evidence, outperforming MEMM by **+1.05% Accuracy** and **+1.45% Macro F1**.
- **Notebook:** [`NLP_Lab_Assignment_9.ipynb`](./Assignment_9/NLP_Lab_Assignment_9.ipynb).

---

## Repository File Tree

```
NLP/
├── .gitignore
├── README.md                                           # Global repository documentation (This file)
│
├── Assignment_1/                                       # Tokenizer Statistics (Telugu Corpus)
│   ├── Final_Tokenizer_Stats.ipynb
│   ├── download_data.py
│   └── README.md
│
├── Assignment_2/                                       # Text Segmentation (Greedy & Dynamic Programming)
│   ├── Lab-2_/Lab-2/
│   │   ├── greedy_segmentation.ipynb
│   │   └── dp_segmentation.ipynb
│   └── README.md
│
├── Assignment_3/                                       # Finite Automata for NLP (DFA & FST)
│   ├── Question_1_DFA.ipynb
│   ├── Question_2_FST.ipynb
│   └── README.md
│
├── Assignment_4/                                       # N-Gram Language Models & Laplace Smoothing
│   ├── Question_1_NGram_Models.ipynb
│   ├── Question_2_Laplace_Smoothing.ipynb
│   └── README.md
│
├── Assignment_5/                                       # Add-K Smoothing
│   ├── Add_K_Smoothing.ipynb
│   └── README.md
│
├── Assignment_6/                                       # Advanced Smoothing (Good-Turing, Katz, Kneser-Ney)
│   ├── N-GRAM_SMOOTHING.ipynb
│   └── code.ipynb
│
├── Assignment_7/                                       # Hindi POS Tagging (Naive Bayes & HMM Viterbi)
│   ├── NLP_Lab_Assignment_7.pdf
│   ├── POS_Tagging_Hindi.ipynb
│   ├── hindi-train-pos.txt
│   ├── hindi-dev-pos.txt
│   ├── hindi-test-pos.txt
│   └── README.md
│
├── Assignment_8/                                       # Midsem Solutions & Noisy Channel Spell Correction
│   ├── NLP-Lab-Assignment-8-2026.pdf
│   ├── NLP-Midsem-Questions-Monsoon-2026.pdf           # SVNIT Monsoon 2026 Midsem Exam Paper (Q1–Q5)
│   ├── midsem_questions.pdf                            # Earlier Midsem Exam Paper (Q1–Q3)
│   ├── NLP_Lab_Assignment_8.ipynb                     # Executed notebook covering all questions + spell correction
│   ├── README.md
│   └── Spell-Correction/
│       ├── count_1edit.txt                             # Empirical single-edit confusion counts (Peter Norvig)
│       ├── count_unigram.txt                           # Google Web 1T Trillion unigram word counts
│       └── spell-errors.txt                            # 39,000+ real-world spelling error pairs
│
└── Assignment_9/                                       # MEMM & CRF for Hindi POS Tagging (4-Way Study)
    ├── NLP-Lab-Assignment-9-2026.pdf                   # Official assignment specification
    ├── NLP_Lab_Assignment_9.ipynb                     # Executed notebook with MEMM, CRF, plots, & benchmarks
    ├── README.md
    └── Hindi-POS-Data/
        ├── hindi-train-pos.txt                         # 15,087 sentences (325,710 tokens)
        ├── hindi-dev-pos.txt                           # 1,866 sentences (40,762 tokens)
        └── hindi-test-pos.txt                          # 1,910 sentences (41,415 tokens)
```

---

## Getting Started & Installation

### 1. Clone Repository
```bash
git clone https://github.com/RathanPaida/NLP.git
cd NLP
```

### 2. Environment Setup
Create and activate a virtual environment with Python 3.10+:
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter nbformat nbclient tqdm pyarrow
# Specifically for Assignment 9 (CRF)
pip install sklearn-crfsuite
# Specifically for Assignment 3 (Automata)
pip install automathon PyPDF2 graphviz
```

### 4. Running the Notebooks
Launch Jupyter Notebook and open any assignment folder:
```bash
jupyter notebook
```
Every notebook has been pre-executed with all cell outputs, confusion matrices, tables, and visualization charts saved and rendered.

---

## Academic Context

- **Course:** Natural Language Processing (AI357)
- **Institution:** Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat
- **Department:** Department of Artificial Intelligence
- **Semester:** 5th Semester, B.Tech Artificial Intelligence

---

## Author

**Rathan Paida**  
Department of Artificial Intelligence  
Sardar Vallabhbhai National Institute of Technology (SVNIT), Surat  
GitHub: [@RathanPaida](https://github.com/RathanPaida)
