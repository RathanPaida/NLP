# NLP Lab Assignment 8: Midsem Solutions & Noisy Channel Spell Correction

## Overview

This assignment covers comprehensive programming solutions for natural language processing tasks as specified in [`NLP-Lab-Assignment-8-2026.pdf`](file:///d:/Assignments_sem_5/NLP/Assignment_8/NLP-Lab-Assignment-8-2026.pdf):
1. **Programming Solutions to Midsem Examination Questions (Paper 1 & Monsoon 2026 Paper)**.
2. **Statistical Noisy Channel Spell Correction Model** using Peter Norvig's error and language model corpora (`count_1edit.txt`, `count_unigram.txt`, evaluated on `spell-errors.txt`).

---

## Part 1A: Mid Semester Exam Solutions (Paper 1)

### Question 1: Text Preprocessing & Log-Normalized TF-IDF Engine
- **Preprocessing pipeline (`preprocess`)**:
  - URLs $\to$ `URL`
  - Numbers $\to$ `NUMBER`
  - Punctuations $\to$ `PUNCT`
  - Lowercase normalization
- **TF-IDF Computation**:
  - `compute_tf_with_normalization(sentence, vocab, smoothing=False)`: Normalized by word count, log-scaled: $\log(1 + \text{norm\_tf})$. Supports Add-1 Laplace smoothing.
  - `compute_idf(sentence, sentences, vocab, smoothing=False)`: Log-scaled IDF: $\log(N / DF)$ or smoothed $\log((N+1)/(DF+1)) + 1$.
  - `compute_tf_idf_scores(sentences)`: Computes TF-IDF matrix across the corpus.

### Question 2: WordPiece Subword Tokenization Algorithm
- **Dataset (5 sentences)**:
  - *The boy hugs the cat.*
  - *The boys are hugging the dogs.*
  - *The dogs are chasing the cats.*
  - *The dog and the cat sit quietly.*
  - *The boy is sitting on the dog.*
- **Algorithm**:
  - Lowercase and punctuation-based tokenization.
  - WordPiece subword prefix representation (`##` for continuation characters).
  - 20 merge iterations using exact likelihood-ratio scoring:
    $$\text{score}(A, B) = \frac{\text{count}(A, B)}{\text{count}(A) \times \text{count}(B)}$$
- **Test Sentence Tokenization**:
  - Input: *"The cat is chasing the dog quietly."*
  - Output Tokens: `['t', '##h', '##e', 'cat', 'is', 'c', '##h', '##a', '##s', '##in', '##g', 't', '##h', '##e', 'do', '##g', 'qui', '##e', '##tly']`

### Question 3: Sentence Classification with Naive Bayes & Add-K Smoothing
- **Training Set (9 sentences across 3 classes)**:
  - `Inform` (3 sentences)
  - `Promo` (3 sentences)
  - `Reminder` (3 sentences)
- **Features**:
  - Structural/binary and frequency features for `URL`, `NUMBER`, and `PUNCT`.
  - Bag-of-Words bigram representation.
  - Add-K language model bigram probabilities ($K=0.3$):
    $$P_{add-K}(w_2 \mid w_1, c) = \frac{\text{count}_c(w_1, w_2) + K}{\text{count}_c(w_1) + K \cdot |V|}$$
- **Classification**:
  - Test Sentence: *"You will get an exclusive offer in the meeting!"*
  - Prediction: **`PROMO`** (driven by strong promotional bigrams `exclusive offer`, `get free`).

---

## Part 1B: SVNIT Monsoon 2026 Midsem Exam Solutions (`NLP-Midsem-Questions-Monsoon-2026.pdf`)

### Question 1: Finite State Transducer (FST) Morphological Analyzer
- **Problem**: Design an FST for English noun plural morphology where regular nouns take suffix `-s`, while nouns ending in $s, x, z, ch, sh$ take suffix `-es`. Represent features as `+SG`, `+PL`, and `+N`.
- **Architecture**:
  - Morphotactic transducer (lexical $\to$ intermediate): `fox+N+PL` $\to$ `fox^s#`.
  - Orthographic rule transducer (intermediate $\to$ surface): E-insertion rule ($\epsilon \to e / \{s, x, z, ch, sh\} \text{\textasciicircum} \_ s\#$).
- **Implementation**:
  - Bidirectional `EnglishNounFST` class supporting Analysis (Surface $\to$ Lexical, e.g. `foxes` $\to$ `fox+N+PL`, `cat` $\to$ `cat+N+SG`) and Generation (Lexical $\to$ Surface).
  - Complete state transition table and execution trace across 10 test nouns (`cat`, `dog`, `bus`, `fox`, `buzz`, `watch`, `dish`).

### Question 2: WordPiece Subword Tokenization on Monsoon Corpus
- **Corpus (16 words)**: `low lower low lower lowest low lowest low lowest low new new new newer newer new`
- **Merge 1**: Candidate `('##s', '##t')` $\to$ `##st` (Count=3, Score=0.1111).
- **Merge 2**: Candidate `('l', '##o')` $\to$ `lo` (Count=10, Score=0.1000).
- **Tokenization of unseen word `newest`**:
  - Subword greedy longest match: `['n', '##e', '##w', '##e', '##st']`.

### Question 3: TF-IDF on Toy Dataset & Sentence Closeness
- **Sentences**:
  - S1: `cat dog animal`
  - S2: `cat cat pet`
  - S3: `dog pet animal`
  - S4: `cat dog pet`
- **TF**: Raw counts; **IDF**: Unsmoothed $\log(N / DF)$.
- **Pairwise Cosine Similarity**:
  - S1 & S3: **0.8719** (Natural Log) / **0.8660** (Log10)
  - S1 & S4: 0.6975
  - S3 & S4: 0.6975
  - S2 & S4: 0.5369
- **Closest Sentence Pair**: **S1 (`cat dog animal`) and S3 (`dog pet animal`)**.

### Question 4: Naive Bayes Email Spam Classifier on Binary Features
- **Dataset**: 7 emails classified as `Spam` (4) or `NotSpam` (3) on binary features `free`, `win`, `offer`.
- **Query Email E8**: Features `(free=0, win=0, offer=0)`.
- **Results**:
  - Unsmoothed Posterior: $P(\text{NotSpam} \mid E8) = 93.43\%$, $P(\text{Spam} \mid E8) = 6.57\%$.
  - Laplace (Add-1) Posterior: $P(\text{NotSpam} \mid E8) = 81.39\%$, $P(\text{Spam} \mid E8) = 18.61\%$.
  - **Prediction**: **`NotSpam`** under both unsmoothed and smoothed formulations.

### Question 5: Kneser-Ney Language Model Smoothing ($d = 0.5$)
- **Corpus (5 sentences)**: `I like NLP`, `I like AI`, `You like NLP`, `You love AI`, `We love NLP`.
- **Continuation Probabilities**:
  - $P_{\text{cont}}(\text{AI}) = |\{\text{like}, \text{love}\}| / 8 = 2/8 = 0.2500$
  - $P_{\text{cont}}(\text{NLP}) = |\{\text{like}, \text{love}\}| / 8 = 2/8 = 0.2500$
- **Backoff Weights**:
  - $\lambda(\text{love}) = (0.5 / 2) \times 2 = 0.5000$
  - $\lambda(\text{like}) = (0.5 / 3) \times 2 = 1/3 \approx 0.3333$
- **Evaluations**:
  - **$P_{KN}(\text{AI} \mid \text{love}) = \frac{\max(1 - 0.5, 0)}{2} + 0.5 \times 0.25 = 0.2500 + 0.1250 = \mathbf{0.3750}$ ($3/8$)**
  - **$P_{KN}(\text{NLP} \mid \text{like}) = \frac{\max(2 - 0.5, 0)}{3} + \frac{1}{3} \times 0.25 = 0.5000 + 0.0833 = \mathbf{0.5833}$ ($7/12$)**

---

## Part 2: Noisy Channel Spell Correction Model

### Mathematical Formulation
$$\hat{c} = \arg\max_{c \in \mathcal{V}} P(c \mid w) = \arg\max_{c \in \mathcal{V}} P(w \mid c) \cdot P(c)$$

1. **Language Model Prior $P(c)$**:
   - Estimated from `count_unigram.txt` (Google Web 1T Unigram corpus, 333,333 words, $N \approx 1.025 \times 10^{12}$ tokens).
   - Unknown word length backoff: $P(c) = \frac{10}{N \cdot 10^{\text{len}(c)}}$.
2. **Channel Error Likelihood $P(w \mid c)$**:
   - Single-edit confusion counts from `count_1edit.txt` ($N_{\text{edits}} = 39,061$) covering insertions, deletions, substitutions, and transpositions.
3. **Candidate Search**:
   - Efficient prefix-tree search across 1-edit and 2-edit candidate words.

### Quantitative Evaluation on `spell-errors.txt` (1,000 Test Pairs)

| Evaluation Subset | Sample Count | Top-1 Accuracy (%) | Top-3 Accuracy (%) | Top-5 Accuracy (%) |
|:---|:---:|:---:|:---:|:---:|
| **1-Edit Distance Errors** | 366 | **77.32%** | **91.53%** | **94.26%** |
| **Overall Corpus (All Edit Distances)** | 1,000 | **36.80%** | **42.80%** | **43.80%** |

---

## Files in this Directory

| File | Description |
|:---|:---|
| [`NLP_Lab_Assignment_8.ipynb`](file:///d:/Assignments_sem_5/NLP/Assignment_8/NLP_Lab_Assignment_8.ipynb) | Complete, executed Jupyter Notebook with all outputs, tables, and visualization figures. |
| [`NLP-Lab-Assignment-8-2026.pdf`](file:///d:/Assignments_sem_5/NLP/Assignment_8/NLP-Lab-Assignment-8-2026.pdf) | Original assignment problem statement. |
| [`NLP-Midsem-Questions-Monsoon-2026.pdf`](file:///d:/Assignments_sem_5/NLP/Assignment_8/NLP-Midsem-Questions-Monsoon-2026.pdf) | SVNIT Monsoon 2026 Midsem exam paper (Questions 1 to 5). |
| [`midsem_questions.pdf`](file:///d:/Assignments_sem_5/NLP/Assignment_8/midsem_questions.pdf) | Earlier Midsem exam question paper (Questions 1 to 3). |
| [`Spell-Correction/count_unigram.txt`](file:///d:/Assignments_sem_5/NLP/Assignment_8/Spell-Correction/count_unigram.txt) | Unigram word frequencies from Google Web Trillion corpus. |
| [`Spell-Correction/count_1edit.txt`](file:///d:/Assignments_sem_5/NLP/Assignment_8/Spell-Correction/count_1edit.txt) | 1-edit distance confusion rules and counts. |
| [`Spell-Correction/spell-errors.txt`](file:///d:/Assignments_sem_5/NLP/Assignment_8/Spell-Correction/spell-errors.txt) | Real-world spelling error pairs dataset. |
| [`README.md`](file:///d:/Assignments_sem_5/NLP/Assignment_8/README.md) | Comprehensive documentation of methods, derivations, and experimental results. |

---

## How to Run

```bash
cd Assignment_8
jupyter notebook NLP_Lab_Assignment_8.ipynb
```
All 27 cells execute top-to-bottom cleanly with pre-rendered tables and figures.
