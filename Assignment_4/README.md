# Assignment 4 — N-gram Language Models & Laplace Smoothing

## Overview

This assignment builds **four n-gram language models** from the tokenized Telugu corpus (IndicCorpV2) produced in Assignment 1, and then applies **Laplace (Add-1) Smoothing** to handle unseen n-grams.

## Data

- **Source:** Tokenized Telugu sentences from Assignment 1 (`indiccorp_te_tokenized_part-00001.parquet`)
- **Total sentences:** 1,000,000
- **Split:**

| Set | Sentences |
|-----|-----------|
| Training | 998,000 |
| Development | 1,000 |
| Test | 1,000 |

## Question 1 — N-gram Language Models

**Notebook:** [`Question_1_NGram_Models.ipynb`](./Question_1_NGram_Models.ipynb)

Builds four language models from training data:

| Model | Order | Formula |
|-------|-------|---------|
| Unigram | 1 | P(wᵢ) = C(wᵢ) / N |
| Bigram | 2 | P(wᵢ \| wᵢ₋₁) = C(wᵢ₋₁, wᵢ) / C(wᵢ₋₁) |
| Trigram | 3 | P(wᵢ \| wᵢ₋₂, wᵢ₋₁) = C(wᵢ₋₂, wᵢ₋₁, wᵢ) / C(wᵢ₋₂, wᵢ₋₁) |
| Quadrigram | 4 | P(wᵢ \| wᵢ₋₃, wᵢ₋₂, wᵢ₋₁) = C(wᵢ₋₃, wᵢ₋₂, wᵢ₋₁, wᵢ) / C(wᵢ₋₃, wᵢ₋₂, wᵢ₋₁) |

Evaluates each model using **perplexity** on the development and test sets.

## Question 2 — Laplace (Add-1) Smoothing

**Notebook:** [`Question_2_Laplace_Smoothing.ipynb`](./Question_2_Laplace_Smoothing.ipynb)

Applies the Laplace smoothing formula:

```
P_Laplace(wₙ | w₁, w₂, ..., wₙ₋₁) = (C(w₁, w₂, ..., wₙ) + 1) / (C(w₁, w₂, ..., wₙ₋₁) + V)
```

where V is the vocabulary size. Compares smoothed vs. unsmoothed perplexity.

## How to Run

### Prerequisites

- Python 3.7+
- Jupyter Notebook / JupyterLab
- `pyarrow`, `tqdm`

### Steps

```bash
pip install pyarrow tqdm jupyter
cd Assignment_4
jupyter notebook
```

1. Run **`Question_1_NGram_Models.ipynb`** — cells execute top to bottom.
2. Run **`Question_2_Laplace_Smoothing.ipynb`** — cells execute top to bottom.

## File Structure

```
Assignment_4/
├── Question_1_NGram_Models.ipynb        # Q1: Unigram/Bigram/Trigram/Quadrigram models
├── Question_2_Laplace_Smoothing.ipynb   # Q2: Add-1 (Laplace) smoothing
└── README.md                            # This file
```

## References

- Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing*, Ch. 3 — N-gram Language Models.
