# Assignment 5 — Add-K Smoothing

## Overview

This assignment applies **Add-K Smoothing** (with $K = 0.3$) to the four n-gram language models developed in Assignment 4, and evaluates them on the development and test sets.

## Formula

$$P_{add\text{-}k}(w_i \mid w_{i-N+1}^{i-1}) = \frac{C(w_{i-N+1}^{i-1}, w_i) + k}{C(w_{i-N+1}^{i-1}) + k \cdot V}$$

where:
- $C(\cdot)$ = n-gram / context count from training data
- $k = 0.3$
- $V$ = vocabulary size

## Data

- **Source:** Tokenized Telugu sentences from Assignment 1
- **Total sentences:** 1,000,000
- **Split:** 998,000 train / 1,000 dev / 1,000 test (same split as Assignment 4)

## Notebook

| File | Description |
|------|-------------|
| [`Add_K_Smoothing.ipynb`](./Add_K_Smoothing.ipynb) | Add-K smoothing for Unigram, Bigram, Trigram, Quadrigram; perplexity evaluation; 3-way comparison (MLE vs. Laplace vs. Add-K) |

## How to Run

```bash
pip install pyarrow tqdm jupyter
cd Assignment_5
jupyter notebook Add_K_Smoothing.ipynb
```

Run all cells top to bottom.

## References

- Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing*, Ch. 3 — N-gram Language Models.
