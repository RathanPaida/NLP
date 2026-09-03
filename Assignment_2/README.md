# Assignment 2 — Text Segmentation

## Overview

This assignment implements two **text segmentation** algorithms to recover word boundaries from unsegmented (concatenated) text. A vocabulary of 1,500 words derived from the **Brown Corpus** is used to guide the segmentation.

## Problem Statement

Given a string with no spaces (e.g., `"itthatthecitytakestepstothisproblem"`), recover the original words (e.g., `"it that the city take steps to this problem"`).

## Approaches

### 1. Greedy Segmentation (Longest Match First)

- Scans left-to-right, always consuming the **longest vocabulary match** at the current position.
- **Pros:** Simple, fast (linear time).
- **Cons:** No backtracking — a greedy choice can cascade into incorrect downstream splits.

### 2. Dynamic Programming Segmentation

- Computes **log-probabilities** from word frequencies and finds the segmentation that **maximises total log-probability** using DP.
- **Recurrence:** `dp[i] = max over j { dp[j] + log P(text[j:i]) }` for all valid vocabulary words `text[j:i]`.
- **Pros:** Globally optimal under the unigram model; frequency-aware.
- **Cons:** Slightly slower (still efficient at `O(n × max_word_len)`).

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Exact-Match Accuracy** | Fraction of test cases where prediction exactly matches ground truth |
| **Word-Level Edit Distance** | Levenshtein distance between predicted and ground-truth word lists (averaged over all test cases) |

## How to Run

### Prerequisites
- Python 3.7+
- Jupyter Notebook / JupyterLab
- `matplotlib`

### Steps

```bash
pip install matplotlib jupyter
jupyter notebook
```

1. Navigate to `Lab-2_/Lab-2/`.
2. Run **`greedy_segmentation.ipynb`** — cells execute top to bottom.
3. Run **`dp_segmentation.ipynb`** — cells execute top to bottom.
4. Compare the accuracy and edit-distance results printed at the end of each notebook.

## File Structure

```
Assignment_2/
├── Lab-2_/
│   ├── Lab-2/
│   │   ├── text_segmentation_dataset.json   # Vocabulary + 1000 test cases
│   │   ├── greedy_segmentation.ipynb        # Greedy longest-match approach
│   │   ├── dp_segmentation.ipynb            # Dynamic programming approach
│   │   ├── add_output_cell.py               # Helper script
│   │   ├── NLP-Lab2-Assignment.pdf          # Assignment specification
│   │   └── README.md                        # Detailed README
│   └── Lab-2_.zip                           # Archived submission
└── README.md                                # This file
```

## References

- Brown Corpus — Francis, W. N., & Kucera, H. (1979). *Brown Corpus Manual*. Brown University.
- Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing*, Ch. 2 — Text Normalization and Word Segmentation.
