# Text Segmentation Assignment — NLP Lab 2

## Overview

This assignment implements two **text segmentation** techniques on unsegmented text (words concatenated without spaces) using a vocabulary derived from the **Brown corpus**. The goal is to recover the original word boundaries.

### Dataset

The file `text_segmentation_dataset.json` contains:

| Key | Description |
|-----|-------------|
| `metadata` | Vocabulary size (1500 words), number of test cases (1000), total corpus word count (735,040) |
| `word_counts` | Dictionary mapping each vocabulary word to its frequency in the Brown corpus |
| `test_cases` | 1000 test cases, each with an `input` (concatenated string), `ground_truth` (space-separated words), and `word_count` |

**Example test case:**
```
Input:        "itthatthecitytakestepstothisproblem"
Ground Truth: "it that the city take steps to this problem"
```

---

## Notebooks

### 1. `greedy_segmentation.ipynb` — Greedy (Longest Match) Approach

**Algorithm:**
1. Start from the beginning of the unsegmented string.
2. Try to match the **longest possible prefix** that exists in the vocabulary.
3. If a match is found, consume those characters and repeat from step 2.
4. If no match is found at any length, consume a single character as an unknown token.

**How it works (example):**
```
Input: "thecityis"

Step 1: Try "thecityis" (9 chars) → not in vocab
        Try "thecityi"  (8 chars) → not in vocab
        ...
        Try "the"       (3 chars) → MATCH! → output "the"
Step 2: Remaining: "cityis"
        Try "cityis"    (6 chars) → not in vocab
        ...
        Try "city"      (4 chars) → MATCH! → output "city"
Step 3: Remaining: "is"
        Try "is"        (2 chars) → MATCH! → output "is"

Result: "the city is"
```

**Strengths:**
- Simple and fast (linear scan)
- Works well when longest match is unambiguous

**Weaknesses:**
- No backtracking — a locally greedy choice can ruin subsequent segmentation
- Ignores word frequency — treats rare and common words equally
- Example failure: `"therein"` → greedy picks `"therein"` but ground truth might be `"there in"`

---

### 2. `dp_segmentation.ipynb` — Dynamic Programming Approach

**Algorithm:**
1. Compute **log probability** for each vocabulary word: `log₁₀(count / total_corpus_words)`
2. Use dynamic programming to find the segmentation that **maximises the total log probability**.

**DP Formulation:**
- Let `dp[i]` = maximum total log-probability for segmenting `text[0:i]`
- **Base case:** `dp[0] = 0` (empty string)
- **Recurrence:**
  ```
  dp[i] = max over all j where text[j:i] is in vocabulary:
           dp[j] + log P(text[j:i])
  ```
- A `backtrack` array records the start index of the last word, enabling reconstruction.

**How it works (example):**
```
Input: "therain"

Possible segmentations:
  "the rain"     → log P("the") + log P("rain") = -1.02 + -3.63 = -4.65
  "the r a in"   → much lower probability
  "therein"      → not in this vocabulary

DP selects: "the rain"  (highest log probability)
```

**Strengths:**
- Finds the **globally optimal** segmentation (not just locally greedy)
- Uses **word frequencies** — prefers common, natural words
- Effectively considers all possible segmentations in O(n × max_word_len) time

**Weaknesses:**
- Slightly slower than greedy (but still efficient)
- Relies on unigram probabilities (no word-order / bigram context)

---

## Evaluation Metrics

### 1. Accuracy (Exact Match)
```
Accuracy = (number of test cases where prediction == ground truth) / total test cases
```
A prediction is correct **only if** it matches the ground truth exactly (every word boundary is correct).

### 2. Word-Level Edit Distance (Levenshtein)
The **Levenshtein edit distance** between the predicted word list and the ground truth word list, where each operation (insert, delete, substitute) applies to a **whole word**.

```
Example:
  Prediction:   ["the", "city", "is", "good"]
  Ground Truth: ["the", "cit", "y", "is", "good"]
  Edit Distance: 2  (substitute "city"→"cit", insert "y")
```

The **average edit distance** across all test cases is reported.

---

## How to Run

### Prerequisites
- Python 3.7+
- Jupyter Notebook or JupyterLab
- `matplotlib` (for visualizations)

### Steps
1. Place `text_segmentation_dataset.json` in the same directory as the notebooks.
2. Open and run **`greedy_segmentation.ipynb`** — cells execute top to bottom.
3. Open and run **`dp_segmentation.ipynb`** — cells execute top to bottom.
4. Compare the results printed at the end of each notebook.

```bash
# Install dependencies (if not already installed)
pip install matplotlib jupyter

# Launch Jupyter
jupyter notebook
```

---

## Expected Results

| Metric | Greedy (Longest Match) | Dynamic Programming |
|--------|----------------------|---------------------|
| **Accuracy** | Lower | Higher |
| **Avg Edit Distance** | Higher | Lower |

The DP approach is expected to **outperform** the greedy approach because:
1. It considers all possible segmentations (not just the longest-first greedy choice).
2. It uses word frequency information to prefer common, natural words.
3. It guarantees the globally optimal segmentation under the unigram model.

---

## File Structure

```
Lab-2/
├── text_segmentation_dataset.json   # Dataset (vocabulary + test cases)
├── greedy_segmentation.ipynb        # Greedy longest-match approach
├── dp_segmentation.ipynb            # Dynamic programming approach
└── README.md                        # This file
```

---

## References

- Brown Corpus — Francis, W. N., & Kucera, H. (1979). *Brown Corpus Manual*. Brown University.
- Jurafsky, D., & Martin, J. H. (2024). *Speech and Language Processing*, Chapter 2 — Text Normalization and Word Segmentation.
