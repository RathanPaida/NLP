import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("Assignment_10/plots", exist_ok=True)
sns.set_theme(style="whitegrid", font="sans-serif")
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

# 1. Load results
with open("Assignment_10/results/glove_results.json", "r") as f:
    glove = json.load(f)
with open("Assignment_10/results/fasttext_results.json", "r") as f:
    fasttext = json.load(f)
with open("Assignment_10/results/word2vec_results.json", "r") as f:
    word2vec = json.load(f)

models_data = {
    'GloVe': glove,
    'FastText': fasttext,
    'Word2Vec': word2vec
}

# -------------------------------------------------------------------------
# Plot 1: Macro F1-Score and Accuracy Comparison (Mean vs TF-IDF)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
models = ['Word2Vec', 'FastText', 'GloVe']
x = np.arange(len(models))
width = 0.35

f1_mean = [models_data[m]['nn_mean']['macro_f1'] for m in models]
f1_tfidf = [models_data[m]['nn_tfidf']['macro_f1'] for m in models]

rects1 = ax.bar(x - width/2, f1_mean, width, label='Mean Pooling', color='#2b5c8f', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x + width/2, f1_tfidf, width, label='TF-IDF Weighted Pooling', color='#e26d5c', edgecolor='black', alpha=0.9)

ax.set_ylabel('Test Macro F1-Score (%)', fontsize=13, fontweight='bold')
ax.set_title('Test Macro F1-Score Comparison Across Word Embeddings & Pooling Strategies', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(['Word2Vec\n(Google News 300d)', 'FastText\n(Subwords 300d)', 'GloVe\n(Wikipedia+Gigaword 300d)'], fontsize=12)
ax.legend(frameon=True, fontsize=11, loc='lower right')
ax.set_ylim(70, 85)

for rect in rects1:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')
for rect in rects2:
    height = rect.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4), textcoords="offset points",
                ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("Assignment_10/plots/macro_f1_comparison.png")
plt.close()
print("Saved macro_f1_comparison.png")

# -------------------------------------------------------------------------
# Plot 2: Analogy Benchmark Accuracy
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
analogy_accs = [models_data[m]['analogy_benchmark']['overall_accuracy'] for m in models]
colors = ['#388E3C', '#1976D2', '#F57C00']

bars = ax.bar(models, analogy_accs, color=colors, edgecolor='black', width=0.5, alpha=0.88)
ax.set_ylabel('Analogy Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_title('Google Analogy Benchmark (questions-words.txt) Accuracy', fontsize=14, fontweight='bold', pad=15)
ax.set_xticklabels(['Word2Vec\n(Google News 300d)', 'FastText\n(Subwords 300d)', 'GloVe\n(6B 300d)'], fontsize=12)
ax.set_ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5), textcoords="offset points",
                ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("Assignment_10/plots/analogy_benchmark_comparison.png")
plt.close()
print("Saved analogy_benchmark_comparison.png")

# -------------------------------------------------------------------------
# Plot 3: Training Loss & Accuracy Learning Curves
# -------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

palette = {
    'Word2Vec (Mean)': ('#1b4965', '-'),
    'Word2Vec (TF-IDF)': ('#1b4965', '--'),
    'FastText (Mean)': ('#2a9d8f', '-'),
    'FastText (TF-IDF)': ('#2a9d8f', '--'),
    'GloVe (Mean)': ('#e76f51', '-'),
    'GloVe (TF-IDF)': ('#e76f51', '--'),
}

for name, key in [('Word2Vec', 'Word2Vec'), ('FastText', 'FastText'), ('GloVe', 'GloVe')]:
    h_m = models_data[key]['nn_mean']['history']
    h_t = models_data[key]['nn_tfidf']['history']
    epochs = range(1, len(h_m['train_loss']) + 1)
    
    col_m, ls_m = palette[f'{name} (Mean)']
    col_t, ls_t = palette[f'{name} (TF-IDF)']
    
    ax1.plot(epochs, h_m['train_loss'], label=f'{name} (Mean)', color=col_m, linestyle=ls_m, linewidth=2)
    ax1.plot(epochs, h_t['train_loss'], label=f'{name} (TF-IDF)', color=col_t, linestyle=ls_t, linewidth=2)
    
    ax2.plot(epochs, h_m['train_acc'], label=f'{name} (Mean)', color=col_m, linestyle=ls_m, linewidth=2)
    ax2.plot(epochs, h_t['train_acc'], label=f'{name} (TF-IDF)', color=col_t, linestyle=ls_t, linewidth=2)

ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax1.set_ylabel('Training Cross-Entropy Loss', fontsize=12, fontweight='bold')
ax1.set_title('Training Loss Dynamics', fontsize=13, fontweight='bold')
ax1.legend(frameon=True, fontsize=10)

ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax2.set_ylabel('Training Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_title('Training Accuracy Dynamics', fontsize=13, fontweight='bold')
ax2.legend(frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig("Assignment_10/plots/training_curves.png")
plt.close()
print("Saved training_curves.png")

# -------------------------------------------------------------------------
# Plot 4: Confusion Matrices
# -------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)

sub_models = [('Word2Vec', 'nn_mean', 'Word2Vec (Mean)', 0, 0),
              ('FastText', 'nn_mean', 'FastText (Mean)', 0, 1),
              ('GloVe', 'nn_mean', 'GloVe (Mean)', 0, 2),
              ('Word2Vec', 'nn_tfidf', 'Word2Vec (TF-IDF)', 1, 0),
              ('FastText', 'nn_tfidf', 'FastText (TF-IDF)', 1, 1),
              ('GloVe', 'nn_tfidf', 'GloVe (TF-IDF)', 1, 2)]

for m_key, strat, title, r, c in sub_models:
    cm = np.array(models_data[m_key][strat]['confusion_matrix'])
    ax = axes[r, c]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=11)
    ax.set_xlabel('Predicted Class', fontsize=11)

plt.tight_layout()
plt.savefig("Assignment_10/plots/confusion_matrices.png")
plt.close()
print("Saved confusion_matrices.png")

print("All plots generated successfully!")

