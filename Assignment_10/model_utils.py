import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

def get_sentence_representation(tokens, model, weights_dict=None, dim=300):
    """
    Compute sentence representation:
    - Mean pooling: weights_dict is None
    - TF-IDF weighted pooling: weights_dict provides word weights
    """
    if not tokens or len(tokens) == 0:
        return np.zeros(dim, dtype=np.float32)
    
    vecs = []
    weights = []
    for w in tokens:
        if w in model:
            vecs.append(model[w])
            if weights_dict is not None:
                weights.append(weights_dict.get(w, 1.0))
            else:
                weights.append(1.0)
                
    if len(vecs) == 0:
        return np.zeros(dim, dtype=np.float32)
    
    vecs = np.array(vecs, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)
    total_weight = np.sum(weights)
    if total_weight > 0:
        return np.sum(vecs * weights[:, None], axis=0) / total_weight
    else:
        return np.mean(vecs, axis=0)

def compute_corpus_embeddings(df, model, tfidf_vectorizer=None, dim=300):
    """
    Computes both mean and tfidf-weighted sentence embeddings for a dataframe.
    """
    mean_embeddings = []
    tfidf_embeddings = []
    
    # Pre-extract vocabulary idf or word weights if tfidf_vectorizer is available
    feature_names = None
    if tfidf_vectorizer is not None:
        feature_names = set(tfidf_vectorizer.vocabulary_.keys())
        idf_dict = dict(zip(tfidf_vectorizer.get_feature_names_out(), tfidf_vectorizer.idf_))
    else:
        idf_dict = None
        
    for idx, row in df.iterrows():
        token_str = row.get('tokens', '')
        if isinstance(token_str, str) and len(token_str) > 0:
            tokens = token_str.split(',')
        else:
            tokens = []
            
        # 1. Mean pooling
        mean_vec = get_sentence_representation(tokens, model, weights_dict=None, dim=dim)
        mean_embeddings.append(mean_vec)
        
        # 2. TF-IDF weighted pooling
        tfidf_vec = get_sentence_representation(tokens, model, weights_dict=idf_dict, dim=dim)
        tfidf_embeddings.append(tfidf_vec)
        
    return np.array(mean_embeddings, dtype=np.float32), np.array(tfidf_embeddings, dtype=np.float32)

def evaluate_custom_analogy(model, positive, negative, topn=5):
    """
    Solves analogies of type: a - b + c (positive=[a, c], negative=[b])
    """
    # check if all words in model
    for w in positive + negative:
        if w not in model:
            # try lowercase
            if w.lower() in model:
                positive = [p.lower() for p in positive]
                negative = [n.lower() for n in negative]
                break
            else:
                return f"Word '{w}' not in vocabulary."
            
    results = model.most_similar(positive=positive, negative=negative, topn=topn)
    return results

def evaluate_analogy_dataset(model, analogy_file_path, max_per_section=150):
    """
    Evaluates word analogies from questions-words.txt across categories.
    Subsamples per section if max_per_section is specified to ensure balanced, fast evaluation.
    """
    results = {}
    current_section = None
    section_data = {}
    
    with open(analogy_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(':'):
                current_section = line[1:].strip()
                section_data[current_section] = []
            else:
                words = line.split()
                if len(words) == 4 and current_section:
                    section_data[current_section].append(words)
                    
    total_correct = 0
    total_tested = 0
    section_metrics = {}
    
    for section, questions in section_data.items():
        if max_per_section and len(questions) > max_per_section:
            # Sample evenly
            step = len(questions) / max_per_section
            sampled_questions = [questions[int(i * step)] for i in range(max_per_section)]
        else:
            sampled_questions = questions
            
        sec_correct = 0
        sec_tested = 0
        
        for q in sampled_questions:
            # Format: a b c d  =>  a : b :: c : d  =>  b - a + c = d
            a, b, c, d = q
            
            # Case handling: check if lowercase works better (for GloVe/FastText)
            cand_a, cand_b, cand_c, cand_d = a, b, c, d
            if cand_a not in model and cand_a.lower() in model:
                cand_a, cand_b, cand_c, cand_d = a.lower(), b.lower(), c.lower(), d.lower()
                
            if cand_a in model and cand_b in model and cand_c in model:
                sec_tested += 1
                try:
                    preds = model.most_similar(positive=[cand_b, cand_c], negative=[cand_a], topn=1)
                    if preds and preds[0][0].lower() == cand_d.lower():
                        sec_correct += 1
                except Exception:
                    pass
                    
        acc = (sec_correct / sec_tested * 100.0) if sec_tested > 0 else 0.0
        section_metrics[section] = {
            'total_in_section': len(questions),
            'tested': sec_tested,
            'correct': sec_correct,
            'accuracy': acc
        }
        total_correct += sec_correct
        total_tested += sec_tested
        
    overall_acc = (total_correct / total_tested * 100.0) if total_tested > 0 else 0.0
    return {
        'overall_accuracy': overall_acc,
        'total_tested': total_tested,
        'total_correct': total_correct,
        'section_metrics': section_metrics
    }

class SimpleNN(nn.Module):
    """
    Simple Neural Network with a single hidden layer as prescribed:
    Linear(input_dim -> hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim -> output_dim)
    """
    def __init__(self, input_dim=300, hidden_dim=128, output_dim=2, dropout_prob=0.2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def train_and_evaluate_nn(X_train, y_train, X_test, y_test, hidden_dim=128, epochs=15, batch_size=64, lr=1e-3, device='cpu'):
    """
    Trains SimpleNN and computes test metrics (Accuracy, Macro Precision, Macro Recall, Macro F1).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Convert labels 1, 2 to 0, 1 if needed
    if np.min(y_train) == 1:
        y_train = y_train - 1
    if np.min(y_test) == 1:
        y_test = y_test - 1
        
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model = SimpleNN(input_dim=X_train.shape[1], hidden_dim=hidden_dim, output_dim=2, dropout_prob=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    history = {'train_loss': [], 'train_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)
            
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100.0
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        
    # Evaluate on test set
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = accuracy_score(all_targets, all_preds) * 100.0
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    cm = confusion_matrix(all_targets, all_preds)
    
    return {
        'model': model,
        'accuracy': acc,
        'macro_precision': prec * 100.0,
        'macro_recall': rec * 100.0,
        'macro_f1': f1 * 100.0,
        'confusion_matrix': cm.tolist(),
        'history': history,
        'report': classification_report(all_targets, all_preds, target_names=['Negative', 'Positive'], digits=4)
    }

