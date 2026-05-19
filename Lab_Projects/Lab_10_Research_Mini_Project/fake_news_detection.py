"""
Lab 10: Research-Oriented NLP Mini Project - Fake News Detection
End-to-end research-grade NLP application with literature, baseline, advanced model, and analysis
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("LAB 10: RESEARCH PROJECT - FAKE NEWS DETECTION")
print("="*70)

# 1. Dataset preparation
print("\n1. PREPARING DATASET...")

# Sample fake news dataset
fake_news_data = [
    ("COVID vaccine causes autism", 1, "Health"),
    ("Study proves vaccines safe", 0, "Health"),
    ("Famous actor secretly arrested", 1, "Celebrity"),
    ("Actor announces new movie", 0, "Celebrity"),
    ("election rigged says candidate", 1, "Politics"),
    ("Official election results released", 0, "Politics"),
    ("Miracle cure found for cancer", 1, "Health"),
    ("New cancer treatment shows promise", 0, "Health"),
    ("Government hiding aliens", 1, "Conspiracy"),
    ("NASA releases new discovery", 0, "Science"),
] * 20

texts = [item[0] for item in fake_news_data]
labels = [item[1] for item in fake_news_data]
categories = [item[2] for item in fake_news_data]

X_train, X_test, y_train, y_test, cat_train, cat_test = train_test_split(
    texts, labels, categories, test_size=0.2, random_state=42
)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")
print(f"   True: {sum(y_train)}, Fake: {len(y_train) - sum(y_train)}")

# 2. Baseline model (Classical ML)
print("\n2. BASELINE MODEL - TF-IDF + RANDOM FOREST...")

vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
baseline_model.fit(X_train_tfidf, y_train)

baseline_preds = baseline_model.predict(X_test_tfidf)
baseline_acc = accuracy_score(y_test, baseline_preds)
baseline_precision, baseline_recall, baseline_f1, _ = precision_recall_fscore_support(
    y_test, baseline_preds, average='weighted'
)

print(f"   Accuracy:  {baseline_acc:.4f}")
print(f"   Precision: {baseline_precision:.4f}")
print(f"   Recall:    {baseline_recall:.4f}")
print(f"   F1-Score:  {baseline_f1:.4f}")

# 3. Advanced model (LSTM)
print("\n3. ADVANCED MODEL - LSTM CLASSIFIER...")

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab_size=1000, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.vocab = self._build_vocab()
    
    def _build_vocab(self):
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for text in self.texts:
            for word in text.lower().split():
                if word not in vocab and len(vocab) < self.vocab_size:
                    vocab[word] = len(vocab)
        return vocab
    
    def text_to_sequence(self, text):
        seq = []
        for word in text.lower().split():
            idx = self.vocab.get(word, self.vocab['<UNK>'])
            seq.append(idx)
        seq = seq[:self.max_len]
        seq += [0] * (self.max_len - len(seq))
        return seq[:self.max_len]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        seq = self.text_to_sequence(self.texts[idx])
        return torch.tensor(seq, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

class LSTMFakeNewsDetector(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        logits = self.fc(hidden)
        return logits

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare data
train_dataset = TextDataset(X_train, y_train)
test_dataset = TextDataset(X_test, y_test, vocab_size=1000)

# Copy vocabulary to test dataset
test_dataset.vocab = train_dataset.vocab

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model
advanced_model = LSTMFakeNewsDetector(len(train_dataset.vocab), embedding_dim=100, hidden_dim=128)
advanced_model = advanced_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(advanced_model.parameters(), lr=0.001)

print("   Training LSTM model...")
train_losses = []
for epoch in range(10):
    advanced_model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = advanced_model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    train_losses.append(total_loss / len(train_loader))
    if (epoch + 1) % 3 == 0:
        print(f"   Epoch {epoch+1}/10 - Loss: {train_losses[-1]:.4f}")

# Evaluate advanced model
advanced_model.eval()
advanced_preds = []
with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        logits = advanced_model(batch_x)
        preds = torch.argmax(logits, dim=1)
        advanced_preds.extend(preds.cpu().numpy())

advanced_acc = accuracy_score(y_test, advanced_preds)
advanced_precision, advanced_recall, advanced_f1, _ = precision_recall_fscore_support(
    y_test, advanced_preds, average='weighted'
)

print(f"\n   LSTM Results:")
print(f"   Accuracy:  {advanced_acc:.4f}")
print(f"   Precision: {advanced_precision:.4f}")
print(f"   Recall:    {advanced_recall:.4f}")
print(f"   F1-Score:  {advanced_f1:.4f}")

# 4. Comparative analysis
print("\n4. COMPARATIVE ANALYSIS...")

comparison_data = {
    'Model': ['Baseline (TF-IDF+RF)', 'Advanced (LSTM)'],
    'Accuracy': [baseline_acc, advanced_acc],
    'Precision': [baseline_precision, advanced_precision],
    'Recall': [baseline_recall, advanced_recall],
    'F1-Score': [baseline_f1, advanced_f1],
}

comparison_df = pd.DataFrame(comparison_data)
print("\n" + comparison_df.to_string(index=False))

improvement = (advanced_acc - baseline_acc) / baseline_acc * 100
print(f"\n   Improvement: {improvement:+.2f}%")

# 5. Error analysis
print("\n5. ERROR ANALYSIS...")

# Misclassified samples
test_dataset_eval = TextDataset(X_test, y_test)
baseline_errors = [(X_test[i], y_test[i], baseline_preds[i]) 
                   for i in range(len(y_test)) if baseline_preds[i] != y_test[i]]
advanced_errors = [(X_test[i], y_test[i], advanced_preds[i]) 
                   for i in range(len(y_test)) if advanced_preds[i] != y_test[i]]

print(f"\n   Baseline errors: {len(baseline_errors)}/{len(y_test)}")
print(f"   Advanced errors: {len(advanced_errors)}/{len(y_test)}")

if advanced_errors:
    print(f"\n   Example errors (Advanced Model):")
    for text, true_label, pred_label in advanced_errors[:3]:
        label_str = "Real" if true_label == 0 else "Fake"
        pred_str = "Real" if pred_label == 0 else "Fake"
        print(f"      Text: '{text[:50]}...'")
        print(f"      True: {label_str}, Predicted: {pred_str}")

# 6. Visualizations
print("\n6. GENERATING VISUALIZATIONS...")

# Model comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
baseline_vals = [baseline_acc, baseline_precision, baseline_recall, baseline_f1]
advanced_vals = [advanced_acc, advanced_precision, advanced_recall, advanced_f1]

axes[0, 0].bar(['Baseline', 'Advanced'], [baseline_acc, advanced_acc], color=['blue', 'green'])
axes[0, 0].set_title('Accuracy Comparison')
axes[0, 0].set_ylim([0, 1])

axes[0, 1].bar(['Baseline', 'Advanced'], [baseline_precision, advanced_precision], color=['blue', 'green'])
axes[0, 1].set_title('Precision Comparison')
axes[0, 1].set_ylim([0, 1])

axes[1, 0].bar(['Baseline', 'Advanced'], [baseline_recall, advanced_recall], color=['blue', 'green'])
axes[1, 0].set_title('Recall Comparison')
axes[1, 0].set_ylim([0, 1])

axes[1, 1].bar(['Baseline', 'Advanced'], [baseline_f1, advanced_f1], color=['blue', 'green'])
axes[1, 1].set_title('F1-Score Comparison')
axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
print("   ✓ Saved model_comparison.png")

# Training curve
plt.figure(figsize=(10, 6))
plt.plot(train_losses, marker='o', linewidth=2, color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LSTM Training Convergence')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
print("   ✓ Saved training_curve.png")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cm_baseline = confusion_matrix(y_test, baseline_preds)
cm_advanced = confusion_matrix(y_test, advanced_preds)

sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
axes[0].set_title('Baseline Model Confusion Matrix')
axes[0].set_ylabel('True')
axes[0].set_xlabel('Predicted')

sns.heatmap(cm_advanced, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
axes[1].set_title('Advanced Model Confusion Matrix')
axes[1].set_ylabel('True')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
print("   ✓ Saved confusion_matrices.png")

# 7. Save comprehensive results
print("\n7. SAVING RESULTS...")

comparison_df.to_csv("model_comparison_results.csv", index=False)
print("   ✓ Saved model_comparison_results.csv")

# Summary statistics
summary = pd.DataFrame({
    'Metric': ['Total Samples', 'Train Samples', 'Test Samples', 'Baseline Accuracy', 
               'Advanced Accuracy', 'Improvement %'],
    'Value': [len(texts), len(X_train), len(X_test), f"{baseline_acc:.4f}", 
              f"{advanced_acc:.4f}", f"{improvement:+.2f}%"]
})
summary.to_csv("summary_statistics.csv", index=False)
print("   ✓ Saved summary_statistics.csv")

print("\n" + "="*70)
print("Lab 10 Complete - Research Project Finished!")
print("="*70)
print("\nProject Output:")
print("  - Model comparison and evaluation")
print("  - Error analysis and misclassification patterns")
print("  - Visualizations (confusion matrices, training curves)")
print("  - CSV results for further analysis")
print("\nNext Steps:")
print("  - Literature review on fake news detection (15+ papers)")
print("  - Hyperparameter tuning and validation")
print("  - Cross-domain evaluation")
print("  - Write research paper (10-15 pages)")
print("  - Prepare presentation and viva")
