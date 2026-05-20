"""
Course: Natural Language Processing
Academic Year: 2025-2026
Student Portfolio Submission

Lab 6 BERT Text Classification
"""


import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    print("Install transformers: pip install transformers")

# Sample sentiment dataset
SAMPLE_DATA = [
    ("This movie is absolutely fantastic!", 1),
    ("I loved every minute of it", 1),
    ("The best film I've seen all year", 1),
    ("Terrible waste of time", 0),
    ("Extremely disappointed with this", 0),
    ("Horrible acting and boring plot", 0),
    ("Amazing cinematography and storytelling", 1),
    ("Not worth watching", 0),
    ("Brilliant performance by the cast", 1),
    ("Completely unwatchable", 0),
] * 20

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text, max_length=self.max_length, truncation=True,
            padding='max_length', return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

print("
--- Starting Lab 6 BERT Text Classification ---")

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # 1. Load pretrained model
    print("\n[*] Loading pretrained bert model...")
    model_name = "distilbert-base-uncased"  # Using DistilBERT for efficiency
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)
    print(f"   ✓ Loaded {model_name}")
    
    # 2. Prepare data
    print("\n[*] Preparing data...")
    texts = [text for text, _ in SAMPLE_DATA]
    labels = [label for _, label in SAMPLE_DATA]
    
    split_idx = int(0.8 * len(texts))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    print(f"   Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # 3. Fine-tune
    print("\n[*] Fine-tuning model...")
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(3):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation
        model.eval()
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(labels.cpu().numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        val_accuracies.append(val_acc)
        print(f"   Epoch {epoch+1}/3 - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 4. Evaluate
    print("\n[*] Evaluation...")
    precision, recall, f1, _ = precision_recall_fscore_support(val_true, val_preds, average='weighted')
    print(f"   Accuracy:  {val_acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # 5. Visualizations
    print("\n[*] Saving visualizations...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('BERT Fine-tuning Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    print("    -> Saved: Saved training_curve.png")
    
    # Save results
    results = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [val_acc, precision, recall, f1]
    })
    results.to_csv("evaluation_results.csv", index=False)
    print("    -> Saved: Saved evaluation_results.csv")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure to install transformers: pip install transformers")

print("
--- Lab 6 BERT Text Classification Execution Finished ---")
