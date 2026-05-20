"""
Course: Natural Language Processing
Academic Year: 2025-2026
Student Portfolio Submission

Lab 3 BiLSTM Sequence Labeling
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
from typing import List, Tuple, Dict, Set
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import nltk
from nltk.corpus import brown
import warnings
warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')


class Vocabulary:
    """Build and manage vocabulary"""
    
    def __init__(self, pad_token='<PAD>', unk_token='<UNK>'):
        self.word2idx = {pad_token: 0, unk_token: 1}
        self.idx2word = {0: pad_token, 1: unk_token}
        self.tag2idx = {}
        self.idx2tag = {}
        self.word_count = Counter()
        self.pad_token = pad_token
        self.unk_token = unk_token
    
    def add_word(self, word: str):
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        self.word_count[word] += 1
    
    def add_tag(self, tag: str):
        if tag not in self.tag2idx:
            idx = len(self.tag2idx)
            self.tag2idx[tag] = idx
            self.idx2tag[idx] = tag
    
    def get_word_idx(self, word: str) -> int:
        return self.word2idx.get(word, self.word2idx[self.unk_token])
    
    def get_tag_idx(self, tag: str) -> int:
        return self.tag2idx.get(tag, 0)
    
    def build_from_sentences(self, sentences: List[List[Tuple[str, str]]], min_freq=1):
        """Build vocabulary from sentences"""
        # Count words
        for sentence in sentences:
            for word, tag in sentence:
                self.add_word(word)
                self.add_tag(tag)
        
        # Filter by frequency
        freq_words = {w for w, c in self.word_count.items() if c >= min_freq}
        self.word2idx = {
            self.pad_token: 0,
            self.unk_token: 1
        }
        self.idx2word = {0: self.pad_token, 1: self.unk_token}
        
        for i, word in enumerate(sorted(freq_words), start=2):
            self.word2idx[word] = i
            self.idx2word[i] = word
        
        print(f"✓ Vocabulary built: {len(self.word2idx)} words, {len(self.tag2idx)} tags")


class SequenceLabelingDataset(Dataset):
    """PyTorch Dataset for sequence labeling"""
    
    def __init__(self, sentences: List[List[Tuple[str, str]]], vocab: Vocabulary):
        self.sentences = sentences
        self.vocab = vocab
        self.word_sequences = []
        self.tag_sequences = []
        self.lengths = []
        
        for sentence in sentences:
            words, tags = zip(*sentence)
            word_idxs = [vocab.get_word_idx(w) for w in words]
            tag_idxs = [vocab.get_tag_idx(t) for t in tags]
            
            self.word_sequences.append(torch.tensor(word_idxs, dtype=torch.long))
            self.tag_sequences.append(torch.tensor(tag_idxs, dtype=torch.long))
            self.lengths.append(len(word_idxs))
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        return {
            'words': self.word_sequences[idx],
            'tags': self.tag_sequences[idx],
            'length': self.lengths[idx]
        }


class BiLSTMSequenceLabeler(nn.Module):
    """BiLSTM model for sequence labeling"""
    
    def __init__(self, vocab_size: int, num_tags: int, 
                 embedding_dim: int = 100, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3, 
                 pretrained_embeddings: torch.Tensor = None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        if pretrained_embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification layer
        self.fc = nn.Linear(hidden_dim * 2, num_tags)
    
    def forward(self, word_idxs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            word_idxs: (batch_size, seq_len)
            lengths: (batch_size,)
            
        Returns:
            logits: (batch_size, seq_len, num_tags)
        """
        # Embedding
        embedded = self.dropout(self.embedding(word_idxs))
        
        # Pack padded sequences
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), 
            batch_first=True, enforce_sorted=False
        )
        
        # BiLSTM
        lstm_output, _ = self.lstm(packed)
        
        # Unpack
        output, _ = pad_packed_sequence(lstm_output, batch_first=True)
        
        # Dropout + classification
        output = self.dropout(output)
        logits = self.fc(output)
        
        return logits


class BiLSTMTrainer:
    """Trainer for BiLSTM sequence labeling"""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in train_loader:
            words = batch['words'].to(self.device)
            tags = batch['tags'].to(self.device)
            lengths = batch['length'].to(self.device)
            
            optimizer.zero_grad()
            
            # Forward
            logits = self.model(words, lengths)  # (batch_size, seq_len, num_tags)
            
            # Reshape for loss
            logits_flat = logits.view(-1, logits.size(-1))
            tags_flat = tags.view(-1)
            
            # Loss
            loss = criterion(logits_flat, tags_flat)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_tags = []
        
        with torch.no_grad():
            for batch in val_loader:
                words = batch['words'].to(self.device)
                tags = batch['tags'].to(self.device)
                lengths = batch['length'].to(self.device)
                
                # Forward
                logits = self.model(words, lengths)
                
                # Loss
                logits_flat = logits.view(-1, logits.size(-1))
                tags_flat = tags.view(-1)
                loss = criterion(logits_flat, tags_flat)
                total_loss += loss.item()
                
                # Predictions
                preds = torch.argmax(logits, dim=-1)
                for pred, tag, length in zip(preds, tags, lengths):
                    predictions.extend(pred[:length].cpu().numpy())
                    true_tags.extend(tag[:length].cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(true_tags, predictions)
        self.val_losses.append(avg_loss)
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 10, lr: float = 0.001):
        """Train the model"""
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        best_acc = 0
        patience = 5
        patience_count = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.evaluate(val_loader, criterion)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | " +
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if val_acc > best_acc:
                best_acc = val_acc
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return best_acc
    
    def predict(self, test_loader: DataLoader) -> Tuple[List, List]:
        """Make predictions on test set"""
        self.model.eval()
        all_preds = []
        all_tags = []
        
        with torch.no_grad():
            for batch in test_loader:
                words = batch['words'].to(self.device)
                tags = batch['tags'].to(self.device)
                lengths = batch['length'].to(self.device)
                
                logits = self.model(words, lengths)
                preds = torch.argmax(logits, dim=-1)
                
                for pred, tag, length in zip(preds, tags, lengths):
                    all_preds.extend(pred[:length].cpu().numpy())
                    all_tags.extend(tag[:length].cpu().numpy())
        
        return all_preds, all_tags


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LAB 3: NEURAL SEQUENCE LABELING USING BiLSTM")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    print("\n[*] Loading training data...")
    sentences = brown.tagged_sents()[:500]
    print(f"   Loaded {len(sentences)} sentences")
    
    # Build vocabulary
    print("\n[*] Building vocabulary...")
    vocab = Vocabulary()
    vocab.build_from_sentences(sentences, min_freq=2)
    
    # Split data
    train_size = int(0.8 * len(sentences))
    train_sentences = sentences[:train_size]
    test_sentences = sentences[train_size:]
    
    # Create datasets
    print("\n[*] Creating datasets...")
    train_dataset = SequenceLabelingDataset(train_sentences, vocab)
    test_dataset = SequenceLabelingDataset(test_sentences, vocab)
    
    # Create dataloaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\n[*] Initializing bilstm model...")
    model = BiLSTMSequenceLabeler(
        vocab_size=len(vocab.word2idx),
        num_tags=len(vocab.tag2idx),
        embedding_dim=100,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {total_params:,}")
    
    # Train model
    print("\n[*] Training model...")
    trainer = BiLSTMTrainer(model, device=device)
    best_acc = trainer.train(train_loader, test_loader, epochs=15, lr=0.001)
    
    # Evaluate
    print("\n[*] Evaluating model...")
    all_preds, all_tags = trainer.predict(test_loader)
    
    accuracy = accuracy_score(all_tags, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_tags, all_preds, average='weighted', zero_division=0
    )
    
    print(f"\n   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Comparison with random embeddings
    print("\n[*] Testing with random embeddings...")
    model_random = BiLSTMSequenceLabeler(
        vocab_size=len(vocab.word2idx),
        num_tags=len(vocab.tag2idx),
        embedding_dim=100,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )
    trainer_random = BiLSTMTrainer(model_random, device=device)
    best_acc_random = trainer_random.train(train_loader, test_loader, epochs=15, lr=0.001)
    
    all_preds_random, _ = trainer_random.predict(test_loader)
    acc_random = accuracy_score(all_tags, all_preds_random)
    
    print(f"\n   Random embeddings accuracy: {acc_random:.4f}")
    print(f"   Learned embeddings accuracy: {accuracy:.4f}")
    print(f"   Improvement: {(accuracy - acc_random):.4f}")
    
    # Visualizations
    print("\n[*] Generating visualizations...")
    
    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(trainer.train_losses, label='Train Loss', marker='o')
    axes[0].plot(trainer.val_losses, label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Convergence')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(trainer.train_losses, label='Learned Embeddings', marker='o')
    axes[1].plot(trainer_random.train_losses, label='Random Embeddings', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Embedding Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("    -> Saved: Saved training_curves.png")
    
    # Per-tag accuracy
    per_tag_acc = {}
    for tag_idx in range(len(vocab.tag2idx)):
        mask = [t == tag_idx for t in all_tags]
        if sum(mask) > 0:
            correct = sum(p == t for p, t, m in zip(all_preds, all_tags, mask) if m)
            per_tag_acc[vocab.idx2tag[tag_idx]] = correct / sum(mask)
    
    plt.figure(figsize=(12, 6))
    tags = list(per_tag_acc.keys())
    accs = list(per_tag_acc.values())
    plt.bar(range(len(tags)), accs, color='skyblue')
    plt.xticks(range(len(tags)), tags, rotation=45)
    plt.ylabel('Accuracy')
    plt.title('Per-Tag Accuracy')
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig("per_tag_accuracy.png", dpi=150)
    print("    -> Saved: Saved per_tag_accuracy.png")
    
    # Save results
    print("\n[*] Saving results...")
    results = {
        'Model': ['BiLSTM (Learned)', 'BiLSTM (Random)'],
        'Accuracy': [accuracy, acc_random],
        'Precision': [precision, 0],
        'Recall': [recall, 0],
        'F1-Score': [f1, 0]
    }
    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)
    print("    -> Saved: Saved evaluation_results.csv")
    
    print("\n" + "="*70)
    print("Lab 3 Complete!")
    print("="*70)
