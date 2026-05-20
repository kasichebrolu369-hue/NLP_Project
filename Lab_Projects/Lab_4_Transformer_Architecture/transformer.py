"""
Course: Natural Language Processing
Academic Year: 2025-2026
Student Portfolio Submission

Lab 4 Transformer Architecture
"""


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
import math
import warnings
warnings.filterwarnings('ignore')


class PositionalEncoding(nn.Module):
    """Positional Encoding for Transformer"""
    
    def __init__(self, d_model: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.scale = 1.0 / math.sqrt(d_model)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, query, key, value, mask=None):
        # (batch_size, num_heads, seq_len_q, d_k)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = self.softmax(scores)
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention Mechanism"""
    
    def __init__(self, d_model: int, num_heads: int = 8):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        query = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, attn_weights = self.attention(query, key, value, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        
        # Final linear layer
        output = self.W_o(attn_output)
        
        return output, attn_weights


class FeedForwardNetwork(nn.Module):
    """Feed-Forward Network"""
    
    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))


class TransformerEncoderLayer(nn.Module):
    """Single Transformer Encoder Layer"""
    
    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, attn_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """Multi-layer Transformer Encoder"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, num_layers: int = 6,
                 num_heads: int = 8, d_ff: int = 2048, max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.d_model = d_model
        self.num_layers = num_layers
    
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        attention_weights = []
        for layer in self.layers:
            x, attn_w = layer(x, mask)
            attention_weights.append(attn_w)
        
        return x, attention_weights


class TextClassifier(nn.Module):
    """Transformer-based Text Classifier"""
    
    def __init__(self, vocab_size: int, num_classes: int, d_model: int = 512,
                 num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        
        self.transformer = TransformerEncoder(
            vocab_size, d_model, num_layers, num_heads
        )
        
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_classes)
        )
    
    def forward(self, x, mask=None):
        encoder_output, attention_weights = self.transformer(x, mask)
        # Use [CLS] token representation (first token)
        cls_representation = encoder_output[:, 0, :]
        logits = self.fc(cls_representation)
        return logits, attention_weights


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LAB 4: TRANSFORMER ARCHITECTURE FROM SCRATCH")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Parameters
    print("\n[*] Setting up transformer...")
    vocab_size = 1000
    d_model = 512
    num_heads = 8
    num_layers = 4
    num_classes = 4
    batch_size = 32
    seq_len = 128
    
    print(f"   Model parameters:")
    print(f"   - d_model: {d_model}")
    print(f"   - num_heads: {num_heads}")
    print(f"   - num_layers: {num_layers}")
    print(f"   - num_classes: {num_classes}")
    
    # Initialize model
    print("\n[*] Initializing model...")
    model = TextClassifier(vocab_size, num_classes, d_model, num_layers, num_heads)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Create dummy data
    print("\n[*] Creating dummy data...")
    X_train = torch.randint(0, vocab_size, (100, seq_len))
    y_train = torch.randint(0, num_classes, (100,))
    X_test = torch.randint(0, vocab_size, (20, seq_len))
    y_test = torch.randint(0, num_classes, (20,))
    
    train_loader = DataLoader(
        list(zip(X_train, y_train)), 
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        list(zip(X_test, y_test)), 
        batch_size=batch_size
    )
    
    # Train
    print("\n[*] Training model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses = []
    for epoch in range(10):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"   Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}")
    
    # Evaluate
    print("\n[*] Evaluating...")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits, _ = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)
    
    accuracy = correct / total
    print(f"   Test Accuracy: {accuracy:.4f}")
    
    # Visualize attention
    print("\n[*] Visualizing attention...")
    sample_input = X_test[:1].to(device)
    with torch.no_grad():
        _, attention_weights = model(sample_input)
    
    # Get attention from first layer, first head
    attn = attention_weights[0][0, 0, :, :].cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn[:20, :20], cmap='viridis', cbar=True)
    plt.title('Attention Weights (Layer 1, Head 1)')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.tight_layout()
    plt.savefig("attention_visualization.png", dpi=150)
    print("    -> Saved: Saved attention_visualization.png")
    
    # Training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, marker='o', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Transformer Training Convergence')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curve.png", dpi=150)
    print("    -> Saved: Saved training_curve.png")
    
    print("\n" + "="*70)
    print("Lab 4 Complete!")
    print("="*70)
