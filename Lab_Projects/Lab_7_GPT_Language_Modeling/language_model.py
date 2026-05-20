"""
Course: Natural Language Processing
Academic Year: 2025-2026
Student Portfolio Submission

Lab 7 GPT Language Modeling
"""


import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Custom GPT model
class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        logits = self.fc(lstm_out)
        return logits

class LanguageGenerator:
    def __init__(self, model, vocab_size, device='cpu', temperature=1.0):
        self.model = model
        self.vocab_size = vocab_size
        self.device = device
        self.temperature = temperature
    
    def greedy_decoding(self, prompt, max_length=50):
        """Generate text using greedy decoding"""
        self.model.eval()
        generated = prompt.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                x = torch.tensor([generated], dtype=torch.long).to(self.device)
                logits = self.model(x)
                next_token = torch.argmax(logits[0, -1, :]).item()
                generated.append(next_token)
                
                if next_token == 0:  # EOS token
                    break
        
        return generated
    
    def top_k_sampling(self, prompt, max_length=50, k=5):
        """Generate text using top-k sampling"""
        self.model.eval()
        generated = prompt.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                x = torch.tensor([generated], dtype=torch.long).to(self.device)
                logits = self.model(x) / self.temperature
                probs = torch.softmax(logits[0, -1, :], dim=0)
                
                top_k_probs, top_k_indices = torch.topk(probs, k)
                top_k_probs /= top_k_probs.sum()
                
                next_token = top_k_indices[torch.multinomial(top_k_probs, 1)].item()
                generated.append(next_token)
                
                if next_token == 0:
                    break
        
        return generated
    
    def nucleus_sampling(self, prompt, max_length=50, p=0.9):
        """Generate text using nucleus (top-p) sampling"""
        self.model.eval()
        generated = prompt.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                x = torch.tensor([generated], dtype=torch.long).to(self.device)
                logits = self.model(x) / self.temperature
                probs = torch.softmax(logits[0, -1, :], dim=0)
                
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=0)
                
                sorted_indices_to_remove = cumsum_probs > p
                sorted_indices_to_remove[0] = False
                
                sorted_probs[sorted_indices_to_remove] = 0
                sorted_probs /= sorted_probs.sum()
                
                next_token = sorted_indices[torch.multinomial(sorted_probs, 1)].item()
                generated.append(next_token)
                
                if next_token == 0:
                    break
        
        return generated

print("
--- Starting Lab 7 GPT Language Modeling ---")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# 1. Setup
print("\n[*] Setting up language model...")
vocab_size = 1000
embedding_dim = 128
hidden_dim = 256
num_layers = 2

model = SimpleGPT(vocab_size, embedding_dim, hidden_dim, num_layers)
model = model.to(device)
print(f"   ✓ Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

# 2. Training (dummy training on random data)
print("\n[*] Training language model...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train = torch.randint(0, vocab_size, (100, 20))
y_train = torch.randint(0, vocab_size, (100, 20))

X_train = X_train.to(device)
y_train = y_train.to(device)

train_losses = []
for epoch in range(10):
    model.train()
    total_loss = 0
    
    for i in range(0, len(X_train), 16):
        batch_x = X_train[i:i+16]
        batch_y = y_train[i:i+16]
        
        optimizer.zero_grad()
        
        logits = model(batch_x)
        loss = criterion(logits.reshape(-1, vocab_size), batch_y.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / 7
    train_losses.append(avg_loss)
    if (epoch + 1) % 2 == 0:
        print(f"   Epoch {epoch+1}/10: Loss = {avg_loss:.4f}")

# 3. Text generation
print("\n[*] Generating text with different strategies...")
generator = LanguageGenerator(model, vocab_size, device=device, temperature=1.0)

prompt = [1, 2, 3]  # Starting prompt
max_gen_len = 20

greedy_output = generator.greedy_decoding(prompt, max_length=max_gen_len)
topk_output = generator.top_k_sampling(prompt, max_length=max_gen_len, k=5)
nucleus_output = generator.nucleus_sampling(prompt, max_length=max_gen_len, p=0.9)

print(f"   Greedy:        {greedy_output[:10]} ... ")
print(f"   Top-k (k=5):   {topk_output[:10]} ...")
print(f"   Nucleus (p=0.9): {nucleus_output[:10]} ...")

# 4. Perplexity calculation
print("\n[*] Calculating perplexity...")
model.eval()
test_data = torch.randint(0, vocab_size, (50, 20)).to(device)
test_targets = torch.randint(0, vocab_size, (50, 20)).to(device)

with torch.no_grad():
    logits = model(test_data)
    loss = criterion(logits.reshape(-1, vocab_size), test_targets.reshape(-1))
    perplexity = torch.exp(loss).item()

print(f"   Perplexity: {perplexity:.4f}")

# 5. Visualizations
print("\n[*] Generating visualizations...")

plt.figure(figsize=(10, 6))
plt.plot(train_losses, marker='o', color='blue', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Language Model Training - Perplexity Convergence')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("training_curve.png", dpi=150)
print("    -> Saved: Saved training_curve.png")

# Decoding strategy comparison
decoding_methods = ['Greedy', 'Top-k (k=5)', 'Nucleus (p=0.9)']
output_lengths = [len(greedy_output), len(topk_output), len(nucleus_output)]

plt.figure(figsize=(10, 6))
plt.bar(decoding_methods, output_lengths, color=['blue', 'green', 'orange'])
plt.ylabel('Generated Length')
plt.title('Generated Text Length by Decoding Strategy')
plt.ylim(0, max_gen_len + 10)
plt.tight_layout()
plt.savefig("decoding_comparison.png", dpi=150)
print("    -> Saved: Saved decoding_comparison.png")

# Save results
results = pd.DataFrame({
    'Metric': ['Final Loss', 'Perplexity', 'Vocab Size', 'Hidden Dim'],
    'Value': [train_losses[-1], perplexity, vocab_size, hidden_dim]
})
results.to_csv("evaluation_results.csv", index=False)
print("    -> Saved: Saved evaluation_results.csv")

print("
--- Lab 7 GPT Language Modeling Execution Finished ---")
