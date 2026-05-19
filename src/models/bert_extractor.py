"""
Task 1: Fine-tune BERT for Structured Information Extraction
Extracts CVE ID, Description, CWE mapping, and Exploit type
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
from loguru import logger
import sys
from datetime import datetime

# Configure logger
logger.remove()
logger.add(sys.stderr, format="{time} | {level: <8} | {message}")


class CVEBertDataset(Dataset):
    """Dataset for BERT fine-tuning on CVE tasks"""
    
    def __init__(self, texts: List[str], labels: List[Dict], tokenizer, max_length: int = 512):
        """
        Initialize dataset
        
        Args:
            texts: List of CVE descriptions
            labels: List of label dictionaries with 'cwe', 'exploit_type' keys
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label mappings
        self.cwe_label_map = self._create_label_map(
            [label.get('cwe', 'OTHER') for label in labels]
        )
        self.exploit_label_map = self._create_label_map(
            [label.get('exploit_type', 'UNKNOWN') for label in labels]
        )
    
    def _create_label_map(self, labels: List[str]) -> Dict[str, int]:
        """Create mapping from label to index"""
        unique_labels = sorted(set(labels))
        return {label: idx for idx, label in enumerate(unique_labels)}
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label_data = self.labels[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get labels
        cwe_label = self.cwe_label_map.get(label_data.get('cwe', 'OTHER'), 0)
        exploit_label = self.exploit_label_map.get(label_data.get('exploit_type', 'UNKNOWN'), 0)
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'cwe_label': torch.tensor(cwe_label, dtype=torch.long),
            'exploit_label': torch.tensor(exploit_label, dtype=torch.long)
        }


class BertCVEExtractor(nn.Module):
    """BERT-based model for CVE information extraction"""
    
    def __init__(self, bert_model: str = "bert-base-uncased", 
                 num_cwe_classes: int = 10, num_exploit_classes: int = 7):
        """
        Initialize BERT extractor
        
        Args:
            bert_model: BERT model name
            num_cwe_classes: Number of CWE classes
            num_exploit_classes: Number of exploit type classes
        """
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(0.1)
        
        # Classification heads
        self.cwe_classifier = nn.Linear(768, num_cwe_classes)
        self.exploit_classifier = nn.Linear(768, num_exploit_classes)
        
        self.num_cwe_classes = num_cwe_classes
        self.num_exploit_classes = num_exploit_classes
    
    def forward(self, input_ids, attention_mask):
        """
        Forward pass
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
            
        Returns:
            Tuple of (cwe_logits, exploit_logits)
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        pooled_output = self.dropout(pooled_output)
        
        cwe_logits = self.cwe_classifier(pooled_output)
        exploit_logits = self.exploit_classifier(pooled_output)
        
        return cwe_logits, exploit_logits


class BertCVETrainer:
    """Trainer for BERT-based CVE extractor"""
    
    def __init__(self, model: BertCVEExtractor, device: str = 'cuda'):
        """
        Initialize trainer
        
        Args:
            model: BERT CVE extractor model
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.loss_history = []
    
    def train_epoch(self, train_loader: DataLoader, optimizer, scheduler) -> float:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Average loss for epoch
        """
        self.model.train()
        total_loss = 0
        
        loss_fn = nn.CrossEntropyLoss()
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            cwe_labels = batch['cwe_label'].to(self.device)
            exploit_labels = batch['exploit_label'].to(self.device)
            
            cwe_logits, exploit_logits = self.model(input_ids, attention_mask)
            
            cwe_loss = loss_fn(cwe_logits, cwe_labels)
            exploit_loss = loss_fn(exploit_logits, exploit_labels)
            loss = cwe_loss + exploit_loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        self.loss_history.append(avg_loss)
        return avg_loss
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with metrics
        """
        self.model.eval()
        
        cwe_correct = 0
        exploit_correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                cwe_labels = batch['cwe_label'].to(self.device)
                exploit_labels = batch['exploit_label'].to(self.device)
                
                cwe_logits, exploit_logits = self.model(input_ids, attention_mask)
                
                cwe_preds = torch.argmax(cwe_logits, dim=1)
                exploit_preds = torch.argmax(exploit_logits, dim=1)
                
                cwe_correct += (cwe_preds == cwe_labels).sum().item()
                exploit_correct += (exploit_preds == exploit_labels).sum().item()
                total += cwe_labels.size(0)
        
        return {
            'cwe_accuracy': cwe_correct / total,
            'exploit_accuracy': exploit_correct / total
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
             epochs: int = 3, learning_rate: float = 2e-5) -> Dict:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training history
        """
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        history = {
            'train_loss': [],
            'val_metrics': []
        }
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            history['train_loss'].append(train_loss)
            logger.info(f"  Training Loss: {train_loss:.4f}")
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            history['val_metrics'].append(val_metrics)
            logger.info(f"  CWE Accuracy: {val_metrics['cwe_accuracy']:.4f}")
            logger.info(f"  Exploit Accuracy: {val_metrics['exploit_accuracy']:.4f}")
        
        return history
    
    def predict(self, texts: List[str], tokenizer, batch_size: int = 32) -> List[Dict]:
        """
        Make predictions on new texts
        
        Args:
            texts: List of CVE descriptions
            tokenizer: BERT tokenizer
            batch_size: Batch size for prediction
            
        Returns:
            List of predictions
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                encodings = tokenizer(
                    batch_texts,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                cwe_logits, exploit_logits = self.model(input_ids, attention_mask)
                
                cwe_preds = torch.argmax(cwe_logits, dim=1)
                exploit_preds = torch.argmax(exploit_logits, dim=1)
                
                cwe_probs = torch.softmax(cwe_logits, dim=1)
                exploit_probs = torch.softmax(exploit_logits, dim=1)
                
                for j in range(len(batch_texts)):
                    predictions.append({
                        'text': batch_texts[j],
                        'cwe_pred': cwe_preds[j].item(),
                        'exploit_pred': exploit_preds[j].item(),
                        'cwe_confidence': cwe_probs[j].max().item(),
                        'exploit_confidence': exploit_probs[j].max().item()
                    })
        
        return predictions
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Model loaded from {path}")


if __name__ == "__main__":
    # Example usage
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Initialize model
    model = BertCVEExtractor(num_cwe_classes=10, num_exploit_classes=7)
    trainer = BertCVETrainer(model, device=device)
    
    # Prepare tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Sample data
    sample_texts = [
        "A remote code execution vulnerability in Application X allows attackers to execute arbitrary code",
        "Cross-site scripting vulnerability in web application allows script injection"
    ]
    
    sample_labels = [
        {'cwe': 'CWE-119', 'exploit_type': 'RCE'},
        {'cwe': 'CWE-79', 'exploit_type': 'XSS'}
    ]
    
    # Create dataset
    dataset = CVEBertDataset(sample_texts, sample_labels, tokenizer)
    data_loader = DataLoader(dataset, batch_size=2)
    
    # Make predictions
    predictions = trainer.predict(sample_texts, tokenizer)
    print(f"Predictions: {predictions}")
