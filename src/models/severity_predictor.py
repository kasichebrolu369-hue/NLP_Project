"""
Task 2: Severity Prediction Model
Predicts CVSS score from CVE description
Compares Classical ML and Transformer-based approaches
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Tuple, Optional
import json
import pickle
from loguru import logger
import sys

# Configure logger
logger.remove()
logger.add(sys.stderr, format="{time} | {level: <8} | {message}")


class ClassicalSeverityPredictor:
    """Classical ML approach for CVSS score prediction"""
    
    def __init__(self, model_type: str = 'svm'):
        """
        Initialize severity predictor
        
        Args:
            model_type: 'svm' or 'rf' for Random Forest
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.scaler = StandardScaler()
        
        if model_type == 'svm':
            self.model = SVR(kernel='rbf', C=100, epsilon=0.1)
        elif model_type == 'rf':
            self.model = RandomForestRegressor(n_estimators=100, max_depth=20)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Initialized {model_type.upper()} severity predictor")
    
    def train(self, texts: List[str], scores: List[float], 
             test_size: float = 0.2) -> Dict[str, float]:
        """
        Train classical model
        
        Args:
            texts: List of CVE descriptions
            scores: List of CVSS scores
            test_size: Test set size
            
        Returns:
            Dictionary with metrics
        """
        # Vectorize texts
        X = self.vectorizer.fit_transform(texts).toarray()
        y = np.array(scores)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Train model
        logger.info(f"Training {self.model_type.upper()} model on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'test_size': len(X_test)
        }
        
        logger.info(f"Model trained. MAE: {metrics['mae']:.4f}, RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict CVSS scores
        
        Args:
            texts: List of CVE descriptions
            
        Returns:
            Array of predicted CVSS scores
        """
        X = self.vectorizer.transform(texts).toarray()
        X = self.scaler.transform(X)
        return np.clip(self.model.predict(X), 0, 10)
    
    def save(self, path: str):
        """Save model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'scaler': self.scaler
            }, f)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
            self.scaler = data['scaler']
        logger.info(f"Model loaded from {path}")


class SeverityDataset(Dataset):
    """Dataset for transformer-based severity prediction"""
    
    def __init__(self, texts: List[str], scores: List[float], tokenizer, max_length: int = 512):
        """
        Initialize dataset
        
        Args:
            texts: CVE descriptions
            scores: CVSS scores
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.scores = torch.tensor(scores, dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        score = self.scores[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'score': score
        }


class BertSeverityPredictor(nn.Module):
    """Transformer-based CVSS score prediction"""
    
    def __init__(self, bert_model: str = "bert-base-uncased"):
        """
        Initialize BERT regressor
        
        Args:
            bert_model: BERT model name
        """
        super().__init__()
        
        self.bert = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(0.1)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        score = self.regressor(pooled_output)
        # Clamp to valid CVSS range [0, 10]
        score = torch.clamp(score, min=0, max=10)
        
        return score


class TransformerSeverityPredictor:
    """Trainer for transformer-based severity prediction"""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize predictor
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device
        self.model = BertSeverityPredictor().to(device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.loss_history = {'train': [], 'val': []}
    
    def train(self, texts: List[str], scores: List[float],
             epochs: int = 3, batch_size: int = 32,
             learning_rate: float = 2e-5, test_size: float = 0.2) -> Dict:
        """
        Train transformer model
        
        Args:
            texts: CVE descriptions
            scores: CVSS scores
            epochs: Number of epochs
            batch_size: Batch size
            learning_rate: Learning rate
            test_size: Test set size
            
        Returns:
            Training history and final metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, scores, test_size=test_size, random_state=42
        )
        
        # Create datasets
        train_dataset = SeverityDataset(X_train, y_train, self.tokenizer)
        test_dataset = SeverityDataset(X_test, y_test, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        loss_fn = nn.MSELoss()
        
        # Training loop
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Train
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                scores = batch['score'].to(self.device).unsqueeze(1)
                
                predictions = self.model(input_ids, attention_mask)
                loss = loss_fn(predictions, scores)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.loss_history['train'].append(train_loss)
            logger.info(f"  Training Loss: {train_loss:.4f}")
            
            # Validate
            val_metrics = self._evaluate(test_loader, y_test)
            self.loss_history['val'].append(val_metrics['mae'])
            logger.info(f"  Validation MAE: {val_metrics['mae']:.4f}, RMSE: {val_metrics['rmse']:.4f}")
        
        return {
            'history': self.loss_history,
            'final_metrics': val_metrics
        }
    
    def _evaluate(self, test_loader, y_true) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                scores = self.model(input_ids, attention_mask)
                predictions.extend(scores.cpu().numpy().squeeze().tolist())
        
        predictions = np.array(predictions)
        y_true = np.array(y_true)
        
        return {
            'mae': mean_absolute_error(y_true, predictions),
            'rmse': np.sqrt(mean_squared_error(y_true, predictions)),
            'r2': r2_score(y_true, predictions)
        }
    
    def predict(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Predict CVSS scores"""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                encodings = self.tokenizer(
                    batch_texts,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encodings['input_ids'].to(self.device)
                attention_mask = encodings['attention_mask'].to(self.device)
                
                scores = self.model(input_ids, attention_mask)
                predictions.extend(scores.cpu().numpy().squeeze().tolist())
        
        return np.array(predictions)
    
    def save_model(self, path: str):
        """Save model"""
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model"""
        self.model.load_state_dict(torch.load(path))
        logger.info(f"Model loaded from {path}")


class SeverityPredictionComparison:
    """Compare classical ML and transformer approaches"""
    
    @staticmethod
    def compare_models(texts: List[str], scores: List[float]) -> pd.DataFrame:
        """
        Compare all severity prediction models
        
        Args:
            texts: CVE descriptions
            scores: CVSS scores
            
        Returns:
            DataFrame with comparison results
        """
        results = {}
        
        # Classical ML - SVM
        logger.info("Training SVM model...")
        svm_predictor = ClassicalSeverityPredictor('svm')
        svm_metrics = svm_predictor.train(texts, scores)
        results['SVM'] = svm_metrics
        
        # Classical ML - Random Forest
        logger.info("Training Random Forest model...")
        rf_predictor = ClassicalSeverityPredictor('rf')
        rf_metrics = rf_predictor.train(texts, scores)
        results['Random Forest'] = rf_metrics
        
        # Transformer - BERT
        logger.info("Training BERT model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        transformer_predictor = TransformerSeverityPredictor(device=device)
        transformer_result = transformer_predictor.train(texts, scores)
        results['BERT'] = transformer_result['final_metrics']
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T
        
        logger.info("\n=== Model Comparison ===")
        logger.info(comparison_df)
        
        return comparison_df


if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "A remote code execution vulnerability in Application X allows attackers to execute arbitrary code",
        "Cross-site scripting vulnerability in web application",
        "Buffer overflow vulnerability in memory management",
        "SQL injection vulnerability in database queries"
    ]
    
    sample_scores = [9.8, 6.5, 7.2, 5.4]
    
    # Compare models
    comparison = SeverityPredictionComparison.compare_models(sample_texts, sample_scores)
    print(comparison)
