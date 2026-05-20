"""
Course: Natural Language Processing
Academic Year: 2025-2026
Student Portfolio Submission

Lab 2 CRF NER
"""


import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Set
import nltk
from nltk import pos_tag, word_tokenize
from nltk.tokenize import sent_tokenize
import spacy
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
warnings.filterwarnings('ignore')

try:
    import sklearn_crfsuite
    from sklearn_crfsuite import CRF
except ImportError:
    print("Warning: sklearn-crfsuite not installed, CRF models will use dummy implementation")

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class FeatureExtractor:
    """Extract features for NER sequence labeling"""
    
    @staticmethod
    def word_features(word: str) -> Dict:
        """Extract basic word features"""
        features = {
            'bias': 1.0,
            'word': word,
            'word.lower': word.lower(),
            'word.isupper': word.isupper(),
            'word.istitle': word.istitle(),
            'word.isdigit': word.isdigit(),
        }
        
        # Character features
        if len(word) > 0:
            features['word.prefix'] = word[:2]
            features['word.suffix'] = word[-2:]
        if len(word) > 3:
            features['word.prefix3'] = word[:3]
            features['word.suffix3'] = word[-3:]
        
        # Shape features
        if re.match(r'^\d+$', word):
            features['word.shape'] = 'NUMBER'
        elif re.match(r'^[A-Z][a-z]+$', word):
            features['word.shape'] = 'TITLECASE'
        elif re.match(r'^[A-Z]+$', word):
            features['word.shape'] = 'UPPERCASE'
        elif re.match(r'^[a-z]+$', word):
            features['word.shape'] = 'LOWERCASE'
        
        # Capitalization
        features['word.caps'] = sum(c.isupper() for c in word) / max(len(word), 1)
        
        return features
    
    @staticmethod
    def get_features(sentence: List[str], pos: int, pos_tags: List[str] = None) -> Dict:
        """
        Extract contextual features for position in sentence
        
        Args:
            sentence (List[str]): Tokenized sentence
            pos (int): Position in sentence
            pos_tags (List[str]): POS tags (optional)
            
        Returns:
            Dict: Features dictionary
        """
        features = {}
        
        # Word features
        word = sentence[pos]
        features.update(FeatureExtractor.word_features(word))
        
        # Context features
        if pos > 0:
            features.update({
                'prev.word': sentence[pos-1],
                'prev.word.lower': sentence[pos-1].lower(),
                'prev.word.isupper': sentence[pos-1].isupper(),
            })
            features.update({f'prev.{k}': v for k, v in 
                            FeatureExtractor.word_features(sentence[pos-1]).items()})
        else:
            features['BOS'] = True
        
        if pos < len(sentence) - 1:
            features.update({
                'next.word': sentence[pos+1],
                'next.word.lower': sentence[pos+1].lower(),
                'next.word.isupper': sentence[pos+1].isupper(),
            })
            features.update({f'next.{k}': v for k, v in 
                            FeatureExtractor.word_features(sentence[pos+1]).items()})
        else:
            features['EOS'] = True
        
        # POS tag features
        if pos_tags:
            features['pos'] = pos_tags[pos]
            if pos > 0:
                features['prev.pos'] = pos_tags[pos-1]
            if pos < len(pos_tags) - 1:
                features['next.pos'] = pos_tags[pos+1]
        
        return features


class NERDataLoader:
    """Load and process NER datasets"""
    
    @staticmethod
    def load_conll2003_demo() -> List[Tuple[List[str], List[str]]]:
        """
        Load demo data resembling CoNLL-2003 format
        Returns: [(tokens, tags), ...]
        """
        # Sample data for demonstration
        demo_data = [
            (["John", "works", "for", "Google"], ["PERSON", "O", "O", "ORG"]),
            (["Mary", "lives", "in", "Paris"], ["PERSON", "O", "O", "LOC"]),
            (["The", "CEO", "of", "Apple", "is", "Tim", "Cook"], 
             ["O", "O", "O", "ORG", "O", "PERSON", "PERSON"]),
            (["Visit", "Microsoft", "in", "Seattle"], ["O", "ORG", "O", "LOC"]),
            (["Google", "and", "Amazon", "are", "big"], ["ORG", "O", "ORG", "O", "O"]),
        ]
        return demo_data * 20  # Repeat to get more data


class CRF_NER:
    """Conditional Random Fields based Named Entity Recognition"""
    
    def __init__(self):
        try:
            self.model = CRF(
                algorithm='lbfgs',
                c1=0.1,
                c2=0.1,
                max_iterations=100,
                all_transitions=True,
                verbose=0
            )
        except:
            self.model = None
        
        self.entities = ['PERSON', 'ORG', 'LOC', 'MISC', 'O']
        self.feature_extractor = FeatureExtractor()
    
    def prepare_data(self, data: List[Tuple[List[str], List[str]]], 
                    use_pos: bool = True) -> Tuple[List, List]:
        """
        Prepare features and labels for CRF
        
        Args:
            data: List of (tokens, tags) tuples
            use_pos: Whether to use POS tags as features
            
        Returns:
            (X, y) for training
        """
        X, y = [], []
        
        for tokens, tags in data:
            # Get POS tags
            pos_tags = None
            if use_pos:
                pos_tags = [tag for word, tag in pos_tag(tokens)]
            
            # Extract features for each token
            features = []
            for i in range(len(tokens)):
                feats = self.feature_extractor.get_features(tokens, i, pos_tags)
                features.append(feats)
            
            X.append(features)
            y.append(tags)
        
        return X, y
    
    def train(self, X: List, y: List):
        """Train CRF model"""
        if self.model is None:
            print("Warning: CRF model not available")
            return
        
        self.model.fit(X, y)
        print(f"✓ CRF Model trained on {len(X)} sequences")
    
    def predict(self, X: List) -> List:
        """Predict tags for sequences"""
        if self.model is None:
            return [['O'] * len(x) for x in X]
        
        return self.model.predict(X)
    
    def predict_single(self, tokens: List[str]) -> List[str]:
        """Predict tags for single sentence"""
        pos_tags = [tag for word, tag in pos_tag(tokens)]
        features = []
        for i in range(len(tokens)):
            feats = self.feature_extractor.get_features(tokens, i, pos_tags)
            features.append(feats)
        
        if self.model is None:
            return ['O'] * len(tokens)
        
        return self.model.predict([features])[0]


class NER_Evaluation:
    """Evaluate NER models"""
    
    @staticmethod
    def token_level_metrics(true_tags: List[str], pred_tags: List[str]) -> Dict:
        """Calculate token-level precision, recall, F1"""
        if len(true_tags) != len(pred_tags):
            raise ValueError("Tag sequences must match")
        
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        
        accuracy = accuracy_score(true_tags, pred_tags)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_tags, pred_tags, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    @staticmethod
    def per_entity_metrics(true_tags: List[str], pred_tags: List[str], 
                          entities: List[str]) -> Dict:
        """Calculate metrics per entity type"""
        results = {}
        
        for entity in entities:
            if entity == 'O':
                continue
            
            tp = sum(1 for t, p in zip(true_tags, pred_tags) if t == entity and p == entity)
            fp = sum(1 for t, p in zip(true_tags, pred_tags) if t != entity and p == entity)
            fn = sum(1 for t, p in zip(true_tags, pred_tags) if t == entity and p != entity)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[entity] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn
            }
        
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LAB 2: NAMED ENTITY RECOGNITION USING CRF")
    print("="*70)
    
    # Load data
    print("\n[*] Loading ner data...")
    data = NERDataLoader.load_conll2003_demo()
    print(f"   Loaded {len(data)} annotated samples")
    print(f"   Sample: {data[0]}")
    
    # Split data
    print("\n[*] Preparing data...")
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    print(f"   Training set: {len(train_data)} sentences")
    print(f"   Test set: {len(test_data)} sentences")
    
    # Initialize CRF model
    print("\n[*] Training crf model...")
    ner_model = CRF_NER()
    
    # Prepare features
    X_train, y_train = ner_model.prepare_data(train_data, use_pos=True)
    X_test, y_test = ner_model.prepare_data(test_data, use_pos=True)
    
    # Train
    ner_model.train(X_train, y_train)
    
    # Evaluate
    print("\n[*] Evaluating model...")
    y_pred = ner_model.predict(X_test)
    
    # Flatten for evaluation
    all_true = []
    all_pred = []
    for true_seq, pred_seq in zip(y_test, y_pred):
        all_true.extend(true_seq)
        all_pred.extend(pred_seq)
    
    # Token-level metrics
    token_metrics = NER_Evaluation.token_level_metrics(all_true, all_pred)
    print(f"\n   Token-level Accuracy: {token_metrics['accuracy']:.4f}")
    print(f"   Token-level Precision: {token_metrics['precision']:.4f}")
    print(f"   Token-level Recall: {token_metrics['recall']:.4f}")
    print(f"   Token-level F1: {token_metrics['f1']:.4f}")
    
    # Per-entity metrics
    print("\n[*] Per-entity performance...")
    entity_metrics = NER_Evaluation.per_entity_metrics(all_true, all_pred, ner_model.entities)
    for entity, metrics in entity_metrics.items():
        print(f"\n   {entity}:")
        print(f"      Precision: {metrics['precision']:.4f}")
        print(f"      Recall: {metrics['recall']:.4f}")
        print(f"      F1: {metrics['f1']:.4f}")
        print(f"      Support: {metrics['support']}")
    
    # Feature ablation study
    print("\n[*] Feature ablation study...")
    print("   Testing with/without POS tags...")
    
    X_train_no_pos, y_train_no_pos = ner_model.prepare_data(train_data, use_pos=False)
    X_test_no_pos, y_test_no_pos = ner_model.prepare_data(test_data, use_pos=False)
    
    model_no_pos = CRF_NER()
    model_no_pos.train(X_train_no_pos, y_train_no_pos)
    y_pred_no_pos = model_no_pos.predict(X_test_no_pos)
    
    all_true_no_pos = []
    all_pred_no_pos = []
    for true_seq, pred_seq in zip(y_test_no_pos, y_pred_no_pos):
        all_true_no_pos.extend(true_seq)
        all_pred_no_pos.extend(pred_seq)
    
    metrics_no_pos = NER_Evaluation.token_level_metrics(all_true_no_pos, all_pred_no_pos)
    
    print(f"\n   With POS tags - F1: {token_metrics['f1']:.4f}")
    print(f"   Without POS tags - F1: {metrics_no_pos['f1']:.4f}")
    print(f"   POS tag contribution: {(token_metrics['f1'] - metrics_no_pos['f1']):.4f}")
    
    # Example predictions
    print("\n[*] Example predictions...")
    example_sent = ["John", "works", "at", "Google", "in", "New", "York"]
    predictions = ner_model.predict_single(example_sent)
    print(f"\n   Sentence: {' '.join(example_sent)}")
    for word, tag in zip(example_sent, predictions):
        print(f"      {word:15s} → {tag}")
    
    # Visualization
    print("\n[*] Generating visualizations...")
    
    # Entity distribution
    entity_counts_true = {}
    entity_counts_pred = {}
    for e in ner_model.entities:
        entity_counts_true[e] = all_true.count(e)
        entity_counts_pred[e] = all_pred.count(e)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(entity_counts_true.keys(), entity_counts_true.values(), color='blue', alpha=0.7)
    axes[0].set_title("True Entity Distribution")
    axes[0].set_ylabel("Count")
    
    axes[1].bar(entity_counts_pred.keys(), entity_counts_pred.values(), color='orange', alpha=0.7)
    axes[1].set_title("Predicted Entity Distribution")
    axes[1].set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig("entity_distribution.png", dpi=150)
    print("    -> Saved: Saved entity_distribution.png")
    
    # F1 scores comparison
    f1_scores = {entity: metrics['f1'] for entity, metrics in entity_metrics.items()}
    plt.figure(figsize=(10, 6))
    plt.barh(list(f1_scores.keys()), list(f1_scores.values()), color='green', alpha=0.7)
    plt.xlabel("F1 Score")
    plt.title("Per-Entity F1 Scores")
    plt.xlim([0, 1])
    for i, (entity, f1) in enumerate(f1_scores.items()):
        plt.text(f1 + 0.02, i, f"{f1:.4f}", va='center')
    plt.tight_layout()
    plt.savefig("entity_f1_scores.png", dpi=150)
    print("    -> Saved: Saved entity_f1_scores.png")
    
    # Save results
    print("\n[*] Saving results...")
    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [
            token_metrics['accuracy'],
            token_metrics['precision'],
            token_metrics['recall'],
            token_metrics['f1']
        ]
    })
    results_df.to_csv("evaluation_results.csv", index=False)
    print("    -> Saved: Saved evaluation_results.csv")
    
    print("\n" + "="*70)
    print("Lab 2 Complete!")
    print("="*70)
