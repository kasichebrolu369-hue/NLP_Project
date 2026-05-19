"""
Lab 1: Advanced PoS Tagging using Hidden Markov Models and Viterbi Decoding
Comprehensive HMM-based Part-of-Speech Tagger with Statistical Analysis
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import math
import nltk
from nltk.corpus import brown, universal_tagset
from nltk import pos_tag, word_tokenize
import spacy
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('corpora/brown')
except LookupError:
    nltk.download('brown')


class HMM_PoS_Tagger:
    """
    Hidden Markov Model-based Part-of-Speech Tagger
    Implements transition and emission probability matrices with Viterbi decoding
    """
    
    def __init__(self, laplace_smoothing=True, smoothing_factor=1.0):
        """
        Initialize HMM PoS Tagger
        
        Args:
            laplace_smoothing (bool): Apply Laplace smoothing to probabilities
            smoothing_factor (float): Laplace smoothing parameter
        """
        self.laplace_smoothing = laplace_smoothing
        self.smoothing_factor = smoothing_factor
        
        # Probability matrices
        self.transition_probs = defaultdict(lambda: defaultdict(float))  # P(tag_i | tag_i-1)
        self.emission_probs = defaultdict(lambda: defaultdict(float))     # P(word | tag)
        self.initial_probs = defaultdict(float)                           # P(tag | START)
        
        # Collections for training
        self.tag_counts = Counter()
        self.word_tag_counts = defaultdict(Counter)
        self.tag_transitions = defaultdict(Counter)
        self.initial_tags = Counter()
        
        # Vocabulary and tags
        self.vocab = set()
        self.tags = set()
        self.oov_token = "<OOV>"
        
    def train(self, sentences: List[List[Tuple[str, str]]], min_freq: int = 1):
        """
        Train HMM model on POS-tagged sentences
        
        Args:
            sentences (List[List[Tuple]]): List of [(word, tag), ...] sentences
            min_freq (int): Minimum word frequency threshold (for vocabulary)
        """
        # Count occurrences
        word_freq = Counter()
        for sentence in sentences:
            for word, tag in sentence:
                word_freq[word] += 1
                self.tags.add(tag)
                self.tag_counts[tag] += 1
                self.word_tag_counts[word][tag] += 1
                
                if len(sentence) > 0:
                    first_word, first_tag = sentence[0]
                    self.initial_tags[first_tag] += 1
                
                # Tag transitions
                idx = sentence.index((word, tag))
                if idx > 0:
                    prev_tag = sentence[idx-1][1]
                    self.tag_transitions[prev_tag][tag] += 1
        
        # Build vocabulary with OOV handling
        self.vocab = {word for word, freq in word_freq.items() if freq >= min_freq}
        self.vocab.add(self.oov_token)
        
        # Calculate probabilities
        self._calculate_probabilities()
        
        print(f"✓ Training complete: {len(self.vocab)} vocab items, {len(self.tags)} tags")
        
    def _calculate_probabilities(self):
        """Calculate transition, emission, and initial probabilities"""
        vocab_size = len(self.vocab)
        tag_size = len(self.tags)
        
        # Initial probabilities P(tag | START)
        total_initial = sum(self.initial_tags.values())
        for tag in self.tags:
            if self.laplace_smoothing:
                self.initial_probs[tag] = (self.initial_tags[tag] + self.smoothing_factor) / \
                                          (total_initial + self.smoothing_factor * tag_size)
            else:
                self.initial_probs[tag] = self.initial_tags[tag] / total_initial if total_initial > 0 else 1/tag_size
        
        # Transition probabilities P(tag_j | tag_i)
        for prev_tag in self.tags:
            total_trans = sum(self.tag_transitions[prev_tag].values())
            for curr_tag in self.tags:
                if self.laplace_smoothing:
                    self.transition_probs[prev_tag][curr_tag] = \
                        (self.tag_transitions[prev_tag][curr_tag] + self.smoothing_factor) / \
                        (total_trans + self.smoothing_factor * tag_size)
                else:
                    self.transition_probs[prev_tag][curr_tag] = \
                        self.tag_transitions[prev_tag][curr_tag] / total_trans if total_trans > 0 else 1/tag_size
        
        # Emission probabilities P(word | tag)
        for word in self.vocab:
            for tag in self.tags:
                if self.laplace_smoothing:
                    self.emission_probs[tag][word] = \
                        (self.word_tag_counts[word][tag] + self.smoothing_factor) / \
                        (self.tag_counts[tag] + self.smoothing_factor * vocab_size)
                else:
                    self.emission_probs[tag][word] = \
                        self.word_tag_counts[word][tag] / self.tag_counts[tag] if self.tag_counts[tag] > 0 else 1/vocab_size
    
    def _get_emission_prob(self, word: str, tag: str) -> float:
        """Get emission probability, handling OOV words"""
        if word in self.vocab:
            return self.emission_probs[tag].get(word, 1e-10)
        else:
            return self.emission_probs[tag].get(self.oov_token, 1e-10)
    
    def viterbi_decode(self, sentence: List[str]) -> List[str]:
        """
        Viterbi algorithm for finding optimal tag sequence
        
        Args:
            sentence (List[str]): Tokenized sentence
            
        Returns:
            List[str]: Optimal POS tags
        """
        if not sentence:
            return []
        
        n = len(sentence)
        
        # Initialize
        viterbi = [defaultdict(float) for _ in range(n)]
        backpointer = [defaultdict(str) for _ in range(n)]
        
        # Base case: first word
        for tag in self.tags:
            word = sentence[0]
            emission = self._get_emission_prob(word, tag)
            viterbi[0][tag] = math.log(self.initial_probs[tag] + 1e-10) + math.log(emission + 1e-10)
            backpointer[0][tag] = ""
        
        # Recursion: words 1 to n-1
        for t in range(1, n):
            word = sentence[t]
            for curr_tag in self.tags:
                max_prob = float('-inf')
                best_prev_tag = None
                
                for prev_tag in self.tags:
                    transition = self.transition_probs[prev_tag][curr_tag]
                    emission = self._get_emission_prob(word, curr_tag)
                    prob = viterbi[t-1][prev_tag] + \
                           math.log(transition + 1e-10) + \
                           math.log(emission + 1e-10)
                    
                    if prob > max_prob:
                        max_prob = prob
                        best_prev_tag = prev_tag
                
                viterbi[t][curr_tag] = max_prob
                backpointer[t][curr_tag] = best_prev_tag
        
        # Backtrace to find best path
        best_path = []
        last_tag = max(viterbi[n-1], key=viterbi[n-1].get)
        best_path.append(last_tag)
        
        for t in range(n-1, 0, -1):
            last_tag = backpointer[t][last_tag]
            best_path.append(last_tag)
        
        best_path.reverse()
        return best_path
    
    def greedy_decode(self, sentence: List[str]) -> List[str]:
        """
        Greedy tagging (most likely tag per word, independent of neighbors)
        
        Args:
            sentence (List[str]): Tokenized sentence
            
        Returns:
            List[str]: Greedy POS tags
        """
        tags = []
        for word in sentence:
            best_tag = max(self.tags, 
                          key=lambda tag: self._get_emission_prob(word, tag))
            tags.append(best_tag)
        return tags
    
    def tag(self, sentence: List[str], method: str = "viterbi") -> List[str]:
        """
        Tag a sentence
        
        Args:
            sentence (List[str]): Tokenized sentence
            method (str): 'viterbi' or 'greedy'
            
        Returns:
            List[str]: POS tags
        """
        if method == "viterbi":
            return self.viterbi_decode(sentence)
        elif method == "greedy":
            return self.greedy_decode(sentence)
        else:
            raise ValueError("Method must be 'viterbi' or 'greedy'")


class PoS_Evaluation:
    """Evaluation utilities for PoS tagging models"""
    
    @staticmethod
    def accuracy(true_tags: List[str], pred_tags: List[str]) -> float:
        """Calculate tag accuracy"""
        if len(true_tags) != len(pred_tags):
            raise ValueError("Tag sequences must have same length")
        correct = sum(t == p for t, p in zip(true_tags, pred_tags))
        return correct / len(true_tags) if len(true_tags) > 0 else 0
    
    @staticmethod
    def unknown_word_accuracy(true_tags: List[str], pred_tags: List[str], 
                             sentences: List[List[str]], vocab: Set[str]) -> float:
        """Accuracy on out-of-vocabulary words"""
        if len(true_tags) != len(pred_tags):
            raise ValueError("Tag sequences must have same length")
        
        correct_oov = 0
        total_oov = 0
        idx = 0
        
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    if true_tags[idx] == pred_tags[idx]:
                        correct_oov += 1
                    total_oov += 1
                idx += 1
        
        return correct_oov / total_oov if total_oov > 0 else 0
    
    @staticmethod
    def ambiguous_word_accuracy(true_tags: List[str], pred_tags: List[str],
                               word_tag_counts: Dict[str, Counter],
                               sentences: List[List[str]]) -> float:
        """Accuracy on ambiguous words (can have multiple tags)"""
        correct_amb = 0
        total_amb = 0
        idx = 0
        
        for sentence in sentences:
            for word in sentence:
                if len(word_tag_counts.get(word, Counter())) > 1:
                    if true_tags[idx] == pred_tags[idx]:
                        correct_amb += 1
                    total_amb += 1
                idx += 1
        
        return correct_amb / total_amb if total_amb > 0 else 0
    
    @staticmethod
    def per_tag_accuracy(true_tags: List[str], pred_tags: List[str], 
                        tags: Set[str]) -> Dict[str, float]:
        """Calculate accuracy per tag"""
        results = {}
        for tag in tags:
            mask = [t == tag for t in true_tags]
            if sum(mask) > 0:
                correct = sum(t == p for t, p, m in zip(true_tags, pred_tags, mask) if m)
                results[tag] = correct / sum(mask)
        return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("LAB 1: HMM PoS TAGGING WITH VITERBI DECODING")
    print("="*70)
    
    # Load training data from Brown Corpus
    print("\n1. LOADING TRAINING DATA...")
    sentences = brown.tagged_sents()[:1000]  # Use first 1000 sentences for demo
    print(f"   Loaded {len(sentences)} sentences from Brown Corpus")
    
    # Split into train/test
    train_size = int(0.8 * len(sentences))
    train_sentences = sentences[:train_size]
    test_sentences = sentences[train_size:]
    
    print(f"   Training set: {len(train_sentences)} sentences")
    print(f"   Test set: {len(test_sentences)} sentences")
    
    # Train HMM model
    print("\n2. TRAINING HMM MODEL...")
    hmm_tagger = HMM_PoS_Tagger(laplace_smoothing=True, smoothing_factor=1.0)
    hmm_tagger.train(train_sentences, min_freq=1)
    
    # Prepare test data
    print("\n3. TESTING HMM MODEL...")
    test_words = []
    test_tags = []
    test_sentences_only = []
    
    for sentence in test_sentences:
        words, tags = zip(*sentence)
        test_words.extend(words)
        test_tags.extend(tags)
        test_sentences_only.append(list(words))
    
    # Test with different methods
    print("\n   Testing Viterbi Decoding...")
    test_pred_viterbi = []
    for sentence in test_sentences:
        words, _ = zip(*sentence)
        tags = hmm_tagger.tag(list(words), method="viterbi")
        test_pred_viterbi.extend(tags)
    
    print("   Testing Greedy Decoding...")
    test_pred_greedy = []
    for sentence in test_sentences:
        words, _ = zip(*sentence)
        tags = hmm_tagger.tag(list(words), method="greedy")
        test_pred_greedy.extend(tags)
    
    # Comparison with NLTK
    print("\n4. COMPARING WITH NLTK TAGGER...")
    nltk_predictions = []
    for sentence in test_sentences:
        words, _ = zip(*sentence)
        nltk_tags = [tag for word, tag in nltk.pos_tag(list(words))]
        nltk_predictions.extend(nltk_tags)
    
    # Comparison with spaCy
    print("   Comparing with spaCy...")
    try:
        nlp = spacy.load("en_core_web_sm")
        spacy_predictions = []
        for sentence in test_sentences:
            words, _ = zip(*sentence)
            doc = nlp(" ".join(words))
            spacy_tags = [token.pos_ for token in doc]
            spacy_predictions.extend(spacy_tags)
    except:
        print("   Warning: spaCy model not available, skipping spaCy comparison")
        spacy_predictions = None
    
    # Evaluation
    print("\n5. EVALUATION RESULTS...")
    accuracy_viterbi = PoS_Evaluation.accuracy(test_tags, test_pred_viterbi)
    accuracy_greedy = PoS_Evaluation.accuracy(test_tags, test_pred_greedy)
    accuracy_nltk = PoS_Evaluation.accuracy(test_tags, nltk_predictions)
    
    print(f"\n   Viterbi Accuracy:  {accuracy_viterbi:.4f}")
    print(f"   Greedy Accuracy:   {accuracy_greedy:.4f}")
    print(f"   NLTK Accuracy:     {accuracy_nltk:.4f}")
    
    if spacy_predictions:
        accuracy_spacy = PoS_Evaluation.accuracy(test_tags, spacy_predictions)
        print(f"   spaCy Accuracy:    {accuracy_spacy:.4f}")
    
    # OOV Analysis
    print("\n6. OUT-OF-VOCABULARY ANALYSIS...")
    train_words = set()
    for sentence in train_sentences:
        for word, _ in sentence:
            train_words.add(word)
    
    oov_accuracy = PoS_Evaluation.unknown_word_accuracy(
        test_tags, test_pred_viterbi, test_sentences_only, train_words
    )
    print(f"   OOV Word Accuracy: {oov_accuracy:.4f}")
    
    # Ambiguous words analysis
    print("\n7. AMBIGUOUS WORD ANALYSIS...")
    amb_accuracy = PoS_Evaluation.ambiguous_word_accuracy(
        test_tags, test_pred_viterbi, hmm_tagger.word_tag_counts, test_sentences_only
    )
    print(f"   Ambiguous Word Accuracy: {amb_accuracy:.4f}")
    
    # Per-tag accuracy
    print("\n8. PER-TAG ACCURACY...")
    per_tag = PoS_Evaluation.per_tag_accuracy(test_tags, test_pred_viterbi, hmm_tagger.tags)
    top_tags = sorted(per_tag.items(), key=lambda x: x[1], reverse=True)[:10]
    for tag, acc in top_tags:
        print(f"   {tag:10s}: {acc:.4f}")
    
    # Create visualizations
    print("\n9. GENERATING VISUALIZATIONS...")
    
    # Confusion matrix
    cm = confusion_matrix(test_tags, test_pred_viterbi, labels=sorted(hmm_tagger.tags))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=sorted(hmm_tagger.tags),
                yticklabels=sorted(hmm_tagger.tags), cbar=True)
    plt.title("Confusion Matrix: HMM PoS Tagger")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("   ✓ Saved confusion_matrix.png")
    
    # Method comparison
    plt.figure(figsize=(10, 6))
    methods = ["Viterbi", "Greedy", "NLTK"]
    accuracies = [accuracy_viterbi, accuracy_greedy, accuracy_nltk]
    if spacy_predictions:
        methods.append("spaCy")
        accuracies.append(PoS_Evaluation.accuracy(test_tags, spacy_predictions))
    
    plt.bar(methods, accuracies, color=['blue', 'orange', 'green', 'red'][:len(methods)])
    plt.ylabel("Accuracy")
    plt.title("PoS Tagging Method Comparison")
    plt.ylim([0, 1])
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f"{acc:.4f}", ha='center')
    plt.tight_layout()
    plt.savefig("method_comparison.png", dpi=150)
    print("   ✓ Saved method_comparison.png")
    
    # Save results
    print("\n10. SAVING RESULTS...")
    results_df = pd.DataFrame({
        'Method': methods,
        'Accuracy': accuracies,
        'Sentences_Evaluated': [len(test_sentences)] * len(methods)
    })
    results_df.to_csv("evaluation_results.csv", index=False)
    print("   ✓ Saved evaluation_results.csv")
    
    print("\n" + "="*70)
    print("Lab 1 Complete!")
    print("="*70)
