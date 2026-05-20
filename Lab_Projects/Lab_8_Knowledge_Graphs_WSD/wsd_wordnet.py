"""
Course: Natural Language Processing
Academic Year: 2025-2026
Student Portfolio Submission

Lab 8 Knowledge Graphs WSD
"""


import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Download required data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

print("
--- Starting Lab 8 Knowledge Graphs WSD ---")

# 1. WordNet exploration
print("\n[*] Exploring wordnet...")

word = "bank"
synsets = wn.synsets(word)
print(f"\n   Word: '{word}'")
print(f"   Number of meanings: {len(synsets)}")

for i, synset in enumerate(synsets):
    print(f"\n   Sense {i+1}: {synset.name()}")
    print(f"      Definition: {synset.definition()}")
    print(f"      Examples: {synset.examples()}")

# 2. Semantic relationships
print("\n[*] Semantic relationships...")

test_word = "dog"
synset = wn.synsets(test_word)[0]

print(f"\n   Word: '{test_word}' (using sense: {synset.name()})")
print(f"   Definition: {synset.definition()}")

# Hypernyms (more general concepts)
hypernyms = synset.hypernyms()
print(f"\n   Hypernyms (broader meaning):")
for h in hypernyms[:3]:
    print(f"      - {h.name()}: {h.definition()}")

# Hyponyms (more specific concepts)
hyponyms = synset.hyponyms()
print(f"\n   Hyponyms (narrower meaning):")
for h in hyponyms[:3]:
    print(f"      - {h.name()}: {h.definition()}")

# Meronyms (part-of)
meronyms = synset.part_meronyms()
print(f"\n   Meronyms (parts):")
for m in meronyms[:3]:
    print(f"      - {m.name()}: {m.definition()}")

# 3. Word similarity
print("\n[*] Word similarity metrics...")

word1 = "dog"
word2 = "cat"
word3 = "car"

synset1_dog = wn.synsets(word1)[0]
synset1_cat = wn.synsets(word2)[0]
synset1_car = wn.synsets(word3)[0]

print(f"\n   Comparing: {word1}, {word2}, {word3}")
print(f"\n   Synset similarity (dog vs cat):")
print(f"      Path-based: {synset1_dog.path_similarity(synset1_cat):.4f}")
print(f"      LCH: {synset1_dog.lch_similarity(synset1_cat):.4f}")
print(f"      WUPA: {synset1_dog.wup_similarity(synset1_cat):.4f}")

print(f"\n   Synset similarity (dog vs car):")
print(f"      Path-based: {synset1_dog.path_similarity(synset1_car):.4f}")
print(f"      LCH: {synset1_dog.lch_similarity(synset1_car):.4f}")
print(f"      WUPA: {synset1_dog.wup_similarity(synset1_car):.4f}")

# 4. Lesk algorithm for Word Sense Disambiguation
print("\n[*] Word sense disambiguation (lesk algorithm)...")

class LeskAlgorithm:
    @staticmethod
    def lesk(context, target_word, synsets=None):
        """
        Lesk algorithm for word sense disambiguation
        """
        if synsets is None:
            synsets = wn.synsets(target_word)
        
        context_tokens = set(word_tokenize(context.lower()))
        
        best_synset = None
        max_overlap = 0
        
        for synset in synsets:
            # Definition + Examples
            definition = synset.definition().lower()
            examples = ' '.join(synset.examples()).lower()
            glosses = (definition + ' ' + examples).split()
            
            # Hyperphone glosses
            for related in synset.hypernyms():
                glosses.extend(related.definition().lower().split())
            
            gloss_tokens = set(glosses)
            overlap = len(context_tokens & gloss_tokens)
            
            if overlap > max_overlap:
                max_overlap = overlap
                best_synset = synset
        
        return best_synset

# Test sentences
test_sentences = [
    ("I went to the bank to withdraw money", "bank"),
    ("I sat on the bank of the river", "bank"),
]

lesk = LeskAlgorithm()

print("\n   Testing Lesk Algorithm:")
for sentence, word in test_sentences:
    result = lesk.lesk(sentence, word)
    print(f"\n   Sentence: '{sentence}'")
    print(f"   Target word: '{word}'")
    if result:
        print(f"   Best sense: {result.name()}")
        print(f"   Definition: {result.definition()}")

# 5. Visualizations
print("\n[*] Generating visualizations...")

# Similarity matrix
words_sample = ["dog", "cat", "car", "taxi", "puppy"]
synsets_sample = [wn.synsets(w)[0] if wn.synsets(w) else None for w in words_sample]

similarity_matrix = []
for s1 in synsets_sample:
    if s1:
        row = []
        for s2 in synsets_sample:
            if s2:
                sim = s1.wup_similarity(s2)
                row.append(sim if sim else 0)
            else:
                row.append(0)
        similarity_matrix.append(row)
    else:
        similarity_matrix.append([0] * len(synsets_sample))

import numpy as np
import seaborn as sns

plt.figure(figsize=(8, 7))
sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='YlGn',
            xticklabels=words_sample, yticklabels=words_sample)
plt.title('Word Similarity Matrix (WordUp Similarity)')
plt.tight_layout()
plt.savefig("similarity_matrix.png", dpi=150)
print("    -> Saved: Saved similarity_matrix.png")

# Synset distribution
synset_counts = Counter()
for word in words_sample:
    synsets = wn.synsets(word)
    synset_counts[word] = len(synsets)

plt.figure(figsize=(10, 6))
plt.bar(synset_counts.keys(), synset_counts.values(), color='skyblue')
plt.ylabel('Number of Synsets')
plt.title('Polysemy - Number of Senses per Word')
plt.tight_layout()
plt.savefig("synset_distribution.png", dpi=150)
print("    -> Saved: Saved synset_distribution.png")

# Save results
results = pd.DataFrame({
    'Word': list(synset_counts.keys()),
    'Num Senses': list(synset_counts.values())
})
results.to_csv("word_senses.csv", index=False)
print("    -> Saved: Saved word_senses.csv")

print("
--- Lab 8 Knowledge Graphs WSD Execution Finished ---")
