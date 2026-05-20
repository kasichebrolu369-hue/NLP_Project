"""
Course: Natural Language Processing
Academic Year: 2025-2026
Student Portfolio Submission

Lab 9 Multilingual NLP
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("
--- Starting Lab 9 Multilingual NLP ---")

# 1. Multilingual data
print("\n[*] Preparing multilingual data...")

multilingual_data = {
    'English': [
        "The cat sits on the mat",
        "I love learning languages",
        "Machine learning is powerful",
    ],
    'Spanish': [
        "El gato se sienta en la estera",
        "Me encanta aprender idiomas",
        "El aprendizaje automático es poderoso",
    ],
    'French': [
        "Le chat s'assoit sur le tapis",
        "J'aime apprendre les langues",
        "L'apprentissage automatique est puissant",
    ],
    'German': [
        "Die Katze sitzt auf der Matte",
        "Ich liebe es, Sprachen zu lernen",
        "Maschinelles Lernen ist mächtig",
    ],
}

print(f"   Languages: {list(multilingual_data.keys())}")
print(f"   Sentences per language: {len(multilingual_data['English'])}")

# 2. Language statistics
print("\n[*] Language statistics...")

stats = {}
for lang, sentences in multilingual_data.items():
    total_words = sum(len(s.split()) for s in sentences)
    avg_length = total_words / len(sentences)
    vocab_size = len(set(' '.join(sentences).split()))
    
    stats[lang] = {
        'Total Words': total_words,
        'Avg Sentence Length': avg_length,
        'Vocabulary Size': vocab_size
    }
    
    print(f"\n   {lang}:")
    print(f"      Total words: {total_words}")
    print(f"      Avg sentence length: {avg_length:.2f}")
    print(f"      Vocabulary size: {vocab_size}")

# 3. Cross-lingual similarity (mock implementation)
print("\n[*] Cross-lingual similarity...")

# Language pairs and expected similarities
language_pairs = [
    ('English', 'Spanish'),
    ('English', 'French'),
    ('French', 'German'),
    ('Spanish', 'German'),
]

# Mock similarity scores (in real scenario, compute from learned embeddings)
mock_similarities = {
    ('English', 'Spanish'): 0.72,
    ('English', 'French'): 0.68,
    ('French', 'German'): 0.65,
    ('Spanish', 'German'): 0.60,
}

print("\n   Cross-lingual Document Similarity:")
for pair, sim in mock_similarities.items():
    print(f"      {pair[0]} ↔ {pair[1]}: {sim:.4f}")

# 4. Language family relationships
print("\n[*] Language family analysis...")

language_families = {
    'Germanic': ['English', 'German'],
    'Romance': ['Spanish', 'French'],
}

print("\n   Language Families:")
for family, languages in language_families.items():
    print(f"      {family}: {', '.join(languages)}")

# 5. Zero-shot transfer scenario
print("\n[*] Zero-shot transfer learning scenario...")

print("\n   Scenario: Train sentiment classifier on English, test on Spanish")
print("   Steps:")
print("      1. Train multilingual embedding model on parallel corpus")
print("      2. Fine-tune classifier on English sentiment data")
print("      3. Use same model for Spanish (zero-shot transfer)")
print("      4. Evaluate performance on Spanish hold-out set")

transfer_results = {
    'Language': ['English (Training)', 'Spanish (Zero-shot)', 'Performance Drop'],
    'Accuracy': [0.92, 0.78, '15.2%']
}

print("\n   Expected Results:")
for lang, acc in zip(transfer_results['Language'], transfer_results['Accuracy']):
    print(f"      {lang}: {acc}")

# 6. Visualizations
print("\n[*] Generating visualizations...")

# Language similarity heatmap
languages_list = ['English', 'Spanish', 'French', 'German']
similarity_matrix = np.array([
    [1.00, 0.72, 0.68, 0.55],
    [0.72, 1.00, 0.75, 0.60],
    [0.68, 0.75, 1.00, 0.65],
    [0.55, 0.60, 0.65, 1.00],
])

import seaborn as sns
plt.figure(figsize=(8, 7))
sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            xticklabels=languages_list, yticklabels=languages_list,
            vmin=0, vmax=1, square=True)
plt.title('Cross-Lingual Similarity Matrix')
plt.tight_layout()
plt.savefig("cross_lingual_similarity.png", dpi=150)
print("    -> Saved: Saved cross_lingual_similarity.png")

# Vocabulary size comparison
vocab_sizes = [stats[lang]['Vocabulary Size'] for lang in languages_list]
plt.figure(figsize=(10, 6))
plt.bar(languages_list, vocab_sizes, color=['blue', 'red', 'green', 'orange'])
plt.ylabel('Vocabulary Size')
plt.title('Vocabulary Size by Language')
plt.tight_layout()
plt.savefig("vocabulary_comparison.png", dpi=150)
print("    -> Saved: Saved vocabulary_comparison.png")

# Zero-shot transfer performance
plt.figure(figsize=(10, 6))
transfer_langs = ['English\n(Training)','Spanish\n(Zero-shot)', 'French\n(Zero-shot)', 'German\n(Zero-shot)']
transfer_accs = [0.92, 0.78, 0.75, 0.72]
colors = ['green'] + ['orange']*3

plt.bar(transfer_langs, transfer_accs, color=colors)
plt.ylabel('Accuracy')
plt.title('Zero-Shot Transfer Learning Performance')
plt.ylim([0.6, 1.0])
for i, acc in enumerate(transfer_accs):
    plt.text(i, acc + 0.02, f"{acc:.2%}", ha='center')
plt.tight_layout()
plt.savefig("zero_shot_transfer.png", dpi=150)
print("    -> Saved: Saved zero_shot_transfer.png")

# Save results
results_df = pd.DataFrame({
    'Language': languages_list,
    'Vocabulary Size': vocab_sizes,
    'German-Romance Family': ['Germanic', 'Romance', 'Romance', 'Germanic']
})
results_df.to_csv("multilingual_analysis.csv", index=False)
print("    -> Saved: Saved multilingual_analysis.csv")

print("
--- Lab 9 Multilingual NLP Execution Finished ---")
