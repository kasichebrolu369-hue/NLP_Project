"""
Course: Natural Language Processing
Academic Year: 2025-2026
Student Portfolio Submission

Lab 5 Distributional Semantics
"""


import numpy as np
from gensim.models import Word2Vec, FastText
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Generate sample corpus
CORPUS = [
    "the king and the queen live in the castle".split(),
    "the prince and the princess study at the school".split(),
    "the dog and the cat play in the park".split(),
    "machine learning is powerful for data analysis".split(),
    "artificial intelligence is transforming technology".split(),
    "deep learning networks process images and text".split(),
    "natural language processing understands human language".split(),
    "embeddings map words to vector spaces".split(),
] * 50

print("
--- Starting Lab 5 Distributional Semantics ---")

# 1. Train Word2Vec (Skip-Gram)
print("\n[*] Training word2vec (skip-gram)...")
w2v_sg = Word2Vec(CORPUS, vector_size=100, window=5, min_count=1, sg=1, epochs=10)
print(f"   ✓ Word2Vec Skip-Gram vocabulary size: {len(w2v_sg.wv)}")

# 2. Train Word2Vec (CBOW)
print("\n[*] Training word2vec (cbow)...")
w2v_cbow = Word2Vec(CORPUS, vector_size=100, window=5, min_count=1, sg=0, epochs=10)
print(f"   ✓ Word2Vec CBOW vocabulary size: {len(w2v_cbow.wv)}")

# 3. Train FastText
print("\n[*] Training fasttext...")
ft_model = FastText(CORPUS, vector_size=100, window=5, min_count=1, epochs=10)
print(f"   ✓ FastText vocabulary size: {len(ft_model.wv)}")

# 4. Word similarity analysis
print("\n[*] Word similarity analysis...")
test_pairs = [
    ("king", "queen"),
    ("king", "prince"),
    ("learning", "machine"),
    ("dog", "cat"),
]

for word1, word2 in test_pairs:
    if word1 in w2v_sg.wv and word2 in w2v_sg.wv:
        sim_sg = w2v_sg.wv.similarity(word1, word2)
        sim_cbow = w2v_cbow.wv.similarity(word1, word2)
        sim_ft = ft_model.wv.similarity(word1, word2)
        print(f"\n   {word1} vs {word2}:")
        print(f"      Skip-Gram: {sim_sg:.4f}")
        print(f"      CBOW:      {sim_cbow:.4f}")
        print(f"      FastText:  {sim_ft:.4f}")

# 5. Word analogies
print("\n[*] Word analogies...")
try:
    analogy = w2v_sg.wv.most_similar(positive=["king", "woman"], negative=["man"], topn=1)
    print(f"   king - man + woman ≈ {analogy[0][0]} (similarity: {analogy[0][1]:.4f})")
except:
    print("   (Analogy task requires sufficient vocabulary)")

# 6. Visualizations
print("\n[*] Generating visualizations...")

# Get word vectors
words = list(w2v_sg.wv.index_to_key[:50])
vectors = np.array([w2v_sg.wv[w] for w in words])

# PCA visualization
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)

plt.figure(figsize=(12, 10))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.6, s=100)
for i, word in enumerate(words):
    plt.annotate(word, xy=(vectors_2d[i, 0], vectors_2d[i, 1]), fontsize=9)
plt.title("Word Embeddings - PCA Projection (Skip-Gram)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("embeddings_pca.png", dpi=150)
print("    -> Saved: Saved embeddings_pca.png")

# TSNE visualization
print("   Computing t-SNE (this may take a moment)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1))
vectors_tsne = tsne.fit_transform(vectors)

plt.figure(figsize=(12, 10))
plt.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1], alpha=0.6, s=100)
for i, word in enumerate(words):
    plt.annotate(word, xy=(vectors_tsne[i, 0], vectors_tsne[i, 1]), fontsize=9)
plt.title("Word Embeddings - t-SNE Projection (Skip-Gram)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("embeddings_tsne.png", dpi=150)
print("    -> Saved: Saved embeddings_tsne.png")

# 7. Model comparison
print("\n[*] Embedding model comparison...")
comparison_data = {
    'Model': ['Skip-Gram', 'CBOW', 'FastText'],
    'Vocab Size': [len(w2v_sg.wv), len(w2v_cbow.wv), len(ft_model.wv)],
    'Vector Dim': [100, 100, 100],
}
comparison_df = pd.DataFrame(comparison_data)
comparison_df.to_csv("embedding_comparison.csv", index=False)
print("    -> Saved: Saved embedding_comparison.csv")

print("
--- Lab 5 Distributional Semantics Execution Finished ---")
