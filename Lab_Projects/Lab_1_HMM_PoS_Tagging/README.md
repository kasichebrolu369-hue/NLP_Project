# Lab 1: Advanced PoS Tagging using Hidden Markov Models and Viterbi Decoding

## Objective
Implement an advanced probabilistic Part-of-Speech tagging system using Hidden Markov Models (HMM) and Viterbi decoding.

## Tasks

### Required Tasks
1. Build a custom HMM-based PoS tagger from scratch
   - Transition Probability Matrix
   - Emission Probability Matrix
   - Viterbi Decoding Algorithm

2. Train the model on:
   - Brown Corpus or
   - Universal Dependencies Dataset

3. Compare performance with:
   - NLTK PoS Tagger
   - spaCy Tagger

4. Perform error analysis for:
   - Unknown words
   - Ambiguous words
   - Morphologically rich tokens

### Advanced Requirements
- [ ] Add Laplace smoothing
- [ ] Handle Out-of-Vocabulary (OOV) words
- [ ] Generate confusion matrix and accuracy metrics
- [ ] Compare greedy decoding vs Viterbi decoding

## Expected Outcome
Students should develop a statistically sound sequence labeling system with quantitative evaluation.

## Tools
- Python
- NumPy
- NLTK
- Pandas
- Matplotlib

## Deliverables
- [ ] Source code (`hmm_pos_tagger.py`)
- [ ] Technical report (`REPORT.md`)
- [ ] Performance evaluation table (`evaluation_results.csv`)
- [ ] Error analysis (`error_analysis.py`)

## Dataset
- Brown Corpus (NLTK)
- Universal Dependencies Dataset

## References
- Viterbi Algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
- HMM PoS Tagging: Manning & Schuetze (1999)
