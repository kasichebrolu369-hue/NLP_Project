# Lab 3: Neural PoS Tagging and NER using BiLSTM

## Objective
Implement deep learning-based sequence labeling systems using BiLSTM networks.

## Tasks

### Required Tasks
1. Build architecture components:
   - Embedding Layer
   - BiLSTM Layer
   - Dense Classification Layer

2. Train models for:
   - PoS tagging
   - NER

3. Use pre-trained embeddings:
   - GloVe
   - FastText

4. Compare:
   - Random embeddings
   - Pre-trained embeddings

5. Analyze training convergence

### Advanced Requirements
- [ ] Add attention mechanism
- [ ] Experiment with different embedding dimensions
- [ ] Implement dropout and regularization
- [ ] Compare BiLSTM with vanilla RNN

## Expected Outcome
Students should understand neural sequence modeling in NLP.

## Tools
- TensorFlow / PyTorch
- Keras
- NumPy
- Matplotlib

## Deliverables
- [ ] Model architecture (`bilstm_model.py`)
- [ ] Training curves (`training_analysis.py`)
- [ ] Performance comparison (`REPORT.md`)
- [ ] Technical report (`TECHNICAL_REPORT.md`)

## Dataset
- Penn Treebank for PoS
- CoNLL-2003 for NER

## References
- BiLSTM for Sequence Tagging: Huang et al. (2015)
- GloVe: Pennington et al. (2014)
