# Lab 2: Named Entity Recognition using Conditional Random Fields (CRF)

## Objective
Design a high-performance Named Entity Recognition system using CRF.

## Tasks

### Required Tasks
1. Use CoNLL-2003 Dataset

2. Extract handcrafted features:
   - Prefix/suffix
   - Word shape
   - Capitalization
   - Context window
   - POS tags

3. Train a CRF model for:
   - PERSON
   - LOCATION
   - ORGANIZATION
   - MISC entities

4. Evaluate using:
   - Precision
   - Recall
   - F1-score

5. Compare CRF performance with HMM-based sequence labeling

### Advanced Requirements
- [ ] Integrate domain adaptation
- [ ] Implement feature ablation study
- [ ] Perform cross-domain testing
- [ ] Visualize entity spans

## Expected Outcome
Students should understand feature engineering and probabilistic sequence labeling deeply.

## Tools
- sklearn-crfsuite
- spaCy
- seqeval
- Python

## Deliverables
- [ ] CRF implementation (`crf_ner.py`)
- [ ] Comparative analysis report (`REPORT.md`)
- [ ] Visualization results (`visualizations.py`)
- [ ] Feature ablation study (`feature_ablation.py`)

## Dataset
- CoNLL-2003 NER dataset

## References
- CRF Paper: Lafferty et al. (2001)
- Feature Engineering for NER: Nadeau & Sekine (2007)
