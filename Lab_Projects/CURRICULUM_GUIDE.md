# NLP Curriculum Implementation Guide

## Complete Advanced NLP Course Structure

This document provides comprehensive guidance for implementing the 10-lab NLP curriculum.

## Lab Progression & Dependencies

```
Lab 1 (HMM & Viterbi)
    ↓
Lab 2 (CRF) ← Requires probabilistic thinking from Lab 1
    ↓
Lab 3 (BiLSTM) ← Neural alternative to Labs 1-2
    ↓
Lab 4 (Transformers) ← Modern architecture foundation
    ↓
Lab 5 (Word Embeddings) ← Feeds into all downstream tasks
    ↓
Lab 6 (BERT Classification) ← Direct application of Labs 4-5
    ↓
Lab 7 (GPT Language Model) ← Generative Transformer model
    ↓
Lab 8 (Knowledge Graphs & WSD) ← Semantic understanding
    ↓
Lab 9 (Multilingual NLP) ← Cross-lingual extension
    ↓
Lab 10 (Research Project) ← Integration of all concepts
```

## Lab Overview Table

| Lab | Focus | Key Concepts | Difficulty | Hours |
|-----|-------|--------------|-----------|-------|
| 1 | Sequence Tagging | HMM, Viterbi, Smoothing | Intermediate | 30-40 |
| 2 | Feature Engineering | CRF, Feature Selection | Intermediate | 25-35 |
| 3 | Neural Networks | BiLSTM, Embeddings, RNN | Intermediate-Adv | 30-40 |
| 4 | Attention Mechanism | Self-Attention, Multi-head | Advanced | 35-45 |
| 5 | Representation Learning | word2vec, GloVe, FastText | Intermediate | 25-35 |
| 6 | Transfer Learning | Fine-tuning, BERT | Intermediate-Adv | 30-40 |
| 7 | Generative Models | Language Models, GPT | Advanced | 30-40 |
| 8 | Semantic Systems | WordNet, WSD, Graphs | Intermediate-Adv | 25-35 |
| 9 | Multilingual Systems | mBERT, XLM-R, Transfer | Advanced | 25-35 |
| 10 | Research Project | Full Pipeline, Publication | Advanced | 60-80 |

**Total**: ~275-385 hours (~6.8-9.6 weeks full-time, or 3-6 months part-time)

## Phase-wise Breakdown

### Phase 1: Foundations (Weeks 1-4)
**Objective**: Build probabilistic and initial neural foundations

**Lab 1**: HMM PoS Tagging
- Probability theory review
- Transition/emission matrices
- Viterbi algorithm implementation
- Comparison benchmarks

**Lab 2**: CRF for NER
- Feature engineering principles
- CRF fundamentals
- Sequence model evaluation
- Ablation studies

**Lab 3**: BiLSTM Sequence Labeling
- RNN/LSTM fundamentals
- Attention-free encoding
- Embedding initialization
- Training techniques

**Assessment**: Implement all three approaches for same task, compare

### Phase 2: Modern Architectures (Weeks 5-8)
**Objective**: Master Transformer-based systems

**Lab 4**: Transformer Architecture
- Self-attention mathematics
- Multi-head mechanisms
- Positional encoding
- Scaling/optimization

**Lab 5**: Distributional Semantics
- Embedding spaces
- Similarity metrics
- Analogical reasoning
- Domain-specific adaptations

**Lab 6**: BERT Fine-tuning
- Transfer learning principles
- Hyperparameter tuning
- Multiple downstream tasks
- Error analysis

**Assessment**: Fine-tune model on 2-3 different tasks

### Phase 3: Advanced Topics (Weeks 9-12)
**Objective**: Specialized NLP systems

**Lab 7**: Generative Modeling
- Autoregressive generation
- Decoding strategies
- Quality metrics
- Failure mode analysis

**Lab 8**: Semantics & Knowledge
- Lexical databases
- Disambiguation algorithms
- Graph-based reasoning
- Integration strategies

**Lab 9**: Multilingual Systems
- Cross-lingual transfer
- Code-switching
- Low-resource scenarios
- Fairness in multilingual models

**Assessment**: Build multilingual system with evaluation

### Phase 4: Capstone (Weeks 13-20)
**Objective**: Research-grade application

**Lab 10**: Research Project
- Literature review (15+ papers)
- Dataset curation
- Baseline implementation
- Advanced system development
- Paper writing (10-15 pages)
- Presentation & viva

**Assessment**: Full research project with publication-quality output

## Implementation Checklist

### Before Starting Labs
- [ ] Set up Python environment (3.8+)
- [ ] Install required libraries
- [ ] Organize directory structure
- [ ] Study required papers for each lab
- [ ] Review prerequisites

### For Each Lab
- [ ] Read comprehensive README
- [ ] Study reference materials
- [ ] Implement baseline
- [ ] Run experiments
- [ ] Document results
- [ ] Write report
- [ ] Prepare presentation

### Throughout Curriculum
- [ ] Maintain lab notebooks
- [ ] Keep Git commits organized
- [ ] Document all findings
- [ ] Save all results/models
- [ ] Create comparison tables
- [ ] Write thoughtful analyses

## Key Skills Progression

### Conceptual Understanding
```
Statistical Models → Neural Models → Transformer Models → Research Methods
```

### Implementation
```
Simple Scripts → Classes & Modules → System Integration → Production Code
```

### Evaluation
```
Basic Metrics → Comprehensive Metrics → Statistical Testing → Research-grade Analysis
```

## Common Challenges & Solutions

### Lab 1: Understanding Viterbi
- **Issue**: Backpointer logic confusing
- **Solution**: Implement step-by-step with visualization

### Lab 2: Feature Engineering for CRF
- **Issue**: Which features to use?
- **Solution**: Start with baseline features, then ablation study

### Lab 3: Training BiLSTM
- **Issue**: Model not converging
- **Solution**: Check learning rate, embedding quality, data normalization

### Lab 4: Implementing Attention
- **Issue**: Dimensionality mismatch in multi-head attention
- **Solution**: Carefully document tensor shapes at each step

### Lab 5: Embedding Quality
- **Issue**: Poor downstream performance
- **Solution**: Evaluate on multiple tasks, check preprocessing

### Lab 6: BERT Fine-tuning
- **Issue**: Catastrophic forgetting
- **Solution**: Use lower learning rates, warmup scheduler

### Lab 7: Text Generation Quality
- **Issue**: Repetitive or incoherent outputs
- **Solution**: Experiment with different decoding strategies

### Lab 8: WSD Evaluation
- **Issue**: Limited labeled data
- **Solution**: Use SemEval datasets, cross-validation

### Lab 9: Multilingual Transfer
- **Issue**: One language dominates others
- **Solution**: Balance training data, adaptation techniques

### Lab 10: Research Project
- **Issue**: Scope too large
- **Solution**: Define clear problem statement, limit scope initially

## Recommended Reading Order

### Foundational Papers (Before Labs)
1. Sequence Labeling Tutorial (Jurafsky & Martin)
2. Attention is All You Need (Vaswani et al., 2017)
3. BERT Paper (Devlin et al., 2018)

### Per-Lab Reading
- Lab 1: Manning & Schuetze (1999) on HMM tagging
- Lab 2: Lafferty et al. (2001) on CRF
- Lab 3: Huang et al. (2015) on BiLSTM
- Lab 4: More on Transformers
- Lab 5: Mikolov et al. (2013), Pennington et al. (2014)
- Lab 6: BERT and transfer learning papers
- Lab 7: GPT papers (Radford et al.)
- Lab 8: Fellbaum (1998) WordNet, WSD surveys
- Lab 9: Multilingual BERT, XLM-R papers
- Lab 10: Domain-specific papers

## Hardware Requirements

### Minimum
- CPU: 4-core processor
- RAM: 8 GB
- Storage: 50 GB (for models + datasets)

### Recommended
- CPU: 8+ cores
- RAM: 16 GB+
- GPU: NVIDIA GPU with CUDA support (optional but helpful)
- Storage: 100+ GB

### For Lab 10 especially
- GPU highly recommended for training large models
- Consider cloud platforms (AWS, GCP, Colab) if local GPU unavailable

## Expected Outputs Per Lab

### Lab 1
```
1_hmm_pos_tagger/
├── hmm_pos_tagger.py (400+ lines)
├── evaluation_results.csv
├── error_analysis.md
├── REPORT.md
└── visualizations/
    ├── confusion_matrix.png
    └── performance_comparison.png
```

### Lab 2
```
2_crf_ner/
├── crf_ner.py (300+ lines)
├── feature_engineering.py (200+ lines)
├── evaluation_results.csv
├── ablation_study.md
├── REPORT.md
└── visualizations/
```

... (similar structures for other labs)

### Lab 10
```
10_research_project/
├── RESEARCH_PAPER.pdf (10-15 pages)
├── source_code/
│   ├── data_preprocessing.py
│   ├── baseline_model.py
│   ├── advanced_model.py
│   ├── evaluation.py
│   └── visualization.py
├── datasets/
│   └── processed_data.csv
├── results/
│   ├── baseline_results.json
│   ├── model_results.json
│   └── visualizations/
├── README.md
└── PRESENTATION.pdf
```

## Grading Rubric Application

For each lab (except 10 which has stricter criteria):

```
Implementation Complexity (25%)
- Code quality: 10%
- Feature completeness: 10%
- Optimization: 5%

Correctness & Performance (20%)
- Correct algorithm: 10%
- Benchmark comparison: 10%

Experimental Analysis (20%)
- Thorough experiments: 10%
- Ablation studies: 10%

Research Depth (15%)
- Literature review: 8%
- Novelty/insights: 7%

Documentation (10%)
- Code comments: 5%
- Report clarity: 5%

(Lab 10 adds: Viva 10%)
```

## Submission Guidelines

- **Format**: GitHub repository with clear structure
- **Code**: Well-commented, follows PEP 8
- **Reports**: PDF or Markdown with visualizations
- **Results**: Reproducible with provided scripts
- **Deadlines**: As per course schedule

## Success Indicators

By lab completion, students should be able to:

- ✅ Articulate the problem solved
- ✅ Explain their implementation choices
- ✅ Defend their experimental results
- ✅ Compare with baselines and literature
- ✅ Identify limitations and future work
- ✅ Present findings professionally

---

**Version**: 1.0
**Last Updated**: May 2026
**Contact**: [Course Instructor]
**Resources**: https://github.com/[organization]/NLP_Labs
