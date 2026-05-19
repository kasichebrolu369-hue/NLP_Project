# Lab 10: Research-Oriented NLP Mini Project

## Objective
Develop an end-to-end research-grade NLP application.

## Suggested Topics

Choose one of the following or propose an alternative:

- **Fake News Detection**: Multi-modal fake news identification
- **Hate Speech Detection**: Robust multilingual hate speech classifier
- **Cyber Threat Intelligence**: Extract threats from security reports
- **Legal Document Summarization**: Abstractive summarization for contracts
- **Medical NLP**: Clinical note analysis and coding
- **Policy Compliance Extraction**: Regulatory requirement identification
- **AI-based Resume Screening**: Intelligent candidate ranking
- **Financial Sentiment Analysis**: Market sentiment from earnings calls

## Mandatory Requirements

- [ ] Literature survey (min 15 papers)
- [ ] Dataset preprocessing and analysis
- [ ] Baseline implementation (classical ML)
- [ ] Advanced Transformer-based implementation
- [ ] Quantitative comparison and benchmarking
- [ ] Research paper style documentation (PDF)

## Advanced Requirements

- [ ] Explainable AI (XAI) / Interpretability
- [ ] Attention visualization
- [ ] Adversarial robustness testing
- [ ] Bias analysis and mitigation
- [ ] Model compression (quantization/distillation)

## Expected Outcome
Students should demonstrate research-level NLP implementation capability.

## Deliverables

- [ ] **Working Prototype** (GitHub repository)
- [ ] **Research Paper Format Report** (PDF, 10-15 pages)
  - Abstract
  - Introduction
  - Related Work
  - Methodology
  - Experiments
  - Results
  - Analysis & Discussion
  - Conclusion
  - References
  
- [ ] **Source Code** (well-documented, reproducible)
  - `data_preprocessing.py`
  - `baseline_model.py`
  - `transformer_model.py`
  - `evaluation.py`
  - `visualization.py`
  
- [ ] **Presentation Slides** (PDF)
- [ ] **Viva Preparation Notes**

## Evaluation Rubric

| Component | Weightage |
|-----------|-----------|
| Implementation Complexity | 25% |
| Correctness and Performance | 20% |
| Experimental Analysis | 20% |
| Research Depth | 15% |
| Documentation and Report | 10% |
| Viva and Demonstration | 10% |

## Recommended Datasets

- CoNLL-2003
- IMDB Reviews
- AG News
- WikiText
- Universal Dependencies
- SQuAD
- SNLI
- Multi30K
- Hate Speech Datasets
- Twitter Sentiment Dataset

## Tools Stack

- **Core**: Python 3.8+
- **Deep Learning**: PyTorch / TensorFlow
- **NLP**: Transformers, spaCy, NLTK
- **ML**: Scikit-learn
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Metrics**: seqeval, rouge, BLEU

## Project Timeline

- **Week 1-2**: Literature review, dataset exploration, baseline
- **Week 3-4**: Core model development, training pipeline
- **Week 5-6**: Experimentation, hyperparameter tuning, analysis
- **Week 7-8**: Documentation, report writing, presentation prep

## References
- Recent NLP papers (2022-2024)
- Transformer architectures
- Domain-specific benchmarks
- Evaluation methodologies

## Starting Template

```python
# project_structure.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
import pandas as pd

# 1. Load and preprocess data
# 2. Create custom dataset class
# 3. Initialize pre-trained model
# 4. Training loop
# 5. Evaluation metrics
# 6. Visualization and analysis
```
