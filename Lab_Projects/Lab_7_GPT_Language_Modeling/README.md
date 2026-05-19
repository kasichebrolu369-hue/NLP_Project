# Lab 7: Advanced Language Modeling using GPT and Transformer Decoders

## Objective
Implement and evaluate autoregressive language models.

## Tasks

### Required Tasks
1. Train a miniature GPT-style decoder model

2. Perform:
   - Next-word prediction
   - Text generation
   - Beam search decoding

3. Compare:
   - Greedy decoding
   - Top-k sampling
   - Nucleus sampling

4. Evaluate generated text quality

### Advanced Requirements
- [ ] Perplexity analysis
- [ ] Hallucination analysis
- [ ] Toxicity detection in generated text
- [ ] Prompt engineering experiments

## Expected Outcome
Students should understand generative NLP systems and decoding strategies.

## Tools
- Transformers
- PyTorch
- NumPy
- Matplotlib

## Deliverables
- [ ] Generated outputs (`generated_samples.txt`)
- [ ] Decoding comparison report (`REPORT.md`)
- [ ] Perplexity analysis (`perplexity_analysis.py`)
- [ ] Language model implementation (`gpt_model.py`)

## Dataset
- WikiText-103 or similar

## References
- Language Models are Unsupervised Multitask Learners (GPT-2): Radford et al. (2019)
- Decoding strategies: Holtzman et al. (2019)
