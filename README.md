## Table of Contents
- [Project Overview](#project-overview)
- [Methods](#methods)
- [Results](#results)

---

## Project Overview

This is a proof-of-concept for **automatic conversation summarization**.

Acme Communications users were losing important information in long group-chat threads. The goal was to build a model that reads a multi-speaker dialogue and generates a short, clear summary.

- **Dataset:** SAMSum — 16,000 chat dialogues with human-written summaries
- **Model:** BERT Encoder-Decoder (`bert-base-uncased`)
- **Training size:** 500 samples (PoC only)
- **Hardware:** GPU (CUDA 12.8)

---

## Methods

**1. Data Analysis**
- Median dialogue length: 73 words
- Median summary length: 18 words (~4x compression)
- Most dialogues have 2–3 speakers

**2. Preprocessing**
- Tokenized with `BertTokenizer`
- Max input: 512 tokens / Max output: 128 tokens
- Padding tokens replaced with `-100` so they are ignored in loss

**3. Model**
- Encoder and decoder both initialized from `bert-base-uncased`
- Cross-attention layers are randomly initialized (not in the original checkpoint)
- Decoding: beam search, width 4

**4. Training**
- 3 epochs, batch size 4, learning rate 5e-5
- Mixed precision (FP16)
- Total training time: ~4.4 minutes

---

## Results

| Metric | Target | Actual |
|--------|--------|--------|
| ROUGE-1 | ≥ 38% | 13.66% |
| ROUGE-2 | — | 2.06% |
| ROUGE-L | ≥ 30% | 12.03% |

**Why the scores are low:**
The model missed both targets. The main reason is that the 12 cross-attention layers — which let the decoder read the encoder output — start from random weights. 500 training samples are not enough to learn them properly, so the model ignores the input and generates unrelated text.

**Comparison with literature (full 14,732-sample training):**

| Model | ROUGE-1 | ROUGE-L |
|-------|---------|---------|
| BERT-to-BERT (ours, 500 samples) | 13.66% | 12.03% |
| BERT-to-BERT (full data) | 38.31% | 34.67% |
| BART-large (full data) | 45.94% | 43.42% |

**To improve:**
- Train on the full dataset (14,732 samples)
- Use a pre-trained seq2seq model like `facebook/bart-large-cnn` to avoid the cold-start problem
