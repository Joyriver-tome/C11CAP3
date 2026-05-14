# C11_CAP3 Capstone Project

Dialogue summarization using a BERT encoder-decoder model on the SAMSum dataset.

---

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Approach](#approach)
4. [Results](#results)
5. [Limitations and Next Steps](#limitations-and-next-steps)
6. [How to Run](#how-to-run)
7. [References](#references)

---

## Overview

This project builds an abstractive text summarization model for Acme Communications. The goal is to automatically generate short summaries from multi-speaker chat conversations, reducing information overload in enterprise messaging.

The full pipeline follows the CRISP-DM methodology: data understanding, preprocessing, model training, evaluation, and analysis.

---

## Dataset

**SAMSum Corpus** (Gliwa et al., 2019)

- ~16,000 English messenger-style dialogues with human-written summaries
- Split: 14,732 train / 818 validation / 819 test
- Median dialogue length: 73 words
- Median summary length: 18 words (~4x compression)
- Most dialogues fit within BERT's 512-token limit (~99% are under 350 words)

---

## Approach

**Architecture**: `bert-base-uncased` encoder + `bert-base-uncased` decoder (Hugging Face `EncoderDecoderModel`)

The encoder reads the full dialogue and produces contextual hidden states. The decoder generates the summary token by token using cross-attention over those hidden states.

**Key preprocessing decisions**:
- `[CLS]` mapped to BOS (id=101), `[SEP]` mapped to EOS (id=102)
- Max encoder tokens: 512, max decoder tokens: 128
- Padding labels set to -100 (ignored in cross-entropy loss)

**Training setup**:

| Parameter | Value |
|---|---|
| Training samples | 500 (PoC subset) |
| Epochs | 3 |
| Learning rate | 5e-5 |
| Batch size | 4 |
| Warmup steps | 100 |
| Decoding | Beam search, width=4 |
| Best checkpoint | Selected by validation ROUGE-L |

---

## Results

Evaluated on the full SAMSum test set (819 samples).

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| **BERT-to-BERT (ours, 500 samples)** | **13.65%** | **1.88%** | **11.84%** |
| BERT-to-BERT (literature, full 14.7k) | 38.31% | 15.22% | 34.67% |
| BART-base (literature, full 14.7k) | 44.16% | 21.28% | 41.19% |
| BART-large (literature, full 14.7k) | 45.94% | 22.06% | 43.42% |

The model did not meet the target scores (ROUGE-1 >= 38%, ROUGE-L >= 30%). Outputs are hallucinated text unrelated to the input — for example, generating "john is going to his parents ." for every dialogue. Training loss decreased correctly (8.14 → 5.04), but the model learned language patterns rather than input-conditional generation.

**Root cause**: BERT's cross-attention layers are absent from the pretrained checkpoint and randomly initialized. 500 samples is not enough to train them from scratch.

---

## Limitations and Next Steps

| Priority | Action | Expected Impact |
|---|---|---|
| High | Train on full 14,732 samples | ROUGE-1 ~38% per literature |
| High | Switch to `facebook/bart-base` | Pre-trained cross-attention; better with less data |
| Medium | Add BERTScore evaluation | More meaningful than ROUGE alone |
| Medium | Add cosine LR decay | Stabler training on larger dataset |
