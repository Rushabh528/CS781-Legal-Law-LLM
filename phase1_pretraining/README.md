# Phase 1 — Legal Language Model Pretraining

**Task:** given a 512-word legal prompt, generate a 128-word continuation.
**Primary metric:** BERTScore F1 (secondary: ROUGE-L F1, BLEU-4).
**Output:** `prediction.json` = `{test_id: generated_text}`, zipped at the root of `submission.zip`.

Four architectures are implemented in raw PyTorch and trained from random initialisation. The Hugging Face `Trainer`
is used only for the training loop.

| Notebook | Architecture | Parameters |
|---|---|---|
| `decoder_only.ipynb` | Decoder-only causal Transformer | 50.94M |
| `encoder_decoder.ipynb` | Encoder-decoder Transformer | 52.25M |
| `cnn_attention.ipynb` | CNN + causal self-attention hybrid | 71.04M |
| `ssm.ipynb` | Diagonal state-space LM (S4D-inspired) | 67.28M |

## Shared pipeline (all four notebooks)

1. **Data.** The four Phase-1 training sources are streamed from `Exploration-Lab/CS-781-Capstone` with `ijson` and mixed
   with fixed sampling fractions (seed 42): Supreme Court 1.00, High Courts 0.30, Acts 1.00, Tribunals 0.50. The mixed
   corpus is written to `mixed_legal_corpus.jsonl`.
2. **Tokenizer.** A byte-level BPE (`ByteLevelBPE`, implemented in the notebook) with a 16,000-token vocabulary, trained
   on up to 5M characters per source. Special tokens: `<pad>`, `<unk>`, `<s>`, `</s>`. The trained file is
   `tokenizer/legal_bpe.json`.
3. **Token budget.** Documents are tokenized and assigned to 200M training tokens and 5M validation tokens, with `</s>`
   between documents. The result is cached in `tokenized_data.npz`.
4. **Blocks.** The token stream is cut into blocks of 2,048 tokens (97,656 training / 2,441 validation blocks).
5. **Training.** AdamW, learning rate 3e-4, 500 warm-up steps, cosine schedule, fp16, 3 epochs. Periodic evaluation and
   checkpointing with the HF `Trainer`. A short throughput benchmark estimates the run time and restores the initial
   weights before training.
6. **Evaluation.** Validation perplexity and training/validation loss curves.
7. **Generation.** Each test prompt is continued until 128 words are produced (temperature + top-p sampling;
   settings per model below). The continuation is cut to 128 words and written to `prediction.json` / `submission.zip`.

### Run flags (configuration cell, section 3)

| Flag | Meaning |
|---|---|
| `TRAIN_NEW_TOKENIZER` | `True`: train the BPE tokenizer; `False`: load `TOKENIZER_INPUT_PATH` |
| `TOKENIZED_CACHE_INPUT` | path of an existing `tokenized_data.npz` (skips re-tokenization), or `None` |
| `COUNT_CORPUS_TOKENS` | optionally count the tokens of the whole corpus (slow) |
| `RESUME_FROM_CHECKPOINT` | resume training from a `Trainer` checkpoint, or `None` |

## Architectures

### Decoder-only Transformer (`decoder_only.ipynb`)
- Token embedding + learned position embedding (context 2,048), dropout 0.1.
- 8 pre-norm blocks: LayerNorm → causal multi-head self-attention (8 heads, fused QKV projection) → residual →
  LayerNorm → MLP (640 → 2,560 → 640, GELU) → residual.
- Final LayerNorm and an LM head tied to the token embedding.
- Batch 3 × gradient accumulation 2, weight decay 0.1. Generation: temperature 0.7, top-p 0.95.

### Encoder-decoder Transformer (`encoder_decoder.ipynb`)
- Every 2,048-token block is split into a 1,024-token source (encoder) and a 1,024-token target (decoder, teacher forcing).
- 6 encoder + 6 decoder layers, d_model 512, 8 heads × 64, FFN 2,048 (ReLU), dropout 0.1.
- RMSNorm (scale only) and a bucketed relative-position bias (32 log-spaced buckets, maximum distance 128, one learned
  bias per head). Bias-free projections; tied embeddings.
- Batch 2 × gradient accumulation 2, weight decay 0.1. Generation: temperature 0.1, top-p 0.95.

### CNN + causal self-attention (`cnn_attention.ipynb`)
- Token embedding + learned position embedding (context 2,048).
- 8 hybrid blocks, each with three pre-norm residual sub-layers: causal depth-preserving `Conv1d` (kernel 3, left
  padding) → causal multi-head self-attention (8 heads) → FFN (640 → 2,560 → 640, GELU).
- Final LayerNorm and an LM head.
- Batch 4 × gradient accumulation 3, weight decay 0.01. Generation: temperature 0.8, top-p 0.9.

### Diagonal SSM (`ssm.ipynb`)
- Token embedding and 10 residual blocks: LayerNorm → `DiagonalSSM` → residual.
- `DiagonalSSM`: input projection to 2 × 2,100 channels (signal `u` and gate `z`); a learned diagonal state-space
  recurrence with state dimension 16 per channel (learned `A` (log), `B`, `C`, step size `Δ` and skip `D`), gated by
  `z`, then an output projection back to 700.
- Final LayerNorm and an LM head.
- Batch 4 × gradient accumulation 3, weight decay 0.01. Generation: temperature 0.75, top-p 0.95; prompts that do not reach 128 words
  are retried with a larger token budget.

## Submissions (Codabench, BERTScore F1)

| Submission | Model | BERTScore F1 |
|---|---|---|
| decoder | Decoder-only Transformer | **0.8245** (best) |
| ssm | Diagonal SSM | 0.8223 |
| cnn_attention (1) | CNN + attention | 0.8208 |
| cnn_attention (2) | CNN + attention | 0.8198 |
| encoder_decoder | Encoder-decoder Transformer | 0.8079 |

Each notebook saves its trained weights under `/kaggle/working/cs781_phase1/`. Phases 2–7 load the four Phase-1
backbones from the checkpoint paths set in their configuration sections.
