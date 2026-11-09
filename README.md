# CS781 Capstone — Legal Language Models from Scratch

**Pandya Rushabh Bhavesh (230732)** · CS781, IIT Kanpur

This repository contains the code for every Codabench submission of the seven capstone phases. In Phase 1, four legal
language models with different architectures are pretrained from scratch on Indian legal text, using a tokenizer
trained on the same corpus. Phases 2–7 adapt those four backbones to downstream legal tasks: named entity recognition,
rhetorical roles, judgment prediction, statute identification, summarization and prior-case retrieval. No external
pretrained weights or embedding models are used.

Each phase folder has one or more Kaggle notebooks and a README that describes the task, the architecture, the
adaptation techniques, the training settings and every submission.

## Repository structure

```text
CS781-Legal-Law-LLM/
├── README.md
├── requirements.txt
├── tokenizer/
│   └── legal_bpe.json              # byte-level BPE tokenizer (16,000 tokens) shared by every phase
├── phase1_pretraining/             # Legal language-model pretraining (4 architectures)
│   ├── decoder_only.ipynb
│   ├── encoder_decoder.ipynb
│   ├── cnn_attention.ipynb
│   ├── ssm.ipynb
│   └── README.md
├── phase2_lner/                    # Legal Named Entity Recognition
├── phase3_rrp/                     # Rhetorical Role Prediction
├── phase4_cjpe/                    # Court Judgment Prediction
├── phase5_lsi/                     # Legal Statute Identification
├── phase6_summarization/           # Abstractive Summarization
└── phase7_pcr/                     # Prior Case Retrieval
```

Model weights, tokenized caches and submission archives are not included (see `.gitignore`).

## Results summary (best Codabench submission per phase)

| Phase | Task | Primary metric | Best submission | Score |
|---|---|---|---|---|
| 1 | Legal LM pretraining (128-word continuation) | BERTScore F1 | Decoder-only Transformer | **0.8245** |
| 2 | Legal NER (character spans, 12 types) | Strict Macro-F1 | Decoder, full fine-tuning | **0.4362** |
| 3 | Rhetorical role prediction (13 labels) | Macro-F1 | CNN+Attention, rsLoRA, BiGRU context, mean pooling | **0.4306** |
| 4 | Court judgment prediction (binary) | Macro-F1 | KD: 4-layer decoder student + LORMA (teacher: decoder full FT) | **0.5944** |
| 5 | Legal statute identification (multi-label, 100) | Macro-F1 | CNN+Attention, LORMA, mean pooling | **0.2846** |
| 6 | Abstractive summarization | BERTScore F1 | Decoder, LORMA | **0.7876** |
| 7 | Prior case retrieval (top 7) | Micro-F1@7 | KD: 4-layer decoder student + LORMA (teacher: decoder full FT) | **0.1636** |

All submissions of each phase are listed in the phase README.

## Phase-1 backbones

All four are implemented in raw PyTorch and use the same tokenizer (`tokenizer/legal_bpe.json`, vocabulary 16,000,
special tokens `<pad>`=0, `<unk>`=1, `<s>`=2, `</s>`=3).

| Key (Phases 2–7) | Architecture | Configuration | Parameters |
|---|---|---|---|
| `decoder` | Decoder-only causal Transformer | 8 pre-norm blocks, d=640, 8 heads, FFN 2560 (GELU), learned positions, context 2048, tied embeddings | 50.94M |
| `encdec` | Encoder-decoder Transformer | 6 + 6 layers, d=512, 8 heads × 64, FFN 2048 (ReLU), RMSNorm, bucketed relative-position bias, 1024 + 1024 tokens | 52.25M |
| `cnn` | CNN + causal self-attention hybrid | 8 hybrid blocks (causal Conv1d k=3 → causal attention → FFN ×4), d=640, 8 heads, learned positions, context 2048 | 71.04M |
| `ssm` | Diagonal state-space LM (S4D-inspired) | 10 residual blocks, d=700, state dim 16, expansion 3, gated output | 67.28M |

Each Phase-2–7 notebook rebuilds the backbone classes it uses (Phase 2: `decoder` and `ssm`; Phases 3–7: all four) and exposes `encode(input_ids, attention_mask) → [B, L, D]` (the final
hidden states before the LM head), which the task-specific heads use.

## Adaptation techniques used across phases

| Technique | Phases | Implementation |
|---|---|---|
| Full fine-tuning | 2, 4, 5, 7 | all backbone and head parameters trainable |
| Partial fine-tuning | 2, 4, 5, 7 | only the last layers of the backbone (phase-specific) + head trainable |
| LoRA | 3, 6 | `peft.LoraConfig` (r = 16, α = 32, dropout 0.05) |
| rsLoRA | 3, 4 | `peft.LoraConfig(use_rslora=True)` |
| IA³ | 3, 7 | `peft.IA3Config` on the registered attention / feed-forward layers |
| **LORMA** (compulsory) | 2–7 | Phase 2: additive low-rank update. Phases 3–7: `LowRankMulLinear`, a multiplicative low-rank update `y = (I + (α/r)·B·A)·W₀x + b`, with `B = 0` at initialisation so training starts from the pretrained layer |
| **Knowledge distillation** (compulsory) | 2–7 | a smaller student (first 4 decoder blocks in Phases 3–7; 6 blocks in Phase 2) trained on gold labels + teacher outputs with temperature T = 2 |

## How to run

The notebooks were written for **Kaggle GPU notebooks**.

1. **Data.** Every notebook downloads its split from the Hugging Face dataset `Exploration-Lab/CS-781-Capstone`. Add a
   Kaggle secret named `hf_token_rp` that holds a Hugging Face token with access to this dataset. To run outside Kaggle,
   replace the `UserSecretsClient().get_secret("hf_token_rp")` line with your own token source.
2. **Dependencies.** `pip install -r requirements.txt`. Each notebook also installs what it needs in its first cell.
3. **Phase 1.** Run a notebook in `phase1_pretraining/`. It mixes the corpus, tokenizes, trains, computes validation
   perplexity, generates the 128-word continuations and writes `submission.zip`.
4. **Phases 2–7.** Attach the tokenizer and the Phase-1 checkpoints as Kaggle datasets, and set their paths in the
   configuration section of the notebook:
   - Phase 2: the tokenizer cell in section 3 and the backbone cells in section 6
   - Phases 3–7: section 2, "Paths and configuration"

   Then *Run All*. In Phases 3–7 the run flags in section 2 decide what a run does:

   | Flag | Meaning |
   |---|---|
   | `RUN_PREFLIGHT` | lightweight checks: strict checkpoint loading, `encode()` shapes, data/collator, adaptation parameter counts, forward/backward, submission format |
   | `RUN_TINY_OVERFIT` | overfit a handful of examples as a sanity check |
   | `RUN_TRAINING` | train the configuration `exp_cfg` (section 11) |
   | `RUN_KD` | knowledge distillation (section 15) |
   | `RUN_INFERENCE`, `INFERENCE_SOURCE` | predict the test set with the trained model (`"main"`) or the KD student (`"kd"`) and write `prediction.json` + `submission.zip` |

   In Phases 2 and 4–7 the default configuration reproduces that phase's best submission. In Phase 3 the best
   configuration (`cnn_rslora_bigru_mean`) is one of the commented `exp_cfg` blocks in section 11. To reproduce another
   submission, pick its configuration from the phase README.

In Phases 2–7 every run writes `best_model.pt`, its metrics and its submission into its own run directory under
`/kaggle/working/`. Every `submission.zip` contains a single `prediction.json` at the archive root, as Codabench requires.
