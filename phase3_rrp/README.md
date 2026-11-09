# Phase 3 — Rhetorical Role Prediction (RR)

**Task:** label every sentence of a legal document (Corporate Law `CL` and Information Technology `IT` judgments) with
one of 13 rhetorical roles.
**Primary metric:** Macro-F1 over the 13 labels (secondary: Macro-Precision, Macro-Recall).
**Output:** `prediction.json` = `{doc_id: [label, label, ...]}`, with exactly one label per sentence, in order.

Notebook: `phase3_rrp.ipynb`. Backbones registered: `decoder`, `encdec`, `cnn`, `ssm`.

## Pipeline

1. **Data.** `CL_train` / `IT_train`, `CL_dev` / `IT_dev`, `CL_test` / `IT_test` and `label_vocab.json` are downloaded
   from `Exploration-Lab/CS-781-Capstone`. Both domains are trained together. The notebook validates the label set,
   sentence/label counts, and test IDs and order, and reports dev metrics per domain.
2. **Sentences.** Each sentence is tokenized with the Phase-1 tokenizer as `<s> sentence </s>`, truncated to
   `MAX_SENT_TOKENS = 128`.
3. **Chunks.** Documents are split into chunks of `MAX_SENTS_PER_CHUNK = 64` consecutive sentences. Each chunk keeps its
   document ID and its position in the document.
4. **Batching.** `RR_BATCH_SIZE = 2` chunks per batch, padded at the sentence and token level.

## Model

`backbone.encode()` on every sentence → **sentence pooling** → optional **BiGRU document context** → linear classifier over 13 labels.

- **Sentence pooling:** `last_token` (the last real token), `mean` (masked mean) or `attention` (learned attention weights over tokens).
- **Context:** `use_context=True` runs a bidirectional GRU (hidden size D/2 per direction) over the sentence vectors of
  a chunk, so each sentence sees its neighbours.
- **Loss:** cross-entropy, optionally with class weights computed from the training label frequencies (`use_class_weights`).
- **Inference:** predictions for each chunk are written back to their sentence positions, giving exactly one label per
  original sentence in order.

## Adaptation techniques

| Key | Technique | Implementation |
|---|---|---|
| `lora` | LoRA | `peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05)` |
| `rslora` | Rank-stabilised LoRA | same, with `use_rslora=True` (scale α/√r) |
| `ia3` | IA³ | `peft.IA3Config` on the attention / feed-forward layers of each backbone |
| `lorma` | **LORMA** | `LowRankMulLinear`: `y = (I + (α/r)·B·A)·W₀x + b`, r = 16, α = 32 |
| KD | **Knowledge distillation** | `DistillationExperiment` (section 15) |

The adapters target, per backbone: decoder `qkv`, `out_proj`, `fc_in`, `fc_out`; CNN `q_proj`, `k_proj`, `v_proj`,
`out_proj`; SSM `in_proj`, `out_proj`; encoder-decoder `q`, `k`, `v`, `o`, `wi`, `wo`. The pooling, BiGRU and
classifier are always trainable.

**Training.** AdamW, adapter learning rate 5e-4, head learning rate 3e-4, weight decay 0.01. Batch 2 chunks × gradient
accumulation 4, 8 epochs, mixed precision, gradient clipping 1.0. The best epoch is chosen by dev Macro-F1.

**Knowledge distillation.**
- Teacher: decoder + rsLoRA + BiGRU, last-token pooling (`KD_TEACHER_CHECKPOINT`).
- Student: the first 4 blocks of the Phase-1 decoder (`STUDENT_LAYERS = 4`) with rsLoRA and the same head.
- Loss: `(1 − α)·CE(gold) + α·T²·KL(teacher ‖ student)` on temperature-scaled softmaxes, with T = 2.0 and α = 0.5.
- Optimiser: lr 5e-4, 8 epochs.

## How to run

Set the paths in section 2 and choose `exp_cfg` in section 11, then *Run All*. Run flags: `RUN_PREFLIGHT`,
`RUN_TINY_OVERFIT`, `RUN_TRAINING`, `RUN_KD`, `RUN_INFERENCE`, `INFERENCE_SOURCE` (see the root README). To reproduce the
best submission, uncomment the `cnn_rslora_bigru_mean` configuration in section 11.

## Submissions (Codabench, Macro-F1)

| Submission | Backbone | Adaptation | Pooling | Context | Macro-F1 |
|---|---|---|---|---|---|
| cnn_rslora_bigru_mean | cnn | rsLoRA | mean | BiGRU | **0.4306** (best) |
| dec_rslora_bigru_attn | decoder | rsLoRA | attention | BiGRU | 0.4284 |
| dec_rslora_bigru | decoder | rsLoRA | last_token | BiGRU | 0.4283 |
| cnn_rslora_bigru_attn | cnn | rsLoRA | attention | BiGRU | 0.4207 |
| cnn_attn_rslora_bigru | cnn | rsLoRA | last_token | BiGRU | 0.4207 |
| cnn_rslora_bigru_mean_unweighted | cnn | rsLoRA | mean | BiGRU, no class weights | 0.4201 |
| decoder_lora_lasttoken_bigru | decoder | LoRA | last_token | BiGRU | 0.4151 |
| dec_lorma_bigru_lasttoken | decoder | LORMA | last_token | BiGRU | 0.4026 |
| KD_DEC_4LAYER | decoder (4-layer student) | KD + rsLoRA | last_token | BiGRU | 0.3987 |
| ssm_rslora_bigru | ssm | rsLoRA | last_token | BiGRU | 0.3606 |
| decoder_lora_lasttoken_nocontext | decoder | LoRA | last_token | none | 0.3545 |

All runs use lr 5e-4, r = 16, α = 32, 8 epochs and class weights, unless noted.
