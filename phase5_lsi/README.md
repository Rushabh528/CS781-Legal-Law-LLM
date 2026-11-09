# Phase 5 — Legal Statute Identification (LSI)

**Task:** multi-label classification. Each case cites one or more of 100 statute IDs.
**Primary metric:** Macro-F1 over statutes (secondary: Micro-F1, subset accuracy).
**Output:** `prediction.json` = `[{"id": case_id, "pred": [statute_id, ...]}, ...]`, with unique integer IDs, one object
per test case, at the root of `submission.zip`.

Notebook: `phase5_lsi.ipynb`. Backbones registered: `decoder`, `encdec`, `cnn`, `ssm`.

## Pipeline

1. **Data.** `Phase_5_LSI/Train/train.json`, `Train/dev.json` and `Test/test.json` from `Exploration-Lab/CS-781-Capstone`.
   Statute IDs are mapped to 100 output indices, and the notebook checks IDs, labels and test order.
2. **Long documents.** Each case is split into overlapping 512-token chunks (stride 384), and the first
   `MAX_CHUNKS = 16` chunks are kept.
3. **Batching.** 2 cases per batch; chunks are flattened to `[B·C, L]` with token and chunk masks.

## Model

chunks → `backbone.encode()` → **token pooling** per chunk (`last_token` / `mean` / `attention`) → **document
pooling** over real chunks (`mean` / `max` / `attention`) → dropout → linear head → 100 logits.

- **Loss:** `BCEWithLogitsLoss`. With `use_pos_weight=True`, each statute's positive class is weighted by
  `(negatives / positives)^0.5` in the training set, clipped to [1, 20].
- **Decision rule:** a statute is predicted when `sigmoid(logit) ≥ threshold`. Before test inference the notebook tunes
  one global threshold on dev to maximise dev Macro-F1 (`tune_threshold`); per-class thresholds are also implemented.
- **Model selection:** the best epoch is chosen by dev Macro-F1.

## Adaptation techniques

| Key | Technique | Implementation |
|---|---|---|
| `full` | Full fine-tuning | all backbone + head parameters |
| `partial` | Partial fine-tuning | backbone modules listed in `cfg.partial_layers` + head |
| `lorma` | **LORMA** | `LowRankMulLinear`: `y = (I + (α/r)·B·A)·W₀x + b`, r = 16, α = 32. For the CNN, the targets are the attention projections and the feed-forward layers `ffn.0` / `ffn.2` |
| KD | **Knowledge distillation** | `DistillationExperiment` + `multilabel_kd_loss` (section 15) |

**Training.** AdamW, backbone/adapter learning rate 5e-4, head learning rate 3e-4, weight decay 0.01. Batch 2 ×
gradient accumulation 8, 8 epochs, mixed precision, gradient clipping 1.0.

**Knowledge distillation.**
- Teacher: decoder + full fine-tuning, last-token pooling (`KD_TEACHER_CHECKPOINT`).
- Student: the first 4 decoder blocks with LORMA, trained in fp32.
- Loss: `(1 − α)·BCE(gold) + α·T²·BCE(student/T, sigmoid(teacher/T))`, with T = 2.0 and α = 0.5.
- Optimiser: lr 2e-4, 8 epochs.
- `KD_STUDENT_INIT_SOURCE` / `KD_STUDENT_RESUME_CHECKPOINT` choose whether the student starts from the Phase-1 decoder
  or continues from an earlier student checkpoint.

## How to run

Set the paths in section 2, then *Run All*. The default flags reproduce the best submission: train `exp_cfg` = CNN +
LORMA with mean token pooling, tune the threshold on dev, predict the test set and write `submission.zip`. Set
`RUN_KD = True` (and `INFERENCE_SOURCE = "kd"`) for the distillation run.

## Submissions (Codabench, Macro-F1)

| Submission | Backbone | Adaptation | Token pooling | pos_weight | Macro-F1 | Mean statutes / test case |
|---|---|---|---|---|---|---|
| cnn lorma | cnn | LORMA (attention + FFN layers) | mean | no | **0.2846** (best) | 6.19 |
| dec lorma | decoder | LORMA | last_token | yes | 0.2708 | 6.20 |
| dec full | decoder | full fine-tuning | last_token | yes | 0.2670 | 5.74 |
| KD | decoder, 4-layer student | KD (teacher: decoder full FT) + LORMA | last_token | no | 0.2553 | 7.99 |
