# Phase 4 — Court Judgment Prediction (CJPE)

**Task:** predict the binary outcome (`0` / `1`) of a case from the full judgment text.
**Primary metric:** Macro-F1 (secondary: Precision, Recall).
**Output:** `prediction.json` = `[{"id": case_id, "pred": 0 | 1}, ...]`, one object per test case, at the root of `submission.zip`.

Notebook: `phase4_cjpe.ipynb`. Backbones registered: `decoder`, `encdec`, `cnn`, `ssm`.

## Pipeline

1. **Data.** `Phase_4_CJPE/Train/train.json`, `Train/dev.json` and `Test/test.json` from `Exploration-Lab/CS-781-Capstone`.
   The notebook checks the labels ∈ {0, 1}, the IDs and the test order.
2. **Long-document representation.** Judgments are much longer than the backbone context, so each case is tokenized
   and split into overlapping chunks of `CHUNK_TOKENS = 512` tokens with stride 384. When a case has more than
   `MAX_CHUNKS = 16` chunks, the first 8 and the last 8 are kept (head + tail).
3. **Batching.** `BATCH_SIZE = 2` cases; chunks are flattened to `[B·C, L]` for the backbone, with a chunk mask.

## Model

chunks `[B·C, L]` → `backbone.encode()` → masked mean over tokens (one vector per chunk) → masked mean over the real
chunks (one vector per case) → dropout → linear head → logits `[B, 2]`. Loss: cross-entropy (class weights optional,
off by default). Training stops with an error if the loss becomes NaN, and inference warns when the test predictions
are (almost) all one class. The best epoch is chosen by dev Macro-F1.

## Adaptation techniques

| Key | Technique | Implementation |
|---|---|---|
| `full` | Full fine-tuning | all backbone + head parameters |
| `rslora` | Rank-stabilised LoRA | `peft.LoraConfig(r=16, lora_alpha=32, use_rslora=True)` |
| `partial` | Partial fine-tuning | last 2 decoder blocks + final LayerNorm + head |
| `lorma` | **LORMA** | `LowRankMulLinear`: `y = (I + (α/r)·B·A)·W₀x + b`, r = 16, α = 32 |
| KD | **Knowledge distillation** | `DistillationExperiment` + `binary_kd_loss` (section 15) |

**Training.** AdamW, backbone/adapter learning rate 5e-4, head learning rate 3e-4, weight decay 0.01. Batch 2 ×
gradient accumulation 8, 4 epochs, mixed precision, gradient clipping 1.0.

**Knowledge distillation (best submission).**
- Teacher: decoder + full fine-tuning (`phase4_decoder_full`, loaded strictly from `KD_TEACHER_CHECKPOINT`).
- Student: embeddings + the first 4 blocks of the Phase-1 decoder, with LORMA (r = 16, α = 32) and the same head.
- Loss: `(1 − α)·CE(gold) + α·T²·KL(softmax(teacher/T) ‖ softmax(student/T))`, with T = 2.0 and α = 0.5.
- Optimiser: lr 5e-4, 6 epochs.
- An efficiency report gives the parameter counts of teacher and student.

## How to run

Set the paths in section 2 (tokenizer, Phase-1 checkpoints, `KD_TEACHER_CHECKPOINT`), then *Run All*. The default flags
reproduce the best submission: `RUN_TRAINING = False`, `RUN_KD = True`, `INFERENCE_SOURCE = "kd"`. To train a
configuration instead, set `RUN_TRAINING = True` and choose `exp_cfg` in section 11 (`INFERENCE_SOURCE = "main"`).

## Submissions (Codabench, Macro-F1)

| Submission | Backbone | Adaptation | Macro-F1 | Test predictions (0 / 1) |
|---|---|---|---|---|
| KD_dec_full | decoder, 4-layer student | KD (teacher: decoder full FT) + LORMA | **0.5944** (best) | 566 / 951 |
| dec_full | decoder | full fine-tuning | 0.5655 | 786 / 731 |
| dec_lorma | decoder | LORMA | 0.3320 | 1516 / 1 |
| cnn_lorma | cnn | LORMA | 0.3320 | 1516 / 1 |

`dec_lorma` and `cnn_lorma` predicted class 0 for 1516 of the 1517 test cases. 0.3320 is the score of a constant
prediction.
