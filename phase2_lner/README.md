# Phase 2 — Legal Named Entity Recognition (L-NER)

**Task:** predict exact character spans of 12 entity types in Indian judgments: `APP`, `RESP`, `A.COUNSEL`,
`R.COUNSEL`, `JUDGE`, `WIT`, `AUTH`, `COURT`, `STAT`, `PREC`, `DATE`, `CASENO`.
**Primary metric:** strict Macro-F1 (exact start, end and label; secondary: strict Macro-Precision / Recall).
**Output:** `prediction.json` = `{doc_id: [[start_char, end_char, label], ...]}`, at the root of `submission.zip`.

Notebook: `phase2_lner.ipynb`. Backbones: `decoder` and `ssm` (Phase 1).

## Pipeline

`document → tokenizer with exact character offsets → overlapping windows → Phase-1 backbone encode() → BIO + boundary heads → character spans → prediction.json`

1. **Data.** Both training folds and the label list are downloaded from `Exploration-Lab/CS-781-Capstone`. Gold
   annotations are normalised to `(start_char, end_char, label)`; the end-offset convention is checked on the data.
   Document text is never normalised, so offsets stay exact.
2. **Offsets.** `encode_with_offsets(tokenizer, text)` extends the Phase-1 `ByteLevelBPE` tokenizer to return, for every
   token, its `(start_char, end_char)` in the original string. A cell of test strings checks this mapping.
3. **Targets.** BIO token classification (25 classes) plus two boundary-correction heads. The start head predicts a
   character shift of {0, 1, 3}; the end head predicts a shift of {−1, 0, 1, 2, 3, 4}. These fix span edges that fall
   inside a BPE token.
4. **Windowing.** Each document is split into windows of 2,048 tokens with stride 1,536. Entities cut at a window edge
   are handled at the window boundary, and every window keeps its token offsets.
5. **Split.** A document-level 85 / 15 train / validation split (seed 42).
6. **Class balancing.** Inverse-frequency class weights (power 0.5, capped at 8) for the BIO, start and end heads.

## Model

`Phase2NERModel`: backbone `encode()` → dropout 0.1 → three linear heads (BIO, start shift, end shift).
Loss = weighted cross-entropy (BIO) + 0.3 × (start + end boundary cross-entropy).

**Decoding.** Token predictions are turned into character spans using the stored offsets. Spans from overlapping
windows are merged per document, near-duplicates (IoU ≥ 0.8) are removed, and a global confidence threshold
(`CONFIDENCE_THRESHOLD = 0.95`) is applied. Submission checks: every test ID is present, there are no unknown IDs,
labels are official, and all spans lie inside the document.

## Adaptation techniques

| Key | Technique | Details |
|---|---|---|
| `full` | Full fine-tuning | all backbone + head parameters |
| `partial` | Partial fine-tuning | last 2 decoder blocks + final LayerNorm + the three heads |
| `lorma` | LORMA (low-rank update) | `LowRankLinear` on the decoder blocks' linear layers, rank 8, α = 16; additive form `W₀x + (α/r)·B·A·x`, `B = 0` at init |
| KD | Knowledge distillation | section 14 (below) |

## Training settings

AdamW: backbone learning rate 7e-5, head learning rate 3e-4, weight decay 0.01. Batch 2 × gradient accumulation 2,
8 epochs, mixed precision, gradient clipping 1.0. The best epoch is chosen by validation strict Macro-F1. Each run writes
to `/kaggle/working/phase2_<backbone>_<method>/` (`best_model.pt`, `metrics.json`, `validation_predictions.json`).

**Knowledge distillation (section 14).**
- Teacher: the full-fine-tuned decoder (`phase2_decoder_full/best_model.pt`).
- Student: a 6-layer decoder whose blocks are initialised from teacher blocks `[0, 1, 3, 4, 6, 7]`.
- Loss: masked token-level KL divergence on the BIO and boundary logits (T = 2.0, α = 0.15), combined with the gold loss.
- Optimiser: lr 3e-4, 8 epochs. The student's predictions use a confidence threshold of 0.90.

## How to run

1. Set the tokenizer path (section 3), `DECODER_CHECKPOINT` and `SSM_CHECKPOINT` (section 6).
2. Choose the experiment in section 10 (`exp_cfg`: `backbone_name` ∈ {`decoder`, `ssm`}, `adaptation_method` ∈
   {`full`, `partial`, `lorma`}).
3. *Run All*. The notebook trains, evaluates the confidence threshold on validation, runs KD (it needs the `decoder` / `full` run
   first), predicts the test set and writes `/kaggle/working/prediction.json` and `submission.zip`.

The active configuration is the best submission: `decoder`, `full`, global threshold 0.95.

## Submissions (Codabench, strict Macro-F1)

| Submission | Configuration | Strict Macro-F1 |
|---|---|---|
| submission_2 | decoder, full fine-tuning, global threshold 0.95 | **0.4362** (best) |
| submission_4 | experiment run | 0.4301 |
| dec_full_opt_change | decoder, full fine-tuning, different optimiser settings | 0.4208 |
| submission_7 | experiment run | 0.4181 |
| dec_partial | decoder, partial fine-tuning | 0.4174 |
| submission_5 | experiment run | 0.4124 |
| dec_lorma | decoder, LORMA | 0.4088 |
| dec_full_multi_thres | decoder, full fine-tuning, per-label thresholds | 0.4018 |
| ssm_full | SSM, full fine-tuning | 0.3476 |
| submission_3 | KD student (6-layer decoder) | 0.1037 |
