# Phase 7 — Prior Case Retrieval (PCR)

**Task:** for each query case, rank the candidate cases in **that split's own pool** and return the 7 most relevant
unique candidate IDs.
**Primary metric:** Micro-F1@7 (secondary: Micro-Precision@7, Micro-Recall@7).
**Output:** `prediction.json` = `{query_id: [7 candidate IDs, most → least relevant]}`, at the root of `submission.zip`.

Notebook: `phase7_pcr.ipynb`. Backbones registered: `decoder`, `encdec`, `cnn`, `ssm`.

## Data

Each split comes from `Exploration-Lab/CS-781-Capstone` as a query file and its own candidate pool:
- `train.json` + `train_candidates.json`: 827 queries / 4,320 candidates
- `dev.json` + `dev_candidates.json`: 118 / 1,023
- `test.json` + `test_candidates.json`: 237 / 1,727

Documents are lists of paragraphs, joined into one text. Gold labels are `relevant_candidates`; test queries have none.
Query IDs also appear in their own pool, so a query's own ID is never returned. The notebook checks for duplicate IDs,
for gold candidates missing from the pool, and that no ID is lost or reordered.

## Model — siamese bi-encoder

A single shared encoder turns every document, query or candidate, into one vector:

1. **Chunks.** The document is tokenized into chunks of `CHUNK_TOKENS = 256` (stride 192). At most 2 chunks are kept,
   evenly spaced across the document (`MAX_QUERY_CHUNKS = MAX_CAND_CHUNKS = 2`). `TextChunkCache` tokenizes each
   document only once.
2. **Encoding.** `backbone.encode()` on every chunk → token pooling (`last_token` / `mean` / `attention`) → document
   pooling over the real chunks (`mean` / `max` / `attention`) → linear projection to 256 dimensions → L2 normalisation.
3. **Score.** Cosine similarity / τ (τ = 0.05).
4. **Training data.** `PCRDataset` pairs each query with its gold positives and `NUM_HARD_NEGATIVES = 3` negatives
   sampled from the same pool (never the query itself). `pcr_collator` de-duplicates candidates within a batch and
   builds a `[B, M]` relevance matrix.
5. **Loss.** Multi-positive InfoNCE over in-batch + sampled candidates, **+ 0.3 ×** a pairwise ranking loss. Every
   positive should beat the 3 hardest negatives by a cosine margin of 0.1 (softplus).
6. **Inference** (`run_inference`). The whole pool is embedded once, every query is scored against all candidates with
   a matrix product, the query's own ID is excluded, and the top 7 are kept. `validate_submission_format` checks 7
   unique in-pool IDs per query and that all test IDs are present.
7. **Model selection.** After every epoch, all dev queries are ranked against the full dev pool, and the epoch with the
   best dev Micro-F1@7 is kept.

## Adaptation techniques

| Key | Technique | Implementation |
|---|---|---|
| `full` | Full fine-tuning | every backbone + head parameter |
| `partial` | Partial fine-tuning | last 25 % (≥ 1) of the layers in every `nn.ModuleList` of the backbone + head |
| `ia3` | IA³ | `peft.IA3Config` on the registered attention / feed-forward layers |
| `lorma` | **LORMA** | `LowRankMulLinear`: `y = (I + (α/r)·B·A)·W₀x + b`, r = 16, α = 32. CNN targets include `ffn.0` / `ffn.2` |
| KD | **Knowledge distillation** | `DistillationExperiment` + listwise `listwise_kd_loss` (section 15) |

The retrieval head (token pooling, document pooling, projection) is always trainable.

**Training.** AdamW. The backbone/adapter learning rate is set per configuration: 2e-5 for the best run's full
fine-tuning, and between 5e-4 and 3e-3 in the commented adapter configurations. Head learning rate 3e-4, weight decay 0.01. Batch 1 query × gradient accumulation 4, 6 epochs, mixed precision, gradient clipping 1.0.

**Knowledge distillation (best submission).**
- Teacher: decoder + full fine-tuning, lr 2e-5, attention pooling (`exp_cfg`, trained in section 14, or
  `KD_TEACHER_CHECKPOINT`).
- Student: embeddings + the first 4 decoder blocks, with LORMA and the teacher's pooling.
- Loss: `(1 − α)·contrastive(gold) + α·T²·KL(teacher ‖ student)` over each query's candidate score distribution, with
  T = 2.0 and α = 0.5.
- Optimiser: lr 5e-4, 6 epochs. Student: 32.15M parameters vs 51.10M for the teacher.

## How to run

Set the paths in section 2, then *Run All*. The default flags reproduce the best submission:
1. `RUN_TRAINING = True` trains the teacher (decoder, full fine-tuning, lr 2e-5).
2. `RUN_KD = True` distils it into the 4-layer student.
3. `INFERENCE_SOURCE = "kd"` ranks the test pool with the student and writes
   `decoder_4layer_lorma_kd/submission.zip`.

Use `INFERENCE_SOURCE = "main"` to submit the teacher; it writes `phase7_decoder_full/submission.zip`.

## Submissions (Codabench, Micro-F1@7)

| Submission | Backbone | Adaptation | Micro-F1@7 |
|---|---|---|---|
| kddecfull | decoder, 4-layer student | KD (teacher: decoder full FT, lr 2e-5) + LORMA | **0.1636** (best) |
| decfulllr2e-5 | decoder | different learning rate | 0.1636 |
| submission_5 | — | — | 0.1636 |
| cnnlormaattn | cnn | LORMA, attention pooling | 0.1258 |
| cnnlrchange | cnn | LORMA, different learning rate | 0.1181 |
| cnnlormamean | cnn | LORMA, mean pooling | 0.0963 |
