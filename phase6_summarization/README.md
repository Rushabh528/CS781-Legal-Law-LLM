# Phase 6 — Abstractive Summarization

**Task:** generate one abstractive summary for each long legal document.
**Primary metric:** BERTScore F1 (secondary: ROUGE-1 F1, ROUGE-L F1).
**Output:** `prediction.json` = `[{"id": doc_id, "summary": "..."}, ...]`, one summary per test document, at the root of
`submission.zip`.

Notebook: `phase6_summarization.ipynb`. Backbones registered: `decoder`, `encdec`, `cnn`, `ssm`. BERTScore and ROUGE
are used only to evaluate generated text; the summarizer itself is built only from the Phase-1 models and tokenizer.

## Data

`Phase_6_Summ/Train/train.json` and `Phase_6_Summ/Test/test.json` from `Exploration-Lab/CS-781-Capstone`. There is no
public dev split, so a deterministic 90 / 10 train / validation split (seed 42) is made from `train.json`. The notebook
checks test IDs and order, that test documents carry no summaries, and that the two splits are disjoint.

## Formulations

- **Causal** (`decoder`, `cnn`, `ssm`): one sequence
  `<s> + document (first MAX_SOURCE_TOKENS = 1000 tokens) + " Summary:" + summary + </s>`.
  The loss is computed only on summary tokens (prompt and padding labels are −100). At test time the model continues
  the prompt with the Phase-1 LM head.
- **Encoder-decoder** (`encdec`): the document goes to the encoder and the summary to the decoder (teacher forcing).
  `STANDARD_ENCDEC = True` uses the plain token-level encoder (≤ 1,000 tokens), and both submitted encoder-decoder runs
  use it. `False` selects a hierarchical variant: every 1,000-token chunk (overlap 128) is encoded separately, and 64
  memory states per chunk plus a learned chunk-position embedding are passed to the decoder.
- `check_length_budget` refuses a configuration whose formulation switches do not match the chosen backbone.

**Decoding.** A generic `beam_search` with `causal_generate` / `seq2seq_generate`. The submitted runs use greedy
decoding (`num_beams = 1`) with at most 384 new tokens. `enforce_generation_limit` trims every summary to the length
limit before the submission is validated.

## Adaptation techniques

| Key | Technique | Implementation |
|---|---|---|
| `lora` | LoRA (additive low-rank update) | `peft.LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05)` |
| `lorma` | **LORMA** | `LowRankMulLinear`: `y = (I + (α/r)·B·A)·W₀x + b`, r = 16, α = 32 |
| KD | **Knowledge distillation** | `DistillationExperiment` + token-level `sequence_kd_loss` (section 15) |

**Training.** AdamW, adapter learning rate 5e-4, head learning rate 3e-4, weight decay 0.01. Batch 1 × gradient
accumulation 16, 4 epochs, mixed precision, gradient clipping 1.0. After every epoch, 50 validation documents are
summarised and scored with BERTScore F1, and the best epoch is kept. With `val_generate=False`, selection uses the
validation loss instead.

**Knowledge distillation.**
- Teacher: decoder + LORMA (`dec_lorma`, loaded strictly from `KD_TEACHER_CHECKPOINT`).
- Student: embeddings + the first 4 decoder blocks + final norm + LM head, with LORMA.
- Loss: `(1 − α)·CE(gold) + α·T²·KL(teacher ‖ student)` over the summary tokens, with T = 2.0 and α = 0.5.
- Optimiser: lr 2e-4, 4 epochs.

## How to run

Set the paths, the formulation switches (`ENCODER_DECODER`, `STANDARD_ENCDEC`) and the run flags in section 2, then
*Run All*. The default flags reproduce the best submission: train decoder + LORMA (`run_name="dec_lorma"`), greedily
generate the test summaries and write `submission.zip`. Set `RUN_KD = True` and `INFERENCE_SOURCE = "kd"` for the
distilled student. For the encoder-decoder runs, set `ENCODER_DECODER = True` and `backbone_name="encdec"`.

## Submissions (Codabench, BERTScore F1)

| Submission | Backbone | Adaptation | Formulation | BERTScore F1 | Test summaries |
|---|---|---|---|---|---|
| dec_lorma | decoder | LORMA | causal prompt + summary | **0.7876** (best) | mean 338 words |
| KD | decoder, 4-layer student | KD (teacher: decoder + LORMA) + LORMA | causal | 0.7542 | mean 368 words |
| encderlora | encdec | LoRA | standard encoder-decoder | 0.7309 | every summary 384 words |
| encdeclorma | encdec | LORMA | standard encoder-decoder | 0.7152 | every summary 384 words |
