---
license: apache-2.0
base_model: unsloth/Qwen3.5-0.8B
library_name: unsloth
datasets:
  - LocalLLaMA/typed-decisions
language:
  - en
tags:
  - typed-decisions
  - structured-decisions
  - calibration
  - specialist
  - clef
  - unsloth
  - decision-model
pipeline_tag: text-classification
thumbnail: https://huggingface.co/rahulkumar0612/lazydecider-0.8b/resolve/main/lazydecider.png
model-index:
  - name: lazydecider-0.8b
    results:
      - task:
          type: text-classification
          name: Typed Decisions
        dataset:
          name: Typed Decisions
          type: LocalLLaMA/typed-decisions
        metrics:
          - type: accuracy
            value: 0.77
            name: Accuracy
          - type: kl_divergence
            value: 0.0938
            name: KL from gold
          - type: brier_score
            value: 0.051
            name: Brier
          - type: ece
            value: 0.1454
            name: ECE
---

# lazydecider-0.8b

![lazydecider-0.8b — a tiny, lazy decision model that picks the right option and tells you how sure it is](https://huggingface.co/rahulkumar0612/lazydecider-0.8b/resolve/main/lazydecider.png)

A specialist for the [Typed Decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
benchmark: **[`unsloth/Qwen3.5-0.8B`](https://huggingface.co/unsloth/Qwen3.5-0.8B)** fine-tuned with a
**Clef joint decision head** and a LoRA adapter. Given the *state* of a workflow (an agent trace, a support
ticket, an invoice, a security alert) and a schema of typed questions, it returns a calibrated probability
distribution over each question's options in a single forward pass. It never generates free text, so there
is nothing to parse and nothing to hallucinate.

**This is not zero-shot.** It was fitted on the `train` split of Typed Decisions and is scored on the
400-case `test` split. Per the benchmark's own rules it is a *specialist*, and should not be compared with
generalist models that answer unseen question schemas.

## Results (`test` split, 400 cases, 2,000 decisions)

Scored with the official arithmetic from the dataset's `bench/score_with_jobs.py` — argmax accuracy,
`KL(gold ‖ pred)`, summed Brier, and 10-bin population-weighted ECE — over all 2,000 decisions.

| Metric | This model | Unsloth documented¹ | Zero-shot Qwen3.5-0.8B² |
|---|---|---|---|
| Accuracy | **0.770** | 0.73 | 0.483 |
| KL from gold (lower is better) | **0.0938** | — | — |
| Brier (lower is better) | **0.0510** | — | — |
| ECE (lower is better) | 0.1454 | — | — |

¹ Unsloth documents `Qwen3.5-0.8B` going **36% → 73%** on this benchmark with their recommended settings.
² Published zero-shot reference row for the same base model.

Validation (model-selection) metrics, on a held-out slice of `train`: accuracy `0.782`, ECE `0.155`,
loss `0.837`. Hyperparameters were tuned on that held-out slice, **never on `test`**; the test split was
hashed and sealed before training and scored exactly once.

Sealed `test` SHA-256: `66342f3036e80e17569aa177da94a3c04181752af5c89402377e903f1040b8ac`.
Full per-decision predictions are in `results.json` / `test_predictions.jsonl`.

## Usage

```python
from unsloth import FastDecisionModel

model, tokenizer = FastDecisionModel.from_pretrained("rahulkumar0612/lazydecider-0.8b")
pred = model.predict(case)   # -> {"probabilities": {option: prob, ...}} per decision
```

The head emits `true`/`false` directly for yes/no questions (the official scorer also maps engine
`yes`/`no` → `true`/`false`). Weights are merged and saved standalone, so no separate adapter load is needed.

## Recipe

- **Backbone:** `unsloth/Qwen3.5-0.8B` (`model_type: qwen3_5`), LoRA fine-tuned (backbone unfrozen).
- **Head:** Clef joint decision head — hidden 1024, width 512, 4 layers, 8 heads, 2048 FFN, 2 routing layers.
- **Adapter:** LoRA `r=16`, `alpha=32`, `dropout=0.05` (~10.2M trainable params, 1.18% of 863M).
- **Two learning rates:** head `1e-3`, backbone `2e-4`; cosine schedule, 10% warmup, weight decay `0.01`.
- **2 epochs**, batch 4 × grad-accum 2, `max_len` 1280, label smoothing 0 (see note), on Apple Silicon (MLX).
- Hyperparameters were chosen by an **autonomous search** (an LLM proposed configs, each trial trained on a
  `fit` slice and scored on a `select` slice, weak trials pruned ASHA-style); the winner was retrained once.

```json
{
  "freeze_backbone": false,
  "head_learning_rate": 0.001,
  "learning_rate": 0.0002,
  "label_smoothing": 0.0,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "batch_size": 4,
  "grad_accum": 2,
  "warmup_ratio": 0.1,
  "weight_decay": 0.01,
  "lr_scheduler": "cosine",
  "epochs": 2
}
```

Two findings from the search: unfreezing the backbone (LoRA across the encoder, not just the head) lifted
accuracy from ≈0.52 to ≈0.78 at the cost of calibration (ECE ≈0.05 → ≈0.15); and on these soft teacher-gold
targets, label smoothing ≥0.06 made both loss and ECE *worse*, so it was left at 0.

## Limits

- **Specialist.** It answers the question schemas of the four workflows it was trained on
  (`agent_trace_observability`, `customer_service`, `invoice_processing`, `security_incidents`). Do not
  compare it against zero-shot rows.
- **It learns the teacher's labels.** The gold is the mean of three samples from a ~4B-class teacher, so a
  score measures agreement with that teacher, not correctness (see the benchmark README).
- **Calibration (ECE ≈0.15) is the weak point.** Unfreezing the backbone bought accuracy at the cost of
  calibration; argmax accuracy is strong for the size class.
- **Synthetic data only;** no claim is made about real-world tickets, invoices or incidents.

## Links

- Dataset: https://huggingface.co/datasets/LocalLLaMA/typed-decisions
- Base model: https://huggingface.co/unsloth/Qwen3.5-0.8B

Apache 2.0, inherited from the `unsloth/Qwen3.5-0.8B` base model.
