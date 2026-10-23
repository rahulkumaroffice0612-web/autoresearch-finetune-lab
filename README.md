# lazylab

An overnight lab that fine-tunes a small **decision model** for the
[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
benchmark — while you sleep. A **local, open-weight agent (Gemma 4 E4B)** proposes hyperparameter
configs, a deterministic **ASHA** rule kills losing runs mid-flight, the winner is retrained once and
scored on a **sealed** test set, and a **Tinker** run confirms the recipe on a bigger model. No
hand-tuning.

Built for DEV Hacktoberfest 2026 Week 3 ("Slow and Steady"). Tags: `#devchallenge #hf26challenge`.

- **Model:** [`rahulkumar0612/lazydecider-0.8b`](https://huggingface.co/rahulkumar0612/lazydecider-0.8b)
  — `unsloth/Qwen3.5-0.8B` + a Clef joint decision head, LoRA fine-tuned.

## Results (sealed test, 2,000 decisions)

| Model | How | Accuracy | KL↓ | Brier↓ | ECE↓ |
|---|---|---|---|---|---|
| Uniform / Prior | no model | ~0.47 | 0.35–0.44 | 0.19–0.24 | — |
| Zero-shot base | no training | 0.483 | — | — | — |
| **LazyDecider 0.8B** (submitted) | overnight agent + ASHA search | **0.770** | **0.094** | **0.051** | **0.145** |
| Tinker 4B LoRA (confirm) | Qwen3.5-4B, 60 steps, dollar-capped | 0.601¹ | 0.64 | 0.17 | 0.094 |

¹ Confirm run is a deliberately budget-capped generic causal-LM reframe on a bigger model — it shows a
clear lift over the 0.483 zero-shot baseline, which is what the Tinker category asks for. The headline
0.770 is the Clef-head laptop model.

The headline **0.770** is produced by our own scorer on our sealed split (`artifacts/test_score.json`);
every number in the results table is one we measured ourselves — no borrowed or re-scaled figures.

## How it works

```
read experiments.jsonl  ->  Gemma proposes ONE config (+ one-line reason)
  ->  train (MLX/Unsloth LoRA) with ASHA pruner  ->  evaluate full `select`
  ->  keep if select-loss improves else revert  ->  append to the log  ->  repeat
```

- **Agent:** Gemma 4 E4B, local via LM Studio / Bionic (OpenAI-compatible, no cloud, no cost).
  Proposals land in `artifacts/experiments_arm*.jsonl` tagged `source=gemma`.
- **Runner:** Unsloth `FastDecisionModel` + `DecisionTrainer` (Clef head), MLX on Apple Silicon.
- **Pruner:** ASHA rung percentiles kill weak runs early.
- **Confirm:** `src/tinker_confirm.py` fine-tunes Qwen3.5-4B via the Tinker API with a hard dollar cap.

## The one rule that matters

The test set is **sealed**. Only its SHA-256 (`data/sealed/test.sha256`) is committed; the raw cases are
gitignored and the overnight worker can never read them. The scorer (`src/scorer.py`) copies the
dataset's own `bench/score_with_jobs.py` arithmetic verbatim — it is the ruler, not an approximation.

## Guardrails (the "can't cheat" part)

- Sealed test; scored once per finalist, config committed to git first.
- Official scorer, hash-checked; any tampering aborts.
- **Tinker spend cap in code**, and the overnight worker strips `TINKER_*` from its env at import so it
  is physically unable to spend credit.
- Time cap per run; weak runs pruned.

## Layout

```
src/
  agent.py            Gemma proposer (OpenAI-compatible) + deterministic random fallback
  worker.py           the autonomous overnight loop (propose -> train -> prune -> keep/revert -> log)
  trainer.py          one trial: Clef LoRA fine-tune, scored on `select`
  asha.py             ASHA pruner (rung percentile logic)
  train_config.py     the only knobs the agent may set, with a strict validating schema
  scorer.py           official metrics (accuracy/KL/Brier/ECE), copied from the dataset's bench
  baselines.py        Prior + Uniform reference baselines (no model)
  finalize.py         retrain the winning config and save a submittable model
  score_test.py       the one-shot sealed-test read (hash-verified)
  tinker_confirm.py   the Tinker confirm run (4B LoRA, dollar-capped)
  run_gemma_search.sh launcher that points the agent at local Gemma
tests/                scorer + config unit tests
data/
  sealed/             test.jsonl (gitignored) + test.sha256 (tracked)
  splits/             fit/select/calib JSONL + manifest.json (tracked hashes)
artifacts/
  experiments_arm*.jsonl   the experiment logs (Gemma's configs + reasons + metrics)
  results.json, tinker_confirm_result.json   the scored numbers
  best_model/README.md     the Hugging Face model card (weights published to HF, not git)
```

## Reproduce

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python datasets unsloth tinker
.venv/bin/python src/prepare_data.py          # download, seal, split
.venv/bin/python src/harness_check.py          # validate the scorer against published rows
# Gemma search (needs LM Studio/Bionic serving gemma-4-e4b locally):
bash src/run_gemma_search.sh 20
# Tinker confirm (needs TINKER_API_KEY; dollar-capped):
.venv/bin/python src/tinker_confirm.py --max-steps 60 --n-test 150
```

## Credits

Benchmark, metric definitions, and scoring code: [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
(Apache 2.0). Fine-tuning: [Unsloth](https://github.com/unslothai/unsloth). Local agent: Google
**Gemma 4 E4B**. Confirm stage: **Tinker** by Thinking Machines.
