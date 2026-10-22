# lazylab

An overnight lab that fine-tunes a small decision model for the
[`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
benchmark — while you sleep. A local agent proposes hyperparameter configs, a deterministic ASHA rule
kills losing runs mid-flight, and the morning gives you a trained model, a full experiment log, and a
spoken briefing. No hand-tuning.

Full design: `../docs/superpowers/specs/2026-10-22-typed-decisions-lab-design.md`.

Built for DEV Hacktoberfest 2026 Week 3 ("Slow and Steady"). Tags: `#devchallenge #hf26challenge`.

## Status

**Day 1 scaffold — data sealed, scorer validated.** No training yet.

## Layout

```
src/
  scorer.py          official metrics (accuracy/KL/Brier/ECE), copied from the
                     dataset's own bench/score_with_jobs.py — the ruler
  baselines.py       Prior + Uniform reference baselines (no model)
  data_io.py         JSONL load/write with stable serialisation for hashing
  prepare_data.py    download → seal test (SHA-256) → freeze fit/select/calib splits
  harness_check.py   reproduce the published reference rows — declared test read #1
tests/
  test_scorer.py     hand-computed unit tests for the scorer math (no dataset)
data/
  train.jsonl        1,200 cases (gitignored; regenerate with prepare_data.py)
  sealed/            test.jsonl (gitignored) + test.sha256 (tracked)
  splits/            fit/select/calib JSONL + manifest.json (tracked hashes)
```

## The one rule that matters

The test set is **sealed**. Only its SHA-256 (`data/sealed/test.sha256`) is committed; the raw cases are
gitignored and must never be read by the overnight worker. The test split is read exactly three times,
each declared in advance (design doc §8.9):

1. **Prior / Uniform harness** — no model, validates the scorer (this scaffold).
2. **Replicating Unsloth's 73%** — declared baseline.
3. **The final finalist** — config committed to git first, then one read.

## Reproduce the scaffold

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python datasets
.venv/bin/python src/prepare_data.py     # download, seal, split
.venv/bin/python src/harness_check.py     # validate the scorer
.venv/bin/python tests/test_scorer.py     # unit tests
```

## Harness result (scorer validation)

`accuracy`, `KL from gold`, and `Brier` reproduce the published Prior and Uniform rows to 3–4 decimals:

| baseline | metric | ours | card |
|---|---|---|---|
| Prior | accuracy | 0.4720 | 0.470 |
| Prior | KL from gold | 0.3474 | 0.347 |
| Prior | Brier | 0.1887 | 0.189 |
| Uniform | KL from gold | 0.4445 | 0.444 |
| Uniform | Brier | 0.2382 | 0.238 |

**Prior is keyed per-question, pooled across workflows** — it ignores the input, and the workflow *is*
part of the input. Keying per-workflow leaked information (accuracy 0.481 vs the card's 0.470); the
harness caught it.

ECE (and Uniform's accuracy) are reported but not gated: the card's model-free reference rows were
hand-scored before the current `bench` script and use a different ECE convention. We match the committed
`score_with_jobs.py` arithmetic, which is authoritative (design doc §18.2).

## Credits

Benchmark, metric definitions and scoring code: `LocalLLaMA/typed-decisions` (Apache 2.0). Full credits
in the design doc §14.
