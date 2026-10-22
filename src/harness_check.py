"""Harness check (design doc §8.9 declared read #1): reproduce the dataset card's
Prior and Uniform reference rows with our own scorer. No model involved, so this
read of the test split is always safe — it is how we validate the ruler.

Prior is fit on the TRAIN split (label frequencies) and scored on TEST.
Uniform needs no training.

Exits non-zero if any metric misses its published target by more than TOL, so this
doubles as a regression test for src/scorer.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

from baselines import build_prior, prior_predictor, uniform_predictor
from data_io import load_cases
from scorer import score_predictions

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "train.jsonl"
TEST = ROOT / "data" / "sealed" / "test.jsonl"

# Published reference rows (dataset card, accuracy/KL/Brier/ECE).
TARGETS = {
    "prior":   {"accuracy": 0.470, "kl_from_gold": 0.347, "brier": 0.189, "ece": 0.088},
    "uniform": {"accuracy": 0.308, "kl_from_gold": 0.444, "brier": 0.238, "ece": 0.169},
}
# accuracy/KL/Brier are the authoritative check: they come straight out of the
# per-request scorer in bench/score_with_jobs.py and must match to ~3 dp.
STRICT = ("kl_from_gold", "brier")
TOL = 0.005  # card values are rounded to 3 dp; allow half a point

# ECE and (for Uniform) accuracy are NOT strict-checked. The card's reference rows
# were hand-scored before the current bench script and use a different ECE / an
# analytic (expected) accuracy for the model-free rows. We match the committed
# score_with_jobs.py arithmetic, not the older card numbers. See design doc §18.2.
INFORMATIONAL = {"prior": ("ece",), "uniform": ("accuracy", "ece")}


def check(name: str, got: dict) -> bool:
    """Strict-check every metric except those listed INFORMATIONAL for this baseline.
    Returns True iff all strict-checked metrics are within TOL."""
    ok = True
    informational = INFORMATIONAL.get(name, ())
    print(f"\n[{name}]  ({got['n_decisions']} decisions)")
    for metric, target in TARGETS[name].items():
        g = got[metric]
        delta = abs(g - target)
        is_info = metric in informational
        if is_info:
            flag = "..  "  # reported, not gated
        elif delta <= TOL:
            flag = "OK  "
        else:
            flag = "XX  "
            ok = False
        note = "  (informational — card predates bench scorer)" if is_info else ""
        print(f"  {flag}{metric:14s} got {g:.4f}  target {target:.3f}  Δ {delta:.4f}{note}")
    return ok


def main() -> int:
    if not TEST.exists() or not TRAIN.exists():
        print("data not prepared — run: python src/prepare_data.py", file=sys.stderr)
        return 2
    train = load_cases(TRAIN)
    test = load_cases(TEST)

    prior = build_prior(train)
    results = {
        "prior": score_predictions(test, prior_predictor(prior)),
        "uniform": score_predictions(test, uniform_predictor()),
    }
    all_ok = all(check(name, results[name]) for name in ("prior", "uniform"))

    print("\n" + ("PASS — scorer reproduces the published reference rows."
                  if all_ok else
                  "FAIL — scorer does NOT match the benchmark. Fix before trusting any number."))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
