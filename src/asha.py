"""ASHA mid-run pruning as a TrainerCallback (design doc §8.4, §8.6).

Verified against the MLX trainer (2026-10-23): it calls callback_handler.call_event
with full HF semantics, fires `on_evaluate(..., metrics=...)` at every eval_steps,
and honours `control.should_training_stop`. So a stock transformers.TrainerCallback
works unchanged.

The rule: at each rung (a step count), compare this run's eval loss against the
historical distribution of eval losses *at that same rung* from prior runs
(experiments.jsonl). If it is worse than the 100/eta percentile, abandon.

Cold start: until MIN_HISTORY runs have reached a rung, let runs finish so the
distribution can form.
"""
from __future__ import annotations

import json
from pathlib import Path

try:
    from transformers import TrainerCallback
except Exception:  # pragma: no cover - transformers always present with unsloth
    class TrainerCallback:  # minimal shim so import never fails
        pass

RUNGS = (20, 60, 140, 240)   # §8.4, final rung = 2 epochs over 960 fit cases
ETA = 3
MIN_HISTORY = 5              # runs needed at a rung before pruning activates
EVAL_LOSS_KEY = "eval_loss"
EARLY_EVAL_SUBSAMPLE = 20    # §8.4: early rungs eval on 20 select cases (~5s), not 120 (~36s)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("inf")
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def nearest_rung(step: int, tol: int = 2) -> int | None:
    """Return the rung this step lands on (within tol), else None."""
    for r in RUNGS:
        if abs(step - r) <= tol:
            return r
    return None


def losses_at_rung(log_path: Path, rung: int) -> list[float]:
    """Eval losses recorded at `rung` across all prior runs in experiments.jsonl."""
    if not log_path.exists():
        return []
    out = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for ev in rec.get("rung_evals", []):
                if ev.get("rung") == rung and ev.get("eval_loss") is not None:
                    out.append(float(ev["eval_loss"]))
    return out


class ASHAPruner(TrainerCallback):
    """Prunes a run mid-flight and records why. The worker reads `self.rung_evals`
    afterwards to append to experiments.jsonl."""

    def __init__(self, log_path: str | Path):
        self.log_path = Path(log_path)
        self.rung_evals: list[dict] = []   # this run's (rung, eval_loss) observations
        self.pruned_at: int | None = None
        self.prune_reason: str | None = None

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        metrics = metrics or {}
        loss = metrics.get(EVAL_LOSS_KEY)
        if loss is None:
            return control
        step = int(getattr(state, "global_step", 0))
        rung = nearest_rung(step)
        if rung is None:
            return control

        self.rung_evals.append({"rung": rung, "step": step, "eval_loss": float(loss)})

        hist = losses_at_rung(self.log_path, rung)
        if len(hist) < MIN_HISTORY:
            return control  # cold start: let it run, build the distribution

        cutoff = _percentile(hist, 100.0 / ETA)
        if loss > cutoff:
            control.should_training_stop = True
            self.pruned_at = rung
            self.prune_reason = (
                f"pruned at rung {rung} (step {step}): "
                f"eval_loss {loss:.4f} > p{100.0/ETA:.0f} cutoff {cutoff:.4f} "
                f"of {len(hist)} peers"
            )
        return control
