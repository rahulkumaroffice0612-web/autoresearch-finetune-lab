"""The overnight worker: fully autonomous search loop (design doc §8.6).

  read experiments.jsonl  ->  proposer picks ONE config  ->  train with ASHAPruner
  ->  evaluate full select  ->  keep if loss improves else revert  ->  append log
  ->  repeat until the time budget expires.

No human judgment anywhere here. The only approval gate in the project is Tinker,
which this worker cannot reach: TINKER_* is stripped from the environment at import
(§8.9). Selection is deterministic: lowest select `loss` wins, committed by the
morning script, not here.

Usage:
  python src/worker.py --minutes 480 --arm A
  python src/worker.py --minutes 30 --arm A --random   # no LM Studio needed
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# --- GUARDRAIL (§8.9): the worker must be unable to spend Tinker credit. Strip first. ---
_stripped = [k for k in list(os.environ) if "TINKER" in k.upper()]
for k in _stripped:
    del os.environ[k]

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
ARTIFACTS = ROOT / "artifacts"


def load_split(name: str):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from data_io import load_cases
    return load_cases(SPLITS / f"{name}.jsonl")


def read_history(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    out = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def best_loss(history: list[dict]) -> float:
    kept = [h["select"]["loss"] for h in history
            if not h.get("pruned") and "select" in h and "loss" in h["select"]]
    return min(kept) if kept else float("inf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--arm", choices=["A", "B"], default="A")
    ap.add_argument("--random", action="store_true", help="use RandomProposer (no LM Studio)")
    ap.add_argument("--smoke", action="store_true",
                    help="fast loop validation: force head-only epochs=1, 120-case fit slice")
    ap.add_argument("--run-time-cap-min", type=float, default=20,
                    help="hard wall-clock kill per run (§8.6)")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from agent import GemmaProposer, RandomProposer
    from asha import ASHAPruner
    from train_config import TrainConfig
    import trainer as T

    log_path = ARTIFACTS / f"experiments_arm{args.arm}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Arm A: fit only. Arm B would concat generated cases here (§19). Kept explicit.
    fit_rows = load_split("fit")
    select_rows = load_split("select")
    if args.smoke:
        fit_rows = fit_rows[:120]  # ~15 steps/epoch -> fast loop validation
    if _stripped:
        print(f"[guard] stripped {_stripped} from env", flush=True)
    print(f"[worker] arm={args.arm} fit={len(fit_rows)} select={len(select_rows)} "
          f"budget={args.minutes}min proposer={'random' if args.random else 'gemma'}", flush=True)

    proposer = RandomProposer(seed=int(time.time())) if args.random else GemmaProposer()

    class SmokeProposer:
        """Forces the fast path so loop mechanics can be validated in minutes."""
        def __init__(self, inner): self.inner = inner
        def propose(self, history):
            cfg, src = self.inner.propose(history)
            cfg.freeze_backbone = True
            cfg.epochs = 1
            return cfg, src
    if args.smoke:
        proposer = SmokeProposer(proposer)

    deadline = time.time() + args.minutes * 60
    n = len(read_history(log_path))
    while time.time() < deadline:
        history = read_history(log_path)
        config, source = proposer.propose(history)
        n += 1
        prev_best = best_loss(history)
        print(f"\n[run {n}] source={source} reason={config.reason!r}", flush=True)

        pruner = ASHAPruner(log_path)
        t0 = time.time()
        try:
            result = T.train_and_score(
                config, fit_rows, select_rows, pruner,
                ARTIFACTS / f"_run_arm{args.arm}", eval_steps=20,
            )
        except Exception as e:
            result = {"config": config.key(), "reason": config.reason,
                      "error": f"{type(e).__name__}: {str(e)[:200]}",
                      "rung_evals": pruner.rung_evals, "pruned": False}
            print(f"[run {n}] ERROR {result['error']}", flush=True)

        loss = result.get("select", {}).get("loss")
        kept = (loss is not None) and (not result.get("pruned")) and (loss < prev_best)
        result.update({
            "run": n, "source": source, "arm": args.arm,
            "timestamp": time.time(), "wall_seconds": round(time.time() - t0, 1),
            "prev_best_loss": prev_best if prev_best != float("inf") else None,
            "kept": kept,
        })
        with open(log_path, "a") as f:
            f.write(json.dumps(result) + "\n")

        status = "KEPT (new best)" if kept else ("pruned" if result.get("pruned") else "reverted")
        print(f"[run {n}] loss={loss} {status}  wall={result['wall_seconds']}s", flush=True)

    history = read_history(log_path)
    print(f"\n[worker] done. {len(history)} runs, best select loss = {best_loss(history):.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
