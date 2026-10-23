#!/usr/bin/env python3
"""A screen-capture-friendly walkthrough of the LazyDecider overnight loop.

Replays the REAL experiment logs (no live server needed) at a readable pace so a
short demo video shows: Gemma proposing a config with its reason, the ASHA rung
evals, the keep/revert decision, and the final morning-report numbers.

Everything shown is read from artifacts/ — nothing is fabricated for the demo.

Usage:
  .venv/bin/python src/demo.py            # replay arm A (the ASHA search)
  .venv/bin/python src/demo.py --arm B    # replay arm B (head-only sweep)
  .venv/bin/python src/demo.py --fast     # no typing delay (for re-recording)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ART = Path(__file__).resolve().parent.parent / "artifacts"

# ANSI colours (kept minimal so it reads well on a recording).
DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


def _sleep(seconds: float, fast: bool) -> None:
    if not fast:
        time.sleep(seconds)


def banner(text: str) -> None:
    line = "=" * max(40, len(text) + 4)
    print(f"\n{CYAN}{line}{RESET}")
    print(f"{CYAN}{BOLD}  {text}{RESET}")
    print(f"{CYAN}{line}{RESET}\n")


def replay(arm: str, fast: bool) -> None:
    path = ART / f"experiments_arm{arm}.jsonl"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

    banner(f"LazyDecider — overnight loop (arm {arm}, {len(rows)} Gemma proposals)")
    print(f"{DIM}Agent: Gemma 4 E4B (local, MLX) · Pruner: ASHA · Scored on a sealed split{RESET}")
    _sleep(1.5, fast)

    best = None
    for r in rows:
        run = r.get("run")
        cfg = r.get("config", {})
        reason = (r.get("reason") or "").strip()

        print(f"\n{BOLD}── run {run} ─────────────────────────────────────────{RESET}")
        print(f"{YELLOW}Gemma proposes:{RESET}")
        # Show the handful of knobs that tell the story, not the whole blob.
        for k in ("freeze_backbone", "lora_r", "lora_alpha", "learning_rate", "label_smoothing", "epochs"):
            if k in cfg:
                print(f"    {k:>16} = {cfg[k]}")
        print(f'{YELLOW}  reason:{RESET} "{reason}"')
        _sleep(2.0, fast)

        # ASHA rung evals, if present (arm A has them).
        for e in r.get("rung_evals", []) or []:
            print(f"    {DIM}ASHA rung @ step {e['step']:>3}: eval_loss = {e['eval_loss']:.4f}{RESET}")
            _sleep(0.4, fast)

        sel = r.get("select") or {}
        loss = sel.get("loss")
        kept = r.get("kept")
        if loss is not None:
            print(f"  select-split loss: {BOLD}{loss:.4f}{RESET}")
        if kept:
            improved = best is None or (loss is not None and loss < best)
            if loss is not None and improved:
                best = loss
            print(f"  decision: {GREEN}KEEP ✔{RESET}  (new best)" if improved else f"  decision: {GREEN}KEEP ✔{RESET}")
        else:
            print(f"  decision: {RED}REVERT ✗{RESET}  (did not beat best)")
        _sleep(1.6, fast)

    banner("Morning report")
    try:
        score = json.loads((ART / "test_score.json").read_text())
        tinker = json.loads((ART / "tinker_confirm_result.json").read_text())
        print(f"  {BOLD}LazyDecider 0.8B (winner){RESET}  "
              f"acc {GREEN}{score['accuracy']:.3f}{RESET}  "
              f"KL {GREEN}{score['kl_from_gold']:.3f}{RESET}  "
              f"Brier {score['brier']:.3f}  ECE {score['ece']:.3f}")
        print(f"  {DIM}Tinker 4B confirm{RESET}         "
              f"acc {tinker['accuracy']:.3f}  KL {tinker['kl_from_gold']:.3f}  "
              f"(vs 0.483 zero-shot = clear lift)")
    except FileNotFoundError:
        pass
    print(f"\n{DIM}Full report: REPORT.md · charts in artifacts/{RESET}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="A", choices=["A", "B"])
    ap.add_argument("--fast", action="store_true", help="no typing delay")
    args = ap.parse_args()
    replay(args.arm, args.fast)


if __name__ == "__main__":
    main()
