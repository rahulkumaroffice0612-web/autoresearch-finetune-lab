#!/usr/bin/env python3
"""Per-domain (per-workflow) breakdown of the winning model's sealed-test score.

Uses the predictions already in artifacts/test_predictions.jsonl (no re-run, no new
sealed read) joined with the sealed gold labels, and reports accuracy / KL / Brier /
ECE for each of the four typed-decisions domains. This is the evidence for "which
domain does LazyDecider generalize to best" — real numbers, not claims.

Output: artifacts/per_domain_scores.json  (and a printed table).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scorer import score_one, aggregate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEALED = ROOT / "data" / "sealed" / "test.jsonl"
PREDS = ROOT / "artifacts" / "test_predictions.jsonl"
OUT = ROOT / "artifacts" / "per_domain_scores.json"


def domain_of(case_id: str) -> str:
    return "_".join(case_id.split("_")[:-1])


def main() -> None:
    # gold by id
    gold_by_id = {}
    for line in SEALED.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gold_by_id[row["id"]] = json.loads(row["gold"]) if isinstance(row["gold"], str) else row["gold"]

    # predictions by id
    pred_by_id = {}
    for line in PREDS.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        pred_by_id[row["id"]] = row["answers"]

    # accumulate per domain
    buckets = defaultdict(lambda: {"accs": [], "kls": [], "briers": [], "confs": []})
    overall = {"accs": [], "kls": [], "briers": [], "confs": []}

    for cid, gold in gold_by_id.items():
        answers = pred_by_id.get(cid, {})
        dom = domain_of(cid)
        for qid, gq in gold.items():
            ok, kl, brier, conf = score_one(answers.get(qid, {"probabilities": {}}), gq)
            for tgt in (buckets[dom], overall):
                tgt["accs"].append(ok)
                tgt["kls"].append(kl)
                tgt["briers"].append(brier)
                tgt["confs"].append(conf)

    report = {"overall": aggregate(**{k: overall[k] for k in overall})}
    for dom, b in buckets.items():
        report[dom] = {"n_cases": sum(1 for cid in gold_by_id if domain_of(cid) == dom),
                       **aggregate(**{k: b[k] for k in b})}

    OUT.write_text(json.dumps(report, indent=2))

    # pretty print, best-accuracy first (excluding overall)
    print(f"{'domain':<28} {'cases':>5} {'acc':>7} {'KL':>7} {'Brier':>7} {'ECE':>7}")
    print("-" * 66)
    doms = sorted((d for d in report if d != "overall"), key=lambda d: -report[d]["accuracy"])
    for d in doms:
        r = report[d]
        print(f"{d:<28} {r['n_cases']:>5} {r['accuracy']:>7.3f} {r['kl_from_gold']:>7.3f} "
              f"{r['brier']:>7.3f} {r['ece']:>7.3f}")
    o = report["overall"]
    print("-" * 66)
    print(f"{'OVERALL':<28} {'':>5} {o['accuracy']:>7.3f} {o['kl_from_gold']:>7.3f} "
          f"{o['brier']:>7.3f} {o['ece']:>7.3f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
