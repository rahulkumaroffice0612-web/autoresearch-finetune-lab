"""Official Typed Decisions scorer — arithmetic copied verbatim from the dataset's
own bench/score_with_jobs.py (read 2026-10-23), per design doc §18.2.

This is the ruler. If score_references() does not reproduce the published Prior
and Uniform rows exactly, nothing downstream can be trusted. Do NOT "improve" the
math here — matching the benchmark is the whole point.

Four metrics, per decision:
  accuracy = argmax(pred) == argmax(gold)
  kl       = sum_i g_i * log(g_i / max(p_i, 1e-9))   for g_i > 0   # KL(gold || pred)
  brier    = sum_i (p_i - g_i)^2
  conf     = max(p)                                               # for ECE
ECE = 10 equal-width bins on max(p), population-weighted.
Final numbers are plain means over all decisions.
"""
from __future__ import annotations

import math

# yes/no -> true/false remap: engines report yes/no, gold uses true/false.
_NOUL_REMAP = {"yes": "true", "no": "false"}


def score_one(pred: dict, gold: dict) -> tuple[bool, float, float, float]:
    """Score one decision (one question of one case). Mirrors bench score_one()."""
    opts = list(gold["probabilities"].keys())
    g = [float(gold["probabilities"][o]) for o in opts]

    if gold.get("type") == "noul":
        pred = dict(pred)
        pred["probabilities"] = {
            _NOUL_REMAP.get(k, k): v for k, v in pred.get("probabilities", {}).items()
        }

    p = [float(pred.get("probabilities", {}).get(o, 0.0)) for o in opts]
    s = sum(p)
    p = [v / s for v in p] if s > 0 else [1.0 / len(opts)] * len(opts)

    am_p = p.index(max(p))
    am_g = g.index(max(g))
    kl = sum(gi * math.log(gi / max(pi, 1e-9)) for gi, pi in zip(g, p) if gi > 0)
    brier = sum((pi - gi) ** 2 for pi, gi in zip(p, g))
    return am_p == am_g, kl, brier, max(p)


def aggregate(accs: list[bool], kls: list[float], briers: list[float],
              confs: list[float]) -> dict:
    """Mean accuracy/KL/Brier plus 10-bin population-weighted ECE. Mirrors bench main()."""
    n = len(accs)
    if n == 0:
        raise ValueError("no decisions scored")

    n_bins = 10
    bin_hits = [0] * n_bins
    bin_total = [0] * n_bins
    bin_conf = [0.0] * n_bins
    for ok, conf in zip(accs, confs):
        b = min(int(conf * n_bins), n_bins - 1)
        bin_total[b] += 1
        bin_hits[b] += int(ok)
        bin_conf[b] += conf
    ece = 0.0
    for b in range(n_bins):
        if bin_total[b]:
            ece += bin_total[b] / n * abs(bin_hits[b] / bin_total[b] - bin_conf[b] / bin_total[b])

    return {
        "n_decisions": n,
        "accuracy": sum(accs) / n,
        "kl_from_gold": sum(kls) / n,
        "brier": sum(briers) / n,
        "ece": ece,
    }


def score_predictions(cases: list[dict], predict) -> dict:
    """Score a predictor over cases. `predict(case) -> {qid: {probabilities: {...}}}`.

    Each case has a `gold` dict of {qid: gold_question}. Every question is one decision.
    """
    accs, kls, briers, confs = [], [], [], []
    for case in cases:
        gold = case["gold"]
        answers = predict(case)
        for qid, gq in gold.items():
            ok, kl, brier, conf = score_one(answers.get(qid, {"probabilities": {}}), gq)
            accs.append(ok)
            kls.append(kl)
            briers.append(brier)
            confs.append(conf)
    return aggregate(accs, kls, briers, confs)
