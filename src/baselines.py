"""Prior and Uniform reference baselines — the model-free rows that let us validate
the scorer against the benchmark's published numbers (design doc §8.9 read #1).

Target numbers to reproduce (from the dataset card):
  Prior   : accuracy 0.470, KL 0.347, Brier 0.189, ECE 0.088
  Uniform : accuracy 0.308, KL 0.444, Brier 0.238, ECE 0.169

Prior   = for each (workflow, question) cell, predict the mean gold distribution
          over the TRAIN split, i.e. the label frequencies, ignoring the input.
Uniform = equal probability on every option of every question.

Both are computed per the one-request-per-case shape: the predictor returns a dict
{qid: {"probabilities": {...}, "type": ...}} for every question in the case.
"""
from __future__ import annotations

from collections import defaultdict


def build_prior(train_cases: list[dict]) -> dict:
    """Mean gold distribution per QUESTION id, pooled across all workflows, over train.

    The Prior *ignores the input* — and the workflow is part of the input — so the
    distribution is keyed on question id alone, NOT on (workflow, qid). Keying it per
    workflow leaks information and inflates the Prior above its intended floor; this
    was caught by the harness check (per-workflow gave accuracy 0.481 vs the card's
    0.470; global-per-question gives 0.472, matching to 3 dp).

    Returns {qid: {"probabilities": {option: prob}, "type": ...}}.
    """
    sums: dict = defaultdict(lambda: defaultdict(float))
    counts: dict = defaultdict(int)
    types: dict = {}
    for case in train_cases:
        for qid, gq in case["gold"].items():
            counts[qid] += 1
            types[qid] = gq["type"]
            for opt, pr in gq["probabilities"].items():
                sums[qid][opt] += float(pr)

    prior: dict = {}
    for qid, opt_sums in sums.items():
        n = counts[qid]
        dist = {opt: s / n for opt, s in opt_sums.items()}
        total = sum(dist.values())
        if total > 0:  # renormalise against float drift
            dist = {opt: v / total for opt, v in dist.items()}
        prior[qid] = {"probabilities": dist, "type": types[qid]}
    return prior


def prior_predictor(prior: dict):
    """A predict(case) that returns the train-mean distribution for each question."""
    def predict(case: dict) -> dict:
        out = {}
        for qid, gq in case["gold"].items():
            entry = prior.get(qid)
            opts = list(gq["probabilities"].keys())
            if entry is None:  # unseen question: fall back to uniform over its options
                out[qid] = {"probabilities": {o: 1.0 / len(opts) for o in opts}}
            else:
                # restrict to this question's option set, then renormalise
                dist = {o: entry["probabilities"].get(o, 0.0) for o in opts}
                total = sum(dist.values())
                dist = {o: v / total for o, v in dist.items()} if total > 0 else \
                    {o: 1.0 / len(opts) for o in opts}
                out[qid] = {"probabilities": dist, "type": entry["type"]}
        return out
    return predict


def uniform_predictor():
    """A predict(case) that puts equal mass on every option of every question.

    Option sets are read from the case's gold (same options the scorer uses), so
    this needs no training data.
    """
    def predict(case: dict) -> dict:
        out = {}
        for qid, gq in case["gold"].items():
            opts = list(gq["probabilities"].keys())
            out[qid] = {"probabilities": {o: 1.0 / len(opts) for o in opts}}
        return out
    return predict
