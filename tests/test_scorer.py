"""Unit tests for the scorer — hand-computed expected values, no dataset needed.
Run: python -m pytest tests/ -q   (or: python tests/test_scorer.py)
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from scorer import score_one, aggregate, score_predictions  # noqa: E402


def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol


def test_perfect_prediction():
    gold = {"type": "choice", "probabilities": {"a": 0.7, "b": 0.3}}
    ok, kl, brier, conf = score_one({"probabilities": {"a": 0.7, "b": 0.3}}, gold)
    assert ok is True
    assert approx(kl, 0.0)
    assert approx(brier, 0.0)
    assert approx(conf, 0.7)


def test_argmax_mismatch():
    gold = {"type": "choice", "probabilities": {"a": 0.6, "b": 0.4}}
    ok, _, _, conf = score_one({"probabilities": {"a": 0.3, "b": 0.7}}, gold)
    assert ok is False
    assert approx(conf, 0.7)


def test_renormalisation():
    # prediction sums to 0.5 -> renormalised to {a:0.4, b:0.6}
    gold = {"type": "choice", "probabilities": {"a": 0.5, "b": 0.5}}
    ok, _, brier, conf = score_one({"probabilities": {"a": 0.2, "b": 0.3}}, gold)
    assert approx(conf, 0.6)
    assert approx(brier, (0.4 - 0.5) ** 2 + (0.6 - 0.5) ** 2)


def test_missing_option_scores_zero_then_renorm():
    gold = {"type": "choice", "probabilities": {"a": 0.5, "b": 0.5}}
    # only 'a' predicted -> b=0.0, renormalise -> {a:1.0, b:0.0}
    ok, _, brier, conf = score_one({"probabilities": {"a": 0.9}}, gold)
    assert approx(conf, 1.0)
    assert approx(brier, (1.0 - 0.5) ** 2 + (0.0 - 0.5) ** 2)


def test_empty_prediction_falls_back_to_uniform():
    gold = {"type": "choice", "probabilities": {"a": 0.5, "b": 0.5}}
    ok, _, _, conf = score_one({"probabilities": {}}, gold)
    assert approx(conf, 0.5)  # uniform over 2 options


def test_noul_yes_no_remap():
    gold = {"type": "noul", "probabilities": {"true": 0.8, "false": 0.2}}
    # engine reports yes/no; must remap to true/false before comparing
    ok, kl, brier, conf = score_one({"probabilities": {"yes": 0.8, "no": 0.2}}, gold)
    assert ok is True
    assert approx(kl, 0.0)
    assert approx(brier, 0.0)


def test_kl_direction_and_value():
    gold = {"type": "choice", "probabilities": {"a": 0.5, "b": 0.5}}
    pred = {"probabilities": {"a": 0.9, "b": 0.1}}
    _, kl, _, _ = score_one(pred, gold)
    expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    assert approx(kl, expected)


def test_ece_perfectly_calibrated_is_zero():
    # One decision, confidence 1.0, correct -> bin accuracy == bin confidence.
    out = aggregate([True], [0.0], [0.0], [1.0])
    assert approx(out["ece"], 0.0)


def test_ece_fully_miscalibrated():
    # confidence 1.0 but wrong -> |acc - conf| = |0 - 1| = 1
    out = aggregate([False], [0.0], [0.0], [1.0])
    assert approx(out["ece"], 1.0)


def test_score_predictions_counts_every_question():
    cases = [{
        "gold": {
            "q1": {"type": "choice", "probabilities": {"a": 1.0, "b": 0.0}},
            "q2": {"type": "noul", "probabilities": {"true": 0.5, "false": 0.5}},
        }
    }]

    def predict(case):
        return {
            "q1": {"probabilities": {"a": 1.0, "b": 0.0}},
            "q2": {"probabilities": {"true": 0.5, "false": 0.5}},
        }

    out = score_predictions(cases, predict)
    assert out["n_decisions"] == 2
    assert approx(out["accuracy"], 1.0)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
