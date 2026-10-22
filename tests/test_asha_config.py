"""Unit tests for ASHA pruning + config schema — no model, no dataset.
Run: python tests/test_asha_config.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from asha import ASHAPruner, nearest_rung, losses_at_rung, _percentile, RUNGS  # noqa: E402
from train_config import TrainConfig  # noqa: E402


class FakeControl:
    should_training_stop = False


class FakeState:
    def __init__(self, step):
        self.global_step = step


def test_nearest_rung():
    assert nearest_rung(20) == 20
    assert nearest_rung(21) == 20          # within tol
    assert nearest_rung(60) == 60
    assert nearest_rung(100) is None       # between rungs
    assert nearest_rung(240) == 240


def test_percentile():
    assert _percentile([], 50) == float("inf")
    assert abs(_percentile([1, 2, 3, 4], 100 / 3) - 2.0) < 0.01


def test_cold_start_never_prunes():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "exp.jsonl"
        # only 2 prior runs at rung 20 -> below MIN_HISTORY=5
        with open(log, "w") as f:
            for loss in (1.0, 1.1):
                f.write(json.dumps({"rung_evals": [{"rung": 20, "eval_loss": loss}]}) + "\n")
        pr = ASHAPruner(log)
        ctrl = FakeControl()
        pr.on_evaluate(None, FakeState(20), ctrl, metrics={"eval_loss": 99.0})
        assert ctrl.should_training_stop is False
        assert pr.pruned_at is None
        assert pr.rung_evals == [{"rung": 20, "step": 20, "eval_loss": 99.0}]


def test_prunes_when_worse_than_cutoff():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "exp.jsonl"
        with open(log, "w") as f:
            for loss in (0.5, 0.6, 0.7, 0.8, 0.9):  # 5 peers at rung 20
                f.write(json.dumps({"rung_evals": [{"rung": 20, "eval_loss": loss}]}) + "\n")
        pr = ASHAPruner(log)
        ctrl = FakeControl()
        # cutoff = p33 of [0.5..0.9] ~ 0.633; our 2.0 is far worse -> prune
        pr.on_evaluate(None, FakeState(20), ctrl, metrics={"eval_loss": 2.0})
        assert ctrl.should_training_stop is True
        assert pr.pruned_at == 20
        assert "pruned at rung 20" in pr.prune_reason


def test_survives_when_better_than_cutoff():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "exp.jsonl"
        with open(log, "w") as f:
            for loss in (0.5, 0.6, 0.7, 0.8, 0.9):
                f.write(json.dumps({"rung_evals": [{"rung": 20, "eval_loss": loss}]}) + "\n")
        pr = ASHAPruner(log)
        ctrl = FakeControl()
        pr.on_evaluate(None, FakeState(20), ctrl, metrics={"eval_loss": 0.4})  # best
        assert ctrl.should_training_stop is False
        assert pr.pruned_at is None


def test_losses_at_rung_filters_by_rung():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "exp.jsonl"
        with open(log, "w") as f:
            f.write(json.dumps({"rung_evals": [
                {"rung": 20, "eval_loss": 1.0}, {"rung": 60, "eval_loss": 0.8}]}) + "\n")
        assert losses_at_rung(log, 20) == [1.0]
        assert losses_at_rung(log, 60) == [0.8]
        assert losses_at_rung(log, 140) == []


def test_config_defaults_valid():
    TrainConfig().validate()  # must not raise


def test_config_rejects_out_of_bounds():
    try:
        TrainConfig(head_learning_rate=1.0).validate()
        assert False, "should have raised"
    except ValueError as e:
        assert "head_learning_rate" in str(e)


def test_config_rejects_unknown_key():
    try:
        TrainConfig.from_dict({"nonsense": 1})
        assert False, "should have raised"
    except ValueError as e:
        assert "unknown" in str(e).lower()


def test_config_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "config.json"
        cfg = TrainConfig(head_learning_rate=2e-3, reason="test")
        cfg.save(p)
        loaded = TrainConfig.load(p)
        assert loaded.head_learning_rate == 2e-3
        assert loaded.reason == "test"
        assert "reason" not in loaded.key()


def test_final_rung_is_240():
    assert RUNGS[-1] == 240


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
