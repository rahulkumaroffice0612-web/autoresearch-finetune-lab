"""Retrain the best config found by the search and SAVE a submittable model.

The search worker (worker.py) deliberately does not checkpoint — it only records
configs + metrics to experiments_arm{A,B}.jsonl (save_strategy="no", §8.4). This
script closes that loop: pick the lowest-select-loss kept run, retrain exactly that
config on the full fit split, and persist a standalone merged model.

Usage:
  python src/finalize.py --arm A                 # best of experiments_armA.jsonl
  python src/finalize.py --arm A --out artifacts/best_model
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
ARTIFACTS = ROOT / "artifacts"


def load_split(name: str):
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from data_io import load_cases
    return load_cases(SPLITS / f"{name}.jsonl")


def best_config(log_path: Path):
    """Return (config_dict, record) for the lowest-loss kept, non-pruned run."""
    best = None
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sel = rec.get("select", {})
            loss = sel.get("loss")
            if loss is None or rec.get("pruned") or "config" not in rec:
                continue
            if best is None or loss < best[1]["select"]["loss"]:
                best = (rec["config"], rec)
    if best is None:
        raise SystemExit(f"No eligible run in {log_path}")
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "B"], default="A")
    ap.add_argument("--out", default=str(ARTIFACTS / "best_model"))
    ap.add_argument("--merged", action="store_true", default=True,
                    help="save a standalone merged model (default) vs adapter-only")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from train_config import TrainConfig
    import trainer as T

    log_path = ARTIFACTS / f"experiments_arm{args.arm}.jsonl"
    cfg_dict, rec = best_config(log_path)
    config = TrainConfig.from_dict({**cfg_dict, "reason": "finalize: retrain best"})
    print(f"[finalize] best run {rec.get('run')} select-loss={rec['select']['loss']:.4f} "
          f"acc={rec['select'].get('accuracy'):.3f} ece={rec['select'].get('ece'):.3f}", flush=True)
    print(f"[finalize] config: {json.dumps(config.key())}", flush=True)

    fit_rows = load_split("fit")
    select_rows = load_split("select")

    # Train this one config from a fresh base, score it, and SAVE the same model
    # (single pass — no wasteful retrain).
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = _train_score_save(config, fit_rows, select_rows, out)
    print(f"[finalize] retrained select: {json.dumps(metrics)}", flush=True)
    print(f"[finalize] SAVED model to {out}", flush=True)
    (out / "config.json").write_text(json.dumps(config.key(), indent=2) + "\n")
    (out / "select_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return 0


def _train_score_save(config, fit_rows, select_rows, out: Path) -> dict:
    """Train the config once, score on full select, then persist the model."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import trainer as T
    from unsloth import FastDecisionModel, DecisionTrainer

    model, tok = T.load_base()
    model = FastDecisionModel.get_peft_model(
        model, r=config.lora_r, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )
    if config.freeze_backbone:
        FastDecisionModel.freeze_backbone(model)
    fit_items = T.build_items(fit_rows, tok, model)
    full_select = T.build_items(select_rows, tok, model)
    max_steps = T.steps_for(config, len(fit_rows))
    targs = T.make_args(config, max_steps, ARTIFACTS / "_run_finalize", eval_steps=max_steps)
    trainer = DecisionTrainer(
        model=model, args=targs, train_dataset=fit_items, eval_dataset=full_select,
        head_learning_rate=config.head_learning_rate, tokenizer=tok,
        label_smoothing=config.label_smoothing, callbacks=[],
    )
    trainer.train()
    metrics = {k: float(v) for k, v in
               FastDecisionModel.evaluate(model, tok, full_select).items()}
    # Merged = standalone, submittable; falls back to adapter save for head-only.
    try:
        model.save_pretrained_merged(str(out), tok)
    except Exception:
        model.save_pretrained(str(out), tok)
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
