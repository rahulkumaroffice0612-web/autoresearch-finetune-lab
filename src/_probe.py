"""Direct 2-step training probe to locate the stall. Fully unbuffered."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def log(msg: str) -> None:
    print(f"[probe +{time.time()-T0:7.1f}s] {msg}", flush=True)


T0 = time.time()
log("start imports")

from transformers import TrainerCallback  # noqa: E402

from data_io import load_cases  # noqa: E402
from train_config import TrainConfig  # noqa: E402
import trainer as T  # noqa: E402

log("imports done")


class StepSpy(TrainerCallback):
    """Flush a line at every lifecycle hook so a stall is pinpointed."""

    def on_train_begin(self, args, state, control, **kw):
        log("HOOK on_train_begin")

    def on_step_begin(self, args, state, control, **kw):
        log(f"HOOK on_step_begin step={state.global_step}")

    def on_step_end(self, args, state, control, **kw):
        log(f"HOOK on_step_end  step={state.global_step}")

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        log(f"HOOK on_evaluate  step={state.global_step} metrics={metrics}")

    def on_log(self, args, state, control, logs=None, **kw):
        log(f"HOOK on_log       step={state.global_step} logs={logs}")


def main():
    from unsloth import FastDecisionModel, DecisionTrainer

    log("loading fit/select rows")
    base = Path(__file__).resolve().parent.parent / "data" / "splits"
    fit_rows = load_cases(base / "fit.jsonl")[:40]
    select_rows = load_cases(base / "select.jsonl")[:20]
    log(f"rows: fit={len(fit_rows)} select={len(select_rows)}")

    log("load_base() start")
    model, tok = T.load_base()
    log("load_base() done")

    cfg = TrainConfig(reason="probe")
    model = FastDecisionModel.get_peft_model(
        model, r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
    )
    log("peft attached")
    if cfg.freeze_backbone:
        FastDecisionModel.freeze_backbone(model)
        log("backbone frozen")

    fit_items = T.build_items(fit_rows, tok, model)
    log(f"fit_items built n={len(fit_items)}")
    eval_items = T.build_items(select_rows, tok, model)
    log(f"eval_items built n={len(eval_items)}")

    args = T.make_args(cfg, max_steps=2, output_dir="artifacts/_probe", eval_steps=1)
    log("args built, max_steps=2 eval_steps=1")

    trn = DecisionTrainer(
        model=model, args=args, train_dataset=fit_items, eval_dataset=eval_items,
        head_learning_rate=cfg.head_learning_rate, tokenizer=tok,
        label_smoothing=cfg.label_smoothing, callbacks=[StepSpy()],
    )
    log("DecisionTrainer constructed; calling train() NOW")

    trn.train()
    log("train() RETURNED")

    log("calling evaluate()")
    m = FastDecisionModel.evaluate(model, tok, eval_items)
    log(f"evaluate() done: {m}")
    log("PROBE COMPLETE OK")


if __name__ == "__main__":
    main()
