"""Build and run one training trial from a TrainConfig, scored on `select`.

API verified against unsloth 2026.10.7 + transformers 5.17.0 (2026-10-23):
  - FastDecisionModel.from_pretrained / get_peft_model / freeze_backbone / build_dataset / evaluate
  - DecisionTrainer(model, args, train_dataset, eval_dataset, head_learning_rate=,
    tokenizer=, label_smoothing=, callbacks=)
  - evaluate() -> {'accuracy','ece','loss','record_accuracy'}
  - TrainingArguments on tfm5: warmup_steps (not warmup_ratio), eval_strategy

LoRA must not accumulate across trials, so each trial reloads the base model. The
import stays paid-once (the expensive 13s `import unsloth`); only from_pretrained
(~51s) repeats. If that proves too slow in the worker we can switch to adapter
reset, but reload is correct-by-construction and the measure step will tell us.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

from transformers import TrainingArguments

from asha import RUNGS
from train_config import TrainConfig

MODEL_NAME = "unsloth/Qwen3.5-0.8B"
MAX_SEQ_LENGTH = 1280
FINAL_RUNG = RUNGS[-1]  # 240


def load_base():
    """Load the Clef model + tokenizer. ~51s measured (from_pretrained)."""
    from unsloth import FastDecisionModel
    return FastDecisionModel.from_pretrained(
        MODEL_NAME, max_seq_length=MAX_SEQ_LENGTH, load_in_4bit=False,
    )


def build_items(rows, tok, model):
    from unsloth import FastDecisionModel
    items, _ = FastDecisionModel.build_dataset(rows, tok, model)
    return items


def steps_for(config: TrainConfig, n_fit_cases: int) -> int:
    eff_batch = config.batch_size * config.grad_accum
    per_epoch = max(1, math.ceil(n_fit_cases / eff_batch))
    return per_epoch * config.epochs


def make_args(config: TrainConfig, max_steps: int, output_dir: str | Path,
              eval_steps: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        max_steps=max_steps,
        warmup_steps=int(round(config.warmup_ratio * max_steps)),
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        weight_decay=config.weight_decay,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=eval_steps,
        save_strategy="no",
        report_to=[],
        seed=3407,
    )


def train_and_score(config: TrainConfig, fit_rows, select_rows, pruner,
                    output_dir: str | Path, eval_steps: int = 20) -> dict:
    """Run one trial on a FRESH base model and score it on full `select`.

    ASHA's rung checks use a cheap 20-case subsample as the trainer's eval_dataset
    (~5s/eval vs ~36s for full select — measured §8.4). The keep/revert decision uses
    the FULL select evaluate() after training, which is the authoritative number.

    Returns a result dict: config key, select metrics, step/timing info, prune state.
    """
    from unsloth import FastDecisionModel, DecisionTrainer
    from asha import EARLY_EVAL_SUBSAMPLE

    model, tok = load_base()
    model = FastDecisionModel.get_peft_model(
        model, r=config.lora_r, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )
    if config.freeze_backbone:
        FastDecisionModel.freeze_backbone(model)

    fit_items = build_items(fit_rows, tok, model)
    # Cheap, fixed subsample for in-loop ASHA eval. select_rows is pre-shuffled and
    # identical across runs (frozen split), so [:N] is a stable, fair subsample.
    eval_subsample = build_items(select_rows[:EARLY_EVAL_SUBSAMPLE], tok, model)
    full_select = build_items(select_rows, tok, model)

    max_steps = steps_for(config, len(fit_rows))
    args = make_args(config, max_steps, output_dir, eval_steps)

    trainer = DecisionTrainer(
        model=model, args=args, train_dataset=fit_items, eval_dataset=eval_subsample,
        head_learning_rate=config.head_learning_rate, tokenizer=tok,
        label_smoothing=config.label_smoothing, callbacks=[pruner],
    )

    t0 = time.time()
    trainer.train()
    train_s = time.time() - t0
    final_step = int(getattr(trainer.state, "global_step", 0))

    # Compact per-logging-step curve (every eval_steps, ~12 points for 240 steps, a
    # few KB). transformers builds state.log_history during training; we keep only the
    # fields useful for a learning-curve plot and drop the rest.
    train_curve = _extract_curve(getattr(trainer.state, "log_history", []))

    # Authoritative score: FULL select, regardless of pruning. A pruned run still gets
    # a (worse) number, which the agent learns from. evaluate() -> accuracy/ece/loss/record_accuracy.
    metrics = FastDecisionModel.evaluate(model, tok, full_select)

    return {
        "config": config.key(),
        "reason": config.reason,
        "select": {k: float(v) for k, v in metrics.items()},
        "max_steps": max_steps,
        "final_step": final_step,
        "train_seconds": round(train_s, 2),
        "s_per_step": round(train_s / max(1, final_step), 3),
        "pruned": pruner.pruned_at is not None,
        "prune_reason": pruner.prune_reason,
        "rung_evals": pruner.rung_evals,
        "train_curve": train_curve,
    }


def _extract_curve(log_history: list[dict]) -> list[dict]:
    """Merge train/eval log points into one step-indexed curve for plotting.

    transformers emits separate dicts for train logs ({'loss','learning_rate',...})
    and eval logs ({'eval_loss',...}); we fold them by step and keep only the fields a
    learning-curve plot needs. ~12 points/run, a few KB."""
    by_step: dict[int, dict] = {}
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        pt = by_step.setdefault(int(step), {"step": int(step)})
        if "loss" in entry:
            pt["train_loss"] = round(float(entry["loss"]), 5)
        if "eval_loss" in entry:
            pt["eval_loss"] = round(float(entry["eval_loss"]), 5)
        if "learning_rate" in entry:
            pt["lr"] = float(entry["learning_rate"])
        if "grad_norm" in entry:
            pt["grad_norm"] = round(float(entry["grad_norm"]), 5)
    return [by_step[s] for s in sorted(by_step)]
