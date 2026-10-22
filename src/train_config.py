"""Training config: the contract the overnight agent (Gemma) writes as config.json,
and the only knobs the ASHA loop searches. Validated against a strict schema so a
malformed agent proposal is rejected (then retried, then a random draw) rather than
crashing the worker — design doc §8.6.

Field set is derived from the verified DecisionTrainer / get_peft_model / args API
(inspected 2026-10-23), not guessed:
  - head_learning_rate, label_smoothing  -> DecisionTrainer(**)
  - r, lora_alpha, lora_dropout           -> get_peft_model(**)
  - learning_rate, batch_size, grad_accum, warmup_ratio, weight_decay, lr_scheduler,
    epochs                                -> TrainingArguments
  - freeze_backbone                        -> head-only fast path (§8.4)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

# (min, max) bounds the agent must stay within; also the random-fallback draw range.
BOUNDS = {
    "head_learning_rate": (1e-5, 5e-3),
    "learning_rate": (1e-6, 1e-3),
    "label_smoothing": (0.0, 0.2),
    "lora_r": (4, 64),
    "lora_alpha": (4, 128),
    "lora_dropout": (0.0, 0.2),
    "batch_size": (1, 8),
    "grad_accum": (1, 8),
    "warmup_ratio": (0.0, 0.2),
    "weight_decay": (0.0, 0.2),
    "epochs": (1, 3),
}
SCHEDULERS = ("linear", "cosine", "constant", "constant_with_warmup")


@dataclass
class TrainConfig:
    # head-only (frozen backbone) is the 2.99 s/step fast path; the agent may flip it
    freeze_backbone: bool = True
    head_learning_rate: float = 1e-3
    learning_rate: float = 2e-4          # backbone/LoRA LR when not frozen
    label_smoothing: float = 0.0
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    batch_size: int = 4
    grad_accum: int = 2
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    lr_scheduler: str = "linear"
    epochs: int = 2
    # free-text, logged for the morning report; never affects training
    reason: str = ""

    def validate(self) -> "TrainConfig":
        errs = []
        for name, (lo, hi) in BOUNDS.items():
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                errs.append(f"{name}: expected number, got {v!r}")
            elif not (lo <= v <= hi):
                errs.append(f"{name}: {v} out of bounds [{lo}, {hi}]")
        if self.lr_scheduler not in SCHEDULERS:
            errs.append(f"lr_scheduler: {self.lr_scheduler!r} not in {SCHEDULERS}")
        if not isinstance(self.freeze_backbone, bool):
            errs.append(f"freeze_backbone: expected bool, got {self.freeze_backbone!r}")
        if errs:
            raise ValueError("invalid config:\n  " + "\n  ".join(errs))
        return self

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**d).validate()

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    def key(self) -> dict:
        """The training-relevant fields only (excludes `reason`), for logging/dedup."""
        return {k: v for k, v in asdict(self).items() if k != "reason"}
