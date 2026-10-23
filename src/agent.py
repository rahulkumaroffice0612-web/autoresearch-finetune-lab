"""The researcher: proposes one TrainConfig per experiment.

Two implementations with the same interface `propose(history) -> (TrainConfig, source)`:
  - GemmaProposer: asks Gemma 4 E4B via LM Studio's OpenAI-compatible endpoint,
    feeding the recent experiment history (incl. prune reasons) so proposals are
    informed by what died and why (design doc §8.6 "feed kills back to the agent").
  - RandomProposer: deterministic draw from the search space. The fallback when the
    agent errors or emits an invalid config twice, and the way the loop runs tonight
    if LM Studio is not up. Logged as source="random".

The agent decides WHAT to try; ASHA decides WHEN to quit. The agent never sees test.
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict

from train_config import BOUNDS, SCHEDULERS, TrainConfig

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://127.0.0.1:1234/v1")
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "google/gemma-4-e4b")

SYSTEM_PROMPT = """You are an ML hyperparameter researcher running an overnight search.
You fine-tune a small decision model (Qwen3.5-0.8B + Clef head) on 960 training cases,
scored by holdout LOSS (lower is better; it is KL divergence from soft gold labels).
Each run trains head-only (frozen encoder) by default — fast, ~3.3s/step.

Propose ONE config as strict JSON with these keys and bounds:
{bounds}
lr_scheduler one of: {schedulers}
freeze_backbone: boolean (true = fast head-only path; false = also tune LoRA on encoder)
reason: one short sentence on why this config, given the history.

Output ONLY the JSON object, nothing else."""


def _history_digest(history: list[dict], k: int = 12) -> str:
    """Compact recent runs for the prompt: config deltas, loss, prune reason."""
    lines = []
    for rec in history[-k:]:
        sel = rec.get("select", {})
        loss = sel.get("loss")
        tag = "PRUNED" if rec.get("pruned") else "full"
        cfg = rec.get("config", {})
        brief = {k2: cfg.get(k2) for k2 in ("head_learning_rate", "label_smoothing",
                                            "lora_r", "batch_size", "freeze_backbone")}
        lines.append(f"- loss={loss!r} [{tag}] {json.dumps(brief)} "
                     f"{'(' + rec['prune_reason'] + ')' if rec.get('prune_reason') else ''}")
    return "\n".join(lines) if lines else "(no runs yet)"


class RandomProposer:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def propose(self, history: list[dict]) -> tuple[TrainConfig, str]:
        r = self.rng
        d = {
            "freeze_backbone": r.random() < 0.8,  # favour the fast path
            "head_learning_rate": round(10 ** r.uniform(-4, -2.5), 6),
            "learning_rate": round(10 ** r.uniform(-5, -3.3), 6),
            "label_smoothing": round(r.uniform(*BOUNDS["label_smoothing"]), 3),
            "lora_r": r.choice([8, 16, 32]),
            "lora_alpha": r.choice([8, 16, 32, 64]),
            "lora_dropout": round(r.uniform(0, 0.1), 3),
            "batch_size": r.choice([2, 4]),
            "grad_accum": r.choice([1, 2, 4]),
            "warmup_ratio": round(r.uniform(0, 0.1), 3),
            "weight_decay": round(r.uniform(0, 0.1), 3),
            "lr_scheduler": r.choice(SCHEDULERS),
            "epochs": r.choice([1, 2]),
            "reason": "random draw from search space",
        }
        return TrainConfig.from_dict(d), "random"


class GemmaProposer:
    def __init__(self, fallback: RandomProposer | None = None, retries: int = 2):
        self.fallback = fallback or RandomProposer()
        self.retries = retries

    def _ask(self, history: list[dict]) -> dict:
        import requests  # local import so the module loads without the dep present
        sys_prompt = SYSTEM_PROMPT.format(
            bounds=json.dumps({k: list(v) for k, v in BOUNDS.items()}),
            schedulers=list(SCHEDULERS),
        )
        user = f"History (recent runs):\n{_history_digest(history)}\n\nPropose the next config as JSON."
        resp = requests.post(
            f"{LM_STUDIO_URL}/chat/completions",
            json={
                "model": GEMMA_MODEL,
                "messages": [{"role": "system", "content": sys_prompt},
                             {"role": "user", "content": user}],
                "temperature": 0.8,
                "response_format": {"type": "json_object"},
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return json.loads(content)

    def propose(self, history: list[dict]) -> tuple[TrainConfig, str]:
        for attempt in range(self.retries):
            try:
                raw = self._ask(history)
                return TrainConfig.from_dict(raw), "gemma"
            except Exception as e:  # network error, bad JSON, schema violation
                last = f"{type(e).__name__}: {str(e)[:120]}"
        # give up -> deterministic random draw, logged as the reason
        cfg, _ = self.fallback.propose(history)
        cfg.reason = f"random fallback after {self.retries} invalid agent proposals ({last})"
        return cfg, "random_fallback"
