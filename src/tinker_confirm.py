"""Tinker confirm run — reproduce the Typed Decisions fine-tune on a BIGGER model.

Why this exists (prize: Best Use of Tinker):
  The submitted checkpoint is a 0.8B Clef model trained on an 18 GB Mac (MLX/Unsloth).
  This script is the independent *confirmation* the Tinker category asks for: take the
  winning recipe, fine-tune a 4B model via the Tinker API (LoRA + cross-entropy), and
  score the SAME sealed test with the SAME official scorer. If the 4B run beats the
  zero-shot baseline, the recipe generalises beyond the laptop model.

Task mapping (Tinker = generic causal-LM LoRA, not Clef multi-head):
  Each case -> a text completion. Prompt = state + the 5 questions + allowed labels.
  Target  = a JSON object of soft label distributions (the gold is itself soft; we
            teach the model to emit per-label probabilities). Trained with
            loss_fn="cross_entropy", prompt tokens weighted 0, target tokens weighted 1.
  Eval    = greedy sample on the sealed test, parse the JSON, feed the official scorer
            (accuracy / KL / Brier / ECE).

HARD BUDGET CAP (real money): the loop aborts on ANY of
  --max-steps, --max-seconds, or --max-requests, whichever comes first. Defaults are
  sized for a ~$5 small run. Nothing in here loops unbounded.

Usage:
  python src/tinker_confirm.py --max-steps 60 --n-test 400 \
      --max-seconds 1500 --max-eval-seconds 1500
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ROOT / "data" / "splits"
SEALED = ROOT / "data" / "sealed" / "test.jsonl"
SEAL_MANIFEST = ROOT / "data" / "sealed" / "test.sha256"
ARTIFACTS = ROOT / "artifacts"
BASE_MODEL = "Qwen/Qwen3.5-4B"
LORA_RANK = 16
MAX_PROMPT_TOKENS = 1024
MAX_TARGET_TOKENS = 340

sys.path.insert(0, str(ROOT / "src"))
from data_io import load_cases                      # noqa: E402
from scorer import score_predictions                # noqa: E402


# ---------------------------------------------------------------------------
# Prompt / target construction
# ---------------------------------------------------------------------------
def _label_space(qid: str, q: dict) -> list[str]:
    """Allowed labels for a question, in a stable order."""
    crit = q.get("criteria")
    if isinstance(crit, dict):                 # choice / noul
        return list(crit.keys())
    if isinstance(crit, list):                 # score (0..N-1)
        return [str(i) for i in range(len(crit))]
    return []


def build_prompt(case: dict) -> str:
    state = case["state"]
    questions = case["questions"]
    lines = [
        "You are a calibrated decision model. For the agent trace below, answer EVERY",
        "question with a probability distribution over its allowed labels. Probabilities",
        "per question must sum to 1. Respond with ONLY a JSON object mapping each",
        'question id to {"label": <argmax>, "probabilities": {label: prob, ...}}.',
        "",
        "TRACE STATE:",
        json.dumps(state, ensure_ascii=False, sort_keys=True),
        "",
        "QUESTIONS:",
    ]
    for qid, q in questions.items():
        labels = _label_space(qid, q)
        lines.append(f"- {qid} ({q.get('type')}): {q.get('instructions','')}")
        lines.append(f"  allowed labels: {labels}")
    lines.append("")
    lines.append("JSON:")
    return "\n".join(lines)


def build_target(case: dict) -> str:
    """Teacher-forced target: the gold soft distributions, rounded, as compact JSON."""
    gold = case["gold"]
    out = {}
    for qid, gq in gold.items():
        probs = {k: round(float(v), 4) for k, v in gq.get("probabilities", {}).items()}
        label = max(probs, key=probs.get) if probs else gq.get("label")
        out[qid] = {"label": label, "probabilities": probs}
    return json.dumps(out, ensure_ascii=False, sort_keys=True)


# ---------------------------------------------------------------------------
# Datum construction (cross-entropy: prompt masked, target supervised)
# ---------------------------------------------------------------------------
def make_datum(tok, prompt: str, target: str):
    import tinker
    from tinker import types

    p_ids = tok.encode(prompt, add_special_tokens=False)[:MAX_PROMPT_TOKENS]
    t_ids = tok.encode(target + tok.eos_token, add_special_tokens=False)[:MAX_TARGET_TOKENS]
    # Next-token LM: input = p+t[:-1], predicting t shifted by one. We weight only the
    # positions that predict target tokens.
    full = p_ids + t_ids
    input_ids = full[:-1]
    target_ids = full[1:]
    weights = [0.0] * (len(p_ids) - 1) + [1.0] * len(t_ids)
    weights = weights[: len(target_ids)]
    model_input = tinker.ModelInput.from_ints(tokens=input_ids)
    loss_fn_inputs = {
        "target_tokens": types.TensorData.from_numpy(np.array(target_ids, dtype=np.int64)),
        "weights": types.TensorData.from_numpy(np.array(weights, dtype=np.float32)),
    }
    return tinker.Datum(model_input=model_input, loss_fn_inputs=loss_fn_inputs)


def _mean_loss(fb_out) -> float | None:
    """forward_backward returns a per-datum list of {'elementwise_loss','logprobs'};
    average elementwise_loss across all supervised tokens in the batch."""
    try:
        lo = fb_out.loss_fn_outputs
        if isinstance(lo, list):
            vals = []
            for d in lo:
                el = d.get("elementwise_loss") if hasattr(d, "get") else None
                if el is not None:
                    vals.append(np.asarray(el.to_numpy()))
            if vals:
                return float(np.concatenate(vals).mean())
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------
class Budget:
    def __init__(self, max_steps, max_seconds, max_requests):
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.max_requests = max_requests
        self.t0 = time.time()
        self.steps = 0
        self.requests = 0

    def tick_step(self):
        self.steps += 1

    def tick_request(self, n=1):
        self.requests += n

    def exceeded(self) -> str | None:
        if self.steps >= self.max_steps:
            return f"max_steps={self.max_steps}"
        if time.time() - self.t0 >= self.max_seconds:
            return f"max_seconds={self.max_seconds}"
        if self.requests >= self.max_requests:
            return f"max_requests={self.max_requests}"
        return None

    def elapsed(self) -> float:
        return round(time.time() - self.t0, 1)


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------
def _run(args):
    import hashlib
    import tinker
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=LORA_RANK)
    tok = tc.get_tokenizer()
    print(f"[tinker] training client {tc.get_info().model_id}", flush=True)

    # ---- TRAIN ----
    fit = load_cases(SPLITS / "fit.jsonl")
    if args.n_fit:
        fit = fit[: args.n_fit]
    print(f"[data] fit cases: {len(fit)}  batch={args.batch}", flush=True)

    budget = Budget(args.max_steps, args.max_seconds, args.max_requests)
    adam = tinker.AdamParams(learning_rate=args.lr, beta1=0.9, beta2=0.95,
                             eps=1e-8, weight_decay=0.01)
    data = [(build_prompt(c), build_target(c)) for c in fit]
    losses = []
    step = 0
    while True:
        why = budget.exceeded()
        if why:
            print(f"[budget] stop before step {step+1}: {why}", flush=True)
            break
        batch_cases = [data[(step * args.batch + i) % len(data)] for i in range(args.batch)]
        batch = [make_datum(tok, p, t) for p, t in batch_cases]
        fb = tc.forward_backward(batch, loss_fn="cross_entropy")
        os_fut = tc.optim_step(adam_params=adam)
        fb_out = fb.result()
        os_fut.result()
        budget.tick_step(); budget.tick_request(2)
        loss = _mean_loss(fb_out)
        if loss is not None:
            losses.append(loss)
        step += 1
        print(f"[train] step {step}/{args.max_steps} loss={loss} "
              f"elapsed={budget.elapsed()}s requests={budget.requests}", flush=True)

    (ARTIFACTS / "tinker_confirm_train.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "lora_rank": LORA_RANK, "steps": step,
        "batch": args.batch, "lr": args.lr, "elapsed_s": budget.elapsed(),
        "n_fit": len(fit), "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
    }, indent=2) + "\n")

    # ---- EVAL on sealed test (same process, ephemeral sampler) ----
    raw = SEALED.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(SEAL_MANIFEST.read_text())
    if digest != manifest["sha256"]:
        raise SystemExit("SEAL MISMATCH — refusing to score.")
    print(f"[seal] OK {digest[:16]}... n_cases={manifest['n_cases']}", flush=True)

    sampling = tc.save_weights_and_get_sampling_client()  # ephemeral, current weights
    print("[tinker] sampler ready (fine-tuned weights)", flush=True)

    cases = load_cases(SEALED)
    if args.n_test:
        cases = cases[: args.n_test]
    eval_budget = Budget(10**9, args.max_eval_seconds, args.max_requests)
    sp = tinker.SamplingParams(max_tokens=MAX_TARGET_TOKENS, temperature=0.0, top_p=1.0)

    pf = open(ARTIFACTS / "tinker_test_predictions.jsonl", "w")
    n_parsed = {"ok": 0, "fallback": 0}
    answers_by_id: dict[str, dict] = {}

    # Concurrent sampling: sample() returns a ConcurrentFuture, so we submit a wave of
    # requests, then collect them. This parallelises the slow per-request round-trips and
    # is robust to a single slow request (it doesn't block the whole eval). Cases whose
    # future errors or times out fall back to uniform so the scorer stays valid.
    def submit(case):
        mi = tinker.ModelInput.from_ints(
            tokens=tok.encode(build_prompt(case), add_special_tokens=False)[:MAX_PROMPT_TOKENS])
        return sampling.sample(prompt=mi, num_samples=1, sampling_params=sp)

    wave = max(1, args.concurrency)
    done = 0
    for start in range(0, len(cases), wave):
        if eval_budget.exceeded():
            print(f"[budget] eval stop at {done}/{len(cases)}: {eval_budget.exceeded()}", flush=True)
            break
        chunk = cases[start:start + wave]
        futures = [(c, submit(c)) for c in chunk]
        for c, fut in futures:
            gold = c["gold"]
            try:
                resp = fut.result(timeout=90)
                text = tok.decode(resp.sequences[0].tokens)
            except Exception as e:
                text = ""
                print(f"[warn] sample failed for {c.get('id')}: {e!r}", flush=True)
            eval_budget.tick_request()
            answers_by_id[c.get("id")] = _parse_answers(text, gold, tally=n_parsed)
            pf.write(json.dumps({"id": c.get("id"), "raw": text[:2000]}) + "\n")
            pf.flush()
        done += len(chunk)
        print(f"[eval] {done}/{len(cases)} elapsed={eval_budget.elapsed()}s "
              f"ok={n_parsed['ok']} fallback={n_parsed['fallback']}", flush=True)

    def predict(case: dict) -> dict:
        return answers_by_id.get(case.get("id")) or _parse_answers("", case["gold"])

    metrics = score_predictions(cases, predict)
    pf.close()
    metrics["n_cases"] = len(cases)
    metrics["model"] = f"{BASE_MODEL} (Tinker LoRA confirm)"
    metrics["train_steps"] = step
    metrics["eval_seconds"] = eval_budget.elapsed()
    metrics["parsed_ok"] = n_parsed["ok"]
    metrics["parsed_fallback"] = n_parsed["fallback"]
    print("[result]", json.dumps(metrics, indent=2), flush=True)
    (ARTIFACTS / "tinker_confirm_result.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return 0


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_qnode(text: str, qid: str) -> dict | None:
    """Pull one question's object out of (possibly truncated) JSON by locating
    '"qid": { ... "probabilities": { ... } }'. Robust to the overall JSON being cut off
    before its closing braces, as long as this question's probabilities block closed."""
    key = f'"{qid}"'
    i = text.find(key)
    if i < 0:
        return None
    j = text.find('"probabilities"', i)
    if j < 0:
        return None
    k = text.find("{", j)
    if k < 0:
        return None
    depth, end = 0, -1
    for idx in range(k, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                end = idx
                break
    if end < 0:
        return None
    try:
        return json.loads(text[k:end + 1])
    except Exception:
        return None


def _parse_answers(text: str, gold: dict, tally: dict | None = None) -> dict:
    """Parse the model's JSON into {qid: {probabilities}}. Tries whole-object parse
    first, then falls back to per-question extraction (robust to truncated output);
    uniform over the gold's own labels only if a question can't be recovered at all."""
    out = {}
    whole = {}
    m = _JSON_RE.search(text or "")
    if m:
        try:
            whole = json.loads(m.group(0))
        except Exception:
            whole = {}
    any_ok = False
    for qid, gq in gold.items():
        labels = list(gq.get("probabilities", {}).keys())
        raw = None
        node = whole.get(qid) if isinstance(whole, dict) else None
        if isinstance(node, dict) and isinstance(node.get("probabilities"), dict):
            raw = node["probabilities"]
        if raw is None:
            raw = _extract_qnode(text or "", qid)  # truncation-robust fallback
        p = {}
        if isinstance(raw, dict):
            p = {k: float(v) for k, v in raw.items()
                 if k in labels and isinstance(v, (int, float))}
        if p:
            any_ok = True
        else:
            p = {k: 1.0 / len(labels) for k in labels}
        out[qid] = {"probabilities": p}
    if tally is not None:
        tally["ok" if any_ok else "fallback"] += 1
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument("--max-seconds", type=int, default=1500,
                    help="hard wall-clock cap on TRAINING (seconds)")
    ap.add_argument("--max-eval-seconds", type=int, default=1500,
                    help="hard wall-clock cap on EVAL sampling (seconds)")
    ap.add_argument("--max-requests", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--n-fit", type=int, default=0)
    ap.add_argument("--n-test", type=int, default=400)
    ap.add_argument("--concurrency", type=int, default=16,
                    help="number of sampling requests in flight at once during eval")
    args = ap.parse_args()
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
