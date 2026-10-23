"""ONE-SHOT sealed test read. Score artifacts/best_model/ on data/sealed/test.jsonl
with the OFFICIAL scorer (scorer.py), the same arithmetic as bench/score_with_jobs.py.

This is the real leaderboard number. It reads the sealed test set exactly once;
predictions are cached to artifacts/test_predictions.jsonl so re-scoring never needs
another read. Verifies the sealed SHA-256 before reading (tamper check).

Usage:
  python src/score_test.py --model artifacts/best_model
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEALED = ROOT / "data" / "sealed" / "test.jsonl"
MANIFEST = ROOT / "data" / "sealed" / "test.sha256"
ARTIFACTS = ROOT / "artifacts"


def verify_seal() -> list[dict]:
    """Check the test file matches the sealed hash, then load + parse it."""
    raw = SEALED.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(MANIFEST.read_text())
    if digest != manifest["sha256"]:
        raise SystemExit(
            f"SEAL MISMATCH: test.jsonl sha256={digest} != manifest {manifest['sha256']}. "
            "The sealed test set was modified; refusing to score."
        )
    print(f"[seal] OK sha256={digest[:16]}... n_cases={manifest['n_cases']}", flush=True)

    cases = []
    for line in raw.decode().splitlines():
        line = line.strip()
        if not line:
            continue
        c = json.loads(line)
        # fields arrive as JSON strings in this dataset; parse the two we need.
        for k in ("gold", "questions", "state", "factors"):
            if isinstance(c.get(k), str):
                c[k] = json.loads(c[k])
        cases.append(c)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ARTIFACTS / "best_model"))
    ap.add_argument("--max-seq-length", type=int, default=1280)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from scorer import score_predictions

    cases = verify_seal()

    from unsloth import FastDecisionModel
    print(f"[load] {args.model}", flush=True)
    model, tok = FastDecisionModel.from_pretrained(
        args.model, max_seq_length=args.max_seq_length, load_in_4bit=False,
    )
    print("[load] done", flush=True)

    # Cache raw predictions so we never need to touch the sealed set again.
    pred_path = ARTIFACTS / "test_predictions.jsonl"
    pred_f = open(pred_path, "w")

    def predict(case: dict) -> dict:
        answers = FastDecisionModel.predict(
            model, tok, case["state"], case["questions"],
        )
        # Keep only what the scorer needs ({qid: {probabilities}}), and cache.
        slim = {qid: {"probabilities": a.get("probabilities", {}),
                      "type": a.get("type")} for qid, a in answers.items()}
        pred_f.write(json.dumps({"id": case.get("id"), "answers": slim}) + "\n")
        pred_f.flush()
        return slim

    done = [0]
    _orig = predict

    def predict_logged(case):
        r = _orig(case)
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"[predict] {done[0]}/{len(cases)} cases", flush=True)
        return r

    result = score_predictions(cases, predict_logged)
    pred_f.close()

    print("\n=== OFFICIAL TEST SCORE (best_model) ===", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    (ARTIFACTS / "test_score.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n[saved] {ARTIFACTS / 'test_score.json'} and {pred_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
