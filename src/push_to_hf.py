#!/usr/bin/env python3
"""Push lazydecider-0.8b to the Hugging Face Hub.

Steps:
  1. Upload the merged model folder (artifacts/best_model) to <user>/lazydecider-0.8b.
     The .eval_results/ YAML inside that folder is uploaded as a PR (create_pr=True),
     so it lands on the official Hub leaderboard only once merged.

Auth: set HF_TOKEN in the environment (or run `hf auth login`) before running.

Usage:
  python src/push_to_hf.py --dry-run     # validate, do not upload
  python src/push_to_hf.py               # actually push
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, upload_folder

REPO = "rahulkumar0612/lazydecider-0.8b"
MODEL_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "best_model"
EVAL_YAML = ".eval_results/rahulkumar0612__lazydecider-0.8b.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="validate only, no upload")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--private", action="store_true", help="create the repo as private")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN in the environment (or run `hf auth login`).", file=sys.stderr)
        return 2

    if not MODEL_DIR.is_dir():
        print(f"ERROR: model dir not found: {MODEL_DIR}", file=sys.stderr)
        return 2

    api = HfApi(token=token)
    if not args.dry_run:
        who = api.whoami()
        print(f"[push] authenticated as: {who.get('name')}")

    files = sorted(p.name for p in MODEL_DIR.iterdir())
    print(f"[push] repo         : {args.repo}")
    print(f"[push] model dir    : {MODEL_DIR}")
    print(f"[push] files        : {len(files)}")
    for f in files:
        print(f"         - {f}")
    print(f"[push] eval yaml    : {EVAL_YAML} (uploaded as a PR)")

    if args.dry_run:
        print("[push] DRY RUN — nothing uploaded.")
        return 0

    api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
    print("[push] repo ready. Uploading model weights (everything except .eval_results)...")
    upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=args.repo,
        repo_type="model",
        token=token,
        ignore_patterns=[".eval_results/*"],
        commit_message="Add lazydecider-0.8b: Qwen3.5-0.8B + Clef, fine-tuned on typed-decisions",
    )
    print("[push] model uploaded. Opening a PR with the .eval_results leaderboard YAML...")
    api.upload_file(
        path_or_fileobj=str(MODEL_DIR / EVAL_YAML),
        path_in_repo=EVAL_YAML,
        repo_id=args.repo,
        repo_type="model",
        token=token,
        create_pr=True,
        commit_message="Add eval results for LocalLLaMA/typed-decisions (specialist)",
    )
    print(f"[push] done. Model: https://huggingface.co/{args.repo}")
    print(f"[push] merge the open PR to publish on the leaderboard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
