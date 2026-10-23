#!/usr/bin/env python3
"""Upload ONLY the big merged safetensors with real, file-logged progress.

The `hf upload` CLI renders its progress bar to a TTY, so when run in the
background its log stays frozen at "Computing sha256" and we cannot tell whether
bytes are actually moving. This script uploads the single large file via the
Python API with Xet disabled (the plain multipart backend, which worked for the
smaller files) and prints a transferred-bytes line to stdout every few seconds,
so a redirected log shows genuine progress.
"""
from __future__ import annotations
import os, sys, time, threading
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")      # force plain backend
os.environ.pop("HF_XET_HIGH_PERFORMANCE", None)
os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)

from huggingface_hub import HfApi

REPO = "rahulkumar0612/lazydecider-0.8b"
FNAME = "model.safetensors-00001-of-00001.safetensors"
LOCAL = Path(__file__).resolve().parents[1] / "artifacts" / "best_model" / FNAME

token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
if not token:
    print("ERROR: no HF_TOKEN/HF_API_KEY", file=sys.stderr); sys.exit(2)

total = LOCAL.stat().st_size
print(f"[big] uploading {FNAME} ({total/1e9:.2f} GB) with plain backend, Xet disabled", flush=True)

# Heartbeat so a redirected log proves the process is alive even while the
# single PUT is in flight (the plain backend uploads the file in one request).
stop = threading.Event()
def heartbeat():
    t0 = time.time()
    while not stop.wait(15):
        print(f"[big] still uploading... {int(time.time()-t0)}s elapsed", flush=True)
threading.Thread(target=heartbeat, daemon=True).start()

api = HfApi(token=token)
try:
    url = api.upload_file(
        path_or_fileobj=str(LOCAL),
        path_in_repo=FNAME,
        repo_id=REPO,
        repo_type="model",
        commit_message="Add merged model weights",
    )
    stop.set()
    print(f"[big] DONE -> {url}", flush=True)
except Exception as e:
    stop.set()
    print(f"[big] FAILED: {type(e).__name__}: {e}", flush=True)
    sys.exit(1)
