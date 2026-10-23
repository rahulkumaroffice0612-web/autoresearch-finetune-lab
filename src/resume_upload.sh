#!/usr/bin/env bash
# Resume the HuggingFace upload of lazydecider-0.8b.
# Safe to re-run any time: `hf upload` + Xet skip already-transferred blobs,
# so it picks up where the last run left off. Run from the lazylab/ dir.
set -euo pipefail

cd "$(dirname "$0")/.."                      # -> lazylab/
source .venv/bin/activate
set -a; source ../.env; set +a
export HF_TOKEN="${HF_TOKEN:-$HF_API_KEY}"   # token lives as HF_API_KEY in .env
export HF_XET_HIGH_PERFORMANCE=1             # fast Xet transfer backend
unset HF_HUB_ENABLE_HF_TRANSFER || true      # deprecated; Xet supersedes it

REPO="rahulkumar0612/lazydecider-0.8b"

echo "[resume] whoami:"
hf auth whoami || true

# The uplink drops intermittently and `hf upload` HANGS (doesn't exit) rather
# than erroring. So we run it in the background, watch its progress log, and if
# no bytes move for $STALL_SECS we kill it and retry. Xet skips already-committed
# chunks, so each retry resumes prior progress.
LOG="artifacts/.hf_upload_live.log"
STALL_SECS=90

run_once() {
  : > "$LOG"
  hf upload "$REPO" artifacts/best_model . \
    --repo-type model \
    --exclude ".eval_results/*" \
    --commit-message "Add lazydecider-0.8b: Qwen3.5-0.8B + Clef, fine-tuned on typed-decisions" \
    >> "$LOG" 2>&1 &
  local pid=$!
  local last_size=0 stall=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    local size; size=$(wc -c < "$LOG" 2>/dev/null || echo 0)
    if [ "$size" -eq "$last_size" ]; then
      stall=$((stall + 15))
      if [ "$stall" -ge "$STALL_SECS" ]; then
        echo "[resume] no progress for ${STALL_SECS}s — killing hung upload (pid $pid)."
        kill "$pid" 2>/dev/null; pkill -f "hf upload" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
        return 1
      fi
    else
      stall=0; last_size=$size
    fi
  done
  wait "$pid"   # propagate the upload's real exit code
}

echo "[resume] uploading model files (excluding .eval_results) with stall-watchdog..."
attempt=0
until run_once; do
  attempt=$((attempt + 1))
  echo "[resume] attempt $attempt interrupted/stalled. Retrying in 10s..."
  sleep 10
done
echo "[resume] model files committed after $((attempt + 1)) attempt(s)."

echo "[resume] model files done. Now opening the leaderboard PR with the eval YAML..."
hf upload "$REPO" \
  artifacts/best_model/.eval_results/rahulkumar0612__lazydecider-0.8b.yaml \
  .eval_results/rahulkumar0612__lazydecider-0.8b.yaml \
  --repo-type model \
  --create-pr \
  --commit-message "Add eval results for LocalLLaMA/typed-decisions (specialist)"

echo "[resume] DONE. Model: https://huggingface.co/$REPO"
echo "[resume] Merge the open PR to publish on the leaderboard."
