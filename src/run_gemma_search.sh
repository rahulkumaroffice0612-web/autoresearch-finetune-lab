#!/usr/bin/env bash
# Run a short autonomous search driven by LOCAL Gemma 4 E4B (LM Studio).
# Prereqs (checked below): LM Studio server up on :1234 with gemma-4-e4b loaded.
#
# This produces real experiment-log entries tagged source="gemma" with Gemma's
# one-line reasons — evidence for the "Best Use of Gemma" claim (the agent that
# proposes the hyperparameter configs).
#
# Usage: bash src/run_gemma_search.sh [MINUTES]
set -euo pipefail

cd "$(dirname "$0")/.."                       # -> lazylab/
source .venv/bin/activate

MINUTES="${1:-20}"

# Point the agent at LM Studio / Bionic (NOT the LiteLLM gateway). Override explicitly,
# because agent.py falls back to LITELLM_BASE_URL when AGENT_BASE_URL is unset.
# Bionic runs the local server on :1235 (the old LM Studio held :1234).
export AGENT_BASE_URL="${AGENT_BASE_URL:-http://127.0.0.1:1235/v1}"
export AGENT_API_KEY=""                       # LM Studio needs no key
unset LITELLM_BASE_URL LITELLM_API_KEY || true
# GEMMA_MODEL must match the model id the server reports at /v1/models.
export GEMMA_MODEL="${GEMMA_MODEL:-gemma-4-e4b-it-mlx}"

echo "[gemma-run] endpoint: $AGENT_BASE_URL  model: $GEMMA_MODEL  budget: ${MINUTES}min"
echo "[gemma-run] models visible on LM Studio:"
curl -sS --max-time 10 "$AGENT_BASE_URL/models" | python -c "import sys,json; [print('  -', m['id']) for m in json.load(sys.stdin).get('data',[])]" || {
  echo "ERROR: local server not reachable on $AGENT_BASE_URL. Start it: lms server start && lms load gemma-4-e4b-it-mlx"; exit 1; }

# Separate log so Gemma's provenance is clean and not mixed with the gemini dev runs.
# worker.py writes to experiments_arm<ARM>.jsonl; use arm B as the Gemma-provenance log.
echo "[gemma-run] launching worker (arm B = Gemma-provenance log)..."
python -u src/worker.py --minutes "$MINUTES" --arm B
