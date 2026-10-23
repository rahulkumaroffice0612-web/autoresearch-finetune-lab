#!/usr/bin/env bash
# Launch the LazyDecider interactive playground (local Gradio, real model).
#
# Usage:
#   bash src/run_playground.sh            # http://127.0.0.1:7860
#   bash src/run_playground.sh --share     # also a temporary public gradio.live link
#   PORT=7862 bash src/run_playground.sh   # custom port
#
# On Apple Silicon this loads the MLX build in artifacts/best_model (load ~15-20s).
set -euo pipefail

cd "$(dirname "$0")/.."                       # -> lazylab/
source .venv/bin/activate

# gradio is the only extra dep beyond the training env.
python -c "import gradio" 2>/dev/null || {
  echo "[playground] installing gradio..."; uv pip install "gradio>=5,<6" >/dev/null; }

PORT="${PORT:-7860}"
echo "[playground] loading model + starting server on :$PORT ..."
echo "[playground] the model preloads at startup (~15-20s); wait for 'model ready', then open http://127.0.0.1:$PORT"
exec python -u src/playground.py --port "$PORT" "$@"
