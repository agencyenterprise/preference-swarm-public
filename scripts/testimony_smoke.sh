#!/usr/bin/env bash
# Pre-run checks for testimony_qwen.sh / testimony_llama.sh: unit tests, then the testimony e4 at toy size
# (Llama-3.2-1B, random directions) and the output check. ~3 min.
#   cd /workspace/preference-swarm && git pull && bash scripts/testimony_smoke.sh
set -u
cd "$(dirname "$0")/.."
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
say () { echo "[$(date -u +%FT%TZ)] [testimony-smoke] $*"; }
$PY -m pytest tests -q || { say "unit tests FAILED"; exit 1; }
$PY scripts/run_swarm.py --config configs/v4_testimony_smoke.yaml || { say "smoke run FAILED"; exit 1; }
OUT="$(cat outputs/latest/v4_testimony_smoke.txt)"
$PY scripts/check_run.py --run "$OUT" --strict || { say "check FAILED ($OUT)"; exit 1; }
say "passed ($OUT)"
