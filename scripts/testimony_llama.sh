#!/usr/bin/env bash
# Testimony vs insistence on Llama-3.1-8B (configs/v4_testimony_llama8b_L10.yaml; block-10 data from llama_L10.sh,
# which must have rebuilt it). Safe to start now: it waits for the Llama block-10 run (llama_L10.sh) to finish, then
# runs. ~40 min (60 questions, 7 conditions, Llama ~3.1 s a deliberation).
#   cd /workspace/preference-swarm && git pull && bash scripts/testimony_smoke.sh
#   cd /workspace/preference-swarm && git pull && bash scripts/testimony_llama.sh
set -u
cd "$(dirname "$0")/.."
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
say () { echo "[$(date -u +%FT%TZ)] [testimony-llama] $*"; }
while pgrep -f "configs/v3_llama8b_L10.yaml" >/dev/null; do
  say "the Llama block-10 run is still going; waiting 60 s"
  sleep 60
done
[ -f data/exp50/Llama-3.1-8B-Instruct_L10/vectors.npz ] || { say "no block-10 data (run llama_L10.sh first)"; exit 1; }
say "starting"
$PY scripts/run_swarm.py --config configs/v4_testimony_llama8b_L10.yaml
st=$?
OUT="$(cat outputs/latest/v4_testimony_llama8b_L10.txt 2>/dev/null)"
say "run exited $st -> $OUT"
[ -n "$OUT" ] && { $PY scripts/check_run.py --run "$OUT" || say "check FAILED"; }
