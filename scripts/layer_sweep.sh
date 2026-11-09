#!/usr/bin/env bash
# Layer sweep (2026-11-08): at which block does a preference direction fitted at that block also steer the choice?
# Llama-3.1-8B at every second block (2..30), then Qwen2.5-7B at 8/12/16/20/24 as the positive control (its block-16
# direction steers). 100 request pairs, pushes of 2/4/8/16 s.d. along readout / tercile / random. ~20 min (estimate
# from the swarm's calibration stage: 0.3 min for 1,800 choices).
#   cd /workspace/preference-swarm && git pull && bash scripts/layer_sweep_smoke.sh
#   cd /workspace/preference-swarm && git pull && bash scripts/layer_sweep.sh
set -u
cd "$(dirname "$0")/.."
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
say () { echo "[$(date -u +%FT%TZ)] [layer-sweep] $*"; }
say "Llama-3.1-8B"
$PY scripts/layer_sweep.py --model meta-llama/Llama-3.1-8B-Instruct --exp50 data/exp50/Llama-3.1-8B-Instruct || say "Llama FAILED"
say "Qwen2.5-7B (positive control)"
$PY scripts/layer_sweep.py --model Qwen/Qwen2.5-7B-Instruct --exp50 data/exp50/Qwen2.5-7B-Instruct --layers 8,12,16,20,24 || say "Qwen FAILED"
say "done"
