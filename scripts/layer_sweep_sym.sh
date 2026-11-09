#!/usr/bin/env bash
# Symmetric layer sweep (2026-11-08): every direction pushed +v and -v at 8 and 16 s.d.; preference part (flips with
# the sign) vs push part (does not); 8 random directions as the null. Llama-3.1-8B at blocks 4-16, Qwen2.5-7B at
# 8/12/16. ~20-25 min (estimate from the first sweep's speed, unmeasured).
#   cd /workspace/preference-swarm && git pull && bash scripts/layer_sweep_sym_smoke.sh
#   cd /workspace/preference-swarm && git pull && bash scripts/layer_sweep_sym.sh
set -u
cd "$(dirname "$0")/.."
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
say () { echo "[$(date -u +%FT%TZ)] [sym-sweep] $*"; }
say "Llama-3.1-8B"
$PY scripts/layer_sweep.py --model meta-llama/Llama-3.1-8B-Instruct --exp50 data/exp50/Llama-3.1-8B-Instruct \
  --layers 4,6,8,10,12,14,16 --sds 8,16 --symmetric || say "Llama FAILED"
say "Qwen2.5-7B (control)"
$PY scripts/layer_sweep.py --model Qwen/Qwen2.5-7B-Instruct --exp50 data/exp50/Qwen2.5-7B-Instruct \
  --layers 8,12,16 --sds 8,16 --symmetric || say "Qwen FAILED"
say "done"
