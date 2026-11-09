#!/usr/bin/env bash
# Testimony vs insistence, both models: Qwen2.5-7B as soon as the GPU has room (it can run beside the Llama block-10
# job; ~25-30 min), then -- once the Llama block-10 run (llama_L10.sh) has finished -- Llama-3.1-8B at block 10
# (~40 min). e4 in chunks of 5 questions covering every condition, saved as each finishes.
#   cd /workspace/preference-swarm && git pull && bash scripts/testimony_smoke.sh
#   cd /workspace/preference-swarm && git pull && bash scripts/testimony.sh
set -u
cd "$(dirname "$0")/.."
say () { echo "[$(date -u +%FT%TZ)] [testimony] $*"; }
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
MIN_FREE_GB=45
wait_for_gpu () {
  while true; do
    read -r FREE_MB < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "${FREE_MB:-0}" -ge $((MIN_FREE_GB * 1024)) ] && break
    say "GPU free ${FREE_MB} MiB < ${MIN_FREE_GB} GB; waiting 60 s"
    sleep 60
  done
  say "GPU free ${FREE_MB} MiB"
}
run () {
  $PY scripts/run_swarm.py --config "configs/$1.yaml"
  local st=$?
  local out="$(cat "outputs/latest/$1.txt" 2>/dev/null)"
  say "$1 exited $st -> $out"
  [ -n "$out" ] && { $PY scripts/check_run.py --run "$out" || say "$1: check FAILED"; }
}
say "Qwen2.5-7B"
wait_for_gpu
run v4_testimony_qwen7b
while pgrep -f "configs/v3_llama8b_L10.yaml" >/dev/null; do
  say "the Llama block-10 run is still going; waiting 60 s"
  sleep 60
done
[ -f data/exp50/Llama-3.1-8B-Instruct_L10/vectors.npz ] || { say "no block-10 data (run llama_L10.sh first)"; exit 1; }
say "Llama-3.1-8B at block 10"
run v4_testimony_llama8b_L10
say "done"
