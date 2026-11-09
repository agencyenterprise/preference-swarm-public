#!/usr/bin/env bash
# Pre-run checks for testimony.sh: unit tests; the testimony e4 at toy size (Llama-3.2-1B, random directions); then the
# REAL Qwen2.5-7B on 2 questions at full batch sizes BESIDE the live Llama block-10 run, to show the two fit on the GPU
# together (prints this run's peak GPU memory and the card's free memory after). ~5 min.
#   cd /workspace/preference-swarm && git pull && bash scripts/testimony_smoke.sh
set -u
cd "$(dirname "$0")/.."
say () { echo "[$(date -u +%FT%TZ)] [testimony-smoke] $*"; }
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
$PY -m pytest tests -q || { say "unit tests FAILED"; exit 1; }
say "toy run (1B)"
$PY scripts/run_swarm.py --config configs/v4_testimony_smoke.yaml || { say "toy run FAILED"; exit 1; }
OUT="$(cat outputs/latest/v4_testimony_smoke.txt)"
$PY scripts/check_run.py --run "$OUT" --strict || { say "toy check FAILED ($OUT)"; exit 1; }
say "real Qwen beside the Llama run"
wait_for_gpu
$PY scripts/run_swarm.py --config configs/v4_testimony_qwen7b_smoke.yaml || { say "Qwen concurrency run FAILED (out of memory?)"; exit 1; }
OUT="$(cat outputs/latest/v4_testimony_qwen7b_smoke.txt)"
$PY scripts/check_run.py --run "$OUT" || { say "Qwen check FAILED ($OUT)"; exit 1; }
read -r FREE_MB < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
say "passed; GPU free now ${FREE_MB} MiB ($OUT)"
