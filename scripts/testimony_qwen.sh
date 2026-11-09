#!/usr/bin/env bash
# Testimony vs insistence on Qwen2.5-7B (configs/v4_testimony_qwen7b.yaml). Safe to start while the Llama block-10 run
# is going: it waits until the GPU has at least MIN_FREE_GB free, then runs. ~20-30 min (60 questions, 7 conditions;
# Qwen ~1.6 s a deliberation alone, slower beside another run).
#   cd /workspace/preference-swarm && git pull && bash scripts/testimony_smoke.sh
#   cd /workspace/preference-swarm && git pull && bash scripts/testimony_qwen.sh
set -u
cd "$(dirname "$0")/.."
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
say () { echo "[$(date -u +%FT%TZ)] [testimony-qwen] $*"; }
MIN_FREE_GB=45
while true; do
  read -r FREE_MB < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "${FREE_MB:-0}" -ge $((MIN_FREE_GB * 1024)) ] && break
  say "GPU free ${FREE_MB} MiB < ${MIN_FREE_GB} GB; waiting 60 s"
  sleep 60
done
say "GPU free ${FREE_MB} MiB; starting"
$PY scripts/run_swarm.py --config configs/v4_testimony_qwen7b.yaml
st=$?
OUT="$(cat outputs/latest/v4_testimony_qwen7b.txt 2>/dev/null)"
say "run exited $st -> $OUT"
[ -n "$OUT" ] && { $PY scripts/check_run.py --run "$OUT" || say "check FAILED"; }
