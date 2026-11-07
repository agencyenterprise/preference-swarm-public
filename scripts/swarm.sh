#!/usr/bin/env bash
# The full preference-swarm run on one GPU: experiment 1 (four models + paraphraser), then experiments 3, 4, 2, 5 on
# Llama-3.1-8B, then 4, 2, 5 on Qwen2.5-7B. ~10-12 h (estimate, unmeasured; the smoke prints per-experiment times).
# A run that fails is reported and the next one still runs. SKIP="e1 llama8b qwen7b" skips any of the three.
#   cd /workspace/preference-swarm && git pull && bash scripts/swarm_smoke.sh
#   cd /workspace/preference-swarm && git pull && bash scripts/swarm.sh
set -u
cd "$(dirname "$0")/.."
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
# The box keeps its Hugging Face cache on the volume; use it unless HF_HOME is already set.
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
say () { echo "[$(date -u +%FT%TZ)] [swarm] $*"; }
for R in e1 llama8b qwen7b; do
  case " ${SKIP:-} " in *" $R "*) say "$R skipped"; continue;; esac
  say "$R starting"
  $PY scripts/run_swarm.py --config "configs/$R.yaml"
  st=$?
  OUT="$(cat "outputs/latest/$R.txt" 2>/dev/null)"
  say "$R exited $st -> $OUT"
  [ -n "$OUT" ] && { $PY scripts/check_run.py --run "$OUT" || say "$R: check FAILED"; }
done
say "all done"
