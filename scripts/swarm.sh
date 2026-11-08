#!/usr/bin/env bash
# The full preference-swarm run on one GPU: experiment 1 (four models + paraphraser), then experiments 3, 4, 2, 5 on
# Llama-3.1-8B, then 4, 2, 5 on Qwen2.5-7B. ~10-12 h (estimate, unmeasured; the smoke prints per-experiment times).
# exp50's directions are rebuilt first if this box lacks them. A run that fails is reported and the next one still runs. SKIP="e1 llama8b qwen7b" skips any of the three.
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
# exp50's directions: copied from casper (prepare_data.sh) where casper's outputs exist, otherwise rebuilt from the
# saved tournament in data/exp50_src (~1 min a model plus the download; the axis directions are not rebuilt).
for SPEC in "meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct"; do
  M=$(basename "$SPEC")
  if [ -f "data/exp50/$M/vectors.npz" ] && [ -f "data/exp50/$M/states.npz" ]; then say "exp50 $M present"; continue; fi
  case "$M" in Qwen*) case " ${SKIP:-} " in *" qwen7b "*) continue;; esac;; esac
  say "exp50 $M missing: rebuilding"
  $PY scripts/rebuild_exp50.py --model "$SPEC" || say "exp50 $M rebuild FAILED"
done
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
