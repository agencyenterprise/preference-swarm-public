#!/usr/bin/env bash
# Second run (2026-11-07 redesign): e3/e4/e5 with forced-stance advocates, the distraction control, planet-database
# and contested cases, misinformed and steered minorities. Qwen2.5-7B first (steering works there), then Llama-3.1-8B
# (non-steering conditions only). ~2.5 h (estimate from the first run's timings, unmeasured at this design).
# SKIP="v2_qwen7b v2_llama8b" skips either. The exp50 data must be in data/exp50 (prepare_data.sh or swarm.sh's rebuild).
#   cd /workspace/preference-swarm && git pull && bash scripts/swarm_smoke.sh
#   cd /workspace/preference-swarm && git pull && bash scripts/swarm_v2.sh
set -u
cd "$(dirname "$0")/.."
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
say () { echo "[$(date -u +%FT%TZ)] [swarm-v2] $*"; }
for SPEC in "meta-llama/Llama-3.1-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct"; do
  M=$(basename "$SPEC")
  if [ -f "data/exp50/$M/vectors.npz" ] && [ -f "data/exp50/$M/states.npz" ]; then continue; fi
  say "exp50 $M missing: rebuilding"
  $PY scripts/rebuild_exp50.py --model "$SPEC" || say "exp50 $M rebuild FAILED"
done
for R in v2_qwen7b v2_llama8b; do
  case " ${SKIP:-} " in *" $R "*) say "$R skipped"; continue;; esac
  say "$R starting"
  $PY scripts/run_swarm.py --config "configs/$R.yaml"
  st=$?
  OUT="$(cat "outputs/latest/$R.txt" 2>/dev/null)"
  say "$R exited $st -> $OUT"
  [ -n "$OUT" ] && { $PY scripts/check_run.py --run "$OUT" || say "$R: check FAILED"; }
done
say "all done"
