#!/usr/bin/env bash
# Pre-run checks for scripts/swarm.sh: unit tests, the exp50 data check, then every experiment end to end at toy size
# (Llama-3.2-1B, random stand-in directions; e1 with three small models), then the output check. ~10 min (estimate).
#   cd /workspace/preference-swarm && git pull && bash scripts/swarm_smoke.sh
set -u
cd "$(dirname "$0")/.."
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
# The box keeps its Hugging Face cache on the volume; use it unless HF_HOME is already set.
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
say () { echo "[$(date -u +%FT%TZ)] [pre-run] $*"; }
say "unit tests"
$PY -m pytest tests -q -rs || { say "unit tests FAILED"; exit 1; }
say "exp50 data"
for M in Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct; do
  for f in prompts.jsonl judge_rows.json general_pref.json vectors.npz states.npz; do
    [ -f "data/exp50/$M/$f" ] || { say "data/exp50/$M/$f missing: run bash scripts/prepare_data.sh"; exit 1; }
  done
done
say "toy pipeline (all five experiments)"
$PY scripts/run_swarm.py --config configs/smoke.yaml || { say "smoke run FAILED (see outputs/smoke_*/telemetry.json)"; exit 1; }
OUT="$(cat outputs/latest/smoke.txt)"
$PY scripts/check_run.py --run "$OUT" --strict || { say "check FAILED ($OUT)"; exit 1; }
say "all pre-run checks passed ($OUT)"
