#!/usr/bin/env bash
# Pre-run checks for scripts/swarm.sh: unit tests, the exp50 rebuild on a small model, then every experiment end to end at toy size
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
say "exp50 rebuild (Llama-3.2-1B on 80 requests; swarm.sh rebuilds the real ones if they are missing)"
$PY scripts/rebuild_exp50.py --model meta-llama/Llama-3.2-1B-Instruct --src-name Llama-3.1-8B-Instruct \
  --layer 8 --out outputs/smoke_rebuild --n 80 || { say "rebuild FAILED"; exit 1; }
$PY -c "
from preference_swarm import data as DT
e = DT.load_exp50('outputs/smoke_rebuild')
assert e.V.shape[0] == 7 and 'general_readout' in e.names, e.names
print('[pre-run] rebuilt exp50 loads:', e.model, e.layer, e.V.shape)
" || { say "rebuilt exp50 does not load"; exit 1; }
for M in Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct; do
  [ -f "data/exp50/$M/vectors.npz" ] && say "exp50 $M present" || say "exp50 $M absent: swarm.sh will rebuild it"
done
say "toy pipeline (all five experiments)"
$PY scripts/run_swarm.py --config configs/smoke.yaml || { say "smoke run FAILED (see outputs/smoke_*/telemetry.json)"; exit 1; }
OUT="$(cat outputs/latest/smoke.txt)"
$PY scripts/check_run.py --run "$OUT" --strict || { say "check FAILED ($OUT)"; exit 1; }
say "all pre-run checks passed ($OUT)"
