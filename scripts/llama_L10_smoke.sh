#!/usr/bin/env bash
# Smoke for the Llama block-10 rerun: rebuild directions at block 10 from 80 requests, then every experiment at toy size
# on the real model and block, then the output check. ~5 min (most of it loading the 8B model twice).
#   cd /workspace/preference-swarm && git pull && bash scripts/llama_L10_smoke.sh
set -u
cd "$(dirname "$0")/.."
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
say () { echo "[$(date -u +%FT%TZ)] [L10-smoke] $*"; }
say "rebuild at block 10 (80 requests)"
$PY scripts/rebuild_exp50.py --model meta-llama/Llama-3.1-8B-Instruct --layer 10 --n 80 --out outputs/smoke_L10_exp50 \
  || { say "rebuild FAILED"; exit 1; }
$PY -c "
from preference_swarm import data as DT
e = DT.load_exp50('outputs/smoke_L10_exp50')
assert e.layer == 10 and e.model == 'meta-llama/Llama-3.1-8B-Instruct', (e.model, e.layer)
print('[L10-smoke] directions load at block', e.layer)
" || { say "rebuilt data wrong"; exit 1; }
say "toy run of e2, e3, e4, e5 at block 10"
$PY scripts/run_swarm.py --config configs/v3_llama8b_L10_smoke.yaml || { say "run FAILED"; exit 1; }
OUT="$(cat outputs/latest/v3_llama8b_L10_smoke.txt)"
$PY scripts/check_run.py --run "$OUT" --strict || { say "check FAILED ($OUT)"; exit 1; }
say "passed ($OUT)"
