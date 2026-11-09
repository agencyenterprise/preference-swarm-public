#!/usr/bin/env bash
# Pre-run check for llama_L10_e5.sh: unit tests, then e5 on the newest Llama block-10 SMOKE run. ~3 min.
#   cd /workspace/preference-swarm && git pull && bash scripts/llama_L10_e5_smoke.sh
set -u
cd "$(dirname "$0")/.."
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
say () { echo "[$(date -u +%FT%TZ)] [L10-e5-smoke] $*"; }
$PY -m pytest tests -q || { say "unit tests FAILED"; exit 1; }
shopt -s nullglob
RUNS=(outputs/v3_llama8b_L10_smoke_*)
[ ${#RUNS[@]} -gt 0 ] || { say "no smoke run found: run bash scripts/llama_L10_smoke.sh first"; exit 1; }
OUT="${RUNS[${#RUNS[@]}-1]}"
say "e5 on $OUT"
$PY scripts/run_swarm.py --config configs/v3_llama8b_L10_smoke.yaml --out "$OUT" --exp e5 || { say "e5 FAILED"; exit 1; }
$PY scripts/check_run.py --run "$OUT" || { say "check FAILED"; exit 1; }
say "passed"
