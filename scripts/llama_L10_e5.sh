#!/usr/bin/env bash
# e5 (the monitor) on the newest Llama block-10 full run, on whatever it has saved so far: the jury data, and the e4
# chunks finished up to now. Safe while llama_L10.sh is still in e4 (it reads the saved files; llama_L10.sh runs e5
# again at its end). ~5 min, most of it loading the model.
#   cd /workspace/preference-swarm && git pull && bash scripts/llama_L10_e5_smoke.sh
#   cd /workspace/preference-swarm && git pull && bash scripts/llama_L10_e5.sh
set -u
cd "$(dirname "$0")/.."
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
say () { echo "[$(date -u +%FT%TZ)] [L10-e5] $*"; }
shopt -s nullglob
RUNS=(outputs/v3_llama8b_L10_2*)
[ ${#RUNS[@]} -gt 0 ] || { say "no full run found (outputs/v3_llama8b_L10_2*)"; exit 1; }
OUT="${RUNS[${#RUNS[@]}-1]}"
say "e5 on $OUT"
$PY scripts/run_swarm.py --config configs/v3_llama8b_L10.yaml --out "$OUT" --exp e5 || { say "e5 FAILED"; exit 1; }
$PY scripts/check_run.py --run "$OUT" || say "check reports problems (expected while e4 is unfinished)"
say "done: $OUT/e5_monitor/summary.json"
