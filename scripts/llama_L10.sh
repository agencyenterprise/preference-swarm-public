#!/usr/bin/env bash
# Llama-3.1-8B steered at decoder block 10 (where the layer sweep found its preference direction steers): rebuild the
# exp50 directions at block 10 on all 1,500 requests (~2 min), then e2 (~35 min), e3 (~20 min), e5, and e4 LAST in
# 12 chunks of 5 questions (~16 min each, ~3.3 h; each chunk saved, balanced over kinds and conditions), then e5.
# ~4.5 h in all (estimate from run-2 speeds: e4 ~3 s a deliberation on Llama). Stop any time; the partial e4 is usable.
# Run again after a stop and it continues: if the newest outputs/v3_llama8b_L10_* run already has e3's jury data, e4
# resumes in that folder (saved questions are skipped) and e5 reruns; otherwise a new run starts.
#   cd /workspace/preference-swarm && git pull && bash scripts/llama_L10_smoke.sh
#   cd /workspace/preference-swarm && git pull && bash scripts/llama_L10.sh
set -u
cd "$(dirname "$0")/.."
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
say () { echo "[$(date -u +%FT%TZ)] [L10] $*"; }
D=data/exp50/Llama-3.1-8B-Instruct_L10
if [ -f "$D/vectors.npz" ] && [ -f "$D/states.npz" ]; then say "block-10 directions present"; else
  say "rebuilding directions at block 10"
  $PY scripts/rebuild_exp50.py --model meta-llama/Llama-3.1-8B-Instruct --src-name Llama-3.1-8B-Instruct --layer 10 --out "$D" \
    || { say "rebuild FAILED"; exit 1; }
fi
shopt -s nullglob
RUNS=(outputs/v3_llama8b_L10_2*)              # full runs only (smokes are v3_llama8b_L10_smoke_*); glob order = time order
PREV=""
[ ${#RUNS[@]} -gt 0 ] && PREV="${RUNS[${#RUNS[@]}-1]}"
if [ -n "$PREV" ] && [ -f "$PREV/e3_jury/jury.jsonl" ]; then
  say "continuing $PREV: e4 skips its saved questions, then e5"
  $PY scripts/run_swarm.py --config configs/v3_llama8b_L10.yaml --out "$PREV" --exp e4,e5
else
  $PY scripts/run_swarm.py --config configs/v3_llama8b_L10.yaml
fi
st=$?
OUT="$(cat outputs/latest/v3_llama8b_L10.txt 2>/dev/null)"
say "run exited $st -> $OUT"
[ -n "$OUT" ] && { $PY scripts/check_run.py --run "$OUT" || say "check FAILED"; }
say "done"
