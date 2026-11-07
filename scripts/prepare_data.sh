#!/usr/bin/env bash
# Copy the exp50 inputs from casper's outputs into data/exp50/ (run once on the box). Reads casper, never writes it.
#   CASPER_OUTPUTS=/workspace/casper/outputs bash scripts/prepare_data.sh
set -u
cd "$(dirname "$0")/.."
SRC="${CASPER_OUTPUTS:-/workspace/casper/outputs}/wellbeing_found/exp50_general"
say () { echo "[prepare] $*"; }
if [ ! -f "$SRC/data/prompts.jsonl" ]; then say "no $SRC/data/prompts.jsonl (is casper's exp50 output on this box?)"; exit 1; fi
status=0
for M in Llama-3.1-8B-Instruct Qwen2.5-7B-Instruct; do
  mkdir -p "data/exp50/$M"
  cp "$SRC/data/prompts.jsonl" "data/exp50/$M/prompts.jsonl"
  for f in judge_rows.json general_pref.json vectors.npz; do
    if [ -f "$SRC/$M/$f" ]; then cp "$SRC/$M/$f" "data/exp50/$M/$f"; else say "MISSING $SRC/$M/$f"; status=1; fi
  done
  say "$M: $(ls "data/exp50/$M" | tr '\n' ' ')"
done
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
$PY -c "
import sys
sys.path.insert(0, 'src')
from preference_swarm import data as DT
for m in ('Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct'):
    e = DT.load_exp50('data/exp50/' + m)
    print(f'[prepare] {m}: model {e.model}, layer {e.layer}, {len(e.texts)} requests, {len(e.score)} scored, '
          f'{len(DT.usable_names(e))} usable, {len(e.names)} directions of width {e.V.shape[1]}')
" || status=1
exit $status
