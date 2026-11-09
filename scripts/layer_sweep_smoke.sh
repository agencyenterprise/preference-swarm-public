#!/usr/bin/env bash
# Smoke for the layer sweep: Llama-3.2-1B, random stand-in scores, two blocks, four pairs. Numbers mean nothing;
# it proves the per-layer states, fitting and pushed calibration run end to end. ~2 min.
#   cd /workspace/preference-swarm && git pull && bash scripts/layer_sweep_smoke.sh
set -u
cd "$(dirname "$0")/.."
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
$PY scripts/layer_sweep.py --model meta-llama/Llama-3.2-1B-Instruct --exp50 random --layers 4,8 --sds 2,8 --pairs 4 \
  --out outputs/layer_sweep_smoke || { echo "[sweep-smoke] FAILED"; exit 1; }
$PY -c "
import json
d = json.load(open('outputs/layer_sweep_smoke/layer_sweep.json'))
assert [r['layer'] for r in d['rows']] == [4, 8], d['rows']
for r in d['rows']:
    for k in ('readout', 'tercile', 'random'):
        for s in (2.0, 8.0):
            assert r['calibration'][f'{k}@{s}sd']['n'] == 4, (k, s)
print('[sweep-smoke] ok: two blocks, three directions, two strengths')
" || { echo "[sweep-smoke] output check FAILED"; exit 1; }
