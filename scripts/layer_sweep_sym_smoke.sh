#!/usr/bin/env bash
# Smoke for the symmetric layer sweep: Llama-3.2-1B, random stand-in scores, one block, two random directions, four
# pairs, pushes +v and -v. Numbers mean nothing; it proves the +/- split and the random null run. ~1 min.
#   cd /workspace/preference-swarm && git pull && bash scripts/layer_sweep_sym_smoke.sh
set -u
cd "$(dirname "$0")/.."
PY=python
for P in poetry "$HOME/.local/bin/poetry"; do if command -v "$P" >/dev/null 2>&1; then PY="$P run python"; break; fi; done
if [ -z "${HF_HOME:-}" ] && [ -d /workspace/.cache/huggingface ]; then export HF_HOME=/workspace/.cache/huggingface; fi
$PY scripts/layer_sweep.py --model meta-llama/Llama-3.2-1B-Instruct --exp50 random --layers 8 --sds 8 --pairs 4 \
  --symmetric --n-random 2 --out outputs/layer_sweep_sym_smoke || { echo "[sym-smoke] FAILED"; exit 1; }
$PY -c "
import json
d = json.load(open('outputs/layer_sweep_sym_smoke/layer_sweep.json'))
r = d['rows'][0]
assert d['symmetric'] and r['layer'] == 8
for k in ('readout', 'tercile', 'random0', 'random1'):
    c = r['calibration'][f'{k}@8.0sd']
    assert c['pref']['n'] == 4 and c['push']['n'] == 4, k
    assert abs(c['pref']['mean'] - (c['plus']['mean'] - c['minus']['mean']) / 2) < 1e-9, k
assert r['proj_sd']['readout'] > 0, 'readout direction has no spread'
assert 'readout@8.0sd' in r['null']
print('[sym-smoke] ok: +/- pushes, preference and push parts, random null')
" || { echo "[sym-smoke] output check FAILED"; exit 1; }
