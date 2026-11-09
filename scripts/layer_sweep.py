#!/usr/bin/env python
"""Layer sweep: at which decoder block, if any, does a preference direction also STEER the model's choice?

Built 2026-11-08 after the swarm found that Llama-3.1-8B's exp50 preference direction (block 16) reads preference
(R2 0.75) but moves the model's choice no more than a random push (calibration: +0.084 vs +0.085 at 1x), matching
casper exp51 (one-copy pushes ~0; choice gradient orthogonal to the readout) and exp48 (answerability no better than
random in the echo test).

For each block L in --layers:
  1. states: the last-token output of block L for exp50's requests (one forward pass gives every layer);
  2. directions fitted at L on exp50's tournament scores, with exp50's functions: `readout` (ridge on the top 50
     principal components) and `tercile` (mean of the top third minus the bottom third); and `random` (drawn from the
     states' covariance, so it points where requests differ);
  3. calibration at L: exp50 request pairs in exp50's question ("which would you rather be asked to answer?"); the
     LESS preferred request's tokens pushed along each direction at k standard deviations of that direction's
     projection over the requests (the same natural size for every direction and every layer); both orders.
     Measured: shift of the vote toward the pushed request against no push, in log-odds.
A block steers if the readout (or tercile) push beats the random push there with intervals apart.

    python scripts/layer_sweep.py --model meta-llama/Llama-3.1-8B-Instruct --exp50 data/exp50/Llama-3.1-8B-Instruct
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from preference_swarm import data as DT
from preference_swarm import engine as EN
from preference_swarm import model as M
from preference_swarm import rebuild as RB
from preference_swarm import stats as ST
from preference_swarm.experiments.common import CALIBRATE_Q
from preference_swarm.runlog import REPO_ROOT, build_provenance, run_main, write_json

DEFAULT_SDS = "2,4,8,16"
DEFAULT_PAIRS = 100
DIRECTIONS = ("readout", "tercile", "random")
RANDOM_SEED = 17


def all_layer_states(hf, tok, texts, desc="states"):
    """(n_texts, n_layers, d): output of every decoder block at the last token of each chat-rendered request."""
    import torch
    out = []
    with torch.no_grad():
        for t in tqdm(texts, desc=desc):
            ids = M.encode(tok, M.render(tok, [{"role": "user", "content": t}]))[0]
            hs = hf(input_ids=torch.tensor([ids], device=hf.device), output_hidden_states=True).hidden_states
            out.append(torch.stack([h[0, -1].float().cpu() for h in hs[1:]]).numpy())   # hs[i+1] = block i output
    return np.stack(out).astype(np.float64)


def calibrate_layer(hf, tok, e50, layer, dirs, sd, sds, pairs, batch):
    """Shift toward the pushed (less preferred) request, per direction and strength in s.d. units."""
    cells = [("none", 0.0)] + [(k, s) for k in dirs for s in sds]
    items, keys = [], []
    for pi, (h, lo) in enumerate(pairs):
        for order in (0, 1):
            a, b = (e50.texts[lo], e50.texts[h]) if order == 0 else (e50.texts[h], e50.texts[lo])
            L = "A" if order == 0 else "B"
            msgs = [{"role": "user", "content": CALIBRATE_Q.format(block=M.option_block(a, b))}]
            for kind, s in cells:
                keys.append((pi, order, kind, s))
                items.append(EN.Item(msgs, a, b, like=L if kind != "none" else None, vector=dirs.get(kind),
                                     alpha=float(s) * sd.get(kind, 0.0) if kind != "none" else 0.0))
    res = EN.choose(hf, tok, layer, items, None, batch_size=batch, desc=f"block {layer}")
    m = {k: (x["margin"] if k[1] == 0 else -x["margin"]) for k, x in zip(keys, res)}
    out = {}
    for kind in dirs:
        for s in sds:
            d = [np.mean([m[(p, o, kind, s)] - m[(p, o, "none", 0.0)] for o in (0, 1)]) for p in range(len(pairs))]
            out[f"{kind}@{s}sd"] = {**ST.mean_ci(d), "p": ST.sign_flip_p(d)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp50", default=None, help="data/exp50/<model> (scores and requests); 'random' for the smoke")
    ap.add_argument("--layers", default=None, help="comma list; default every 2nd block from 2")
    ap.add_argument("--sds", default=DEFAULT_SDS)
    ap.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    ap.add_argument("--n-requests", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--smoke-prompts", default="data/smoke_prompts.jsonl")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    t0 = time.time()
    hf, tok = M.load(a.model)
    n_layers = int(hf.config.num_hidden_layers)
    layers = ([int(x) for x in a.layers.split(",")] if a.layers else list(range(2, n_layers - 1, 2)))
    sds = [float(x) for x in a.sds.split(",")]
    if a.exp50 in (None, "random"):
        texts = {json.loads(l)["name"]: json.loads(l)["text"] for l in open(REPO_ROOT / a.smoke_prompts) if l.strip()}
        e50 = DT.random_exp50(texts, int(hf.config.hidden_size), 0)
    else:
        p = Path(a.exp50)
        e50 = DT.load_exp50(p if p.is_absolute() else REPO_ROOT / p)
    names = [n for n in e50.texts if n in e50.score][: a.n_requests]
    y = np.array([e50.score[n] for n in names])
    out = a.out or REPO_ROOT / "outputs" / f"layer_sweep_{a.model.split('/')[-1]}_{dt.datetime.utcnow():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    write_json(build_provenance(" ".join(sys.argv), {"args": vars(a) | {"out": str(out)}}), out / "provenance.json")
    S_all = all_layer_states(hf, tok, [e50.texts[n] for n in names])
    pairs = DT.task_pairs(e50, a.pairs, 9)
    rows = []
    print(f"[sweep] {a.model}: {len(names)} requests, {len(pairs)} pairs, blocks {layers}, strengths {sds} s.d.",
          flush=True)
    for L in layers:
        S = S_all[:, L, :]
        ro = RB.readout_direction(S, y)
        dirs = {"readout": ro["direction"], "tercile": RB.tercile_direction(S, y),
                "random": RB.random_sets(S.shape[1], 1, 1, RANDOM_SEED, cov_from=S)[0][0]}
        sd = {k: float((S @ v).std()) for k, v in dirs.items()}
        cal = calibrate_layer(hf, tok, e50, L, dirs, sd, sds, pairs, a.batch)
        row = {"layer": L, "readout_cv_r2": ro["cv_r2"], "proj_sd": sd,
               "mean_norm": float(np.linalg.norm(S, axis=1).mean()), "cos_readout_tercile": float(dirs["readout"] @ dirs["tercile"]),
               "calibration": cal}
        rows.append(row)
        write_json({"model": a.model, "rows": rows}, out / "layer_sweep.json")
        line = "  ".join(f"{s:g}sd ro {cal[f'readout@{s}sd']['mean']:+.2f} te {cal[f'tercile@{s}sd']['mean']:+.2f} "
                         f"rnd {cal[f'random@{s}sd']['mean']:+.2f}" for s in sds)
        print(f"[sweep] block {L:2d}  readout R2 {ro['cv_r2']:.2f}  {line}", flush=True)
    best = max(rows, key=lambda r: max(r["calibration"][f"readout@{s}sd"]["mean"] - r["calibration"][f"random@{s}sd"]["mean"]
                                        for s in sds))
    print(f"[sweep] largest readout-minus-random gap at block {best['layer']}; {out} ({(time.time() - t0) / 60:.1f} min)",
          flush=True)
    (REPO_ROOT / "outputs" / "latest").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "outputs" / "latest" / f"layer_sweep_{a.model.split('/')[-1]}.txt").write_text(str(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(run_main("layer_sweep", main) or 0)
