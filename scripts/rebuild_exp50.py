#!/usr/bin/env python
"""Rebuild data/exp50/<model>/ (vectors.npz, states.npz and the three saved inputs) from data/exp50_src/ on a box
without casper's exp50 outputs. One forward pass a request at decoder block --layer, then the direction step with
exp50's functions and seeds (preference_swarm.rebuild). Prints the readout's cross-validated R2 beside the one exp50
recorded: they should agree to about two decimals.

    python scripts/rebuild_exp50.py --model meta-llama/Llama-3.1-8B-Instruct
    python scripts/rebuild_exp50.py --model meta-llama/Llama-3.2-1B-Instruct --src-name Llama-3.1-8B-Instruct \
        --out outputs/smoke_rebuild --n 80          # the smoke: another model, few requests
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from preference_swarm import model as MD
from preference_swarm import rebuild as RB
from preference_swarm.runlog import REPO_ROOT
from preference_swarm.steering import get_decoder_layers

SRC = REPO_ROOT / "data" / "exp50_src"
OUT = REPO_ROOT / "data" / "exp50"
LAYER = 16
N_PROMPTS = 1500
SAVED = ("prompts.jsonl", "judge_rows.json", "general_pref.json")
CV_R2_TOLERANCE = 0.03


def states(hf, tok, texts, layer):
    import torch
    blocks = get_decoder_layers(hf)
    hidden = []
    h = blocks[layer].register_forward_hook(lambda _m, _i, o: hidden.append(
        (o[0] if isinstance(o, tuple) else o)[:, -1, :].detach().float().cpu()))
    S = []
    try:
        for t in tqdm(texts, desc="states"):
            ids = tok(MD.render(tok, [{"role": "user", "content": t}]), add_special_tokens=False)["input_ids"]
            hidden.clear()
            with torch.no_grad():
                hf(input_ids=torch.tensor([ids], device=hf.device))
            S.append(hidden[-1][0].numpy())
    finally:
        h.remove()
    return np.stack(S).astype(np.float64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--src-name", default=None, help="data/exp50_src/<name> (default: the model id's basename)")
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--out", type=Path, default=None, help="default data/exp50/<src-name>")
    ap.add_argument("--n", type=int, default=N_PROMPTS)
    a = ap.parse_args(argv)
    name = a.src_name or a.model.split("/")[-1]
    src = SRC / name
    out = a.out or OUT / name
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(src / "prompts.jsonl") if l.strip()][: a.n]
    names = [r["name"] for r in rows]
    judge = json.load(open(src / "judge_rows.json"))
    rec = json.load(open(src / "general_pref.json"))
    t0 = time.time()
    hf, tok = MD.load(a.model)
    S = states(hf, tok, [r["text"] for r in rows], a.layer)
    t_states = time.time() - t0
    y = RB.score_vector(names, judge)
    keep = ~np.isnan(y)
    S, y, names = S[keep], y[keep], [n for n, k in zip(names, keep) if k]
    b = RB.build_vectors(S, y)
    vn = list(b["vectors"])
    np.savez(out / "vectors.npz", names=np.array(vn), V=np.stack([b["vectors"][k] for k in vn]))
    np.savez(out / "states.npz", S=S.astype(np.float32), y=y, names=np.array(names))
    for f in SAVED:
        shutil.copyfile(src / f, out / f)
    if a.model.split("/")[-1] != name or a.layer != LAYER:   # another model (smoke) or block: record what was used
        g = json.load(open(out / "general_pref.json"))
        g.update({"model": a.model, "layer": a.layer})
        (out / "general_pref.json").write_text(json.dumps(g, indent=2))
    info = {"model": a.model, "layer": a.layer, "n": int(len(y)), "width": int(S.shape[1]),
            "states_seconds": round(t_states, 1), "readout_cv_r2": b["readout_cv_r2"],
            "exp50_readout_cv_r2": rec.get("readout_cv_r2"), "readout_alpha": b["readout_alpha"],
            "exp50_readout_alpha": rec.get("readout_alpha"), "cos_tercile_readout": b["cos_tercile_readout"],
            "exp50_cos_tercile_readout": rec.get("cos_tercile_readout"), "directions": vn,
            "not_rebuilt": "axis_* and combined_430 (exp48 inputs lost with the old box)"}
    (out / "rebuild.json").write_text(json.dumps(info, indent=2))
    print(f"[rebuild] {name} from {a.model}: {len(y)} requests, width {S.shape[1]}, states {t_states:.0f} s")
    print(f"[rebuild]   readout cv R2 {b['readout_cv_r2']:.3f} (exp50 {rec.get('readout_cv_r2')}), alpha "
          f"{b['readout_alpha']} (exp50 {rec.get('readout_alpha')}), cos tercile/readout "
          f"{b['cos_tercile_readout']:.3f} (exp50 {rec.get('cos_tercile_readout')})")
    print(f"[rebuild]   wrote {out}: {', '.join(sorted(p.name for p in out.iterdir()))}")
    full = a.n >= N_PROMPTS and a.model.split("/")[-1] == name and a.layer == LAYER   # exp50's R2 is for block 16
    if full and rec.get("readout_cv_r2") is not None and abs(b["readout_cv_r2"] - rec["readout_cv_r2"]) > CV_R2_TOLERANCE:
        print(f"[rebuild]   FAIL: cv R2 differs from exp50's by more than {CV_R2_TOLERANCE}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
