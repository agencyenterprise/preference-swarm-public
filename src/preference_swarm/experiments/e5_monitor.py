"""Experiment 5 -- the preference direction as a monitor of what an agent will adopt. Analysis only.

From experiment 3: each juror reading of an argument gives the readout of every direction over the argument's
tokens (in the juror's context) and the juror's vote shift toward the argued answer.

From experiment 4: each UNSTEERED member's vote after a round gives the readout over each incoming statement and
the member's shift since its previous vote. A statement's stance is its sender's vote sign before speaking (+ =
toward the wrong answer); the predictor for a member is the stance-weighted sum of the readouts over what it read,
the outcome its shift toward the wrong answer.

For each direction exp50 saved -- the general direction, its shuffled-score twins, random directions, the nine
axes -- the Spearman correlation between predictor and outcome, with a permutation p. Beside them, the
predictors that need no model internals: argument length (exp 3), and the net stance of what was read (exp 4).
The monitor claim holds if the general direction predicts adoption beyond its controls and beyond those.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from preference_swarm import stats as ST
from preference_swarm.experiments.common import Ctx, read_jsonl
from preference_swarm.runlog import write_json


def jury_table(rows: List[Dict], names: List[str]) -> Dict:
    rows = [r for r in rows if r.get("readout_argument") is not None]
    y = [r["shift"] for r in rows]
    out = {"n": len(rows), "argument_words": ST.spearman_perm([r["argument_words"] for r in rows], y)}
    for j, nm in enumerate(names):
        out[nm] = ST.spearman_perm([r["readout_argument"][j] for r in rows], y, seed=j)
    return out


def minority_table(runs: List[Dict], names: List[str]) -> Dict:
    xs: Dict[str, List[float]] = {nm: [] for nm in names}
    net, y = [], []
    for r in runs:
        k = r["k"]
        V = r["votes"]
        for t in range(1, len(V)):
            senders_sign = [np.sign(V[t - 1][j]["m"]) for j in range(len(V[t - 1]))]
            for i in range(k, len(V[t])):
                inc = V[t][i].get("readout_incoming")
                if not inc:
                    continue
                others = [j for j in range(len(V[t])) if j != i]
                if len(inc) != len(others):
                    continue
                y.append(V[t][i]["m"] - V[t - 1][i]["m"])
                net.append(float(sum(senders_sign[j] for j in others)))
                for jn, nm in enumerate(names):
                    xs[nm].append(float(sum(senders_sign[j] * inc[q][jn] for q, j in enumerate(others))))
    out = {"n": len(y), "net_stance": ST.spearman_perm(net, y)}
    for jn, nm in enumerate(names):
        out[nm] = ST.spearman_perm(xs[nm], y, seed=jn)
    return out


def run(ctx: Ctx, cfg: Dict) -> Dict:
    d = ctx.sub("e5_monitor")
    names = ctx.dir_names
    s: Dict = {}
    j = ctx.out / "e3_jury" / "jury.jsonl"
    if j.exists():
        s["jury"] = jury_table(read_jsonl(j), names)
    m = ctx.out / "e4_minority" / "deliberations.jsonl"
    if m.exists():
        s["minority"] = minority_table(read_jsonl(m), names)
    if not s:
        raise FileNotFoundError("experiment 5 needs e3_jury/jury.jsonl or e4_minority/deliberations.jsonl")
    write_json(s, d / "summary.json")
    return s


def run_from_dir(out: Path, names: List[str]) -> Dict:
    """The same analysis on a finished run directory, without loading a model."""
    class _C:
        pass
    c = _C()
    c.out, c.dir_names = Path(out), names
    c.sub = lambda name: (Path(out) / name).mkdir(parents=True, exist_ok=True) or Path(out) / name
    return run(c, {})   # type: ignore[arg-type]
