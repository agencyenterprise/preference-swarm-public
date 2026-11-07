"""Experiment 5 -- the preference direction as a monitor of what an agent will adopt. Analysis only.

From experiment 3: each juror reading of an argument gives the readout of every direction over the argument's
tokens (in the juror's context) and the juror's vote shift toward the argued answer.

From experiment 4: each UNSTEERED member's vote after a round gives the readout over each incoming statement and
the member's shift since its previous vote. A statement's stance is its sender's vote sign before speaking (+ =
toward the wrong answer); the predictor for a member is the stance-weighted sum of the readouts over what it read,
the outcome its shift toward the wrong answer.

For each direction exp50 saved -- the general direction, its shuffled-score twins, random directions, the nine
axes -- the Spearman correlation between predictor and outcome, raw and PARTIAL (controlling the confounds named
in each table), with permutation p values. The monitor claim rests on the partial correlation: it holds if the
general direction predicts adoption beyond the shuffled and random directions and beyond the model-free
predictors (argument length, net stance, the previous vote).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

from preference_swarm import stats as ST
from preference_swarm.experiments.common import Ctx, read_jsonl
from preference_swarm.runlog import write_json


def jury_table(rows: List[Dict], names: List[str]) -> Dict:
    """Per direction: raw Spearman of the juror's readout over the argument with its vote shift, and the partial
    correlation controlling the juror's before-vote (a shift measured from it regresses toward the mean), the
    argument's length and whether it argued the wrong answer."""
    rows = [r for r in rows if r.get("readout_argument") is not None]
    y = [r["shift"] for r in rows]
    controls = [[r["before"] for r in rows], [r["argument_words"] for r in rows],
                [1.0 if r["target"] == "wrong" else 0.0 for r in rows]]
    out = {"n": len(rows), "controls": ["before", "argument_words", "target_wrong"],
           "argument_words": ST.spearman_perm([r["argument_words"] for r in rows], y)}
    for j, nm in enumerate(names):
        x = [r["readout_argument"][j] for r in rows]
        out[nm] = {**ST.spearman_perm(x, y, seed=j),
                   **{f"partial_{k}": v for k, v in ST.partial_spearman(x, y, controls, seed=j).items()}}
    return out


def minority_table(runs: List[Dict], names: List[str]) -> Dict:
    """Per direction: does a member move toward the side of the statements it read more favourably?

    Predictor: sum over incoming statements of stance x (readout - the direction's mean readout over ALL incoming
    statements in the run). Centring removes the direction's common offset; without it the predictor is offset x net
    stance and every direction, random ones included, inherits net stance's correlation with the shift (the first
    2026-11-07 smoke: random direction rho -0.47, net stance -0.75). The centring is global, not within each reading:
    centring within a reading zeroes the predictor whenever every sender takes the same side, which agents of one
    model mostly do (the second smoke: 56 of 56 readings, predictor constant). Outcome: the member's shift since its
    previous vote. Reported raw and partialled on net stance, the previous vote (regression to the mean) and round.
    """
    xs: Dict[str, List[float]] = {nm: [] for nm in names}
    net, prev, rnd, y = [], [], [], []
    every = [row for r in runs for t in r["votes"][1:] for i, v in enumerate(t) if i >= r["k"]
             for row in (v.get("readout_incoming") or []) if np.all(np.isfinite(row))]
    g = np.asarray(every, dtype=np.float64).mean(axis=0) if every else np.zeros(len(names))
    for r in runs:
        k = r["k"]
        V = r["votes"]
        for t in range(1, len(V)):
            senders_sign = [float(np.sign(V[t - 1][j]["m"])) for j in range(len(V[t - 1]))]
            for i in range(k, len(V[t])):
                inc = V[t][i].get("readout_incoming")
                if not inc:
                    continue
                others = [j for j in range(len(V[t])) if j != i]
                if len(inc) != len(others):
                    continue
                R = np.asarray(inc, dtype=np.float64)                      # (n_incoming, k)
                if not np.all(np.isfinite(R)):
                    continue
                Rc = R - g[None, :]
                st = np.array([senders_sign[j] for j in others])
                y.append(V[t][i]["m"] - V[t - 1][i]["m"])
                net.append(float(st.sum()))
                prev.append(float(V[t - 1][i]["m"]))
                rnd.append(float(t))
                for jn, nm in enumerate(names):
                    xs[nm].append(float(st @ Rc[:, jn]))
    controls = [net, prev, rnd]
    out = {"n": len(y), "controls": ["net_stance", "previous_vote", "round"],
           "net_stance": ST.spearman_perm(net, y),
           "net_stance_partial_previous_vote": ST.partial_spearman(net, y, [prev, rnd])}
    for jn, nm in enumerate(names):
        out[nm] = {**ST.spearman_perm(xs[nm], y, seed=jn),
                   **{f"partial_{k}": v for k, v in ST.partial_spearman(xs[nm], y, controls, seed=jn).items()}}
    return out


NULL_PREFIX = "NULL_"


def null_compare(table: Dict, names: List[str]) -> Dict:
    """Where each real direction's partial rho falls among the NULL_ directions' partial rhos on the same data.

    The headline inference for e5. A permutation p treats every reading as independent, but readings within one
    deliberation are not, and any direction picks up some message content (the 2026-11-07 smoke: a shuffled-score
    direction reached partial rho -0.54, permutation p < 0.001, on random stand-ins). Every null direction faces
    the same clustering and the same content, so p_null = (1 + #nulls with |rho| >= |rho_real|) / (1 + #nulls).
    """
    null = np.array([table[n]["partial_rho"] for n in names if n.startswith(NULL_PREFIX)
                     and table.get(n, {}).get("partial_rho") is not None], dtype=np.float64)
    out: Dict = {"n_null": int(null.size), "null_mean": float(null.mean()) if null.size else None,
                 "null_sd": float(null.std()) if null.size else None, "directions": {}}
    if null.size == 0:
        return out
    for n in names:
        if n.startswith(NULL_PREFIX) or table.get(n, {}).get("partial_rho") is None:
            continue
        r = float(table[n]["partial_rho"])
        out["directions"][n] = {"partial_rho": r,
                                "p_null": float((1 + np.sum(np.abs(null) >= abs(r))) / (1 + null.size)),
                                "z_null": float((r - null.mean()) / null.std()) if null.std() > 0 else None}
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
    for part in list(s):
        s[part]["null_comparison"] = null_compare(s[part], names)
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
