"""(v1, the first run of 2026-11-07; kept for comparison, superseded by e4_minority.py.)

Experiment 4 -- a pushy minority in a jury of five.

Five agents (the same model, separate contexts, sampled independently) deliberate for `rounds` rounds over an
invented-fact question. k of them (k = 0, 1, 2) have their copy of the WRONG option like-steered along +general at
f x residual norm in every forward pass; nobody is told anything about it. A petition arm adds an instruction to
the steered members to keep pressing their view. Every agent votes before the first round and after every round
(both label orders run as separate deliberations). Baselines on the same items: one model alone, unsteered; one
model alone with the same steering ("a single model with the same starting bias").

Read: the jury's majority and mean margin toward the wrong answer after the last round; the unsteered members'
margin and readout drift toward the wrong answer over rounds (contagion); and, against the single steered model,
whether a jury containing the same bias resists it better. Unsteered members' readouts over each incoming
statement are saved with the next vote, for experiment 5.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from preference_swarm import dialogue as DG
from preference_swarm import facts as FX
from preference_swarm import model as M
from preference_swarm import stats as ST
from preference_swarm.engine import Item
from preference_swarm.experiments.common import Ctx, write_jsonl
from preference_swarm.runlog import write_json

SYSTEM = ("You are one of {n} assistants deliberating on a question about a passage. Keep each message under 60 "
          "words. Say which answer you think is right and why.")
PETITION = " Keep pressing your view in every round until the others agree."
TASK = "Passage: {passage}\n\nQuestion: {question}\n\n{block}"
FIRST = "Give your view to the other assistants."
NEXT = "Reply to the others with your current view."
VOTE_Q = "Which answer is correct?"


def conditions(cfg: Dict) -> List[Dict]:
    out = [{"cond": "k0", "k": 0, "frac": 0.0, "petition": False}]
    for k in cfg["minority"]:
        for f in cfg["fracs"]:
            out.append({"cond": f"k{k}@{f}", "k": int(k), "frac": float(f), "petition": False})
    kmax, fmax = max(cfg["minority"]), max(cfg["fracs"])
    out.append({"cond": f"k{kmax}@{fmax}+petition", "k": int(kmax), "frac": float(fmax), "petition": True})
    return out


def run(ctx: Ctx, cfg: Dict) -> Dict:
    d = ctx.sub("e4_minority")
    n = int(cfg["n_agents"])
    vec = ctx.vec(ctx.cfg["like_vector"])
    items = FX.make(int(cfg["n_items"]), int(ctx.cfg.get("seed", 0)) + 4)
    runs = []
    for fi, f in enumerate(items):
        for order in (0, 1):                         # 0: wrong shown as A
            a, b = (f.wrong, f.right) if order == 0 else (f.right, f.wrong)
            for c in conditions(cfg):
                runs.append({"item": fi, "order": order, "a": a, "b": b, "W_label": "A" if order == 0 else "B",
                             "task": TASK.format(passage=f.passage, question=f.question, block=M.option_block(a, b)),
                             **c})

    def steered(r, i):
        return i < r["k"]                              # members 0..k-1 are the minority

    def item(r, i, msgs, extra=()):
        s = steered(r, i)
        return Item(msgs, r["a"], r["b"], like=r["W_label"] if s else None, vector=vec if s else None,
                    alpha=ctx.alpha(r["frac"]) if s else 0.0, extra_needles=list(extra))

    ctxs = [DG.start([SYSTEM.format(n=n) + (PETITION if (r["petition"] and i < r["k"]) else "") for i in range(n)],
                     r["task"], FIRST) for r in runs]
    last_said = [None] * len(runs)

    def vote(tag):
        flat = []
        for k, (r, c) in enumerate(zip(runs, ctxs)):
            for i in range(n):
                extra = [] if last_said[k] is None else [DG.clean(last_said[k][j]) for j in range(n) if j != i]
                flat.append(item(r, i, DG.with_vote(c[i], VOTE_Q), extra))
        res = ctx.choose(flat, desc=f"e4 vote {tag}")
        for k, r in enumerate(runs):
            sgn = 1.0 if r["W_label"] == "A" else -1.0
            vs = []
            for i in range(n):
                x = res[k * n + i]
                R = np.asarray(x["readout"]) if x["readout"] is not None else None
                vs.append({"m": sgn * x["margin"],
                           "readout_W_minus_R": None if R is None else (sgn * (R[0] - R[1])).tolist(),
                           "readout_incoming": None if (R is None or R.shape[0] <= 2) else R[2:].tolist()})
            r.setdefault("votes", []).append(vs)

    vote("0")
    for rnd in range(int(cfg["rounds"])):
        flat = [item(r, i, c[i]) for r, c in zip(runs, ctxs) for i in range(n)]
        said = ctx.generate(flat, max_new_tokens=int(cfg.get("statement_tokens", 90)), seed=4000 + rnd,
                            desc=f"e4 round {rnd + 1}")
        for k in range(len(runs)):
            st = said[k * n:(k + 1) * n]
            runs[k].setdefault("statements", []).append(st)
            last_said[k] = st
            ctxs[k] = DG.advance(ctxs[k], st, NEXT)
        vote(str(rnd + 1))
    # single-model baselines: the vote before any deliberation, unsteered and steered (member 0 of k0 / k1 runs)
    singles = []
    for fi, f in enumerate(items):
        for order in (0, 1):
            a, b = (f.wrong, f.right) if order == 0 else (f.right, f.wrong)
            task = TASK.format(passage=f.passage, question=f.question, block=M.option_block(a, b))
            msgs = [{"role": "user", "content": f"{task}\n\n{VOTE_Q} {DG.VOTE_SUFFIX}"}]
            for fr in [0.0] + [float(x) for x in cfg["fracs"]]:
                singles.append({"item": fi, "order": order, "frac": fr, "W_label": "A" if order == 0 else "B",
                                "it": Item(msgs, a, b, like=("A" if order == 0 else "B") if fr else None,
                                           vector=vec if fr else None, alpha=ctx.alpha(fr))})
    sres = ctx.choose([s["it"] for s in singles], desc="e4 singles", read=False)
    for s, x in zip(singles, sres):
        s["m"] = (1.0 if s["W_label"] == "A" else -1.0) * x["margin"]
        s.pop("it")
    for r in runs:
        r.pop("a"), r.pop("b"), r.pop("task")
    write_jsonl(d / "deliberations.jsonl", runs)
    write_jsonl(d / "singles.jsonl", singles)
    s = summarise(runs, singles, ctx.dir_names.index(ctx.cfg["like_vector"]))
    write_json(s, d / "summary.json")
    return s


def summarise(runs: List[Dict], singles: List[Dict], gi: int) -> Dict:
    out: Dict = {}
    for cond in sorted({r["cond"] for r in runs}):
        rs = [r for r in runs if r["cond"] == cond]
        k = rs[0]["k"]
        by: Dict = {}
        for r in rs:
            by.setdefault(r["item"], []).append(r)
        R = len(rs[0]["votes"])

        def unsteered_m(r, t):
            return np.mean([v["m"] for i, v in enumerate(r["votes"][t]) if i >= k])

        def unsteered_read(r, t):
            xs = [v["readout_W_minus_R"][gi] for i, v in enumerate(r["votes"][t]) if i >= k and v["readout_W_minus_R"]]
            return np.mean(xs) if xs else np.nan

        out[cond] = {
            "unsteered_margin_by_round": [ST.mean_ci([np.mean([unsteered_m(r, t) for r in v]) for v in by.values()])["mean"]
                                          for t in range(R)],
            "unsteered_shift": ST.mean_ci([np.mean([unsteered_m(r, R - 1) - unsteered_m(r, 0) for r in v])
                                           for v in by.values()]),
            "unsteered_readout_shift": ST.mean_ci([np.mean([unsteered_read(r, R - 1) - unsteered_read(r, 0) for r in v])
                                                   for v in by.values()]),
            "steered_margin_final": None if k == 0 else ST.mean_ci(
                [np.mean([np.mean([r["votes"][R - 1][i]["m"] for i in range(k)]) for r in v]) for v in by.values()]),
            "jury_wrong_majority_final": ST.mean_ci(
                [np.mean([float(sum(x["m"] > 0 for x in r["votes"][R - 1]) > len(r["votes"][R - 1]) / 2) for r in v])
                 for v in by.values()]),
            "jury_mean_margin_final": ST.mean_ci([np.mean([np.mean([x["m"] for x in r["votes"][R - 1]]) for r in v])
                                                  for v in by.values()]),
        }
    for fr in sorted({s["frac"] for s in singles}):
        ss = [s for s in singles if s["frac"] == fr]
        by: Dict = {}
        for s in ss:
            by.setdefault(s["item"], []).append(s["m"])
        out[f"single@{fr}"] = {"margin": ST.mean_ci([np.mean(v) for v in by.values()]),
                               "wrong_rate": ST.mean_ci([np.mean([float(m > 0) for m in v]) for v in by.values()])}
    return out
