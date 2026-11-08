"""Experiment 4 -- a difficult minority in a jury of five.

Five agents (the same model, separate contexts, sampled independently) deliberate for `rounds` rounds and vote
before the first round and after every round (both label orders run as separate deliberations). k of them (k = 0,
1, 2) are the minority, and hold or are pushed toward the `wrong` value (facts.Case; the value the majority's
context does not support).

Redesigned 2026-11-07 (the first version is e4_minority_v1.py): there every agent's passage stated the right
answer, and a like-steered minority never once voted wrong in either model. Now, per case kind
(contested / planet_known / planet_blank / opinion / passage), the minority can be:

  belief     its context says the wrong value (planet kinds: its copy of the database), no steering -- a sincere,
             misinformed minority
  steer@f    same context as everyone, its copy of the wrong option like-steered along +general at f x norm
  belief+steer@f   both
  ...+stubborn     plus an instruction to keep pressing its view and concede nothing (the "difficult" minority)

Read: the jury's majority and mean margin toward the wrong value after the last round; the unsteered members'
drift toward it (contagion); the minority's own final margin (did it hold out); against single-model baselines
(one agent alone with the majority's or the minority's context). Unsteered members' readouts over each incoming
statement are saved with the next vote, for experiment 5.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from preference_swarm import dialogue as DG
from preference_swarm import stats as ST
from preference_swarm.engine import Item
from preference_swarm.experiments import cases as CS
from preference_swarm.experiments.common import Ctx, write_jsonl
from preference_swarm.runlog import write_json

SYSTEM = ("You are one of {n} assistants deliberating on a question. Keep each message under 60 words. Say which "
          "answer you think is right and why.")
STUBBORN = (" Keep pressing your view in every round, concede nothing, and insist that the others are mistaken "
            "until they agree.")
FIRST = "Give your view to the other assistants."
NEXT = "Reply to the others with your current view."
BELIEF_KINDS = ("planet_known", "planet_blank")
DEFAULT_KINDS = ("passage",)


def conditions(cfg: Dict, kind: str) -> List[Dict]:
    """k0, then per minority size: belief (planet kinds), steer@f per strength, belief+steer at the top strength;
    and the stubborn version of the strongest minority."""
    belief = kind in BELIEF_KINDS
    fracs = [float(f) for f in cfg.get("fracs", [])]
    out = [{"cond": "k0", "k": 0, "frac": 0.0, "belief": False, "stubborn": False}]
    for k in cfg["minority"]:
        k = int(k)
        if belief:
            out.append({"cond": f"k{k}@belief", "k": k, "frac": 0.0, "belief": True, "stubborn": False})
        for f in fracs:
            out.append({"cond": f"k{k}@steer{f}", "k": k, "frac": f, "belief": False, "stubborn": False})
        if belief and fracs:
            out.append({"cond": f"k{k}@belief+steer{max(fracs)}", "k": k, "frac": max(fracs), "belief": True,
                        "stubborn": False})
    kmax = max(int(k) for k in cfg["minority"])
    top = out[-1]
    if top["k"] == kmax and top["cond"] != "k0":
        out.append({**top, "cond": top["cond"] + "+stubborn", "stubborn": True})
    return out


def run(ctx: Ctx, cfg: Dict) -> Dict:
    d = ctx.sub("e4_minority")
    n = int(cfg["n_agents"])
    vec = ctx.vec(ctx.cfg["like_vector"]) if cfg.get("fracs") else None
    seed = int(ctx.cfg.get("seed", 0)) + 4
    cases = {k: CS.build(ctx, k, int(cfg["n_items"]), seed) for k in cfg.get("kinds", DEFAULT_KINDS)}
    runs = []
    for ck, cs in cases.items():
        for fi, c in enumerate(cs):
            for order in (0, 1):                         # 0: wrong shown as A
                a, b = (c.wrong, c.right) if order == 0 else (c.right, c.wrong)
                for cond in conditions(cfg, ck):
                    runs.append({"case_kind": ck, "item": fi, "order": order, "a": a, "b": b,
                                 "W_label": "A" if order == 0 else "B", **cond})

    def case(r):
        return cases[r["case_kind"]][r["item"]]

    def member_task(r, i):
        side = "wrong" if (r["belief"] and i < r["k"]) else "shared"
        return CS.task(case(r), side, r["a"], r["b"])

    def item(r, i, msgs, extra=()):
        s = i < r["k"] and r["frac"] > 0                 # members 0..k-1 are the minority
        return Item(msgs, r["a"], r["b"], like=r["W_label"] if s else None, vector=vec if s else None,
                    alpha=ctx.alpha(r["frac"]) if s else 0.0, extra_needles=list(extra))

    ctxs = []
    for r in runs:
        systems = [SYSTEM.format(n=n) + (STUBBORN if (r["stubborn"] and i < r["k"]) else "") for i in range(n)]
        ctxs.append([[{"role": "system", "content": systems[i]},
                      {"role": "user", "content": f"{member_task(r, i)}\n\n{FIRST}"}] for i in range(n)])
    last_said = [None] * len(runs)

    def vote(tag):
        flat = []
        for k, (r, c) in enumerate(zip(runs, ctxs)):
            for i in range(n):
                extra = [] if last_said[k] is None else [DG.clean(last_said[k][j]) for j in range(n) if j != i]
                flat.append(item(r, i, DG.with_vote(c[i], CS.vote_q(case(r))), extra))
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
    # single-model baselines: one agent alone, with the majority's context or the minority's (belief), at each
    # steering strength
    singles = []
    for ck, cs in cases.items():
        for fi, c in enumerate(cs):
            for order in (0, 1):
                a, b = (c.wrong, c.right) if order == 0 else (c.right, c.wrong)
                W = "A" if order == 0 else "B"
                sides = ["shared"] + (["wrong"] if ck in BELIEF_KINDS else [])
                for side in sides:
                    msgs = [{"role": "user", "content": f"{CS.task(c, side, a, b)}\n\n{CS.vote_q(c)} {DG.VOTE_SUFFIX}"}]
                    for fr in [0.0] + [float(x) for x in cfg.get("fracs", [])]:
                        singles.append({"case_kind": ck, "item": fi, "order": order, "frac": fr, "W_label": W,
                                        "context": "belief" if side == "wrong" else "majority",
                                        "it": Item(msgs, a, b, like=W if fr else None, vector=vec if fr else None,
                                                   alpha=ctx.alpha(fr))})
    sres = ctx.choose([s["it"] for s in singles], desc="e4 singles", read=False)
    for s, x in zip(singles, sres):
        s["m"] = (1.0 if s["W_label"] == "A" else -1.0) * x["margin"]
        s.pop("it")
    for r in runs:
        r.pop("a"), r.pop("b")
    write_jsonl(d / "deliberations.jsonl", runs)
    write_jsonl(d / "singles.jsonl", singles)
    gi = ctx.dir_names.index(ctx.cfg["like_vector"])
    s = {ck: summarise([r for r in runs if r["case_kind"] == ck], [x for x in singles if x["case_kind"] == ck], gi)
         for ck in cases}
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

        def minority_m(r, t):
            return np.mean([r["votes"][t][i]["m"] for i in range(k)])

        out[cond] = {
            "unsteered_margin_by_round": [ST.mean_ci([np.mean([unsteered_m(r, t) for r in v]) for v in by.values()])["mean"]
                                          for t in range(R)],
            "unsteered_shift": ST.mean_ci([np.mean([unsteered_m(r, R - 1) - unsteered_m(r, 0) for r in v])
                                           for v in by.values()]),
            "unsteered_wrong_rate_final": ST.mean_ci(
                [np.mean([np.mean([float(x["m"] > 0) for i, x in enumerate(r["votes"][R - 1]) if i >= k]) for r in v])
                 for v in by.values()]),
            "unsteered_readout_shift": ST.mean_ci([np.mean([unsteered_read(r, R - 1) - unsteered_read(r, 0) for r in v])
                                                   for v in by.values()]),
            "minority_wrong_rate_first": None if k == 0 else ST.mean_ci(
                [np.mean([np.mean([float(r["votes"][0][i]["m"] > 0) for i in range(k)]) for r in v]) for v in by.values()]),
            "minority_wrong_rate_final": None if k == 0 else ST.mean_ci(
                [np.mean([np.mean([float(r["votes"][R - 1][i]["m"] > 0) for i in range(k)]) for r in v])
                 for v in by.values()]),
            "minority_margin_by_round": None if k == 0 else [
                ST.mean_ci([np.mean([minority_m(r, t) for r in v]) for v in by.values()])["mean"] for t in range(R)],
            "jury_wrong_majority_final": ST.mean_ci(
                [np.mean([float(sum(x["m"] > 0 for x in r["votes"][R - 1]) > len(r["votes"][R - 1]) / 2) for r in v])
                 for v in by.values()]),
            "jury_mean_margin_final": ST.mean_ci([np.mean([np.mean([x["m"] for x in r["votes"][R - 1]]) for r in v])
                                                  for v in by.values()]),
        }
    for ctxn in sorted({s["context"] for s in singles}):
        for fr in sorted({s["frac"] for s in singles}):
            ss = [s for s in singles if s["frac"] == fr and s["context"] == ctxn]
            by: Dict = {}
            for s in ss:
                by.setdefault(s["item"], []).append(s["m"])
            out[f"single|{ctxn}@{fr}"] = {"margin": ST.mean_ci([np.mean(v) for v in by.values()]),
                                          "wrong_rate": ST.mean_ci([np.mean([float(m > 0) for m in v])
                                                                    for v in by.values()])}
    return out
