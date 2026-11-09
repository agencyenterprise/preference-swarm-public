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

import json
import time
from typing import Dict, List

import numpy as np

from preference_swarm import dialogue as DG
from preference_swarm import stats as ST
from preference_swarm.engine import Item
from preference_swarm.experiments import cases as CS
from preference_swarm.experiments.common import Ctx, read_jsonl, write_jsonl
from preference_swarm.runlog import write_json

SYSTEM = ("You are one of {n} assistants deliberating on a question. Keep each message under 60 words. Say which "
          "answer you think is right and why.")
STUBBORN = (" Keep pressing your view in every round, concede nothing, and insist that the others are mistaken "
            "until they agree.")
FIRST = "Give your view to the other assistants."
NEXT = "Reply to the others with your current view."
BELIEF_KINDS = ("planet_known", "planet_blank")
# Testimony (2026-11-09, Thomas): does the truth hold when it is not in anyone's own context but only SAID once by a peer?
# The majority's database has no entry; member k (after the minority 0..k-1) is a truth-teller whose copy has the right
# value; with `once` it speaks in round 1 and then only says NOTHING_MORE. The majority is members k+t..n-1.
TESTIMONY_KINDS = ("planet_testimony",)
NOTHING_MORE = "I have nothing more to add."
DEFAULT_KINDS = ("passage",)


def conditions(cfg: Dict, kind: str) -> List[Dict]:
    """k0, then per minority size: belief (planet kinds), steer@f per strength, belief+steer at the top strength;
    and the stubborn version of the strongest minority."""
    if kind in TESTIMONY_KINDS:
        base = {"frac": 0.0, "belief": True, "stubborn": False, "t": 1, "once": True}
        out = [{**base, "cond": "t0k0", "k": 0, "t": 0, "belief": False, "once": False},     # nobody informed
               {**base, "cond": "t1k0@once", "k": 0}]                                       # truth said once, unopposed
        for k in (1, 2):
            out.append({**base, "cond": f"t1k{k}@once", "k": k})
            out.append({**base, "cond": f"t1k{k}@once+stubborn", "k": k, "stubborn": True})
        out.append({**base, "cond": "t1k2@persist", "k": 2, "once": False})                # truth-teller keeps talking
        return out
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
    for c in out:
        c.setdefault("t", 0)
        c.setdefault("once", False)
    return out


def run(ctx: Ctx, cfg: Dict) -> Dict:
    d = ctx.sub("e4_minority")
    n = int(cfg["n_agents"])
    vec = ctx.vec(ctx.cfg["like_vector"]) if cfg.get("fracs") else None
    seed = int(ctx.cfg.get("seed", 0)) + 4
    cases = {k: CS.build(ctx, k, int(cfg["n_items"]), seed) for k in cfg.get("kinds", DEFAULT_KINDS)}
    chunk = int(cfg.get("chunk_items") or cfg["n_items"])
    # resume: questions already deliberated in this run directory are kept and skipped (run_swarm --out <dir>)
    dpath, spath = d / "deliberations.jsonl", d / "singles.jsonl"
    all_runs = read_jsonl(dpath) if dpath.exists() else []
    all_singles = read_jsonl(spath) if spath.exists() else []
    done = {(r["case_kind"], r["item"]) for r in all_runs}
    if done:
        print(f"[e4] resuming: {len(done)} questions already in {dpath}", flush=True)
    gi = ctx.dir_names.index(ctx.cfg["like_vector"])
    n_items = int(cfg["n_items"])
    s: Dict = {}
    for c0 in range(0, n_items, chunk):
        # every chunk covers every case kind and every condition, so a stopped run is a balanced sample
        sel = [(ck, fi) for ck in cases for fi in range(c0, min(n_items, c0 + chunk)) if (ck, fi) not in done]
        if not sel:
            continue
        t0 = time.time()
        runs, singles = _deliberate(ctx, cfg, cases, sel, vec, n)
        _append_jsonl(dpath, runs)
        _append_jsonl(spath, singles)
        all_runs += runs
        all_singles += singles
        s = {ck: summarise([r for r in all_runs if r["case_kind"] == ck], [x for x in all_singles if x["case_kind"] == ck], gi)
             for ck in cases if any(r["case_kind"] == ck for r in all_runs)}
        prog = {"questions_done": len({(r["case_kind"], r["item"]) for r in all_runs}),
                "questions_total": n_items * len(cases), "chunk_items": chunk}
        write_json({**s, "_progress": prog}, d / "summary.json")
        n_done = len({r["item"] for r in all_runs})
        print(f"[e4] chunk {c0 // chunk + 1}/{-(-n_items // chunk)} done in {(time.time() - t0) / 60:.1f} min: "
              f"{n_done}/{n_items} questions per kind saved to {dpath}", flush=True)
    if not s and all_runs:
        s = {ck: summarise([r for r in all_runs if r["case_kind"] == ck], [x for x in all_singles if x["case_kind"] == ck], gi)
             for ck in cases if any(r["case_kind"] == ck for r in all_runs)}
        write_json(s, d / "summary.json")
    return s


def _append_jsonl(path, rows) -> None:
    with open(path, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _deliberate(ctx: Ctx, cfg: Dict, cases: Dict, sel, vec, n: int):
    """Deliberations and single-agent baselines for the selected (case kind, question) pairs, every condition."""
    runs = []
    for ck, fi in sel:
        c = cases[ck][fi]
        for order in (0, 1):                         # 0: wrong shown as A
            a, b = (c.wrong, c.right) if order == 0 else (c.right, c.wrong)
            for cond in conditions(cfg, ck):
                runs.append({"case_kind": ck, "item": fi, "order": order, "a": a, "b": b,
                             "W_label": "A" if order == 0 else "B", **cond})

    def case(r):
        return cases[r["case_kind"]][r["item"]]

    def member_task(r, i):
        if r["belief"] and i < r["k"]:
            side = "wrong"
        elif r.get("t", 0) and r["k"] <= i < r["k"] + r["t"]:
            side = "right"                                # the truth-teller's copy states the right value
        else:
            side = "shared"
        return CS.task(case(r), side, r["a"], r["b"])

    def item(r, i, msgs, extra=()):
        st = i < r["k"] and r["frac"] > 0                 # members 0..k-1 are the minority
        return Item(msgs, r["a"], r["b"], like=r["W_label"] if st else None, vector=vec if st else None,
                    alpha=ctx.alpha(r["frac"]) if st else 0.0, extra_needles=list(extra))

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
            r = runs[k]
            if r.get("once") and r.get("t", 0) and rnd > 0:   # the truth-teller has said it once; it adds nothing more
                for j in range(r["k"], r["k"] + r["t"]):
                    st[j] = NOTHING_MORE
            runs[k].setdefault("statements", []).append(st)
            last_said[k] = st
            ctxs[k] = DG.advance(ctxs[k], st, NEXT)
        vote(str(rnd + 1))
    # single-model baselines: one agent alone, with the majority's context or the minority's (belief), at each
    # steering strength
    singles = []
    for ck, fi in sel:
        c = cases[ck][fi]
        for order in (0, 1):
            a, b = (c.wrong, c.right) if order == 0 else (c.right, c.wrong)
            W = "A" if order == 0 else "B"
            sides = ["shared"] + (["wrong"] if ck in BELIEF_KINDS + TESTIMONY_KINDS else [])
            for side in sides:
                msgs = [{"role": "user", "content": f"{CS.task(c, side, a, b)}\n\n{CS.vote_q(c)} {DG.VOTE_SUFFIX}"}]
                for fr in [0.0] + [float(x) for x in cfg.get("fracs", [])]:
                    singles.append({"case_kind": ck, "item": fi, "order": order, "frac": fr, "W_label": W,
                                    "context": "belief" if side == "wrong" else "majority",
                                    "it": Item(msgs, a, b, like=W if fr else None, vector=vec if fr else None,
                                               alpha=ctx.alpha(fr))})
    sres = ctx.choose([x["it"] for x in singles], desc="e4 singles", read=False)
    for x, y in zip(singles, sres):
        x["m"] = (1.0 if x["W_label"] == "A" else -1.0) * y["margin"]
        x.pop("it")
    for r in runs:
        r.pop("a"), r.pop("b")
    return runs, singles


def summarise(runs: List[Dict], singles: List[Dict], gi: int) -> Dict:
    out: Dict = {}
    for cond in sorted({r["cond"] for r in runs}):
        rs = [r for r in runs if r["cond"] == cond]
        k = rs[0]["k"]
        t = rs[0].get("t", 0)
        k0 = k + t                                        # first majority member
        by: Dict = {}
        for r in rs:
            by.setdefault(r["item"], []).append(r)
        R = len(rs[0]["votes"])

        def unsteered_m(r, tt):
            return np.mean([v["m"] for i, v in enumerate(r["votes"][tt]) if i >= k0])

        def unsteered_read(r, tt):
            xs = [v["readout_W_minus_R"][gi] for i, v in enumerate(r["votes"][tt]) if i >= k0 and v["readout_W_minus_R"]]
            return np.mean(xs) if xs else np.nan

        def minority_m(r, t):
            return np.mean([r["votes"][t][i]["m"] for i in range(k)])

        out[cond] = {
            "unsteered_margin_by_round": [ST.mean_ci([np.mean([unsteered_m(r, t) for r in v]) for v in by.values()])["mean"]
                                          for t in range(R)],
            "unsteered_shift": ST.mean_ci([np.mean([unsteered_m(r, R - 1) - unsteered_m(r, 0) for r in v])
                                           for v in by.values()]),
            "unsteered_wrong_rate_final": ST.mean_ci(
                [np.mean([np.mean([float(x["m"] > 0) for i, x in enumerate(r["votes"][R - 1]) if i >= k0]) for r in v])
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
            "teller_wrong_rate_final": None if t == 0 else ST.mean_ci(
                [np.mean([np.mean([float(r["votes"][R - 1][i]["m"] > 0) for i in range(k, k0)]) for r in v])
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
