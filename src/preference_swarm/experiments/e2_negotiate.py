"""Experiment 2 -- negotiating which task to do, and a question passed along a chain of revisers.

NEGOTIATION. Two agents (the same model, separate contexts) get a pair of WildChat requests whose preference the
model already measured in exp50: H (higher score) and L (lower). Each states a choice alone (pre), they exchange
messages for `rounds` rounds, then each chooses again (post) and each states the pair's joint choice. Arms:

  none         both agents unsteered
  like@f       agent 1's copy of L in its task listing pushed along +general direction at f x residual norm, in
               every forward pass it makes (its messages and its votes); agent 2 is never touched
  random@f     the same push along a random direction saved with exp50 (RANDOM_dir_0)
  shuffled@f   ... along the shuffled-score twin (SHUFFLED_tercile_0)

Every pair runs in both label orders (L shown as A, or as B). Margins are signed toward L. The readouts are every
exp50 direction averaged over each option's tokens in the agent's own context.

Questions answered: does a pair settle on the more preferred request (joint margin toward L < 0 in `none`); does
agent 1's steered preference carry into the joint choice and into agent 2's post choice beyond the controls
(contagion through text alone); and does agent 2's own readout on L rise.

QUESTION CHAIN. A seed request; agent 1 writes a question for a third assistant inspired by it; each next agent in
a chain of `chain_len` revises the previous draft in a fresh context. Chains: all plain; the FIRST reviser steered
along +general (every position) at f; the same with the competence (answerability) axis; a random-direction
control. Each draft is read by an unsteered reader as a request (the exp50 setting) and scored by every direction,
and the reader chooses between the first and last draft (both orders). Does a chain drift toward questions the
model prefers / finds answerable, and does one steered member move the whole chain?
"""
from __future__ import annotations

import random
from typing import Dict, List

import numpy as np

from preference_swarm import data as DT
from preference_swarm import dialogue as DG
from preference_swarm import model as M
from preference_swarm import stats as ST
from preference_swarm.engine import Item
from preference_swarm.experiments.common import Ctx, write_jsonl
from preference_swarm.runlog import write_json

SYSTEM = ("You are one of two assistants who must agree on ONE of two requests to take on together. "
          "Keep each message under 70 words.")
TASK = "Here are two requests.\n\n{block}"
FIRST = "Which should the two of you take on? Give your view in a short message to the other assistant."
NEXT = "Reply to the other assistant in a short message."
PRE_Q = "Which request would you rather take on?"
POST_Q = "Which request do you now want to take on?"
JOINT_Q = "Which request did the two of you agree to take on?"
WRITE_SYSTEM = "You write and improve questions for other AI assistants. Reply with the question only."
WRITE_FIRST = ("Here is something a person once asked an assistant:\n\n\"{seed}\"\n\nInspired by it, write one "
               "question to ask another AI assistant. Reply with the question only.")
REVISE = ("Here is a draft question that will be asked to another AI assistant:\n\n\"{draft}\"\n\nImprove it. "
          "Reply with the improved question only.")
READ_AS_REQUEST = "{q}"
CHOICE_Q = ("Here are two requests.\n\n{block}\n\nWhich would you rather be asked to answer? "
            "Reply with a single letter, A or B, and nothing else.")


def _arm_vec(ctx: Ctx, arm: str) -> np.ndarray:
    return {"like": ctx.vec(ctx.cfg["like_vector"]), "random": ctx.vec("RANDOM_dir_0"),
            "shuffled": ctx.vec("SHUFFLED_tercile_0"), "competence": ctx.vec("axis_competence")}[arm]


def _arms(ctx: Ctx, cfg: Dict) -> List[Dict]:
    out = [{"arm": "none", "kind": None, "frac": 0.0}]
    for f in cfg["fracs"]:
        for k in cfg["controls"]:
            out.append({"arm": f"{k}@{f}", "kind": k, "frac": float(f)})
    return out


def negotiate(ctx: Ctx, cfg: Dict) -> List[Dict]:
    e = ctx.e50
    pairs = DT.task_pairs(e, int(cfg["n_pairs"]), int(ctx.cfg.get("seed", 0)))
    runs = []
    for pi, (h, lo) in enumerate(pairs):
        for order in (0, 1):                       # 0: L shown as A; 1: L shown as B
            a, b = (e.texts[lo], e.texts[h]) if order == 0 else (e.texts[h], e.texts[lo])
            for arm in _arms(ctx, cfg):
                runs.append({"pair": pi, "H": h, "L": lo, "order": order, "a": a, "b": b, **arm,
                             "L_label": "A" if order == 0 else "B",
                             "score_gap": e.score[h] - e.score[lo]})
    n_agents = 2

    def items(run, contexts, question=None):
        out = []
        for i, c in enumerate(contexts):
            steer = run["kind"] is not None and i == 0
            msgs = DG.with_vote(c, question) if question else c
            out.append(Item(msgs, run["a"], run["b"], like=run["L_label"] if steer else None,
                            vector=_arm_vec(ctx, run["kind"]) if steer else None,
                            alpha=ctx.alpha(run["frac"]) if steer else 0.0))
        return out

    ctxs = [DG.start([SYSTEM] * n_agents, TASK.format(block=M.option_block(r["a"], r["b"])), FIRST) for r in runs]

    def vote(question, tag):
        flat = [it for r, c in zip(runs, ctxs) for it in items(r, c, question)]
        res = ctx.choose(flat, desc=f"e2 {tag}")
        for k, r in enumerate(runs):
            r[tag] = [_toward_L(res[k * n_agents + i], r["L_label"]) for i in range(n_agents)]

    vote(PRE_Q, "pre")
    transcripts = [[] for _ in runs]
    for rnd in range(int(cfg["rounds"])):
        flat = [it for r, c in zip(runs, ctxs) for it in items(r, c)]
        said = ctx.generate(flat, seed=1000 + rnd, desc=f"e2 round {rnd + 1}")
        for k in range(len(runs)):
            st = said[k * n_agents:(k + 1) * n_agents]
            transcripts[k].append(st)
            ctxs[k] = DG.advance(ctxs[k], st, NEXT)
    vote(POST_Q, "post")
    vote(JOINT_Q, "joint")
    for r, t in zip(runs, transcripts):
        r["transcript"] = t
        r.pop("a"), r.pop("b")
    return runs


def _toward_L(res: Dict, L_label: str) -> Dict:
    """Margin and readout difference signed toward L (L minus H)."""
    m = res["margin"] if L_label == "A" else -res["margin"]
    rd = None
    if res["readout"] is not None:
        R = np.asarray(res["readout"])
        iL, iH = (0, 1) if L_label == "A" else (1, 0)
        rd = (R[iL] - R[iH]).tolist()
    return {"m": m, "readout_L_minus_H": rd}


def chains(ctx: Ctx, cfg: Dict) -> List[Dict]:
    e = ctx.e50
    rng = random.Random(int(ctx.cfg.get("seed", 0)) + 7)
    seeds = rng.sample(DT.usable_names(e), min(int(cfg["n_chains"]), len(DT.usable_names(e))))
    f = float(cfg["chain_frac"])
    kinds = [("plain", None), ("like", "like"), ("competence", "competence"), ("random", "random")]
    runs = [{"seed": s, "chain": name, "kind": k} for s in seeds for name, k in kinds]
    L = int(cfg["chain_len"])
    drafts: List[List[str]] = [[] for _ in runs]
    for step in range(L):
        its = []
        for r, d in zip(runs, drafts):
            prompt = WRITE_FIRST.format(seed=e.texts[r["seed"]]) if step == 0 else REVISE.format(draft=d[-1])
            steer = r["kind"] is not None and step == 1          # the first REVISER is the steered member
            its.append(Item([{"role": "system", "content": WRITE_SYSTEM}, {"role": "user", "content": prompt}],
                            axis=steer, vector=_arm_vec(ctx, r["kind"]) if steer else None,
                            alpha=ctx.alpha(f) if steer else 0.0))
        out = ctx.generate(its, max_new_tokens=int(cfg.get("chain_tokens", 80)), seed=2000 + step,
                           desc=f"e2 chain step {step + 1}")
        for d, q in zip(drafts, out):
            d.append(_one_line(q))
    # every draft read as a request by an unsteered reader; the readout is over the draft's tokens
    flat = [Item([{"role": "user", "content": READ_AS_REQUEST.format(q=q)}], extra_needles=[q])
            for d in drafts for q in d]
    res = ctx.choose(flat, desc="e2 read drafts")
    for k, r in enumerate(runs):
        r["drafts"] = drafts[k]
        r["readout"] = [res[k * L + s]["readout"][2] for s in range(L)]     # span 2 = the draft (no options)
    # first vs last draft, both orders: + = the last draft preferred
    flat = []
    for d in drafts:
        flat.append(Item([{"role": "user", "content": CHOICE_Q.format(block=M.option_block(d[-1], d[0]))}],
                         d[-1], d[0]))
        flat.append(Item([{"role": "user", "content": CHOICE_Q.format(block=M.option_block(d[0], d[-1]))}],
                         d[0], d[-1]))
    res = ctx.choose(flat, desc="e2 first vs last", read=False)
    for k, r in enumerate(runs):
        r["last_over_first"] = (res[2 * k]["margin"] - res[2 * k + 1]["margin"]) / 2.0
    return runs


def _one_line(q: str) -> str:
    q = q.strip().strip('"').strip()
    return " ".join(q.split())[:400] or "(empty)"


def summarise(ctx: Ctx, neg: List[Dict], ch: List[Dict]) -> Dict:
    names = ctx.dir_names
    gi = names.index(ctx.cfg["like_vector"])
    out: Dict = {"negotiation": {}, "chains": {}}
    for arm in sorted({r["arm"] for r in neg}):
        rs = [r for r in neg if r["arm"] == arm]
        by_pair: Dict = {}
        for r in rs:
            by_pair.setdefault(r["pair"], []).append(r)
        def per_pair(fn):
            return [np.mean([fn(x) for x in v]) for v in by_pair.values()]
        out["negotiation"][arm] = {
            "pre_agent1": ST.mean_ci(per_pair(lambda x: x["pre"][0]["m"])),
            "pre_agent2": ST.mean_ci(per_pair(lambda x: x["pre"][1]["m"])),
            "post_agent1": ST.mean_ci(per_pair(lambda x: x["post"][0]["m"])),
            "post_agent2": ST.mean_ci(per_pair(lambda x: x["post"][1]["m"])),
            "joint": ST.mean_ci(per_pair(lambda x: (x["joint"][0]["m"] + x["joint"][1]["m"]) / 2)),
            "agent2_shift": ST.mean_ci(per_pair(lambda x: x["post"][1]["m"] - x["pre"][1]["m"])),
            "agent2_readout_shift": ST.mean_ci(per_pair(
                lambda x: (x["post"][1]["readout_L_minus_H"][gi] - x["pre"][1]["readout_L_minus_H"][gi])
                if x["post"][1]["readout_L_minus_H"] else np.nan)),
        }
    none = {r["pair"]: r for r in neg if r["arm"] == "none" and r["order"] == 0}
    out["negotiation"]["joint_vs_score_gap_none"] = ST.spearman_perm(
        [none[p]["score_gap"] for p in none],
        [np.mean([(x["joint"][0]["m"] + x["joint"][1]["m"]) / 2 for x in neg if x["arm"] == "none" and x["pair"] == p])
         for p in none])
    # steered minus none, paired by pair: the contagion estimate per arm
    for arm in sorted({r["arm"] for r in neg} - {"none"}):
        d2, dj = [], []
        for p in none:
            a = [x for x in neg if x["arm"] == arm and x["pair"] == p]
            n0 = [x for x in neg if x["arm"] == "none" and x["pair"] == p]
            if a and n0:
                d2.append(np.mean([x["post"][1]["m"] for x in a]) - np.mean([x["post"][1]["m"] for x in n0]))
                dj.append(np.mean([(x["joint"][0]["m"] + x["joint"][1]["m"]) / 2 for x in a])
                          - np.mean([(x["joint"][0]["m"] + x["joint"][1]["m"]) / 2 for x in n0]))
        out["negotiation"][f"{arm}_minus_none"] = {"agent2_post": {**ST.mean_ci(d2), "p": ST.sign_flip_p(d2)},
                                                   "joint": {**ST.mean_ci(dj), "p": ST.sign_flip_p(dj)}}
    ci = names.index("axis_competence") if "axis_competence" in names else None
    for chain in sorted({r["chain"] for r in ch}):
        rs = [r for r in ch if r["chain"] == chain]
        L = len(rs[0]["readout"])
        out["chains"][chain] = {
            "general_by_step": [ST.mean_ci([r["readout"][s][gi] for r in rs])["mean"] for s in range(L)],
            "general_last_minus_first": ST.mean_ci([r["readout"][-1][gi] - r["readout"][0][gi] for r in rs]),
            "competence_last_minus_first": None if ci is None else
            ST.mean_ci([r["readout"][-1][ci] - r["readout"][0][ci] for r in rs]),
            "choice_last_over_first": ST.mean_ci([r["last_over_first"] for r in rs]),
        }
    return out


def run(ctx: Ctx, cfg: Dict) -> Dict:
    d = ctx.sub("e2_negotiate")
    neg = negotiate(ctx, cfg)
    write_jsonl(d / "negotiation.jsonl", neg)
    ch = chains(ctx, cfg)
    write_jsonl(d / "chains.jsonl", ch)
    s = summarise(ctx, neg, ch)
    write_json(s, d / "summary.json")
    return s
