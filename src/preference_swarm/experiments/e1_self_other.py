"""Experiment 1 -- does a model prefer its own answers, and by name, by voice, or by content?

A pool of local models answers the same requests (usable WildChat requests from exp50). A paraphraser OUTSIDE the
pool rewrites every answer in neutral wording, keeping the content. Each pool model then judges pairs of answers to
the same request -- its own against each other model's, plus one pair of two others' -- told who it is in the system
prompt, under four conditions:

  hidden       no author labels
  true         each answer labelled with the model that wrote it
  swapped      the two labels exchanged (its own answer carries the other's name and vice versa)
  paraphrased  the paraphrased versions, no labels (voice removed, content kept)

Both orders. Margins are signed toward the judge's OWN answer (for other-vs-other pairs, toward the first-named
author, which is a position-free baseline near zero). Reading: self-preference in `hidden` that survives
`paraphrased` is about content; one that disappears was voice; `true` minus `hidden` is the effect of the name;
in `swapped`, following the label means preferring the other model's answer.

Judges that have exp50 directions (config `exp50_dirs`) also give the readout of every direction over each
answer's tokens -- does the internal preference agree with the stated choice, and does it favour the own answer?
"""
from __future__ import annotations

import gc
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np

from preference_swarm import data as DT
from preference_swarm import engine as EN
from preference_swarm import model as M
from preference_swarm import stats as ST
from preference_swarm.engine import Item
from preference_swarm.experiments.common import read_jsonl, write_jsonl
from preference_swarm.runlog import REPO_ROOT, write_json

ANSWER_SYSTEM = "You are a helpful assistant. Answer in at most 150 words."
PARA_SYSTEM = "You rewrite text. Reply with the rewritten text only."
PARA = ("Rewrite the following answer in plain, neutral wording. Keep every fact, claim and suggestion, and keep "
        "roughly the same length; change the phrasing and style completely.\n\nAnswer:\n{a}")
JUDGE_SYSTEM = "You are {name}."
JUDGE = ("Here is a request and two answers to it.\n\nRequest: {q}\n\n{block}\n\nWhich answer is better? "
         "Reply with a single letter, A or B, and nothing else.")
CONDITIONS = ("hidden", "true", "swapped", "paraphrased")


def _free() -> None:
    """Release GPU memory once the caller has dropped its references (del hf, tok first)."""
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def _one(text: str) -> str:
    return " ".join(text.split()) or "(empty)"


def answers(cfg: Dict, out: Path, questions: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """{model_name: {qid: answer}}; resumable per model."""
    res = {}
    for m in cfg["models"]:
        f = out / f"answers_{m['name']}.jsonl"
        if f.exists():
            res[m["name"]] = {r["q"]: r["answer"] for r in read_jsonl(f)}
            continue
        hf, tok = M.load(m["id"])
        qs = sorted(questions)
        items = [Item([{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": questions[q]}])
                 for q in qs]
        outs = EN.generate(hf, tok, 1, items, max_new_tokens=int(cfg["answer_tokens"]),
                           temperature=float(cfg.get("temperature", 0.7)), batch_size=int(cfg.get("batch_generate", 32)),
                           seed=11, desc=f"e1 answers {m['name']}")
        res[m["name"]] = {q: _one(a) for q, a in zip(qs, outs)}
        write_jsonl(f, [{"q": q, "answer": res[m["name"]][q]} for q in qs])
        del hf, tok
        _free()
    return res


def paraphrase(cfg: Dict, out: Path, ans: Dict[str, Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    f = out / "paraphrases.jsonl"
    if f.exists():
        res: Dict[str, Dict[str, str]] = {}
        for r in read_jsonl(f):
            res.setdefault(r["model"], {})[r["q"]] = r["text"]
        return res
    hf, tok = M.load(cfg["paraphraser"])
    keys = [(m, q) for m in sorted(ans) for q in sorted(ans[m])]
    items = [Item([{"role": "system", "content": PARA_SYSTEM}, {"role": "user", "content": PARA.format(a=ans[m][q])}])
             for m, q in keys]
    outs = EN.generate(hf, tok, 1, items, max_new_tokens=int(cfg["answer_tokens"]) + 40, temperature=0.7,
                       batch_size=int(cfg.get("batch_generate", 32)), seed=12, desc="e1 paraphrase")
    res = {}
    for (m, q), t in zip(keys, outs):
        res.setdefault(m, {})[q] = _one(t)
    write_jsonl(f, [{"model": m, "q": q, "text": res[m][q]} for m, q in keys])
    del hf, tok
    _free()
    return res


def pairs_for(judge: str, models: List[str], qs: List[str], seed: int) -> List[Dict]:
    """Own vs every other model, plus one other-vs-other pair, per request."""
    rng = random.Random(seed)
    others = [m for m in models if m != judge]
    out = []
    for q in qs:
        for o in others:
            out.append({"q": q, "x": judge, "y": o, "kind": "own"})
        if len(others) >= 2:
            x, y = rng.sample(others, 2)
            out.append({"q": q, "x": x, "y": y, "kind": "others"})
    return out


def judge_items(judge_name: str, p: Dict, cond: str, order: int, questions: Dict[str, str],
                ans: Dict[str, Dict[str, str]], para: Dict[str, Dict[str, str]]) -> Item:
    src = para if cond == "paraphrased" else ans
    tx, ty = src[p["x"]][p["q"]], src[p["y"]][p["q"]]
    lx, ly = p["x"], p["y"]
    if cond == "swapped":
        lx, ly = ly, lx
    if cond in ("true", "swapped"):
        ox, oy = f"[Written by {lx}] {tx}", f"[Written by {ly}] {ty}"
    else:
        ox, oy = tx, ty
    a, b = (ox, oy) if order == 0 else (oy, ox)
    msgs = [{"role": "system", "content": JUDGE_SYSTEM.format(name=judge_name)},
            {"role": "user", "content": JUDGE.format(q=questions[p["q"]], block=M.option_block(a, b))}]
    return Item(msgs, a, b, extra_needles=[tx, ty])


def judge_all(cfg: Dict, out: Path, questions, ans, para) -> List[Dict]:
    names = [m["name"] for m in cfg["models"]]
    rows: List[Dict] = []
    for m in cfg["models"]:
        f = out / f"judge_{m['name']}.jsonl"
        if f.exists():
            rows.extend(read_jsonl(f))
            continue
        hf, tok = M.load(m["id"])
        e50_dir = (cfg.get("exp50_dirs") or {}).get(m["name"])
        V, layer, dnames = None, 1, []
        if e50_dir:
            e = DT.load_exp50(REPO_ROOT / e50_dir)
            if e.model != m["id"] or e.V.shape[1] != int(hf.config.hidden_size):
                raise ValueError(f"exp50 directions in {e50_dir} belong to {e.model} (width {e.V.shape[1]}), "
                                 f"not to judge {m['id']}")
            V, layer, dnames = e.V, e.layer, e.names
        ps = pairs_for(m["name"], names, sorted(questions), int(cfg.get("seed", 0)))
        keys, items = [], []
        for p in ps:
            for cond in CONDITIONS:
                for order in (0, 1):
                    keys.append((p, cond, order))
                    items.append(judge_items(m["name"], p, cond, order, questions, ans, para))
        res = EN.choose(hf, tok, layer, items, V, batch_size=int(cfg.get("batch_choose", 8)),
                        desc=f"e1 judge {m['name']}")
        jr = []
        for (p, cond, order), x in zip(keys, res):
            toward_x = x["margin"] if order == 0 else -x["margin"]
            R = np.asarray(x["readout"]) if x["readout"] is not None else None
            jr.append({"judge": m["name"], **p, "cond": cond, "order": order, "toward_x": toward_x,
                       "readout_x_minus_y": None if R is None else (R[2] - R[3]).tolist(),
                       "directions": dnames if R is not None else None})
        write_jsonl(f, jr)
        rows.extend(jr)
        del hf, tok
        _free()
    return rows


def summarise(rows: List[Dict]) -> Dict:
    s: Dict = {}
    for judge in sorted({r["judge"] for r in rows}):
        js = [r for r in rows if r["judge"] == judge]
        s[judge] = {}
        for kind in ("own", "others"):
            for cond in CONDITIONS:
                by: Dict = {}
                for r in js:
                    if r["kind"] == kind and r["cond"] == cond:
                        by.setdefault((r["q"], r["y"]), []).append(r["toward_x"])
                v = [np.mean(x) for x in by.values() if len(x) == 2]
                s[judge][f"{kind}|{cond}"] = {**ST.mean_ci(v), "p": ST.sign_flip_p(v)}
        own = {}
        for r in js:
            if r["kind"] == "own":
                own.setdefault((r["q"], r["y"], r["cond"]), []).append(r["toward_x"])
        def diff(c1, c2):
            dd = []
            for (q, y, c), v in own.items():
                if c == c1 and (q, y, c2) in own:
                    dd.append(np.mean(v) - np.mean(own[(q, y, c2)]))
            return {**ST.mean_ci(dd), "p": ST.sign_flip_p(dd)}
        s[judge]["name_effect_true_minus_hidden"] = diff("true", "hidden")
        s[judge]["voice_effect_hidden_minus_paraphrased"] = diff("hidden", "paraphrased")
        s[judge]["label_follow_true_minus_swapped"] = diff("true", "swapped")
        rd = [r for r in js if r["readout_x_minus_y"] is not None]
        if rd:
            names = rd[0]["directions"]
            gi = names.index("general_readout")
            hid = [r for r in rd if r["cond"] == "hidden"]
            s[judge]["readout_vs_choice_hidden"] = ST.spearman_perm([r["readout_x_minus_y"][gi] for r in hid],
                                                                     [r["toward_x"] for r in hid])
            s[judge]["readout_own_minus_other_hidden"] = ST.mean_ci(
                [r["readout_x_minus_y"][gi] for r in hid if r["kind"] == "own"])
    return s


def run(cfg: Dict, out: Path) -> Dict:
    out.mkdir(parents=True, exist_ok=True)
    src = cfg["questions_from"]
    e = DT.load_exp50(REPO_ROOT / src) if src != "smoke" else None
    if e is not None:
        names = DT.usable_names(e)
        texts = e.texts
    else:
        texts = {r["name"]: r["text"] for r in read_jsonl(REPO_ROOT / cfg.get("smoke_prompts", "data/smoke_prompts.jsonl"))}
        names = sorted(texts)
    rng = random.Random(int(cfg.get("seed", 0)) + 1)
    qs = rng.sample(names, min(int(cfg["n_questions"]), len(names)))
    questions = {q: texts[q] for q in qs}
    (out / "questions.json").write_text(json.dumps(questions, indent=1))
    ans = answers(cfg, out, questions)
    para = paraphrase(cfg, out, ans)
    rows = judge_all(cfg, out, questions, ans, para)
    s = summarise(rows)
    write_json(s, out / "summary.json")
    return s
