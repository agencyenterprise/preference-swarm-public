"""The exp50 inputs: requests, each request's preference score, and the preference directions.

data/exp50/<Model>/ holds four files copied from casper's outputs/wellbeing_found/exp50_general (scripts/
prepare_data.sh does the copy on the box): prompts.jsonl (the 1,500 WildChat requests), judge_rows.json (every A/B
match the model judged), general_pref.json (model id and layer), vectors.npz (the directions: general_readout,
general_tercile, three shuffled-score twins, two random directions, the nine axes, combined_430).

The score is exp50's own: each pair judged in both orders, the order effect removed within the pair, a request's
score the mean margin over its opponents (casper src/wellbeing_found/prompt_pref.py paired_margin / scores).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Dict, List, NamedTuple, Sequence, Tuple

import numpy as np

# Requests kept for the swarm tasks: short enough to quote in a conversation, and none of the role-play or sexual
# requests WildChat is full of (they would turn every transcript into a refusal study).
MIN_CHARS, MAX_CHARS = 30, 360
EXCLUDE = re.compile(r"\b(role ?play|rp\b|sexy|sex|erotic|nsfw|fetish|grope|kiss|naked|nude|horny|lewd|seduc|"
                     r"porn|breast|butt|feet|foot fetish|vore|hypno|diaper|age regression|tickl|spank|bdsm|"
                     r"slave|torture|kill|suicide|rape|onlyfans|futa|hentai|waifu)", re.I)
# Controls saved beside the real directions in vectors.npz.
CONTROL_PREFIXES = ("SHUFFLED_", "RANDOM_")


class Exp50(NamedTuple):
    model: str
    layer: int
    texts: Dict[str, str]
    score: Dict[str, float]
    names: List[str]          # direction names
    V: np.ndarray             # (k, d), rows unit


def paired_margin(rows: Sequence[Dict]) -> List[Tuple[str, str, float]]:
    """(x, y, theta) with theta > 0 = x preferred; the two presentation orders averaged."""
    by: Dict[Tuple[str, str], Dict[str, float]] = {}
    for r in rows:
        key = tuple(sorted((str(r["a"]), str(r["b"]))))
        by.setdefault(key, {})[f'{r["a"]}|{r["b"]}'] = float(r["logit"])
    out = []
    for (x, y), seen in sorted(by.items()):
        fwd, rev = seen.get(f"{x}|{y}"), seen.get(f"{y}|{x}")
        if fwd is None or rev is None:
            continue
        out.append((x, y, (fwd - rev) / 2.0))
    return out


def scores(judgements: Sequence[Tuple[str, str, float]]) -> Dict[str, float]:
    tot: Dict[str, float] = {}
    n: Dict[str, int] = {}
    for x, y, th in judgements:
        tot[x] = tot.get(x, 0.0) + th
        n[x] = n.get(x, 0) + 1
        tot[y] = tot.get(y, 0.0) - th
        n[y] = n.get(y, 0) + 1
    return {k: tot[k] / max(1, n[k]) for k in tot}


def load_exp50(d: Path) -> Exp50:
    d = Path(d)
    for f in ("prompts.jsonl", "judge_rows.json", "general_pref.json", "vectors.npz"):
        if not (d / f).exists():
            raise FileNotFoundError(f"{d / f} missing: run scripts/prepare_data.sh on the box first")
    texts = {}
    for line in open(d / "prompts.jsonl"):
        if line.strip():
            r = json.loads(line)
            texts[r["name"]] = r["text"]
    score = scores(paired_margin(json.load(open(d / "judge_rows.json"))))
    rep = json.load(open(d / "general_pref.json"))
    z = np.load(d / "vectors.npz")
    names = [str(n) for n in z["names"]]
    V = np.asarray(z["V"], dtype=np.float64)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    return Exp50(str(rep["model"]), int(rep["layer"]), texts, score, names, V)


def random_exp50(texts: Dict[str, str], d: int, layer: int, seed: int = 0) -> Exp50:
    """A stand-in for the smoke test on a model without exp50 directions: random unit directions, random scores."""
    rng = np.random.default_rng(seed)
    names = ["general_readout", "general_tercile", "SHUFFLED_tercile_0", "RANDOM_dir_0", "axis_competence",
             "axis_refusal"]
    V = rng.standard_normal((len(names), d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    score = {k: float(rng.standard_normal()) for k in texts}
    return Exp50("smoke", layer, dict(texts), score, names, V)


def usable(text: str) -> bool:
    t = text.strip()
    return MIN_CHARS <= len(t) <= MAX_CHARS and not EXCLUDE.search(t) and "\n\n" not in t


def usable_names(e: Exp50) -> List[str]:
    return sorted([k for k, t in e.texts.items() if usable(t) and k in e.score], key=lambda k: int(k[1:])
                  if k[1:].isdigit() else k)


def direction(e: Exp50, name: str) -> np.ndarray:
    return e.V[e.names.index(name)]


def task_pairs(e: Exp50, n: int, seed: int) -> List[Tuple[str, str]]:
    """n request pairs spread over the score gap: pairs drawn across score quintiles, each pair (preferred,
    less preferred) by the model's own exp50 score."""
    names = usable_names(e)
    rng = random.Random(seed)
    ranked = sorted(names, key=lambda k: e.score[k])
    q = max(1, len(ranked) // 5)
    bins = [ranked[i * q:(i + 1) * q] for i in range(5)]
    out, seen = [], set()
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        i, j = rng.randrange(5), rng.randrange(5)
        x, y = rng.choice(bins[i]), rng.choice(bins[j])
        if x == y or (x, y) in seen or (y, x) in seen:
            continue
        seen.add((x, y))
        hi, lo = (x, y) if e.score[x] >= e.score[y] else (y, x)
        out.append((hi, lo))
    return out
