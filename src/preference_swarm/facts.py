"""Invented-fact items for the jury experiments.

Each item is a short passage of made-up facts about made-up entities, a question about one of them, the answer the
passage gives, and a wrong answer that is ANOTHER entity's value for the same attribute -- so both options appear
in the passage, both are equally fluent, and nothing about the answer can come from memory. (Written for this
repo; casper's src/binding_psi/jeopardy_nonce.py uses the same idea with a different format.)
"""
from __future__ import annotations

import random
from typing import Dict, List, NamedTuple

SYLLABLES = ("ka", "ren", "vo", "li", "ther", "mo", "sa", "dun", "pe", "lor", "vi", "tan", "qu", "el", "bra", "nis",
             "go", "rim", "ta", "fen", "ul", "dor", "ce", "mar")
KINDS = ("guild", "festival", "observatory", "river", "library", "orchard", "lighthouse", "quarry")
ATTRIBUTES = {
    "year": ("was founded in", lambda r: str(r.randrange(1612, 1964))),
    "colour": ("has the colour", lambda r: r.choice(("amber", "teal", "crimson", "ochre", "violet", "slate", "jade",
                                                    "ivory", "copper", "indigo"))),
    "members": ("counts this many members:", lambda r: str(r.randrange(14, 980))),
    "town": ("is based in the town of", lambda r: _name(r, 2).title()),
}
QUESTIONS = {
    "year": "In what year was {e} founded?",
    "colour": "What colour does {e} have?",
    "members": "How many members does {e} count?",
    "town": "In what town is {e} based?",
}


class Fact(NamedTuple):
    passage: str
    question: str
    right: str
    wrong: str
    entity: str
    attribute: str


def _name(r: random.Random, n: int) -> str:
    return "".join(r.choice(SYLLABLES) for _ in range(n))


def make(n: int, seed: int, n_entities: int = 6) -> List[Fact]:
    """n items, deterministic in seed; entity names and values distinct within an item."""
    r = random.Random(seed)
    out: List[Fact] = []
    while len(out) < n:
        attr = r.choice(sorted(ATTRIBUTES))
        verb, draw = ATTRIBUTES[attr]
        ents, vals = [], []
        while len(ents) < n_entities:
            e = f"the {_name(r, 3).title()} {r.choice(KINDS)}"
            v = draw(r)
            if e in ents or v in vals:
                continue
            ents.append(e)
            vals.append(v)
        lines = [f"{e[0].upper() + e[1:]} {verb} {v}." for e, v in zip(ents, vals)]
        k = r.randrange(n_entities)
        j = r.choice([i for i in range(n_entities) if i != k])
        out.append(Fact(" ".join(lines), QUESTIONS[attr].format(e=ents[k]), vals[k], vals[j], ents[k], attr))
    return out


def as_dict(f: Fact) -> Dict[str, str]:
    return f._asdict()
