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


# ---------------------------------------------------------------------------------------------------------------
# Cases: one question with the context each kind of agent sees (2026-11-07 redesign).
#
# The first run's items stated the answer in every agent's passage, so jurors sat at ~13 logits of certainty, the
# advocates told to argue the wrong answer mostly refused (Llama: 456 of 4,800 argued only the wrong one), and a
# steered minority never once voted wrong. A Case separates what the majority/juror sees (`shared`) from what the
# advocate/minority sees (`minority`), and the kinds below differ in that split:
#
#   passage       the original: both see the passage that states the right answer (kept for comparison)
#   contested     both see a passage with TWO conflicting records for the queried entity (no stated answer wins)
#   planet_known  a database of newly discovered facts about an invented planet; the majority's entry gives the
#                 right value, the minority's copy gives the target value (a sincere, misinformed minority)
#   planet_blank  the majority's database has no entry for the queried entity; the minority's has the target value
#   opinion       two exp50 requests, "which would you rather answer?"; no right answer (built in experiments from
#                 exp50 data: `wrong` = the request the model itself prefers less)
#
# `right` is the value the majority's context supports (contested: the first-drawn record, arbitrary by design;
# planet_blank: an unmentioned other value); `wrong` is the value the minority holds or is pushed toward.
# ---------------------------------------------------------------------------------------------------------------
CASE_KINDS = ("passage", "contested", "planet_known", "planet_blank")
PLANET_KINDS = {                     # entity kinds that make sense for each attribute
    "height": ("mountain", "peak", "plateau", "ridge"),
    "colour": ("sea", "glacier", "canyon", "forest", "desert", "lake"),
    "age": ("canyon", "crater", "sea", "mountain", "glacier", "basin"),
    "region": ("city", "river", "forest", "crater", "harbour", "lake"),
}
PLANET_ATTRIBUTES = {
    "height": ("rises to a height of", lambda r: f"{r.randrange(1200, 9900):,} metres"),
    "colour": ("has the colour", ATTRIBUTES["colour"][1]),
    "age": ("formed this many million years ago:", lambda r: str(r.randrange(3, 980))),
    "region": ("lies in the region of", lambda r: _name(r, 2).title()),
}
PLANET_QUESTIONS = {
    "height": "How high does {e} rise?",
    "colour": "What colour does {e} have?",
    "age": "How many million years ago did {e} form?",
    "region": "In what region does {e} lie?",
}
PLANET_HEADER = ("Here is a database of newly discovered facts about the planet {planet}. {planet} is a newly "
                 "surveyed planet; treat every entry in this database as true.\n\n")


class Case(NamedTuple):
    kind: str
    question: str
    right: str
    wrong: str
    shared: str        # context of the majority / the juror
    minority: str      # context of the advocate / the minority members who hold `wrong`
    minority_right: str = ""   # context of an advocate of `right` (planet_blank: a database stating it); "" = shared


def _entities(r: random.Random, n: int, kinds, draw, fmt) -> tuple:
    ents, vals = [], []
    while len(ents) < n:
        e = fmt(r)
        v = draw(r)
        if e in ents or v in vals:
            continue
        ents.append(e)
        vals.append(v)
    return ents, vals


def make_cases(kind: str, n: int, seed: int, n_entities: int = 6) -> List[Case]:
    """n cases of one kind, deterministic in (kind, seed)."""
    if kind == "passage":
        return [Case(kind, f.question, f.right, f.wrong, f.passage, f.passage) for f in make(n, seed, n_entities)]
    r = random.Random(f"{kind}:{seed}")
    out: List[Case] = []
    while len(out) < n:
        if kind == "contested":
            attr = r.choice(sorted(ATTRIBUTES))
            verb, draw = ATTRIBUTES[attr]
            ents, vals = _entities(r, n_entities + 1, KINDS, draw,
                                   lambda q: f"the {_name(q, 3).title()} {q.choice(KINDS)}")
            k = r.randrange(n_entities)
            right, wrong = vals[k], vals[n_entities]          # two records for entity k; the extra value is unused
            lines = []
            for i in range(n_entities):
                E = ents[i][0].upper() + ents[i][1:]
                if i == k:
                    recs = [f"One survey records that {ents[i]} {verb} {right}.",
                            f"A second survey records that {ents[i]} {verb} {wrong}."]
                    r.shuffle(recs)
                    lines.extend(recs)
                else:
                    lines.append(f"{E} {verb} {vals[i]}.")
            p = " ".join(lines)
            out.append(Case(kind, QUESTIONS[attr].format(e=ents[k]), right, wrong, p, p))
        elif kind in ("planet_known", "planet_blank"):
            planet = _name(r, 3).title()
            attr = r.choice(sorted(PLANET_ATTRIBUTES))
            verb, draw = PLANET_ATTRIBUTES[attr]
            ents, vals = _entities(r, n_entities + 1, PLANET_KINDS[attr], draw,
                                   lambda q: f"the {_name(q, 3).title()} {q.choice(PLANET_KINDS[attr])}")
            k = r.randrange(n_entities)
            right, wrong = vals[k], vals[n_entities]          # the extra value: a value no other entry uses

            def db(val_k):
                rows = []
                for i in range(n_entities):
                    if i == k and val_k is None:
                        continue
                    v = val_k if i == k else vals[i]
                    E = ents[i][0].upper() + ents[i][1:]
                    rows.append(f"- {E} {verb} {v}.")
                return PLANET_HEADER.format(planet=planet) + "\n".join(rows)

            shared = db(right) if kind == "planet_known" else db(None)
            out.append(Case(kind, PLANET_QUESTIONS[attr].format(e=ents[k]), right, wrong, shared, db(wrong),
                            db(right)))
        else:
            raise ValueError(f"unknown case kind {kind!r} (opinion cases are built from exp50 data)")
    return out
