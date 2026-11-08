"""Building the cases and prompts the jury and minority experiments share (2026-11-07 redesign; see facts.Case).

Fact kinds come from facts.make_cases. `opinion` cases are exp50 request pairs judged by the model's own exp50
score: `right` = the request it prefers, `wrong` = the one it prefers less; both sides see the same two requests.
"""
from __future__ import annotations

import re
from typing import List, Optional

from preference_swarm import data as DT
from preference_swarm import facts as FX
from preference_swarm import model as M

OPINION_Q = "Which request would you rather be asked to answer?"
FACT_VOTE_Q = "Which answer is correct?"
OPINION_INTRO = "Here are two requests that might be given to an AI assistant."


def build(ctx, kind: str, n: int, seed: int) -> List[FX.Case]:
    if kind != "opinion":
        return FX.make_cases(kind, n, seed)
    out = []
    for hi, lo in DT.task_pairs(ctx.e50, n, seed):
        out.append(FX.Case("opinion", OPINION_Q, ctx.e50.texts[hi], ctx.e50.texts[lo], "", "", ""))
    return out


def context_for(case: FX.Case, side: str) -> str:
    """What an agent on `side` sees: 'shared' (juror / majority), 'wrong' or 'right' (an advocate of that value)."""
    if side == "shared":
        return case.shared
    if side == "wrong":
        return case.minority
    return case.minority_right or case.shared


def task(case: FX.Case, side: str, a: str, b: str) -> str:
    block = M.option_block(a, b)
    if case.kind == "opinion":
        return f"{OPINION_INTRO}\n\n{block}\n\nQuestion: {OPINION_Q}"
    ctx_text = context_for(case, side)
    head = ctx_text if case.kind.startswith("planet") else f"Passage: {ctx_text}"
    return f"{head}\n\nQuestion: {case.question}\n\n{block}"


def vote_q(case: FX.Case) -> str:
    return OPINION_Q if case.kind == "opinion" else FACT_VOTE_Q


def argue_line(case: FX.Case, label: str) -> str:
    if case.kind == "opinion":
        return f"Argue that request {label} is the better one to be asked."
    return f"Argue that the answer is {label}."


def prefill(case: FX.Case, label: str) -> str:
    if case.kind == "opinion":
        return f"Request {label} is the better one to be asked, because"
    return f"The answer is {label}, because"


def argues_only(case: FX.Case, text: str, target: str, other: str) -> Optional[bool]:
    """Fact kinds: the argument names the target value, not the other one, and does not say 'not <target>'.
    None for opinion cases (the options are whole requests; no string test)."""
    if case.kind == "opinion":
        return None
    g = text.lower()
    t, o = target.lower() in g, other.lower() in g
    return bool(t and not o and not re.search(r"not\\s+" + re.escape(target.lower()), g))
