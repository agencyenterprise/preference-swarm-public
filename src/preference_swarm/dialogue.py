"""Agent contexts for a deliberation among n agents, built turn by turn.

Agent i sees: its system prompt, the task as a user turn, then for each round its OWN statement as an assistant
turn and the OTHER agents' statements of that round collected into the next user turn. Votes are read by appending
one more user turn that asks for a single letter. All agents of a deliberation speak simultaneously within a round,
so nobody sees a statement from the round it is answering in.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Sequence

Message = Dict[str, str]
VOTE_SUFFIX = "Reply with a single letter, A or B, and nothing else."


def clean(text: str) -> str:
    """One line, single spaces: how every statement enters a context, so a readout can find it again by its text
    (the 2026-11-07 smoke lost half its e4 readouts to statements containing newlines)."""
    return " ".join(str(text).split())


def label(i: int) -> str:
    return f"Assistant {i + 1}"


def start(systems: Sequence[str], task: str, first_prompt: str) -> List[List[Message]]:
    """One context per agent; systems may differ (e.g. a petition instruction for some)."""
    return [[{"role": "system", "content": s}, {"role": "user", "content": f"{task}\n\n{first_prompt}"}]
            for s in systems]


def others_turn(i: int, statements: Sequence[str], prompt: str, n_agents: int) -> str:
    lines = [f"{label(j)} said: \"{clean(statements[j])}\"" for j in range(n_agents) if j != i]
    return "\n\n".join(lines) + f"\n\n{prompt}"


def advance(contexts: List[List[Message]], statements: Sequence[str], next_prompt: str) -> List[List[Message]]:
    """Append each agent's own statement and the others' statements; returns new contexts."""
    n = len(contexts)
    if len(statements) != n:
        raise ValueError(f"{len(statements)} statements for {n} agents")
    out = []
    for i, c in enumerate(contexts):
        c2 = copy.deepcopy(c)
        c2.append({"role": "assistant", "content": clean(statements[i])})
        c2.append({"role": "user", "content": others_turn(i, statements, next_prompt, n)})
        out.append(c2)
    return out


def with_vote(context: List[Message], question: str) -> List[Message]:
    """The context plus a final user turn asking for a letter.

    The last user turn already carries the others' statements; the vote question is appended to it so the
    conversation keeps strict user/assistant alternation (some templates reject two user turns in a row).
    """
    c = copy.deepcopy(context)
    if c and c[-1]["role"] == "user":
        c[-1] = {"role": "user", "content": c[-1]["content"] + f"\n\n{question} {VOTE_SUFFIX}"}
    else:
        c.append({"role": "user", "content": f"{question} {VOTE_SUFFIX}"})
    return c
