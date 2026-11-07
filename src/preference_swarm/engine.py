"""Batched forward passes and generation with per-row steering and span readouts.

Every experiment goes through two calls:

  choose(...)    a forward pass per item whose context ends by asking for a single letter, A or B; returns the
                 log-probability margin logp(A) - logp(B) (letters summed over the forms a model emits, casper
                 src/wellbeing_steer/preference.py) and the preference readouts over each option's tokens and any
                 extra spans.
  generate(...)  one sampled turn per item, same steering.

An item's steering is described in TEXT terms (which option to push, or the whole agent) and resolved to token
spans here, after the chat template has rendered the context, so callers never handle token offsets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm.auto import tqdm

from preference_swarm import model as M
from preference_swarm.steering import BlockHook, Steer, attached

OPTION_LABELS = ("A", "B")
OPTION_FORMS = ("{}", " {}", "{}.", "**{}")
MAX_TOKENS = 6144


@dataclass
class Item:
    """One agent context.

    messages      the conversation; the chat template's generation prompt is appended
    a, b          the two option texts as listed in the task (needed for like-steering and option readouts)
    like          None, "A" or "B": push that option's tokens along `vector` (prefill only)
    axis          True: push every position (prompt and generated tail) along `vector`
    vector/alpha  the push; alpha in residual-norm units already scaled by the caller
    extra         extra character spans of the rendered text to read (filled by `extra_needles`)
    extra_needles strings whose LAST occurrence in the rendered text is read (e.g. an incoming message)
    """
    messages: List[Dict[str, str]]
    a: Optional[str] = None
    b: Optional[str] = None
    like: Optional[str] = None
    axis: bool = False
    vector: Optional[np.ndarray] = None
    alpha: float = 0.0
    extra_needles: List[str] = field(default_factory=list)


def option_token_ids(tok) -> Dict[str, List[int]]:
    """First tokens of each letter's forms, minus any token the two letters share.

    A form like "**A" is split by Llama's tokenizer into "**" + "A", so its first token is the same for both letters
    and says nothing about the choice; such tokens are dropped from both sides (the 2026-11-07 smoke stopped on
    exactly this: A [32, 334, 362], B [33, 334, 426], 334 = "**").
    """
    raw: Dict[str, set] = {}
    for label in OPTION_LABELS:
        ids = set()
        for form in OPTION_FORMS:
            enc = tok.encode(form.format(label), add_special_tokens=False)
            if enc:
                ids.add(int(enc[0]))
        raw[label] = ids
    shared = raw["A"] & raw["B"]
    out = {k: sorted(v - shared) for k, v in raw.items()}
    if not out["A"] or not out["B"]:
        raise ValueError(f"no letter-specific first token left after dropping shared ones: {raw}")
    return out


def margin_from_logprobs(lp: np.ndarray, ids: Dict[str, List[int]]) -> float:
    """logsumexp over A's forms minus logsumexp over B's forms; + = A."""
    def lse(x):
        m = np.max(x)
        return m + np.log(np.exp(x - m).sum())
    return float(lse(lp[ids["A"]]) - lse(lp[ids["B"]]))


def _encode(tok, it: Item):
    text = M.render(tok, it.messages)
    ids, offs = M.encode(tok, text)
    if len(ids) > MAX_TOKENS:
        raise ValueError(f"context of {len(ids)} tokens exceeds {MAX_TOKENS}")
    char_spans: List[Optional[Tuple[int, int]]] = [None, None]
    if it.a is not None and it.b is not None:
        sa, sb = M.block_option_spans(text, it.a, it.b)
        char_spans = [sa, sb]
    for nd in it.extra_needles:
        nd = nd.strip()
        i = text.rfind(nd) if nd else -1
        char_spans.append(None if i < 0 else (i, i + len(nd)))
    tspans = M.char_to_token_spans(offs, char_spans, text)
    steer = None
    if it.vector is not None and it.alpha != 0.0:
        if it.axis:
            steer = Steer(it.vector, it.alpha, None)
        elif it.like in OPTION_LABELS:
            sp = tspans[OPTION_LABELS.index(it.like)]
            if sp is None:
                raise ValueError("like-steering an option whose tokens were not found")
            steer = Steer(it.vector, it.alpha, [sp])
    return ids, tspans, steer


def _batches(n: int, size: int):
    for i in range(0, n, size):
        yield range(i, min(n, i + size))


def _pad(tok, seqs: Sequence[List[int]]):
    import torch

    T = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    ids = torch.full((len(seqs), T), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), T), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, T - len(s):] = torch.tensor(s, dtype=torch.long)
        att[r, T - len(s):] = 1
    return ids, att


def choose(hf, tok, layer: int, items: Sequence[Item], directions: Optional[np.ndarray] = None,
           batch_size: int = 8, desc: str = "choose") -> List[Dict]:
    """Margin and readouts per item. readouts[j] = (n_spans, k) with span 0 = option A, 1 = option B, then extras."""
    import torch

    ids_map = option_token_ids(tok)
    enc = [_encode(tok, it) for it in items]
    hook = BlockHook(directions)
    out: List[Dict] = [None] * len(items)   # type: ignore[list-item]
    order = sorted(range(len(items)), key=lambda i: len(enc[i][0]))   # similar lengths together: less padding
    with attached(hf, layer, hook), torch.no_grad():
        for rng in tqdm(list(_batches(len(order), batch_size)), desc=desc, leave=False):
            idx = [order[i] for i in rng]
            seqs = [enc[i][0] for i in idx]
            hook.prepare([len(s) for s in seqs], [enc[i][2] for i in idx], [enc[i][1] for i in idx])
            x, att = _pad(tok, seqs)
            try:    # only the last position is read; full logits for 16 x 1500 tokens x 128k vocab is ~6 GB
                logits = hf(input_ids=x.to(hf.device), attention_mask=att.to(hf.device),
                            logits_to_keep=1).logits[:, -1, :]
            except TypeError:
                logits = hf(input_ids=x.to(hf.device), attention_mask=att.to(hf.device)).logits[:, -1, :]
            lp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            for r, i in enumerate(idx):
                read = None if hook.readings is None else hook.readings[r]
                out[i] = {"margin": margin_from_logprobs(lp[r], ids_map),
                          "readout": None if read is None else read.tolist(),
                          "n_tokens": len(enc[i][0])}
    return out


def generate(hf, tok, layer: int, items: Sequence[Item], max_new_tokens: int = 160, temperature: float = 1.0,
             top_p: float = 0.95, batch_size: int = 32, seed: int = 0, desc: str = "generate") -> List[str]:
    """One sampled reply per item, steering applied as described by each item."""
    import torch

    enc = [_encode(tok, it) for it in items]
    hook = BlockHook(None)
    out: List[str] = [""] * len(items)
    order = sorted(range(len(items)), key=lambda i: len(enc[i][0]))
    torch.manual_seed(seed)
    with attached(hf, layer, hook), torch.no_grad():
        for rng in tqdm(list(_batches(len(order), batch_size)), desc=desc, leave=False):
            idx = [order[i] for i in rng]
            seqs = [enc[i][0] for i in idx]
            hook.prepare([len(s) for s in seqs], [enc[i][2] for i in idx], None)
            x, att = _pad(tok, seqs)
            gen = hf.generate(input_ids=x.to(hf.device), attention_mask=att.to(hf.device),
                              max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                              temperature=temperature if temperature > 0 else None,
                              top_p=top_p if temperature > 0 else None, pad_token_id=tok.pad_token_id)
            new = gen[:, x.shape[1]:]
            for r, i in enumerate(idx):
                out[i] = tok.decode(new[r], skip_special_tokens=True).strip()
    return out


def residual_scale(hf, tok, layer: int, texts: Sequence[str], batch_size: int = 8) -> float:
    """Mean per-token L2 norm of blocks[layer]'s output over chat-rendered `texts`: the unit for alpha."""
    import torch

    from preference_swarm.steering import get_decoder_layers

    norms: List[float] = []
    cur: Dict[str, object] = {}

    def grab(_m, _i, o):
        cur["h"] = (o[0] if isinstance(o, tuple) else o).detach()

    handle = get_decoder_layers(hf)[layer].register_forward_hook(grab)
    try:
        with torch.no_grad():
            for rng in _batches(len(texts), batch_size):
                seqs = [M.encode(tok, M.render(tok, [{"role": "user", "content": texts[i]}]))[0] for i in rng]
                x, att = _pad(tok, seqs)
                hf(input_ids=x.to(hf.device), attention_mask=att.to(hf.device))
                h = cur["h"].float()
                n = torch.linalg.vector_norm(h, dim=-1)
                m = att.to(n.device).bool()
                T = m.shape[1]
                for r, s in enumerate(seqs):          # drop each row's first real token: the attention sink,
                    m[r, T - len(s)] = False          # whose norm is far above the rest in Llama/Qwen
                norms.extend(n[m].cpu().tolist())
    finally:
        handle.remove()
    if not norms:
        raise RuntimeError("residual_scale saw no tokens")
    return float(np.mean(norms))
