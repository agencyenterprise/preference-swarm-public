"""Loading a local model and rendering chat text.

Loading follows casper's src/michelangelo/local.py (left padding for batched generation, `dtype` with the
`torch_dtype` fallback); rendering goes through the chat template as a STRING and is then tokenised, the repo
pattern that avoids apply_chat_template returning a BatchEncoding on some transformers versions.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

LOG = logging.getLogger("preference_swarm.model")

DEFAULT_DTYPE = "bfloat16"
# Qwen3 thinks before answering unless told not to; every other template ignores the switch.
TEMPLATE_KWARGS = {"enable_thinking": False}

Message = Dict[str, str]


def load(model_id: str, dtype: str = DEFAULT_DTYPE):
    """Model and tokenizer, eval mode, left-padded."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype), device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=getattr(torch, dtype), device_map="auto")
    model.eval()
    LOG.info("loaded %s (%s)", model_id, dtype)
    return model, tok


def render(tok, messages: Sequence[Message], add_generation_prompt: bool = True) -> str:
    """The conversation through the chat template, as text."""
    try:
        return tok.apply_chat_template(list(messages), add_generation_prompt=add_generation_prompt, tokenize=False,
                                       **TEMPLATE_KWARGS)
    except (TypeError, ValueError):
        return tok.apply_chat_template(list(messages), add_generation_prompt=add_generation_prompt, tokenize=False)


def encode(tok, text: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Token ids and character offsets of rendered text (no special tokens added: the template has them)."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offs = [tuple(o) for o in enc["offset_mapping"]]
    if ids and isinstance(ids[0], (list, tuple)):
        ids, offs = list(ids[0]), [tuple(o) for o in offs[0]]
    return ids, offs


def char_to_token_spans(offsets: Sequence[Tuple[int, int]], char_spans: Sequence[Optional[Tuple[int, int]]]
                        ) -> List[Optional[Tuple[int, int]]]:
    """Map character spans to half-open token spans by STRICT containment (casper prompt_pref.echo_spans rule).

    A token straddling a span's edge also carries the neighbouring text (a label, a newline), so it is left
    out. None in, or no token strictly inside, gives None.
    """
    out: List[Optional[Tuple[int, int]]] = []
    for cs in char_spans:
        if cs is None:
            out.append(None)
            continue
        lo, hi = cs
        idx = [i for i, (st, en) in enumerate(offsets) if st >= lo and en <= hi and en > st]
        out.append((idx[0], idx[-1] + 1) if idx else None)
    return out


OPTION_BLOCK = "A. {a}\n\nB. {b}"


def option_block(a: str, b: str) -> str:
    """How two options are listed in every prompt of this repo."""
    return OPTION_BLOCK.format(a=a.strip(), b=b.strip())


def block_option_spans(text: str, a: str, b: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Character spans of options A and B inside the FIRST listing of their block in `text`.

    Agents may quote an option later in a conversation; the steered or read copy is always the one in the task
    listing, found as the whole block (both options together), which cannot be confused with a quotation of one.
    """
    block = option_block(a, b)
    i = text.find(block)
    if i < 0:
        raise ValueError("the option block is not in the rendered text (template mangled it?)")
    a_lo = i + len("A. ")
    b_lo = i + block.index("\n\nB. ") + len("\n\nB. ")
    return (a_lo, a_lo + len(a.strip())), (b_lo, b_lo + len(b.strip()))
