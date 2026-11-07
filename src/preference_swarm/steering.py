"""Per-row steering and span readout on one decoder block's output.

One forward hook on ``blocks[layer]`` (block index, 0-based -- the convention of casper's exp50, whose preference
directions were read from exactly this tensor) does two things in order:

  steer   each batch row gets its own unit vector and strength, added either at EVERY position (prefill and every
          decode step: "axis" steering) or only inside given token spans of the prompt (prefill only: "like"
          steering of one option, the exp51 one-copy-pushed design). Unsteered rows get nothing.
  read    on the prefill pass, the block output (after steering) projected on a fixed set of unit directions and
          averaged over each requested token span, per row.

casper's src/michelangelo/steer.py steers a whole batch with ONE vector; a swarm whose agents differ needs one per
row, which is the new part here. Spans are given in each row's own (unpadded) token coordinates and shifted by the
row's left padding inside the hook.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

Span = Tuple[int, int]


def get_decoder_layers(model):
    """The decoder block ModuleList for Llama/Qwen-style HF models (casper src/vector_sleeper/steering.py)."""
    m = model
    for attr in ("model", "transformer", "gpt_neox"):
        if hasattr(m, attr):
            m = getattr(m, attr)
            break
    if not any(hasattr(m, a) for a in ("layers", "h", "blocks")):
        for attr in ("language_model", "text_model"):
            if hasattr(m, attr):
                inner = getattr(m, attr)
                m = getattr(inner, "model", inner)
                break
    for attr in ("layers", "h", "blocks"):
        if hasattr(m, attr):
            return getattr(m, attr)
    raise AttributeError(f"could not locate decoder layers on {type(model).__name__}")


@dataclass
class Steer:
    """One row's steering: unit vector, strength in residual-stream units, and where.

    spans None = every position including generated tokens; spans given = only those prompt tokens.
    """
    vector: np.ndarray
    alpha: float
    spans: Optional[Sequence[Span]] = None


def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("cannot normalise a zero vector")
    return v / n


def position_mask(lengths: Sequence[int], T: int, rows: Sequence[Optional[Steer]]) -> np.ndarray:
    """(B, T) 0/1 mask of the prefill positions each row is steered at, under left padding to width T."""
    B = len(lengths)
    M = np.zeros((B, T), dtype=np.float32)
    for r in range(B):
        s = rows[r] if r < len(rows) else None
        if s is None or s.alpha == 0.0:
            continue
        off = T - int(lengths[r])
        if s.spans is None:
            M[r, off:] = 1.0
        else:
            for lo, hi in s.spans:
                lo, hi = max(0, int(lo)) + off, min(int(lengths[r]), int(hi)) + off
                if hi > lo:
                    M[r, lo:hi] = 1.0
    return M


def shifted(spans: Sequence[Optional[Span]], length: int, T: int) -> List[Optional[Span]]:
    """A row's spans in padded coordinates."""
    off = T - int(length)
    return [None if s is None else (int(s[0]) + off, int(s[1]) + off) for s in spans]


class BlockHook:
    """State for one batch; see the module docstring."""

    def __init__(self, directions: Optional[np.ndarray] = None) -> None:
        self.D = None if directions is None else np.stack([unit(d) for d in np.asarray(directions)])
        self._D_t = None
        self.lengths: List[int] = []
        self.rows: List[Optional[Steer]] = []
        self.read_spans: List[List[Optional[Span]]] = []
        self.readings: Optional[np.ndarray] = None        # (B, n_spans, k) after a prefill with read spans
        self._prefill_done = False

    def prepare(self, lengths: Sequence[int], rows: Optional[Sequence[Optional[Steer]]] = None,
                read_spans: Optional[Sequence[Sequence[Optional[Span]]]] = None) -> None:
        self.lengths = [int(x) for x in lengths]
        self.rows = list(rows) if rows is not None else [None] * len(self.lengths)
        self.read_spans = [list(x) for x in read_spans] if read_spans is not None else []
        self.readings = None
        self._prefill_done = False

    def __call__(self, module, inputs, output):
        import torch

        h = output[0] if isinstance(output, tuple) else output
        B, T, d = h.shape
        prefill = T > 1 or not self._prefill_done
        if any(s is not None and s.alpha != 0.0 for s in self.rows):
            V = np.zeros((B, d), dtype=np.float32)
            for r, s in enumerate(self.rows[:B]):
                if s is not None and s.alpha != 0.0:
                    V[r] = (float(s.alpha) * unit(s.vector)).astype(np.float32)
            if prefill:
                M = position_mask(self.lengths, T, self.rows)
            else:   # a decode step: only rows steered everywhere keep being steered
                M = np.array([[1.0 if (s is not None and s.alpha != 0.0 and s.spans is None) else 0.0]
                              for s in self.rows[:B]], dtype=np.float32)
            Vt = torch.from_numpy(V).to(h.device, h.dtype)
            Mt = torch.from_numpy(M).to(h.device, h.dtype)
            h = h + Mt[:, :, None] * Vt[:, None, :]
        if prefill and self.D is not None and self.read_spans:
            if self._D_t is None or self._D_t.device != h.device:
                self._D_t = torch.from_numpy(self.D.astype(np.float32)).to(h.device)
            proj = (h.float() @ self._D_t.T).cpu().numpy()          # (B, T, k)
            n = max(len(x) for x in self.read_spans)
            out = np.full((B, n, self.D.shape[0]), np.nan, dtype=np.float64)
            for r in range(min(B, len(self.read_spans))):
                for j, sp in enumerate(shifted(self.read_spans[r], self.lengths[r], T)):
                    if sp is not None and sp[1] > sp[0]:
                        out[r, j] = proj[r, sp[0]:sp[1]].mean(axis=0)
            self.readings = out
        self._prefill_done = True
        if isinstance(output, tuple):
            return (h,) + tuple(output[1:])
        return h


@contextmanager
def attached(model, layer: int, hook: BlockHook) -> Iterator[BlockHook]:
    """Register `hook` on blocks[layer] for the duration of the block."""
    handle = get_decoder_layers(model)[layer].register_forward_hook(hook)
    try:
        yield hook
    finally:
        handle.remove()
