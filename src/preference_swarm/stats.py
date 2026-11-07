"""Small statistics, numpy only: means with bootstrap intervals, Spearman, permutation tests."""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

N_BOOT = 2000
N_PERM = 2000


def mean_ci(x: Sequence[float], seed: int = 0, n_boot: int = N_BOOT) -> Dict[str, Optional[float]]:
    """Mean and a percentile bootstrap 95% interval (n, NaNs dropped)."""
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    if a.size == 1:
        return {"mean": float(a[0]), "lo": float(a[0]), "hi": float(a[0]), "n": 1}
    rng = np.random.default_rng(seed)
    boots = a[rng.integers(0, a.size, (n_boot, a.size))].mean(axis=1)
    return {"mean": float(a.mean()), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)),
            "n": int(a.size)}


def rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty_like(x)
    r[order] = np.arange(x.size, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < xs.size:
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return None
    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_perm(x: Sequence[float], y: Sequence[float], seed: int = 0, n_perm: int = N_PERM
                  ) -> Dict[str, Optional[float]]:
    """Spearman rho and a two-sided permutation p."""
    rho = spearman(x, y)
    if rho is None:
        return {"rho": None, "p": None, "n": 0}
    a, b = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b)
    ra, rb = rank(a[keep]), rank(b[keep])
    rng = np.random.default_rng(seed)
    null = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(n_perm)])
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))
    return {"rho": rho, "p": p, "n": int(keep.sum())}


def sign_flip_p(x: Sequence[float], seed: int = 0, n_perm: int = N_PERM) -> Optional[float]:
    """Two-sided p for mean(x) != 0 by random sign flips (paired differences)."""
    a = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size < 2:
        return None
    rng = np.random.default_rng(seed)
    obs = abs(a.mean())
    null = np.abs((a[None, :] * rng.choice((-1.0, 1.0), (n_perm, a.size))).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))
