"""Rebuilding exp50's direction files from its saved tournament, without casper.

The swarm needs exp50's vectors.npz (the general preference direction and its controls) and states.npz (the
request states, for the on-manifold null directions). Both came from a box that no longer exists. The tournament
that defines the score (judge_rows.json, ~29 min a model) survives in data/exp50_src/; the states are one forward
pass a request (~40 s a model). This module redoes the direction step of casper's scripts/general_preference.py
`run` with the same functions, ported verbatim from casper src/wellbeing_found/general_pref.py and combined.py,
and the same seeds, so the rebuilt directions are those of exp50 up to the states' numerical noise.

Not rebuilt: the nine axis directions and combined_430. They came from exp48's 430-prompt sweep, whose states
and axis_directions.npz were on the same box. The swarm uses them only as comparisons (e5) and one chain arm (e2,
skipped when the competence axis is absent).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from preference_swarm.data import paired_margin, scores

SEED = 0                       # casper general_preference.py --seed default
SHUFFLED_TWINS = 3
RANDOM_DIRS = 2
RANDOM_SEED_OFFSET = 7         # cw.random_sets(..., a.seed + 7, cov_from=S)
N_PCS_READOUT = 50
N_PCS_RANDOM = 256


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + 1e-12)


def tercile_direction(S: np.ndarray, y: np.ndarray, frac: float = 1 / 3) -> np.ndarray:
    """Mean state of the top `frac` by score minus the bottom `frac`, unit length."""
    order = np.argsort(y)
    k = max(1, int(round(len(y) * frac)))
    return unit(S[order[-k:]].mean(0) - S[order[:k]].mean(0))


def readout_direction(S: np.ndarray, y: np.ndarray, n_pcs: int = N_PCS_READOUT, alphas=(0.1, 1, 10, 100, 1000),
                      folds: int = 5, seed: int = 0) -> Dict[str, object]:
    """Ridge of the standardised score on the states' top principal components (penalty by cross-validated R2),
    mapped back to one residual-stream direction."""
    C = S - S.mean(0)
    _, sv, vt = np.linalg.svd(C, full_matrices=False)
    # components with ~zero variance (fewer requests than n_pcs, e.g. the layer-sweep smoke's 40) would dominate the
    # back-mapping through 1/scale and leave a direction with no spread over the requests; drop them. exp50's own fits
    # (1,500 requests, 50 components) never reach this, so they are unchanged.
    k = min(n_pcs, int(np.sum(sv > sv[0] * 1e-6)))
    Z = C @ vt[:k].T
    Z = Z / (Z.std(0) + 1e-12)
    ys = (y - y.mean()) / (y.std() + 1e-12)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    best = None
    for al in alphas:
        pred = np.empty(len(y))
        for f in range(folds):
            te = idx[f::folds]
            tr = np.setdiff1d(idx, te)
            A = Z[tr]
            beta = np.linalg.solve(A.T @ A + al * np.eye(k), A.T @ ys[tr])
            pred[te] = Z[te] @ beta
        r2 = float(1 - ((ys - pred) ** 2).sum() / ((ys - ys.mean()) ** 2).sum())
        if best is None or r2 > best[1]:
            best = (al, r2)
    beta = np.linalg.solve(Z.T @ Z + best[0] * np.eye(k), Z.T @ ys)
    scale = (C @ vt[:k].T).std(0) + 1e-12
    w = (beta / scale) @ vt[:k]
    return {"direction": unit(w), "alpha": best[0], "cv_r2": best[1], "n_pcs": k}


def random_sets(d: int, k: int, n_sets: int, seed: int, cov_from: Optional[np.ndarray] = None,
                n_pcs: int = N_PCS_RANDOM) -> List[np.ndarray]:
    """n_sets random sets of k unit directions, isotropic or drawn from the states' top-n_pcs covariance."""
    rng = np.random.default_rng(seed)
    if cov_from is None:
        return [np.stack([unit(rng.normal(size=d)) for _ in range(k)]) for _ in range(n_sets)]
    C = cov_from - cov_from.mean(0)
    _, sv, vt = np.linalg.svd(C, full_matrices=False)
    r = min(n_pcs, len(sv))
    scale, basis = sv[:r] / np.sqrt(len(C)), vt[:r]
    return [np.stack([unit((rng.normal(size=r) * scale) @ basis) for _ in range(k)]) for _ in range(n_sets)]


def score_vector(names: List[str], judge_rows: List[Dict]) -> np.ndarray:
    sc = scores(paired_margin(judge_rows))
    return np.array([sc.get(n, np.nan) for n in names], dtype=np.float64)


def build_vectors(S: np.ndarray, y: np.ndarray, seed: int = SEED) -> Dict:
    """The direction step of casper general_preference.py `run`, in its order (the rng draws depend on it)."""
    terc = tercile_direction(S, y)
    ro = readout_direction(S, y)
    rng = np.random.default_rng(seed)
    vecs = {"general_tercile": terc, "general_readout": ro["direction"]}
    for i in range(SHUFFLED_TWINS):
        vecs[f"SHUFFLED_tercile_{i}"] = tercile_direction(S, rng.permutation(y))
    for i, u in enumerate(random_sets(S.shape[1], 1, RANDOM_DIRS, seed + RANDOM_SEED_OFFSET, cov_from=S)):
        vecs[f"RANDOM_dir_{i}"] = u[0]
    return {"vectors": vecs, "readout_cv_r2": ro["cv_r2"], "readout_alpha": ro["alpha"],
            "cos_tercile_readout": float(terc @ ro["direction"])}
