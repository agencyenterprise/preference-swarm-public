"""The exp50 rebuild's direction step (no torch): the ported functions on states with a planted preference
direction, the vector names and order exp50 saved, determinism, and e2's chains without the competence axis."""
from __future__ import annotations

import json

import numpy as np
import yaml

from preference_swarm import data as DT
from preference_swarm import rebuild as RB
from preference_swarm.runlog import REPO_ROOT

EXP50_NAMES = ["general_tercile", "general_readout", "SHUFFLED_tercile_0", "SHUFFLED_tercile_1",
               "SHUFFLED_tercile_2", "RANDOM_dir_0", "RANDOM_dir_1"]


def _planted(n=300, d=64, seed=0):
    rng = np.random.default_rng(seed)
    w = RB.unit(rng.standard_normal(d))
    S = rng.standard_normal((n, d)) * 0.5
    y = rng.standard_normal(n)
    S += np.outer(y, w)
    return S, y, w


def test_directions_recover_the_planted_one_and_controls_do_not():
    S, y, w = _planted()
    b = RB.build_vectors(S, y)
    v = b["vectors"]
    assert list(v) == EXP50_NAMES
    assert all(abs(np.linalg.norm(x) - 1) < 1e-9 for x in v.values())
    assert v["general_tercile"] @ w > 0.9 and v["general_readout"] @ w > 0.9
    assert b["readout_cv_r2"] > 0.5
    assert all(abs(v[f"SHUFFLED_tercile_{i}"] @ w) < 0.5 for i in range(3))


def test_rebuild_is_deterministic():
    S, y, _ = _planted()
    a, b = RB.build_vectors(S, y)["vectors"], RB.build_vectors(S, y)["vectors"]
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_saved_tournament_scores_every_request():
    src = REPO_ROOT / "data" / "exp50_src"
    for m in ("Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"):
        names = [json.loads(l)["name"] for l in open(src / m / "prompts.jsonl") if l.strip()][:1500]
        y = RB.score_vector(names, json.load(open(src / m / "judge_rows.json")))
        rec = json.load(open(src / m / "general_pref.json"))
        assert int((~np.isnan(y)).sum()) == rec["n_prompts"]
        assert abs(float(np.nanstd(y)) - rec["score_sd"]) < 1e-6


def test_chains_without_the_competence_axis(tmp_path):
    from preference_swarm.experiments import e2_negotiate as E2
    from test_flows import FakeCtx
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "smoke.yaml"))
    ctx = FakeCtx(tmp_path, cfg)
    keep = [i for i, n in enumerate(ctx.e50.names) if not n.startswith("axis_")]
    ctx.e50 = DT.Exp50(ctx.e50.model, ctx.e50.layer, ctx.e50.texts, ctx.e50.score,
                       [ctx.e50.names[i] for i in keep], ctx.e50.V[keep])
    ch = E2.chains(ctx, cfg["e2"])
    assert {r["chain"] for r in ch} == {"plain", "like", "random"}
    assert len(ch) == cfg["e2"]["n_chains"] * 3
