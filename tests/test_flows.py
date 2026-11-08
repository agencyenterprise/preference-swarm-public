"""The experiments' orchestration end to end with a fake model context (no torch): rounds, votes, arms, orders,
readout indexing, summaries, and the run checker on the written records."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from preference_swarm import data as DT
from preference_swarm.experiments import e2_negotiate as E2
from preference_swarm.experiments import e3_jury as E3
from preference_swarm.experiments import e4_minority as E4
from preference_swarm.experiments import e5_monitor as E5
from preference_swarm.runlog import REPO_ROOT


class FakeCtx:
    def __init__(self, out: Path, cfg: dict):
        texts = {json.loads(l)["name"]: json.loads(l)["text"] for l in open(REPO_ROOT / "data" / "smoke_prompts.jsonl")}
        self.e50 = DT.random_exp50(texts, 16, 4)
        nc = cfg.get("null_directions") or {}
        if nc:                                    # as common.open_ctx does (the smoke has no states: all isotropic)
            self.e50 = DT.null_directions(self.e50, 16, int(nc["cov"]), int(nc["iso"]), 0, None)
        self.cfg, self.out, self.scale = cfg, out, 10.0
        self.rng = np.random.default_rng(0)
        self.calls = {"choose": 0, "generate": 0}

    def vec(self, name):
        return DT.direction(self.e50, name)

    @property
    def V(self):
        return self.e50.V

    @property
    def dir_names(self):
        return self.e50.names

    def alpha(self, f):
        return float(f) * self.scale

    def choose(self, items, desc="", read=True):
        self.calls["choose"] += len(items)
        for it in items:                       # every like-steered item must name an option and carry a vector
            assert it.like in (None, "A", "B") and (it.like is None or it.vector is not None)
        k = len(self.V)
        return [{"margin": float(self.rng.standard_normal()),
                 "readout": self.rng.standard_normal((2 + len(it.extra_needles), k)).tolist() if read else None,
                 "n_tokens": 10} for it in items]

    def generate(self, items, max_new_tokens=None, seed=0, desc=""):
        self.calls["generate"] += len(items)
        return [f"statement {seed} {i} about something" for i in range(len(items))]

    def sub(self, name):
        p = self.out / name
        p.mkdir(parents=True, exist_ok=True)
        return p


def test_smoke_config_flows_and_checker(tmp_path):
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "smoke.yaml"))
    ctx = FakeCtx(tmp_path, cfg)
    s2 = E2.run(ctx, cfg["e2"])
    s3 = E3.run(ctx, cfg["e3"])
    s4 = E4.run(ctx, cfg["e4"])
    s5 = E5.run(ctx, {})
    assert "none" in s2["negotiation"] and "plain" in s2["chains"] and "like" in s2["chains"]
    assert set(s3) == set(cfg["e3"]["kinds"])
    assert all(any(k.startswith("arm_minus_none|") for k in s3[ck]) for ck in s3)
    assert all("net_persuasion|wrong|plain" in s3[ck] for ck in s3)
    assert set(s4) == set(cfg["e4"]["kinds"])
    assert "k1@belief" in s4["planet_known"] and "single|belief@0.0" in s4["planet_known"]
    assert "k1@belief" not in s4["opinion"] and "single|majority@0.0" in s4["opinion"]
    jury = [json.loads(l) for l in open(tmp_path / "e3_jury" / "jury.jsonl")]
    assert any(r["arm"] == "other_case" for r in jury)
    assert all(r["argues_target"] is None for r in jury if r["case_kind"] == "opinion")
    assert s5["jury"]["n"] > 0 and s5["minority"]["n"] > 0
    neg = [json.loads(l) for l in open(tmp_path / "e2_negotiate" / "negotiation.jsonl")]
    c = cfg["e2"]
    assert len(neg) == c["n_pairs"] * 2 * (1 + len(c["fracs"]) * len(c["controls"]))
    assert all(len(r["transcript"]) == c["rounds"] and len(r["pre"]) == 2 for r in neg)
    dl = [json.loads(l) for l in open(tmp_path / "e4_minority" / "deliberations.jsonl")]
    assert {r["case_kind"] for r in dl} == set(cfg["e4"]["kinds"])
    assert all(len(r["votes"]) == cfg["e4"]["rounds"] + 1 for r in dl)
    assert all(len(v["readout_incoming"]) == cfg["e4"]["n_agents"] - 1 for r in dl for t in r["votes"][1:] for v in t)
    # the run checker on these records (e1 left out: it needs real models)
    cfg_run = dict(cfg, experiments=["e2", "e3", "e4", "e5"])
    (tmp_path / "config.json").write_text(json.dumps(cfg_run))
    (tmp_path / "telemetry.json").write_text(json.dumps({"stages": [{"stage": "e2", "status": "ok", "seconds": 1.0}]}))
    import check_run
    assert check_run.main(["--run", str(tmp_path), "--strict"]) == 0


def test_e2_steers_only_agent_one(tmp_path):
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "smoke.yaml"))
    ctx = FakeCtx(tmp_path, cfg)
    seen = []
    orig = ctx.choose

    def spy(items, desc="", read=True):
        if "pre" in desc:
            seen.extend(items)
        return orig(items, desc, read)

    ctx.choose = spy
    E2.negotiate(ctx, cfg["e2"])
    agent2 = seen[1::2]
    agent1 = seen[0::2]
    assert all(it.vector is None and it.alpha == 0.0 for it in agent2)
    assert any(it.vector is not None for it in agent1) and any(it.vector is None for it in agent1)


def test_calibration_runs_on_the_fake_context(tmp_path):
    from preference_swarm.experiments import common
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "smoke.yaml"))
    ctx = FakeCtx(tmp_path, cfg)
    out = common.calibrate(ctx, cfg["calibrate"])
    assert out["n_pairs"] == cfg["calibrate"]["n"]
    for f in cfg["calibrate"]["fracs"]:
        assert out[f"like@{float(f)}"]["n"] == cfg["calibrate"]["n"]
    assert (tmp_path / "calibration.json").exists()


def test_v2_minority_sees_the_wrong_entry_and_advocates_are_prefilled(tmp_path):
    from preference_swarm import facts as FX
    from preference_swarm.experiments import cases as CS
    cfg = yaml.safe_load(open(REPO_ROOT / "configs" / "smoke.yaml"))
    ctx = FakeCtx(tmp_path, cfg)
    seen = []
    orig = ctx.generate

    def spy(items, max_new_tokens=None, seed=0, desc=""):
        seen.append((desc, items))
        return orig(items, max_new_tokens, seed, desc)

    ctx.generate = spy
    E4.run(ctx, dict(cfg["e4"], kinds=["planet_known"], minority=[1], fracs=[]))
    c0 = FX.make_cases("planet_known", cfg["e4"]["n_items"], int(cfg.get("seed", 0)) + 4)[0]
    n = cfg["e4"]["n_agents"]
    conds = [c["cond"] for c in E4.conditions(dict(cfg["e4"], minority=[1], fracs=[]), "planet_known")]
    first_round = seen[0][1]
    ib = conds.index("k1@belief")                    # item 0, order 0 runs come first, conditions in order
    members = first_round[ib * n:(ib + 1) * n]
    entry = [l for l in c0.minority.splitlines() if c0.wrong in l][0]
    assert entry in members[0].messages[1]["content"]
    assert all(entry not in m.messages[1]["content"] for m in members[1:])
    seen.clear()
    E3.run(ctx, dict(cfg["e3"], kinds=["planet_known"]))
    adv = seen[0][1]
    assert all(it.prefill.startswith("The answer is ") for it in adv)
    assert all(c0.wrong in it.messages[1]["content"] for it in adv[:4] if "A. " + c0.wrong in it.messages[1]["content"])
