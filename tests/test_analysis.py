"""Summaries and the monitor analysis on synthetic records with known answers; configs; the repo."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import yaml

from preference_swarm.experiments import e1_self_other as E1
from preference_swarm.experiments import e3_jury as E3
from preference_swarm.experiments import e3_jury_v1 as E3v1
from preference_swarm.experiments import e4_minority_v1 as E4v1
from preference_swarm.experiments import e4_minority as E4
from preference_swarm.experiments import e5_monitor as E5
from preference_swarm.runlog import REPO_ROOT

NAMES = ["general_readout", "SHUFFLED_tercile_0", "RANDOM_dir_0"]


def test_e5_jury_finds_a_planted_monitor_and_not_a_null():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        g = rng.standard_normal()
        rows.append({"shift": 2.0 * g + 0.1 * rng.standard_normal(), "argument_words": int(rng.integers(20, 60)),
                     "readout_argument": [g, rng.standard_normal(), rng.standard_normal()],
                     "before": float(rng.standard_normal()), "target": "wrong" if i % 2 else "right"})
    t = E5.jury_table(rows, NAMES)
    assert t["general_readout"]["rho"] > 0.9 and t["general_readout"]["p"] < 0.01
    assert t["general_readout"]["partial_rho"] > 0.9 and t["general_readout"]["partial_p"] < 0.01
    assert abs(t["RANDOM_dir_0"]["rho"]) < 0.25 and abs(t["RANDOM_dir_0"]["partial_rho"]) < 0.25


def _run(k, n, rounds, rng, planted=True):
    votes = [[{"m": float(rng.choice((-1.0, 1.0)))} for _ in range(n)]]
    for t in range(rounds):
        prev = votes[-1]
        cur = []
        for i in range(n):
            inc = [[rng.standard_normal(), rng.standard_normal(), rng.standard_normal()] for j in range(n) if j != i]
            push = sum(np.sign(prev[j]["m"]) * inc[q][0] for q, j in enumerate([j for j in range(n) if j != i]))
            cur.append({"m": prev[i]["m"] + (push if planted else 0.0) + 0.01 * rng.standard_normal(),
                        "readout_incoming": inc})
        votes.append(cur)
    return {"k": k, "votes": votes}


def test_e5_minority_finds_a_planted_monitor():
    rng = np.random.default_rng(1)
    runs = [_run(1, 5, 3, rng) for _ in range(40)]
    t = E5.minority_table(runs, NAMES)
    assert t["n"] == 40 * 3 * 4                     # unsteered members (4) x rounds (3)
    assert t["general_readout"]["rho"] > 0.8
    assert abs(t["SHUFFLED_tercile_0"]["rho"]) < 0.3


def test_e4_conditions_and_summary_shape():
    cfg = {"minority": [1, 2], "fracs": [0.25, 0.5]}
    conds = E4v1.conditions(cfg)
    assert [c["cond"] for c in conds][0] == "k0" and conds[-1]["petition"] and len(conds) == 1 + 4 + 1
    rng = np.random.default_rng(2)
    runs = []
    for item in range(4):
        for c in conds:
            r = _run(c["k"], 5, 2, rng, planted=False)
            for t in r["votes"]:
                for v in t:
                    v["readout_W_minus_R"] = [0.1, 0.0, 0.0]
            runs.append({"item": item, **c, **r})
    singles = [{"item": i, "frac": f, "m": float(rng.standard_normal())} for i in range(4) for f in (0.0, 0.25)]
    s = E4v1.summarise(runs, singles, 0)
    assert set(s) >= {"k0", "k1@0.25", "k2@0.5+petition", "single@0.0", "single@0.25"}
    assert len(s["k0"]["unsteered_margin_by_round"]) == 3 and s["k0"]["steered_margin_final"] is None


def test_e3_summary_keys():
    rows = []
    for item in range(5):
        for target in ("wrong", "right"):
            for style in ("plain", "majority"):
                for arm in ("none", "like@0.5"):
                    for order in (0, 1):
                        rows.append({"item": item, "target": target, "style": style, "arm": arm, "order": order,
                                     "shift": 1.0 if style == "majority" else 0.0, "before": -1.0, "after": 0.5})
    s = E3v1.summarise(rows)
    assert s["style_minus_plain|majority"]["mean"] == 1.0
    assert s["wrong|plain|none"]["flip_rate"]["mean"] == 1.0
    assert "arm_minus_none|like@0.5" in s


def test_e1_pairs_and_summary():
    ps = E1.pairs_for("m1", ["m1", "m2", "m3"], ["q1", "q2"], 0)
    assert sum(p["kind"] == "own" for p in ps) == 4 and sum(p["kind"] == "others" for p in ps) == 2
    rows = []
    for p in ps:
        for cond in E1.CONDITIONS:
            for order in (0, 1):
                rows.append({"judge": "m1", **p, "cond": cond, "order": order,
                             "toward_x": 1.0 if (p["kind"] == "own" and cond != "paraphrased") else 0.0,
                             "readout_x_minus_y": None, "directions": None})
    s = E1.summarise(rows)["m1"]
    assert s["own|hidden"]["mean"] == 1.0 and s["own|paraphrased"]["mean"] == 0.0
    assert s["voice_effect_hidden_minus_paraphrased"]["mean"] == 1.0


def test_e1_swapped_labels_swap_the_names_only():
    q = {"q1": "a question"}
    ans = {"m1": {"q1": "own answer"}, "m2": {"q1": "other answer"}}
    it = E1.judge_items("m1", {"q": "q1", "x": "m1", "y": "m2"}, "swapped", 0, q, ans, ans)
    assert it.a == "[Written by m2] own answer" and it.b == "[Written by m1] other answer"
    it = E1.judge_items("m1", {"q": "q1", "x": "m1", "y": "m2"}, "true", 1, q, ans, ans)
    assert it.a == "[Written by m2] other answer"


# ---------------------------------------------------------------- configs
@pytest.mark.parametrize("name", ["llama8b", "qwen7b", "e1", "smoke"])
def test_configs_load(name):
    c = yaml.safe_load(open(REPO_ROOT / "configs" / f"{name}.yaml"))
    assert c["run_name"] and c["experiments"]
    for x in c["experiments"]:
        assert x in ("e1", "e2", "e3", "e4", "e5")
        if x in ("e1", "e2", "e3", "e4"):
            assert x in c
    if "e5" in c["experiments"]:
        assert "e3" in c["experiments"] or "e4" in c["experiments"]
        assert c["experiments"].index("e5") == len(c["experiments"]) - 1      # e5 reads e3 / e4: runs last


def test_model_configs_point_at_matching_exp50():
    for name, model in (("llama8b", "meta-llama/Llama-3.1-8B-Instruct"), ("qwen7b", "Qwen/Qwen2.5-7B-Instruct")):
        c = yaml.safe_load(open(REPO_ROOT / "configs" / f"{name}.yaml"))
        assert c["model"] == model and c["exp50_dir"].endswith(model.split("/")[1])
        assert c["like_vector"] == "general_readout"


# ---------------------------------------------------------------- repo
def _code_lines(path: Path):
    return [l for l in path.read_text().splitlines() if not l.lstrip().startswith("#")]


def test_every_module_compiles():
    files = list((REPO_ROOT / "src").rglob("*.py")) + list((REPO_ROOT / "scripts").glob("*.py")) + \
        list((REPO_ROOT / "tests").glob("*.py"))
    assert len(files) > 15
    for f in files:
        compile(f.read_text(), str(f), "exec")


def test_no_casper_imports():
    for f in list((REPO_ROOT / "src").rglob("*.py")) + list((REPO_ROOT / "scripts").glob("*.py")):
        assert not re.search(r"^\s*(from|import)\s+src\b", f.read_text(), re.M), f.name


def test_scripts_never_swallow_output_and_sync_only_collects():
    for f in (REPO_ROOT / "scripts").glob("*.sh"):
        body = "\n".join(_code_lines(f))
        assert not re.search(r"\|\s*(tail|head|grep|less)\b", body), f.name
    sync = "\n".join(_code_lines(REPO_ROOT / "scripts" / "sync_outputs.sh"))
    for bad in ("--delete", "--remove-source-files"):
        assert bad not in sync
    assert not re.search(r"(^|[;&|]\s*|\s)rm(\s|$)", sync, re.M)
    assert "/workspace/preference-swarm/outputs" in sync


def test_prepare_only_reads_casper():
    body = "\n".join(_code_lines(REPO_ROOT / "scripts" / "prepare_data.sh"))
    for line in body.splitlines():
        if line.strip().startswith("cp "):
            assert line.rstrip().split()[-1].strip('"').startswith("data/exp50/"), line
    assert ">" not in "".join(l for l in body.splitlines() if "$SRC" in l and "echo" not in l and "say" not in l)


# ---------------------------------------------------------------- the 2026-11-07 smoke confound
def test_partial_spearman_removes_a_confound_and_keeps_a_real_effect():
    from preference_swarm import stats as ST
    rng = np.random.default_rng(3)
    stance = rng.choice((-3.0, -1.0, 1.0, 3.0), 400)
    y = -stance + 0.3 * rng.standard_normal(400)
    x_confounded = 2.0 * stance + 0.3 * rng.standard_normal(400)      # an offset times stance, no own signal
    raw = ST.spearman_perm(x_confounded, y, n_perm=200)
    part = ST.partial_spearman(x_confounded, y, [stance], n_perm=200)
    assert raw["rho"] < -0.8 and abs(part["rho"]) < 0.15
    real = rng.standard_normal(400)
    y2 = -stance + real
    part2 = ST.partial_spearman(real + 0.5 * stance, y2, [stance], n_perm=200)
    assert part2["rho"] > 0.6 and part2["p"] < 0.01


def _confounded_runs(rng, n_runs=60, n=5, rounds=3, real=False):
    """Votes that regress to the mean and follow the crowd's stance; readouts = a per-direction OFFSET plus noise
    (what made every direction 'predict' in the smoke). With real=True the shift also follows the CENTRED readout
    of direction 0."""
    offsets = np.array([1.5, -2.0, 0.7])
    runs = []
    for _ in range(n_runs):
        base = rng.standard_normal()
        votes = [[{"m": base + 0.5 * rng.standard_normal()} for _ in range(n)]]
        for t in range(rounds):
            prev = votes[-1]
            cur = []
            for i in range(n):
                others = [j for j in range(n) if j != i]
                R = offsets[None, :] + rng.standard_normal((n - 1, 3))
                st = np.array([np.sign(prev[j]["m"]) for j in others])
                m = 0.5 * prev[i]["m"] + 0.2 * st.sum() + 0.3 * rng.standard_normal()
                if real:
                    m += 1.0 * float(st @ (R[:, 0] - R[:, 0].mean()))
                cur.append({"m": m, "readout_incoming": R.tolist()})
            votes.append(cur)
        runs.append({"k": 0, "votes": votes})
    return runs


def test_minority_table_is_not_fooled_by_offsets():
    rng = np.random.default_rng(4)
    t = E5.minority_table(_confounded_runs(rng), NAMES)
    for nm in NAMES:                                   # no direction carries signal of its own
        assert t[nm]["partial_p"] > 0.01 and abs(t[nm]["partial_rho"]) < 0.15, (nm, t[nm])
    assert abs(t["net_stance"]["rho"]) > 0.2           # the confound is there to be fooled by


def test_minority_table_finds_a_real_centred_effect():
    rng = np.random.default_rng(5)
    t = E5.minority_table(_confounded_runs(rng, real=True), NAMES)
    # the planted effect follows the WITHIN-reading centred readout; the (global) predictor still finds it clearly
    assert t["general_readout"]["partial_rho"] > 0.25 and t["general_readout"]["partial_p"] < 0.01
    assert abs(t["RANDOM_dir_0"]["partial_rho"]) < 0.15



def test_minority_table_works_when_every_sender_agrees():
    """All senders on one side (one model mostly agrees with itself); the receiver moves more after statements read
    more favourably along direction 0. Centring within each reading would make the predictor identically zero."""
    rng = np.random.default_rng(6)
    runs = []
    for _ in range(80):
        votes = [[{"m": 1.0 + 0.1 * rng.random()} for _ in range(5)]]       # everyone leans the same way
        for t in range(2):
            cur = []
            for i in range(5):
                R = np.array([0.8, -1.0, 0.3])[None, :] + rng.standard_normal((4, 3))
                cur.append({"m": votes[-1][i]["m"] + 0.5 * float(R[:, 0].sum()) * 0.3 + 0.05 * rng.standard_normal(),
                            "readout_incoming": R.tolist()})
            votes.append(cur)
        runs.append({"k": 0, "votes": votes})
    t = E5.minority_table(runs, NAMES)
    assert t["general_readout"]["rho"] is not None
    assert t["general_readout"]["partial_rho"] > 0.5 and t["general_readout"]["partial_p"] < 0.01
    assert abs(t["RANDOM_dir_0"]["partial_rho"]) < 0.2



def test_null_directions_shape_and_null_compare():
    from preference_swarm import data as DT
    e = DT.random_exp50({"w0": "a request that is long enough to use"}, 12, 4)
    S = np.random.default_rng(0).standard_normal((50, 12)) @ np.diag(np.r_[np.ones(3) * 10, np.ones(9) * 0.1])
    e2 = DT.null_directions(e, 12, 5, 2, 0, S)
    assert e2.names[-7:] == [f"NULL_cov_{i}" for i in range(5)] + ["NULL_iso_0", "NULL_iso_1"]
    assert np.allclose(np.linalg.norm(e2.V, axis=1), 1.0)
    cov_mass = np.mean([np.sum(e2.V[len(e.names) + i, :3] ** 2) for i in range(5)])
    assert cov_mass > 0.9                       # drawn where the states vary (the first three axes)
    names = ["general_readout"] + [f"NULL_cov_{i}" for i in range(19)]
    table = {"general_readout": {"partial_rho": 0.5}}
    table.update({f"NULL_cov_{i}": {"partial_rho": 0.01 * i - 0.1} for i in range(19)})
    nc = E5.null_compare(table, names)
    assert nc["n_null"] == 19 and nc["directions"]["general_readout"]["p_null"] == 1 / 20
    table["general_readout"]["partial_rho"] = 0.0
    assert E5.null_compare(table, names)["directions"]["general_readout"]["p_null"] > 0.5


# -- the 2026-11-07 redesign -----------------------------------------------------------------------------------------
def test_v2_e4_conditions_by_kind():
    cfg = {"minority": [1, 2], "fracs": [0.25, 0.5]}
    planet = [c["cond"] for c in E4.conditions(cfg, "planet_known")]
    assert planet == ["k0", "k1@belief", "k1@steer0.25", "k1@steer0.5", "k1@belief+steer0.5",
                      "k2@belief", "k2@steer0.25", "k2@steer0.5", "k2@belief+steer0.5", "k2@belief+steer0.5+stubborn"]
    assert [c["cond"] for c in E4.conditions(cfg, "opinion")][-1] == "k2@steer0.5+stubborn"
    llama = [c["cond"] for c in E4.conditions({"minority": [1, 2], "fracs": []}, "planet_blank")]
    assert llama == ["k0", "k1@belief", "k2@belief", "k2@belief+stubborn"]


def test_v2_e3_net_persuasion_subtracts_the_distraction_control():
    rows = []
    for item in range(6):
        for target in ("wrong", "right"):
            for order in (0, 1):
                for arm, shift, ok in (("none", 3.0, True), ("other_case", 2.0, None)):
                    rows.append({"case_kind": "planet_known", "item": item, "target": target, "order": order,
                                 "style": "plain", "arm": arm, "shift": shift, "before": -1.0, "after": -1.0 + shift,
                                 "argues_target": ok})
    s = E3.summarise(rows)["planet_known"]
    assert s["net_persuasion|wrong|plain"]["mean"] == 1.0
    assert s["wrong|plain|none"]["flip_rate"]["mean"] == 1.0
    assert s["compliance"]["wrong|plain|none"] == 1.0
