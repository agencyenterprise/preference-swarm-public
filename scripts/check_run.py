#!/usr/bin/env python
"""Check a finished run directory: every experiment the config lists ran, wrote records of the right shape, and
has readouts; print the headline numbers. --strict exits 1 on any problem.

    python scripts/check_run.py --run outputs/llama8b_<ts> [--strict]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FOLDERS = {"e1": "e1_self_other", "e2": "e2_negotiate", "e3": "e3_jury", "e4": "e4_minority", "e5": "e5_monitor"}


def _f(x):
    return "None" if x is None else f"{x:+.3f}"


def rows(p: Path):
    return [json.loads(l) for l in open(p) if l.strip()] if p.exists() else []


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    run = a.run
    bad = []
    cfg = json.load(open(run / "config.json"))
    tel = json.load(open(run / "telemetry.json"))
    for s in tel.get("stages", []):
        print(f"  stage {s['stage']:5s} {s['status']:6s} {s['seconds'] / 60:7.1f} min  {s.get('error', '')}")
        if s["status"] != "ok":
            bad.append(f"{s['stage']} {s['status']}: {s.get('error')}")
    for x in cfg["experiments"]:
        d = run / FOLDERS[x]
        if not (d / "summary.json").exists():
            bad.append(f"{x}: no summary.json")
            continue
        if x == "e1":
            names = [m["name"] for m in cfg["e1"]["models"]]
            for n in names:
                jr = rows(d / f"judge_{n}.jsonl")
                if not jr:
                    bad.append(f"e1: no judgements by {n}")
                conds = {r["cond"] for r in jr}
                if jr and conds != {"hidden", "true", "swapped", "paraphrased"}:
                    bad.append(f"e1: {n} conditions {sorted(conds)}")
            print(f"  e1: judges {names}")
        if x == "e2":
            neg = rows(d / "negotiation.jsonl")
            c = cfg["e2"]
            want = c["n_pairs"] * 2 * (1 + len(c["fracs"]) * len(c["controls"]))
            if len(neg) != want:
                bad.append(f"e2: {len(neg)} negotiations, want {want}")
            if neg and any(r["post"][1]["readout_L_minus_H"] is None for r in neg):
                bad.append("e2: negotiation votes without readouts")
            if neg and any(len(r["transcript"]) != c["rounds"] for r in neg):
                bad.append("e2: a transcript has the wrong number of rounds")
            ch = rows(d / "chains.jsonl")
            kinds = {r["chain"] for r in ch}
            if not {"plain", "like", "random"} <= kinds:
                bad.append(f"e2: chain kinds {sorted(kinds)}")
            if len(ch) != c["n_chains"] * len(kinds):     # 3 kinds when the competence axis was not rebuilt
                bad.append(f"e2: {len(ch)} chains, want {c['n_chains'] * len(kinds)}")
            if ch and any(any(np.isnan(np.asarray(v, dtype=float)).all() for v in r["readout"]) for r in ch):
                bad.append("e2: a chain draft has no readout")
            s = json.load(open(d / "summary.json"))["negotiation"]
            for k in sorted(s):
                if k.endswith("_minus_none"):
                    v = s[k]["agent2_post"]
                    print(f"  e2 {k}: agent2 post shift toward L {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] "
                          f"p {v['p']}")
            pre = {k: s[k]["pre_agent1"]["mean"] for k in s if isinstance(s[k], dict) and "pre_agent1" in s[k]}
            print(f"  e2 agent 1 pre margin toward L by arm (manipulation check): "
                  + ", ".join(f"{k} {v:+.2f}" for k, v in sorted(pre.items())))
        if x == "e3":
            jr = rows(d / "jury.jsonl")
            c = cfg["e3"]
            kinds = c.get("kinds", ["passage"])
            n_arms = 1 + len(c.get("controls", []))
            want_min = c["n_items"] * len(kinds) * 2 * 2 * len(c["styles"]) * n_arms
            if len(jr) < want_min:
                bad.append(f"e3: {len(jr)} juror readings, want at least {want_min}")
            if jr and sum(r["readout_argument"] is None or all(np.isnan(r["readout_argument"])) for r in jr) > len(jr) * 0.02:
                bad.append("e3: more than 2% of arguments not located for the readout")
            if jr and sum(not r["argument"].strip() for r in jr) > len(jr) * 0.02:
                bad.append("e3: more than 2% empty arguments")
            if jr and "case_kind" in jr[0] and not any(r["arm"] == "other_case" for r in jr):
                bad.append("e3: no other_case (distraction control) readings")
            s = json.load(open(d / "summary.json"))
            for ck, sk in sorted(s.items()):
                if not isinstance(sk, dict) or "compliance" not in sk:      # v1 summaries are flat
                    continue
                comp = [v for kk, v in sk["compliance"].items() if kk.startswith("wrong|") and kk.endswith("|none")]
                for kk in sorted(sk):
                    if kk.startswith("net_persuasion|wrong|"):
                        v = sk[kk]
                        print(f"  e3 {ck} {kk}: {v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] p {v['p']:.3g}")
                fl = [sk[kk]["flip_rate"]["mean"] for kk in sk if kk.startswith("wrong|") and kk.endswith("|none")]
                bf = [sk[kk]["before"]["mean"] for kk in sk if kk.startswith("wrong|") and kk.endswith("|none")]
                print(f"  e3 {ck}: juror before {np.mean(bf):+.2f}, wrong-target flip rate {np.mean(fl):.2f}"
                      + (f", advocates arguing only the wrong value {np.mean(comp):.2f}" if comp else ""))
        if x == "e4":
            dl = rows(d / "deliberations.jsonl")
            c = cfg["e4"]
            if dl and any(len(r["votes"]) != c["rounds"] + 1 or len(r["statements"]) != c["rounds"] for r in dl):
                bad.append("e4: a deliberation has the wrong number of votes or rounds")
            if dl and any(v["readout_incoming"] is None for r in dl for t in r["votes"][1:] for v in t):
                bad.append("e4: votes after a round without incoming-statement readouts")
            inc = [row for r in dl for t in r["votes"][1:] for v in t for row in (v["readout_incoming"] or [])]
            lost = sum(1 for row in inc if not np.all(np.isfinite(np.asarray(row, dtype=float))))
            if inc and lost > 0.05 * len(inc):    # the 2026-11-07 smoke lost about half to unnormalised newlines
                bad.append(f"e4: {lost} of {len(inc)} incoming statements not located for the readout")
            kinds = c.get("kinds", ["passage"])
            for ck in kinds:
                nk = len({(r["item"], r["order"], r["cond"]) for r in dl if r.get("case_kind", "passage") == ck})
                if nk < c["n_items"] * 2 * 2:
                    bad.append(f"e4: case kind {ck} has {nk} deliberations")
            s = json.load(open(d / "summary.json"))
            prog = s.pop("_progress", None)          # e4 runs in chunks; a stopped run says how far it got
            if prog:
                print(f"  e4 progress: {prog['questions_done']} of {prog['questions_total']} questions")
            nested = all(isinstance(v, dict) and all(isinstance(w, dict) for w in v.values()) for v in s.values()) \
                and not any("jury_wrong_majority_final" in v for v in s.values())
            for ck, sk in (sorted(s.items()) if nested else [("", s)]):
                for k in sorted(sk):
                    v = sk[k]
                    if "jury_wrong_majority_final" in v:
                        mn = v.get("minority_wrong_rate_final")
                        mn = "-" if mn is None else f"{mn['mean']:.2f}"
                        print(f"  e4 {ck} {k}: wrong majority {v['jury_wrong_majority_final']['mean']:.2f}, "
                              f"unsteered wrong {v.get('unsteered_wrong_rate_final', {}).get('mean', float('nan')):.2f}, "
                              f"minority wrong {mn}, "
                              f"unsteered shift {v['unsteered_shift']['mean']:+.3f}")
                    elif "wrong_rate" in v:
                        print(f"  e4 {ck} {k}: wrong rate {v['wrong_rate']['mean']:.2f}")
        if x == "e5":
            s = json.load(open(d / "summary.json"))
            for part in sorted(s):
                if isinstance(s[part], dict) and "n" in s[part]:
                    g0 = s[part].get("general_readout") or {}
                    if s[part]["n"] > 10 and g0.get("rho") is None:   # the second 2026-11-07 smoke: constant predictor
                        bad.append(f"e5 {part}: the general_readout predictor has no variance (n {s[part]['n']})")
                    nc = s[part].get("null_comparison") or {}
                    for nm in ("general_readout", "SHUFFLED_tercile_0", "RANDOM_dir_0"):
                        g = s[part].get(nm) or {}
                        pn = (nc.get("directions") or {}).get(nm, {}).get("p_null")
                        print(f"  e5 {part} {nm}: n {s[part]['n']}, raw rho {_f(g.get('rho'))}, partial rho "
                              f"{_f(g.get('partial_rho'))}, p against {nc.get('n_null')} null directions {_f(pn)}")
                    if cfg.get("null_directions") and not nc.get("n_null"):
                        bad.append(f"e5 {part}: no null directions to compare against")
    cal = run / "calibration.json"
    if cal.exists():
        c = json.load(open(cal))
        for f in c["fracs"]:
            print(f"  calibrate push {f}: like {_f(c[f'like@{f}']['mean'])}, random {_f(c[f'random@{f}']['mean'])}")
    for b in bad:
        print(f"  PROBLEM {b}")
    print(f"[check] {run}: {'ok' if not bad else f'{len(bad)} problem(s)'}")
    return 1 if (bad and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
