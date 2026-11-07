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
            if len(ch) != c["n_chains"] * 4:
                bad.append(f"e2: {len(ch)} chains, want {c['n_chains'] * 4}")
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
            want = c["n_items"] * 2 * 2 * len(c["styles"]) * (1 + len(c["controls"]))
            if len(jr) != want:
                bad.append(f"e3: {len(jr)} juror readings, want {want}")
            if jr and sum(r["readout_argument"] is None or all(np.isnan(r["readout_argument"])) for r in jr) > len(jr) * 0.02:
                bad.append("e3: more than 2% of arguments not located for the readout")
            if jr and sum(not r["argument"].strip() for r in jr) > len(jr) * 0.02:
                bad.append("e3: more than 2% empty arguments")
            s = json.load(open(d / "summary.json"))
            for k in sorted(s):
                if k.startswith("wrong|") and k.endswith("|none"):
                    print(f"  e3 {k}: shift {s[k]['shift']['mean']:+.3f}, flips {s[k]['flip_rate']['mean']:.2f}")
        if x == "e4":
            dl = rows(d / "deliberations.jsonl")
            c = cfg["e4"]
            n_cond = 1 + len(c["minority"]) * len(c["fracs"]) + 1
            if len(dl) != c["n_items"] * 2 * n_cond:
                bad.append(f"e4: {len(dl)} deliberations, want {c['n_items'] * 2 * n_cond}")
            if dl and any(len(r["votes"]) != c["rounds"] + 1 or len(r["statements"]) != c["rounds"] for r in dl):
                bad.append("e4: a deliberation has the wrong number of votes or rounds")
            if dl and any(v["readout_incoming"] is None for r in dl for t in r["votes"][1:] for v in t):
                bad.append("e4: votes after a round without incoming-statement readouts")
            s = json.load(open(d / "summary.json"))
            for k in sorted(s):
                if "jury_wrong_majority_final" in s[k]:
                    print(f"  e4 {k}: wrong majority {s[k]['jury_wrong_majority_final']['mean']:.2f}, unsteered shift "
                          f"{s[k]['unsteered_shift']['mean']:+.3f}")
                elif "wrong_rate" in s[k]:
                    print(f"  e4 {k}: wrong rate {s[k]['wrong_rate']['mean']:.2f}")
        if x == "e5":
            s = json.load(open(d / "summary.json"))
            for part in ("jury", "minority"):
                if part in s:
                    g = s[part].get("general_readout") or {}
                    print(f"  e5 {part}: n {s[part]['n']}, general_readout rho {g.get('rho')} p {g.get('p')}")
    for b in bad:
        print(f"  PROBLEM {b}")
    print(f"[check] {run}: {'ok' if not bad else f'{len(bad)} problem(s)'}")
    return 1 if (bad and a.strict) else 0


if __name__ == "__main__":
    sys.exit(main())
