#!/usr/bin/env python
"""Run preference-swarm experiments from a config.

    python scripts/run_swarm.py --config configs/llama8b.yaml               # the config's experiment list
    python scripts/run_swarm.py --config configs/llama8b.yaml --exp e3,e5   # a subset
    python scripts/run_swarm.py --config configs/e1.yaml                    # experiment 1 (multi-model)
    python scripts/run_swarm.py --config configs/llama8b.yaml --out outputs/llama8b_<ts> --exp e5

Each invocation writes outputs/<run_name>_<UTC>/ (or --out): config.json, provenance.json, telemetry.json (per
experiment status, seconds, error), one folder per experiment with its JSONL records and summary.json. An
experiment that raises is recorded and the next one still runs; the exit status is 1 if any failed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import yaml

from preference_swarm.runlog import REPO_ROOT, Telemetry, build_provenance, run_main, write_json

SINGLE_MODEL = ("e2", "e3", "e4", "e5")


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--exp", default=None, help="comma-separated subset of the config's experiments")
    ap.add_argument("--out", default=None, type=Path, help="reuse a run directory (e.g. to rerun e5 only)")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    exps = [x.strip() for x in (a.exp or ",".join(cfg["experiments"])).split(",") if x.strip()]
    bad = [x for x in exps if x not in ("e1",) + SINGLE_MODEL]
    if bad:
        ap.error(f"unknown experiment(s) {bad}")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = a.out if a.out else REPO_ROOT / "outputs" / f"{cfg['run_name']}_{stamp}"
    out = out if out.is_absolute() else REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    write_json(cfg, out / "config.json")
    write_json(build_provenance(str(a.config)), out / "provenance.json")
    tel = Telemetry(out / "telemetry.json", {"run": cfg["run_name"], "out": str(out), "experiments": exps,
                                             "started": dt.datetime.now(dt.timezone.utc).isoformat()})
    print(f"[swarm] {cfg['run_name']}: {exps} -> {out}", flush=True)
    failed = []
    if "e1" in exps:
        from preference_swarm.experiments import e1_self_other
        t0 = time.time()
        try:
            e1_self_other.run(cfg["e1"] | {k: cfg[k] for k in ("seed",) if k in cfg}, out / "e1_self_other")
            tel.record(cfg["run_name"], "e1", "ok", time.time() - t0)
        except Exception as exc:          # recorded, then the next experiment runs
            failed.append("e1")
            tel.record(cfg["run_name"], "e1", "failed", time.time() - t0, error=repr(exc),
                       traceback=traceback.format_exc()[-3000:])
            traceback.print_exc()
    single = [x for x in exps if x in SINGLE_MODEL]
    if single:
        from preference_swarm.experiments import common, e2_negotiate, e3_jury, e4_minority, e5_monitor
        mods = {"e2": e2_negotiate, "e3": e3_jury, "e4": e4_minority, "e5": e5_monitor}
        t0 = time.time()
        try:
            ctx = common.open_ctx(cfg, out)
            tel.record(cfg["model"], "load", "ok", time.time() - t0, residual_scale=ctx.scale, layer=ctx.layer)
        except Exception as exc:
            tel.record(cfg["model"], "load", "failed", time.time() - t0, error=repr(exc),
                       traceback=traceback.format_exc()[-3000:])
            traceback.print_exc()
            tel.finish()
            return 1
        if any(x in single for x in ("e2", "e3", "e4")) and cfg.get("calibrate") is not None:
            t0 = time.time()
            try:     # the manipulation check first: the log shows within minutes whether the pushes move choices
                cal = common.calibrate(ctx, cfg["calibrate"])
                tel.record(cfg["model"], "calibrate", "ok", time.time() - t0,
                           like={k: v["mean"] for k, v in cal.items() if k.startswith("like@")})
            except Exception as exc:
                failed.append("calibrate")
                tel.record(cfg["model"], "calibrate", "failed", time.time() - t0, error=repr(exc),
                           traceback=traceback.format_exc()[-3000:])
                traceback.print_exc()
        for x in single:
            t0 = time.time()
            print(f"[swarm] {x} starting", flush=True)
            try:
                mods[x].run(ctx, cfg.get(x, {}))
                tel.record(cfg["model"], x, "ok", time.time() - t0)
                print(f"[swarm] {x} ok in {(time.time() - t0) / 60:.1f} min -> {out}", flush=True)
            except Exception as exc:
                failed.append(x)
                tel.record(cfg["model"], x, "failed", time.time() - t0, error=repr(exc),
                           traceback=traceback.format_exc()[-3000:])
                traceback.print_exc()
    try:     # this process's peak GPU memory: tells whether two runs can share the card
        import torch
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            print(f"[swarm] peak GPU memory allocated by this run: {peak:.1f} GiB", flush=True)
            tel.record(cfg["model"], "gpu_peak", "ok", 0.0, peak_gib=round(peak, 2))
    except Exception:
        pass
    tel.finish()
    print(f"[swarm] done; failed: {failed or 'none'}; {out}", flush=True)
    (REPO_ROOT / "outputs" / "latest").mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "outputs" / "latest" / f"{cfg['run_name']}.txt").write_text(str(out) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run_main("preference_swarm", main) or 0)
