"""What every single-model experiment shares: the model, the exp50 inputs, the steering unit, the output folder."""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from preference_swarm import data as DT
from preference_swarm import engine as EN
from preference_swarm.runlog import REPO_ROOT, write_json

LOG = logging.getLogger("preference_swarm")
N_SCALE_TEXTS = 64


@dataclass
class Ctx:
    cfg: Dict
    out: Path
    hf: object = None
    tok: object = None
    e50: Optional[DT.Exp50] = None
    layer: int = 0
    scale: float = 0.0
    timings: Dict[str, float] = field(default_factory=dict)

    # -- directions -----------------------------------------------------------------------------------------------
    def vec(self, name: str) -> np.ndarray:
        return DT.direction(self.e50, name)

    @property
    def V(self) -> np.ndarray:
        return self.e50.V

    @property
    def dir_names(self) -> List[str]:
        return self.e50.names

    def alpha(self, frac: float) -> float:
        """A strength given as a fraction of the mean residual norm at the steered block."""
        return float(frac) * self.scale

    # -- model calls ----------------------------------------------------------------------------------------------
    def choose(self, items: List[EN.Item], desc: str = "choose", read: bool = True) -> List[Dict]:
        return EN.choose(self.hf, self.tok, self.layer, items, self.V if read else None,
                         batch_size=int(self.cfg.get("batch_choose", 8)), desc=desc)

    def generate(self, items: List[EN.Item], max_new_tokens: Optional[int] = None, seed: int = 0,
                 desc: str = "generate") -> List[str]:
        return EN.generate(self.hf, self.tok, self.layer, items,
                           max_new_tokens=int(max_new_tokens or self.cfg.get("max_new_tokens", 140)),
                           temperature=float(self.cfg.get("temperature", 1.0)),
                           batch_size=int(self.cfg.get("batch_generate", 32)), seed=seed, desc=desc)

    # -- output ---------------------------------------------------------------------------------------------------
    def sub(self, name: str) -> Path:
        p = self.out / name
        p.mkdir(parents=True, exist_ok=True)
        return p


CALIBRATE_Q = ("Here are two requests.\n\n{block}\n\nWhich would you rather be asked to answer? "
               "Reply with a single letter, A or B, and nothing else.")


def calibrate(ctx: "Ctx", cfg: Dict) -> Dict:
    """The like-steering manipulation check, run first so the log shows within minutes whether a strength moves
    the model's own choice: exp50 request pairs in exp50's own question, the LESS preferred request's tokens pushed
    along the like vector or a random direction at each strength, both orders. Reported per strength: mean shift
    of the margin toward the pushed request against the unpushed choice, with a bootstrap interval."""
    from preference_swarm import model as M
    from preference_swarm import stats as ST

    pairs = DT.task_pairs(ctx.e50, int(cfg.get("n", 100)), int(ctx.cfg.get("seed", 0)) + 9)
    fracs = [float(f) for f in cfg.get("fracs", [0.25, 0.5, 1.0, 2.0])]
    dirs = {"like": ctx.vec(ctx.cfg["like_vector"]), "random": ctx.vec("RANDOM_dir_0")}
    cells = [("none", 0.0)] + [(k, f) for k in dirs for f in fracs]
    items, keys = [], []
    for pi, (h, lo) in enumerate(pairs):
        for order in (0, 1):
            a, b = (ctx.e50.texts[lo], ctx.e50.texts[h]) if order == 0 else (ctx.e50.texts[h], ctx.e50.texts[lo])
            L = "A" if order == 0 else "B"
            msgs = [{"role": "user", "content": CALIBRATE_Q.format(block=M.option_block(a, b))}]
            for kind, f in cells:
                keys.append((pi, order, kind, f))
                items.append(EN.Item(msgs, a, b, like=L if kind != "none" else None,
                                     vector=dirs.get(kind), alpha=ctx.alpha(f) if kind != "none" else 0.0))
    res = ctx.choose(items, desc="calibrate", read=False)
    m = {}
    for (pi, order, kind, f), x in zip(keys, res):
        m[(pi, order, kind, f)] = x["margin"] if order == 0 else -x["margin"]      # toward the pushed (L) request
    out = {"n_pairs": len(pairs), "fracs": fracs, "none_toward_L": ST.mean_ci(
        [np.mean([m[(p, o, "none", 0.0)] for o in (0, 1)]) for p in range(len(pairs))])}
    for kind in dirs:
        for f in fracs:
            d = [np.mean([m[(p, o, kind, f)] - m[(p, o, "none", 0.0)] for o in (0, 1)]) for p in range(len(pairs))]
            out[f"{kind}@{f}"] = {**ST.mean_ci(d), "p": ST.sign_flip_p(d)}
    write_json(out, ctx.out / "calibration.json")
    print(f"[calibrate] {len(pairs)} pairs; unpushed margin toward the less preferred request "
          f"{out['none_toward_L']['mean']:+.2f}", flush=True)
    for f in fracs:
        a_, r_ = out[f"like@{f}"], out[f"random@{f}"]
        print(f"[calibrate] push {f:4.2f} x norm: like {a_['mean']:+.3f} [{a_['lo']:+.3f}, {a_['hi']:+.3f}]   "
              f"random {r_['mean']:+.3f} [{r_['lo']:+.3f}, {r_['hi']:+.3f}]  (logits toward the pushed request)",
              flush=True)
    return out


def write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    n = 0
    with open(path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> List[Dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def open_ctx(cfg: Dict, out: Path) -> Ctx:
    """Load the model and the exp50 inputs (or random stand-ins for the smoke), and measure the steering unit."""
    from preference_swarm import model as M

    t0 = time.time()
    ctx = Ctx(cfg=cfg, out=out)
    ctx.hf, ctx.tok = M.load(cfg["model"], cfg.get("dtype", M.DEFAULT_DTYPE))
    d = int(ctx.hf.config.hidden_size)
    src = cfg["exp50_dir"]
    if src == "random":
        texts = {}
        prompts = REPO_ROOT / cfg.get("smoke_prompts", "data/smoke_prompts.jsonl")
        for line in open(prompts):
            r = json.loads(line)
            texts[r["name"]] = r["text"]
        ctx.e50 = DT.random_exp50(texts, d, int(cfg["layer"]), int(cfg.get("seed", 0)))
    else:
        p = Path(src)
        ctx.e50 = DT.load_exp50(p if p.is_absolute() else REPO_ROOT / p)
        if ctx.e50.model != cfg["model"]:
            raise ValueError(f"exp50 directions were fitted on {ctx.e50.model}, the config loads {cfg['model']}")
        if ctx.e50.V.shape[1] != d:
            raise ValueError(f"exp50 directions have width {ctx.e50.V.shape[1]}, the model {d}")
    nc = dict(cfg.get("null_directions") or {})
    if nc:
        st = None if src == "random" else DT.load_states(Path(src) if Path(src).is_absolute() else REPO_ROOT / src)
        ctx.e50 = DT.null_directions(ctx.e50, d, int(nc.get("cov", 30)), int(nc.get("iso", 10)),
                                     int(cfg.get("seed", 0)), st)
    ctx.layer = int(cfg.get("layer") or ctx.e50.layer)
    if src != "random" and ctx.layer != ctx.e50.layer:
        raise ValueError(f"config layer {ctx.layer} differs from the layer exp50 read the directions at "
                         f"({ctx.e50.layer})")
    names = DT.usable_names(ctx.e50)[:N_SCALE_TEXTS]
    ctx.scale = EN.residual_scale(ctx.hf, ctx.tok, ctx.layer, [ctx.e50.texts[k] for k in names])
    ctx.timings["load"] = round(time.time() - t0, 1)
    write_json({"model": cfg["model"], "layer": ctx.layer, "residual_scale": ctx.scale, "directions": ctx.dir_names,
                "n_usable_requests": len(DT.usable_names(ctx.e50))}, out / "context.json")
    LOG.info("model %s layer %d residual scale %.2f, %d directions", cfg["model"], ctx.layer, ctx.scale, len(ctx.V))
    return ctx
