"""Fixed turn-solver speed/quality benchmark: ``python -m river_net.turn_bench``.

The same 20 turn spots as :mod:`river_net.turn_check` (``RangeGenerator(7)``, indices 0..19,
``street="turn"``) are solved with each configuration, timed single-threaded
(``torch`` 1 thread, no exploitability checks inside the timed solve), and then scored:

* ``expl@exact`` (% of the turn pot): exploitability of the configuration's average turn
  strategy when every river leaf × river card is re-solved **exactly** (river solver) at the
  leaf ranges the strategy induces — the comparable quality measure across leaf models and
  speed tricks;
* ``expl own``: exploitability in the configuration's own leaf model (no sampling, no
  stale values).

Configurations are named ``<leaf>-<iters>[-r<k>][-s<m>][+<leaf2>-<iters2>]``:
``net`` (river network on every river card), ``pre`` (pre-river network), ``cd``
(check-down); ``r<k>`` refreshes leaf values every k iterations (after ``w<n>``
warm-up iterations that refresh every time, default 10); ``s<m>`` samples m river cards per
iteration; ``+...`` continues the solve with another leaf model (warm start).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .model import RiverValueNet
from .ranges import RangeGenerator
from .turn import CheckdownLeaf, FixedLeaf, NetworkLeaf, TurnSolver
from .turn_check import exact_leaf_values

RIVER_MODEL = "river_net/models/river_k64.pt"
PRERIVER_MODEL = "river_net/models/preriver_k256.pt"
DEFAULT_CONFIGS = ["cd-300", "net-300", "net-100", "net-200-s8", "net-150-s8", "net-100-s8",
                   "net-100-s4", "net-60-s8", "pre-100", "pre-100+net-50", "pre-60+net-40-s8"]

_NETS: dict = {}


def _leaf(kind: str):
    if kind == "cd":
        return CheckdownLeaf()
    if kind == "net":
        if "net" not in _NETS:
            _NETS["net"] = RiverValueNet.load(RIVER_MODEL)
        return NetworkLeaf(_NETS["net"])
    if kind == "pre":
        from .preriver import PreRiverLeaf, PreRiverNet

        if "pre" not in _NETS:
            _NETS["pre"] = PreRiverNet.load(PRERIVER_MODEL)
        return PreRiverLeaf(_NETS["pre"])
    raise ValueError(f"unknown leaf {kind!r}")


STAGE = re.compile(r"^(cd|net|pre)-(\d+)((?:-[rsw]\d+)*)$")


def parse(name: str) -> list:
    stages = []
    for part in name.split("+"):
        m = STAGE.match(part)
        if not m:
            raise ValueError(f"bad configuration {name!r}")
        opts = dict(re.findall(r"-([rsw])(\d+)", m.group(3)))
        stages.append({"leaf": m.group(1), "iters": int(m.group(2)),
                       "refresh": int(opts["r"]) if "r" in opts else None,
                       "samples": int(opts["s"]) if "s" in opts else None,
                       "warmup": int(opts.get("w", 10))})
    return stages


def run(spot, name: str):
    """Solves ``spot`` with configuration ``name``: (solver, seconds)."""
    stages = parse(name)
    t0 = time.perf_counter()
    first = stages[0]
    ts = TurnSolver(spot, _leaf(first["leaf"]), refresh=first["refresh"],
                    refresh_warmup=first["warmup"], river_samples=first["samples"])
    for k, st in enumerate(stages):
        if k:
            ts.set_leaf(_leaf(st["leaf"]))
            ts.refresh, ts.river_samples = st["refresh"], st["samples"]
            ts.refresh_warmup = st["warmup"]
        ts.solve(iters=st["iters"], target_expl=0.0, check_every=0)
    return ts, time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spots", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/turn_speed/bench.json"))
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    gen = RangeGenerator(a.seed)
    spots = [gen.generate(i, street="turn") for i in range(a.spots)]
    for name in a.configs:  # load models before timing
        for st in parse(name):
            _leaf(st["leaf"])
    solved = {}
    for name in a.configs:
        rows = []
        for spot in spots:
            ts, sec = run(spot, name)
            rows.append((ts, sec))
        solved[name] = rows
        print(f"{name:>18}: median solve {np.median([s for _, s in rows]):.3f}s", file=sys.stderr)
    results = {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for name, rows in solved.items():
            per = []
            for i, (ts, sec) in enumerate(rows):
                avg = [ts._avg(0), ts._avg(1)]
                ts.refresh, ts.river_samples = None, None
                own = ts.exploitability(avg)
                R1, R2 = ts.leaf_ranges(avg)
                Vex, _ = exact_leaf_values(ts, R1, R2, pool)
                ts.set_leaf(FixedLeaf(Vex))
                ex = ts.exploitability(avg)
                per.append({"spot": i, "seconds": sec, "expl_own": own, "expl_exact": ex,
                            "spr": spots[i].spr})
            results[name] = per
            e = np.array([r["expl_exact"] for r in per]) * 100
            print(f"{name:>18}: time median {np.median([r['seconds'] for r in per]):.3f}s  "
                  f"expl@exact median {np.median(e):.2f}% mean {e.mean():.2f}%  own median "
                  f"{100 * np.median([r['expl_own'] for r in per]):.2f}%", file=sys.stderr)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"spots": a.spots, "seed": a.seed, "results": results},
                                indent=1))
    print(f"\n{'config':>18} | {'time med':>8} | {'time mean':>9} | {'expl@exact med':>14} | "
          f"{'mean':>6} | {'own med':>7}")
    for name, per in results.items():
        t = np.array([r["seconds"] for r in per])
        e = np.array([r["expl_exact"] for r in per]) * 100
        o = np.array([r["expl_own"] for r in per]) * 100
        print(f"{name:>18} | {np.median(t):7.3f}s | {t.mean():8.3f}s | {np.median(e):13.2f}% | "
              f"{e.mean():5.2f}% | {np.median(o):6.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
