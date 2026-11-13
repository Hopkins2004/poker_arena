"""Flop-solver check: ``python -m river_net.flop_check --spots 10``.

For flop spots from the range generator (``RangeGenerator(seed).generate(i, street="flop")``)
each configuration's flop strategy is solved (timed single-threaded, no checks inside the
timed solve) and scored by re-solving every flop leaf × turn card with the **turn solver**
(river-network leaves, 150 iterations on 8 sampled river cards, the fast plan) at the leaf
ranges the strategy induces, holding those values fixed:

* ``expl@turn`` (% of the flop pot): exploitability of the flop strategy when the turn is
  played by the turn solver: the comparable quality measure across leaf models;
* ``expl own``: exploitability in the configuration's own leaf model;
* leaf error (network configurations): turn network vs turn-solver values at the induced
  ranges (turn-pot units, range-weighted and unweighted).

Configurations: ``tnet-<iters>[-s<m>]`` (turn network, ``m`` sampled turn cards per
iteration) and ``cd-<iters>`` (check-down leaves).
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

from .cards import HAND_CARDS
from .flop import (CheckdownFlopLeaf, FixedTurnLeaf, FlopSolver, TurnNetLeaf, TurnValueNet,
                   turn_continuation_values)
from .ranges import RangeGenerator

TURN_MODEL = "river_net/models/turn_k256.pt"
CONFIG = re.compile(r"^(tnet|cd)-(\d+)(?:-s(\d+))?$")


def run(spot, name, net):
    m = CONFIG.match(name)
    if not m:
        raise ValueError(f"bad configuration {name!r}")
    leaf = TurnNetLeaf(net) if m.group(1) == "tnet" else CheckdownFlopLeaf()
    samples = int(m.group(3)) if m.group(3) else None
    t0 = time.perf_counter()
    fs = FlopSolver(spot, leaf, turn_samples=samples)
    fs.solve(iters=int(m.group(2)), target_expl=0.0, check_every=0)
    return fs, time.perf_counter() - t0


def leaf_errors(fs, R1, R2, Vnet, Vex):
    num_w = num_u = den = 0.0
    for l in range(len(fs.leaves)):
        for k, t in enumerate(fs.boards.cards):
            valid = fs.valid & ~(HAND_CARDS == t).any(axis=1)
            t1, t2 = R1[l][valid].sum(), R2[l][valid].sum()
            weight = t1 * t2
            if weight <= 1e-12:
                continue
            for p, R, tt in ((0, R1[l], t1), (1, R2[l], t2)):
                err = np.abs(Vnet[p][l, k] - Vex[p][l, k])
                num_w += weight * float((err * R * valid).sum() / tt) / 2
                num_u += weight * float(err[valid].mean()) / 2
            den += weight
    return num_w / max(den, 1e-30), num_u / max(den, 1e-30)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spots", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--configs", nargs="*", default=["cd-150", "tnet-150-s8"])
    ap.add_argument("--model", default=TURN_MODEL)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("results/flop/flop_check.json"))
    a = ap.parse_args(argv)
    torch.set_num_threads(1)
    net = TurnValueNet.load(a.model)
    gen = RangeGenerator(a.seed)
    spots = [gen.generate(i, street="flop") for i in range(a.spots)]
    solved = {}
    for name in a.configs:
        solved[name] = [run(s, name, net) for s in spots]
        print(f"{name:>14}: median solve {np.median([t for _, t in solved[name]]):.2f}s",
              file=sys.stderr)
    results = {}
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for name, rows in solved.items():
            per = []
            for i, (fs, sec) in enumerate(rows):
                avg = [fs._avg(0), fs._avg(1)]
                fs.refresh, fs.river_samples = None, None
                own = fs.exploitability(avg)
                R1, R2 = fs.leaf_ranges(avg)
                Vex, turn_expl = turn_continuation_values(fs, R1, R2, pool)
                row = {"spot": i, "seconds": sec, "expl_own": own, "spr": spots[i].spr,
                       "turn_expl": turn_expl, "scenario": spots[i].meta["scenario"]}
                if isinstance(fs.leaf, TurnNetLeaf):
                    Vnet, _ = fs.leaf.card_values(R1, R2, fs.leaf_c, root=fs.r)
                    row["leaf_wmae"], row["leaf_mae"] = leaf_errors(fs, R1, R2, Vnet, Vex)
                fs.set_leaf(FixedTurnLeaf(Vex))
                row["expl_turn"] = fs.exploitability(avg)
                per.append(row)
                print(f"{name:>14} spot {i:2d} SPR {row['spr']:5.2f} {sec:5.2f}s "
                      f"own {100 * own:.2f}% @turn {100 * row['expl_turn']:.2f}%"
                      + (f" leaf wMAE {row['leaf_wmae']:.4f}" if "leaf_wmae" in row else ""),
                      file=sys.stderr)
            results[name] = per
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"spots": a.spots, "seed": a.seed, "model": a.model,
                                 "results": results}, indent=1))
    print(f"\n{a.spots} flop spots (seed {a.seed}); turn continuation: turn solver "
          f"(river-network leaves, 150 iterations, 8 river samples)")
    print(f"{'config':>14} | {'time med':>8} | {'expl@turn med':>13} | {'mean':>6} | "
          f"{'own med':>7} | {'leaf wMAE':>9} | {'leaf MAE':>8}")
    for name, per in results.items():
        t = np.array([r["seconds"] for r in per])
        e = np.array([r["expl_turn"] for r in per]) * 100
        o = np.array([r["expl_own"] for r in per]) * 100
        lw = np.mean([r["leaf_wmae"] for r in per]) if "leaf_wmae" in per[0] else float("nan")
        lu = np.mean([r["leaf_mae"] for r in per]) if "leaf_mae" in per[0] else float("nan")
        print(f"{name:>14} | {np.median(t):7.2f}s | {np.median(e):12.2f}% | {e.mean():5.2f}% | "
              f"{np.median(o):6.2f}% | {lw:9.4f} | {lu:8.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
