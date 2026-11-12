"""Turn-solver check: ``python -m river_net.turn_check --spots 20``.

For turn spots from the range generator (source mix as configured, cut on the turn), solves
the turn with network leaves and with check-down leaves, then re-solves every river leaf ×
river card **exactly** (river solver, induced average ranges) and reports:

* leaf error: the network's per-hand river values vs the exact re-solve on the ranges that
  actually reach the leaves (pot units of the river pot; range-weighted and unweighted,
  aggregated with weights = joint leaf reach);
* exploitability of each turn strategy in its own leaf model, and with the river continuation
  fixed to the exact re-solves (``expl@exact``, % of the turn pot): the comparable measure of
  turn-strategy quality across leaf models;
* time per turn solve.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time

import numpy as np

from .cards import HAND_CARDS, N_COMBOS
from .model import RiverValueNet
from .ranges import RangeGenerator
from .solver import Spot, solve
from .turn import CheckdownLeaf, FixedLeaf, NetworkLeaf, TurnSolver


def _river_job(args):
    board, pot, stack, r1, r2 = args
    res = solve(Spot(np.asarray(board, dtype=np.uint8), pot, stack, r1, r2))
    return res.v1, res.v2, res.expl


def exact_leaf_values(ts: TurnSolver, R1, R2, pool):
    """Exact per-card conditional values ``V[p][l, card, h]`` at the given leaf ranges."""
    cards = ts.boards.cards
    L, C = len(ts.leaves), len(cards)
    V = [np.zeros((L, C, N_COMBOS)), np.zeros((L, C, N_COMBOS))]
    jobs, keys = [], []
    for l in range(L):
        c = ts.leaf_c[l]
        for k, card in enumerate(cards):
            dead = (HAND_CARDS == card).any(axis=1)
            r = []
            for p, R in enumerate((R1[l], R2[l])):
                x = np.where(dead, 0.0, R)
                if x.sum() <= 1e-12:  # nobody of this player reaches: value vs root range
                    x = np.where(dead, 0.0, ts.r[p])
                r.append((x / x.sum()).astype(np.float32))
            jobs.append((list(ts.board) + [card], 1 + 2 * c, ts.S - c, r[0], r[1]))
            keys.append((l, k))
    out = pool.map(_river_job, jobs, chunksize=8)
    expl = []
    for (l, k), (v1, v2, e) in zip(keys, out):
        V[0][l, k], V[1][l, k] = v1, v2
        expl.append(e)
    return V, float(np.median(expl)) if expl else 0.0


def leaf_errors(ts: TurnSolver, R1, R2, Vnet, Vex):
    """Range-weighted and unweighted MAE of network vs exact leaf values."""
    cards = ts.boards.cards
    num_w = num_u = den = 0.0
    for l in range(len(ts.leaves)):
        for k, card in enumerate(cards):
            valid = ts.valid & ~(HAND_CARDS == card).any(axis=1)
            t1, t2 = R1[l][valid].sum(), R2[l][valid].sum()
            weight = t1 * t2
            if weight <= 1e-12:
                continue
            for p, R, t in ((0, R1[l], t1), (1, R2[l], t2)):
                err = np.abs(Vnet[p][l, k] - Vex[p][l, k])
                num_w += weight * float((err * R * valid).sum() / t) / 2
                num_u += weight * float(err[valid].mean()) / 2
            den += weight
    return num_w / max(den, 1e-30), num_u / max(den, 1e-30)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spots", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--model", default="river_net/models/river_k64.pt")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    net = RiverValueNet.load(a.model)
    gen = RangeGenerator(a.seed)
    rows = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for i in range(a.spots):
            spot = gen.generate(i, street="turn")
            row = {"i": i, "spr": spot.spr, "source": spot.meta["source"],
                   "line": spot.meta["line"]}
            for name, leaf in (("net", NetworkLeaf(net, threads=a.threads)),
                               ("cd", CheckdownLeaf())):
                ts = TurnSolver(spot, leaf)
                res = ts.solve(iters=a.iters, target_expl=0.0, check_every=a.iters)
                R1, R2 = ts.leaf_ranges()
                Vex, river_expl = exact_leaf_values(ts, R1, R2, pool)
                row[f"{name}.seconds"] = res.seconds
                row[f"{name}.expl"] = res.expl
                if name == "net":
                    Vnet, _ = leaf.card_values(R1, R2, ts.leaf_c, root=ts.r)
                    row["leaf.wmae"], row["leaf.mae"] = leaf_errors(ts, R1, R2, Vnet, Vex)
                    row["river_expl"] = river_expl
                own_root = float((ts.r[0] * res.cfv1).sum())
                ts.leaf = FixedLeaf(Vex)
                ts.leaf.bind(ts.boards, ts.S)
                row[f"{name}.expl_exact"] = ts.exploitability()
                ex = ts._profile_values([ts._avg(0), ts._avg(1)])
                row[f"{name}.root_err"] = abs(own_root - float((ts.r[0] * ex[0]).sum()))
            rows.append(row)
            print(f"spot {i:2d} {row['source']} SPR {row['spr']:5.2f} {row['line']:>8} | "
                  f"net {row['net.seconds']:4.1f}s expl {100 * row['net.expl']:.2f}% "
                  f"@exact {100 * row['net.expl_exact']:.2f}% | check-down expl@exact "
                  f"{100 * row['cd.expl_exact']:.2f}% | leaf wMAE {row['leaf.wmae']:.4f}",
                  file=sys.stderr)
    med = lambda k: float(np.median([r[k] for r in rows]))  # noqa: E731
    mean = lambda k: float(np.mean([r[k] for r in rows]))  # noqa: E731
    print(f"\n{len(rows)} turn spots, {a.iters} iterations")
    print(f"  time per turn solve: network {med('net.seconds'):.1f}s, check-down "
          f"{med('cd.seconds'):.1f}s (median)")
    print(f"  leaf error (network vs exact river re-solve, river-pot units): range-weighted "
          f"{mean('leaf.wmae'):.4f}, unweighted {mean('leaf.mae'):.4f}")
    print(f"  exploitability in own leaf model (% turn pot): network {100 * med('net.expl'):.2f}, "
          f"check-down {100 * med('cd.expl'):.2f} (median)")
    print(f"  exploitability with exact river continuation (% turn pot): network "
          f"{100 * med('net.expl_exact'):.2f}, check-down {100 * med('cd.expl_exact'):.2f} "
          f"(median); mean {100 * mean('net.expl_exact'):.2f} vs {100 * mean('cd.expl_exact'):.2f}")
    print(f"  root value error vs exact continuation (turn pot): network "
          f"{mean('net.root_err'):.4f}, check-down {mean('cd.root_err'):.4f}")
    wins = sum(r["net.expl_exact"] < r["cd.expl_exact"] for r in rows)
    print(f"  network turn strategy less exploitable than check-down on {wins}/{len(rows)} spots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
