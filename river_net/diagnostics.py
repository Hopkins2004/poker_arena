"""Range-generator diagnostics: ``python -m river_net.diagnostics [--n 1000] [--seed 0]``.

For ``n`` spots per source, prints text histograms of the SPR, the effective number of
combos ``1/Σr²`` (both players) and a polarization index, plus a few example ranges as their
top-20 hand classes. Polarization: each hand's river showdown equity against the opponent's
range; the index is the range mass with equity < 0.2 or ≥ 0.8 (bottom and top equity
quintiles), so 0 = all middling hands and 1 = fully polarized.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from pokerarena import _native, preflop

from .cards import board_str
from .ranges import SOURCES, RangeGenerator, effective_combos


def polarization(spot, p: int) -> float:
    me, opp = (spot.r1, spot.r2) if p == 0 else (spot.r2, spot.r1)
    eq, _ = _native.solver_river_equity(spot.board.tolist(), opp.astype(np.float32))
    return float(me[(eq < 0.2) | (eq >= 0.8)].sum() / me.sum())


def histogram(values, bins, label, width=40) -> str:
    counts, edges = np.histogram(values, bins=bins)
    top = max(counts.max(), 1)
    lines = [f"  {label}: median {np.median(values):.2f}, p5 {np.percentile(values, 5):.2f}, "
             f"p95 {np.percentile(values, 95):.2f}"]
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        lines.append(f"    [{lo:7.2f}, {hi:7.2f})  {'#' * int(round(width * c / top)):<{width}} {c}")
    return "\n".join(lines)


def top_classes(r, k=20) -> str:
    w = np.bincount(preflop.COMBO_CLASS, weights=r, minlength=169)
    idx = np.argsort(-w)[:k]
    return " ".join(f"{preflop.CLASSES[i]}:{100 * w[i]:.1f}" for i in idx if w[i] > 0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=1000, help="spots per source")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--examples", type=int, default=2, help="example ranges per source")
    a = ap.parse_args(argv)
    gen = RangeGenerator(a.seed)
    for src in SOURCES:
        t0 = time.time()
        spots = [gen.generate(i, source=src) for i in range(a.n)]
        dt = time.time() - t0
        spr = np.array([s.spr for s in spots])
        eff = np.array([effective_combos(r) for s in spots for r in (s.r1, s.r2)])
        pol = np.array([polarization(s, p) for s in spots for p in (0, 1)])
        print(f"\n=== source {src}: {a.n} spots, {a.n / dt:.0f} spots/s (one core)")
        print(histogram(spr, [0.1, 0.5, 1, 2, 3, 5, 8, 12, 20], "SPR"))
        print(histogram(eff, [0, 20, 50, 100, 200, 400, 700, 1100], "effective combos"))
        print(histogram(pol, np.linspace(0, 1, 11), "polarization (mass with eq<.2 or ≥.8)"))
        if src != "B":
            kinds = {}
            for s in spots:
                kinds[s.meta["pot_type"]] = kinds.get(s.meta["pot_type"], 0) + 1
            print(f"  pot types: {kinds}")
        for s in spots[: a.examples]:
            m = s.meta
            print(f"\n  example: board {board_str(s.board)}, pot {s.pot:.1f}bb, "
                  f"stack {s.stack:.1f}bb (SPR {s.spr:.2f}), {m['scenario']} | {m['line']}")
            print(f"    P1 (OOP) top-20 classes (% of range): {top_classes(s.r1)}")
            print(f"    P2 (IP)  top-20 classes (% of range): {top_classes(s.r2)}")


if __name__ == "__main__":
    main()
