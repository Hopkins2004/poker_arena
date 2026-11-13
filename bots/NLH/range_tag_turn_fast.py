"""range_solver_turn with the fast turn solve (river sampling, fixed iteration plan).

Plays exactly like ``range_solver_turn`` (chart preflop, TAG flop, heads-up turn solve with
river-network leaves, exact river solve from the turn solve's ranges) except for how the
turn is solved: instead of up to 200 DCFR iterations on all 48 river cards with
exploitability checks, it runs the stages of ``TURN_PLAN`` back to back on one solver,
with no checks. Each stage is (leaf model, iterations, river cards sampled per
iteration or None for all 48). The default, 150 iterations of river-network leaves on 8
sampled river cards, was chosen on the fixed turn benchmark
(``python -m river_net.turn_bench``; see bots/NLH/SOLVERS.md): 0.36 s per solve and 1.44%
exploitability with exact river play, against 1.57 s and 2.09% for 300 full-card
iterations.
"""

import numpy as np

from bots.NLH.range_solver_turn import RangeSolverTurn

MODELS = {"net": "river_net/models/river_k64.pt", "pre": "river_net/models/preriver_k256.pt"}
_NETS: dict = {}


def _leaf(kind: str, threads: int):
    """A fresh leaf evaluator: "net" = river network on every river card, "pre" = pre-river
    network (one query per leaf). Models load once per process."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if kind not in _NETS:
        if kind == "net":
            from river_net.model import RiverValueNet

            _NETS[kind] = RiverValueNet.load(str(root / MODELS[kind]))
        else:
            from river_net.preriver import PreRiverNet

            _NETS[kind] = PreRiverNet.load(str(root / MODELS[kind]))
    if kind == "net":
        from river_net.turn import NetworkLeaf

        return NetworkLeaf(_NETS[kind], threads=threads)
    from river_net.preriver import PreRiverLeaf

    return PreRiverLeaf(_NETS[kind], threads=threads)


class RangeTagTurnFast(RangeSolverTurn):
    """range_solver_turn with a fixed, sampled turn-solve plan."""

    NAME = "range_tag_turn_fast"
    ANCHOR = None

    TURN_PLAN = (("net", 150, 8),)   # 0.36 s, 1.44% expl@exact (benchmark median)

    def _solve_turn(self, obs, live):
        from river_net.solver import Spot
        from river_net.turn import TurnSolver

        n = obs.n_seats
        board = [int(c) for c in obs.board]
        pot0 = int(obs.pot) - int(np.sum(obs.street_contrib))
        behind = [int(obs.stacks[s]) + int(obs.street_contrib[s]) for s in live]
        if len(board) != 4 or pot0 <= 0 or min(behind) <= 0:
            return None
        order = [(obs.button + k) % n for k in range(1, n + 1)]
        players = [s for s in order if s in live]  # [OOP, IP]
        me = players.index(obs.seat)
        ranges = self._river_ranges(obs, players, board, streets=((1, 3),))
        if ranges is None:
            return None
        spot = Spot(np.array(board, dtype=np.uint8), float(pot0), float(min(behind)),
                    ranges[0], ranges[1])
        ts = res = None
        for kind, iters, samples in self.TURN_PLAN:
            leaf = _leaf(kind, self.TORCH_THREADS)
            if ts is None:
                ts = TurnSolver(spot, leaf, river_samples=samples,
                                seed=int(self.nrng.integers(1 << 31)))
            else:
                ts.set_leaf(leaf)
                ts.river_samples = samples
            res = ts.solve(iters=iters, target_expl=0.0, check_every=0)
        self.stats["turn_solves"] += 1
        return res, float(pot0), players, me, ts.r
