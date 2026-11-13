"""Chart preflop, then a solver on every heads-up postflop street.

* Preflop: ``range_tag`` (the 6-max 100bb chart; off-chart lines answered as the last
  aggressive action).
* Flop, heads-up: both ranges from the preflop chart line (uniform when the hand left the
  chart), solved once per hand with ``river_net.flop.FlopSolver``: DCFR on the flop tree,
  turn leaves valued by the turn value network ``river_net/models/turn_k256.pt`` on
  ``FLOP_TURN_SAMPLES`` sampled turn cards per iteration, flop all-ins exact over all
  turn + river runouts. The bot plays its combo from the average strategy.
* Turn, heads-up: the turn solve of ``range_tag_turn_fast`` (river-network leaves, 150
  iterations on 8 sampled river cards), starting from the flop solve's ranges carried
  through both players' flop strategies along the line played (Bayes), restricted to the
  turn card.
* River, heads-up: the exact river solve, from the turn solve's ranges carried the same way.
* Whenever a step can't use the previous street's solve (line off the tree, a range
  emptied, multiway, errors), it falls back to the parent bots' logic: soft-filter range
  estimates, then TAG play.
"""

import numpy as np

from pokerarena import preflop

from bots.NLH.range_solver import COMBO_INDEX, LIVE
from bots.NLH.range_tag_turn_fast import RangeTagTurnFast

FLOP, TURN = 1, 2
TURN_MODEL = "river_net/models/turn_k256.pt"
_TURN_NET = None


def _turn_net():
    global _TURN_NET
    if _TURN_NET is None:
        from pathlib import Path

        from river_net.flop import TurnValueNet

        _TURN_NET = TurnValueNet.load(str(Path(__file__).resolve().parents[2] / TURN_MODEL))
    return _TURN_NET


class FullSolver(RangeTagTurnFast):
    """Flop, turn and river solves chained through Bayesian range updates."""

    NAME = "full_solver"
    ANCHOR = None

    FLOP_ITERS = 150          # flop DCFR iterations (no exploitability checks)
    FLOP_TURN_SAMPLES = 8     # turn cards sampled per iteration (of 49)

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self._flop = None
        self.stats.update({"flop_solves": 0, "flop_fallbacks": 0, "turn_from_flop": 0})

    def new_hand(self, info):
        super().new_hand(info)
        self._flop = None

    def act(self, obs):
        if obs.street == FLOP and obs.variant == "nlhe":
            try:
                a = self._flop_action(obs)
            except Exception:  # noqa: BLE001 - never crash a table; fall back
                a = None
            if a is not None:
                return a
            self.stats["flop_fallbacks"] += 1
        return super().act(obs)

    # -- flop ----------------------------------------------------------------------------

    def _flop_action(self, obs):
        live = [s for s in range(obs.n_seats) if int(obs.status[s]) in LIVE]
        if len(live) != 2 or obs.seat not in live:
            return None
        if self._flop is None:
            self._flop = self._solve_flop(obs, live) or False
        if not self._flop:
            return None
        res, pot0, players, me, _r = self._flop
        t = res.tree
        node, _path = self._walk(t, obs.history, FLOP, players, pot0)
        if node is None or t.kind[node] != "decision" or players[int(t.player[node])] != obs.seat:
            return None
        hole = sorted((int(obs.hole[0]), int(obs.hole[1])), reverse=True)
        probs = res.strategy_at(node)[:, COMBO_INDEX[tuple(hole)]].astype(np.float64)
        if probs.sum() <= 0:
            return None
        k = int(self.nrng.choice(len(probs), p=probs / probs.sum()))
        child = list(t.children(node))[k]
        return self._to_action(obs, t, child, me, pot0)

    def _solve_flop(self, obs, live):
        from river_net.flop import FlopSolver, TurnNetLeaf
        from river_net.solver import Spot

        n = obs.n_seats
        board = [int(c) for c in obs.board]
        pot0 = int(obs.pot) - int(np.sum(obs.street_contrib))
        behind = [int(obs.stacks[s]) + int(obs.street_contrib[s]) for s in live]
        if len(board) != 3 or pot0 <= 0 or min(behind) <= 0:
            return None
        order = [(obs.button + k) % n for k in range(1, n + 1)]
        players = [s for s in order if s in live]  # [OOP, IP]
        me = players.index(obs.seat)
        ranges = self._river_ranges(obs, players, board, streets=())  # preflop chart only
        if ranges is None:
            return None
        spot = Spot(np.array(board, dtype=np.uint8), float(pot0), float(min(behind)),
                    ranges[0], ranges[1])
        fs = FlopSolver(spot, TurnNetLeaf(_turn_net(), threads=self.TORCH_THREADS),
                        turn_samples=self.FLOP_TURN_SAMPLES,
                        seed=int(self.nrng.integers(1 << 31)))
        res = fs.solve(iters=self.FLOP_ITERS, target_expl=0.0, check_every=0)
        self.stats["flop_solves"] += 1
        return res, float(pot0), players, me, fs.r

    # -- turn ranges from the flop solve -------------------------------------------------

    def _river_ranges(self, obs, players, board, streets=((1, 3), (2, 4))):
        if obs.street == TURN and self._flop:
            r = self._ranges_after_flop(obs, players, board)
            if r is not None:
                self.stats["turn_from_flop"] += 1
                return r
        return super()._river_ranges(obs, players, board, streets)

    def _ranges_after_flop(self, obs, players, board):
        res, pot0, flop_players, _me, root = self._flop
        if flop_players != players:
            return None
        t = res.tree
        node, path = self._walk(t, obs.history, FLOP, players, pot0)
        if node is None or t.kind[node] != "showdown":
            return None
        r = [root[0].copy(), root[1].copy()]
        for p, at, k in path:
            r[p] *= res.strategy_at(at)[k]
        dead = (np.asarray(preflop.COMBOS) == board[3]).any(axis=1)
        out = []
        for x in r:
            x = np.where(dead, 0.0, x)
            if x.sum() <= 1e-12:
                return None
            out.append((x / x.sum()).astype(np.float32))
        return out
