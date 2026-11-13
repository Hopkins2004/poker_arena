"""range_solver plus a heads-up turn solve with the river value network as leaf evaluator.

* Preflop and flop: exactly ``range_tag`` (chart preflop, Monte Carlo equity on the flop).
* Turn, heads-up: estimates both turn ranges (preflop chart line, then the flop actions as
  equity-percentile soft filters, like ``range_solver``), solves the turn once per hand with
  ``river_net.turn.TurnSolver`` (DCFR, river leaves valued by the 3x512 river network
  ``river_net/models/river_k64.pt``, plan ``TURN_PLAN``) and plays its own hand from the
  average strategy. Opponent sizes off the tree map to the nearest tree action.
* River, heads-up: the exact river solve of ``range_solver``. When the turn was solved and
  the turn line stayed on the tree, the river ranges are the turn solve's ranges carried
  through both players' turn strategies along the line actually played (Bayes), restricted
  to the river card; otherwise the soft-filter estimate.
* Anything else (multiway, odd states, errors) falls back to the parent bot's logic.
"""

import numpy as np

from pokerarena import preflop

from bots.NLH.range_solver import COMBO_INDEX, LIVE, RangeSolver

TURN = 2
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


class RangeSolverTurn(RangeSolver):
    """range_solver, plus a depth-limited heads-up turn solve (network river leaves)."""

    NAME = "range_solver_turn"
    ANCHOR = None

    # Turn solve plan: stages of (leaf model, DCFR iterations, river cards sampled per
    # iteration or None for all 48), run back to back on one solver (a later stage continues
    # the earlier one's regrets). No exploitability checks: chosen on the fixed benchmark
    # (``python -m river_net.turn_bench``, see bots/NLH/SOLVERS.md).
    TURN_PLAN = (("net", 150, 8),)   # 0.36 s, 1.44% expl@exact (benchmark median)
    TORCH_THREADS = 1         # arena runs tables in parallel processes

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self._turn = None
        self.stats.update({"turn_solves": 0, "turn_fallbacks": 0, "river_from_turn": 0})

    def new_hand(self, info):
        super().new_hand(info)
        self._turn = None

    def act(self, obs):
        if obs.street == TURN and obs.variant == "nlhe":
            try:
                a = self._turn_action(obs)
            except Exception:  # noqa: BLE001 - never crash a table; fall back
                a = None
            if a is not None:
                return a
            self.stats["turn_fallbacks"] += 1
        return super().act(obs)

    # -- turn ----------------------------------------------------------------------------

    def _turn_action(self, obs):
        live = [s for s in range(obs.n_seats) if int(obs.status[s]) in LIVE]
        if len(live) != 2 or obs.seat not in live:
            return None
        if self._turn is None:
            self._turn = self._solve_turn(obs, live) or False
        if not self._turn:
            return None
        res, pot0, players, me, _r = self._turn
        t = res.tree
        node, _path = self._walk(t, obs.history, TURN, players, pot0)
        if node is None or t.kind[node] != "decision" or players[int(t.player[node])] != obs.seat:
            return None
        hole = sorted((int(obs.hole[0]), int(obs.hole[1])), reverse=True)
        probs = res.strategy_at(node)[:, COMBO_INDEX[tuple(hole)]].astype(np.float64)
        if probs.sum() <= 0:
            return None
        k = int(self.nrng.choice(len(probs), p=probs / probs.sum()))
        child = list(t.children(node))[k]
        return self._to_action(obs, t, child, me, pot0)

    def _walk(self, t, history, street, players, pot0):
        """Follow ``street``'s actions down the tree: (node, [(player, node, action index)])
        or (None, None) when the line leaves the tree."""
        node, path = 0, []
        for seat, st, kind, _amount, to, all_in in history:
            if st != street:
                continue
            if t.kind[node] != "decision" or players[int(t.player[node])] != seat:
                return None, None
            child = self._child(t, node, kind, to / pot0, all_in)
            if child is None:
                return None, None
            path.append((int(t.player[node]), node, list(t.children(node)).index(child)))
            node = child
        return node, path

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
        ts = None
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

    # -- river ranges from the turn solve ------------------------------------------------

    def _river_ranges(self, obs, players, board, streets=((1, 3), (2, 4))):
        if obs.street == TURN + 1 and self._turn:
            r = self._ranges_after_turn(obs, players, board)
            if r is not None:
                self.stats["river_from_turn"] += 1
                return r
        return super()._river_ranges(obs, players, board, streets)

    def _ranges_after_turn(self, obs, players, board):
        res, pot0, turn_players, _me, root = self._turn
        if turn_players != players:
            return None
        t = res.tree
        node, path = self._walk(t, obs.history, TURN, players, pot0)
        if node is None or t.kind[node] != "showdown":
            return None
        r = [root[0].copy(), root[1].copy()]
        for p, at, k in path:
            r[p] *= res.strategy_at(at)[k]
        river = board[4]
        dead = (np.asarray(preflop.COMBOS) == river).any(axis=1)
        out = []
        for x in r:
            x = np.where(dead, 0.0, x)
            if x.sum() <= 1e-12:
                return None
            out.append((x / x.sum()).astype(np.float32))
        return out
