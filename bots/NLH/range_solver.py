"""Chart preflop, TAG equity on the flop and turn, exact river solve (proof of concept).

* Preflop and flop/turn: exactly ``range_tag`` (chart preflop, Monte Carlo equity after).
* River, heads-up: estimates both players' river ranges, solves the river exactly with the
  ``river_net`` solver (vectorized DCFR, stops at 0.2% of the pot) and plays its own hand
  from the solver's average strategy.
  - Ranges: the preflop chart's ranges along the hand's preflop line (uniform when the hand
    left the chart), then every flop and turn action applied as a soft filter on the actor's
    range by equity percentile against the other range (bets/raises polarized, calls
    condensed, checks capped; fixed mid-range parameters of ``river_net.ranges``).
  - The solve happens once per hand at the bot's first river decision; later river
    decisions navigate the same solution. Opponent sizes off the solver's tree map to the
    nearest tree action.
  - Anything else (multiway river, odd states) falls back to the TAG logic.
"""

import numpy as np

from pokerarena import Action, _native, clamp_raise, preflop
from pokerarena.preflop import POSITION_BY_OFFSET, _match

from bots.NLH.range_tag import RangeTag

RIVER = 3
LIVE = (1, 2)
DB = preflop.load()
COMBO_INDEX = {(int(a), int(b)): i for i, (a, b) in enumerate(preflop.COMBOS)}


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _percentiles(score, w):
    q = np.zeros(len(w))
    idx = np.flatnonzero(w > 0)
    if idx.size:
        order = idx[np.argsort(score[idx], kind="stable")]
        cw = np.cumsum(w[order])
        q[order] = (cw - 0.5 * w[order]) / cw[-1]
    return q


# Deterministic mid-range versions of the range generator's soft filters.
def _filter(kind, q, facing_bet):
    T = 0.06
    if kind == "check":
        return 1 - 0.7 * _sig((q - 0.72) / T)
    if kind == "bet":
        return np.clip(_sig((q - 0.62) / T) + 0.45 * _sig((0.17 - q) / T), 0, 1)
    if kind == "raise":
        return np.clip(_sig((q - 0.81) / T) + 0.45 * _sig((0.11 - q) / T), 0, 1)
    p = _sig((q - 0.35) / T)  # call
    return p * (1 - 0.5 * _sig((q - 0.91) / T)) if facing_bet else p


def preflop_ranges(obs):
    """Seat -> 169 class weights along the chart line, or None if the hand left the chart."""
    n = obs.n_seats
    if n != 6 or any(int(obs.status[s]) == 0 for s in range(n)):
        return None
    pos = {s: POSITION_BY_OFFSET[(s - obs.button) % n] for s in range(n)}
    w = {s: np.ones(169) for s in range(n)}
    path = []
    for seat, street, kind, _amount, to, all_in in obs.history:
        if street != 0 or kind in ("ante", "sb", "bb"):
            continue
        node = DB.at(path)
        if node is None:
            if kind != "fold":
                return None
            path.append(0)
            continue
        if node.actor != pos[seat]:
            return None
        idx, _ = _match(node, kind, to / obs.bb, all_in)
        if idx is None:
            return None
        w[seat] = w[seat] * node.strategy[idx]
        path.append(idx)
    return w


class RangeSolver(RangeTag):
    """range_tag, plus an exact heads-up river solve over estimated ranges."""

    NAME = "range_solver"
    ANCHOR = None

    SOLVE_ITERS = 500         # maximum DCFR iterations
    TARGET_EXPL = 0.002       # stop at 0.2% of the pot
    FLOP_SAMPLES = 32         # Monte Carlo runouts for flop range equity

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.nrng = np.random.default_rng(seed ^ 0x5EED5)
        self._river = None
        self.stats = {"solves": 0, "fallbacks": 0}

    def new_hand(self, info):
        self._river = None

    def act(self, obs):
        if obs.street == RIVER and obs.variant == "nlhe":
            try:
                a = self._river_action(obs)
            except Exception:  # noqa: BLE001 - never crash a table; fall back
                a = None
            if a is not None:
                return a
            self.stats["fallbacks"] += 1
        return super().act(obs)

    # -- river ---------------------------------------------------------------------------

    def _river_action(self, obs):
        live = [s for s in range(obs.n_seats) if int(obs.status[s]) in LIVE]
        if len(live) != 2 or obs.seat not in live:
            return None
        if self._river is None:
            self._river = self._solve(obs, live)
        if self._river is None:
            return None
        res, pot0, players, me = self._river
        t = res.tree
        node = 0
        for seat, street, kind, _amount, to, all_in in obs.history:
            if street != RIVER:
                continue
            if t.kind[node] != "decision" or players[int(t.player[node])] != seat:
                return None
            node = self._child(t, node, kind, to / pot0, all_in)
            if node is None:
                return None
        if t.kind[node] != "decision" or players[int(t.player[node])] != obs.seat:
            return None
        hole = sorted((int(obs.hole[0]), int(obs.hole[1])), reverse=True)
        probs = res.strategy_at(node)[:, COMBO_INDEX[tuple(hole)]].astype(np.float64)
        if probs.sum() <= 0:
            return None
        k = int(self.nrng.choice(len(probs), p=probs / probs.sum()))
        child = list(t.children(node))[k]
        return self._to_action(obs, t, child, me, pot0)

    @staticmethod
    def _child(t, node, kind, to_pot, all_in):
        kids = list(t.children(node))
        edges = [t.edge[c] for c in kids]
        if kind in ("check", "call", "fold"):
            for c, e in zip(kids, edges):
                if e == kind:
                    return c
            return None
        sized = [(c, e) for c, e in zip(kids, edges) if e.split()[0] in ("bet", "raise", "allin")]
        if not sized:
            return None
        if all_in:
            for c, e in sized:
                if e.startswith("allin"):
                    return c
        p = int(t.player[node])
        return min(sized, key=lambda ce: abs(t.contrib[ce[0]][p] - to_pot))[0]

    @staticmethod
    def _to_action(obs, t, child, me, pot0):
        la = obs.legal
        e = t.edge[child]
        if e in ("check", "call"):
            return Action.check() if la.can_check else Action.call()
        if e == "fold":
            return Action.check() if la.can_check else Action.fold()
        if not la.can_raise:
            return Action.check() if la.can_check else Action.call()
        if e.startswith("allin"):
            return Action.raise_to(la.max_raise_to)
        return clamp_raise(obs, round(float(t.contrib[child][me]) * pot0))

    def _solve(self, obs, live):
        from river_net.solver import Spot, solve

        n = obs.n_seats
        board = [int(c) for c in obs.board]
        pot0 = int(obs.pot) - int(np.sum(obs.street_contrib))
        behind = [int(obs.stacks[s]) + int(obs.street_contrib[s]) for s in live]
        if pot0 <= 0 or min(behind) <= 0:
            return None
        order = [(obs.button + k) % n for k in range(1, n + 1)]
        players = [s for s in order if s in live]  # [OOP, IP]
        me = players.index(obs.seat)
        ranges = self._river_ranges(obs, players, board)
        if ranges is None:
            return None
        spot = Spot(np.array(board, dtype=np.uint8), float(pot0), float(min(behind)),
                    ranges[0], ranges[1])
        res = solve(spot, iters=self.SOLVE_ITERS, target_expl=self.TARGET_EXPL, strategy=True)
        self.stats["solves"] += 1
        return res, float(pot0), players, me

    def _river_ranges(self, obs, players, board):
        w169 = preflop_ranges(obs)
        r = {}
        for s in players:
            w = w169[s] if w169 is not None else np.ones(169)
            if w.sum() <= 0:
                w = np.ones(169)
            r[s] = preflop.expand(w, dead=board)
        a, b = players
        for street, nb in ((1, 3), (2, 4)):
            facing = False
            for seat, st, kind, _amount, _to, _ai in obs.history:
                if st != street or seat not in r or kind in ("ante", "sb", "bb"):
                    continue
                if kind == "fold":
                    continue
                other = b if seat == a else a
                opp = r[other]
                if opp.sum() <= 0 or r[seat].sum() <= 0:
                    return None
                eq = _native.solver_range_equity(
                    board[:nb], (opp / opp.sum()).astype(np.float32), self.FLOP_SAMPLES,
                    int(self.nrng.integers(1 << 62)))
                q = _percentiles(eq, r[seat])
                r[seat] = r[seat] * _filter(kind, q, facing)
                if kind in ("bet", "raise"):
                    facing = True
        out = []
        for s in players:
            v = r[s]
            if v.sum() <= 0:
                return None
            out.append((v / v.sum()).astype(np.float32))
        return out
