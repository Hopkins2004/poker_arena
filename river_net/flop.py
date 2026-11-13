"""Flop solver: public-tree DCFR on the flop with turn leaves valued by the turn value network.

The same principle as the turn solver (:mod:`river_net.turn`), one street earlier. A flop
spot is ``(board[3], pot P, effective stack S, r1, r2)``, player 1 out of position,
amounts in units of the flop pot, ``u1 = (collected − put in on the flop, turn and
river)/P − ½``. The tree is the river/turn bet abstraction at the flop SPR. Terminals:

* **fold**: as on every street;
* **all-in** (both all-in on the flop): exact showdown over the turn + river runouts,
  ``(½ + S)/990 · Σ_runouts net(h)`` (one product with the flop showdown matrix
  ``_native.solver_flop_showdown_matrix``; every deal sees C(45, 2) = 990 runouts);
* **turn leaf** (flop betting closed with chips behind): for each of the 49 turn cards the
  turn value network gives conditional turn values ``v_t(h)`` (turn-pot units) for the
  leaf's ranges conditioned on the card. A turn pot is ``1 + 2c`` flop pots, so
  ``cfv(h) = (1 + 2c)/45 · Σ_t m_t(h) · v_t(h)`` (each deal sees 45 turn cards), with the
  turn SPR ``(S − c)/(1 + 2c)``. ``turn_samples=m`` evaluates ``m`` random turn cards per
  iteration (scaled by 49/m, unbiased), like the turn solver's river sampling.

Turn value network: :class:`TurnValueNet` (the river network's architecture on turn
buckets, :func:`river_net.preriver.turn_bucket_map`), trained on turn spots solved by the
turn solver with river-network leaves (``python -m river_net.datagen --kind turn``).

:func:`turn_continuation_values` re-solves every flop leaf × turn card with the turn solver
(the "exact" continuation used to score flop strategies, :mod:`river_net.flop_check`).
"""

from __future__ import annotations

import numpy as np

from pokerarena import _native

from .cards import HAND_CARDS, N_COMBOS
from .preriver import DEFAULT_SCHEME, PreRiverNet, turn_bucket_map, turn_features
from .turn import NetworkLeaf, TurnSolver, compat_mass

DEALS_PER_TURN = 45.0   # turn cards each (h, h') deal sees on a flop
RUNOUTS_PER_DEAL = 990.0  # unordered turn + river runouts each deal sees


class TurnValueNet(PreRiverNet):
    """A trained turn value network: ``[r̄1, r̄2, SPR, log(1 + SPR)]`` on turn buckets →
    conditional turn values per bucket (turn-pot units), zero-sum projected. Same wrapper
    as the pre-river network (both are :class:`river_net.model.RiverNet` on turn buckets);
    ``predict(board[4], pot, stack, r1, r2)`` gives per-hand turn values."""


class FlopBoards:
    """The 49 turn cards of a flop board and its all-in showdown matrix (built lazily)."""

    def __init__(self, board):
        self.board = [int(c) for c in board]
        self.cards = [c for c in range(52) if c not in self.board]
        self._W = None

    def showdown_matrix(self) -> np.ndarray:
        if self._W is None:
            self._W = np.asarray(_native.solver_flop_showdown_matrix(self.board))
        return self._W


class FlopSolver(TurnSolver):
    """DCFR on a flop spot; leaves valued per turn card (``turn_samples`` per iteration).

    Same API and options as :class:`river_net.turn.TurnSolver` (``refresh``, resumable
    ``solve``, ``set_leaf``); ``turn_samples`` is its ``river_samples``."""

    BOARD_CARDS = 3
    ALLIN_DEALS = RUNOUTS_PER_DEAL

    def _make_boards(self):
        return FlopBoards(self.board)

    def __init__(self, spot, leaf, abstraction=None, turn_samples: int | None = None,
                 seed: int = 0, **options):
        super().__init__(spot, leaf, abstraction, river_samples=turn_samples, seed=seed,
                         **options)


class TurnNetLeaf(NetworkLeaf):
    """Flop leaves valued by a :class:`TurnValueNet` on every turn card (or a sample)."""

    name = "turn-network"
    deals = DEALS_PER_TURN

    def bind(self, boards: FlopBoards, stack: float):
        self.boards = boards
        self.stack = stack
        scheme = getattr(self.net, "scheme", DEFAULT_SCHEME)
        maps = []
        for t in boards.cards:
            b4 = np.array(boards.board + [t], dtype=np.uint8)
            maps.append(turn_bucket_map(b4, self.K, scheme, turn_features(b4)))
        self.bmap = np.stack(maps)                                         # [49, 1326]
        self.valid = (self.bmap >= 0).astype(np.float64)
        self.index = _native.LeafIndex(list(boards.cards),
                                       self.bmap.astype(np.int32).ravel(), self.K)
        self._dense = None

    def card_values(self, R1, R2, c, root=None):
        """Per-turn-card conditional values ``V[p][l, card, h]`` (turn-pot units) and
        opponent masses ``M[p][l, card, h]`` (diagnostics; one query per card row)."""
        L, C, K = len(c), len(self.boards.cards), self.K
        y, _ = self.query(R1, R2, c, root)
        y = y.reshape(L, C, 2 * K)
        V = [np.zeros((L, C, N_COMBOS)), np.zeros((L, C, N_COMBOS))]
        for k in range(C):
            ok = self.bmap[k] >= 0
            g = np.maximum(self.bmap[k], 0)
            for p in range(2):
                V[p][:, k] = np.where(ok, y[:, k, p * K + g], 0.0)
        R = [np.asarray(R1, dtype=np.float64), np.asarray(R2, dtype=np.float64)]
        M = [card_masses(R[1 - p], self.boards.cards) for p in range(2)]
        return V, M


class CheckdownFlopLeaf:
    """Flop leaves as a showdown with no turn or river betting (exact, zero-sum baseline)."""

    name = "check-down"

    def bind(self, boards: FlopBoards, stack: float):
        self.boards = boards

    def values(self, R1, R2, c):
        W = self.boards.showdown_matrix().reshape(N_COMBOS, N_COMBOS)
        k = (0.5 + np.asarray(c, dtype=np.float64))[:, None] / RUNOUTS_PER_DEAL
        n1 = (np.asarray(R2, dtype=np.float32) @ W.T).astype(np.float64)
        n2 = (np.asarray(R1, dtype=np.float32) @ W.T).astype(np.float64)
        return k * n1, k * n2


def card_masses(q: np.ndarray, cards) -> np.ndarray:
    """Opponent mass compatible with each combo on ``board + t``: ``[L, cards, 1326]``."""
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    out = np.zeros((q.shape[0], len(cards), N_COMBOS))
    for k, t in enumerate(cards):
        dead = (HAND_CARDS == t).any(axis=1)
        out[:, k] = np.where(dead, 0.0, compat_mass(np.where(dead, 0.0, q)))
    return out


class FixedTurnLeaf:
    """Flop leaves with fixed per-turn-card conditional values ``V[p][l, card, h]``."""

    name = "fixed"

    def __init__(self, V):
        self.V = V

    def bind(self, boards, stack):
        self.boards = boards

    def values(self, R1, R2, c):
        scale = (1 + 2 * np.asarray(c, dtype=np.float64))[:, None] / DEALS_PER_TURN
        R = [R1, R2]
        return tuple(scale * (card_masses(R[1 - p], self.boards.cards) * self.V[p]).sum(axis=1)
                     for p in range(2))


def turn_continuation_values(fs: FlopSolver, R1, R2, pool, iters=150, river_samples=8):
    """Turn-solver values ``V[p][l, card, h]`` at every flop leaf × turn card for the leaf
    ranges ``R1, R2`` ([L, 1326]): each turn spot is solved by the turn solver with
    river-network leaves. Returns ``(V, median own-model exploitability)``."""
    L, C = len(fs.leaves), len(fs.boards.cards)
    V = [np.zeros((L, C, N_COMBOS)), np.zeros((L, C, N_COMBOS))]
    jobs, keys = [], []
    for l in range(L):
        c = fs.leaf_c[l]
        for k, t in enumerate(fs.boards.cards):
            dead = (HAND_CARDS == t).any(axis=1)
            r = []
            for p, R in enumerate((R1[l], R2[l])):
                x = np.where(dead, 0.0, R)
                if x.sum() <= 1e-12:
                    x = np.where(dead, 0.0, fs.r[p])
                r.append((x / x.sum()).astype(np.float32))
            jobs.append((fs.boards.board + [t], 1 + 2 * c, fs.S - c, r[0], r[1], iters,
                         river_samples, int(1000 * l + k)))
            keys.append((l, k))
    out = pool.map(_turn_job, jobs, chunksize=4)
    expl = []
    for (l, k), (v1, v2, e) in zip(keys, out):
        V[0][l, k], V[1][l, k] = v1, v2
        expl.append(e)
    return V, float(np.median(expl)) if expl else 0.0


_RIVER_NET = None


def _turn_job(args):
    global _RIVER_NET
    board, pot, stack, r1, r2, iters, samples, seed = args
    import torch

    from .model import RiverValueNet
    from .solver import Spot

    torch.set_num_threads(1)
    if _RIVER_NET is None:
        _RIVER_NET = RiverValueNet.load("river_net/models/river_k64.pt")
    spot = Spot(np.asarray(board, dtype=np.uint8), pot, stack, r1, r2)
    ts = TurnSolver(spot, NetworkLeaf(_RIVER_NET), river_samples=samples, seed=seed)
    res = ts.solve(iters=iters, target_expl=0.0, check_every=iters)
    return res.v1, res.v2, res.expl
