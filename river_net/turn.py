"""Turn solver: public-tree DCFR on the turn with river leaves valued by a leaf evaluator.

Same principle and conventions as the river solver (:mod:`river_net.solver`), one street
earlier: a turn spot is ``(board[4], pot P, effective stack S, r1, r2)``, player 1 is out of
position, amounts are in units of the turn pot, ``u1 = (collected − put in on turn and
river)/P − ½``, ``v_i(h)`` is the conditional per-hand value under the average strategy.

The turn tree is the river solver's tree (same bet abstraction). Its terminals:

* **fold**: ``u_f = −½ − c_f`` for the folder, ``½ + c_f`` for the other player;
* **all-in** (betting closed with both players all-in): exact showdown averaged over the
  river cards (``(½ + S) · Σ_card net_card(h) / 44``; every deal sees 44 river cards);
* **river leaf** (betting closed with chips behind): for each of the 48 river cards the
  leaf evaluator gives conditional river values ``v_card(h)`` (in river-pot units) for the
  leaf's ranges conditioned on the card. A river pot is ``1 + 2c`` turn pots and river
  payoffs convert as ``u_turn = (1 + 2c) · u_river``, so the leaf's counterfactual value is
  ``cfv(h) = (1 + 2c)/44 · Σ_card m_card(h) · v_card(h)``, with ``m_card(h)`` the opponent
  reach compatible with ``h`` and the card.

Leaf evaluators: :class:`NetworkLeaf` (the river value network, one batched forward pass per
iteration over every leaf × river card) and :class:`CheckdownLeaf` (showdown, no river
betting; exact and zero-sum, used for tests and as a baseline).

CFR: DCFR (α, β, γ) = (1.5, 0, 2) with **simultaneous** updates, so each iteration needs a
single leaf query for both players. Exploitability is measured in the turn game with the
leaf values fixed at the current average ranges (the standard depth-limited measure).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from pokerarena import _native

from .buckets import bucket_map
from .cards import HAND_CARDS, N_COMBOS, parse_board
from .solver import Spot
from .tree import Tree, build_tree

DEALS_PER_RIVER = 44.0  # river cards each (h, h') deal can see on a turn board


def _incidence() -> np.ndarray:
    a = np.zeros((N_COMBOS, 52), dtype=np.float64)
    a[np.arange(N_COMBOS), HAND_CARDS[:, 0]] = 1
    a[np.arange(N_COMBOS), HAND_CARDS[:, 1]] = 1
    return a


INCIDENCE = _incidence()


def compat_mass(q: np.ndarray) -> np.ndarray:
    """Compatible mass of opponent reach ``q`` ([..., 1326]) for every combo."""
    T = q.sum(axis=-1, keepdims=True)
    C = q @ INCIDENCE
    return T - C[..., HAND_CARDS[:, 0]] - C[..., HAND_CARDS[:, 1]] + q


# ------------------------------------------------------------------------- leaf evaluators

class CheckdownLeaf:
    """River leaves as a showdown with no river betting (exact, zero-sum)."""

    name = "check-down"

    def bind(self, boards, stack: float):
        self.boards = boards

    def values(self, R1, R2, c):
        """Counterfactual leaf values ``[L, 1326]`` for both players."""
        out1 = np.zeros_like(R1)
        out2 = np.zeros_like(R2)
        for l in range(len(c)):
            k = (0.5 + c[l]) / DEALS_PER_RIVER
            out1[l] = k * self.boards.showdown_net_sum(R2[l].astype(np.float32))
            out2[l] = k * self.boards.showdown_net_sum(R1[l].astype(np.float32))
        return out1, out2


class NetworkLeaf:
    """River leaves valued by a :class:`river_net.model.RiverValueNet`."""

    name = "network"

    def __init__(self, net, threads: int | None = None):
        self.net = net
        self.K = net.K
        if net.board_features:
            raise NotImplementedError("board-feature models are not supported as leaves yet")
        if threads:
            torch.set_num_threads(threads)
        self.queries = 0

    def bind(self, boards, stack: float):
        self.boards = boards
        self.stack = stack
        n_cards = len(boards.cards)
        st = np.asarray(boards.strengths()).reshape(n_cards, N_COMBOS)
        self.bmap = np.stack([bucket_map(s, self.K) for s in st])          # [C, 1326]
        self.valid = (self.bmap >= 0).astype(np.float64)                  # [C, 1326]
        onehot = np.zeros((n_cards, N_COMBOS, self.K), dtype=np.float64)
        c_idx, h_idx = np.nonzero(self.bmap >= 0)
        onehot[c_idx, h_idx, self.bmap[c_idx, h_idx]] = 1.0
        self.B = onehot                                                   # [C, 1326, K]
        self.Bflat = onehot.transpose(1, 0, 2).reshape(N_COMBOS, -1)      # [1326, C*K]
        self.BT = onehot.transpose(0, 2, 1)                               # [C, K, 1326]

    def values(self, R1, R2, c, root=None):
        V, M = self.card_values(R1, R2, c, root)
        scale = (1 + 2 * np.asarray(c)) / DEALS_PER_RIVER
        return tuple(scale[:, None] * (M[p] * V[p]).sum(axis=1) for p in range(2))

    def card_values(self, R1, R2, c, root=None):
        """Per-card conditional river values ``V[p][l, card, h]`` (river-pot units) and
        opponent masses ``M[p][l, card, h]``."""
        L = len(c)
        C, K = self.bmap.shape[0], self.K
        R = [np.asarray(R1, dtype=np.float64), np.asarray(R2, dtype=np.float64)]
        # Opponent masses per card: M[p][l, c, h] (opponent of p is 1 - p).
        M = [np.asarray(self.boards.masses(R[1 - p].astype(np.float32).ravel()),
                        dtype=np.float64).reshape(L, C, N_COMBOS) for p in range(2)]
        tot = [R[p] @ self.valid.T for p in range(2)]                       # [L, C]
        # Ranges with no mass on a card (e.g. an action nobody takes yet) fall back to the
        # root range so their counterfactual values stay defined.
        for p in range(2):
            empty = tot[p] <= 1e-12
            if empty.any() and root is not None:
                for l in np.unique(np.nonzero(empty)[0]):
                    R[p][l] = np.where(R[p][l].sum() > 0, R[p][l], root[p])
                tot[p] = R[p] @ self.valid.T
        safe = [np.where(t > 0, t, 1.0) for t in tot]
        rb = [(R[p] @ self.Bflat).reshape(L, C, K) / safe[p][..., None] for p in range(2)]
        # Projection weights w_p(k) = Σ_{h∈k} r_p(h) m_p(h), both normalised.
        w = []
        for p in range(2):
            rm = R[p][:, None, :] * M[p] / (safe[p] * safe[1 - p])[..., None]   # [L, C, H]
            w.append(np.matmul(rm.transpose(1, 0, 2), self.B).transpose(1, 0, 2))  # [L,C,K]
        spr = np.clip((self.stack - np.asarray(c)) / (1 + 2 * np.asarray(c)), 0, None)
        spr = np.broadcast_to(spr[:, None], (L, C))
        x = np.concatenate([rb[0], rb[1], spr[..., None], np.log1p(spr)[..., None]], axis=2)
        wcat = np.concatenate(w, axis=2)
        y = self.net.predict_buckets(x.reshape(L * C, -1).astype(np.float32),
                                     wcat.reshape(L * C, -1).astype(np.float32))
        y = y.reshape(L, C, 2 * K).astype(np.float64)
        self.queries += L * C
        V = []
        for p in range(2):
            vb = y[..., p * K:(p + 1) * K]                                  # [L, C, K]
            V.append(np.matmul(vb.transpose(1, 0, 2), self.BT).transpose(1, 0, 2))
        return V, M


class FixedLeaf:
    """River leaves with fixed per-card conditional values ``V[p][l, card, h]`` (e.g. from
    exact re-solves); counterfactual values follow the opponent reach through the masses."""

    name = "fixed"

    def __init__(self, V):
        self.V = V

    def bind(self, boards, stack: float):
        self.boards = boards

    def values(self, R1, R2, c):
        L, C = self.V[0].shape[:2]
        R = [R1, R2]
        M = [np.asarray(self.boards.masses(R[1 - p].astype(np.float32).ravel()),
                        dtype=np.float64).reshape(L, C, N_COMBOS) for p in range(2)]
        scale = (1 + 2 * np.asarray(c)) / DEALS_PER_RIVER
        return tuple(scale[:, None] * (M[p] * self.V[p]).sum(axis=1) for p in range(2))


# ------------------------------------------------------------------------- solver

@dataclass(eq=False)
class TurnResult:
    v1: np.ndarray
    v2: np.ndarray
    m1: np.ndarray
    m2: np.ndarray
    cfv1: np.ndarray
    cfv2: np.ndarray
    expl: float
    iters: int
    history: list
    tree: Tree
    strategy1: np.ndarray
    strategy2: np.ndarray
    seconds: float
    leaf_queries: int = 0
    meta: dict = field(default_factory=dict)

    def strategy_at(self, node: int) -> np.ndarray:
        s = (self.strategy1, self.strategy2)[int(self.tree.player[node])]
        return s[int(self.tree.slot[node]), : int(self.tree.n_children[node])]


class TurnSolver:
    """DCFR on a turn spot with a pluggable river-leaf evaluator."""

    def __init__(self, spot: Spot, leaf, abstraction: dict | None = None,
                 alpha=1.5, beta=0.0, gamma=2.0):
        self.board = parse_board(spot.board)
        if len(self.board) != 4:
            raise ValueError("a turn spot has a 4-card board")
        self.valid = ~np.isin(HAND_CARDS, self.board).any(axis=1)
        r = [np.where(self.valid, np.asarray(x, dtype=np.float64), 0.0) for x in (spot.r1, spot.r2)]
        if min(x.sum() for x in r) <= 0:
            raise ValueError("a range has no mass on valid combos")
        self.r = [x / x.sum() for x in r]
        self.S = float(spot.spr)
        self.tree = build_tree(self.S, abstraction)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        t = self.tree
        self.n = len(t)
        self.A = t.max_actions
        self.decisions = [i for i in range(self.n) if t.kind[i] == "decision"]
        sd = [i for i in range(self.n) if t.kind[i] == "showdown"]
        self.allin = [i for i in sd if t.contrib[i][0] >= self.S - 1e-6]
        self.leaves = [i for i in sd if t.contrib[i][0] < self.S - 1e-6]
        self.leaf_c = np.array([float(t.contrib[i][0]) for i in self.leaves])
        self.boards = _native.TurnBoards(self.board.tolist())
        self.leaf = leaf
        leaf.bind(self.boards, self.S)
        self.regret = [np.zeros((t.n_slots[p], self.A, N_COMBOS)) for p in range(2)]
        self.ssum = [np.zeros((t.n_slots[p], self.A, N_COMBOS)) for p in range(2)]
        self.m = [compat_mass(self.r[1]) * self.valid, compat_mass(self.r[0]) * self.valid]

    # -- strategies and reaches --------------------------------------------------------

    def _rm(self, p):
        t = self.tree
        out = np.zeros_like(self.regret[p])
        for i in self.decisions:
            if int(t.player[i]) != p:
                continue
            s, k = int(t.slot[i]), int(t.n_children[i])
            pos = np.maximum(self.regret[p][s, :k], 0)
            tot = pos.sum(axis=0)
            out[s, :k] = np.where(tot > 0, pos / np.where(tot > 0, tot, 1), 1.0 / k)
        return out

    def _avg(self, p):
        t = self.tree
        out = np.zeros_like(self.ssum[p])
        for i in self.decisions:
            if int(t.player[i]) != p:
                continue
            s, k = int(t.slot[i]), int(t.n_children[i])
            tot = self.ssum[p][s, :k].sum(axis=0)
            out[s, :k] = np.where(tot > 0, self.ssum[p][s, :k] / np.where(tot > 0, tot, 1),
                                  1.0 / k)
        return out

    def _seq(self, strat, p):
        """Sequence reach of player p at every node (root 1) under ``strat``."""
        t = self.tree
        reach = np.zeros((self.n, N_COMBOS))
        reach[0] = 1.0
        for i in self.decisions:
            for a, c in enumerate(t.children(i)):
                reach[c] = reach[i] * (strat[int(t.slot[i]), a] if int(t.player[i]) == p else 1)
        return reach

    # -- values ------------------------------------------------------------------------

    def _leaf_values(self, seq):
        if not self.leaves:
            return [np.zeros((0, N_COMBOS))] * 2
        R1 = self.r[0] * seq[0][self.leaves]
        R2 = self.r[1] * seq[1][self.leaves]
        if isinstance(self.leaf, NetworkLeaf):
            return self.leaf.values(R1, R2, self.leaf_c, root=self.r)
        return self.leaf.values(R1, R2, self.leaf_c)

    def leaf_ranges(self, strat=None):
        """Both players' reach ``[L, 1326]`` at the river leaves (default: average profile)."""
        strat = strat or [self._avg(0), self._avg(1)]
        seq = [self._seq(strat[0], 0), self._seq(strat[1], 1)]
        return self.r[0] * seq[0][self.leaves], self.r[1] * seq[1][self.leaves]

    def _backward(self, p, strat_p, opp_reach, leaf_cfv, mode):
        """cfv of player p at every node; ``mode``: "follow" | "br" | "cfr" (returns
        instantaneous regrets too)."""
        t = self.tree
        cfv = np.zeros((self.n, N_COMBOS))
        inst = np.zeros_like(self.regret[p]) if mode == "cfr" else None
        leaf_pos = {node: k for k, node in enumerate(self.leaves)}
        allin = set(self.allin)
        for i in range(self.n - 1, -1, -1):
            kind = t.kind[i]
            if kind == "fold":
                f = int(t.player[i])
                c = t.contrib[i]
                u = -0.5 - c[f] if f == p else 0.5 + c[f]
                cfv[i] = u * compat_mass(opp_reach[i])
            elif kind == "showdown":
                if i in allin:
                    q = opp_reach[i].astype(np.float32)
                    cfv[i] = (0.5 + self.S) / DEALS_PER_RIVER * np.asarray(
                        self.boards.showdown_net_sum(q), dtype=np.float64)
                else:
                    cfv[i] = leaf_cfv[leaf_pos[i]]
            else:
                kids = list(t.children(i))
                if int(t.player[i]) == p:
                    s = int(t.slot[i])
                    if mode == "br":
                        cfv[i] = np.max(cfv[kids], axis=0)
                    else:
                        st = strat_p[s, :len(kids)]
                        cfv[i] = (st * cfv[kids]).sum(axis=0)
                        if mode == "cfr":
                            inst[s, :len(kids)] = cfv[kids] - cfv[i]
                else:
                    cfv[i] = cfv[kids].sum(axis=0)
        return cfv, inst

    def _profile_values(self, strat, mode="follow"):
        seq = [self._seq(strat[0], 0), self._seq(strat[1], 1)]
        leaf = self._leaf_values(seq)
        out = []
        for p in range(2):
            o = 1 - p
            cfv, _ = self._backward(p, strat[p], self.r[o] * seq[o], leaf[p], mode)
            out.append(cfv[0] * self.valid)
        return out

    def exploitability(self) -> float:
        avg = [self._avg(0), self._avg(1)]
        br = self._profile_values(avg, mode="br")
        return 0.5 * float((self.r[0] * br[0]).sum() + (self.r[1] * br[1]).sum())

    # -- main loop ---------------------------------------------------------------------

    def solve(self, iters=300, target_expl=0.005, check_every=25) -> TurnResult:
        t0 = time.time()
        history = []
        expl = float("nan")
        t_done = 0
        for it in range(1, iters + 1):
            cur = [self._rm(0), self._rm(1)]
            seq = [self._seq(cur[0], 0), self._seq(cur[1], 1)]
            leaf = self._leaf_values(seq)
            ta, tb = it ** self.alpha, it ** self.beta
            dpos, dneg = ta / (ta + 1), tb / (tb + 1)
            dstrat = (it / (it + 1)) ** self.gamma
            for p in range(2):
                o = 1 - p
                _, inst = self._backward(p, cur[p], self.r[o] * seq[o], leaf[p], "cfr")
                self.regret[p] += inst
                self.regret[p] *= np.where(self.regret[p] > 0, dpos, dneg)
                t = self.tree
                for i in self.decisions:
                    if int(t.player[i]) == p:
                        s, k = int(t.slot[i]), int(t.n_children[i])
                        self.ssum[p][s, :k] = (self.ssum[p][s, :k]
                                               + seq[p][i] * cur[p][s, :k]) * dstrat
            t_done = it
            if it % check_every == 0 or it == iters:
                expl = self.exploitability()
                history.append((it, expl))
                if expl < target_expl:
                    break
        avg = [self._avg(0), self._avg(1)]
        cfv = self._profile_values(avg)
        v = [np.where(self.m[p] > 0, cfv[p] / np.where(self.m[p] > 0, self.m[p], 1), 0.0)
             for p in range(2)]
        return TurnResult(v[0], v[1], self.m[0], self.m[1], cfv[0], cfv[1], expl, t_done,
                          history, self.tree, avg[0], avg[1], time.time() - t0,
                          getattr(self.leaf, "queries", 0))


def solve_turn(spot: Spot, leaf, iters=300, target_expl=0.005, check_every=25,
               abstraction: dict | None = None) -> TurnResult:
    return TurnSolver(spot, leaf, abstraction).solve(iters, target_expl, check_every)
