"""Exact river solver: Python interface to the Rust core (``crates/solver``).

Value convention
================

A river spot is ``(board[5], pot P, effective stack S, r1[1326], r2[1326])``. Player 1 is out
of position and acts first. Ranges are nonnegative, zero on blocked combos and normalised to
sum to 1 (``solve`` renormalises over the valid combos).

All amounts are in units of the starting river pot ``P``; the solver only sees ``S / P``.
For a deal ``(h1, h2)`` ending at a terminal::

    u1 = (chips player 1 collects from the final pot − chips player 1 put in on the river)/P − ½
    u2 = −u1

so a fold by player f gives ``u_f = −½ − c_f`` (c_f = f's river chips / P) and the other
player ``½ + c_f``; a showdown with river contributions c each gives the winner ``½ + c``,
the loser ``−½ − c``, a tie 0.

* ``c(h, h') = 1`` if the combos share no card (and neither is blocked), else 0.
* ``m_i(h) = Σ_h' c(h,h') r_{-i}(h')`` — compatible opponent mass.
* ``v_i(h) = (1/m_i(h)) Σ_h' c(h,h') r_{-i}(h') E_σ̄[u_i | h, h']`` — the conditional per-hand
  value under the solver's average strategy σ̄. This is the network's target. It is defined
  for every valid hand, including hands with ``r_i(h) = 0`` (their average strategy is the
  average of their own counterfactual-regret play), and 0 on blocked combos.
* ``ṽ_i(h) = m_i(h) v_i(h)`` — counterfactual value; ``Σ r1 ṽ1 + Σ r2 ṽ2 = 0`` (zero sum).
* Exploitability: ``½ (Σ r1 BR̃1 + Σ r2 BR̃2)`` in pot units (×100 for % of pot).

Default bet abstraction: bets {½ pot, pot, all-in}, raises {pot-size raise, all-in}, at most
3 bets/raises, any size leaving less than ⅓ of the pre-action stack behind becomes all-in.
Solver: vectorized public-tree DCFR (α, β, γ) = (1.5, 0, 2), alternating updates,
exploitability checked every 50 iterations, stop below ``target_expl`` or at ``max_iters``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pokerarena import _native

from .cards import N_COMBOS, parse_board
from .tree import DEFAULT_ABSTRACTION, Tree

__all__ = ["DEFAULT_ABSTRACTION", "Spot", "SolveResult", "solve", "solve_abstract"]


@dataclass(eq=False)
class Spot:
    """A river spot. ``pot`` and ``stack`` share a unit (e.g. big blinds)."""

    board: np.ndarray
    pot: float
    stack: float
    r1: np.ndarray
    r2: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def spr(self) -> float:
        return float(self.stack) / float(self.pot)


@dataclass(eq=False)
class SolveResult:
    """Solver output. Per-hand arrays have 1326 entries (river) or one per abstract hand."""

    v1: np.ndarray
    v2: np.ndarray
    m1: np.ndarray
    m2: np.ndarray
    cfv1: np.ndarray
    cfv2: np.ndarray
    expl: float
    iters: int
    history: list
    tree: Optional[Tree] = None
    strategy1: Optional[np.ndarray] = None
    """Average strategy of player 1, ``[slot, action, hand]`` (actions past a node's count
    are unused)."""
    strategy2: Optional[np.ndarray] = None

    def strategy_at(self, node: int) -> np.ndarray:
        """``[action, hand]`` average strategy at a decision node."""
        t = self.tree
        s = (self.strategy1, self.strategy2)[int(t.player[node])]
        return s[int(t.slot[node]), : int(t.n_children[node])]


def _result(d: dict, n: int) -> SolveResult:
    tree = Tree.from_dict(d["tree"]) if "tree" in d else None
    strat = [None, None]
    if tree is not None:
        for p in range(2):
            strat[p] = d[f"strategy{p + 1}"].reshape(tree.n_slots[p], tree.max_actions, n)
    return SolveResult(d["v1"], d["v2"], d["m1"], d["m2"], d["cfv1"], d["cfv2"],
                       float(d["expl"]), int(d["iters"]), list(d["history"]), tree,
                       strat[0], strat[1])


def solve(spot: Spot, iters: int = 500, target_expl: float = 0.002, check_every: int = 50,
          abstraction: dict | None = None, strategy: bool = False) -> SolveResult:
    """Solves a river spot. ``iters`` is the maximum; set ``target_expl=0`` to run them all.
    Releases the GIL."""
    r1 = np.ascontiguousarray(spot.r1, dtype=np.float32)
    r2 = np.ascontiguousarray(spot.r2, dtype=np.float32)
    if r1.shape != (N_COMBOS,) or r2.shape != (N_COMBOS,):
        raise ValueError("ranges must have 1326 entries")
    d = _native.solver_river_solve(parse_board(spot.board).tolist(), float(spot.spr), r1, r2,
                                   abstraction, iters, target_expl, check_every, strategy)
    return _result(d, N_COMBOS)


def solve_abstract(strength, r1, r2, stack: float, cards=None, abstraction: dict | None = None,
                   iters: int = 500, target_expl: float = 0.0, check_every: int = 50
                   ) -> SolveResult:
    """Solves an abstract game: ``strength[i]`` (higher wins), optional ``cards[i] = (a, b)``
    for card removal (None: every pair of hands is compatible). Ranges are used as given.
    Arrays come back in the input hand order, with the average strategy and tree."""
    strength = [int(s) for s in strength]
    n = len(strength)
    r1 = np.ascontiguousarray(r1, dtype=np.float32)
    r2 = np.ascontiguousarray(r2, dtype=np.float32)
    cards = None if cards is None else [tuple(int(x) for x in c) for c in cards]
    d = _native.solver_abstract_solve(strength, r1, r2, float(stack), cards, abstraction,
                                      iters, target_expl, check_every, True)
    return _result(d, n)
