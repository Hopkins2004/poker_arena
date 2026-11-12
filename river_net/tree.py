"""Python view of the solver's flat public tree (built in Rust).

Nodes are in breadth-first order, so each node's children are the contiguous range
``first_child[i] : first_child[i] + n_children[i]``. ``contrib[i] = (c1, c2)`` are the river
contributions in pot units; ``player[i]`` is the actor (decisions) or the folder (folds), with
0 = player 1 (out of position). ``slot[i]`` indexes a decision among its player's decisions,
which is how strategies are laid out: ``strategy[slot, action, hand]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pokerarena import _native

DEFAULT_ABSTRACTION = {
    "oop_bets": [0.5, 1.0],
    "ip_bets": [0.5, 1.0],
    "raises": [1.0],
    "allin": True,
    "max_bets": 3,
    "allin_threshold": 1.0 / 3.0,
}


@dataclass(frozen=True, eq=False)
class Tree:
    kind: tuple
    player: np.ndarray
    first_child: np.ndarray
    n_children: np.ndarray
    contrib: np.ndarray
    parent: np.ndarray
    slot: np.ndarray
    edge: tuple
    n_slots: tuple
    max_actions: int
    stack: float

    @classmethod
    def from_dict(cls, d: dict) -> "Tree":
        return cls(tuple(d["kind"]), d["player"], d["first_child"], d["n_children"],
                   d["contrib"].reshape(-1, 2), d["parent"], d["slot"], tuple(d["edge"]),
                   tuple(d["n_slots"]), int(d["max_actions"]), float(d["stack"]))

    def __len__(self) -> int:
        return len(self.kind)

    def children(self, i: int) -> range:
        f = int(self.first_child[i])
        return range(f, f + int(self.n_children[i]))

    def path(self, i: int) -> list:
        """Edge labels from the root to node ``i``."""
        out = []
        while int(self.parent[i]) != 0xFFFFFFFF:
            out.append(self.edge[i])
            i = int(self.parent[i])
        return out[::-1]

    def find(self, *edges: str) -> int:
        """Node reached by a sequence of edge-label prefixes, e.g. ``find("bet", "call")``."""
        i = 0
        for e in edges:
            i = next(c for c in self.children(i) if self.edge[c].startswith(e))
        return i


def build_tree(stack: float, abstraction: dict | None = None) -> Tree:
    """River tree for effective stack ``stack`` (pot units)."""
    return Tree.from_dict(_native.solver_build_tree(float(stack), abstraction))
