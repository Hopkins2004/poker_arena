"""Preflop range database: 6-max 100bb NLHE strategy by hand class, and the ranges it implies.

Data: ``ranges/nlhe_6max_100bb.json``, built by ``ranges/build_preflop.py`` from the Leakproof
preflop ranges (6max-100bb-norake 1.0.0, https://github.com/a-hanamoto/leakproof-preflop-ranges,
CC BY 4.0, https://creativecommons.org/licenses/by/4.0/; converted, strategy unchanged).

Quick use::

    from pokerarena import preflop
    db = preflop.load()

    db.node("", "BTN").freqs("A5s")            # {'fold': 0.0, 'raise 2.5bb': 1.0}
    db.node("BTN raise 2.5bb", "BB").freqs("K9o")
    spot = db.spot("BTN raise 2.5bb, BB call")  # flop spot: pot, stacks, both ranges
    spot.ranges["BB"]                           # np.float64[169], class weights
    spot.combos("BTN", dead=board)             # np.float64[1326], combo weights

    # Inside a bot (6-handed NLHE tables, preflop):
    d = db.locate(obs)
    if d is not None and d.node is not None:
        return d.sample(obs, rng)              # a legal pokerarena.Action

Hand classes are strings like ``"AA"``, ``"AKs"``, ``"T9o"``; :data:`CLASSES` fixes their
order (13x13 grid, A first; suited above the diagonal). Combos are the 1326 two-card hands
``(hi, lo)`` with card ints ``rank * 4 + suit``, in :data:`COMBOS` order.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from ._native import Action
from .catalog import REPO_ROOT
from .helpers import clamp_raise

__all__ = [
    "CLASSES", "COMBOS", "COMBO_CLASS", "DEFAULT_PATH", "Decision", "Node", "PreflopAction",
    "PreflopDB", "Spot", "class_index", "class_name", "combo_counts", "expand", "load",
    "range_percent",
]

DEFAULT_PATH = REPO_ROOT / "ranges" / "nlhe_6max_100bb.json"
RANKS = "23456789TJQKA"


def _class_names() -> tuple:
    out = []
    for i in range(12, -1, -1):
        for j in range(12, -1, -1):
            if i == j:
                out.append(RANKS[i] * 2)
            elif i > j:
                out.append(RANKS[i] + RANKS[j] + "s")
            else:
                out.append(RANKS[j] + RANKS[i] + "o")
    return tuple(out)


CLASSES: tuple = _class_names()
"""The 169 hand classes, in database order."""
_INDEX = {c: i for i, c in enumerate(CLASSES)}


def class_index(a, b=None) -> int:
    """Index in :data:`CLASSES` of a class name (``"AKs"``) or of two cards (ints 0..52)."""
    if b is None:
        if isinstance(a, str):
            return _INDEX[a]
        a, b = (int(x) for x in a)
    a, b = int(a), int(b)
    ra, rb = a >> 2, b >> 2
    hi, lo = max(ra, rb), min(ra, rb)
    if ra == rb:
        return (12 - hi) * 14
    if (a & 3) == (b & 3):
        return (12 - hi) * 13 + (12 - lo)
    return (12 - lo) * 13 + (12 - hi)


def class_name(a, b=None) -> str:
    """Class name of two cards (``class_name(51, 47) == "AKs"``) or of an index."""
    if b is None and isinstance(a, (int, np.integer)):
        return CLASSES[int(a)]
    return CLASSES[class_index(a, b)]


def _combos():
    pairs = [(hi, lo) for hi in range(52) for lo in range(hi)]
    arr = np.array(pairs, dtype=np.uint8)
    cls = np.array([class_index(h, l) for h, l in pairs], dtype=np.int16)
    return arr, cls


COMBOS, COMBO_CLASS = _combos()
COMBOS.setflags(write=False)
COMBO_CLASS.setflags(write=False)
"""``COMBOS[i] = (hi, lo)`` card ints (hi > lo); ``COMBO_CLASS[i]`` = its class index."""


def combo_counts() -> np.ndarray:
    """Combos per class (6 pairs, 4 suited, 12 offsuit), in class order."""
    return np.bincount(COMBO_CLASS, minlength=169).astype(np.float64)


def expand(weights, dead: Iterable[int] = ()) -> np.ndarray:
    """Class weights (169) -> combo weights (1326), zeroing combos that use a dead card."""
    w = np.asarray(weights, dtype=np.float64)[COMBO_CLASS]
    dead = [int(c) for c in dead]
    if dead:
        blocked = np.isin(COMBOS, dead).any(axis=1)
        w = np.where(blocked, 0.0, w)
    return w


def range_percent(weights) -> float:
    """Share of the 1326 starting combos in a class-weight range, in percent."""
    return float(np.dot(weights, combo_counts()) / 1326 * 100)


# ----------------------------------------------------------------------------- types

@dataclass(frozen=True)
class PreflopAction:
    """One action of the tree: ``kind`` is fold / call / raise / allin."""

    kind: str
    label: str
    to_bb: Optional[float] = None
    """Raise-to total in big blinds (raises only)."""


@dataclass(frozen=True, eq=False)
class Node:
    """One decision: ``actor`` acts after ``line`` (voluntary actions, folds omitted)."""

    id: int
    path: tuple
    """Action index of every decision so far, folds included (the source's ``history``)."""
    actor: str
    line: str
    raises: int
    pot_bb: float
    to_call_bb: float
    actions: tuple
    children: tuple
    """Next decision id per action, or None (terminal, or followed only by implied folds)."""
    reach: np.ndarray
    """How often the actor's own earlier actions bring each class here (169)."""
    strategy: np.ndarray
    """``strategy[a, c]``: frequency of action ``a`` with class ``c``. Rows of classes that
    cannot reach the node are zero."""

    def freqs(self, hand) -> Optional[dict]:
        """``{action label: frequency}`` for a class name or two cards; None if the class
        cannot reach this decision under the strategy."""
        c = class_index(hand) if isinstance(hand, str) else class_index(*hand)
        if self.reach[c] <= 0:
            return None
        return {a.label: float(f) for a, f in zip(self.actions, self.strategy[:, c])}

    def range(self, action: Optional[int] = None) -> np.ndarray:
        """The actor's class weights arriving here, or after taking ``action``."""
        if action is None:
            return self.reach.copy()
        return self.reach * self.strategy[action]

    def action_index(self, label_or_kind: str) -> int:
        for i, a in enumerate(self.actions):
            if label_or_kind in (a.label, a.kind):
                return i
        raise KeyError(f"{label_or_kind!r} not in {[a.label for a in self.actions]}")

    def __repr__(self) -> str:
        return f"Node({self.id}, {self.actor} after {self.line or 'nothing'!r})"


@dataclass(frozen=True, eq=False)
class Spot:
    """A line that ends preflop with two players in: a flop (``allin`` False) or an all-in."""

    line: str
    node: int
    action: int
    players: tuple
    """Positions in postflop order (first acts first, i.e. is out of position)."""
    pot_bb: float
    stack_bb: float
    """Chips behind for each player, in big blinds."""
    allin: bool
    ranges: dict
    """Position -> class weights (169)."""

    def combos(self, position: str, dead: Iterable[int] = ()) -> np.ndarray:
        """1326 combo weights of a player's range, minus combos using ``dead`` cards."""
        return expand(self.ranges[position], dead)

    def __repr__(self) -> str:
        return f"Spot({self.line!r}, pot {self.pot_bb}bb, {self.stack_bb}bb behind)"


POSITION_BY_OFFSET = {1: "SB", 2: "BB", 3: "UTG", 4: "HJ", 5: "CO", 0: "BTN"}


@dataclass(frozen=True, eq=False)
class Decision:
    """Where an arena observation sits in the tree (see :meth:`PreflopDB.locate`)."""

    node: Optional[Node]
    """The current decision, or None when the tree folds here implicitly (a third player
    entering a pot the tree keeps heads-up)."""
    position: str
    positions: dict
    """Seat -> position name."""
    approx: bool
    """True if an earlier bet size was mapped to the nearest size in the tree."""

    def freqs(self, obs) -> Optional[dict]:
        if self.node is None:
            return {"fold": 1.0}
        return self.node.freqs((int(obs.hole[0]), int(obs.hole[1])))

    def to_action(self, obs, index: int):
        """The arena action for tree action ``index`` at this decision."""
        la = obs.legal
        if self.node is None:
            return Action.check() if la.can_check else Action.fold()
        a = self.node.actions[index]
        if a.kind == "fold":
            return Action.check() if la.can_check else Action.fold()
        if a.kind == "call":
            return Action.check() if la.can_check else Action.call()
        if not la.can_raise:  # calling a shove ("allin" in the tree), or raising is closed
            return Action.call()
        if a.kind == "allin":
            return Action.raise_to(la.max_raise_to)
        return clamp_raise(obs, round(a.to_bb * obs.bb))

    def sample(self, obs, rng) -> Optional[Action]:
        """Samples the tree strategy for the bot's hand (``rng``: numpy Generator or
        ``random.Random``); None if the hand cannot be here under the strategy."""
        f = self.freqs(obs)
        if f is None:
            return None
        p = np.fromiter(f.values(), dtype=np.float64)
        u = rng.random()
        i = int(min(np.searchsorted(np.cumsum(p), u * p.sum(), side="right"), len(p) - 1))
        if self.node is None:
            return self.to_action(obs, 0)
        return self.to_action(obs, i)


# ----------------------------------------------------------------------------- database

class PreflopDB:
    """The loaded database. Use :func:`load` for the shared default instance."""

    def __init__(self, data: dict):
        if data.get("schema") != "pokerarena.preflop/1":
            raise ValueError(f"unknown preflop database schema {data.get('schema')!r}")
        if tuple(data["classes"]) != CLASSES:
            raise ValueError("database class order does not match pokerarena.preflop.CLASSES")
        self.game: dict = data["game"]
        self.source: dict = data["source"]
        self.positions: tuple = tuple(data["positions"])
        self.nodes: list = []
        for n in data["nodes"]:
            acts = tuple(PreflopAction(a["kind"], a["label"], a.get("to_bb"))
                         for a in n["actions"])
            reach = np.array(n["reach"], dtype=np.float64)
            strat = np.array(n["strategy"], dtype=np.float64)
            reach.setflags(write=False)
            strat.setflags(write=False)
            self.nodes.append(Node(n["id"], tuple(n["path"]), n["actor"], n["line"],
                                   n["raises"], n["pot_bb"], n["to_call_bb"], acts,
                                   tuple(n["children"]), reach, strat))
        self.spots: list = []
        for s in data["spots"]:
            ranges = {}
            for p, w in s["ranges"].items():
                arr = np.array(w, dtype=np.float64)
                arr.setflags(write=False)
                ranges[p] = arr
            self.spots.append(Spot(s["line"], s["node"], s["action"], tuple(s["players"]),
                                   s["pot_bb"], s["stack_bb"], s["allin"], ranges))
        self._by_path = {n.path: n for n in self.nodes}
        self._by_line = {(n.line, n.actor): n for n in self.nodes}
        self._spots = {s.line: s for s in self.spots}

    @classmethod
    def from_file(cls, path) -> "PreflopDB":
        return cls(json.loads(Path(path).read_text()))

    @property
    def root(self) -> Node:
        return self._by_path[()]

    def node(self, line: str, actor: str) -> Node:
        """The decision of ``actor`` after ``line`` (e.g. ``"UTG raise 2bb, BTN raise 6bb"``,
        ``"UTG"``); ``""`` is the unopened pot."""
        try:
            return self._by_line[(line, actor)]
        except KeyError:
            raise KeyError(f"no decision for {actor} after {line!r}") from None

    def at(self, path) -> Optional[Node]:
        """The decision at a path of action indices (None if there is none)."""
        return self._by_path.get(tuple(path))

    def spot(self, line: str) -> Spot:
        """A flop or all-in spot by its full line, e.g. ``"CO raise 2.5bb, BB call"``."""
        try:
            return self._spots[line]
        except KeyError:
            raise KeyError(f"no spot {line!r}") from None

    def flop_spots(self) -> list:
        return [s for s in self.spots if not s.allin]

    # -- arena observations -----------------------------------------------------------

    def locate(self, obs) -> Optional[Decision]:
        """Places a preflop observation in the tree.

        Returns None when the tree does not apply: not NLHE, not preflop, not exactly six
        occupied seats, or the hand left the tree (a limp, a cold call the tree does not
        allow, a check, ...). Bet sizes off the tree are mapped to the tree's raise (or to
        all-in when that is the only raise) and flagged ``approx``. Stack depth and blinds
        are not checked: the tree assumes 100bb and no ante.
        """
        if obs.variant != "nlhe" or obs.street != 0:
            return None
        n = obs.n_seats
        status = obs.status
        if n != 6 or any(int(status[s]) == 0 for s in range(n)):
            return None
        positions = {s: POSITION_BY_OFFSET[(s - obs.button) % n] for s in range(n)}
        bb = obs.bb
        path: list = []
        approx = False
        for seat, street, kind, _amount, to, all_in in obs.history:
            if street != 0 or kind in ("ante", "sb", "bb"):
                continue
            node = self._by_path.get(tuple(path))
            if node is None:  # implied fold in the tree
                if kind != "fold":
                    return None
                path.append(0)
                continue
            if node.actor != positions[seat]:
                return None
            idx, off = _match(node, kind, to / bb, all_in)
            if idx is None:
                return None
            approx |= off
            path.append(idx)
        node = self._by_path.get(tuple(path))
        me = positions[obs.seat]
        if node is not None and node.actor != me:
            return None
        return Decision(node, me, positions, approx)


def _match(node: Node, kind: str, to_bb: float, all_in: bool):
    """(tree action index, size approximated?) for an arena action at ``node``."""
    kinds = [a.kind for a in node.actions]
    if kind == "fold":
        return kinds.index("fold"), False
    if kind == "call":
        if all_in and "allin" in kinds and "call" not in kinds:
            return kinds.index("allin"), False   # calling a shove
        if "call" in kinds:
            return kinds.index("call"), False
        return None, False
    if kind in ("bet", "raise"):
        if all_in and "allin" in kinds:
            return kinds.index("allin"), False
        if "raise" in kinds:
            i = kinds.index("raise")
            return i, abs(node.actions[i].to_bb - to_bb) > 0.25
        if "allin" in kinds:
            return kinds.index("allin"), True
    return None, False


@lru_cache(maxsize=None)
def _load_default(path: str) -> PreflopDB:
    return PreflopDB.from_file(path)


def load(path=None) -> PreflopDB:
    """The preflop database (default: ``ranges/nlhe_6max_100bb.json`` in the repository).
    Loaded once per process and shared; nodes and ranges are read-only."""
    return _load_default(str(Path(path) if path else DEFAULT_PATH))
