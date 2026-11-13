"""Training-spot generator: river spots that look like rivers reached by sensible play.

Sources (mix configurable, default 70 / 20 / 10):

* **A. Simulated lines.** A preflop scenario from the imported chart (``pokerarena.preflop``,
  6-max 100bb), light jitter, a random board, then a flop line and a turn line, each action
  soft-filtering the acting player's range by equity percentile against the opponent's
  current range (``_native.solver_range_equity``: flop Monte Carlo, turn exact).
* **B. DeepStack recursive split.** Random board; per player, river-strength-sorted tie
  groups with probability mass split recursively at uniform random fractions; optionally
  mixed 50/50 with uniform.
* **C. Noisy baselines.** A scenario's chart ranges, filtered by the board only, times
  Dirichlet noise.

Pot and stack: scenario pot / stack (big blinds), grown by the sampled flop and turn lines
(source A filters ranges along the way; B and C only track the money).

Preflop scenarios follow the user's procedure: the opener is uniform over UTG, HJ, CO, BTN,
SB (earlier seats fold); each later seat then calls / 3-bets / folds with the combo-weighted
frequencies of its chart range, and so on through 4-bets; lines that end preflop are redrawn
with the same opener. The result is one of the chart's 38 heads-up flop spots (8 single
raised, 15 3-bet, 15 4-bet pots). No limped pots (the chart has none); stacks are the
chart's 100bb.

Spots: ranges zero on blocked combos and normalised; rejected (and redrawn) when either range
has fewer than 20 effective combos (``1 / Σ r²``) or the SPR is outside [0.1, 20].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pokerarena import _native, preflop

from .cards import HAND_CARDS, N_COMBOS
from .solver import Spot

SOURCES = ("A", "B", "C")
DEFAULT_MIX = {"A": 0.7, "B": 0.2, "C": 0.1}
LINES = ("cc", "bc", "cbc", "brc", "crc")
DEFAULT_LINE_WEIGHTS = {"cc": 0.30, "bc": 0.25, "cbc": 0.20, "brc": 0.10, "crc": 0.15}
BET_SIZES = (0.33, 0.5, 0.75, 1.0)
RAISE_MULT = 3.0
MIN_EFFECTIVE = 20.0
SPR_RANGE = (0.1, 20.0)
OPENERS = ("UTG", "HJ", "CO", "BTN", "SB")
FLOP_SAMPLES = 32


# ----------------------------------------------------------------------------- scenarios

@dataclass(frozen=True, eq=False)
class Scenario:
    """A heads-up flop spot of the chart."""

    line: str
    oop: str
    ip: str
    pot_bb: float
    stack_bb: float
    r_oop: np.ndarray   # 169 class weights
    r_ip: np.ndarray
    prob: float         # sampling probability (user's procedure)
    pot_type: str       # "srp" / "3bet" / "4bet"


def _action_probs(node) -> np.ndarray:
    """Combo-weighted share of the actor's range (arriving here) taking each action."""
    w = node.reach * preflop.combo_counts()
    tot = w.sum()
    return (node.strategy * w).sum(axis=1) / tot


def scenario_table(db=None) -> list:
    """The 38 flop scenarios with their sampling probabilities."""
    db = db or preflop.load()
    spot_of = {(s.node, s.action): s for s in db.flop_spots()}
    mass: dict = {}

    def walk(node_id, p, opener_mass):
        node = db.nodes[node_id]
        for a, pa in enumerate(_action_probs(node)):
            if pa <= 0:
                continue
            child = node.children[a]
            if child is not None:
                walk(child, p * pa, opener_mass)
            elif (node_id, a) in spot_of:
                key = spot_of[(node_id, a)].line
                opener_mass.setdefault(key, 0.0)
                opener_mass[key] += p * pa
            # else: the hand ended preflop (fold-out or all-in): redrawn

    for opener in OPENERS:
        node = db.root
        while node.actor != opener:  # earlier seats fold
            node = db.nodes[node.children[node.action_index("fold")]]
        opened = db.nodes[node.children[node.action_index("raise")]]
        m: dict = {}
        walk(opened.id, 1.0, m)
        z = sum(m.values())
        for k, v in m.items():  # redraw with the same opener: renormalise per opener
            mass[k] = mass.get(k, 0.0) + v / z / len(OPENERS)
    out = []
    for s in db.flop_spots():
        raises = s.line.count("raise")
        out.append(Scenario(s.line, s.players[0], s.players[1], s.pot_bb, s.stack_bb,
                            np.asarray(s.ranges[s.players[0]]),
                            np.asarray(s.ranges[s.players[1]]),
                            mass.get(s.line, 0.0), {1: "srp", 2: "3bet", 3: "4bet"}[raises]))
    return out


# ----------------------------------------------------------------------------- helpers

_GRID_NEIGHBOURS = None


def _grid_neighbours() -> np.ndarray:
    """169x169 adjacency of the 13x13 hand grid (the "edge" of a chart range)."""
    global _GRID_NEIGHBOURS
    if _GRID_NEIGHBOURS is None:
        g = np.zeros((169, 169), dtype=bool)
        for i in range(13):
            for j in range(13):
                for di, dj in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    a, b = i + di, j + dj
                    if 0 <= a < 13 and 0 <= b < 13:
                        g[i * 13 + j, a * 13 + b] = True
        _GRID_NEIGHBOURS = g
    return _GRID_NEIGHBOURS


def jitter_classes(w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light per-spot jitter of chart class weights: log-normal noise, a power transform
    (widen < 1 < tighten) and a little weight on zero classes bordering the range."""
    sigma = rng.uniform(0.05, 0.25)
    gamma = float(np.exp(rng.uniform(np.log(0.7), np.log(1.4))))
    out = np.power(np.clip(w, 0, 1), gamma) * np.exp(rng.normal(0, sigma, size=169))
    if rng.random() < 0.5:
        edge = (w == 0) & (_grid_neighbours() @ (w > 0))
        out = out + edge * rng.uniform(0, 0.3) * rng.random(169)
    return np.clip(out, 0, 1)


def expand(w169: np.ndarray, dead) -> np.ndarray:
    return preflop.expand(w169, dead=[int(c) for c in dead])


def normalise(r: np.ndarray) -> np.ndarray:
    s = r.sum()
    return r / s if s > 0 else r


def effective_combos(r: np.ndarray) -> float:
    s2 = float(np.square(r).sum())
    return 1.0 / s2 if s2 > 0 else 0.0


def percentiles(score: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted mid-rank percentile of each combo's score within the range ``w``."""
    q = np.zeros(N_COMBOS)
    idx = np.flatnonzero(w > 0)
    if idx.size == 0:
        return q
    order = idx[np.argsort(score[idx], kind="stable")]
    cw = np.cumsum(w[order])
    q[order] = (cw - 0.5 * w[order]) / cw[-1]
    return q


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def filter_aggressive(q, rng, raise_=False):
    """Bet / raise: value at the top plus bluffs from the bottom (polarized)."""
    t_hi = rng.uniform(0.70, 0.92) if raise_ else rng.uniform(0.45, 0.80)
    t_lo = rng.uniform(0.03, 0.20) if raise_ else rng.uniform(0.05, 0.30)
    beta = rng.uniform(0.15, 0.8)
    T = rng.uniform(0.03, 0.10)
    return np.clip(_sig((q - t_hi) / T) + beta * _sig((t_lo - q) / T), 0, 1)


def filter_call(q, rng, raise_possible=True):
    """Call: a bump over the middle; the top partly removed when raising was possible."""
    t_c = rng.uniform(0.15, 0.55)
    T = rng.uniform(0.03, 0.10)
    p = _sig((q - t_c) / T)
    if raise_possible:
        rho, t_top = rng.uniform(0.3, 0.8), rng.uniform(0.85, 0.97)
        p = p * (1 - rho * _sig((q - t_top) / T))
    return p


def filter_check(q, rng):
    """Check: capped (top percentiles down-weighted)."""
    kappa, t_cap = rng.uniform(0.5, 0.95), rng.uniform(0.6, 0.85)
    T = rng.uniform(0.03, 0.10)
    return 1 - kappa * _sig((q - t_cap) / T)


# ----------------------------------------------------------------------------- generator

@dataclass
class GeneratorConfig:
    mix: dict = field(default_factory=lambda: dict(DEFAULT_MIX))
    line_weights: dict = field(default_factory=lambda: dict(DEFAULT_LINE_WEIGHTS))
    scenario_weights: dict | None = None   # line -> weight; None = the user's procedure
    flop_samples: int = FLOP_SAMPLES
    max_attempts: int = 50


# street -> (board cards, street lines played, extra seed word)
STREETS = {"river": (5, 2, None), "turn": (4, 1, 4), "turn_end": (4, 2, 44)}


class RangeGenerator:
    """Deterministic: ``generate(i)`` depends only on ``(seed, i)`` and the config."""

    def __init__(self, seed: int = 0, config: GeneratorConfig | None = None):
        self.seed = int(seed)
        self._nb = 5     # board cards of the spot being generated (5 river, 4 turn)
        self._lines = 2  # street lines played before the spot (flop, turn)
        self.config = config or GeneratorConfig()
        self.scenarios = scenario_table()
        w = np.array([s.prob for s in self.scenarios])
        if self.config.scenario_weights is not None:
            w = np.array([self.config.scenario_weights.get(s.line, 0.0)
                          for s in self.scenarios])
        self.scenario_p = w / w.sum()
        mix = self.config.mix
        self.source_p = np.array([mix.get(s, 0.0) for s in SOURCES], dtype=float)
        self.source_p /= self.source_p.sum()
        lw = self.config.line_weights
        self.line_p = np.array([lw.get(l, 0.0) for l in LINES], dtype=float)
        self.line_p /= self.line_p.sum()

    # -- public ------------------------------------------------------------------------

    def generate(self, index: int, source: str | None = None, street: str = "river") -> Spot:
        """Spot number ``index`` (optionally forcing a source).

        * ``street="river"``: 5-card board, flop and turn lines played (river solver data);
        * ``street="turn"``: the same process cut one street earlier, 4-card board, only the
          flop line played (turn-solver spots; source B splits mass over turn equity vs a
          uniform range instead of river strength);
        * ``street="turn_end"``: 4-card board, flop and turn lines played, i.e. a river spot
          before the river card (pre-river network data, :mod:`river_net.preriver`).
        """
        if street not in STREETS:
            raise ValueError(f"street must be one of {STREETS}, got {street!r}")
        self._nb, self._lines, tag = STREETS[street]
        seed = [self.seed, int(index)] + ([tag] if tag is not None else [])
        rng = np.random.default_rng(seed)
        src = source or SOURCES[int(rng.choice(3, p=self.source_p))]
        for _ in range(self.config.max_attempts):
            spot = {"A": self._source_a, "B": self._source_b, "C": self._source_c}[src](rng)
            if spot is not None and self._accept(spot):
                spot.meta["index"] = int(index)
                return spot
        raise RuntimeError(f"no acceptable spot after {self.config.max_attempts} attempts")

    # -- pieces ------------------------------------------------------------------------

    def _scenario(self, rng) -> Scenario:
        return self.scenarios[int(rng.choice(len(self.scenarios), p=self.scenario_p))]

    def _sample_line(self, rng, pot, stack):
        """A street line feasible with the stack: (line, bet fraction)."""
        lines = list(LINES)
        p = self.line_p.copy()
        f = float(rng.choice(BET_SIZES))
        for k, line in enumerate(lines):
            need = f * pot * (RAISE_MULT if "r" in line else 1.0)
            if line != "cc" and need >= stack * 0.95:
                p[k] = 0.0
        p /= p.sum()
        return lines[int(rng.choice(len(lines), p=p))], f

    @staticmethod
    def _money(line, f, pot, stack):
        b = f * pot
        put = {"cc": 0.0, "bc": b, "cbc": b, "brc": RAISE_MULT * b, "crc": RAISE_MULT * b}[line]
        return pot + 2 * put, stack - put

    def _accept(self, spot: Spot) -> bool:
        if not (SPR_RANGE[0] <= spot.spr <= SPR_RANGE[1]):
            return False
        return min(effective_combos(spot.r1), effective_combos(spot.r2)) >= MIN_EFFECTIVE

    def _finish(self, board, pot, stack, r1, r2, meta) -> Spot | None:
        dead = np.isin(HAND_CARDS, board).any(axis=1)
        r1 = np.where(dead, 0.0, r1)
        r2 = np.where(dead, 0.0, r2)
        if r1.sum() <= 0 or r2.sum() <= 0:
            return None
        return Spot(board.astype(np.uint8), float(pot), float(stack),
                    normalise(r1).astype(np.float32), normalise(r2).astype(np.float32), meta)

    def _money_path(self, rng, sc):
        pot, stack = sc.pot_bb, sc.stack_bb
        lines = []
        for _ in range(self._lines):
            line, f = self._sample_line(rng, pot, stack)
            pot, stack = self._money(line, f, pot, stack)
            lines.append(line if line == "cc" else f"{line}{f:g}")
        return pot, stack, "/".join(lines)

    def _source_a(self, rng) -> Spot | None:
        sc = self._scenario(rng)
        board = rng.choice(52, size=5, replace=False).astype(np.uint8)
        r = [expand(jitter_classes(sc.r_oop, rng), board[:3]),
             expand(jitter_classes(sc.r_ip, rng), board[:3])]
        pot, stack = sc.pot_bb, sc.stack_bb
        lines = []
        for street, nb in ((1, 3), (2, 4))[: self._lines]:
            vis = board[:nb]
            if nb == 4:  # the turn card is now dead
                turn_dead = (HAND_CARDS == board[3]).any(axis=1)
                r = [np.where(turn_dead, 0.0, x) for x in r]
            line, f = self._sample_line(rng, pot, stack)
            r = self._apply_line(rng, line, vis, r)
            if r is None:
                return None
            pot, stack = self._money(line, f, pot, stack)
            lines.append(line if line == "cc" else f"{line}{f:g}")
        return self._finish(board[: self._nb], pot, stack, r[0], r[1],
                            {"source": "A", "scenario": sc.line, "pot_type": sc.pot_type,
                             "line": "/".join(lines)})

    def _equity_pct(self, rng, board, me, opp):
        eq = _native.solver_range_equity(board.tolist(), normalise(opp).astype(np.float32),
                                         self.config.flop_samples, int(rng.integers(1 << 62)))
        return percentiles(eq, me)

    def _apply_line(self, rng, line, board, r):
        """Applies a street line's actions as soft filters; r = [oop, ip]."""
        steps = {
            "cc": [(0, "check"), (1, "check")],
            "bc": [(0, "bet"), (1, "call")],
            "cbc": [(0, "check"), (1, "bet"), (0, "call")],
            "brc": [(0, "bet"), (1, "raise"), (0, "call")],
            "crc": [(0, "check"), (1, "bet"), (0, "raise"), (1, "call")],
        }[line]
        r = [x.copy() for x in r]
        for k, (p, act) in enumerate(steps):
            if r[0].sum() <= 0 or r[1].sum() <= 0:
                return None
            q = self._equity_pct(rng, board, r[p], r[1 - p])
            if act == "check":
                r[p] *= filter_check(q, rng)
            elif act == "bet":
                r[p] *= filter_aggressive(q, rng)
            elif act == "raise":
                r[p] *= filter_aggressive(q, rng, raise_=True)
            else:
                # A raise is possible when calling a bet; after a raise, less so.
                r[p] *= filter_call(q, rng, raise_possible=(steps[k - 1][1] == "bet"))
        return r

    def _source_b(self, rng) -> Spot | None:
        sc = self._scenario(rng)
        pot, stack, line = self._money_path(rng, sc)
        board = rng.choice(52, size=5, replace=False).astype(np.uint8)[: self._nb]
        if self._nb == 5:
            strength = _native.solver_river_strengths(board.tolist()).astype(np.int64)
        else:  # turn: order by equity against a uniform range (exact over rivers)
            uni = (~np.isin(HAND_CARDS, board).any(axis=1)).astype(np.float32)
            eq = _native.solver_range_equity(board.tolist(), uni / uni.sum(), 1, 0)
            strength = np.where(uni > 0, 1 + np.round(eq * 1e6), 0).astype(np.int64)
        valid = np.flatnonzero(strength > 0)
        r = [recursive_split(strength, valid, rng) for _ in range(2)]
        return self._finish(board, pot, stack, r[0], r[1],
                            {"source": "B", "scenario": sc.line, "pot_type": sc.pot_type,
                             "line": line})

    def _source_c(self, rng) -> Spot | None:
        sc = self._scenario(rng)
        pot, stack, line = self._money_path(rng, sc)
        board = rng.choice(52, size=5, replace=False).astype(np.uint8)[: self._nb]
        alpha = rng.uniform(0.5, 2.0)
        r = [expand(w, board) * rng.gamma(alpha, size=N_COMBOS) for w in (sc.r_oop, sc.r_ip)]
        return self._finish(board, pot, stack, r[0], r[1],
                            {"source": "C", "scenario": sc.line, "pot_type": sc.pot_type,
                             "line": line})


def recursive_split(strength: np.ndarray, valid: np.ndarray, rng) -> np.ndarray:
    """DeepStack-style range: mass split recursively between the lower and upper halves of
    the strength-sorted tie groups at uniform random fractions, down to single groups (mass
    spread evenly over a group's combos); mixed 50/50 with uniform half the time."""
    order = valid[np.argsort(strength[valid], kind="stable")]
    s = strength[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    bounds = np.r_[starts, len(order)]
    n_groups = len(starts)
    r = np.zeros(N_COMBOS)
    stack = [(0, n_groups, 1.0)]
    while stack:
        lo, hi, m = stack.pop()
        if hi - lo == 1:
            members = order[bounds[lo]:bounds[lo + 1]]
            r[members] = m / len(members)
            continue
        mid = (lo + hi) // 2
        u = rng.random()
        stack.append((lo, mid, m * u))
        stack.append((mid, hi, m * (1 - u)))
    if rng.random() < 0.5:
        uni = np.zeros(N_COMBOS)
        uni[valid] = 1.0 / len(valid)
        r = 0.5 * r + 0.5 * uni
    return r
