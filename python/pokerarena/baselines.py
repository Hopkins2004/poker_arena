"""Python ports of the Rust baseline bots (crates/bots), for comparing Rust and Python bots.

The decision logic mirrors the Rust code line for line and takes the same params, so a
config can switch ``kind = "tag"`` to ``kind = "python:pokerarena.baselines.TightAggressive"``
without other changes (``examples.baselines`` re-exports this module). Random streams differ (Python's ``random`` vs Xoshiro256++), so
results agree statistically, not bit-for-bit. ``TightAggressive`` uses the Rust-backed
``pokerarena.equity`` for its Monte Carlo equity, just as the Rust bot does.
"""

import math
import random

from pokerarena import Action, BaseBot, equity

PREFLOP = 0


def _clamp_raise(la, to):
    return Action.raise_to(min(max(to, la.min_raise_to), la.max_raise_to))


def _pot_fraction_to(obs, frac):
    # Rust's f64::round rounds half away from zero (Python's round() is banker's rounding).
    return obs.current_bet + math.floor((obs.pot + obs.legal.call_amount) * frac + 0.5)


class CallingStation(BaseBot):
    """Always checks or calls."""

    def act(self, obs):
        return Action.check() if obs.legal.can_check else Action.call()


class RandomLegal(BaseBot):
    """Random fold / call / raise with weights and pot-fraction raise sizes.

    Same params as the Rust ``random`` kind: ``fold``, ``call``, ``raise``, ``raise_sizes``.
    """

    def __init__(self, seed=0, fold=1.0, call=2.0, raise_sizes=(0.5, 1.0), **kw):
        super().__init__(seed)
        self.rng = random.Random(seed)
        self.fold_w = float(fold)
        self.call_w = float(call)
        self.raise_w = float(kw.pop("raise", 1.0))  # `raise` is a Python keyword
        if kw:
            raise TypeError(f"unknown params: {sorted(kw)}")
        self.sizes = list(raise_sizes)

    def act(self, obs):
        la = obs.legal
        raise_w = self.raise_w if la.can_raise else 0.0
        total = self.fold_w + self.call_w + raise_w
        u = self.rng.random() * total
        if u < self.fold_w + self.call_w or total <= 0.0:
            if la.can_check:
                return Action.check()
            return Action.fold() if u < self.fold_w else Action.call()
        frac = self.rng.choice(self.sizes) if self.sizes else 1.0
        return _clamp_raise(la, _pot_fraction_to(obs, frac))


def chen_score(a, b):
    """Chen formula score of two Hold'em hole cards (ints 0..52)."""
    ra, rb = a >> 2, b >> 2
    if ra >= rb:
        hi, lo = a, b
    else:
        hi, lo = b, a
    rh, rl = hi >> 2, lo >> 2
    pts = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}
    s = pts.get(rh, (rh + 2.0) / 2.0)
    if rh == rl:
        return max(s * 2.0, 5.0)
    if (hi & 3) == (lo & 3):
        s += 2.0
    gap = rh - rl - 1
    s -= {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0}.get(gap, 5.0)
    if gap <= 1 and rh < 10:
        s += 1.0
    return float(math.ceil(s))


class TightAggressive(BaseBot):
    """Chen-formula preflop, Monte Carlo equity postflop (and preflop in PLO).

    Same params as the Rust ``tag`` kind: ``samples``, ``tightness``, ``call_margin``.
    """

    def __init__(self, seed=0, samples=300, tightness=0.0, call_margin=0.05):
        super().__init__(seed)
        self.rng = random.Random(seed)
        self.samples = int(samples)
        self.tightness = float(tightness)
        self.call_margin = float(call_margin)

    def act(self, obs):
        if obs.street == PREFLOP and obs.variant != "plo":
            return self._preflop_holdem(obs)
        return self._by_equity(obs)

    def _preflop_holdem(self, obs):
        la = obs.legal
        hole = obs.hole
        chen = chen_score(int(hole[0]), int(hole[1])) - self.tightness
        history = obs.history
        raises = sum(1 for r in history if r[1] == PREFLOP and r[2] == "raise")
        fold_or_check = Action.check() if la.can_check else Action.fold()
        behind = _players_behind_preflop(obs)
        if raises == 0:
            open_ = 7.0 if behind <= 1 else 8.0 if behind <= 3 else 9.5
            if chen >= open_ and la.can_raise:
                limpers = sum(1 for r in history if r[1] == PREFLOP and r[2] == "call")
                return _clamp_raise(la, obs.bb * (3 + limpers))
            return fold_or_check
        if raises == 1:
            if chen >= 12.0 and la.can_raise:
                return _clamp_raise(la, obs.current_bet * 3)
            if chen >= 9.0:
                my_stack = int(obs.stacks[obs.seat]) + int(obs.street_contrib[obs.seat])
                return Action.call() if la.call_amount * 6 <= my_stack else fold_or_check
            return fold_or_check
        if chen >= 15.0:
            return _clamp_raise(la, la.max_raise_to) if la.can_raise else Action.call()
        if chen >= 12.0:
            return Action.call()
        return fold_or_check

    def _by_equity(self, obs):
        la = obs.legal
        status = obs.status
        opponents = sum(1 for s in range(obs.n_seats) if s != obs.seat and status[s] in (1, 2))
        samples = max(self.samples // 2, 50) if obs.street == PREFLOP else self.samples
        e = equity(obs.hole, obs.board, opponents, samples,
                   seed=self.rng.getrandbits(63), variant=obs.variant)
        fair = 1.0 / (opponents + 1.0)
        strong = min(max(1.6 * fair, 0.55), 0.85)
        very_strong = min(strong + 0.15, 0.92)
        if la.can_check:
            if e >= very_strong and la.can_raise:
                return _clamp_raise(la, _pot_fraction_to(obs, 0.75))
            if e >= strong and la.can_raise:
                return _clamp_raise(la, _pot_fraction_to(obs, 0.5))
            return Action.check()
        odds = la.call_amount / (obs.pot + la.call_amount)
        if e >= very_strong and la.can_raise:
            return _clamp_raise(la, _pot_fraction_to(obs, 0.8))
        if e >= odds + self.call_margin:
            return Action.call()
        return Action.fold()


def _players_behind_preflop(obs):
    n = obs.n_seats
    bb_seat = next((r[0] for r in obs.history if r[2] == "bb"), obs.button)
    status = obs.status
    behind = 0
    s = obs.seat
    while s != bb_seat:
        s = (s + 1) % n
        if status[s] in (1, 2):
            behind += 1
    return behind
