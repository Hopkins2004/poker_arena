"""Tight-aggressive rule bot. Anchored at Elo 1300.

* Preflop (Hold'em): Chen-formula hand score against position-dependent thresholds; opens,
  3-bets premiums, calls strong hands, continues only with the top of the range vs 3-bets.
* Postflop (and preflop in Omaha): Monte Carlo equity against the live opponents; bets or
  raises strong equity, calls when equity beats the pot odds plus a margin.
"""

import math
import random

from pokerarena import Action, BaseBot, clamp_raise, equity, pot_fraction_to

PREFLOP = 0


def chen_score(a: int, b: int) -> float:
    """Chen formula score of two Hold'em hole cards (ints 0..52; AA = 20, 72o = -1)."""
    hi, lo = (a, b) if a >> 2 >= b >> 2 else (b, a)
    rh, rl = hi >> 2, lo >> 2
    s = {12: 10.0, 11: 8.0, 10: 7.0, 9: 6.0}.get(rh, (rh + 2.0) / 2.0)
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
    """Chen-formula preflop, Monte Carlo equity postflop."""

    NAME = "tag"
    ANCHOR = 1300

    SAMPLES = 300        # Monte Carlo samples per postflop decision (half preflop in PLO)
    TIGHTNESS = 0.0      # added to every preflop threshold (positive = tighter)
    CALL_MARGIN = 0.05   # equity edge over pot odds required to call

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.rng = random.Random(seed)

    def act(self, obs):
        if obs.street == PREFLOP and obs.variant != "plo":
            return self._preflop_holdem(obs)
        return self._by_equity(obs)

    def _preflop_holdem(self, obs):
        la = obs.legal
        chen = chen_score(int(obs.hole[0]), int(obs.hole[1])) - self.TIGHTNESS
        history = obs.history
        raises = sum(1 for r in history if r[1] == PREFLOP and r[2] == "raise")
        fold_or_check = Action.check() if la.can_check else Action.fold()
        behind = _players_behind_preflop(obs)
        if raises == 0:
            open_ = 7.0 if behind <= 1 else 8.0 if behind <= 3 else 9.5
            if chen >= open_ and la.can_raise:
                limpers = sum(1 for r in history if r[1] == PREFLOP and r[2] == "call")
                return clamp_raise(obs, obs.bb * (3 + limpers))
            return fold_or_check
        if raises == 1:
            if chen >= 12.0 and la.can_raise:
                return clamp_raise(obs, obs.current_bet * 3)
            if chen >= 9.0:
                my_stack = int(obs.stacks[obs.seat]) + int(obs.street_contrib[obs.seat])
                return Action.call() if la.call_amount * 6 <= my_stack else fold_or_check
            return fold_or_check
        if chen >= 15.0:
            return clamp_raise(obs, la.max_raise_to) if la.can_raise else Action.call()
        if chen >= 12.0:
            return Action.call()
        return fold_or_check

    def _by_equity(self, obs):
        la = obs.legal
        status = obs.status
        opponents = sum(1 for s in range(obs.n_seats) if s != obs.seat and status[s] in (1, 2))
        samples = max(self.SAMPLES // 2, 50) if obs.street == PREFLOP else self.SAMPLES
        e = equity(obs.hole, obs.board, opponents, samples,
                   seed=self.rng.getrandbits(63), variant=obs.variant)
        fair = 1.0 / (opponents + 1.0)
        strong = min(max(1.6 * fair, 0.55), 0.85)
        very_strong = min(strong + 0.15, 0.92)
        if la.can_check:
            if e >= very_strong and la.can_raise:
                return clamp_raise(obs, pot_fraction_to(obs, 0.75))
            if e >= strong and la.can_raise:
                return clamp_raise(obs, pot_fraction_to(obs, 0.5))
            return Action.check()
        odds = la.call_amount / (obs.pot + la.call_amount)
        if e >= very_strong and la.can_raise:
            return clamp_raise(obs, pot_fraction_to(obs, 0.8))
        if e >= odds + self.CALL_MARGIN:
            return Action.call()
        return Action.fold()


def _players_behind_preflop(obs):
    """Players still to act after me in the first preflop orbit (0 for the big blind)."""
    n = obs.n_seats
    bb_seat = next((r[0] for r in obs.history if r[2] == "bb"), obs.button)
    behind, s = 0, obs.seat
    while s != bb_seat:
        s = (s + 1) % n
        if obs.status[s] in (1, 2):
            behind += 1
    return behind
