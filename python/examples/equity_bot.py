"""An equity-threshold bot using the Rust Monte Carlo ``pokerarena.equity`` helper.

Bets when its equity against the live opponents is well above a fair share, calls when it
beats the pot odds, and otherwise checks or folds.
"""

import random

from pokerarena import Action, BaseBot, equity


class EquityBot(BaseBot):
    def __init__(self, seed: int = 0, samples: int = 200, bet_edge: float = 1.5,
                 call_margin: float = 0.03):
        super().__init__(seed)
        self.rng = random.Random(seed)
        self.samples = samples
        self.bet_edge = bet_edge
        self.call_margin = call_margin

    def act(self, obs):
        la = obs.legal
        opponents = int(sum(1 for s, st in enumerate(obs.status) if s != obs.seat and st in (1, 2)))
        e = equity(obs.hole, obs.board, opponents, self.samples,
                   seed=self.rng.getrandbits(63), variant=obs.variant)
        fair = 1.0 / (opponents + 1)
        if la.can_raise and e >= min(0.9, fair * self.bet_edge):
            to = obs.current_bet + round(0.66 * (obs.pot + la.call_amount))
            return Action.raise_to(min(max(to, la.min_raise_to), la.max_raise_to))
        if la.can_check:
            return Action.check()
        odds = la.call_amount / (obs.pot + la.call_amount)
        return Action.call() if e >= odds + self.call_margin else Action.fold()
