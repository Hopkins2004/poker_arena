"""A random bot: fold / call / raise with configurable weights and pot-fraction sizes."""

import random

from pokerarena import Action, BaseBot


class RandomBot(BaseBot):
    def __init__(self, seed: int = 0, fold: float = 1.0, call: float = 2.0, raise_: float = 1.0,
                 sizes=(0.5, 1.0)):
        super().__init__(seed)
        self.rng = random.Random(seed)
        self.w = (fold, call, raise_)
        self.sizes = list(sizes)

    def act(self, obs):
        la = obs.legal
        fold, call, raise_ = self.w
        if not la.can_raise:
            raise_ = 0.0
        u = self.rng.random() * (fold + call + raise_)
        if u < fold + call:
            if la.can_check:
                return Action.check()
            return Action.fold() if u < fold else Action.call()
        frac = self.rng.choice(self.sizes)
        to = obs.current_bet + round((obs.pot + la.call_amount) * frac)
        return Action.raise_to(min(max(to, la.min_raise_to), la.max_raise_to))
