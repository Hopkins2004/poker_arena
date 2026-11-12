"""Random legal play with fixed weights and pot-fraction raise sizes. Anchored at Elo 850."""

import random

from pokerarena import Action, BaseBot, clamp_raise, pot_fraction_to


class RandomLegal(BaseBot):
    """Folds, calls or raises at random (weights 1 : 2 : 1); raises 0.5 or 1 pot. When it
    can check, the fold weight checks instead (it never folds for free)."""

    NAME = "random"
    ANCHOR = 850

    FOLD_WEIGHT = 1.0
    CALL_WEIGHT = 2.0
    RAISE_WEIGHT = 1.0
    RAISE_SIZES = (0.5, 1.0)  # fractions of the pot after calling

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.rng = random.Random(seed)

    def act(self, obs):
        la = obs.legal
        raise_w = self.RAISE_WEIGHT if la.can_raise else 0.0
        total = self.FOLD_WEIGHT + self.CALL_WEIGHT + raise_w
        u = self.rng.random() * total
        if u < self.FOLD_WEIGHT + self.CALL_WEIGHT:
            if la.can_check:
                return Action.check()
            return Action.fold() if u < self.FOLD_WEIGHT else Action.call()
        return clamp_raise(obs, pot_fraction_to(obs, self.rng.choice(self.RAISE_SIZES)))
