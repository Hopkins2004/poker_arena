"""The example bot from BOTS.md (tested in tests/test_guide_example.py).

Calls when its Monte Carlo equity beats the pot odds, value-raises strong hands, and plays
tighter when the hand has already been raised twice.

Config entry:
    [[bots]]
    name = "pot_odds"
    kind = "python:examples.pot_odds_bot.PotOdds"
    params = { samples = 200 }
"""

import random

from pokerarena import Action, BaseBot, equity


class PotOdds(BaseBot):
    def __init__(self, seed: int = 0, samples: int = 200):
        super().__init__(seed)
        self.rng = random.Random(seed)  # all randomness from the seed => reproducible runs
        self.samples = samples
        self.raises_this_hand = 0

    def new_hand(self, info):
        self.raises_this_hand = 0  # never carry hand state across hands

    def observe(self, event):
        if event["type"] == "action" and event["kind"] == "raise":
            self.raises_this_hand += 1

    def act(self, obs):
        la = obs.legal
        # Opponents still in the hand: status 1 (active) or 2 (all-in), excluding me.
        opponents = sum(1 for s, st in enumerate(obs.status) if s != obs.seat and st in (1, 2))
        e = equity(obs.hole, obs.board, opponents, self.samples,
                   seed=self.rng.getrandbits(63), variant=obs.variant)
        threshold = 0.85 if self.raises_this_hand >= 2 else 0.70
        if la.can_raise and e >= threshold:
            # Raise-to = current bet + 2/3 of the pot after calling, clamped to the legal range.
            to = obs.current_bet + (obs.pot + la.call_amount) * 2 // 3
            return Action.raise_to(min(max(to, la.min_raise_to), la.max_raise_to))
        if la.can_check:
            return Action.check()
        pot_odds = la.call_amount / (obs.pot + la.call_amount)
        return Action.call() if e >= pot_odds else Action.fold()
