"""Bots used by the test-suite (importable from worker processes)."""

import random

from pokerarena import Action, BaseBot


class SpyBot(BaseBot):
    """Checks information hiding from the Python side. On any violation it returns an
    illegal action, which shows up as a non-zero `illegal` count in the results."""

    def __init__(self, seed=0):
        super().__init__(seed)
        self.rng = random.Random(seed)
        self.hole = None
        self.violation = False
        self.revealed = set()
        self.folded = set()

    def new_hand(self, info):
        self.hole = list(info["hole"])
        self.seat = info["seat"]
        self.folded = set()
        self.board_seen = []

    def observe(self, ev):
        if ev["type"] == "action" and ev["kind"] == "fold":
            self.folded.add(ev["seat"])
        if ev["type"] == "street":
            self.board_seen = list(ev["board"])
        if ev["type"] == "showdown":
            if ev["seat"] in self.folded:
                self.violation = True

    def act(self, obs):
        if list(obs.hole) != self.hole or obs.seat != self.seat:
            self.violation = True
        if list(obs.board) != self.board_seen[: len(obs.board)]:
            self.violation = True
        if set(obs.board) & set(self.hole):
            self.violation = True
        public = {name for name in dir(obs) if not name.startswith("_")}
        expected = {"seat", "n_seats", "button", "street", "variant", "max_seats", "hole",
                    "board", "stacks", "street_contrib", "total_contrib", "status", "pot",
                    "current_bet", "sb", "bb", "ante", "legal", "history"}
        if public != expected:
            self.violation = True
        if self.violation:
            return Action.bet_to(0)  # always illegal
        la = obs.legal
        if la.can_raise and self.rng.random() < 0.2:
            return ("raise_to", la.min_raise_to)  # tuple form
        return "call"  # string form


class Crasher(BaseBot):
    """Raises in act() every few decisions."""

    def __init__(self, seed=0):
        super().__init__(seed)
        self.n = 0

    def act(self, obs):
        self.n += 1
        if self.n % 5 == 0:
            raise RuntimeError("boom")
        return Action.check() if obs.legal.can_check else Action.call()


class Sleeper(BaseBot):
    """Hangs forever in act() (simulates a stuck Python bot)."""

    def act(self, obs):
        import time

        time.sleep(3600)
        return Action.fold()


class Exiter(BaseBot):
    """Kills its own worker process mid-hand (simulates a crashing extension/bot)."""

    def act(self, obs):
        import os

        os._exit(3)
