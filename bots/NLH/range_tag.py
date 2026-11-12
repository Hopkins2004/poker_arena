"""Chart preflop, TAG postflop (proof of concept for the preflop range database).

* Preflop: plays the 6-max 100bb chart (``pokerarena.preflop``, from the Leakproof ranges, CC BY
  4.0). When the hand leaves the chart (limpers, cold callers, short-handed tables, odd sizes),
  it answers the last aggressive action as if the chart's line had happened: unopened pots
  with limpers are played as an open (raise first in), one raise as facing an open, two raises
  as facing a 3-bet, and so on, with cold callers ignored. Raise sizes keep the chart's
  multiple of the bet being faced.
* Postflop: the anchored TAG bot's Monte Carlo equity logic, unchanged.
"""

import random

from pokerarena import Action, clamp_raise, preflop

from bots.NLH.tag import TightAggressive

PREFLOP = 0
ORDER = ("UTG", "HJ", "CO", "BTN", "SB", "BB")
DB = preflop.load()


def positions_by_seat(obs) -> dict:
    """Seat -> chart position for any table size: BTN, SB, BB, then CO, HJ, UTG counting
    back from the button (heads-up: the button is the SB). Empty seats are skipped."""
    n = obs.n_seats
    occupied = [(obs.button + k) % n for k in range(n)]
    occupied = [s for s in occupied if int(obs.status[s]) != 0]  # button first
    m = len(occupied)
    if m == 2:
        return {occupied[0]: "SB", occupied[1]: "BB"}
    out = {occupied[0]: "BTN", occupied[1]: "SB", occupied[2]: "BB"}
    back = ("CO", "HJ", "UTG")
    for i, s in enumerate(reversed(occupied[3:])):
        out[s] = back[min(i, len(back) - 1)]
    return out


def _aggressor(node) -> str | None:
    """Position that made the last bet/raise in a node's line (None when unopened)."""
    return node.line.split(", ")[-1].split()[0] if node.line else None


def _facing_bb(node) -> float:
    """Raise-to total (bb) the actor faces at a node: the last chart raise, else the BB."""
    for part in reversed(node.line.split(", ") if node.line else []):
        word = part.split()
        if word[1] == "raise":
            return float(word[2][:-2])
    return 1.0


class RangeTag(TightAggressive):
    """Preflop chart (6-max 100bb), TAG equity play after the flop."""

    NAME = "range_tag"
    ANCHOR = None  # not anchored (TightAggressive's ANCHOR would otherwise be inherited)

    def __init__(self, seed: int = 0):
        super().__init__(seed)
        self.prng = random.Random(seed ^ 0x9E3779B97F4A7C15)

    def act(self, obs):
        if obs.street == PREFLOP and obs.variant == "nlhe":
            return self._preflop_chart(obs)
        return super().act(obs)

    # -- preflop --------------------------------------------------------------------------

    def _preflop_chart(self, obs):
        d = DB.locate(obs)
        if d is not None:
            action = d.sample(obs, self.prng)
            if action is not None:
                return action
        return self._last_aggression(obs)

    def _last_aggression(self, obs):
        """Off the chart: answer the last aggressive action with the closest chart node."""
        la = obs.legal
        pos = positions_by_seat(obs)
        raises, last_raiser, limpers = 0, None, 0
        for seat, street, kind, _amount, _to, _all_in in obs.history:
            if street != PREFLOP:
                continue
            if kind in ("bet", "raise"):
                raises += 1
                last_raiser = pos.get(seat)
                limpers = 0
            elif kind == "call" and raises == 0:
                limpers += 1
        node = self._closest_node(min(raises, 4), pos[obs.seat], last_raiser)
        if node is None:
            return Action.check() if la.can_check else Action.fold()
        freqs = node.freqs((int(obs.hole[0]), int(obs.hole[1])))
        if freqs is None:  # the chart never brings this hand here: give up the pot
            return Action.check() if la.can_check else Action.fold()
        acts, weights = node.actions, list(freqs.values())
        a = self.prng.choices(range(len(acts)), weights=weights)[0]
        kind = acts[a].kind
        if kind == "fold":
            return Action.check() if la.can_check else Action.fold()
        if kind == "call" or not la.can_raise:
            return Action.check() if la.can_check else Action.call()
        if kind == "allin":
            return Action.raise_to(la.max_raise_to)
        if raises == 0:  # open, isolating limpers: chart size plus one bb per limper
            return clamp_raise(obs, round((acts[a].to_bb + limpers) * obs.bb))
        mult = acts[a].to_bb / _facing_bb(node)
        return clamp_raise(obs, round(obs.current_bet * mult))

    @staticmethod
    def _closest_node(raises: int, me: str, aggressor):
        """Chart node at this raise level, preferring my position, then the aggressor's."""
        best, best_key = None, None
        for n in DB.nodes:
            if n.raises != raises:
                continue
            key = (abs(ORDER.index(n.actor) - ORDER.index(me)),
                   0 if aggressor is None or _aggressor(n) is None
                   else abs(ORDER.index(_aggressor(n)) - ORDER.index(aggressor)))
            if best_key is None or key < best_key:
                best, best_key = n, key
        return best
