"""Small helpers for writing bots."""

import math

from ._native import Action


def clamp_raise(obs, to: int) -> Action:
    """A raise to ``to`` (street total), clamped into the legal range of ``obs.legal``."""
    la = obs.legal
    return Action.raise_to(min(max(int(to), la.min_raise_to), la.max_raise_to))


def pot_fraction_to(obs, frac: float) -> int:
    """Raise-to total for a bet/raise of ``frac`` times the pot after calling (rounded half
    away from zero)."""
    return obs.current_bet + math.floor((obs.pot + obs.legal.call_amount) * frac + 0.5)
