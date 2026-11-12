"""The Python bot protocol, mirroring the Rust ``arena::bot::Bot`` trait.

Only :meth:`Bot.act` is required. The optional hooks receive plain dicts:

* ``new_match(info)``: ``variant, max_seats, n_seats, seat, sb, bb, ante,
  starting_stack, hands, tournament, duplicate``
* ``new_hand(info)``: ``hand_id, seat, hole (np.uint8[]), button, stacks (np.uint64[]),
  sb, bb, ante``
* ``observe(event)``: ``{"type": "action", seat, street, kind, amount, to, all_in}``,
  ``{"type": "street", street, board}``, ``{"type": "showdown", seat, hole, strength}``,
  ``{"type": "pot_awarded", seat, amount}``, ``{"type": "uncalled_returned", seat, amount}``
* ``end_hand(result)``: ``hand_id, net (np.int64[]), stacks, board, showdown``

``act`` receives an :class:`pokerarena.Observation` (own hole cards and public information
only) and returns an :class:`pokerarena.Action` (or a kind string such as ``"call"``, or a
``("raise_to", amount)`` tuple). Amounts are *total street contributions* ("raise-to").
``obs.legal`` holds ``can_check, call_amount, can_raise, min_raise_to, max_raise_to``.

Bots are constructed as ``ClassName(seed=<int>, **params)`` once per table seat; seed any
randomness from ``seed`` so that runs are reproducible. In duplicate mode bots must not
carry hand-specific state across hands, and must read their seat from ``new_hand`` /
``obs.seat``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ._native import Action, Observation


@runtime_checkable
class Bot(Protocol):
    """Structural type of a Python bot. Only ``act`` is required."""

    def act(self, obs: Observation) -> Action: ...


class BaseBot:
    """Convenience base class with no-op optional hooks."""

    def __init__(self, seed: int = 0, **params: Any) -> None:
        self.seed = seed
        self.params = params

    def new_match(self, info: dict) -> None:
        pass

    def new_hand(self, info: dict) -> None:
        pass

    def act(self, obs: Observation) -> Action:  # pragma: no cover - abstract
        raise NotImplementedError

    def observe(self, event: dict) -> None:
        pass

    def end_hand(self, result: dict) -> None:
        pass
