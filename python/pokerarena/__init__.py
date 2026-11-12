"""pokerarena: run Python poker bots against each other on a Rust engine.

Quick start::

    import pokerarena
    result = pokerarena.run("configs/6max_nlh.toml")
    print(result.table)

All bots are Python bots. They implement :class:`pokerarena.Bot` (only ``act`` is required)
and are referenced from a config as ``kind = "python:module.ClassName"``; they are
constructed as ``ClassName(seed=<int>, **params)``. Bots live in the repository's
``bots/<FORMAT>/`` directories (the anchored baselines are in ``bots/NLH``).
See BOTS.md for the full bot-author guide.
"""

from ._native import (
    Action,
    Cancelled,
    LegalActions,
    Observation,
    Run,
    STREETS,
    card_to_str,
    equity,
    evaluate,
    hand_category,
    parse_cards,
)
from .bot import BaseBot, Bot
from .catalog import BotInfo, discover, ensure_importable
from .helpers import clamp_raise, pot_fraction_to
from .job import ArenaJob, Progress, RunResult, is_free_threaded
from .runner import run

ensure_importable()

__all__ = [
    "Action",
    "BotInfo",
    "clamp_raise",
    "discover",
    "pot_fraction_to",
    "ArenaJob",
    "Cancelled",
    "Progress",
    "BaseBot",
    "Bot",
    "LegalActions",
    "Observation",
    "Run",
    "RunResult",
    "STREETS",
    "card_to_str",
    "equity",
    "evaluate",
    "hand_category",
    "is_free_threaded",
    "parse_cards",
    "run",
]
