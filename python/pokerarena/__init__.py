"""pokerarena: run poker bots (Rust and Python) against each other.

Quick start::

    import pokerarena
    result = pokerarena.run("configs/6max_nlh.toml")
    print(result.table)

Python bots implement :class:`pokerarena.Bot` (only ``act`` is required) and are referenced
from a config as ``kind = "python:module.ClassName"``. They are constructed as
``ClassName(seed=<int>, **params)``.
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
    rust_bot_kinds,
)
from .bot import BaseBot, Bot
from .job import ArenaJob, Progress, RunResult, is_free_threaded
from .runner import run

__all__ = [
    "Action",
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
    "rust_bot_kinds",
]
