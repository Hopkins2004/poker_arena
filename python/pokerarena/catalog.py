"""Bot catalog: the bots that live in the repository's ``bots/<FORMAT>/`` directories.

Every ``.py`` file in ``bots/NLH`` (except names starting with ``_``) is scanned for bot
classes defined in that file (classes with an ``act`` method). Each class may declare:

* ``NAME``: the default player name (otherwise the file name);
* ``ANCHOR``: a fixed Elo rating (the bot is an anchor; see the README).

Bots are referenced as ``python:bots.<FORMAT>.<module>.<Class>``. Importing :mod:`pokerarena`
puts the repository root on ``sys.path`` so that ``bots`` is importable (worker processes
inherit it).
"""

from __future__ import annotations

import importlib
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
BOTS_DIR = REPO_ROOT / "bots"
DEFAULT_FORMAT_DIR = "NLH"


def ensure_importable() -> None:
    """Makes ``bots.<FORMAT>...`` importable (adds the repository root to ``sys.path``)."""
    root = str(REPO_ROOT)
    if BOTS_DIR.is_dir() and root not in sys.path:
        sys.path.append(root)


@dataclass(frozen=True)
class BotInfo:
    """One bot found in a format directory."""

    name: str
    """Default player name (``NAME`` or the file name)."""
    kind: str
    """Config kind, e.g. ``python:bots.NLH.tag.TightAggressive``."""
    label: str
    """Human-readable label (first docstring line)."""
    anchor: Optional[float]
    """Fixed Elo, or None."""
    path: str
    """Source file."""


def discover(format_dir: str = DEFAULT_FORMAT_DIR) -> list[BotInfo]:
    """All bots in ``bots/<format_dir>``, sorted by name. Files that fail to import are
    skipped with a warning on stderr."""
    ensure_importable()
    folder = BOTS_DIR / format_dir
    out: list[BotInfo] = []
    if not folder.is_dir():
        return out
    for path in sorted(folder.glob("*.py")):
        if path.name.startswith("_"):
            continue
        modname = f"bots.{format_dir}.{path.stem}"
        try:
            module = importlib.import_module(modname)
        except Exception as e:  # noqa: BLE001 - a broken bot file must not break the catalog
            print(f"pokerarena: skipping {path}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        for cls_name, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != modname or not callable(getattr(cls, "act", None)):
                continue
            doc = " ".join((inspect.getdoc(cls) or cls_name).split("\n\n")[0].split())
            anchor = getattr(cls, "ANCHOR", None)
            out.append(BotInfo(
                name=str(getattr(cls, "NAME", path.stem)),
                kind=f"python:{modname}.{cls_name}",
                label=doc,
                anchor=None if anchor is None else float(anchor),
                path=str(path),
            ))
    return sorted(out, key=lambda b: b.name)


def anchored_lineup(format_dir: str = DEFAULT_FORMAT_DIR) -> list[dict]:
    """Config ``[[bots]]`` entries for every anchored bot in a format directory."""
    return [{"name": b.name, "kind": b.kind, "anchor": b.anchor}
            for b in discover(format_dir) if b.anchor is not None]
