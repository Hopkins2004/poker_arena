"""``pokerarena.run``: run a configured tournament with Rust and Python bots (blocking).

Rust-only tables run inside the Rust engine on a thread pool with the GIL released. Tables
that seat a Python bot are partitioned over a ``multiprocessing`` pool; each worker calls into
Rust for its tables and returns serialized results, which the parent merges in table-id order
before applying the round's rating update. On a free-threaded CPython (3.13t+)
``threaded=True`` runs Python tables on threads instead. See :mod:`pokerarena.job` for the
controllable (background, stoppable) version used by the GUI.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Union

from .job import ArenaJob, RunResult, is_free_threaded

__all__ = ["run", "RunResult", "is_free_threaded"]


def run(
    config: Union[str, os.PathLike, dict],
    *,
    processes: Optional[int] = None,
    threaded: bool = False,
    seed: Optional[int] = None,
    rounds: Optional[int] = None,
    threads: Optional[int] = None,
    fresh: bool = False,
    quiet: bool = False,
    save: bool = True,
    leaderboard: Optional[Union[str, os.PathLike]] = None,
    results_dir: Optional[Union[str, os.PathLike]] = None,
) -> RunResult:
    """Run the tournament described by ``config`` (a TOML path, TOML/JSON text or a dict).

    Args:
        processes: worker processes for tables with Python bots (default: CPU count).
        threaded: on free-threaded CPython, run Python tables on threads instead of
            processes (falls back to processes, with a warning, on GIL builds).
        seed, rounds, threads: override the config.
        fresh: ignore stored ratings for these bots.
        save: write ``results/<run_id>/`` and update the leaderboard file.
        leaderboard, results_dir: override the config's paths.

    Results are identical for any ``processes``/``threads`` value.
    """

    def report(rnd, n_rounds, n_tables, n_py):
        print(f"round {rnd + 1}/{n_rounds}: {n_tables} tables ({n_py} with Python bots)",
              file=sys.stderr)

    job = ArenaJob(config, processes=processes, threaded=threaded, seed=seed, rounds=rounds,
                   threads=threads, fresh=fresh, save=save, leaderboard=leaderboard,
                   results_dir=results_dir, on_round=None if quiet else report)
    return job.run()
