"""Run a configured tournament with Rust and Python bots.

Rust-only tables run inside the Rust engine on a thread pool with the GIL released.
Tables that seat a Python bot are partitioned over a ``multiprocessing`` pool; each worker
calls into Rust for its tables and returns serialized results, which the parent merges in
table-id order before applying the round's rating update. On a free-threaded CPython
(3.13t+) ``threaded=True`` runs Python tables on threads instead.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import sysconfig
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from ._native import Run


def is_free_threaded() -> bool:
    """True on a free-threaded CPython build with the GIL actually disabled."""
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        return False
    check = getattr(sys, "_is_gil_enabled", None)
    return check is not None and not check()


@dataclass
class RunResult:
    """Outcome of :func:`run`."""

    rows: list = field(default_factory=list)
    """Leaderboard rows (dicts), best first."""
    ratings: dict = field(default_factory=dict)
    """Final rating per bot key."""
    table: str = ""
    """The leaderboard rendered as text."""
    results_dir: Optional[str] = None
    """``results/<run_id>`` directory, if results were saved."""


def _worker(args):
    text, seed, rounds, round_idx, table_ids = args
    r = Run(text, seed=seed, rounds=rounds, load_leaderboard=False)
    return table_ids, r.run_tables(round_idx, table_ids, threads=1)


def _chunks(ids, n):
    n = max(1, min(n, len(ids)))
    return [ids[i::n] for i in range(n)]


def run(
    config: Union[str, os.PathLike],
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
    """Run the tournament described by ``config`` (a TOML path or TOML text).

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
    text = _read_config(config)
    lb = os.fspath(leaderboard) if leaderboard is not None else None
    r = Run(text, seed=seed, rounds=rounds, threads=threads, leaderboard=lb, fresh=fresh,
            load_leaderboard=save)
    if threaded and not is_free_threaded():
        warnings.warn("threaded=True needs a free-threaded CPython (3.13t+) with the GIL "
                      "disabled; using processes instead", RuntimeWarning, stacklevel=2)
        threaded = False
    n_proc = processes or os.cpu_count() or 1
    pool = None
    try:
        for rnd in range(r.n_rounds):
            sched = r.schedule(rnd)
            py_ids = r.python_tables(rnd)
            py_set = set(py_ids)
            rust_ids = [t for t in range(len(sched)) if t not in py_set]
            results: dict[int, str] = {}
            pending = None
            if py_ids and not threaded:
                if pool is None:
                    pool = mp.get_context("spawn").Pool(n_proc)
                jobs = [(text, seed, rounds, rnd, chunk) for chunk in _chunks(py_ids, n_proc)]
                pending = pool.map_async(_worker, jobs)
            if rust_ids:
                results.update(zip(rust_ids, r.run_tables(rnd, rust_ids)))
            if py_ids and threaded:
                results.update(zip(py_ids, r.run_tables(rnd, py_ids, threads=threads)))
            if pending is not None:
                for ids, res in pending.get():
                    results.update(zip(ids, res))
            r.apply_round(rnd, [results[t] for t in range(len(sched))])
            if not quiet:
                print(f"round {rnd + 1}/{r.n_rounds}: {len(sched)} tables "
                      f"({len(py_ids)} with Python bots)", file=sys.stderr)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    out = RunResult(rows=r.rows(), ratings=dict(zip(r.keys, r.ratings)), table=r.render())
    if save:
        rd = os.fspath(results_dir) if results_dir is not None else None
        out.results_dir = r.finish(rd, True)
    return out


def _read_config(config) -> str:
    if isinstance(config, (str, os.PathLike)):
        s = os.fspath(config)
        if "\n" not in s and Path(s).exists():
            return Path(s).read_text()
        if "\n" in s:
            return s
    raise FileNotFoundError(f"config not found: {config!r}")
