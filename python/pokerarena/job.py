"""A controllable run: background execution, live progress and a hard stop.

:class:`ArenaJob` drives one run. It is used by :func:`pokerarena.run` (blocking) and by the
GUI (background thread + polling).

How tables execute:

* Every table (all bots are Python) runs in a ``multiprocessing`` pool of worker processes,
  one table per task. Each worker plays its table through the Rust engine and reports
  finished hands into a slot of a shared ``RawArray``, so the parent can show hand-level
  progress without locks.
* On free-threaded CPython with ``threaded=True``, tables run in-process on a Rust thread
  pool instead, with the GIL released.
* Results are merged in table-id order and ratings are updated once per round. The output is
  identical for any process or thread count.
* If enabled (``[run] replay_hands_per_bot``), a random sample of hands per bot is recorded
  and written to ``hand_replays/<run_id>/`` after a completed run.

**Stopping** (:meth:`ArenaJob.stop`) is a hard stop:

* The worker pool is ``terminate()``-d and joined, which kills worker processes even if a
  Python bot is stuck in an infinite loop.
* The Rust cancellation flag is set, so in-process tables stop before their next hand.
* Nothing is written: results, the leaderboard and hand replays are saved only after the
  last round, and those writes are atomic (temp file or directory, then rename).
* All cleanup happens on the driver thread in a ``finally`` block, so after :meth:`wait`
  returns no worker processes, pool threads or result handles are left behind.
"""

from __future__ import annotations

import gc
import multiprocessing as mp
import os
import signal
import sys
import sysconfig
import threading
import time
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from ._native import Cancelled, Run

__all__ = ["ArenaJob", "Progress", "RunResult", "is_free_threaded"]


def is_free_threaded() -> bool:
    """True on a free-threaded CPython build with the GIL actually disabled."""
    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        return False
    check = getattr(sys, "_is_gil_enabled", None)
    return check is not None and not check()


@dataclass
class RunResult:
    """Outcome of a finished run."""

    rows: list = field(default_factory=list)
    """Leaderboard rows (dicts), best first."""
    ratings: dict = field(default_factory=dict)
    """Final rating per bot key."""
    table: str = ""
    """The leaderboard rendered as text."""
    results_dir: Optional[str] = None
    """``results/<run_id>`` directory, if results were saved."""
    replays_dir: Optional[str] = None
    """``hand_replays/<run_id>`` directory, if hand replays were enabled."""
    hands: int = 0
    """Hands played."""
    tables: int = 0
    """Tables played."""
    elapsed: float = 0.0
    """Wall-clock seconds."""
    tournament: bool = False
    """True for tournament runs."""


@dataclass(frozen=True)
class Progress:
    """A snapshot of a job's progress."""

    state: str
    """``ready``, ``running``, ``stopping``, ``done``, ``stopped`` or ``error``."""
    hands_done: int
    hands_total: int
    """Planned hands (an upper bound for tournaments)."""
    tables_done: int
    tables_total: int
    rounds_done: int
    rounds_total: int
    elapsed: float
    tournament: bool = False
    """Tournaments end early, so ``hands_total`` is only an upper bound there."""

    @property
    def fraction(self) -> float:
        """Completed fraction in [0, 1] (by hands; by tables for tournaments)."""
        if self.state == "done":
            return 1.0
        if self.hands_total and not self.tournament:
            return min(1.0, self.hands_done / self.hands_total)
        return self.tables_done / self.tables_total if self.tables_total else 0.0

    @property
    def hands_per_sec(self) -> float:
        return self.hands_done / self.elapsed if self.elapsed > 0 else 0.0


class _Stopped(Exception):
    pass


# --------------------------------------------------------------------------- worker side

_slots = None
_runs: dict = {}


def _init_worker(slots) -> None:
    global _slots
    _slots = slots
    # The parent owns Ctrl-C handling and terminates the pool; workers just die with it.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _work(args):
    text, seed, rounds, rnd, table_id, slot = args
    key = (text, seed, rounds)
    run = _runs.get(key)
    if run is None:
        _runs.clear()
        run = Run(text, seed=seed, rounds=rounds, load_leaderboard=False)
        _runs[key] = run
    start = run.hands_done

    def report(n, _slot=slot, _start=start):
        _slots[_slot] = n - _start

    res = run.run_tables(rnd, [table_id], threads=1, on_progress=report, every=128)
    _slots[slot] = run.hands_done - start
    return table_id, res[0]


# --------------------------------------------------------------------------- parent side


def _read_config(config) -> str:
    """Accepts a path to a TOML file, TOML text, JSON text, or a dict (as JSON)."""
    if isinstance(config, dict):
        import json

        return json.dumps(config)
    s = os.fspath(config)
    stripped = s.lstrip()
    if stripped.startswith("{") or "\n" in s:
        return s
    if Path(s).exists():
        return Path(s).read_text()
    raise FileNotFoundError(f"config not found: {config!r}")


class ArenaJob:
    """One run, executed with :meth:`run` (blocking) or :meth:`start` (background thread).

    Args:
        config: a TOML path, TOML or JSON text, or a dict with the config structure.
        processes: worker processes for tables with Python bots (default: CPU count).
        threaded: on free-threaded CPython, run Python tables on threads instead.
        seed, rounds, threads: override the config.
        fresh: ignore stored ratings for these bots.
        save: write ``results/<run_id>/`` and update the leaderboard when the run completes.
        leaderboard, results_dir, replays_dir: override the config's paths.
        on_round: called as ``on_round(round_index, n_rounds, n_tables, n_python_tables)``
            on the driver thread after each round.

    Construction validates the config and imports every Python bot, so errors surface
    immediately in the caller's thread.
    """

    def __init__(
        self,
        config,
        *,
        processes: Optional[int] = None,
        threaded: bool = False,
        seed: Optional[int] = None,
        rounds: Optional[int] = None,
        threads: Optional[int] = None,
        fresh: bool = False,
        save: bool = True,
        leaderboard: Optional[Union[str, os.PathLike]] = None,
        results_dir: Optional[Union[str, os.PathLike]] = None,
        replays_dir: Optional[Union[str, os.PathLike]] = None,
        on_round: Optional[Callable[[int, int, int, int], None]] = None,
    ) -> None:
        self.text = _read_config(config)
        self._seed, self._rounds, self._threads = seed, rounds, threads
        lb = os.fspath(leaderboard) if leaderboard is not None else None
        self._run = Run(self.text, seed=seed, rounds=rounds, threads=threads, leaderboard=lb,
                        fresh=fresh, load_leaderboard=save)
        if threaded and not is_free_threaded():
            warnings.warn("threaded=True needs a free-threaded CPython (3.13t+) with the GIL "
                          "disabled; using processes instead", RuntimeWarning, stacklevel=2)
            threaded = False
        self.threaded = threaded
        self.processes = processes or os.cpu_count() or 1
        self.save = save
        self.results_dir = os.fspath(results_dir) if results_dir is not None else None
        self.replays_dir = os.fspath(replays_dir) if replays_dir is not None else None
        self.on_round = on_round

        run = self._run
        self.keys = list(run.keys)
        self.run_id = run.run_id
        self.initial_ratings = dict(zip(run.keys, run.initial_ratings))
        self.anchors = dict(zip(run.keys, run.anchors))
        self._schedules = [run.schedule(r) for r in range(run.n_rounds)]
        self._py_tables = [run.python_tables(r) for r in range(run.n_rounds)]
        self._offsets = []
        total = 0
        for s in self._schedules:
            self._offsets.append(total)
            total += len(s)
        self.tables_total = total
        self.hands_total = total * run.hands_per_table
        self.has_python = any(self._py_tables)
        self.tournament = run.is_tournament

        self._state = "ready"
        self._stop = threading.Event()
        self._finishing = False
        self._thread: Optional[threading.Thread] = None
        self._slots = None
        self._workers: list = []
        self._tables_done = 0
        self._rounds_done = 0
        self._t0: Optional[float] = None
        self._t1: Optional[float] = None
        self.result: Optional[RunResult] = None
        self.error: Optional[BaseException] = None
        self.error_traceback: str = ""

    # -- control ------------------------------------------------------------------------

    def start(self) -> None:
        """Starts the run on a background thread."""
        if self._state != "ready":
            raise RuntimeError(f"job already {self._state}")
        self._state = "running"
        self._thread = threading.Thread(target=self._main, name="arena-job", daemon=False)
        self._thread.start()

    def run(self) -> RunResult:
        """Runs to completion on the calling thread; raises on error or stop."""
        if self._state != "ready":
            raise RuntimeError(f"job already {self._state}")
        self._state = "running"
        self._main()
        if self._state == "stopped":
            raise Cancelled("run stopped")
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result

    def stop(self) -> None:
        """Hard stop. Safe from any thread at any time; returns immediately (use
        :meth:`wait` to wait for cleanup). Ignored once results are being saved."""
        if self._finishing or self._state in ("done", "stopped", "error"):
            return
        self._stop.set()
        if self._state == "running":
            self._state = "stopping"
        self._run.cancel()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Waits for the background thread; True once it has fully finished (cleanup done)."""
        if self._thread is None:
            return self._state != "running"
        self._thread.join(timeout)
        return not self._thread.is_alive()

    @property
    def state(self) -> str:
        return self._state

    def progress(self) -> Progress:
        """Thread-safe snapshot (cheap enough to call many times per second)."""
        try:
            done = self._run.hands_done
        except RuntimeError:  # borrowed mutably for an instant while merging a round
            done = 0
        slots = self._slots
        if slots is not None:
            done += sum(slots)
        if self._t0 is None:
            elapsed = 0.0
        else:
            elapsed = (self._t1 or time.perf_counter()) - self._t0
        return Progress(self._state, done, self.hands_total, self._tables_done,
                        self.tables_total, self._rounds_done, len(self._schedules), elapsed,
                        self.tournament)

    # -- driver -------------------------------------------------------------------------

    def _check(self) -> None:
        if self._stop.is_set():
            raise _Stopped()

    def _main(self) -> None:
        run = self._run
        pool = None
        it = None
        status = "error"
        self._t0 = time.perf_counter()
        try:
            for rnd, sched in enumerate(self._schedules):
                self._check()
                py_ids = self._py_tables[rnd]
                py_set = set(py_ids)
                rust_ids = [t for t in range(len(sched)) if t not in py_set]
                results: dict[int, str] = {}
                it = None
                if py_ids and not self.threaded:
                    if pool is None:
                        pool = self._make_pool()
                    tasks = [(self.text, self._seed, self._rounds, rnd, t, self._offsets[rnd] + t)
                             for t in py_ids]
                    it = pool.imap_unordered(_work, tasks, chunksize=1)
                if rust_ids:
                    results.update(zip(rust_ids, run.run_tables(rnd, rust_ids)))
                    self._tables_done += len(rust_ids)
                if py_ids and self.threaded:
                    results.update(zip(py_ids, run.run_tables(rnd, py_ids, threads=self._threads)))
                    self._tables_done += len(py_ids)
                if it is not None:
                    remaining = len(py_ids)
                    while remaining:
                        self._check()
                        self._check_workers()
                        try:
                            t, res = it.next(timeout=0.05)
                        except mp.TimeoutError:
                            continue
                        results[t] = res
                        remaining -= 1
                        self._tables_done += 1
                self._check()
                run.apply_round(rnd, [results[t] for t in range(len(sched))])
                self._rounds_done += 1
                if self.on_round:
                    self.on_round(rnd, len(self._schedules), len(sched), len(py_ids))
            self._check()
            self._finishing = True
            hands = self.progress().hands_done
            res = RunResult(rows=run.rows(), ratings=dict(zip(run.keys, run.ratings)),
                            table=run.render(), hands=hands, tables=self.tables_total,
                            elapsed=time.perf_counter() - self._t0)
            res.tournament = self.tournament
            if self.save:
                res.results_dir = run.finish(self.results_dir, True)
            # Hand replays are independent of saving results/leaderboard.
            res.replays_dir = run.write_replays(self.replays_dir)
            self.result = res
            status = "done"
        except (Cancelled, _Stopped):
            status = "stopped"
        except BaseException as e:  # noqa: BLE001 - reported to the caller, then re-raised by run()
            # Keep the traceback as text only: live frames would pin the pool, dead worker
            # processes and their pipes (file descriptors) for as long as the job exists.
            self.error_traceback = traceback.format_exc()
            self.error = e.with_traceback(None)
            status = "error"
        finally:
            if pool is not None:
                if status == "done":
                    pool.close()
                else:
                    pool.terminate()
                pool.join()
                # A terminated Pool's pipes and semaphores live in reference cycles between
                # its handler threads, caches and result iterators; without a collection they
                # stay open until some later GC (~16 fds per pool). Release them now.
                it = pool = None
                self._workers = []
                gc.collect()
            self._t1 = time.perf_counter()
            self._state = status

    def _make_pool(self):
        ctx = mp.get_context("spawn")
        self._slots = ctx.RawArray("Q", max(1, self.tables_total))
        n = max(1, min(self.processes, max(len(p) for p in self._py_tables)))
        pool = ctx.Pool(n, initializer=_init_worker, initargs=(self._slots,))
        # Pool silently replaces dead workers, and the task a dead worker held never
        # completes, so a crash would hang the run. Keep the original workers to detect it.
        self._workers = list(getattr(pool, "_pool", []))
        return pool

    def _check_workers(self) -> None:
        for p in self._workers:
            code = p.exitcode
            if code is not None:
                raise RuntimeError(
                    f"a Python worker process died unexpectedly (exit code {code}); its "
                    "table cannot finish. A Python bot probably crashed the interpreter or "
                    "called exit(); see the log / stderr for details.")
