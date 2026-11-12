import json
import warnings

import pytest

import pokerarena as pa

MIXED = """
[run]
rounds = 2
seed = 17

[format]
variant = "nlhe"
max_seats = 6

[table]
hands_per_table = 60
starting_stack_bb = 100
blinds = { sb = 1, bb = 2 }

[[bots]]
name = "tag"
kind = "tag"
params = { samples = 50 }

[[bots]]
name = "rust_random"
kind = "random"

[[bots]]
name = "station"
kind = "calling_station"

[[bots]]
name = "py_random"
kind = "python:examples.random_bot.RandomBot"
params = { fold = 1.0, call = 2.0, raise_ = 1.0, sizes = [0.5, 1.0] }

[[bots]]
name = "py_equity"
kind = "python:examples.equity_bot.EquityBot"
params = { samples = 60 }

[[bots]]
name = "spy"
kind = "python:tests.bots_for_tests.SpyBot"
"""


def _run(**kw):
    return pa.run(MIXED, save=False, quiet=True, **kw)


def _canon(res):
    return json.dumps(res.rows, sort_keys=True)


def test_mixed_run_deterministic_across_processes():
    a = _run(processes=1)
    b = _run(processes=4)
    c = _run(processes=4, threads=1)
    assert _canon(a) == _canon(b) == _canon(c)
    assert a.ratings == b.ratings
    keys = {r["key"] for r in a.rows}
    assert keys == {"tag", "rust_random", "station", "py_random", "py_equity", "spy"}
    for r in a.rows:
        assert r["illegal"] == 0, r
        assert r["hands"] > 0
    assert abs(sum(a.ratings.values()) - 6 * 1500) < 1e-6


def test_different_seed_differs():
    assert _canon(_run(processes=2)) != _canon(_run(processes=2, seed=18))


def test_python_and_rust_tables_split():
    r = pa.Run(MIXED, load_leaderboard=False)
    sched = r.schedule(0)
    py = set(r.python_tables(0))
    assert py, "some tables must contain Python bots"
    for t, seats in enumerate(sched):
        has_py = any(r.python_bots[b] for b in seats)
        assert has_py == (t in py)


def test_threaded_falls_back_on_gil_build():
    if pa.is_free_threaded():
        res = _run(threaded=True)
        assert _canon(res) == _canon(_run(processes=2))
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            res = _run(threaded=True, processes=2)
        assert any("free-threaded" in str(x.message) for x in w)
        assert _canon(res) == _canon(_run(processes=1))


def test_rust_only_run_threads_identical():
    cfg = MIXED.split("[[bots]]\nname = \"py_random\"")[0]
    a = pa.run(cfg, save=False, quiet=True, threads=1)
    b = pa.run(cfg, save=False, quiet=True, threads=8)
    assert _canon(a) == _canon(b)


def test_exceptions_count_as_illegal_actions():
    cfg = MIXED.replace("tests.bots_for_tests.SpyBot", "tests.bots_for_tests.Crasher")
    res = pa.run(cfg, save=False, quiet=True, processes=2, rounds=1)
    crasher = next(r for r in res.rows if r["key"] == "spy")
    assert crasher["illegal"] > 0
    others = [r for r in res.rows if r["key"] != "spy"]
    assert all(r["illegal"] == 0 for r in others)


def test_bad_python_kind_fails_fast():
    cfg = MIXED.replace("tests.bots_for_tests.SpyBot", "tests.bots_for_tests.Missing")
    with pytest.raises(AttributeError):
        pa.Run(cfg, load_leaderboard=False)


def test_save_results_and_leaderboard(tmp_path):
    lb = tmp_path / "lb.json"
    res = pa.run(MIXED, quiet=True, processes=2, rounds=1, leaderboard=lb,
                 results_dir=tmp_path / "results")
    assert res.results_dir and (tmp_path / "results").exists()
    stored = json.loads(lb.read_text())["entries"]
    assert set(stored) == set(res.ratings)
    # Second run starts from the stored ratings.
    r = pa.Run(MIXED, leaderboard=str(lb))
    assert r.ratings == [res.ratings[k] for k in r.keys]
