"""Hard-stop ("nuke") and leak tests for ArenaJob."""

import os
import subprocess
import sys
import textwrap
import threading
import time

import pytest

import pokerarena as pa

PY = "python:pokerarena.baselines."

BASE = """
[run]
rounds = 3
seed = 5
[format]
variant = "nlhe"
max_seats = 6
[table]
hands_per_table = {hands}
blinds = {{ sb = 1, bb = 2 }}
"""
BASE_BOTS = """
[[bots]]
name = "tag"
kind = "python:pokerarena.baselines.TightAggressive"
params = { samples = 50 }
anchor = 1300
[[bots]]
name = "random"
kind = "python:pokerarena.baselines.RandomLegal"
anchor = 850
[[bots]]
name = "station"
kind = "python:pokerarena.baselines.CallingStation"
anchor = 400
"""
PY_BOTS = """
[[bots]]
name = "py_tag"
kind = "python:pokerarena.baselines.TightAggressive"
params = { samples = 50 }
[[bots]]
name = "py_station"
kind = "python:pokerarena.baselines.CallingStation"
"""


REPLAYS = ""  # appended bot TOML (none); replays are enabled through the [run] section


def cfg(hands, py=True, extra="", replays=0):
    base = BASE.format(hands=hands)
    if replays:
        base = base.replace("seed = 5", f"seed = 5\nreplay_hands_per_bot = {replays}")
    return base + BASE_BOTS + (PY_BOTS if py else "") + extra


def children():
    out = subprocess.run(["pgrep", "-P", str(os.getpid())], capture_output=True, text=True).stdout
    return sorted(int(p) for p in out.split())


def os_threads():
    out = subprocess.run(["ps", "-M", "-p", str(os.getpid())], capture_output=True, text=True).stdout
    return len(out.strip().splitlines()) - 1


def fds():
    return len(os.listdir("/dev/fd"))


def wait_for(pred, timeout=30.0):
    t0 = time.time()
    while not pred():
        assert time.time() - t0 < timeout, "timed out"
        time.sleep(0.01)


def baseline():
    # Settle: rayon pool threads exit asynchronously after a run.
    time.sleep(0.3)
    return children(), threading.active_count(), fds(), os_threads()


def assert_no_leak(before, slack=0):
    """Same child processes; no growth in Python threads, file descriptors or OS threads.
    (Counts may shrink when helpers from earlier tests finish exiting.)"""
    after = baseline()
    assert after[0] == before[0], f"child processes leaked: {before[0]} -> {after[0]}"
    assert after[1] <= before[1], f"Python threads leaked: {before[1]} -> {after[1]}"
    assert after[2] <= before[2] + slack, f"fds leaked: {before[2]} -> {after[2]}"
    assert after[3] <= before[3] + slack, f"OS threads leaked: {before[3]} -> {after[3]}"


@pytest.fixture(scope="module", autouse=True)
def warm_up():
    # Starts multiprocessing's resource tracker (a long-lived helper child) before baselines.
    pa.ArenaJob(cfg(20), save=False, rounds=1, processes=2).run()


def test_completes_with_exact_progress():
    job = pa.ArenaJob(cfg(200), save=False, processes=2)
    seen = []
    job.start()
    while job.state == "running":
        seen.append(job.progress().hands_done)
        time.sleep(0.005)
    assert job.wait(10)
    p = job.progress()
    assert job.state == "done" and p.fraction == 1.0
    assert p.hands_done == p.hands_total == job.tables_total * 200
    assert seen == sorted(seen), "progress must be monotone"
    assert job.result.rows and job.result.hands == p.hands_total


def test_stop_is_fast_and_writes_nothing(tmp_path):
    before = baseline()
    lb = tmp_path / "lb.json"
    job = pa.ArenaJob(cfg(5_000_000, py=False, replays=100), save=True, leaderboard=lb,
                      results_dir=tmp_path / "results", replays_dir=tmp_path / "replays",
                      processes=3)
    job.start()
    wait_for(lambda: job.progress().hands_done > 5000)
    t0 = time.perf_counter()
    job.stop()
    assert job.wait(5)
    latency = time.perf_counter() - t0
    assert job.state == "stopped" and job.result is None and job.error is None
    assert latency < 3.0, latency
    n = job.progress().hands_done
    time.sleep(0.1)
    assert job.progress().hands_done == n, "hands kept running after stop"
    assert not lb.exists() and not (tmp_path / "results").exists(), "a stopped run wrote files"
    assert not (tmp_path / "replays").exists(), "a stopped run wrote hand replays"
    assert_no_leak(before)


def test_stop_with_python_workers():
    before = baseline()
    job = pa.ArenaJob(cfg(2_000_000), save=False, processes=3)
    job.start()
    # Hand-level progress also flows back from worker processes.
    wait_for(lambda: job.progress().hands_done > 20_000)
    assert len(children()) > len(before[0]), "workers should be running"
    t0 = time.perf_counter()
    job.stop()
    assert job.wait(10)
    assert time.perf_counter() - t0 < 3.0
    assert job.state == "stopped"
    assert_no_leak(before)


def test_stop_kills_hung_python_bot():
    before = baseline()
    extra = '\n[[bots]]\nname = "sleeper"\nkind = "python:tests.bots_for_tests.Sleeper"\n'
    job = pa.ArenaJob(cfg(1000, py=False, extra=extra), save=False, processes=2)
    job.start()
    time.sleep(1.5)  # workers are now stuck inside Sleeper.act
    assert job.state == "running"
    t0 = time.perf_counter()
    job.stop()
    assert job.wait(10), "driver thread did not finish"
    assert time.perf_counter() - t0 < 3.0
    assert job.state == "stopped"
    assert_no_leak(before)


def test_repeated_start_stop_no_leaks():
    before = baseline()
    for i in range(8):
        job = pa.ArenaJob(cfg(1_000_000, py=(i % 2 == 0)), save=False, processes=2)
        job.start()
        time.sleep(0.05 * (i % 4))
        job.stop()
        assert job.wait(10)
        assert job.state in ("stopped", "done")
    assert_no_leak(before)


def test_stop_edge_cases():
    job = pa.ArenaJob(cfg(50), save=False)
    job.stop()  # before start: the run is stopped immediately
    with pytest.raises(pa.Cancelled):
        job.run()
    done = pa.ArenaJob(cfg(20, py=False), save=False)
    done.run()
    done.stop()  # after completion: no-op
    assert done.state == "done" and done.result is not None


def test_no_semaphore_or_process_leaks_at_exit(tmp_path):
    # A fresh interpreter that starts and nukes jobs, then exits: multiprocessing's resource
    # tracker would print "leaked semaphore" warnings if anything was left behind.
    script = tmp_path / "nuke.py"
    # The __main__ guard is required with the "spawn" start method: workers re-import the
    # main script, and without it each worker would start its own jobs.
    script.write_text(textwrap.dedent(f"""
        import time, pokerarena as pa

        if __name__ == "__main__":
            cfg = {cfg(1_000_000)!r}
            for i in range(3):
                job = pa.ArenaJob(cfg, save=False, processes=2)
                job.start()
                time.sleep(0.4)
                job.stop()
                assert job.wait(10) and job.state == "stopped"
            print("ok")
    """))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=here,
                       timeout=120, env={**os.environ, "PYTHONPATH": here})
    assert p.returncode == 0, p.stderr
    assert p.stdout.strip() == "ok"
    assert "leaked" not in p.stderr.lower(), p.stderr
    assert "Traceback" not in p.stderr, p.stderr


def test_worker_crash_fails_fast_instead_of_hanging():
    before = baseline()
    extra = '\n[[bots]]\nname = "exiter"\nkind = "python:tests.bots_for_tests.Exiter"\n'
    job = pa.ArenaJob(cfg(1000, py=False, extra=extra), save=False, processes=2)
    t0 = time.perf_counter()
    job.start()
    assert job.wait(15), "job hung after a worker died"
    assert time.perf_counter() - t0 < 10
    assert job.state == "error"
    assert "worker process died" in str(job.error) and "exit code 3" in str(job.error)
    assert_no_leak(before)


def test_hand_replays_written_per_bot(tmp_path):
    job = pa.ArenaJob(cfg(300, replays=25), save=False, processes=2,
                      replays_dir=tmp_path / "replays")
    res = job.run()
    d = tmp_path / "replays" / job.run_id
    assert res.replays_dir == str(d)
    files = sorted(p.name for p in d.iterdir())
    assert files == sorted(f"{k}.txt" for k in job.keys)
    for k in job.keys:
        text = (d / f"{k}.txt").read_text()
        assert text.count("### Round") == 25, k
        assert text.count("Hand #") == 25
        # Every sampled hand involves that bot.
        for hand in text.split("### Round")[1:]:
            assert f": {k} (" in hand, (k, hand[:200])
    # Replays don't change results.
    plain = pa.ArenaJob(cfg(300), save=False, processes=2).run()
    assert plain.rows == res.rows


def test_saved_ratings_are_reused_except_anchors(tmp_path):
    lb = tmp_path / "lb.json"
    first = pa.ArenaJob(cfg(200), save=True, leaderboard=lb, results_dir=tmp_path / "r",
                        processes=2).run()
    job2 = pa.ArenaJob(cfg(200), save=True, leaderboard=lb, results_dir=tmp_path / "r",
                       processes=2)
    for key, start in job2.initial_ratings.items():
        assert start == first.ratings[key], key
    assert (job2.initial_ratings["tag"], job2.initial_ratings["random"],
            job2.initial_ratings["station"]) == (1300, 850, 400)
    fresh = pa.ArenaJob(cfg(200), save=False, processes=2)
    assert fresh.initial_ratings["py_tag"] == 1500
