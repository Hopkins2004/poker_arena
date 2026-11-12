"""Range generator: scenario table, normalisation, blocking, determinism, throughput."""

import time

import numpy as np
import pytest

from river_net import cards
from river_net.ranges import (
    RangeGenerator, effective_combos, percentiles, recursive_split, scenario_table,
)


def test_scenario_table_follows_the_chart():
    sc = scenario_table()
    assert len(sc) == 38
    assert sum(s.prob for s in sc) == pytest.approx(1.0)
    by = {}
    for s in sc:
        by[s.pot_type] = by.get(s.pot_type, 0) + s.prob
    assert by["srp"] > by["3bet"] > by["4bet"] > 0
    # Each opener gets 1/5 of the mass (the user's procedure).
    for opener in ("UTG", "HJ", "CO", "BTN", "SB"):
        m = sum(s.prob for s in sc if s.line.startswith(opener + " raise"))
        assert m == pytest.approx(0.2)
    btn = next(s for s in sc if s.line == "BTN raise 2.5bb, BB call")
    assert (btn.oop, btn.ip) == ("BB", "BTN") and btn.pot_bb == 5.5 and btn.stack_bb == 97.5


@pytest.mark.parametrize("source", ["A", "B", "C"])
def test_spots_are_normalised_blocked_and_in_bounds(source):
    g = RangeGenerator(1)
    for i in range(20):
        s = g.generate(i, source=source)
        blocked = cards.blocked(s.board)
        for r in (s.r1, s.r2):
            assert r.shape == (1326,) and r.dtype == np.float32
            assert np.all(r >= 0) and np.all(r[blocked] == 0)
            assert r.sum() == pytest.approx(1.0, abs=1e-5)
            assert effective_combos(r) >= 20
        assert 0.1 <= s.spr <= 20
        assert s.meta["source"] == source and s.meta["scenario"]


def test_generation_is_deterministic():
    a, b = RangeGenerator(5), RangeGenerator(5)
    for i in (0, 3, 17):
        x, y = a.generate(i), b.generate(i)
        assert np.array_equal(x.r1, y.r1) and np.array_equal(x.r2, y.r2)
        assert np.array_equal(x.board, y.board) and x.pot == y.pot and x.stack == y.stack
        assert x.meta == y.meta
    assert not np.array_equal(a.generate(1).r1, RangeGenerator(6).generate(1).r1)


def test_source_a_filters_shape_the_ranges():
    # A bet-call line moves mass towards stronger hands than a check-check line.
    g = RangeGenerator(2)
    s = [g.generate(i, source="A") for i in range(60)]
    assert {x.meta["line"].split("/")[0][:2] for x in s} >= {"cc", "bc"}


def test_recursive_split_and_percentiles():
    rng = np.random.default_rng(0)
    board = cards.random_board(rng)
    st = cards.river_strengths(board).astype(np.int64)
    valid = np.flatnonzero(st > 0)
    r = recursive_split(st, valid, rng)
    assert r.sum() == pytest.approx(1.0) and np.all(r[st == 0] == 0)
    # Equal strengths get equal weight.
    for s in np.unique(st[valid])[:20]:
        g = r[st == s]
        assert np.allclose(g, g[0])
    w = np.zeros(1326)
    w[valid] = 1.0
    q = percentiles(st.astype(float), w)
    assert 0 < q[valid].min() and q[valid].max() < 1


def test_throughput_per_core():
    g = RangeGenerator(9)
    g.generate(0)  # warm up (evaluator tables)
    t = time.time()
    n = 60
    for i in range(1, n + 1):
        g.generate(i)
    rate = n / (time.time() - t)
    assert rate >= 20, f"{rate:.1f} spots/s"
