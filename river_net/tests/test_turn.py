"""Turn solver: per-card helpers vs brute force, check-down exactness, zero sum, convergence,
determinism, network leaves, turn spot generation."""

import numpy as np
import pytest

from pokerarena import _native
from river_net import cards
from river_net.model import RiverValueNet
from river_net.ranges import RangeGenerator, effective_combos
from river_net.solver import Spot
from river_net.turn import CheckdownLeaf, NetworkLeaf, TurnSolver, compat_mass

MODEL = "river_net/models/river_k64.pt"


def turn_spot(seed, spr=3.0):
    rng = np.random.default_rng(seed)
    b = cards.random_board(rng, 4)
    v = ~cards.blocked(b)
    r1 = rng.random(1326) * v * (rng.random(1326) > 0.4)
    r2 = rng.random(1326) * v * (rng.random(1326) > 0.4)
    return Spot(b, 10.0, 10.0 * spr, (r1 / r1.sum()).astype(np.float32),
                (r2 / r2.sum()).astype(np.float32))


def test_turn_boards_match_river_equity():
    s = turn_spot(0)
    tb = _native.TurnBoards(s.board.tolist())
    assert len(tb.cards) == 48 and not set(tb.cards) & set(s.board.tolist())
    q = s.r2.astype(np.float32)
    m = np.asarray(tb.masses(np.stack([q, 2 * q]).ravel())).reshape(2, 48, 1326)
    net_sum = np.zeros(1326)
    for k, c in enumerate(tb.cards):
        eq, mass = _native.solver_river_equity(s.board.tolist() + [c], q)
        assert np.allclose(m[0, k], mass, atol=1e-6) and np.allclose(m[1, k], 2 * mass, atol=1e-5)
        net_sum += (2 * eq - 1) * mass   # win - loss = (2 eq - 1) * mass
    assert np.allclose(tb.showdown_net_sum(q), net_sum, atol=1e-5)
    # Turn-board compatible mass in numpy.
    valid = ~cards.blocked(s.board)
    h = 17 if valid[17] else int(np.flatnonzero(valid)[0])
    a, b = cards.HAND_CARDS[h]
    brute = sum(q[g] for g in range(1326) if not {a, b} & set(cards.HAND_CARDS[g].tolist()))
    assert compat_mass(q.astype(np.float64))[h] == pytest.approx(brute, rel=1e-6)


def test_checkdown_leaf_at_zero_stack_is_turn_equity():
    # SPR 0: no betting, the only line is check-check to a river check-down: v = equity - 1/2.
    s = turn_spot(1, spr=0.0)
    ts = TurnSolver(s, CheckdownLeaf())
    res = ts.solve(iters=5, target_expl=0.0, check_every=5)
    eq = _native.solver_range_equity(s.board.tolist(), s.r2.astype(np.float32), 1, 0)
    valid = ~cards.blocked(s.board) & (res.m1 > 0)
    assert np.allclose(res.v1[valid], eq[valid] - 0.5, atol=1e-5)


@pytest.mark.parametrize("seed,spr", [(2, 1.5), (3, 6.0)])
def test_checkdown_solver_zero_sum_converges_and_is_deterministic(seed, spr):
    s = turn_spot(seed, spr)
    a = TurnSolver(s, CheckdownLeaf()).solve(iters=300, target_expl=0.0, check_every=50)
    ts = TurnSolver(s, CheckdownLeaf())
    zs = (ts.r[0] * a.cfv1).sum() + (ts.r[1] * a.cfv2).sum()
    assert abs(zs) < 1e-8
    h = [e for _, e in a.history]
    assert h[-1] < h[0] and h[-1] < 0.01
    b = TurnSolver(s, CheckdownLeaf()).solve(iters=300, target_expl=0.0, check_every=50)
    assert np.array_equal(a.v1, b.v1) and np.array_equal(a.strategy2, b.strategy2)
    assert np.all(a.v1[cards.blocked(s.board)] == 0)


def test_network_leaf_solver_runs_and_is_zero_sum():
    net = RiverValueNet.load(MODEL)
    s = RangeGenerator(3).generate(0, street="turn")
    ts = TurnSolver(s, NetworkLeaf(net, threads=2))
    res = ts.solve(iters=60, target_expl=0.0, check_every=30)
    assert np.isfinite(res.v1).all() and np.isfinite(res.v2).all()
    zs = (ts.r[0] * res.cfv1).sum() + (ts.r[1] * res.cfv2).sum()
    assert abs(zs) < 1e-5  # bucket-level projection carries through the card average
    # One batch per iteration + one per exploitability check (2) + the final value pass.
    assert res.leaf_queries == (60 + 2 + 1) * len(ts.leaves) * 48
    assert res.expl < 0.05


@pytest.mark.parametrize("source", ["A", "B", "C"])
def test_turn_spots(source):
    g = RangeGenerator(4)
    for i in range(5):
        s = g.generate(i, source=source, street="turn")
        assert len(s.board) == 4
        blocked = cards.blocked(s.board)
        for r in (s.r1, s.r2):
            assert r.sum() == pytest.approx(1, abs=1e-5) and np.all(r[blocked] == 0)
            assert effective_combos(r) >= 20
        assert 0.1 <= s.spr <= 20 and "/" not in s.meta["line"]
    x, y = g.generate(2, street="turn"), RangeGenerator(4).generate(2, street="turn")
    assert np.array_equal(x.r1, y.r1) and np.array_equal(x.board, y.board)
