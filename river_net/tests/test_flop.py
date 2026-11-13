"""Flop solver: showdown matrix and features vs references, check-down exactness, zero sum,
convergence, flop spots, turn-network leaves (when the model is trained)."""

from pathlib import Path

import numpy as np
import pytest

from pokerarena import _native
from river_net import cards
from river_net.flop import (CheckdownFlopLeaf, FixedTurnLeaf, FlopSolver, TurnNetLeaf,
                            TurnValueNet)
from river_net.preriver import turn_features, turn_features_reference
from river_net.ranges import RangeGenerator, effective_combos
from river_net.solver import Spot

TURN_MODEL = "river_net/models/turn_k256.pt"
have_model = pytest.mark.skipif(not Path(TURN_MODEL).exists(), reason="turn model not trained")


def test_native_turn_features_match_reference():
    b = cards.parse_board("Td9d4s2h")
    a, r = turn_features(b), turn_features_reference(b)
    assert np.array_equal(a[0], r[0]) and np.array_equal(a[1], r[1])


def test_flop_showdown_matrix_matches_turn_sums():
    board = [51, 30, 9]
    W = np.asarray(_native.solver_flop_showdown_matrix(board)).reshape(1326, 1326)
    assert np.array_equal(W, -W.T)
    q = np.random.default_rng(0).random(1326).astype(np.float32)
    q[cards.blocked(np.array(board, dtype=np.uint8))] = 0
    q /= q.sum()
    ref = sum(np.asarray(_native.TurnBoards(board + [t]).showdown_net_sum(q), dtype=np.float64)
              for t in range(52) if t not in board)
    assert np.allclose(W @ q, ref / 2, atol=1e-3)  # ordered (turn, river) pairs count twice


def test_flop_spots():
    g = RangeGenerator(5)
    for i in range(6):
        s = g.generate(i, street="flop")
        assert len(s.board) == 3 and s.meta["line"] == ""
        for r in (s.r1, s.r2):
            assert r.sum() == pytest.approx(1, abs=1e-5) and effective_combos(r) >= 20


def test_checkdown_flop_at_zero_stack_is_runout_equity():
    s = RangeGenerator(7).generate(1, street="flop")
    fs = FlopSolver(Spot(s.board, s.pot, 0.0, s.r1, s.r2), CheckdownFlopLeaf())
    res = fs.solve(iters=3, target_expl=0.0, check_every=0)
    W = fs.boards.showdown_matrix().reshape(1326, 1326)
    m = fs.m[0]
    ok = m > 0
    eq = 0.5 + 0.5 * (W @ fs.r[1]) / (990 * np.where(ok, m, 1))
    assert np.allclose(res.v1[ok], eq[ok] - 0.5, atol=1e-5)


def test_checkdown_flop_solver_zero_sum_and_converges():
    s = RangeGenerator(7).generate(0, street="flop")
    fs = FlopSolver(s, CheckdownFlopLeaf())
    res = fs.solve(iters=150, target_expl=0.0, check_every=50)
    assert abs((fs.r[0] * res.cfv1).sum() + (fs.r[1] * res.cfv2).sum()) < 1e-6
    h = [e for _, e in res.history]
    assert h[-1] < h[0] and h[-1] < 0.03


@have_model
def test_turn_net_leaf_flop_solver():
    net = TurnValueNet.load(TURN_MODEL)
    s = RangeGenerator(7).generate(3, street="flop")
    fs = FlopSolver(s, TurnNetLeaf(net, threads=2), turn_samples=8, seed=1)
    res = fs.solve(iters=30, target_expl=0.0, check_every=30)
    assert np.isfinite(res.expl) and np.isfinite(res.v1).all()
    assert abs((fs.r[0] * res.cfv1).sum() + (fs.r[1] * res.cfv2).sum()) < 1e-5
    # Per-card values through the fixed leaf reproduce the network leaf's sum.
    R1, R2 = fs.leaf_ranges()
    V, _ = fs.leaf.card_values(R1, R2, fs.leaf_c, root=fs.r)
    direct = fs.leaf.values(R1, R2, fs.leaf_c, root=fs.r)
    fixed = FixedTurnLeaf(V)
    fixed.bind(fs.boards, fs.S)
    via = fixed.values(R1, R2, fs.leaf_c)
    for p in range(2):
        assert np.allclose(direct[p], via[p], atol=1e-6)
