"""Turn-solver speed paths: native leaf kernels (card subsets), refresh / sampling options,
resumable solves, turn buckets, the pre-river teacher and leaf."""

from pathlib import Path

import numpy as np
import pytest

from pokerarena import _native
from river_net import cards
from river_net.model import RiverValueNet
from river_net.preriver import Teacher, turn_bucket_map, turn_features
from river_net.ranges import RangeGenerator
from river_net.turn import NetworkLeaf, TurnSolver, compat_mass

MODEL = "river_net/models/river_k64.pt"
PRE_MODEL = "river_net/models/preriver_k256.pt"


@pytest.fixture(scope="module")
def net():
    return RiverValueNet.load(MODEL)


@pytest.fixture(scope="module")
def solved(net):
    s = RangeGenerator(7).generate(1, street="turn")
    ts = TurnSolver(s, NetworkLeaf(net, threads=2))
    ts.solve(iters=20, target_expl=0.0, check_every=0)
    return s, ts


def test_fast_values_match_dense_reference(solved):
    _, ts = solved
    R1, R2 = ts.leaf_ranges()
    R1[0] = 0  # a leaf one player never reaches: root fallback
    fast = ts.leaf.values(R1, R2, ts.leaf_c, root=ts.r)
    V, M = ts.leaf.card_values(R1.copy(), R2.copy(), ts.leaf_c, root=ts.r)
    scale = (1 + 2 * ts.leaf_c) / 44.0
    for p in range(2):
        ref = scale[:, None] * (M[p] * V[p]).sum(axis=1)
        assert np.allclose(fast[p], ref, atol=1e-6)


def test_card_subsets_partition_the_full_sum(solved):
    # Splitting the 48 river cards into disjoint subsets: the subset sums add up to the full
    # sum, so the scaled subset estimate is unbiased; the all-in sum likewise.
    _, ts = solved
    leaf = ts.leaf
    R1, R2 = ts.leaf_ranges()
    full = leaf.expand(leaf.query(R1, R2, ts.leaf_c, root=ts.r), R1, R2, ts.leaf_c)
    parts = np.array_split(np.random.default_rng(0).permutation(48), 6)
    acc = [np.zeros_like(full[0]), np.zeros_like(full[1])]
    for sub in parts:
        sub = np.sort(sub)
        est = leaf.expand(leaf.query(R1, R2, ts.leaf_c, root=ts.r, cards=sub), R1, R2, ts.leaf_c)
        for p in range(2):
            acc[p] += est[p] * len(sub) / 48
    for p in range(2):
        assert np.allclose(acc[p], full[p], atol=1e-6)
    q = ts.r[1].astype(np.float32)
    tot = np.asarray(ts.boards.showdown_net_sum(q))
    split = sum(np.asarray(ts.boards.showdown_net_sum(q, [int(c) for c in sub])) for sub in parts)
    assert np.allclose(split, tot, atol=1e-5)


def test_refresh_every_iteration_is_the_plain_solver(net):
    s = RangeGenerator(7).generate(2, street="turn")
    a = TurnSolver(s, NetworkLeaf(net)).solve(iters=15, target_expl=0.0, check_every=0)
    b = TurnSolver(s, NetworkLeaf(net), refresh=1).solve(iters=15, target_expl=0.0, check_every=0)
    assert np.allclose(a.strategy1, b.strategy1, atol=1e-9)
    assert np.allclose(a.v2, b.v2, atol=1e-9)


def test_stale_and_sampled_solves_run(net):
    s = RangeGenerator(7).generate(2, street="turn")
    for kw in ({"refresh": 3, "refresh_warmup": 5}, {"river_samples": 8, "seed": 3}):
        ts = TurnSolver(s, NetworkLeaf(net), **kw)
        res = ts.solve(iters=20, target_expl=0.0, check_every=20)
        assert np.isfinite(res.expl) and np.isfinite(res.v1).all()


def test_solve_continues_from_its_state(net):
    s = RangeGenerator(7).generate(3, street="turn")
    a = TurnSolver(s, NetworkLeaf(net))
    a.solve(iters=10, target_expl=0.0, check_every=0)
    ra = a.solve(iters=10, target_expl=0.0, check_every=0)
    rb = TurnSolver(s, NetworkLeaf(net)).solve(iters=20, target_expl=0.0, check_every=0)
    assert ra.iters == rb.iters == 20
    assert np.allclose(ra.strategy2, rb.strategy2, atol=1e-12)


@pytest.mark.parametrize("scheme,K", [("ehs", 64), ("ehs2:8", 256)])
def test_turn_buckets(scheme, K):
    board = cards.parse_board("AsKd7h2c")
    f = turn_features(board)
    bm = turn_bucket_map(board, K, scheme, f)
    valid = ~cards.blocked(board)
    assert np.all(bm[~valid] == -1) and np.all((bm[valid] >= 0) & (bm[valid] < K))
    # Suit isomorphs (same features) share a bucket; EHS is in [0, 1].
    assert np.all((f[0][valid] >= 0) & (f[0][valid] <= 1))
    key = np.round(np.stack(f, 1) * 1e9)
    for u in np.unique(key[valid], axis=0)[:50]:
        same = valid & np.all(key == u, axis=1)
        assert len(np.unique(bm[same])) == 1
    if scheme == "ehs":  # 1-D buckets are ordered by EHS
        order = np.argsort(f[0][valid], kind="stable")
        assert np.all(np.diff(bm[valid][order]) >= 0)


def test_teacher_targets_zero_sum_and_match_network_leaf(net):
    s = RangeGenerator(11).generate(5, street="turn_end")
    v1, v2, m1, m2 = Teacher(net, threads=2)(s.board, s.spr, s.r1, s.r2)
    r1, r2 = s.r1.astype(np.float64), s.r2.astype(np.float64)
    assert abs((r1 * m1 * v1).sum() + (r2 * m2 * v2).sum()) < 1e-6
    assert np.allclose(m1, compat_mass(r2) * ~cards.blocked(s.board))
    # Same as the turn solver's network leaf at c = 0 with stack = river SPR.
    leaf = NetworkLeaf(net)
    leaf.bind(_native.TurnBoards(s.board.tolist()), s.spr)
    cfv = leaf.values(r1[None], r2[None], np.zeros(1))
    assert np.allclose(cfv[0][0], m1 * v1, atol=1e-7)


@pytest.mark.skipif(not Path(PRE_MODEL).exists(), reason="pre-river model not trained")
def test_preriver_leaf_solver_runs_and_is_near_zero_sum():
    from river_net.preriver import PreRiverLeaf, PreRiverNet

    pre = PreRiverNet.load(PRE_MODEL)
    s = RangeGenerator(7).generate(4, street="turn")
    ts = TurnSolver(s, PreRiverLeaf(pre))
    res = ts.solve(iters=40, target_expl=0.0, check_every=40)
    zs = (ts.r[0] * res.cfv1).sum() + (ts.r[1] * res.cfv2).sum()
    assert abs(zs) < 1e-5 and np.isfinite(res.expl)
    assert res.leaf_queries == (40 + 1 + 1) * len(ts.leaves)  # iterations, 1 check, final
