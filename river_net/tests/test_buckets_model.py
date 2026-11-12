"""Buckets, dataset split, model projection and the predict interface."""

import numpy as np
import pytest
import torch

from river_net import cards
from river_net.buckets import bucket_counts, bucket_map, bucket_targets, broadcast, to_buckets
from river_net.dataset import board_split
from river_net.model import RiverNet, RiverValueNet
from river_net.ranges import RangeGenerator
from river_net.solver import solve


@pytest.mark.parametrize("K", [16, 64, 256])
def test_buckets_are_strength_ordered_and_respect_ties(K):
    rng = np.random.default_rng(K)
    for board in [cards.random_board(rng) for _ in range(5)] + [cards.parse_board("AsKsQsJsTs")]:
        st = cards.river_strengths(board).astype(np.int64)
        b = bucket_map(st, K)
        valid = st > 0
        assert np.all(b[~valid] == -1) and b[valid].min() >= 0 and b[valid].max() < K
        # Monotone in strength, and tie groups never split.
        order = np.argsort(st[valid], kind="stable")
        assert np.all(np.diff(b[valid][order]) >= 0)
        for s in np.unique(st[valid]):
            assert len(np.unique(b[st == s])) == 1
        counts = bucket_counts(b, K)
        assert counts.sum() == valid.sum()
        if len(np.unique(st[valid])) > 4 * K:  # many groups: roughly equal sizes
            assert counts.max() <= 3 * valid.sum() / K


def test_bucket_helpers():
    rng = np.random.default_rng(0)
    board = cards.random_board(rng)
    b = bucket_map(cards.river_strengths(board), 16)
    r = rng.random(1326) * (b >= 0)
    assert to_buckets(r, b, 16).sum() == pytest.approx(r.sum())
    v, m = rng.normal(size=1326), rng.random(1326) * (b >= 0)
    vb = bucket_targets(v, m, b, 16)
    k = int(b[b >= 0][0])
    sel = b == k
    assert vb[k] == pytest.approx((v[sel] * m[sel]).sum() / m[sel].sum())
    assert bucket_targets(v, np.zeros(1326), b, 16)[k] == pytest.approx(v[sel].mean())
    out = broadcast(vb, b)
    assert np.all(out[b < 0] == 0) and np.allclose(out[sel], vb[k])
    # Batched form equals the per-sample form.
    B = np.stack([b, b])
    assert np.allclose(to_buckets(np.stack([r, 2 * r]), B, 16)[1], 2 * to_buckets(r, b, 16))


def test_board_split_is_by_board_and_balanced():
    rng = np.random.default_rng(1)
    boards = [cards.random_board(rng) for _ in range(4000)]
    s = [board_split(b) for b in boards]
    assert board_split(boards[0]) == board_split(boards[0][::-1])  # order-free
    frac = {k: s.count(k) / len(s) for k in ("train", "val", "test")}
    assert frac["train"] == pytest.approx(0.9, abs=0.02)
    assert frac["test"] == pytest.approx(0.05, abs=0.015)


def test_projection_zero_sum_and_predict_interface():
    torch.manual_seed(0)
    K = 16
    m = RiverNet(K, hidden=32, depth=2)
    x = torch.randn(8, 2 * K + 2)
    w = torch.rand(8, 2 * K)
    y, delta = m(x, w)
    assert torch.allclose((w * y).sum(1), torch.zeros(8), atol=1e-5)
    net = RiverValueNet(m)
    spot = RangeGenerator(0).generate(0)
    v1, v2 = net.predict(spot.board, spot.pot, spot.stack, spot.r1, spot.r2)
    blocked = cards.blocked(spot.board)
    assert v1.shape == (1326,) and np.all(v1[blocked] == 0) and np.all(v2[blocked] == 0)
    # The projection makes the bucket-level root values zero-sum.
    res = solve(spot)
    b = bucket_map(cards.river_strengths(spot.board), K)
    w1 = to_buckets(spot.r1 * res.m1, b, K)
    w2 = to_buckets(spot.r2 * res.m2, b, K)
    vb1, vb2 = [np.array([vv[b == k][0] if (b == k).any() else 0 for k in range(K)])
                for vv in (v1, v2)]
    assert abs((w1 * vb1).sum() + (w2 * vb2).sum()) < 1e-4
