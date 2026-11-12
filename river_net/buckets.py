"""Per-board strength buckets.

The valid hands of a board are ordered by river strength and cut into ``K`` buckets of
roughly equal numbers of combos, never splitting a tie group: each tie group goes to the
bucket containing its midpoint, ``floor(K · (combos before the group + size/2) / n)``. A
bucket's index therefore encodes relative showdown strength. Groups larger than ``n/K``
(e.g. everyone playing the board) leave some buckets empty.

A bucket map is an int16[1326] array: the bucket of each combo, −1 for blocked combos.
Batched helpers take ``bmap[N, 1326]`` and work on all samples at once.
"""

from __future__ import annotations

import numpy as np

from .cards import N_COMBOS, river_strengths


def bucket_map(strength: np.ndarray, K: int) -> np.ndarray:
    """Bucket of every combo from its river strength (0 = blocked)."""
    strength = np.asarray(strength)
    valid = np.flatnonzero(strength > 0)
    n = len(valid)
    out = np.full(N_COMBOS, -1, dtype=np.int16)
    if n == 0:
        return out
    order = valid[np.argsort(strength[valid], kind="stable")]
    s = strength[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    ends = np.r_[starts[1:], n]
    mid = (starts + ends) / 2.0
    gb = np.minimum((K * mid / n).astype(np.int64), K - 1)
    sizes = ends - starts
    out[order] = np.repeat(gb, sizes).astype(np.int16)
    return out


def board_bucket_map(board, K: int) -> np.ndarray:
    return bucket_map(river_strengths(board), K)


def _flat(bmap: np.ndarray, K: int):
    """Flattened (sample * K + bucket) indices and a validity mask for batched bincounts."""
    bmap = np.atleast_2d(bmap)
    valid = bmap >= 0
    rows = np.arange(bmap.shape[0])[:, None]
    return np.where(valid, rows * K + bmap, 0), valid, bmap.shape[0]


def to_buckets(r: np.ndarray, bmap: np.ndarray, K: int) -> np.ndarray:
    """Range mass per bucket: ``[K]`` (or ``[N, K]`` for batches)."""
    single = np.ndim(bmap) == 1
    idx, valid, N = _flat(bmap, K)
    out = np.bincount(idx[valid], weights=np.atleast_2d(r)[valid], minlength=N * K)
    out = out.reshape(N, K)
    return out[0] if single else out


def bucket_counts(bmap: np.ndarray, K: int) -> np.ndarray:
    single = np.ndim(bmap) == 1
    idx, valid, N = _flat(bmap, K)
    out = np.bincount(idx[valid], minlength=N * K).reshape(N, K)
    return out[0] if single else out


def bucket_targets(v: np.ndarray, weights: np.ndarray, bmap: np.ndarray, K: int) -> np.ndarray:
    """Weighted mean of per-hand values per bucket (weights ``m_i(h)``), falling back to the
    plain mean over the bucket's hands where all weights are 0; 0 for empty buckets."""
    single = np.ndim(bmap) == 1
    idx, valid, N = _flat(bmap, K)
    v2, w2 = np.atleast_2d(v), np.atleast_2d(weights)
    num = np.bincount(idx[valid], weights=(v2 * w2)[valid], minlength=N * K)
    den = np.bincount(idx[valid], weights=w2[valid], minlength=N * K)
    plain = np.bincount(idx[valid], weights=v2[valid], minlength=N * K)
    cnt = np.bincount(idx[valid], minlength=N * K)
    out = np.where(den > 0, num / np.where(den > 0, den, 1),
                   np.where(cnt > 0, plain / np.maximum(cnt, 1), 0.0))
    out = out.reshape(N, K)
    return out[0] if single else out


def broadcast(vb: np.ndarray, bmap: np.ndarray) -> np.ndarray:
    """Per-hand values from bucket values (each hand gets its bucket's value; 0 if blocked)."""
    vb = np.atleast_2d(vb)
    b2 = np.atleast_2d(bmap)
    rows = np.arange(b2.shape[0])[:, None]
    out = np.where(b2 >= 0, vb[rows, np.maximum(b2, 0)], 0.0)
    return out[0] if np.ndim(bmap) == 1 else out
