"""Dataset access: shards, the board split, and bucketed training arrays.

* Split by board, not by sample: a board's sorted 5-card set is hashed to 0..99; < 90 is
  train, < 95 validation, the rest test. Test boards are never seen in training.
* Bucketing is a one-off preprocessing pass (``bucketed(data_dir, K)``), cached as
  ``<data_dir>/bucketed_K<K>.npz``: per sample the bucket masses ``rb1, rb2``, the bucket
  targets ``vb1, vb2`` (``m``-weighted means), the projection weights ``w1, w2``
  (``Σ_{h∈k} r·m``) and a non-empty-bucket mask. Precomputing is much faster than
  bucketing on the fly (one strength evaluation per board instead of one per epoch).
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .buckets import bucket_counts, bucket_map, bucket_targets, to_buckets
from .cards import river_strengths

SPLITS = ("train", "val", "test")
CACHE_VERSION = 1


def board_split(board) -> str:
    key = bytes(sorted(int(c) for c in board))
    h = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little") % 100
    return "train" if h < 90 else "val" if h < 95 else "test"


def manifest(data_dir) -> dict:
    return json.loads((Path(data_dir) / "manifest.json").read_text())


def shard_paths(data_dir) -> list:
    d = Path(data_dir)
    return [d / s["file"] for s in manifest(d)["shards"]]


def load_shard(path, keys=None) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in (keys or z.files)}


def bucket_maps(boards: np.ndarray, K: int, scheme: str | None = None) -> np.ndarray:
    """Bucket maps of a batch of boards: river buckets for 5-card boards, turn buckets
    (:func:`river_net.preriver.turn_bucket_map`, ``scheme``) for 4-card boards."""
    boards = np.asarray(boards)
    if boards.shape[1] == 4:
        from .preriver import DEFAULT_SCHEME, turn_bucket_map

        return np.stack([turn_bucket_map(b, K, scheme or DEFAULT_SCHEME) for b in boards])
    return np.stack([bucket_map(river_strengths(b), K) for b in boards])


def _bucket_shard(args) -> dict:
    path, K, scheme = args
    z = load_shard(path)
    bmap = bucket_maps(z["board"], K, scheme)
    out = {k: z[k] for k in ("index", "board", "pot", "stack", "spr", "expl", "source",
                             "scenario", "line")}
    for p in ("1", "2"):
        r, v, m = z["r" + p], z["v" + p], z["m" + p]
        out["rb" + p] = to_buckets(r, bmap, K).astype(np.float32)
        out["vb" + p] = bucket_targets(v, m, bmap, K).astype(np.float32)
        out["w" + p] = to_buckets(r * m, bmap, K).astype(np.float32)
    out["nonempty"] = bucket_counts(bmap, K) > 0
    out["split"] = np.array([board_split(b) for b in z["board"]])
    return out


def bucketed(data_dir, K: int, workers: int | None = None, scheme: str | None = None) -> dict:
    """All samples bucketed at ``K`` (cached). Arrays are concatenated in shard order.
    Pre-river data (4-card boards) uses turn buckets of ``scheme`` (default
    :data:`river_net.preriver.DEFAULT_SCHEME`), cached per scheme."""
    d = Path(data_dir)
    man = manifest(d)
    n = man.get("n", 0)
    if man["config"].get("kind") in ("preriver", "turn"):
        from .preriver import DEFAULT_SCHEME

        scheme = scheme or DEFAULT_SCHEME
        cache = d / f"bucketed_K{K}_{scheme.replace(':', '-')}.npz"
    else:
        scheme = None
        cache = d / f"bucketed_K{K}.npz"
    if cache.exists():
        with np.load(cache) as z:
            if int(z["n"]) == n and int(z["version"]) == CACHE_VERSION:
                return {k: z[k] for k in z.files if k not in ("n", "version")}
    tasks = [(p, K, scheme) for p in shard_paths(d)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_bucket_shard, tasks)
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    tmp = cache.with_suffix(".tmp.npz")
    np.savez(tmp, n=n, version=CACHE_VERSION, **out)
    tmp.replace(cache)
    return out


@dataclass
class Arrays:
    """Model inputs and targets for one split."""

    x: np.ndarray        # features [N, F]
    y: np.ndarray        # targets [N, 2K]
    w: np.ndarray        # projection weights [N, 2K]
    rb: np.ndarray       # bucket masses [N, 2K] (loss weights)
    mask: np.ndarray     # non-empty buckets [N, 2K]
    meta: dict           # index, source, scenario, line, spr, board


def features(data: dict, idx, board_features: bool = False) -> np.ndarray:
    spr = data["spr"][idx].astype(np.float32)
    cols = [data["rb1"][idx], data["rb2"][idx], spr[:, None], np.log1p(spr)[:, None]]
    if board_features:
        oh = np.zeros((len(spr), 52), dtype=np.float32)
        np.put_along_axis(oh, data["board"][idx].astype(np.int64), 1.0, axis=1)
        cols.append(oh)
    return np.concatenate(cols, axis=1).astype(np.float32)


def arrays(data: dict, split: str, board_features: bool = False, limit: int | None = None,
           seed: int = 0) -> Arrays:
    idx = np.flatnonzero(data["split"] == split)
    if limit is not None and limit < len(idx):
        idx = np.sort(np.random.default_rng(seed).choice(idx, size=limit, replace=False))
    cat = lambda a, b: np.concatenate([data[a][idx], data[b][idx]], axis=1)  # noqa: E731
    meta = {k: data[k][idx] for k in ("index", "source", "scenario", "line", "spr", "board")}
    return Arrays(features(data, idx, board_features), cat("vb1", "vb2"), cat("w1", "w2"),
                  cat("rb1", "rb2"), np.concatenate([data["nonempty"][idx]] * 2, axis=1),
                  meta)
