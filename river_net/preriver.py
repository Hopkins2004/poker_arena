"""Pre-river network: one query per turn leaf instead of one per river card.

DeepStack's auxiliary-network trick. A turn leaf (turn betting closed, chips behind) is a
*turn-end spot* ``(board[4], P, S, r1, r2)``: the river pot ``P``, the stack behind ``S``
and both ranges at the end of turn betting. Its value is the chance average of the river
values over the 48 river cards. With the river network ``f`` (conditional values
``v_c(h)``, river-pot units) the target is

    v̄_p(h) = Σ_c m_c(h) · v_c(h) / Σ_c m_c(h),        Σ_c m_c(h) = 44 · m(h),

the river values averaged over the river cards ``h`` doesn't block, weighted by the
opponent mass ``m_c(h)`` compatible with ``h`` and ``c`` (``m(h)`` is the compatible mass on
the turn board; every compatible opponent hand leaves 44 river cards). ``v̄_p(h)`` is
defined where ``m(h) > 0`` (0 elsewhere). A turn solver then needs
``cfv_p(h) = (1 + 2c) · m_p(h) · v̄_p(h)`` at a leaf where each player put ``c`` turn pots in,
which is exactly :class:`river_net.turn.NetworkLeaf`'s per-card sum, from one network call.

Training data needs no solves: turn-end spots from the range generator
(``street="turn_end"``) labelled by :class:`Teacher` (the river network over all rivers),
written by ``python -m river_net.datagen --kind preriver``.

Turn buckets (``scheme``), per turn board, from the hand's river strengths against a
uniform random hand (``HS_c(h)``, the river equity vs a uniform range on ``board + c``):

* ``"ehs"``: equal-combo buckets of ``EHS(h) = mean_c HS_c(h)`` (1-D, like the river
  buckets but on expected strength);
* ``"ehs2:<K2>"``: potential-aware, ``K/K2`` equal-combo EHS groups, each split into ``K2``
  equal-combo sub-buckets by ``E[HS²]`` (same EHS, more variance = more drawing).

Ties (identical features, e.g. suit isomorphs) never straddle a bucket boundary.
"""

from __future__ import annotations

import numpy as np

from pokerarena import _native

from .buckets import bucket_map
from .cards import HAND_CARDS, N_COMBOS, parse_board
from .turn import NetworkLeaf, compat_mass

DEFAULT_SCHEME = "ehs2:8"


# ------------------------------------------------------------------------- turn buckets

def turn_features(board) -> tuple:
    """``(EHS, E[HS²])`` of every combo on a turn board (0 for blocked combos), with
    ``HS_c`` the river equity against a uniform random hand on ``board + c``."""
    b = parse_board(board)
    valid = ~np.isin(HAND_CARDS, b).any(axis=1)
    s1 = np.zeros(N_COMBOS)
    s2 = np.zeros(N_COMBOS)
    n = np.zeros(N_COMBOS)
    for c in range(52):
        if c in b:
            continue
        ok = valid & ~(HAND_CARDS == c).any(axis=1)
        uni = ok.astype(np.float32)
        eq, _ = _native.solver_river_equity(b.tolist() + [c], uni / uni.sum())
        eq = np.asarray(eq, dtype=np.float64)
        s1 += np.where(ok, eq, 0.0)
        s2 += np.where(ok, eq * eq, 0.0)
        n += ok
    n = np.where(n > 0, n, 1)
    return np.where(valid, s1 / n, 0.0), np.where(valid, s2 / n, 0.0)


def _as_strength(x: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Positive integer keys preserving the order of ``x`` with exact ties (1e-9 grid)."""
    return np.where(valid, 1 + np.round(x * 1e9).astype(np.int64), 0)


def turn_bucket_map(board, K: int, scheme: str = DEFAULT_SCHEME, features=None) -> np.ndarray:
    """Bucket of every combo on a turn board (−1 for blocked), see the module docs."""
    ehs, ehs2 = features if features is not None else turn_features(board)
    b = parse_board(board)
    valid = ~np.isin(HAND_CARDS, b).any(axis=1)
    if scheme == "ehs":
        return bucket_map(_as_strength(ehs, valid), K)
    if scheme.startswith("ehs2:"):
        k2 = int(scheme.split(":")[1])
        if K % k2:
            raise ValueError(f"K = {K} is not a multiple of K2 = {k2}")
        g = bucket_map(_as_strength(ehs, valid), K // k2)
        out = np.full(N_COMBOS, -1, dtype=np.int16)
        for j in range(K // k2):
            members = g == j
            if not members.any():
                continue
            sub = bucket_map(np.where(members, _as_strength(ehs2, valid), 0), k2)
            out[members] = j * k2 + sub[members]
        return out
    raise ValueError(f"unknown turn bucket scheme {scheme!r}")


# ------------------------------------------------------------------------- teacher

class Teacher:
    """Labels turn-end spots with the river network averaged over all river cards."""

    def __init__(self, net, threads: int | None = 1):
        self.leaf = NetworkLeaf(net, threads=threads)

    def __call__(self, board, spr: float, r1, r2):
        """``(v1, v2, m1, m2)``, each ``[1326]``: conditional chance-averaged values (river
        pot units, 0 where ``m = 0``) and compatible opponent masses on the turn board."""
        b = parse_board(board)
        valid = ~np.isin(HAND_CARDS, b).any(axis=1)
        r = [np.where(valid, np.asarray(x, dtype=np.float64), 0.0) for x in (r1, r2)]
        r = [x / x.sum() for x in r]
        self.leaf.bind(_native.TurnBoards(b.tolist()), float(spr))
        cfv = self.leaf.values(r[0][None], r[1][None], np.zeros(1))  # (1/44) Σ_c m_c v_c
        out_v, out_m = [], []
        for p in range(2):
            m = compat_mass(r[1 - p]) * valid
            v = np.where(m > 1e-12, cfv[p][0] / np.where(m > 1e-12, m, 1.0), 0.0)
            out_v.append(v)
            out_m.append(m)
        return out_v[0], out_v[1], out_m[0], out_m[1]


# ------------------------------------------------------------------------- network + leaf

class PreRiverNet:
    """A trained pre-river model (:class:`river_net.model.RiverNet` on turn buckets) plus its
    bucket scheme. Inputs and outputs are those of the river network with turn buckets:
    ``[r̄1, r̄2, SPR, log(1 + SPR)]`` → ``[v̄1, v̄2]`` per bucket, zero-sum projected."""

    def __init__(self, model, scheme: str):
        self.model = model.eval()
        self.K = model.config["K"]
        self.scheme = scheme
        if model.config["board_features"]:
            raise NotImplementedError("board-feature pre-river models are not supported")

    @classmethod
    def load(cls, path) -> "PreRiverNet":
        import torch

        from .model import RiverNet

        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = RiverNet(**ck["config"])
        m.load_state_dict(ck["state_dict"])
        scheme = (ck.get("train_config") or {}).get("scheme") or DEFAULT_SCHEME
        return cls(m, scheme)

    def predict_buckets(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            y, _ = self.model(torch.from_numpy(x), torch.from_numpy(w))
        return y.numpy()

    def predict(self, board, pot, stack, r1, r2):
        """``(v1[1326], v2[1326])``: chance-averaged conditional river values of a turn-end
        spot (river-pot units; pot and stack in any common unit)."""
        b = parse_board(board)
        leaf = PreRiverLeaf(self)
        leaf.bind_board(b, float(stack) / float(pot))
        r = [np.asarray(x, dtype=np.float64)[None] for x in (r1, r2)]
        v = leaf.query(r[0], r[1], np.zeros(1))
        return v[0][0], v[1][0]


class PreRiverLeaf:
    """Turn leaves valued by a :class:`PreRiverNet`: one network row per leaf (instead of
    one per leaf × river card), ``cfv_p(h) = (1 + 2c) · m_p(h) · v̄_p(h)``."""

    name = "pre-river"

    def __init__(self, net: PreRiverNet, threads: int | None = None):
        self.net = net
        self.K = net.K
        if threads:
            import torch

            torch.set_num_threads(threads)
        self.queries = 0

    def bind(self, boards, stack: float):
        rivers = set(int(c) for c in boards.cards)
        self.bind_board(np.array([c for c in range(52) if c not in rivers], dtype=np.uint8),
                        stack)

    def bind_board(self, board, stack: float):
        self.stack = float(stack)
        self.bmap = turn_bucket_map(board, self.K, self.net.scheme)
        self.valid = (self.bmap >= 0).astype(np.float64)
        B = np.zeros((N_COMBOS, self.K))
        ok = np.flatnonzero(self.bmap >= 0)
        B[ok, self.bmap[ok]] = 1.0
        self.B = B
        self.gather = np.maximum(self.bmap, 0).astype(np.int64)

    def query(self, R1, R2, c, root=None, cards=None):
        """Conditional chance-averaged values ``[v̄1, v̄2]`` (``[L, 1326]`` each). ``cards``
        is ignored (there is nothing to sample)."""
        Q = [np.asarray(R1, dtype=np.float64), np.asarray(R2, dtype=np.float64)]
        R = [q.copy() for q in Q]
        if root is not None:
            for p in range(2):
                empty = R[p].sum(axis=1) <= 0
                if empty.any():
                    R[p][empty] = root[p]
        c = np.asarray(c, dtype=np.float64)
        L, K = len(c), self.K
        tot = [R[p] @ self.valid for p in range(2)]
        safe = [np.where(t > 0, t, 1.0) for t in tot]
        M = [compat_mass(Q[1 - p]) * self.valid for p in range(2)]
        rb = [(R[p] @ self.B) / safe[p][:, None] for p in range(2)]
        w = [((R[p] * M[p]) @ self.B) / (safe[p] * safe[1 - p])[:, None] for p in range(2)]
        spr = np.clip((self.stack - c) / (1 + 2 * c), 0, None)[:, None]
        x = np.concatenate([rb[0], rb[1], spr, np.log1p(spr)], axis=1).astype(np.float32)
        y = self.net.predict_buckets(x, np.concatenate(w, axis=1).astype(np.float32))
        self.queries += L
        return [y[:, p * K + self.gather] * self.valid for p in range(2)]

    def expand(self, v, R1, R2, c):
        scale = (1 + 2 * np.asarray(c, dtype=np.float64))[:, None]
        R = [np.asarray(R1, dtype=np.float64), np.asarray(R2, dtype=np.float64)]
        return tuple(scale * compat_mass(R[1 - p]) * self.valid * v[p] for p in range(2))

    def values(self, R1, R2, c, root=None):
        return self.expand(self.query(R1, R2, c, root), R1, R2, c)


# ------------------------------------------------------------------------- evaluation

def _eval_job(args):
    path, idx, model_path = args
    from .buckets import broadcast, bucket_targets
    from .dataset import load_shard

    net = PreRiverNet.load(model_path)
    z = load_shard(path)
    out = []
    for j in idx:
        board = z["board"][j]
        r = [z["r1"][j].astype(np.float64), z["r2"][j].astype(np.float64)]
        v = [z["v1"][j], z["v2"][j]]
        m = [z["m1"][j], z["m2"][j]]
        leaf = PreRiverLeaf(net)
        leaf.bind_board(board, float(z["spr"][j]))
        pred = leaf.query(r[0][None], r[1][None], np.zeros(1))
        row = {"source": str(z["source"][j]), "spr": float(z["spr"][j])}
        for name in ("net", "floor", "checkdown"):
            row[name] = [0.0, 0.0]
        for p in range(2):
            ok = m[p] > 1e-12
            w = r[p] * ok
            floor = broadcast(bucket_targets(v[p], m[p], leaf.bmap, leaf.K), leaf.bmap)
            eq = np.asarray(_native.solver_range_equity(board.tolist(),
                                                        z["r" + "21"[p]][j], 1, 0))
            for name, est in (("net", pred[p][0]), ("floor", floor), ("checkdown", eq - 0.5)):
                err = np.abs(est - v[p])
                row[name][0] += float((err * w).sum() / w.sum()) / 2
                row[name][1] += float(err[ok].mean()) / 2
        row["root"] = abs(float((r[0] * m[0] * (pred[0][0] - v[0])).sum()))
        out.append(row)
    return out


def evaluate(model_path, data_dir, n: int = 3000, workers: int | None = None) -> dict:
    """Test-board errors of a pre-river model against its teacher targets."""
    import multiprocessing as mp

    from .dataset import board_split, load_shard, shard_paths

    tasks, total = [], 0
    for path in shard_paths(data_dir):
        z = load_shard(path, ["board"])
        idx = [j for j, b in enumerate(z["board"]) if board_split(b) == "test"]
        idx = idx[: max(0, n - total)]
        if idx:
            tasks += [(path, idx[k:k + 50], str(model_path)) for k in range(0, len(idx), 50)]
            total += len(idx)
        if total >= n:
            break
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = [r for part in pool.map(_eval_job, tasks) for r in part]
    agg = {name: [float(np.mean([r[name][0] for r in rows])),
                  float(np.mean([r[name][1] for r in rows]))]
           for name in ("net", "floor", "checkdown")}
    agg["root"] = float(np.mean([r["root"] for r in rows]))
    agg["n"] = len(rows)
    agg["by_source"] = {s: float(np.mean([r["net"][0] for r in rows if r["source"] == s]))
                        for s in sorted({r["source"] for r in rows})}
    return agg


def main(argv=None) -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Evaluate a pre-river network on test boards.")
    ap.add_argument("--model", default="river_net/models/preriver_k256.pt")
    ap.add_argument("--data", default="data/preriver_v1")
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args(argv)
    res = evaluate(a.model, a.data, a.n)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
