"""River value network: bucketed ranges + SPR in, per-bucket conditional values out.

Inputs: bucket masses ``r̄1, r̄2`` (K each), SPR, ``log(1 + SPR)``, optionally a 52-card board
one-hot. Outputs: ``v̂1, v̂2`` (K each, pot units), then a zero-sum projection: with bucket
weights ``w_i(k) = Σ_{h∈k} r_i(h) m_i(h)``,
``δ = (Σ w1·v̂1 + Σ w2·v̂2) / (Σ w1 + Σ w2)`` is subtracted from every output.

:class:`RiverValueNet` wraps a trained model with the interface a turn solver will call::

    net = RiverValueNet.load("river_net/models/river_k64.pt")
    v1, v2 = net.predict(board, pot, stack, r1, r2)        # 1326 conditional values each
    V1, V2 = net.predict_batch(boards, sprs, R1, R2)        # [B, 1326] each
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from pokerarena import _native

from .buckets import broadcast, bucket_map, to_buckets
from .cards import N_COMBOS, parse_board, river_strengths


class RiverNet(nn.Module):
    def __init__(self, K: int, hidden: int = 64, depth: int = 3, board_features: bool = False):
        super().__init__()
        self.config = {"K": K, "hidden": hidden, "depth": depth,
                       "board_features": board_features}
        d = 2 * K + 2 + (52 if board_features else 0)
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.GELU()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, 2 * K)

    def raw(self, x: torch.Tensor) -> torch.Tensor:
        """Outputs before the zero-sum projection, ``[B, 2K]`` (player 1 then player 2)."""
        return self.head(self.body(x))

    def forward(self, x: torch.Tensor, w: torch.Tensor):
        """Projected outputs ``[B, 2K]`` and the subtracted offset ``δ`` ``[B]``."""
        y = self.raw(x)
        delta = (w * y).sum(dim=1) / w.sum(dim=1).clamp_min(1e-12)
        return y - delta[:, None], delta


def save(model: RiverNet, path, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict(), **extra}, path)


def load(path) -> RiverNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = RiverNet(**ck["config"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def _norm(r: np.ndarray, valid: np.ndarray) -> np.ndarray:
    r = np.where(valid, np.asarray(r, dtype=np.float64), 0.0)
    s = r.sum(axis=-1, keepdims=True)
    return r / np.where(s > 0, s, 1.0)


class RiverValueNet:
    """A trained network plus bucketing: the leaf-evaluator interface."""

    def __init__(self, model: RiverNet):
        self.model = model.eval()
        self.K = model.config["K"]
        self.board_features = model.config["board_features"]

    @classmethod
    def load(cls, path) -> "RiverValueNet":
        return cls(load(path))

    def inputs(self, boards, sprs, R1, R2):
        """Features, projection weights and bucket maps for a batch."""
        boards = np.asarray(boards, dtype=np.uint8).reshape(-1, 5)
        B = len(boards)
        st = np.stack([river_strengths(b) for b in boards])
        valid = st > 0
        R1, R2 = _norm(np.reshape(R1, (B, N_COMBOS)), valid), _norm(np.reshape(R2, (B, N_COMBOS)), valid)
        bmap = np.stack([bucket_map(s, self.K) for s in st])
        M1 = np.stack([_native.solver_river_equity(b.tolist(), r.astype(np.float32))[1]
                       for b, r in zip(boards, R2)])
        M2 = np.stack([_native.solver_river_equity(b.tolist(), r.astype(np.float32))[1]
                       for b, r in zip(boards, R1)])
        spr = np.asarray(sprs, dtype=np.float32).reshape(B)
        cols = [to_buckets(R1, bmap, self.K), to_buckets(R2, bmap, self.K), spr[:, None],
                np.log1p(spr)[:, None]]
        if self.board_features:
            oh = np.zeros((B, 52), dtype=np.float32)
            np.put_along_axis(oh, boards.astype(np.int64), 1.0, axis=1)
            cols.append(oh)
        x = np.concatenate(cols, axis=1).astype(np.float32)
        w = np.concatenate([to_buckets(R1 * M1, bmap, self.K),
                            to_buckets(R2 * M2, bmap, self.K)], axis=1).astype(np.float32)
        return x, w, bmap

    @torch.no_grad()
    def predict_buckets(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        y, _ = self.model(torch.from_numpy(x), torch.from_numpy(w))
        return y.numpy()

    def predict_batch(self, boards, sprs, R1, R2):
        """Conditional values ``[B, 1326]`` for both players (0 on blocked combos)."""
        x, w, bmap = self.inputs(boards, sprs, R1, R2)
        y = self.predict_buckets(x, w)
        K = self.K
        return broadcast(y[:, :K], bmap), broadcast(y[:, K:], bmap)

    def predict(self, board, pot, stack, r1, r2):
        """``(v1[1326], v2[1326])`` for one river spot (pot and stack in any common unit)."""
        b = parse_board(board)
        v1, v2 = self.predict_batch(b[None], [float(stack) / float(pot)], r1, r2)
        return v1[0], v2[0]
