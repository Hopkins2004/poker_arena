"""Train the river value network: ``python -m river_net.train --data data/v1 --K 64 --out runs/k64``.

Loss: Huber (δ = 0.1 pot) per bucket, weighted by ``λ + (1 − λ)·K·r̄_i(k)`` (λ = 0.5; empty
buckets weight 0), summed over both players' buckets and averaged over the batch. AdamW
(lr 1e-3, cosine decay to 0 over the planned steps, weight decay 1e-4), batch 512, early
stopping on validation loss. Writes ``config.json``, ``log.json`` and ``best.pt`` to
``--out``; ``--save-model`` also copies the best model there.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import arrays, bucketed
from .model import RiverNet, save

LAMBDA = 0.5
HUBER_DELTA = 0.1


def loss_fn(pred, y, rb, mask, K: int) -> torch.Tensor:
    w = (LAMBDA + (1 - LAMBDA) * K * rb) * mask
    h = torch.nn.functional.huber_loss(pred, y, delta=HUBER_DELTA, reduction="none")
    return (h * w).sum(dim=1).mean()


def _tensors(a):
    return [torch.from_numpy(np.ascontiguousarray(t, dtype=np.float32))
            for t in (a.x, a.y, a.w, a.rb, a.mask)]


@torch.no_grad()
def evaluate_loss(model, tensors, K, batch=4096) -> float:
    model.eval()
    x, y, w, rb, mask = tensors
    tot = 0.0
    for s in range(0, len(x), batch):
        sl = slice(s, s + batch)
        pred, _ = model(x[sl], w[sl])
        tot += float(loss_fn(pred, y[sl], rb[sl], mask[sl], K)) * len(x[sl])
    return tot / max(len(x), 1)


def train(data_dir, K=64, out=None, board_features=False, hidden=64, depth=3, epochs=60,
          batch=512, lr=1e-3, weight_decay=1e-4, patience=6, n_train=None, seed=0,
          threads=None, time_limit=None, quiet=False, data=None) -> dict:
    """Trains one model; returns a summary dict (and the model under "model")."""
    torch.manual_seed(seed)
    if threads:
        torch.set_num_threads(threads)
    data = data if data is not None else bucketed(data_dir, K)
    tr = arrays(data, "train", board_features, limit=n_train, seed=seed)
    va = arrays(data, "val", board_features)
    T, V = _tensors(tr), _tensors(va)
    model = RiverNet(K, hidden, depth, board_features)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(T[0]) / batch)
    total = epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad, log = math.inf, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(T[0]), generator=gen)
        run = 0.0
        for s in range(0, len(perm), batch):
            i = perm[s:s + batch]
            x, y, w, rb, mask = (t[i] for t in T)
            pred, _ = model(x, w)
            loss = loss_fn(pred, y, rb, mask, K)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            run += float(loss) * len(i)
        tl, vl = run / len(perm), evaluate_loss(model, V, K)
        log.append({"epoch": ep + 1, "train": tl, "val": vl, "seconds": time.time() - t0})
        if not quiet:
            print(f"epoch {ep + 1:3d}  train {tl:.5f}  val {vl:.5f}  ({time.time() - t0:.0f}s)",
                  file=sys.stderr)
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if time_limit and time.time() - t0 > time_limit:
            break
    model.load_state_dict(best_state)
    model.eval()
    cfg = {"data": str(data_dir), "K": K, "board_features": board_features, "hidden": hidden,
           "depth": depth, "epochs": epochs, "batch": batch, "lr": lr,
           "weight_decay": weight_decay, "patience": patience, "n_train": len(T[0]),
           "n_val": len(V[0]), "seed": seed, "loss": f"huber({HUBER_DELTA}) weighted "
           f"{LAMBDA} + {1 - LAMBDA}*K*rb"}
    summary = {"config": cfg, "best_val": best, "epochs_run": len(log),
               "seconds": time.time() - t0, "log": log, "model": model}
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(cfg, indent=1))
        (out / "log.json").write_text(json.dumps(log, indent=1))
        save(model, out / "best.pt", train_config=cfg, best_val=best)
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=Path("data/v1"))
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--board-features", action="store_true")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=None, help="limit training samples")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--time-limit", type=float, default=None, help="seconds")
    ap.add_argument("--save-model", type=Path, default=None, help="copy best.pt here")
    a = ap.parse_args(argv)
    s = train(a.data, a.K, a.out, a.board_features, a.hidden, a.depth, a.epochs, a.batch, a.lr,
              patience=a.patience, n_train=a.n_train, threads=a.threads,
              time_limit=a.time_limit)
    print(f"best val loss {s['best_val']:.5f} after {s['epochs_run']} epochs "
          f"({s['seconds']:.0f}s, {s['config']['n_train']} train samples)", file=sys.stderr)
    if a.save_model:
        a.save_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(a.out / "best.pt", a.save_model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
