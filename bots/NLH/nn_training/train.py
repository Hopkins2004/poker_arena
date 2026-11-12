"""Training for bots/NLH/neural_net.py. Run from the repository root:

    python/.venv/bin/python -m bots.NLH.nn_training.train bc   [--hands 300000]
    python/.venv/bin/python -m bots.NLH.nn_training.train es   [--gens 40 --pop 24]
    python/.venv/bin/python -m bots.NLH.nn_training.train eval [--hands 60000]

* ``bc``  behaviour cloning: plays the anchored TAG bot (as a recording "teacher") against
  random and station, records (features, abstract action) for each TAG decision, and fits
  the network with cross-entropy (numpy + Adam). Fast: a TAG-like policy in ~1 minute.
* ``es``  evolution strategies: perturbs the weights, scores each perturbation by bb/hand
  against the three anchored bots (duplicate 6-max, common random numbers per generation)
  and steps along the rank-weighted perturbations. Starts from the current weights, so run
  ``bc`` first. Optimises exactly what the arena measures; no gradients needed.
* ``eval`` a quick arena run (nn + the anchored bots, nothing saved) printing the board.

Weights are written to bots/NLH/nn_training/weights.npz, which the bot loads.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

from bots.NLH import neural_net as nn
from bots.NLH.nn_training import workers

DATA_DIR = Path(__file__).resolve().parent / "data"


def _pool(processes):
    return mp.get_context("spawn").Pool(processes or os.cpu_count())


# ----------------------------------------------------------------------- behaviour cloning

def collect(hands: int, seed: int, processes=None, hands_per_table=500, tables_per_task=3):
    per_task = hands_per_table * tables_per_task
    tasks = [(seed * 1_000_003 + i, hands_per_table, tables_per_task)
             for i in range(max(1, -(-hands // per_task)))]
    t0 = time.time()
    with _pool(processes) as pool:
        parts = pool.map(workers.collect, tasks)
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    print(f"collected {len(y):,} teacher decisions from {len(tasks) * per_task:,} hands "
          f"in {time.time() - t0:.1f}s", file=sys.stderr)
    return x, y


def _targets_mask(y):
    """Allowed classes per sample: one class, or {0, 1} for a free check."""
    mask = np.zeros((len(y), nn.N_OUT))
    single = y < workers.CHECK
    mask[single, y[single]] = 1.0
    mask[~single, 0] = mask[~single, 1] = 1.0
    return mask


def _grad(layers, x, mask):
    """Loss -log(sum of allowed-class probabilities) and its gradients."""
    hs = [x]
    h = x
    for w, b in layers[:-1]:
        h = np.tanh(h @ w + b)
        hs.append(h)
    w, b = layers[-1]
    z = h @ w + b
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    allowed = (p * mask).sum(axis=1)
    loss = -np.log(np.maximum(allowed, 1e-12)).mean()
    target = p * mask / np.maximum(allowed, 1e-12)[:, None]
    dz = (p - target) / len(x)
    grads = []
    for i in range(len(layers) - 1, -1, -1):
        w, _ = layers[i]
        grads.append((hs[i].T @ dz, dz.sum(axis=0)))
        if i:
            dz = (dz @ w.T) * (1.0 - hs[i] ** 2)
    return loss, grads[::-1], p


def _accuracy(p, mask):
    return float(mask[np.arange(len(p)), p.argmax(axis=1)].mean())


def fit(x, y, *, epochs=40, batch=512, lr=3e-3, seed=0, layers=None):
    rng = np.random.default_rng(seed)
    x = x.astype(np.float64)
    mask = _targets_mask(y)
    idx = rng.permutation(len(y))
    n_val = max(len(y) // 10, 1)
    val, tr = idx[:n_val], idx[n_val:]
    layers = [(w.copy(), b.copy()) for w, b in (layers or nn.init_layers(seed))]
    m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in layers]
    v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in layers]
    b1, b2, eps, t = 0.9, 0.999, 1e-8, 0
    for ep in range(epochs):
        rng.shuffle(tr)
        for s in range(0, len(tr), batch):
            j = tr[s:s + batch]
            _, grads, _ = _grad(layers, x[j], mask[j])
            t += 1
            for k, ((w, b), (gw, gb)) in enumerate(zip(layers, grads)):
                mw, mb = m[k]
                vw, vb = v[k]
                mw[:] = b1 * mw + (1 - b1) * gw
                mb[:] = b1 * mb + (1 - b1) * gb
                vw[:] = b2 * vw + (1 - b2) * gw * gw
                vb[:] = b2 * vb + (1 - b2) * gb * gb
                c1, c2 = 1 - b1 ** t, 1 - b2 ** t
                w -= lr * (mw / c1) / (np.sqrt(vw / c2) + eps)
                b -= lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
        if ep % 5 == 4 or ep == epochs - 1:
            vl, _, vp = _grad(layers, x[val], mask[val])
            print(f"epoch {ep + 1:3d}: val loss {vl:.4f}  val acc {_accuracy(vp, mask[val]):.3f}",
                  file=sys.stderr)
    return layers


def cmd_bc(a):
    data = DATA_DIR / f"teacher_{a.hands}_{a.seed}.npz"
    if data.exists() and not a.recollect:
        with np.load(data) as z:
            x, y = z["x"], z["y"]
        print(f"loaded {len(y):,} decisions from {data}", file=sys.stderr)
    else:
        x, y = collect(a.hands, a.seed, a.processes)
        DATA_DIR.mkdir(exist_ok=True)
        np.savez_compressed(data, x=x, y=y)
    counts = np.bincount(y, minlength=4)
    print("labels: fold {} call {} raise {} check {}".format(*counts), file=sys.stderr)
    t0 = time.time()
    layers = fit(x, y, epochs=a.epochs, lr=a.lr, seed=a.seed)
    print(f"trained in {time.time() - t0:.1f}s", file=sys.stderr)
    nn.save_weights(layers, method="bc", decisions=len(y))
    print(f"wrote {nn.WEIGHTS_FILE}", file=sys.stderr)


# ----------------------------------------------------------------------- evolution strategies

def _centered_ranks(f):
    r = np.empty(len(f))
    r[np.argsort(f)] = np.arange(len(f))
    return r / (len(f) - 1) - 0.5


def cmd_es(a):
    try:
        theta = nn.flatten(nn.load_weights())
        print("starting from the current weights", file=sys.stderr)
    except FileNotFoundError:
        theta = nn.flatten(nn.init_layers(a.seed))
        print("no weights yet: starting from a random init (run bc first for a better start)",
              file=sys.stderr)
    rng = np.random.default_rng(a.seed)
    m, v = np.zeros_like(theta), np.zeros_like(theta)
    hpt, nt = a.hands_per_table, a.tables
    with _pool(a.processes) as pool:
        def score(thetas, seed):
            return np.array(pool.map(workers.evaluate,
                                     [(t, seed, hpt, nt) for t in thetas]))

        val_seeds = [10_000_000 + i for i in range(a.val_tasks)]
        best = float(np.mean(pool.map(workers.evaluate,
                                      [(theta, s, hpt, nt) for s in val_seeds])))
        print(f"start: {best * 100:+.1f} bb/100 on the validation seeds", file=sys.stderr)
        best_theta = theta.copy()
        for g in range(a.gens):
            t0 = time.time()
            eps = rng.standard_normal((a.pop // 2, theta.size))
            cands = np.concatenate([theta + a.sigma * eps, theta - a.sigma * eps])
            f = score(cands, a.seed * 7919 + g)  # same cards for every candidate
            w = _centered_ranks(f)
            grad = ((w[:len(eps)] - w[len(eps):]) @ eps) / (len(eps) * a.sigma)
            grad -= a.l2 * theta
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad * grad
            theta = theta + a.lr * (m / (1 - 0.9 ** (g + 1))) / (
                np.sqrt(v / (1 - 0.999 ** (g + 1))) + 1e-8)
            line = (f"gen {g + 1:3d}: pop mean {f.mean() * 100:+7.1f} bb/100  "
                    f"best {f.max() * 100:+7.1f}  ({time.time() - t0:.1f}s)")
            if (g + 1) % a.val_every == 0 or g == a.gens - 1:
                val = float(np.mean(pool.map(workers.evaluate,
                                             [(theta, s, hpt, nt) for s in val_seeds])))
                line += f"  | validation {val * 100:+.1f} bb/100"
                if val > best:
                    best, best_theta = val, theta.copy()
                    nn.save_weights(nn.unflatten(best_theta), method="es",
                                    validation_bb_per_100=best * 100)
                    line += "  (saved)"
            print(line, file=sys.stderr)
    print(f"best validation: {best * 100:+.1f} bb/100; weights in {nn.WEIGHTS_FILE}",
          file=sys.stderr)


# ----------------------------------------------------------------------- arena check

def cmd_eval(a):
    import pokerarena

    hands_per_table = 1200
    rounds = max(1, a.hands // (hands_per_table * 6))
    cfg = {
        "run": {"rounds": rounds, "seed": a.seed},
        "format": {"variant": "nlhe", "max_seats": 6},
        "table": {"hands_per_table": hands_per_table, "starting_stack_bb": 100,
                  "blinds": {"sb": 1, "bb": 2}, "stack_policy": "reset_each_hand",
                  "duplicate_mode": True},
        "matchmaking": {"min_appearances": 4, "min_tables": 6},
        "bots": [{"name": "nn", "kind": "python:bots.NLH.neural_net.NeuralNet"},
                 *({"name": n, "kind": k,
                    "anchor": {"random": 850, "station": 400, "tag": 1300}[n]}
                   for n, k in workers.ANCHORS.items())],
    }
    t0 = time.time()
    res = pokerarena.run(cfg, processes=a.processes, save=False, quiet=True)
    print(res.table)
    print(f"{time.time() - t0:.1f}s", file=sys.stderr)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--processes", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)
    sub = p.add_subparsers(dest="cmd", required=True)

    bc = sub.add_parser("bc", help="behaviour cloning of the TAG bot")
    bc.add_argument("--hands", type=int, default=300_000)
    bc.add_argument("--epochs", type=int, default=40)
    bc.add_argument("--lr", type=float, default=3e-3)
    bc.add_argument("--recollect", action="store_true", help="ignore a cached dataset")
    bc.set_defaults(fn=cmd_bc)

    es = sub.add_parser("es", help="evolution strategies against the anchored bots")
    es.add_argument("--gens", type=int, default=40)
    es.add_argument("--pop", type=int, default=24, help="candidates per generation (even)")
    es.add_argument("--sigma", type=float, default=0.05)
    es.add_argument("--lr", type=float, default=0.02)
    es.add_argument("--l2", type=float, default=0.005)
    es.add_argument("--hands-per-table", type=int, default=300, help="multiple of 6")
    es.add_argument("--tables", type=int, default=4, help="tables per candidate")
    es.add_argument("--val-tasks", type=int, default=24)
    es.add_argument("--val-every", type=int, default=5)
    es.set_defaults(fn=cmd_es)

    ev = sub.add_parser("eval", help="quick arena run: nn vs the anchored bots")
    ev.add_argument("--hands", type=int, default=72_000)
    ev.set_defaults(fn=cmd_eval)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
