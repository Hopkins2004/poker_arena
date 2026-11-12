"""Evaluate river value networks on the test boards and write ``results/river_net_poc.md``.

    python -m river_net.evaluate --data data/v1 --model river_net/models/river_k64.pt

Per-hand errors are in pot units, against the exact solver values ``v_i(h)``:

* **network**: the model's bucket values broadcast to hands;
* **bucketing floor**: ``broadcast(bucket_targets(v, m))`` — a perfect network at this K;
* **check-down**: showdown equity vs the compatible opponent range − ½ (a leaf evaluator that
  ignores river betting).

MAE unweighted (mean over valid hands) and range-weighted (``Σ r_i |err|``), both players
pooled; root value error ``|Σ r_i m_i (v̂_i − v_i)|``; zero-sum residual of the raw outputs
before projection (``|Σ w1 ŷ1 + Σ w2 ŷ2|``). Breakdowns by source, pot type, scenario, river
line (flop/turn line shapes) and SPR bin. ``--curves`` adds learning curves over training-set
size × K × board features, and inference speed is always measured.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pokerarena import _native

from .buckets import broadcast, bucket_map, bucket_targets
from .cards import river_strengths
from .dataset import board_split, bucketed, load_shard, shard_paths
from .model import RiverValueNet, load
from .train import train

SPR_BINS = [(0, 1, "<1"), (1, 3, "1-3"), (3, 8, "3-8"), (8, 1e9, ">8")]


# ------------------------------------------------------------------------- data

def load_split(data_dir, split="test") -> dict:
    keys = ("index", "board", "pot", "stack", "spr", "r1", "r2", "v1", "v2", "m1", "m2",
            "expl", "source", "scenario", "line")
    parts = []
    for p in shard_paths(data_dir):
        z = load_shard(p, keys)
        sel = np.array([board_split(b) == split for b in z["board"]])
        parts.append({k: v[sel] for k, v in z.items()})
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["strength"] = np.stack([river_strengths(b) for b in out["board"]])
    out["pot_type"] = np.array([{1: "srp", 2: "3bet", 3: "4bet"}[s.count("raise")]
                                for s in out["scenario"]])
    out["line_shape"] = np.array([re.sub(r"[0-9.]+", "", s) for s in out["line"]])
    out["spr_bin"] = np.array([next(n for lo, hi, n in SPR_BINS if lo <= s < hi)
                               for s in out["spr"]])
    return out


def checkdown(T: dict) -> tuple:
    """Showdown equity − ½ vs the compatible opponent range, per hand (0 where blocked)."""
    v = []
    for p, opp in (("1", "r2"), ("2", "r1")):
        rows = []
        for b, r in zip(T["board"], T[opp]):
            eq, mass = _native.solver_river_equity(b.tolist(), r.astype(np.float32))
            rows.append(np.where(mass > 0, eq - 0.5, 0.0))
        v.append(np.stack(rows))
    return v[0], v[1]


def floor(T: dict, K: int) -> tuple:
    bm = np.stack([bucket_map(s, K) for s in T["strength"]])
    return (broadcast(bucket_targets(T["v1"], T["m1"], bm, K), bm),
            broadcast(bucket_targets(T["v2"], T["m2"], bm, K), bm))


def predict(net: RiverValueNet, T: dict, chunk=2048):
    """Network values per hand [N,1326] x2 and the pre-projection zero-sum residual [N]."""
    V1, V2, res = [], [], []
    K = net.K
    for s in range(0, len(T["spr"]), chunk):
        sl = slice(s, s + chunk)
        x, w, bm = net.inputs(T["board"][sl], T["spr"][sl], T["r1"][sl], T["r2"][sl])
        with torch.no_grad():
            raw = net.model.raw(torch.from_numpy(x)).numpy()
        y = net.predict_buckets(x, w)
        res.append(np.abs((w * raw).sum(axis=1)))
        V1.append(broadcast(y[:, :K], bm))
        V2.append(broadcast(y[:, K:], bm))
    return np.concatenate(V1), np.concatenate(V2), np.concatenate(res)


# ------------------------------------------------------------------------- metrics

def per_sample(T: dict, V1, V2) -> dict:
    """Per-sample metrics (players pooled: mean of the two)."""
    valid = T["strength"] > 0
    out = {"mae": 0.0, "wmae": 0.0, "root": 0.0}
    for p, (V, v, r, m) in enumerate(((V1, T["v1"], T["r1"], T["m1"]),
                                      (V2, T["v2"], T["r2"], T["m2"]))):
        err = np.abs(V - v)
        out["mae"] = out["mae"] + (err * valid).sum(1) / valid.sum(1) / 2
        out["wmae"] = out["wmae"] + (err * r).sum(1) / r.sum(1) / 2
        out[f"root{p + 1}"] = np.abs((r * m * (V - v)).sum(1))
        out["root"] = out["root"] + out[f"root{p + 1}"] / 2
    return out


def table(T: dict, metrics: dict, group: str | None, names: list, extra=None) -> list:
    """Rows of mean metrics by group (None = overall)."""
    keys = [None] if group is None else sorted(set(T[group].tolist()), key=str)
    rows = []
    for k in keys:
        sel = np.ones(len(T["spr"]), bool) if k is None else T[group] == k
        row = {"group": "all" if k is None else str(k), "n": int(sel.sum())}
        for name in names:
            for m in ("mae", "wmae", "root", "root1", "root2"):
                row[f"{name}.{m}"] = float(metrics[name][m][sel].mean())
        if extra is not None:
            row["zs"] = float(extra[sel].mean())
        rows.append(row)
    return rows


def md_table(rows, names, metric_cols=("mae", "wmae"), zs=False, title_map=None) -> str:
    head = ["group", "n"] + [f"{title_map.get(n, n)} {c}" for n in names for c in metric_cols]
    if zs:
        head.append("zero-sum resid.")
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["group"], str(r["n"])] + [f"{r[f'{n}.{c}']:.4f}" for n in names
                                               for c in metric_cols]
        if zs:
            cells.append(f"{r['zs']:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------- speed

def inference_speed(net: RiverValueNet, T: dict) -> list:
    torch.set_num_threads(1)
    rows = []
    x_all, w_all, _ = net.inputs(T["board"][:1024], T["spr"][:1024], T["r1"][:1024],
                                 T["r2"][:1024])
    for B in (1, 64, 1024):
        x = torch.from_numpy(x_all[:B])
        w = torch.from_numpy(w_all[:B])
        reps = max(20, 20000 // B)
        with torch.no_grad():
            net.model(x, w)
            t0 = time.perf_counter()
            for _ in range(reps):
                net.model(x, w)
            dt = (time.perf_counter() - t0) / reps
        t0 = time.perf_counter()
        n_full = min(B, 64)
        net.predict_batch(T["board"][:n_full], T["spr"][:n_full], T["r1"][:n_full],
                          T["r2"][:n_full])
        full = (time.perf_counter() - t0) / n_full
        rows.append({"batch": B, "forward_ms": dt * 1e3, "spots_per_s": B / dt,
                     "full_predict_ms_per_spot": full * 1e3})
    torch.set_num_threads(torch.get_num_threads())
    return rows


# ------------------------------------------------------------------------- plots

def plots(out_dir: Path, T, V1, rows_curves, model_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    rng = np.random.default_rng(0)
    pick = rng.choice(len(T["spr"]), size=min(6, len(T["spr"])), replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, i in zip(axes.ravel(), pick):
        valid = T["strength"][i] > 0
        r = T["r1"][i][valid]
        sc = ax.scatter(T["v1"][i][valid], V1[i][valid], c=r / r.max(), s=6, cmap="viridis")
        lo = min(T["v1"][i][valid].min(), V1[i][valid].min())
        hi = max(T["v1"][i][valid].max(), V1[i][valid].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_title(f"{T['source'][i]} · SPR {T['spr'][i]:.1f} · {T['line'][i]}", fontsize=8)
        ax.set_xlabel("exact v1 (pot)", fontsize=8)
        ax.set_ylabel("network v1 (pot)", fontsize=8)
    fig.colorbar(sc, ax=axes, label="range weight (relative)")
    fig.suptitle(f"Predicted vs exact per-hand values, player 1, random test spots ({model_name})")
    files["scatter"] = out_dir / "scatter.png"
    fig.savefig(files["scatter"], dpi=110)
    plt.close(fig)
    if rows_curves:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for (K, bf), grp in _group_curves(rows_curves).items():
            n = [r["n_train"] for r in grp]
            style = "-" if not bf else "--"
            axes[0].plot(n, [r["mae"] for r in grp], style, marker="o",
                         label=f"K={K}{' +board' if bf else ''}")
            axes[1].plot(n, [r["wmae"] for r in grp], style, marker="o",
                         label=f"K={K}{' +board' if bf else ''}")
        for ax, t in zip(axes, ("MAE, unweighted (pot)", "MAE, range-weighted (pot)")):
            ax.set_xscale("log")
            ax.set_xlabel("training samples")
            ax.set_title(t)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        files["curves"] = out_dir / "learning_curves.png"
        fig.tight_layout()
        fig.savefig(files["curves"], dpi=110)
        plt.close(fig)
    return files


def _group_curves(rows):
    g = {}
    for r in rows:
        g.setdefault((r["K"], r["board_features"]), []).append(r)
    return {k: sorted(v, key=lambda r: r["n_train"]) for k, v in sorted(g.items())}


# ------------------------------------------------------------------------- curves

def learning_curves(data_dir, T, sizes, Ks, epochs, threads, hidden=256) -> list:
    rows = []
    for K in Ks:
        data = bucketed(data_dir, K)
        n_avail = int((data["split"] == "train").sum())
        fl = per_sample(T, *floor(T, K))
        ns = sorted({min(n, n_avail) for n in sizes})
        for bf in (False, True):
            for n in ns:
                s = train(data_dir, K=K, board_features=bf, hidden=hidden, epochs=epochs,
                          patience=4, n_train=n, threads=threads, quiet=True, data=data)
                V1, V2, _ = predict(RiverValueNet(s["model"]), T)
                m = per_sample(T, V1, V2)
                rows.append({"K": K, "board_features": bf, "n_train": n,
                             "mae": float(m["mae"].mean()), "wmae": float(m["wmae"].mean()),
                             "floor_mae": float(fl["mae"].mean()),
                             "floor_wmae": float(fl["wmae"].mean()),
                             "epochs": s["epochs_run"], "seconds": s["seconds"]})
                print(f"  curve K={K} board={bf} n={n}: mae {rows[-1]['mae']:.4f} "
                      f"wmae {rows[-1]['wmae']:.4f} ({s['seconds']:.0f}s)", file=sys.stderr)
    return rows


# ------------------------------------------------------------------------- report

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data", type=Path, default=Path("data/v1"))
    ap.add_argument("--model", type=Path, default=Path("river_net/models/river_k64.pt"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    ap.add_argument("--curves", action="store_true", help="learning curves and ablations")
    ap.add_argument("--curve-sizes", default="5000,20000,50000,100000")
    ap.add_argument("--curve-ks", default="64,256,512")
    ap.add_argument("--curve-epochs", type=int, default=25)
    ap.add_argument("--curve-hidden", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)

    t0 = time.time()
    T = load_split(a.data, "test")
    net = RiverValueNet(load(a.model))
    K = net.K
    print(f"test split: {len(T['spr'])} samples; model K={K}", file=sys.stderr)
    V1, V2, zs = predict(net, T)
    metrics = {"net": per_sample(T, V1, V2), "floor": per_sample(T, *floor(T, K)),
               "checkdown": per_sample(T, *checkdown(T))}
    names = ["net", "floor", "checkdown"]
    titles = {"net": "network", "floor": "floor", "checkdown": "check-down"}
    groups = {"overall": None, "source": "source", "pot type": "pot_type",
              "river line (flop/turn)": "line_shape", "SPR": "spr_bin", "scenario": "scenario"}
    tables = {g: table(T, metrics, col, names, zs) for g, col in groups.items()}
    speed = inference_speed(net, T)
    curves = []
    if a.curves:
        sizes = [int(x) for x in a.curve_sizes.split(",")]
        Ks = [int(x) for x in a.curve_ks.split(",")]
        curves = learning_curves(a.data, T, sizes, Ks, a.curve_epochs, a.threads,
                                 a.curve_hidden)
    img_dir = a.out / "river_net_poc"
    files = plots(img_dir, T, V1, curves, a.model.name)

    # Beats check-down everywhere?
    beats = all(r["net.mae"] < r["checkdown.mae"] and r["net.wmae"] < r["checkdown.wmae"]
                for rows in tables.values() for r in rows)
    ratio_u = tables["overall"][0]["net.mae"] / tables["overall"][0]["floor.mae"]
    ratio_w = tables["overall"][0]["net.wmae"] / tables["overall"][0]["floor.wmae"]
    summary = {"n_test": int(len(T["spr"])), "K": K, "model": str(a.model),
               "tables": tables, "speed": speed, "curves": curves,
               "beats_checkdown_everywhere": beats, "net_over_floor": [ratio_u, ratio_w],
               "curve_hidden": a.curve_hidden, "curve_epochs": a.curve_epochs,
               "seconds": time.time() - t0}
    (img_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    write_report(a, summary, files, titles)
    print(f"wrote {a.out / 'river_net_poc.md'} ({time.time() - t0:.0f}s)", file=sys.stderr)
    return 0


def write_report(a, S, files, titles):
    names = ["net", "floor", "checkdown"]
    o = S["tables"]["overall"][0]
    lines = [
        "# River value network PoC: results",
        "",
        f"Model `{S['model']}` (K = {S['K']}), evaluated on {S['n_test']:,} test spots whose "
        "boards never appear in training (split by board). All errors are in **pot units** "
        "(fractions of the starting river pot), per hand, against the exact solver's "
        "conditional values `v_i(h)`; both players pooled.",
        "",
        "* **network**: model bucket values broadcast to hands.",
        f"* **floor**: the bucketing floor, a perfect network at K = {S['K']}.",
        "* **check-down**: showdown equity vs the compatible opponent range − ½ (ignores "
        "river betting).",
        "* `mae`: mean over valid hands; `wmae`: weighted by the player's range; root: "
        "`|Σ r m (v̂ − v)|`; zero-sum resid.: `|Σ w1 ŷ1 + Σ w2 ŷ2|` before projection.",
        "",
        "## Overall",
        "",
        f"| | network | floor | check-down |",
        "|---|---|---|---|",
        f"| MAE (unweighted) | {o['net.mae']:.4f} | {o['floor.mae']:.4f} | "
        f"{o['checkdown.mae']:.4f} |",
        f"| MAE (range-weighted) | {o['net.wmae']:.4f} | {o['floor.wmae']:.4f} | "
        f"{o['checkdown.wmae']:.4f} |",
        f"| root value error, player 1 | {o['net.root1']:.4f} | {o['floor.root1']:.4f} | "
        f"{o['checkdown.root1']:.4f} |",
        f"| root value error, player 2 | {o['net.root2']:.4f} | {o['floor.root2']:.4f} | "
        f"{o['checkdown.root2']:.4f} |",
        f"| zero-sum residual before projection | {o['zs']:.4f} | | |",
        "",
        "The raw (pre-projection) residual is not a training target: the projection "
        "subtracts a constant from all outputs, so the network is free to leave any offset "
        "in its raw outputs, and only projected values are ever used.",
        "",
        f"Network / floor: **{S['net_over_floor'][0]:.2f}×** (unweighted), "
        f"**{S['net_over_floor'][1]:.2f}×** (range-weighted). Network beats check-down on "
        f"every breakdown below: **{'yes' if S['beats_checkdown_everywhere'] else 'NO'}**.",
        "",
    ]
    for g in ("source", "pot type", "river line (flop/turn)", "SPR"):
        lines += [f"## By {g}", "", md_table(S["tables"][g], names, title_map=titles, zs=False),
                  ""]
    lines += ["## By preflop scenario", "",
              md_table(S["tables"]["scenario"], names, title_map=titles), ""]
    lines += ["## Inference speed (CPU, 1 thread)", "",
              "| batch | forward ms | spots/s (forward) | full predict ms/spot (bucketing + "
              "masses + forward) |", "|---|---|---|---|"]
    for r in S["speed"]:
        lines.append(f"| {r['batch']} | {r['forward_ms']:.3f} | {r['spots_per_s']:,.0f} | "
                     f"{r['full_predict_ms_per_spot']:.2f} |")
    lines += ["", "## Predicted vs exact", "",
              f"![scatter](river_net_poc/{files['scatter'].name})", ""]
    if S["curves"]:
        lines += ["## Learning curves and ablations", "",
                  "Test MAE (pot units) vs training-set size; each K with and without the "
                  "52-card board one-hot. `floor` is the bucketing floor at that K. Kept "
                  f"cheap: 3x{S.get('curve_hidden', 256)} MLPs, at most "
                  f"{S.get('curve_epochs', 25)} epochs each (the shipped model trains longer).",
                  "",
                  "| K | board | n train | MAE | wMAE | floor MAE | floor wMAE | epochs |",
                  "|---|---|---|---|---|---|---|---|"]
        for (K, bf), grp in _group_curves(S["curves"]).items():
            for r in grp:
                lines.append(f"| {K} | {'yes' if bf else 'no'} | {r['n_train']:,} | "
                             f"{r['mae']:.4f} | {r['wmae']:.4f} | {r['floor_mae']:.4f} | "
                             f"{r['floor_wmae']:.4f} | {r['epochs']} |")
        lines += ["", f"![learning curves](river_net_poc/{files['curves'].name})", ""]
    (a.out / "river_net_poc.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
