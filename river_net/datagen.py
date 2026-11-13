"""Training-data generation: ``python -m river_net.datagen --n 100000 --workers 12 --out data/v1``.

Each sample index ``i`` is generated (``RangeGenerator(seed).generate(i)``) and solved
(``solver.solve``) deterministically, so a dataset is reproducible from (seed, config) and a
run can be resumed or extended: shards already listed in ``manifest.json`` are skipped.

``--kind preriver --teacher river_net/models/river_k64.pt`` writes pre-river network data
instead (:mod:`river_net.preriver`): turn-end spots (``generate(i, street="turn_end")``,
4-card boards) labelled by the river network averaged over all river cards; no solves,
``expl`` and ``iters`` are 0, ``m1, m2`` are turn-board masses, ``spr`` is the river SPR.

Shard ``k`` holds samples ``[k * shard_size, (k + 1) * shard_size)`` as ``shard_00000.npz``
with arrays (float32 unless noted): ``index`` int64, ``board`` uint8[5], ``pot``, ``stack``,
``spr``, ``r1, r2, v1, v2, m1, m2`` [1326], ``expl``, ``iters`` int16, ``source``,
``scenario``, ``line`` (str). Values follow the convention in :mod:`river_net.solver`.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .ranges import GeneratorConfig, RangeGenerator
from .solver import solve
from .tree import DEFAULT_ABSTRACTION

FORMAT = "river_net.data/1"
CHUNK = 50  # samples per worker task
FIELDS_1326 = ("r1", "r2", "v1", "v2", "m1", "m2")

_gen: RangeGenerator | None = None
_solve_cfg: dict = {}
_teacher = None


def _init(seed: int, gen_cfg: dict, solve_cfg: dict) -> None:
    global _gen, _solve_cfg, _teacher
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)  # the parent handles Ctrl-C
    _gen = RangeGenerator(seed, GeneratorConfig(**gen_cfg))
    _solve_cfg = solve_cfg
    if solve_cfg.get("kind") == "preriver":
        from .model import RiverValueNet
        from .preriver import Teacher

        _teacher = Teacher(RiverValueNet.load(solve_cfg["teacher"]), threads=1)


class _Labels:
    def __init__(self, v1, v2, m1, m2):
        self.v1, self.v2, self.m1, self.m2 = v1, v2, m1, m2
        self.expl, self.iters = 0.0, 0


def _work(indices: list) -> dict:
    out: dict = {k: [] for k in ("index", "board", "pot", "stack", "spr", *FIELDS_1326,
                                 "expl", "iters", "source", "scenario", "line", "seconds")}
    for i in indices:
        t0 = time.perf_counter()
        if _teacher is not None:
            spot = _gen.generate(i, street="turn_end")
            res = _Labels(*_teacher(spot.board, spot.spr, spot.r1, spot.r2))
        else:
            spot = _gen.generate(i)
            res = solve(spot, iters=_solve_cfg["max_iters"],
                        target_expl=_solve_cfg["target_expl"],
                        check_every=_solve_cfg["check_every"],
                        abstraction=_solve_cfg["abstraction"])
        out["index"].append(i)
        out["board"].append(spot.board)
        out["pot"].append(spot.pot)
        out["stack"].append(spot.stack)
        out["spr"].append(spot.spr)
        for k, v in (("r1", spot.r1), ("r2", spot.r2), ("v1", res.v1), ("v2", res.v2),
                     ("m1", res.m1), ("m2", res.m2)):
            out[k].append(v)
        out["expl"].append(res.expl)
        out["iters"].append(res.iters)
        out["source"].append(spot.meta["source"])
        out["scenario"].append(spot.meta["scenario"])
        out["line"].append(spot.meta["line"])
        out["seconds"].append(time.perf_counter() - t0)
    return out


def _to_arrays(parts: list) -> dict:
    cat = {k: [x for p in parts for x in p[k]] for k in parts[0]}
    order = np.argsort(cat["index"])
    arr = {
        "index": np.asarray(cat["index"], dtype=np.int64),
        "board": np.asarray(cat["board"], dtype=np.uint8),
        "pot": np.asarray(cat["pot"], dtype=np.float32),
        "stack": np.asarray(cat["stack"], dtype=np.float32),
        "spr": np.asarray(cat["spr"], dtype=np.float32),
        "expl": np.asarray(cat["expl"], dtype=np.float32),
        "iters": np.asarray(cat["iters"], dtype=np.int16),
        "source": np.asarray(cat["source"]),
        "scenario": np.asarray(cat["scenario"]),
        "line": np.asarray(cat["line"]),
        "seconds": np.asarray(cat["seconds"], dtype=np.float32),
    }
    for k in FIELDS_1326:
        arr[k] = np.asarray(cat[k], dtype=np.float32)
    return {k: v[order] for k, v in arr.items()}


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _config(a) -> dict:
    if a.kind == "preriver":
        return {
            "format": FORMAT,
            "kind": "preriver",
            "seed": a.seed,
            "shard_size": a.shard_size,
            "generator": asdict(GeneratorConfig()),
            "street": "turn_end",
            "solver": {"kind": "preriver", "teacher": str(a.teacher)},
            "value_convention": "river-pot units; v = river-network values averaged over the "
                                "river cards, weighted by compatible opponent mass (see "
                                "river_net/preriver.py); m = turn-board compatible mass",
        }
    return {
        "format": FORMAT,
        "seed": a.seed,
        "shard_size": a.shard_size,
        "generator": asdict(GeneratorConfig()),
        "solver": {"max_iters": a.max_iters, "target_expl": a.target_expl,
                   "check_every": a.check_every, "abstraction": DEFAULT_ABSTRACTION,
                   "algorithm": "DCFR(1.5, 0, 2), alternating updates"},
        "value_convention": "pot units of the starting river pot; u1 = (collected - put in "
                            "on the river)/P - 1/2, u2 = -u1; v = conditional per-hand value "
                            "under the average strategy (see river_net/solver.py)",
    }


class Stats:
    def __init__(self):
        self.expl: list = []
        self.iters: list = []
        self.seconds: list = []
        self.t0 = time.time()
        self.done = 0

    def add(self, part: dict, max_iters: int, target: float):
        self.expl += part["expl"]
        self.iters += part["iters"]
        self.seconds += part["seconds"]
        self.done += len(part["index"])

    def line(self, total: int, max_iters: int, target: float) -> str:
        dt = time.time() - self.t0
        e = np.asarray(self.expl) * 100
        it = np.asarray(self.iters)
        capped = float(np.mean((it >= max_iters) & (np.asarray(self.expl) >= target)))
        return (f"{self.done}/{total} samples, {self.done / dt * 3600:,.0f}/hour, "
                f"expl median {np.median(e):.3f}% p95 {np.percentile(e, 95):.3f}% of pot, "
                f"iters median {np.median(it):.0f}, hit max_iters {100 * capped:.1f}%, "
                f"per-sample {1e3 * np.median(self.seconds):.0f} ms")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, required=True, help="total samples (resume/extend ok)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--max-iters", type=int, default=500)
    ap.add_argument("--target-expl", type=float, default=0.002)
    ap.add_argument("--check-every", type=int, default=50)
    ap.add_argument("--log-every", type=float, default=30.0, help="seconds")
    ap.add_argument("--kind", choices=("river", "preriver"), default="river")
    ap.add_argument("--teacher", type=Path, default=Path("river_net/models/river_k64.pt"),
                    help="river network labelling --kind preriver data")
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    mpath = a.out / "manifest.json"
    cfg = _config(a)
    if mpath.exists():
        manifest = json.loads(mpath.read_text())
        if manifest["config"] != json.loads(json.dumps(cfg)):
            print(f"{mpath}: config differs from this run; use another --out", file=sys.stderr)
            return 2
    else:
        manifest = {"config": cfg, "shards": []}
    n_shards = -(-a.n // a.shard_size)

    def full(k):  # samples shard k holds when the dataset has a.n samples
        return min(a.shard_size, a.n - k * a.shard_size)

    # A shard counts as present only if complete for this --n (a partial last shard from a
    # smaller run is regenerated when the dataset is extended).
    have = {s["shard"] for s in manifest["shards"]
            if (a.out / s["file"]).exists() and s["shard"] < n_shards
            and s["count"] == full(s["shard"])}
    manifest["shards"] = [s for s in manifest["shards"] if s["shard"] in have]
    todo = [k for k in range(n_shards) if k not in have]
    total = sum(full(k) for k in todo)
    print(f"{a.out}: {len(have)} shards present, {len(todo)} to generate ({total} samples) "
          f"on {a.workers} workers", file=sys.stderr)
    if not todo:
        return 0

    tasks, owner = [], {}
    for k in todo:
        lo, hi = k * a.shard_size, min((k + 1) * a.shard_size, a.n)
        for s in range(lo, hi, CHUNK):
            tasks.append(list(range(s, min(s + CHUNK, hi))))
            owner[s] = k
    pending = {k: [] for k in todo}
    need = {k: full(k) for k in todo}
    stats = Stats()
    last = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers, initializer=_init,
                  initargs=(a.seed, cfg["generator"], cfg["solver"])) as pool:
        try:
            for part in pool.imap_unordered(_work, tasks):
                k = owner[part["index"][0]]
                pending[k].append(part)
                stats.add(part, a.max_iters, a.target_expl)
                if sum(len(p["index"]) for p in pending[k]) == need[k]:
                    arr = _to_arrays(pending.pop(k))
                    name = f"shard_{k:05d}.npz"
                    tmp = a.out / (name + ".tmp.npz")
                    np.savez(tmp, **arr)
                    tmp.replace(a.out / name)
                    manifest["shards"].append({
                        "shard": k, "file": name, "start": int(arr["index"][0]),
                        "count": int(len(arr["index"])),
                        "expl_median": float(np.median(arr["expl"])),
                        "seconds": float(arr["seconds"].sum()),
                    })
                    manifest["shards"].sort(key=lambda s: s["shard"])
                    manifest["n"] = sum(s["count"] for s in manifest["shards"])
                    _write_atomic(mpath, json.dumps(manifest, indent=1))
                if time.time() - last > a.log_every:
                    print(stats.line(total, a.max_iters, a.target_expl), file=sys.stderr)
                    last = time.time()
        except KeyboardInterrupt:
            pool.terminate()
            print("interrupted; completed shards are kept (rerun to resume)", file=sys.stderr)
            return 130
    print("done: " + stats.line(total, a.max_iters, a.target_expl), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
