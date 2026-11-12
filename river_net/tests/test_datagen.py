"""Data generation: shards, manifest, resume, determinism."""

import json

import numpy as np

from river_net import datagen
from river_net.ranges import RangeGenerator
from river_net.solver import solve


def test_datagen_shards_resume_and_determinism(tmp_path):
    out = tmp_path / "d"
    args = ["--out", str(out), "--shard-size", "20", "--workers", "2", "--seed", "3"]
    assert datagen.main(["--n", "40", *args]) == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["n"] == 40 and [s["shard"] for s in m["shards"]] == [0, 1]
    z = np.load(out / "shard_00001.npz")
    assert list(z["index"]) == list(range(20, 40))
    assert z["v1"].shape == (20, 1326) and z["r2"].dtype == np.float32
    # Extending only generates the missing shard; existing files are untouched.
    before = (out / "shard_00000.npz").stat().st_mtime_ns
    assert datagen.main(["--n", "50", *args]) == 0
    assert (out / "shard_00000.npz").stat().st_mtime_ns == before
    assert json.loads((out / "manifest.json").read_text())["n"] == 50
    # The partial last shard (40..49) is regenerated in full when extending to 60.
    assert json.loads((out / "manifest.json").read_text())["shards"][-1]["count"] == 10
    assert datagen.main(["--n", "60", *args]) == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["n"] == 60 and [s["count"] for s in m["shards"]] == [20, 20, 20]
    assert list(np.load(out / "shard_00002.npz")["index"]) == list(range(40, 60))
    # A sample equals a direct generate + solve of its index.
    spot = RangeGenerator(3).generate(25)
    res = solve(spot)
    k = list(z["index"]).index(25)
    assert np.array_equal(z["r1"][k], spot.r1) and np.array_equal(z["v1"][k], res.v1)
    assert z["source"][k] == spot.meta["source"] and z["expl"][k] == np.float32(res.expl)
    # A different config refuses to mix into the same directory.
    assert datagen.main(["--n", "60", *args[:-1], "4"]) == 2
