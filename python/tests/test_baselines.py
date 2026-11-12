from pathlib import Path

import pokerarena as pa
from pokerarena.baselines import RandomLegal, chen_score

CONFIG = (Path(__file__).parent.parent.parent / "configs" / "trio.toml").read_text()


def test_chen_matches_rust():
    c = lambda s: pa.parse_cards(s)[0]
    assert chen_score(c("As"), c("Ah")) == 20.0
    assert chen_score(c("Ks"), c("Kh")) == 16.0
    assert chen_score(c("As"), c("Ks")) == 12.0
    assert chen_score(c("2s"), c("2h")) == 5.0
    assert chen_score(c("7s"), c("2h")) == -1.0
    assert chen_score(c("Ts"), c("9s")) == 8.0


def test_ports_play_legally_and_match_style():
    cfg = CONFIG.replace("hands_per_table = 1000", "hands_per_table = 300")
    res = pa.run(cfg, save=False, quiet=True, rounds=1, processes=2)
    rows = {r["key"]: r for r in res.rows}
    assert all(r["illegal"] == 0 for r in rows.values())
    assert rows["station"]["pfr"] == 0.0 and rows["station"]["vpip"] > 0.85
    assert rows["tag_loose"]["rating"] > rows["station"]["rating"]


def test_random_legal_accepts_rust_param_names():
    b = RandomLegal(seed=1, **{"fold": 2.0, "call": 3.0, "raise": 0.2, "raise_sizes": [0.5]})
    assert b.raise_w == 0.2
