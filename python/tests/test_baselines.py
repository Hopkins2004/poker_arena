"""The anchored bots in bots/NLH and the catalog that discovers them."""

import pytest

import pokerarena as pa
from bots.NLH.random_legal import RandomLegal
from bots.NLH.tag import chen_score


def test_catalog_finds_the_anchored_bots():
    found = pa.discover()
    bots = {b.name: b for b in found if b.anchor is not None}
    assert set(bots) == {"random", "station", "tag"}  # other bots in the folder are unanchored
    assert (bots["random"].anchor, bots["station"].anchor, bots["tag"].anchor) == (850, 400, 1300)
    assert bots["tag"].kind == "python:bots.NLH.tag.TightAggressive"
    assert all(b.label and b.path.endswith(".py") for b in bots.values())
    assert pa.catalog.anchored_lineup() == [
        {"name": b.name, "kind": b.kind, "anchor": b.anchor} for b in found if b.anchor is not None]


def test_parameters_are_hardcoded():
    with pytest.raises(TypeError):
        RandomLegal(seed=1, fold=2.0)
    assert RandomLegal.RAISE_SIZES == (0.5, 1.0)


def test_chen_score():
    c = lambda s: pa.parse_cards(s)[0]
    assert chen_score(c("As"), c("Ah")) == 20.0
    assert chen_score(c("Ks"), c("Kh")) == 16.0
    assert chen_score(c("As"), c("Ks")) == 12.0
    assert chen_score(c("2s"), c("2h")) == 5.0
    assert chen_score(c("7s"), c("2h")) == -1.0
    assert chen_score(c("Ts"), c("9s")) == 8.0


def test_anchored_lineup_plays_legally():
    cfg = {
        "run": {"rounds": 1, "seed": 3},
        "format": {"variant": "nlhe", "max_seats": 6},
        "table": {"hands_per_table": 300, "blinds": {"sb": 1, "bb": 2}},
        "bots": pa.catalog.anchored_lineup(),
    }
    res = pa.run(cfg, save=False, quiet=True, processes=2)
    rows = {r["key"]: r for r in res.rows}
    assert all(r["illegal"] == 0 and r["anchored"] for r in rows.values())
    assert res.ratings == {"random": 850, "station": 400, "tag": 1300}
    assert rows["tag"]["bb_per_100"] > rows["station"]["bb_per_100"]
