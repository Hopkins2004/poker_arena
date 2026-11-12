"""The preflop range database: hand classes, the tree, spots, and locating arena decisions."""

import json
import random
import subprocess
import sys

import numpy as np
import pytest

from pokerarena import BaseBot, Run, parse_cards, preflop
from pokerarena.catalog import REPO_ROOT

DB = preflop.load()


def pct(weights):
    return round(preflop.range_percent(weights), 1)


def test_hand_classes_and_combos():
    assert len(preflop.CLASSES) == len(set(preflop.CLASSES)) == 169
    assert preflop.CLASSES[:3] == ("AA", "AKs", "AQs") and preflop.CLASSES[13] == "AKo"
    assert preflop.class_name(*parse_cards("AsKs")) == "AKs"
    assert preflop.class_name(*parse_cards("Kd9c")) == "K9o"
    assert preflop.class_name(*parse_cards("7h7c")) == "77"
    for name in preflop.CLASSES:
        assert preflop.CLASSES[preflop.class_index(name)] == name
    counts = preflop.combo_counts()
    assert counts.sum() == 1326
    assert all(counts[i] == (6 if len(c) == 2 else 4 if c[2] == "s" else 12)
               for i, c in enumerate(preflop.CLASSES))
    for (hi, lo), c in zip(preflop.COMBOS, preflop.COMBO_CLASS):
        assert hi > lo and preflop.class_index(int(hi), int(lo)) == c


def test_database_is_up_to_date_with_the_source():
    r = subprocess.run([sys.executable, str(REPO_ROOT / "ranges" / "build_preflop.py"),
                        "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_published_range_sizes():
    # Figures from the source README (% of the 1326 combos).
    opens = {p: pct(DB.node("", p).range(1)) for p in ("UTG", "HJ", "CO", "BTN", "SB")}
    assert opens == {"UTG": 18.2, "HJ": 22.5, "CO": 27.8, "BTN": 42.5, "SB": 43.7}
    bb = DB.node("BTN raise 2.5bb", "BB")
    assert pct(bb.range(bb.action_index("call"))) == 37.6
    btn = DB.node("CO raise 2.5bb", "BTN")
    assert pct(btn.range(btn.action_index("call"))) == 5.6
    assert pct(btn.range(btn.action_index("raise")) + btn.range(btn.action_index("allin"))) \
        == 12.3


def test_strategies_and_tree_links():
    for n in DB.nodes:
        sums = n.strategy.sum(axis=0)
        reached = n.reach > 0
        assert np.allclose(sums[reached], 1.0) and np.all(sums[~reached] == 0)
        for a, child in enumerate(n.children):
            if child is not None:
                c = DB.nodes[child]
                assert c.path[:len(n.path) + 1] == (*n.path, a)
                assert all(x == 0 for x in c.path[len(n.path) + 1:])  # implied folds only
    assert DB.root.actor == "UTG" and DB.root.freqs("AA") == {"fold": 0.0, "raise 2bb": 1.0}
    assert DB.node("", "UTG").freqs("72o") == {"fold": 1.0, "raise 2bb": 0.0}
    assert DB.node("UTG raise 2bb, BB raise 11bb", "UTG").freqs("72o") is None  # never opened


def test_spots_carry_both_ranges():
    s = DB.spot("BTN raise 2.5bb, BB call")
    assert s.players == ("BB", "BTN") and s.pot_bb == 5.5 and s.stack_bb == 97.5
    assert not s.allin
    assert np.array_equal(s.ranges["BTN"], DB.node("", "BTN").range(1))
    bb = DB.node("BTN raise 2.5bb", "BB")
    assert np.array_equal(s.ranges["BB"], bb.range(bb.action_index("call")))
    assert len(DB.flop_spots()) == 38 and len(DB.spots) == 66
    shove = DB.spot("SB raise 3bb, BB raise 9bb, SB allin, BB allin")
    assert shove.allin and shove.pot_bb == 200 and shove.stack_bb == 0
    # Combo expansion with card removal.
    board = parse_cards("AsKsQs")
    w = s.combos("BTN", dead=board)
    assert w.shape == (1326,)
    blocked = np.isin(preflop.COMBOS, board).any(axis=1)
    assert np.all(w[blocked] == 0)
    assert np.array_equal(w[~blocked], preflop.expand(s.ranges["BTN"])[~blocked])


# --------------------------------------------------------------------- live arena check

_log: list = []


class ChartBot(BaseBot):
    """Plays the database preflop and check/folds after; logs every preflop decision."""

    def __init__(self, seed=0):
        super().__init__(seed)
        self.rng = random.Random(seed)

    def act(self, obs):
        if obs.street > 0:
            return "check" if obs.legal.can_check else "fold"
        d = DB.locate(obs)
        action = d.sample(obs, self.rng) if d is not None else None
        _log.append((d, obs.history[-1][2] if obs.history else None, action))
        if action is None:
            return "fold"
        return action


def test_locate_follows_a_live_table():
    _log.clear()
    cfg = {
        "run": {"rounds": 1, "seed": 3},
        "format": {"variant": "nlhe", "max_seats": 6},
        "table": {"hands_per_table": 3000, "starting_stack_bb": 100,
                  "blinds": {"sb": 1, "bb": 2}, "illegal_action_policy": "error"},
        "matchmaking": {"min_appearances": 1, "min_tables": 1},
        "bots": [{"name": "chart", "kind": "python:tests.test_preflop.ChartBot"}],
    }
    run = Run(json.dumps(cfg), seed=3, rounds=1, load_leaderboard=False)
    run.run_tables(0, list(range(len(run.schedule(0)))), threads=1)
    assert len(_log) > 5000
    assert all(d is not None for d, _, _ in _log), "every decision is in the tree"
    assert all(not d.approx for d, _, _ in _log), "sizes match the tree exactly"
    assert all(a is not None for _, _, a in _log), "the bot's hand is always in range"
    # The opening frequency at UTG matches the chart (18.2% of hands).
    utg = [a for d, _, a in _log if d.node is not None and d.node.path == ()]
    opened = sum(a.kind == "raise_to" for a in utg) / len(utg)
    assert abs(opened - 0.182) < 0.03, opened
    # Deep lines are reached too: 4-bets and calls of shoves.
    depths = {d.node.raises for d, _, _ in _log if d.node is not None}
    assert {0, 1, 2, 3} <= depths
    assert any(d.node is None for d, _, _ in _log), "implied folds are handled"


def test_locate_rejects_other_games():
    class Obs:
        variant, street, n_seats = "plo", 0, 6

    assert DB.locate(Obs()) is None
    Obs.variant, Obs.street = "nlhe", 1
    assert DB.locate(Obs()) is None


@pytest.mark.parametrize("bad", [{"schema": "x"}])
def test_rejects_unknown_schema(bad):
    with pytest.raises(ValueError):
        preflop.PreflopDB(bad)
