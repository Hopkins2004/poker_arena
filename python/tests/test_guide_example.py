"""Runs the BOTS.md Python example bot in an arena."""

import pokerarena as pa

CFG = """
[run]
rounds = 2
seed = 1
[format]
variant = "nlhe"
max_seats = 6
[table]
hands_per_table = 300
blinds = { sb = 1, bb = 2 }
[[bots]]
name = "pot_odds"
kind = "python:examples.pot_odds_bot.PotOdds"
params = { samples = 100 }
[[bots]]
name = "tag"
kind = "python:pokerarena.baselines.TightAggressive"
anchor = 1300
[[bots]]
name = "station"
kind = "python:pokerarena.baselines.CallingStation"
anchor = 400
"""


def test_guide_python_bot_runs_legally():
    res = pa.run(CFG, save=False, quiet=True, processes=2)
    rows = {r["key"]: r for r in res.rows}
    assert all(r["illegal"] == 0 and r["hands"] > 0 for r in rows.values())
    assert rows["pot_odds"]["bb_per_100"] > rows["station"]["bb_per_100"]
