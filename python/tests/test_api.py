import numpy as np
import pytest

import pokerarena as pa
from examples.equity_bot import EquityBot
from examples.random_bot import RandomBot


def test_actions():
    assert pa.Action.raise_to(10) == pa.Action("raise_to", 10)
    assert pa.Action.call().kind == "call"
    assert repr(pa.Action.bet_to(5)) == "Action.bet_to(5)"
    with pytest.raises(ValueError):
        pa.Action("shove")


def test_cards_and_eval():
    assert pa.card_to_str(51) == "As"
    assert pa.parse_cards("As Kd") == [51, 45]
    assert pa.hand_category(pa.evaluate("AsKsQsJsTs")) == "StraightFlush"
    assert pa.evaluate("AsAd2c3d4h") < pa.evaluate("AsAdAc3d4h")
    assert pa.evaluate([51, 47, 43, 39, 35, 0, 1]) == pa.evaluate("AsKsQsJsTs2c2d")


def test_equity():
    aa = pa.equity("AsAh", [], 1, 20000, seed=1)
    assert abs(aa - 0.852) < 0.01
    assert pa.equity(np.array([51, 50], dtype=np.uint8), "", 1, 500, seed=7) == \
        pa.equity([51, 50], "", 1, 500, seed=7)
    assert pa.equity("AsKs", "QsJsTs", 3, 100) == 1.0
    plo = pa.equity("AsAhKsKh", "", 1, 4000, seed=2, variant="plo")
    assert 0.6 < plo < 0.75
    with pytest.raises(ValueError):
        pa.equity("AsAs", "", 1, 10)
    with pytest.raises(ValueError):
        pa.equity("AsAh", "", 30, 10)


def test_protocol():
    assert isinstance(RandomBot(seed=1), pa.Bot)
    assert isinstance(EquityBot(seed=1), pa.Bot)
    assert not isinstance(object(), pa.Bot)
