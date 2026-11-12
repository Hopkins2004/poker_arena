"""Cards and combos, shared with the engine and ``pokerarena.preflop``.

Cards are ints ``rank * 4 + suit`` (rank 0 = 2 ... 12 = A; suits c d h s). The 1326 two-card
combos have the fixed index ``hi * (hi - 1) / 2 + lo`` (``hi > lo``): :data:`HAND_CARDS`
is the same table as ``pokerarena.preflop.COMBOS``, and the Rust solver uses the same order.
"""

from __future__ import annotations

import numpy as np

from pokerarena import _native, card_to_str, parse_cards, preflop

N_COMBOS = 1326
HAND_CARDS: np.ndarray = preflop.COMBOS          # (1326, 2) uint8, (hi, lo)
HAND_CLASS: np.ndarray = preflop.COMBO_CLASS     # (1326,) 169-class index per combo
CARD_HANDS = [np.flatnonzero((HAND_CARDS == c).any(axis=1)) for c in range(52)]
"""Combo indices containing each card."""


def combo_index(a: int, b: int) -> int:
    hi, lo = (a, b) if a > b else (b, a)
    return hi * (hi - 1) // 2 + lo


def parse_board(board) -> np.ndarray:
    """Board as uint8 cards from a string ("AsKd7h2c2d"), or a sequence of ints."""
    if isinstance(board, str):
        board = parse_cards(board)
    b = np.asarray(board, dtype=np.uint8)
    if len(set(b.tolist())) != len(b) or (b >= 52).any():
        raise ValueError(f"bad board {board!r}")
    return b


def blocked(board) -> np.ndarray:
    """Mask of combos that use a board card."""
    b = parse_board(board)
    return np.isin(HAND_CARDS, b).any(axis=1)


def valid_mask(board) -> np.ndarray:
    """Mask of combos not blocked by the board."""
    return ~blocked(board)


def river_strengths(board) -> np.ndarray:
    """Engine strength of every combo on a river board (uint16, 0 where blocked)."""
    return _native.solver_river_strengths(parse_board(board).tolist())


def combo_str(i: int) -> str:
    a, b = HAND_CARDS[i]
    return card_to_str(int(a)) + card_to_str(int(b))


def board_str(board) -> str:
    return "".join(card_to_str(int(c)) for c in parse_board(board))


def random_board(rng: np.random.Generator, n: int = 5) -> np.ndarray:
    return rng.choice(52, size=n, replace=False).astype(np.uint8)
