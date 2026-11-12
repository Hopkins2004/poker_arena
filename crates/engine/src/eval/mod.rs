//! Fast hand evaluation.
//!
//! [`Evaluator`] scores 5–7 card Hold'em hands and Omaha hands (exactly two hole cards plus
//! three board cards) as a [`Strength`]: a `u16` where higher is better, encoded as
//! `(category + 1) << 12 | rank within category`, so [`category`] is a shift away.
//!
//! The hot path is branch-light and allocation-free: an incremental [`HandAcc`] sums
//! per-card keys, then a single table lookup produces the strength. See [`tables`] for the
//! scheme.

pub mod omaha;
pub mod reference;
pub mod tables;

use crate::card::Card;
pub use reference::HandCategory;
use tables::{Tables, CARD_KEYS, CARD_MASKS, FLUSH_FLAGS, SUIT_BIAS, SUIT_SHIFT};

/// Comparable hand strength; higher is better. Never zero for a valid hand.
pub type Strength = u16;

/// Category of a strength.
#[inline]
pub fn category(s: Strength) -> HandCategory {
    tables::category(s)
}

/// Incremental hand accumulator: add cards one at a time (or combine partial hands), then
/// evaluate. Combining requires the two parts to be disjoint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HandAcc {
    key: u64,
    mask: u64,
}

impl HandAcc {
    /// Empty accumulator.
    pub const EMPTY: HandAcc = HandAcc { key: 0, mask: 0 };

    /// Returns `self` plus one card.
    #[inline(always)]
    pub fn with(self, c: Card) -> HandAcc {
        HandAcc {
            key: self.key + CARD_KEYS[c.index()],
            mask: self.mask | CARD_MASKS[c.index()],
        }
    }

    /// Returns the union of two disjoint partial hands.
    #[inline(always)]
    pub fn combine(self, o: HandAcc) -> HandAcc {
        HandAcc {
            key: self.key + o.key,
            mask: self.mask | o.mask,
        }
    }

    /// Accumulator of a card slice.
    #[inline]
    pub fn from_cards(cards: &[Card]) -> HandAcc {
        cards.iter().fold(HandAcc::EMPTY, |a, &c| a.with(c))
    }

    /// Number of cards accumulated.
    #[inline]
    pub fn len(self) -> u32 {
        self.mask.count_ones()
    }

    /// True if no cards were added.
    #[inline]
    pub fn is_empty(self) -> bool {
        self.mask == 0
    }
}

/// Handle to the global lookup tables. Cheap to copy; obtain once with [`Evaluator::get`]
/// and reuse in hot loops to avoid repeated `OnceLock` checks.
#[derive(Clone, Copy)]
pub struct Evaluator {
    t: &'static Tables,
}

impl std::fmt::Debug for Evaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Evaluator")
    }
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::get()
    }
}

impl Evaluator {
    /// Returns the evaluator, building tables on first call.
    #[inline]
    pub fn get() -> Evaluator {
        Evaluator {
            t: tables::tables(),
        }
    }

    /// Evaluates an accumulated hand of 5–7 cards.
    #[inline(always)]
    pub fn eval_acc(self, h: HandAcc) -> Strength {
        let k = h.key + SUIT_BIAS;
        let f = k & FLUSH_FLAGS;
        if f != 0 {
            let suit = (f.trailing_zeros() - SUIT_SHIFT - 3) / 4;
            self.t.flush_table[((h.mask >> (16 * suit)) & 0x1fff) as usize]
        } else {
            self.t.rank_table[self.t.hash(k as u32)]
        }
    }

    /// Evaluates exactly seven cards.
    #[inline(always)]
    pub fn eval7(self, c: &[Card; 7]) -> Strength {
        let h = HandAcc::EMPTY
            .with(c[0])
            .with(c[1])
            .with(c[2])
            .with(c[3])
            .with(c[4])
            .with(c[5])
            .with(c[6]);
        self.eval_acc(h)
    }

    /// Evaluates 5, 6 or 7 cards.
    #[inline]
    pub fn eval(self, cards: &[Card]) -> Strength {
        debug_assert!((5..=7).contains(&cards.len()));
        self.eval_acc(HandAcc::from_cards(cards))
    }

    /// Evaluates an Omaha hand: the best hand using exactly two of `hole` (2–6 cards) and
    /// exactly three of the five `board` cards.
    #[inline]
    pub fn eval_omaha(self, hole: &[Card], board: &[Card; 5]) -> Strength {
        omaha::eval_omaha(self, hole, board)
    }
}

/// A rules-specific way to score a player's hand at showdown. Implemented by zero-sized
/// types so the state machine is monomorphised (no dynamic dispatch in evaluation).
pub trait HandEvaluator: Copy {
    /// Strength of `hole` with the full five-card `board`.
    fn strength(ev: Evaluator, hole: &[Card], board: &[Card; 5]) -> Strength;
}

/// Hold'em: best five of the hole cards plus board.
#[derive(Clone, Copy, Debug, Default)]
pub struct HoldemEval;
impl HandEvaluator for HoldemEval {
    #[inline]
    fn strength(ev: Evaluator, hole: &[Card], board: &[Card; 5]) -> Strength {
        let b = HandAcc::from_cards(board);
        ev.eval_acc(HandAcc::from_cards(hole).combine(b))
    }
}

/// Omaha: exactly two hole cards and three board cards.
#[derive(Clone, Copy, Debug, Default)]
pub struct OmahaEval;
impl HandEvaluator for OmahaEval {
    #[inline]
    fn strength(ev: Evaluator, hole: &[Card], board: &[Card; 5]) -> Strength {
        ev.eval_omaha(hole, board)
    }
}

/// Convenience: evaluates 5–7 cards with the global evaluator.
pub fn eval(cards: &[Card]) -> Strength {
    Evaluator::get().eval(cards)
}

#[cfg(test)]
mod tests;
