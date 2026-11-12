//! Omaha evaluation: the best hand using exactly two hole cards and exactly three board cards.
//!
//! Precomputes the ten board-triple accumulators and the hole-pair accumulators, then takes
//! the maximum over all pair × triple combinations (60 lookups for 4 hole cards).

use super::{Evaluator, HandAcc, Strength};
use crate::card::Card;

/// Index triples of the C(5,3) = 10 board subsets.
const TRIPLES: [[usize; 3]; 10] = [
    [0, 1, 2],
    [0, 1, 3],
    [0, 1, 4],
    [0, 2, 3],
    [0, 2, 4],
    [0, 3, 4],
    [1, 2, 3],
    [1, 2, 4],
    [1, 3, 4],
    [2, 3, 4],
];

/// Best Omaha strength. `hole` has 2–6 cards; `board` is the full board.
#[inline]
pub fn eval_omaha(ev: Evaluator, hole: &[Card], board: &[Card; 5]) -> Strength {
    debug_assert!((2..=6).contains(&hole.len()));
    let mut trip = [HandAcc::EMPTY; 10];
    for (t, idx) in trip.iter_mut().zip(TRIPLES.iter()) {
        *t = HandAcc::EMPTY
            .with(board[idx[0]])
            .with(board[idx[1]])
            .with(board[idx[2]]);
    }
    let mut best = 0;
    for a in 0..hole.len() {
        for b in a + 1..hole.len() {
            let pair = HandAcc::EMPTY.with(hole[a]).with(hole[b]);
            for t in &trip {
                best = best.max(ev.eval_acc(pair.combine(*t)));
            }
        }
    }
    best
}
