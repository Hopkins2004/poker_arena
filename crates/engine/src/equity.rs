//! Monte Carlo equity against random opponent hands.

use crate::card::{Card, CardMask};
use crate::eval::{Evaluator, HandEvaluator};
use crate::format::Variant;
use crate::rng::below;
use rand_core::Rng;

/// Estimates the equity (win probability, ties split) of `hole` with the visible `board`
/// against `n_opponents` random hands, from `samples` random runouts, using the showdown
/// rules of `variant` (e.g. exactly two hole cards in PLO). Allocation-free.
///
/// Opponent hands are dealt `hole.len()` cards each, uniformly from the unseen cards.
pub fn equity<R: Rng>(
    ev: Evaluator,
    variant: Variant,
    hole: &[Card],
    board: &[Card],
    n_opponents: usize,
    samples: u32,
    rng: &mut R,
) -> f64 {
    crate::rules::rules(variant).equity(ev, hole, board, n_opponents, samples, rng)
}

/// [`equity`] for a statically known showdown evaluator `E` (what each format's rules call).
pub fn equity_with<E: HandEvaluator, R: Rng + ?Sized>(
    ev: Evaluator,
    hole: &[Card],
    board: &[Card],
    n_opponents: usize,
    samples: u32,
    rng: &mut R,
) -> f64 {
    assert!(board.len() <= 5 && !hole.is_empty());
    if n_opponents == 0 || samples == 0 {
        return 1.0;
    }
    let hc = hole.len();
    let dead = CardMask::from_cards(hole).0 | CardMask::from_cards(board).0;
    let mut live = [Card(0); 52];
    let mut n_live = 0;
    for i in 0..52u8 {
        if dead & (1u64 << i) == 0 {
            live[n_live] = Card(i);
            n_live += 1;
        }
    }
    let need_board = 5 - board.len();
    let need = need_board + n_opponents * hc;
    assert!(
        need <= n_live,
        "not enough cards for {n_opponents} opponents"
    );

    let mut full_board = [Card(0); 5];
    full_board[..board.len()].copy_from_slice(board);
    let mut total = 0.0f64;
    for _ in 0..samples {
        // Partial Fisher–Yates: the first `need` live cards become the sample.
        for i in 0..need {
            let j = i + below(rng, (n_live - i) as u32) as usize;
            live.swap(i, j);
        }
        full_board[board.len()..].copy_from_slice(&live[..need_board]);
        let mine = E::strength(ev, hole, &full_board);
        let mut best_opp = 0;
        let mut ties = 0u32;
        for o in 0..n_opponents {
            let start = need_board + o * hc;
            let s = E::strength(ev, &live[start..start + hc], &full_board);
            if s > best_opp {
                best_opp = s;
                ties = (s == mine) as u32;
            } else if s == best_opp && s == mine {
                ties += 1;
            }
        }
        if mine > best_opp {
            total += 1.0;
        } else if mine == best_opp {
            total += 1.0 / (ties + 1) as f64;
        }
    }
    total / samples as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::card::parse_cards;
    use rand_xoshiro::rand_core::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn eq(hole: &str, board: &str, opp: usize, variant: Variant) -> f64 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        equity(
            Evaluator::get(),
            variant,
            &parse_cards(hole).unwrap(),
            &parse_cards(board).unwrap(),
            opp,
            20_000,
            &mut rng,
        )
    }

    #[test]
    fn known_equities() {
        // AA vs one random hand ≈ 85.2%.
        let aa = eq("AsAh", "", 1, Variant::NLHE);
        assert!((aa - 0.852).abs() < 0.01, "{aa}");
        // 72o vs one random hand ≈ 34.6%.
        let seven_two = eq("7c2d", "", 1, Variant::NLHE);
        assert!((seven_two - 0.346).abs() < 0.015, "{seven_two}");
        // Royal flush on the river: always wins.
        assert_eq!(eq("AsKs", "QsJsTs2c3d", 3, Variant::NLHE), 1.0);
        // AA vs 5 random hands ≈ 49%.
        let aa5 = eq("AsAh", "", 5, Variant::NLHE);
        assert!((aa5 - 0.49).abs() < 0.02, "{aa5}");
        // Board plays (broadway straight, no flush possible): always a split.
        let split = eq("2c3d", "AsKdQhJcTs", 1, Variant::NLHE);
        assert!(split < 0.6 && split > 0.45, "{split}");
    }

    #[test]
    fn omaha_equity_reasonable() {
        let e = eq("AsAhKsKh", "", 1, Variant::PLO);
        assert!(e > 0.6 && e < 0.75, "{e}");
    }
}
