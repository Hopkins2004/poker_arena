//! Fixed-Limit Texas Hold'em: two hole cards, best five of seven, fixed bet sizes (small bet
//! preflop and on the flop, big bet on the turn and river) with a per-street cap.

use crate::card::Card;
use crate::equity::equity_with;
use crate::eval::{Evaluator, HandEvaluator, HoldemEval, Strength};
use crate::format::{Betting, FormatError, GameFormat, Variant};
use crate::rules::FormatRules;
use rand_core::Rng;

/// Fixed-Limit Hold'em rules.
#[derive(Clone, Copy, Debug, Default)]
pub struct Flhe;

/// The FLHE format with the given bet sizes and a cap of 4 (bet, raise, re-raise, cap).
pub const fn format(max_seats: u8, small_bet: u64, big_bet: u64) -> GameFormat {
    GameFormat {
        variant: Variant::FLHE,
        max_seats,
        hole_cards: 2,
        betting: Betting::FixedLimit {
            small_bet,
            big_bet,
            cap: 4,
        },
    }
}

impl FormatRules for Flhe {
    fn variant(&self) -> Variant {
        Variant::FLHE
    }
    fn name(&self) -> &'static str {
        "flhe"
    }
    fn title(&self) -> &'static str {
        "Hold'em Limit"
    }
    fn default_format(&self, max_seats: u8) -> GameFormat {
        format(max_seats, 2, 4)
    }
    fn validate(&self, f: &GameFormat) -> Result<(), FormatError> {
        if f.hole_cards != 2 {
            return Err(FormatError("Hold'em uses 2 hole cards".into()));
        }
        match f.betting {
            Betting::FixedLimit {
                small_bet,
                big_bet,
                cap,
            } if small_bet > 0 && big_bet > 0 && cap > 0 => Ok(()),
            Betting::FixedLimit { .. } => {
                Err(FormatError("fixed-limit sizes and cap must be > 0".into()))
            }
            _ => Err(FormatError("FLHE uses fixed-limit betting".into())),
        }
    }
    fn strength(&self, ev: Evaluator, hole: &[Card], board: &[Card; 5]) -> Strength {
        HoldemEval::strength(ev, hole, board)
    }
    fn equity(
        &self,
        ev: Evaluator,
        hole: &[Card],
        board: &[Card],
        n_opponents: usize,
        samples: u32,
        rng: &mut dyn Rng,
    ) -> f64 {
        equity_with::<HoldemEval, _>(ev, hole, board, n_opponents, samples, rng)
    }
}
