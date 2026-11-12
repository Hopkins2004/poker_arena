//! No-Limit Texas Hold'em: two hole cards, best five of seven, no-limit betting.

use crate::card::Card;
use crate::equity::equity_with;
use crate::eval::{Evaluator, HandEvaluator, HoldemEval, Strength};
use crate::format::{Betting, FormatError, GameFormat, Variant};
use crate::rules::FormatRules;
use rand_core::Rng;

/// No-Limit Hold'em rules.
#[derive(Clone, Copy, Debug, Default)]
pub struct Nlhe;

/// The NLHE format for a table of `max_seats`.
pub const fn format(max_seats: u8) -> GameFormat {
    GameFormat {
        variant: Variant::NLHE,
        max_seats,
        hole_cards: 2,
        betting: Betting::NoLimit,
    }
}

impl FormatRules for Nlhe {
    fn variant(&self) -> Variant {
        Variant::NLHE
    }
    fn name(&self) -> &'static str {
        "nlhe"
    }
    fn title(&self) -> &'static str {
        "Hold'em No Limit"
    }
    fn default_format(&self, max_seats: u8) -> GameFormat {
        format(max_seats)
    }
    fn validate(&self, f: &GameFormat) -> Result<(), FormatError> {
        if f.hole_cards != 2 {
            return Err(FormatError("Hold'em uses 2 hole cards".into()));
        }
        if f.betting != Betting::NoLimit {
            return Err(FormatError("NLHE uses no-limit betting".into()));
        }
        Ok(())
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
