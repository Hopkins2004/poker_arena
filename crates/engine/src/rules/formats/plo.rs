//! Pot-Limit Omaha (high): four hole cards (4–6 supported), the best hand uses exactly two
//! hole cards and exactly three board cards, pot-limit betting.

use crate::card::Card;
use crate::equity::equity_with;
use crate::eval::{Evaluator, HandEvaluator, OmahaEval, Strength};
use crate::format::{Betting, FormatError, GameFormat, Variant, MAX_HOLE};
use crate::rules::FormatRules;
use rand_core::Rng;

/// Pot-Limit Omaha rules.
#[derive(Clone, Copy, Debug, Default)]
pub struct Plo;

/// The PLO format (four hole cards) for a table of `max_seats`.
pub const fn format(max_seats: u8) -> GameFormat {
    GameFormat {
        variant: Variant::PLO,
        max_seats,
        hole_cards: 4,
        betting: Betting::PotLimit,
    }
}

impl FormatRules for Plo {
    fn variant(&self) -> Variant {
        Variant::PLO
    }
    fn name(&self) -> &'static str {
        "plo"
    }
    fn title(&self) -> &'static str {
        "Omaha Pot Limit"
    }
    fn default_format(&self, max_seats: u8) -> GameFormat {
        format(max_seats)
    }
    fn validate(&self, f: &GameFormat) -> Result<(), FormatError> {
        if !(4..=MAX_HOLE).contains(&(f.hole_cards as usize)) {
            return Err(FormatError(format!(
                "Omaha needs 4..={MAX_HOLE} hole cards"
            )));
        }
        if f.betting != Betting::PotLimit {
            return Err(FormatError("PLO uses pot-limit betting".into()));
        }
        Ok(())
    }
    fn strength(&self, ev: Evaluator, hole: &[Card], board: &[Card; 5]) -> Strength {
        OmahaEval::strength(ev, hole, board)
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
        equity_with::<OmahaEval, _>(ev, hole, board, n_opponents, samples, rng)
    }
}
