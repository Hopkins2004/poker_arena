//! Pluggable game rules: everything that differs between poker formats lives here.
//!
//! A **format** (NLHE, PLO, FLHE, ...) is a deal shape (hole cards), a showdown evaluator and
//! a betting structure:
//!
//! * [`formats`]: one file per format, each implementing [`FormatRules`] (how many hole
//!   cards, how hands are scored at showdown, equity, validation, names).
//! * [`betting`]: one file per betting structure (no-limit, pot-limit, fixed-limit), each
//!   implementing [`betting::BettingRules`] (opening bet, legal raise range, caps).
//! * [`actions`]: the structure-independent parts, i.e. computing a player's
//!   [`LegalActions`](crate::action::LegalActions) from the betting state ("what may I do?") and
//!   validating the action a bot returns ("is this allowed?").
//!
//! The hand state machine ([`crate::state`]) knows nothing format-specific: it asks these
//! modules.
//!
//! **Adding a format:** write `formats/<name>.rs` implementing [`FormatRules`], add a variant
//! to [`Variant`] and one line to [`rules`] and [`ALL`]. A new betting structure is a new
//! file in `betting/` plus a [`Betting`] variant.

pub mod actions;
pub mod betting;
pub mod formats;

use crate::card::Card;
use crate::eval::{Evaluator, Strength};
use crate::format::{FormatError, GameFormat, Variant};
use rand_core::Rng;

/// Format-specific rules. Implemented by one zero-sized type per format in [`formats`].
pub trait FormatRules: Send + Sync {
    /// Format id.
    fn variant(&self) -> Variant;
    /// Short config name, e.g. `"nlhe"`.
    fn name(&self) -> &'static str;
    /// Human-readable name, e.g. `"Hold'em No Limit"`.
    fn title(&self) -> &'static str;
    /// The standard format for a table of `max_seats` (small blind 1, big blind 2 where the
    /// betting structure needs sizes).
    fn default_format(&self, max_seats: u8) -> GameFormat;
    /// Format-specific consistency checks (generic ones are in [`GameFormat::validate`]).
    fn validate(&self, format: &GameFormat) -> Result<(), FormatError>;
    /// Showdown strength of `hole` with the full `board` (higher is better).
    fn strength(&self, ev: Evaluator, hole: &[Card], board: &[Card; 5]) -> Strength;
    /// Monte Carlo equity of `hole` with the visible `board` against `n_opponents` random
    /// hands (see [`crate::equity`]).
    fn equity(
        &self,
        ev: Evaluator,
        hole: &[Card],
        board: &[Card],
        n_opponents: usize,
        samples: u32,
        rng: &mut dyn Rng,
    ) -> f64;
}

/// Every registered format.
pub const ALL: &[&'static dyn FormatRules] = &[
    &formats::nlhe::Nlhe,
    &formats::plo::Plo,
    &formats::flhe::Flhe,
];

/// The rules of a format.
#[inline]
pub fn rules(variant: Variant) -> &'static dyn FormatRules {
    match variant {
        Variant::NLHE => &formats::nlhe::Nlhe,
        Variant::PLO => &formats::plo::Plo,
        Variant::FLHE => &formats::flhe::Flhe,
    }
}

/// Looks a format up by its config name (case-insensitive), e.g. `"plo"`.
pub fn by_name(name: &str) -> Option<&'static dyn FormatRules> {
    ALL.iter()
        .copied()
        .find(|r| r.name().eq_ignore_ascii_case(name))
}

pub use crate::format::Betting;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_consistent() {
        for r in ALL {
            assert_eq!(rules(r.variant()).name(), r.name());
            assert_eq!(by_name(r.name()).unwrap().variant(), r.variant());
            let f = r.default_format(6);
            assert_eq!(f.variant, r.variant());
            f.validate().unwrap();
        }
        assert!(by_name("NLHE").is_some());
        assert!(by_name("stud").is_none());
    }
}
