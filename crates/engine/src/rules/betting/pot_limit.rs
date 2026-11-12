//! Pot-limit: raises from a full raise up to the size of the pot after calling.

use super::{BetContext, BettingRules};
use crate::action::Street;

/// Pot-limit betting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PotLimit;

impl BettingRules for PotLimit {
    /// The big blind opens every street (minimum bet = 1 BB).
    fn opening_increment(&self, _street: Street, bb: u64) -> u64 {
        bb
    }
    /// Minimum: current bet + the last full raise increment.
    /// Maximum (pot-limit rule): current bet + (pot after calling), where the pot after
    /// calling is the pot including current-street bets plus the amount to call.
    fn raise_to_range(&self, ctx: &BetContext) -> (u64, u64) {
        (
            ctx.current_bet + ctx.last_raise,
            ctx.current_bet + ctx.pot + ctx.to_call,
        )
    }
}
