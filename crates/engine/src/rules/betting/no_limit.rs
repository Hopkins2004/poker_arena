//! No-limit: any raise from a full raise up to the whole stack.

use super::{BetContext, BettingRules};
use crate::action::Street;

/// No-limit betting.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoLimit;

impl BettingRules for NoLimit {
    /// The big blind opens every street (minimum bet = 1 BB).
    fn opening_increment(&self, _street: Street, bb: u64) -> u64 {
        bb
    }
    /// Minimum: current bet + the last full raise increment. Maximum: the stack.
    fn raise_to_range(&self, ctx: &BetContext) -> (u64, u64) {
        (ctx.current_bet + ctx.last_raise, u64::MAX)
    }
}
