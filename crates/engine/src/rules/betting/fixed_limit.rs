//! Fixed-limit: every bet and raise is exactly one small bet (preflop, flop) or one big bet
//! (turn, river), and a street allows at most `cap` bets + raises.

use super::{BetContext, BettingRules};
use crate::action::Street;

/// Fixed-limit betting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedLimit {
    /// Bet size preflop and on the flop.
    pub small_bet: u64,
    /// Bet size on the turn and river.
    pub big_bet: u64,
    /// Maximum bets + raises per street (the big blind counts as the first preflop).
    pub cap: u8,
}

impl FixedLimit {
    /// The fixed bet size of a street.
    pub fn size(&self, street: Street) -> u64 {
        if street >= Street::Turn {
            self.big_bet
        } else {
            self.small_bet
        }
    }
}

impl BettingRules for FixedLimit {
    fn opening_increment(&self, street: Street, _bb: u64) -> u64 {
        self.size(street)
    }
    fn raise_permitted(&self, ctx: &BetContext) -> bool {
        ctx.bets_this_street < self.cap
    }
    /// Exactly one bet size above the current bet.
    fn raise_to_range(&self, ctx: &BetContext) -> (u64, u64) {
        let to = ctx.current_bet + self.size(ctx.street);
        (to, to)
    }
}
