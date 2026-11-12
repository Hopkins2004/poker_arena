//! Betting structures. Each file implements [`BettingRules`] for one structure; the
//! [`Betting`] enum (part of [`GameFormat`](crate::format::GameFormat)) dispatches to them.

pub mod fixed_limit;
pub mod no_limit;
pub mod pot_limit;

use crate::action::Street;
use crate::format::Betting;

/// The betting state a structure needs to decide what is allowed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BetContext {
    /// Current street.
    pub street: Street,
    /// Big blind of the hand.
    pub bb: u64,
    /// Street total to match.
    pub current_bet: u64,
    /// Size of the last full bet/raise increment on this street.
    pub last_raise: u64,
    /// Full bets + raises so far on this street (the big blind counts preflop).
    pub bets_this_street: u8,
    /// Total chips in the pot, current-street bets included.
    pub pot: u64,
    /// Chips the acting player needs to call (not capped by the stack).
    pub to_call: u64,
}

/// What a betting structure decides.
pub trait BettingRules {
    /// The bet/raise increment that opens `street`: the minimum bet postflop, and the size
    /// of the big blind's "opening bet" preflop.
    fn opening_increment(&self, street: Street, bb: u64) -> u64;
    /// Whether the structure allows another bet or raise (e.g. a fixed-limit cap).
    fn raise_permitted(&self, ctx: &BetContext) -> bool {
        let _ = ctx;
        true
    }
    /// Legal raise-to range `(min, max)` as street totals, before capping at the player's
    /// all-in amount (`u64::MAX` = no upper limit).
    fn raise_to_range(&self, ctx: &BetContext) -> (u64, u64);
}

impl BettingRules for Betting {
    #[inline]
    fn opening_increment(&self, street: Street, bb: u64) -> u64 {
        match *self {
            Betting::NoLimit => no_limit::NoLimit.opening_increment(street, bb),
            Betting::PotLimit => pot_limit::PotLimit.opening_increment(street, bb),
            Betting::FixedLimit {
                small_bet,
                big_bet,
                cap,
            } => fixed_limit::FixedLimit {
                small_bet,
                big_bet,
                cap,
            }
            .opening_increment(street, bb),
        }
    }
    #[inline]
    fn raise_permitted(&self, ctx: &BetContext) -> bool {
        match *self {
            Betting::NoLimit => no_limit::NoLimit.raise_permitted(ctx),
            Betting::PotLimit => pot_limit::PotLimit.raise_permitted(ctx),
            Betting::FixedLimit {
                small_bet,
                big_bet,
                cap,
            } => fixed_limit::FixedLimit {
                small_bet,
                big_bet,
                cap,
            }
            .raise_permitted(ctx),
        }
    }
    #[inline]
    fn raise_to_range(&self, ctx: &BetContext) -> (u64, u64) {
        match *self {
            Betting::NoLimit => no_limit::NoLimit.raise_to_range(ctx),
            Betting::PotLimit => pot_limit::PotLimit.raise_to_range(ctx),
            Betting::FixedLimit {
                small_bet,
                big_bet,
                cap,
            } => fixed_limit::FixedLimit {
                small_bet,
                big_bet,
                cap,
            }
            .raise_to_range(ctx),
        }
    }
}
