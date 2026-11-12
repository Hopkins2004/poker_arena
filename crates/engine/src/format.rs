//! Game format description: variant, table size, hole cards and betting structure.

/// Poker variant. Determines how showdown hands are evaluated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
    /// No-limit Texas Hold'em.
    NLHE,
    /// Pot-limit Omaha (high only): exactly two hole cards and three board cards.
    PLO,
    /// Fixed-limit Texas Hold'em.
    FLHE,
}

/// Betting structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Betting {
    /// Any raise up to the whole stack.
    NoLimit,
    /// Maximum raise-to = current bet + (pot after calling).
    PotLimit,
    /// Fixed bet sizes: `small_bet` preflop and on the flop, `big_bet` on turn and river.
    /// `cap` is the maximum number of bets plus raises per street (the big blind counts as the
    /// opening bet preflop).
    FixedLimit {
        /// Bet and raise size preflop and on the flop.
        small_bet: u64,
        /// Bet and raise size on the turn and river.
        big_bet: u64,
        /// Maximum number of bets + raises per street.
        cap: u8,
    },
}

/// Maximum number of seats at any table.
pub const MAX_SEATS: usize = 10;
/// Maximum number of hole cards for any supported variant.
pub const MAX_HOLE: usize = 6;

/// Full description of a game format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GameFormat {
    /// Variant (decides the showdown evaluator).
    pub variant: Variant,
    /// Maximum number of seats at a table (2..=10).
    pub max_seats: u8,
    /// Hole cards per player (2 for Hold'em, 4 for PLO).
    pub hole_cards: u8,
    /// Betting structure.
    pub betting: Betting,
}

/// Error describing an inconsistent format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatError(pub String);

impl std::fmt::Display for FormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid game format: {}", self.0)
    }
}
impl std::error::Error for FormatError {}

impl GameFormat {
    /// No-limit Hold'em with `seats` seats.
    pub const fn nlhe(seats: u8) -> GameFormat {
        GameFormat {
            variant: Variant::NLHE,
            max_seats: seats,
            hole_cards: 2,
            betting: Betting::NoLimit,
        }
    }
    /// Pot-limit Omaha with `seats` seats.
    pub const fn plo(seats: u8) -> GameFormat {
        GameFormat {
            variant: Variant::PLO,
            max_seats: seats,
            hole_cards: 4,
            betting: Betting::PotLimit,
        }
    }
    /// Fixed-limit Hold'em with the given bet sizes and a cap of 4 (bet, raise, re-raise, cap).
    pub const fn flhe(seats: u8, small_bet: u64, big_bet: u64) -> GameFormat {
        GameFormat {
            variant: Variant::FLHE,
            max_seats: seats,
            hole_cards: 2,
            betting: Betting::FixedLimit {
                small_bet,
                big_bet,
                cap: 4,
            },
        }
    }

    /// Checks internal consistency.
    pub fn validate(&self) -> Result<(), FormatError> {
        let seats = self.max_seats as usize;
        if !(2..=MAX_SEATS).contains(&seats) {
            return Err(FormatError(format!(
                "max_seats {seats} not in 2..={MAX_SEATS}"
            )));
        }
        let hc = self.hole_cards as usize;
        match self.variant {
            Variant::NLHE | Variant::FLHE if hc != 2 => {
                return Err(FormatError("Hold'em needs 2 hole cards".into()))
            }
            Variant::PLO if !(4..=MAX_HOLE).contains(&hc) => {
                return Err(FormatError("Omaha needs 4..=6 hole cards".into()))
            }
            _ => {}
        }
        if seats * hc + 5 > 52 {
            return Err(FormatError("not enough cards for this many seats".into()));
        }
        match (self.variant, self.betting) {
            (
                Variant::FLHE,
                Betting::FixedLimit {
                    small_bet,
                    big_bet,
                    cap,
                },
            ) if small_bet == 0 || big_bet == 0 || cap == 0 => {
                return Err(FormatError("fixed-limit sizes and cap must be > 0".into()));
            }
            (Variant::FLHE, Betting::FixedLimit { .. }) => {}
            (Variant::FLHE, _) => return Err(FormatError("FLHE needs FixedLimit betting".into())),
            (_, Betting::FixedLimit { .. }) => {
                return Err(FormatError(
                    "FixedLimit betting is only supported for FLHE".into(),
                ))
            }
            _ => {}
        }
        Ok(())
    }
}
