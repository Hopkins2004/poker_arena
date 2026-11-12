//! Actions, legal-action summaries and the public action record.

/// A betting street.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Street {
    /// Before the flop.
    Preflop = 0,
    /// Three board cards.
    Flop = 1,
    /// Fourth board card.
    Turn = 2,
    /// Fifth board card.
    River = 3,
}

impl Street {
    /// Number of board cards visible on this street.
    pub const fn board_len(self) -> usize {
        match self {
            Street::Preflop => 0,
            Street::Flop => 3,
            Street::Turn => 4,
            Street::River => 5,
        }
    }
    /// The following street, if any.
    pub const fn next(self) -> Option<Street> {
        match self {
            Street::Preflop => Some(Street::Flop),
            Street::Flop => Some(Street::Turn),
            Street::Turn => Some(Street::River),
            Street::River => None,
        }
    }
}

/// An action chosen by a bot.
///
/// Amounts use the **raise-to** convention: the player's *total contribution on the current
/// street* after the action, not the increment. `BetTo` and `RaiseTo` are interchangeable; the
/// engine only checks that the amount is legal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Action {
    /// Give up the hand. Always legal for the player to act.
    Fold,
    /// Pass without betting. Legal only when there is nothing to call.
    Check,
    /// Match the current bet (all-in if short). When there is nothing to call this is
    /// treated as a check.
    Call,
    /// Open the betting to the given street total.
    BetTo(u64),
    /// Raise to the given street total.
    RaiseTo(u64),
    /// Commit the whole stack. Legal when that is a call, or when it is a raise permitted by
    /// the betting structure (in pot- or fixed-limit a deep stack cannot always shove).
    AllIn,
}

/// Precomputed summary of what the player to act may do. Bots never need to re-derive rules.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct LegalActions {
    /// True when there is no bet to call (check is legal).
    pub can_check: bool,
    /// Chips needed to call (capped at the stack; a short call is an all-in). Zero when
    /// `can_check`.
    pub call_amount: u64,
    /// True when betting/raising is allowed (`min_raise_to..=max_raise_to`).
    pub can_raise: bool,
    /// Minimum legal bet/raise-to (street total). Equals the all-in amount when the stack is
    /// smaller than a full raise. Zero when `!can_raise`.
    pub min_raise_to: u64,
    /// Maximum legal bet/raise-to (street total). Zero when `!can_raise`.
    pub max_raise_to: u64,
}

/// Kind of a recorded public action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActionKind {
    /// Forced ante.
    PostAnte,
    /// Forced small blind.
    PostSmallBlind,
    /// Forced big blind.
    PostBigBlind,
    /// Fold.
    Fold,
    /// Check.
    Check,
    /// Call (possibly all-in for less).
    Call,
    /// Opening bet on a street.
    Bet,
    /// Raise over an existing bet (preflop the big blind counts as the bet).
    Raise,
}

/// One entry of the public action history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActionRecord {
    /// Acting seat.
    pub seat: u8,
    /// Street of the action.
    pub street: Street,
    /// What happened.
    pub kind: ActionKind,
    /// Chips moved from the stack into the pot by this action.
    pub amount: u64,
    /// The seat's street contribution after the action (raise-to total).
    pub to: u64,
    /// True if the action put the player all-in.
    pub all_in: bool,
}

/// Why an action was rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IllegalAction {
    /// The hand is over or it is not this seat's turn.
    NotToAct,
    /// Check while facing a bet.
    CannotCheck,
    /// Bet/raise/all-in-raise when raising is not allowed (betting not reopened, cap reached,
    /// no opponent who could respond, or not enough chips beyond a call).
    RaiseNotAllowed,
    /// Bet/raise below the minimum (and not all-in).
    BelowMinRaise,
    /// Bet/raise above the maximum (stack or pot limit).
    AboveMaxRaise,
}

impl std::fmt::Display for IllegalAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}
impl std::error::Error for IllegalAction {}
