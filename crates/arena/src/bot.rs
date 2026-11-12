//! The bot interface.
//!
//! A bot sees the game only through these types. [`Observation`] is built by the engine for
//! the acting seat and contains that seat's hole cards plus public information; other seats'
//! hole cards only ever appear in [`PublicEvent::Showdown`] for hands that reach showdown.
//!
//! **Duplicate mode:** deals are replayed with bots rotated through the seats, so a bot must
//! not carry hand-specific state (e.g. "my cards last hand") across hands, and must read its
//! seat from [`HandStart::seat`] (not [`MatchInfo::seat`], which is only the initial seat).

use std::sync::Arc;

pub use engine::action::{Action, ActionKind, ActionRecord, LegalActions, Street};
pub use engine::card::Card;
pub use engine::eval::Strength;
pub use engine::format::{Betting, GameFormat, Variant};
pub use engine::state::{Observation, SeatStatus};

/// Information given once per table, before the first hand.
#[derive(Clone, Copy, Debug)]
pub struct MatchInfo {
    /// Game format.
    pub format: GameFormat,
    /// Number of seats at the table.
    pub n_seats: usize,
    /// This bot's initial seat (seats rotate in duplicate mode; see [`HandStart::seat`]).
    pub seat: usize,
    /// Small blind at the start of the match.
    pub sb: u64,
    /// Big blind at the start of the match.
    pub bb: u64,
    /// Ante at the start of the match.
    pub ante: u64,
    /// Starting stack in chips.
    pub starting_stack: u64,
    /// Planned number of hands (an upper bound in tournament mode).
    pub hands: u64,
    /// True for tournament mode (players are eliminated, blinds increase).
    pub tournament: bool,
    /// True when duplicate mode is on.
    pub duplicate: bool,
}

/// Information given at the start of each hand the bot is dealt into.
#[derive(Clone, Copy, Debug)]
pub struct HandStart<'a> {
    /// Hand number within the table, starting at 0.
    pub hand_id: u64,
    /// This bot's seat for this hand.
    pub seat: usize,
    /// This bot's hole cards.
    pub hole: &'a [Card],
    /// Button seat.
    pub button: usize,
    /// Stacks of every seat before blinds and antes (0 = empty seat).
    pub stacks: &'a [u64],
    /// Small blind.
    pub sb: u64,
    /// Big blind.
    pub bb: u64,
    /// Ante.
    pub ante: u64,
}

/// A public event, broadcast to every bot in the hand.
#[derive(Clone, Copy, Debug)]
pub enum PublicEvent<'a> {
    /// An action, including posted blinds and antes.
    Action(ActionRecord),
    /// A new street was dealt; `board` is the full visible board.
    Street {
        /// The new street.
        street: Street,
        /// Visible board cards.
        board: &'a [Card],
    },
    /// A player's hole cards revealed at showdown.
    Showdown {
        /// Revealing seat.
        seat: usize,
        /// Its hole cards.
        hole: &'a [Card],
        /// Its hand strength (higher is better).
        strength: Strength,
    },
    /// Chips a seat collected at the end of the hand (one event per winning seat, summed
    /// over the main and side pots it won).
    PotAwarded {
        /// Receiving seat.
        seat: usize,
        /// Chips awarded.
        amount: u64,
    },
    /// An uncalled bet returned to the bettor before showdown.
    UncalledReturned {
        /// Bettor.
        seat: usize,
        /// Chips returned.
        amount: u64,
    },
}

/// Summary of a finished hand.
#[derive(Clone, Copy, Debug)]
pub struct HandResult<'a> {
    /// Hand number within the table.
    pub hand_id: u64,
    /// Net chips per seat (0 for seats not in the hand).
    pub net: &'a [i64],
    /// Stacks after the hand.
    pub stacks: &'a [u64],
    /// Final visible board.
    pub board: &'a [Card],
    /// True if the hand went to showdown.
    pub showdown: bool,
}

/// A poker bot. Only [`Bot::act`] is required.
///
/// Bots never mutate game state: they return an [`Action`] and the engine validates it.
/// Illegal actions are handled per the table's illegal-action policy and counted in stats.
pub trait Bot: Send {
    /// Display name.
    fn name(&self) -> &str;
    /// Called once per table before the first hand.
    fn new_match(&mut self, _info: &MatchInfo) {}
    /// Called at the start of every hand the bot is dealt into.
    fn new_hand(&mut self, _info: &HandStart) {}
    /// Chooses an action. `obs.legal` describes what is allowed.
    fn act(&mut self, obs: &Observation) -> Action;
    /// Called for every public event of hands the bot is dealt into.
    fn observe(&mut self, _event: &PublicEvent) {}
    /// Called at the end of every hand the bot was dealt into.
    fn end_hand(&mut self, _result: &HandResult) {}
}

/// Creates a fresh bot instance from a seed. Each table gets fresh instances, so bots share
/// no mutable state across tables; shared read-only data should be captured in an `Arc`.
pub type BotFactory = Arc<dyn Fn(u64) -> Box<dyn Bot> + Send + Sync>;
