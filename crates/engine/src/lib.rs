//! Poker rules engine: cards, deck, hand evaluation and the betting state machine.
//!
//! This crate has no I/O, no threads and no knowledge of matchmaking or ratings. Randomness
//! always comes from an explicitly seeded generator passed in by the caller.

#![warn(missing_docs)]

pub mod action;
pub mod card;
pub mod deck;
pub mod equity;
pub mod eval;
pub mod format;
pub mod rng;
pub mod rules;
pub mod state;

pub use action::{Action, ActionKind, ActionRecord, IllegalAction, LegalActions, Street};
pub use card::{Card, CardMask};
pub use deck::{Deck, DeckBuilder};
pub use eval::{Evaluator, HandAcc, HandCategory, Strength};
pub use format::{Betting, GameFormat, Variant, MAX_HOLE, MAX_SEATS};
pub use rules::{rules, FormatRules};
pub use state::{HandSetup, HandState, Observation, Pot, SeatStatus, SetupError};
