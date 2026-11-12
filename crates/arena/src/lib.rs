//! Arena: bot interface, table runner, stats, matchmaking, ratings and the parallel runner.
//!
//! This crate contains no poker rules; those live in `engine`.

#![warn(missing_docs)]
// Table code indexes several parallel per-seat arrays by seat number; range loops over
// seats are clearer than zipped iterators there.
#![allow(clippy::needless_range_loop)]

pub mod bot;
pub mod hand_history;
pub mod stats;
pub mod table;

pub use bot::{Bot, BotFactory, HandResult, HandStart, MatchInfo, PublicEvent};
pub use stats::SeatStats;
pub use table::{
    run_table, BlindLevel, Blinds, IllegalActionPolicy, SeatResult, StackPolicy, TableConfig,
    TableError, TableMode, TableResult,
};
