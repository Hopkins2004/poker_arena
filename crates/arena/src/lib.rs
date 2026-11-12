//! Arena: bot interface, table runner, stats, matchmaking, ratings and the parallel runner.
//!
//! This crate contains no poker rules; those live in `engine`.

#![warn(missing_docs)]
// Table code indexes several parallel per-seat arrays by seat number; range loops over
// seats are clearer than zipped iterators there.
#![allow(clippy::needless_range_loop)]

pub mod bot;
pub mod hand_history;
pub mod leaderboard;
pub mod matchmaking;
pub mod rating;
pub mod runner;
pub mod stats;
pub mod table;

pub use bot::{Bot, BotFactory, HandResult, HandStart, MatchInfo, PublicEvent};
pub use leaderboard::{LeaderboardFile, LeaderboardRow};
pub use matchmaking::{make_schedule, MatchmakingConfig, Schedule};
pub use rating::{PairwiseElo, RatingConfig, RatingSystem, ScoreMode, TableOutcome};
pub use runner::{run, BotEntry, RunError, RunOutput, RunSpec, RunState};
pub use stats::SeatStats;
pub use table::{
    run_table, BlindLevel, Blinds, IllegalActionPolicy, SeatResult, StackPolicy, TableConfig,
    TableError, TableMode, TableResult,
};
