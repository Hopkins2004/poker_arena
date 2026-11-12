//! The parallel runner: rounds of scheduled tables run on a rayon pool, merged in table-id
//! order, with rating updates applied once per round.
//!
//! **Determinism.** Given (spec, master seed, bot set) a run is bit-for-bit reproducible
//! regardless of thread count: the schedule of round `r` is seeded from
//! `mix_seed(master, r, SCHEDULE_STREAM)`, table `t` of round `r` from
//! `mix_seed(master, r, t)`; tables share no state; results are collected in table-id order
//! and ratings are updated sequentially after each round.
//!
//! The pieces are exposed separately ([`RunState`], [`run_tables`]) so that other drivers
//! (e.g. the Python multiprocessing runner) can execute tables elsewhere and merge them.

use crate::bot::BotFactory;
use crate::leaderboard::LeaderboardRow;
use crate::matchmaking::{make_schedule, MatchmakingConfig, MatchmakingError, Schedule};
use crate::rating::{PairwiseElo, RatingConfig, RatingSystem, TableOutcome};
use crate::stats::SeatStats;
use crate::table::{run_table, TableConfig, TableError, TableResult};
use engine::rng::mix_seed;
use engine::GameFormat;
pub use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Seed stream for schedules (never collides with a table id).
pub const SCHEDULE_STREAM: u64 = u64::MAX;

/// Everything that defines a run, apart from the bots.
#[derive(Clone, Debug, PartialEq)]
pub struct RunSpec {
    /// Game format.
    pub format: GameFormat,
    /// Table settings.
    pub table: TableConfig,
    /// Matchmaking settings.
    pub matchmaking: MatchmakingConfig,
    /// Rating settings.
    pub rating: RatingConfig,
    /// Number of rounds.
    pub rounds: usize,
    /// Master seed.
    pub seed: u64,
    /// Worker threads (`None` = all cores).
    pub threads: Option<usize>,
}

/// A participating bot.
#[derive(Clone)]
pub struct BotEntry {
    /// Name (leaderboard key, together with the version).
    pub name: String,
    /// Optional version tag.
    pub version: Option<String>,
    /// Factory creating fresh instances.
    pub factory: BotFactory,
}

impl BotEntry {
    /// Leaderboard key: `name` or `name@version`.
    pub fn key(&self) -> String {
        bot_key(&self.name, self.version.as_deref())
    }
}

/// Leaderboard key for a name and optional version.
pub fn bot_key(name: &str, version: Option<&str>) -> String {
    match version {
        Some(v) if !v.is_empty() => format!("{name}@{v}"),
        _ => name.to_string(),
    }
}

/// Run errors.
#[derive(Debug)]
pub enum RunError {
    /// Invalid configuration.
    Config(String),
    /// Matchmaking failed.
    Matchmaking(MatchmakingError),
    /// A table failed.
    Table {
        /// Round.
        round: usize,
        /// Table id.
        table: usize,
        /// Cause.
        error: TableError,
    },
    /// I/O error.
    Io(std::io::Error),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Config(m) => write!(f, "config: {m}"),
            RunError::Matchmaking(e) => write!(f, "{e}"),
            RunError::Table {
                round,
                table,
                error,
            } => write!(f, "round {round} table {table}: {error}"),
            RunError::Io(e) => write!(f, "io: {e}"),
        }
    }
}
impl std::error::Error for RunError {}
impl From<std::io::Error> for RunError {
    fn from(e: std::io::Error) -> Self {
        RunError::Io(e)
    }
}

/// One played round.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoundRecord {
    /// Round index.
    pub round: usize,
    /// Schedule used.
    pub schedule: Schedule,
    /// Table results in table-id order.
    pub results: Vec<TableResult>,
}

/// Accumulated state of a run: ratings and per-bot statistics.
pub struct RunState {
    spec: RunSpec,
    keys: Vec<String>,
    elo: PairwiseElo,
    initial: Vec<f64>,
    stats: Vec<SeatStats>,
    /// Per (table, seat) bb/100 samples per bot, for confidence intervals.
    samples: Vec<Vec<f64>>,
    places: Vec<Vec<u32>>,
    rounds_applied: usize,
}

impl RunState {
    /// Creates the state for bots with leaderboard `keys` and starting ratings `initial`.
    pub fn new(spec: RunSpec, keys: Vec<String>, initial: Vec<f64>) -> Result<RunState, RunError> {
        if keys.is_empty() {
            return Err(RunError::Config("no bots".into()));
        }
        if initial.len() != keys.len() {
            return Err(RunError::Config(
                "one initial rating per bot required".into(),
            ));
        }
        let mut seen = std::collections::HashSet::new();
        for k in &keys {
            if !seen.insert(k) {
                return Err(RunError::Config(format!("duplicate bot key `{k}`")));
            }
        }
        spec.table
            .validate()
            .map_err(|e| RunError::Config(e.to_string()))?;
        spec.format
            .validate()
            .map_err(|e| RunError::Config(e.to_string()))?;
        let n = keys.len();
        Ok(RunState {
            elo: PairwiseElo::new(spec.rating.clone(), initial.clone()),
            spec,
            keys,
            initial,
            stats: vec![SeatStats::default(); n],
            samples: vec![Vec::new(); n],
            places: vec![Vec::new(); n],
            rounds_applied: 0,
        })
    }

    /// The run spec.
    pub fn spec(&self) -> &RunSpec {
        &self.spec
    }
    /// Bot keys, indexed by bot.
    pub fn keys(&self) -> &[String] {
        &self.keys
    }
    /// Current ratings, indexed by bot.
    pub fn ratings(&self) -> &[f64] {
        self.elo.ratings()
    }
    /// Rounds applied so far.
    pub fn rounds_applied(&self) -> usize {
        self.rounds_applied
    }

    /// Schedule of round `round`.
    pub fn schedule(&self, round: usize) -> Result<Schedule, RunError> {
        make_schedule(
            self.keys.len(),
            self.spec.format.max_seats as usize,
            &self.spec.matchmaking,
            mix_seed(self.spec.seed, round as u64, SCHEDULE_STREAM),
        )
        .map_err(RunError::Matchmaking)
    }

    /// Seed of table `table` in round `round`.
    pub fn table_seed(&self, round: usize, table: usize) -> u64 {
        table_seed(self.spec.seed, round, table)
    }

    /// Merges a round's results (which must be in table-id order) and updates ratings.
    pub fn apply_round(&mut self, results: &[TableResult]) {
        self.elo.begin_round();
        for r in results {
            self.elo.add_table(&TableOutcome::from_result(r));
            for s in &r.seats {
                self.stats[s.bot].merge(&s.stats);
                self.samples[s.bot].push(s.stats.bb_per_100(r.bb));
                if let Some(p) = s.place {
                    self.places[s.bot].push(p);
                }
            }
        }
        self.elo.end_round();
        self.rounds_applied += 1;
    }

    /// Leaderboard rows, sorted by rating (best first).
    pub fn rows(&self) -> Vec<LeaderboardRow> {
        let bb = self.spec.table.reference_bb();
        let mut rows: Vec<LeaderboardRow> = (0..self.keys.len())
            .map(|b| {
                LeaderboardRow::build(
                    &self.keys[b],
                    self.ratings()[b],
                    self.initial[b],
                    &self.stats[b],
                    &self.samples[b],
                    &self.places[b],
                    bb,
                )
            })
            .collect();
        rows.sort_by(|a, b| b.rating.total_cmp(&a.rating).then(a.key.cmp(&b.key)));
        rows
    }
}

/// Seed of a table: `mix_seed(master, round, table)`.
pub fn table_seed(master: u64, round: usize, table: usize) -> u64 {
    mix_seed(master, round as u64, table as u64)
}

/// Runs all tables of a schedule in parallel on `pool`, returning results in table-id order.
pub fn run_tables(
    spec: &RunSpec,
    factories: &[BotFactory],
    schedule: &Schedule,
    round: usize,
    pool: &rayon::ThreadPool,
    progress: &ProgressBar,
) -> Result<Vec<TableResult>, RunError> {
    pool.install(|| {
        schedule
            .tables
            .par_iter()
            .enumerate()
            .map(|(t, seats)| {
                let r = run_table(
                    &spec.format,
                    seats,
                    factories,
                    &spec.table,
                    table_seed(spec.seed, round, t),
                )
                .map_err(|error| RunError::Table {
                    round,
                    table: t,
                    error,
                });
                progress.inc(1);
                r
            })
            .collect()
    })
}

/// Builds the worker pool for a spec.
pub fn build_pool(threads: Option<usize>) -> Result<rayon::ThreadPool, RunError> {
    let mut b = rayon::ThreadPoolBuilder::new();
    if let Some(n) = threads {
        if n == 0 {
            return Err(RunError::Config("threads must be >= 1".into()));
        }
        b = b.num_threads(n);
    }
    b.build().map_err(|e| RunError::Config(e.to_string()))
}

/// Output of a complete run.
pub struct RunOutput {
    /// Final state (ratings, stats).
    pub state: RunState,
    /// Every round's schedule and results.
    pub rounds: Vec<RoundRecord>,
}

/// Runs `spec.rounds` rounds for `bots` with the given starting ratings.
pub fn run(
    spec: &RunSpec,
    bots: &[BotEntry],
    initial: Vec<f64>,
    quiet: bool,
) -> Result<RunOutput, RunError> {
    let keys = bots.iter().map(|b| b.key()).collect();
    let factories: Vec<BotFactory> = bots.iter().map(|b| b.factory.clone()).collect();
    let mut state = RunState::new(spec.clone(), keys, initial)?;
    let pool = build_pool(spec.threads)?;
    let schedules: Vec<Schedule> = (0..spec.rounds)
        .map(|r| state.schedule(r))
        .collect::<Result<_, _>>()?;
    let total: usize = schedules.iter().map(|s| s.tables.len()).sum();
    let pb = if quiet {
        ProgressBar::hidden()
    } else {
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner} round {msg} [{bar:40}] {pos}/{len} tables ({elapsed}, eta {eta})",
            )
            .unwrap()
            .progress_chars("=> "),
        );
        pb
    };
    let mut rounds = Vec::with_capacity(spec.rounds);
    for (r, schedule) in schedules.into_iter().enumerate() {
        pb.set_message(format!("{}/{}", r + 1, spec.rounds));
        let results = run_tables(spec, &factories, &schedule, r, &pool, &pb)?;
        state.apply_round(&results);
        rounds.push(RoundRecord {
            round: r,
            schedule,
            results,
        });
    }
    pb.finish_and_clear();
    Ok(RunOutput { state, rounds })
}

/// Runs a single table of a schedule (for drivers that execute tables outside the pool).
pub fn run_table_external(
    spec: &RunSpec,
    factories: &[BotFactory],
    schedule: &Schedule,
    round: usize,
    table: usize,
) -> Result<TableResult, RunError> {
    let seats = schedule
        .tables
        .get(table)
        .ok_or_else(|| RunError::Config(format!("no table {table} in schedule")))?;
    run_table(
        &spec.format,
        seats,
        factories,
        &spec.table,
        table_seed(spec.seed, round, table),
    )
    .map_err(|error| RunError::Table {
        round,
        table,
        error,
    })
}
