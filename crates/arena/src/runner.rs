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
use crate::table::{
    run_table, run_table_with_hooks, TableConfig, TableError, TableHooks, TableResult,
};
use engine::rng::mix_seed;
use engine::GameFormat;
pub use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

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
    /// Stopped through [`RunProgress::cancel`].
    Cancelled,
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
            RunError::Cancelled => write!(f, "cancelled"),
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
    /// Per (table, bot) pooled bb/100 samples, for confidence intervals.
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
            for s in &r.seats {
                self.stats[s.bot].merge(&s.stats);
                if let Some(p) = s.place {
                    self.places[s.bot].push(p);
                }
            }
            // One pooled bb/100 sample per (table, bot), for the CI.
            let mut seen: Vec<usize> = Vec::with_capacity(r.seats.len());
            for s in &r.seats {
                if seen.contains(&s.bot) {
                    continue;
                }
                seen.push(s.bot);
                let (net, hands) = r
                    .seats
                    .iter()
                    .filter(|o| o.bot == s.bot)
                    .fold((0i64, 0u64), |a, o| {
                        (a.0 + o.stats.net, a.1 + o.stats.hands)
                    });
                if hands > 0 {
                    self.samples[s.bot].push(net as f64 / r.bb as f64 / hands as f64 * 100.0);
                }
            }
            self.elo.add_table(&TableOutcome::from_result(r));
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

/// Live progress of a run, shareable across threads (clone it to observe a run from a UI).
/// Counters only feed displays; they never affect results.
#[derive(Clone)]
pub struct RunProgress {
    /// Terminal progress bar (tables); hidden for quiet runs and UIs.
    pub bar: ProgressBar,
    /// Hands finished so far.
    pub hands_done: Arc<AtomicU64>,
    /// Planned hands (an upper bound for tournaments), set when the run starts.
    pub hands_total: Arc<AtomicU64>,
    /// Tables finished so far.
    pub tables_done: Arc<AtomicU64>,
    /// Planned tables, set when the run starts.
    pub tables_total: Arc<AtomicU64>,
    /// Rounds finished so far.
    pub rounds_done: Arc<AtomicU64>,
    /// Set to stop the run: every table stops before its next hand.
    pub cancel: Arc<AtomicBool>,
}

impl RunProgress {
    /// Progress with a hidden terminal bar.
    pub fn hidden() -> RunProgress {
        RunProgress::with_bar(ProgressBar::hidden())
    }
    /// Progress driving the given terminal bar.
    pub fn with_bar(bar: ProgressBar) -> RunProgress {
        RunProgress {
            bar,
            hands_done: Arc::default(),
            hands_total: Arc::default(),
            tables_done: Arc::default(),
            tables_total: Arc::default(),
            rounds_done: Arc::default(),
            cancel: Arc::default(),
        }
    }

    /// Requests cancellation (takes effect before the next hand of every table).
    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    /// True once cancellation was requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

/// Runs all tables of a schedule in parallel on `pool`, returning results in table-id order.
pub fn run_tables(
    spec: &RunSpec,
    factories: &[BotFactory],
    schedule: &Schedule,
    round: usize,
    pool: &rayon::ThreadPool,
    progress: &RunProgress,
) -> Result<Vec<TableResult>, RunError> {
    pool.install(|| {
        schedule
            .tables
            .par_iter()
            .enumerate()
            .map(|(t, seats)| {
                let hooks = TableHooks {
                    hands_done: Some(&progress.hands_done),
                    cancel: Some(&progress.cancel),
                    on_hand: None,
                };
                let r = run_table_with_hooks(
                    &spec.format,
                    seats,
                    factories,
                    &spec.table,
                    table_seed(spec.seed, round, t),
                    hooks,
                )
                .map_err(|error| table_error(round, t, error));
                progress.tables_done.fetch_add(1, Ordering::Relaxed);
                progress.bar.inc(1);
                r
            })
            .collect()
    })
}

/// Maps a table error, turning cancellation into [`RunError::Cancelled`].
pub fn table_error(round: usize, table: usize, error: TableError) -> RunError {
    match error {
        TableError::Cancelled => RunError::Cancelled,
        error => RunError::Table {
            round,
            table,
            error,
        },
    }
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

/// Runs `spec.rounds` rounds for `bots` with the given starting ratings, with a terminal
/// progress bar unless `quiet`.
pub fn run(
    spec: &RunSpec,
    bots: &[BotEntry],
    initial: Vec<f64>,
    quiet: bool,
) -> Result<RunOutput, RunError> {
    let progress = if quiet {
        RunProgress::hidden()
    } else {
        let pb = ProgressBar::new(0);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner} round {msg} [{bar:40}] {pos}/{len} tables ({elapsed}, eta {eta})",
            )
            .unwrap()
            .progress_chars("=> "),
        );
        RunProgress::with_bar(pb)
    };
    run_with_progress(spec, bots, initial, &progress)
}

/// [`run`] reporting through a caller-provided [`RunProgress`].
pub fn run_with_progress(
    spec: &RunSpec,
    bots: &[BotEntry],
    initial: Vec<f64>,
    progress: &RunProgress,
) -> Result<RunOutput, RunError> {
    let keys = bots.iter().map(|b| b.key()).collect();
    let factories: Vec<BotFactory> = bots.iter().map(|b| b.factory.clone()).collect();
    let mut state = RunState::new(spec.clone(), keys, initial)?;
    let pool = build_pool(spec.threads)?;
    let schedules: Vec<Schedule> = (0..spec.rounds)
        .map(|r| state.schedule(r))
        .collect::<Result<_, _>>()?;
    let total: u64 = schedules.iter().map(|s| s.tables.len() as u64).sum();
    progress.tables_total.store(total, Ordering::Relaxed);
    progress
        .hands_total
        .store(total * spec.table.hands, Ordering::Relaxed);
    progress.bar.set_length(total);
    let mut rounds = Vec::with_capacity(spec.rounds);
    for (r, schedule) in schedules.into_iter().enumerate() {
        progress
            .bar
            .set_message(format!("{}/{}", r + 1, spec.rounds));
        let results = run_tables(spec, &factories, &schedule, r, &pool, progress)?;
        state.apply_round(&results);
        rounds.push(RoundRecord {
            round: r,
            schedule,
            results,
        });
        progress.rounds_done.fetch_add(1, Ordering::Relaxed);
    }
    progress.bar.finish_and_clear();
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
