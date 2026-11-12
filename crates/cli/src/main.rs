//! `arena`: benchmark the engine and show the persisted leaderboard.
//!
//! Runs with bots go through the Poker Arena desktop app or the Python package
//! (`pokerarena.run`): all bots are Python bots. The benchmark drives tables with a small
//! built-in random policy that exists only to measure engine and runner throughput.

use arena::bot::{Action, Bot, BotFactory, Observation};
use arena::runner::{build_pool, run_tables};
use arena::{LeaderboardFile, RunSpec, Schedule, TableConfig};
use clap::{Parser, Subcommand};
use engine::rng::{below, unit_f64};
use engine::{Card, Deck, Evaluator};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "arena", version, about = "Poker bot arena")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Benchmark evaluator and full-hand throughput at 1 thread and at all threads.
    Bench {
        /// Tables per measurement.
        #[arg(long, default_value_t = 96)]
        tables: usize,
        /// Hands per table.
        #[arg(long, default_value_t = 10_000)]
        hands: u64,
        /// Format name from the engine's rules registry (nlhe, plo, flhe, ...).
        #[arg(long, default_value = "nlhe")]
        variant: String,
        /// Enable hand-history writing (to measure its cost).
        #[arg(long)]
        hand_history: bool,
        /// Thread count for the parallel measurement (default: all cores).
        #[arg(long)]
        threads: Option<usize>,
    },
    /// Print the persisted leaderboard.
    Leaderboard {
        /// Leaderboard file.
        #[arg(long, default_value = "leaderboard.json")]
        file: PathBuf,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let r = match cli.cmd {
        Cmd::Bench {
            tables,
            hands,
            variant,
            hand_history,
            threads,
        } => cmd_bench(tables, hands, variant, hand_history, threads),
        Cmd::Leaderboard { file } => cmd_leaderboard(&file),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_leaderboard(file: &std::path::Path) -> Result<(), String> {
    let lb = LeaderboardFile::load(file).map_err(|e| format!("{}: {e}", file.display()))?;
    if lb.entries.is_empty() {
        println!("(no entries in {})", file.display());
    } else {
        print!("{}", lb.render());
    }
    Ok(())
}

fn cmd_bench(
    tables: usize,
    hands: u64,
    variant: String,
    hand_history: bool,
    threads: Option<usize>,
) -> Result<(), String> {
    let all = threads.unwrap_or_else(|| {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    });
    let pool1 = build_pool(Some(1)).map_err(|e| e.to_string())?;
    let pool_n = build_pool(Some(all)).map_err(|e| e.to_string())?;

    // --- Evaluator -------------------------------------------------------------------
    let ev = Evaluator::get();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
    let mut deck = Deck::new();
    let n_hands = 1 << 22;
    let hands7: Vec<[Card; 7]> = (0..n_hands)
        .map(|_| {
            deck.shuffle_prefix(&mut rng, 7);
            deck.cards()[..7].try_into().unwrap()
        })
        .collect();
    let eval_rate = |pool: &rayon::ThreadPool, reps: usize| {
        let t = Instant::now();
        let sum: u64 = pool.install(|| {
            (0..reps)
                .into_par_iter()
                .map(|_| {
                    hands7
                        .par_chunks(1 << 14)
                        .map(|c| {
                            c.iter()
                                .map(|h| ev.eval7(std::hint::black_box(h)) as u64)
                                .sum::<u64>()
                        })
                        .sum::<u64>()
                })
                .sum()
        });
        std::hint::black_box(sum);
        (reps * n_hands) as f64 / t.elapsed().as_secs_f64()
    };
    eval_rate(&pool1, 1); // warm up
    let e1 = eval_rate(&pool1, 8);
    let en = eval_rate(&pool_n, 8 * all);
    println!("evaluator (random 7-card hands)");
    println!("  1 thread : {:>8.1} M evals/s", e1 / 1e6);
    println!(
        "  {all} threads: {:>8.1} M evals/s  (speedup {:.2}x, efficiency {:.0}%)",
        en / 1e6,
        en / e1,
        100.0 * en / e1 / all as f64
    );

    // --- Full hands ------------------------------------------------------------------
    let rules = engine::rules::by_name(&variant).ok_or_else(|| {
        let known: Vec<&str> = engine::rules::ALL.iter().map(|r| r.name()).collect();
        format!("unknown variant `{variant}` (known: {})", known.join(", "))
    })?;
    let format = rules.default_format(6);
    let label = format!("6-max {}", rules.title());
    let mut table = TableConfig::cash(hands, 1, 2, 100);
    table.hand_history = hand_history;
    let spec = RunSpec {
        format,
        table,
        matchmaking: Default::default(),
        rating: Default::default(),
        rounds: 1,
        seed: 42,
        threads: None,
        replay_hands_per_bot: 0,
    };
    let factories: Vec<BotFactory> = vec![Arc::new(|seed| Box::new(BenchPolicy::new(seed)))];
    let schedule = Schedule {
        tables: vec![vec![0; 6]; tables],
    };
    let bar = arena::RunProgress::hidden();
    let measure = |pool: &rayon::ThreadPool| -> Result<f64, String> {
        let t = Instant::now();
        let res = run_tables(&spec, &factories, &schedule, 0, pool, &bar, None)
            .map_err(|e| e.to_string())?;
        let played: u64 = res.iter().map(|r| r.hands_played).sum();
        Ok(played as f64 / t.elapsed().as_secs_f64())
    };
    let h1 = measure(&pool1)?;
    let hn = measure(&pool_n)?;
    println!(
        "\n{label}, random-legal bots, 100bb, {tables} tables x {hands} hands{}",
        if hand_history {
            ", hand history ON"
        } else {
            ""
        }
    );
    println!("  1 thread : {:>10.0} hands/s", h1);
    println!(
        "  {all} threads: {:>10.0} hands/s  (speedup {:.2}x, efficiency {:.0}%)",
        hn,
        hn / h1,
        100.0 * hn / h1 / all as f64
    );
    Ok(())
}

/// Benchmark-only random-legal policy (fold 25% / call 50% / raise 25% to 0.5 or 1 pot when
/// facing a bet; check or bet otherwise). Not a selectable bot.
struct BenchPolicy {
    rng: Xoshiro256PlusPlus,
}

impl BenchPolicy {
    fn new(seed: u64) -> Self {
        BenchPolicy {
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
}

impl Bot for BenchPolicy {
    fn name(&self) -> &str {
        "bench"
    }
    fn act(&mut self, obs: &Observation) -> Action {
        let la = obs.legal;
        let u = unit_f64(&mut self.rng);
        if u < 0.75 || !la.can_raise {
            return if la.can_check {
                Action::Check
            } else if u < 0.25 {
                Action::Fold
            } else {
                Action::Call
            };
        }
        let frac = [0.5, 1.0][below(&mut self.rng, 2) as usize];
        let to = obs.current_bet + ((obs.pot + la.call_amount) as f64 * frac) as u64;
        Action::RaiseTo(to.clamp(la.min_raise_to, la.max_raise_to))
    }
}
