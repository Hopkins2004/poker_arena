//! `arena`: run bot tournaments, benchmark the engine, show the leaderboard.

use arena::config::ConfigFile;
use arena::leaderboard::{render, utc_timestamp, write_results_dir};
use arena::runner::{build_pool, run_tables};
use arena::{run, BotEntry, LeaderboardFile, RunSpec, Schedule, TableConfig};
use clap::{Parser, Subcommand, ValueEnum};
use engine::{Card, Deck, Evaluator, GameFormat};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "arena", version, about = "Poker bot arena")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a configured set of rounds and update the leaderboard.
    Run {
        /// TOML config file.
        #[arg(long)]
        config: PathBuf,
        /// Override the master seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Override the number of worker threads.
        #[arg(long)]
        threads: Option<usize>,
        /// Override the number of rounds.
        #[arg(long)]
        rounds: Option<usize>,
        /// Ignore stored ratings and start every bot from the initial rating.
        #[arg(long)]
        fresh: bool,
        /// No progress bar.
        #[arg(long)]
        quiet: bool,
        /// Override the leaderboard file.
        #[arg(long)]
        leaderboard: Option<PathBuf>,
        /// Override the results directory.
        #[arg(long)]
        results_dir: Option<PathBuf>,
    },
    /// Benchmark evaluator and full-hand throughput at 1 thread and at all threads.
    Bench {
        /// Tables per measurement.
        #[arg(long, default_value_t = 96)]
        tables: usize,
        /// Hands per table.
        #[arg(long, default_value_t = 10_000)]
        hands: u64,
        /// Game variant.
        #[arg(long, value_enum, default_value_t = BenchVariant::Nlhe)]
        variant: BenchVariant,
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

#[derive(Clone, Copy, ValueEnum)]
enum BenchVariant {
    Nlhe,
    Plo,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let r = match cli.cmd {
        Cmd::Run {
            config,
            seed,
            threads,
            rounds,
            fresh,
            quiet,
            leaderboard,
            results_dir,
        } => cmd_run(RunArgs {
            config,
            seed,
            threads,
            rounds,
            fresh,
            quiet,
            leaderboard,
            results_dir,
        }),
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

struct RunArgs {
    config: PathBuf,
    seed: Option<u64>,
    threads: Option<usize>,
    rounds: Option<usize>,
    fresh: bool,
    quiet: bool,
    leaderboard: Option<PathBuf>,
    results_dir: Option<PathBuf>,
}

fn cmd_run(a: RunArgs) -> Result<(), String> {
    let text = std::fs::read_to_string(&a.config)
        .map_err(|e| format!("cannot read {}: {e}", a.config.display()))?;
    let mut cfg = ConfigFile::parse(&text)?;
    let mut overrides = Vec::new();
    if let Some(s) = a.seed {
        cfg.run.seed = s;
        overrides.push(format!("seed = {s}"));
    }
    if let Some(t) = a.threads {
        cfg.run.threads = Some(t);
        overrides.push(format!("threads = {t}"));
    }
    if let Some(r) = a.rounds {
        cfg.run.rounds = r;
        overrides.push(format!("rounds = {r}"));
    }
    let spec = cfg.run_spec()?;

    let mut entries = Vec::with_capacity(cfg.bots.len());
    for b in &cfg.bots {
        if let Some(target) = b.python_target() {
            return Err(format!(
                "bot `{}` is a Python bot ({target}); the `arena` binary runs Rust bots only. \
                 Use `pokerarena.run(config)` from Python for mixed runs.",
                b.name
            ));
        }
        let factory = bots::factory(&b.kind, &b.name, &b.params)
            .map_err(|e| format!("bot `{}`: {e}", b.name))?;
        entries.push(BotEntry {
            name: b.name.clone(),
            version: b.version.clone(),
            factory,
        });
    }
    let keys: Vec<String> = entries.iter().map(|e| e.key()).collect();

    let lb_path = a
        .leaderboard
        .unwrap_or_else(|| PathBuf::from(&cfg.run.leaderboard));
    let mut lb =
        LeaderboardFile::load(&lb_path).map_err(|e| format!("{}: {e}", lb_path.display()))?;
    if a.fresh {
        for k in &keys {
            lb.entries.remove(k);
        }
    }
    let initial = lb.initial_ratings(&keys, spec.rating.initial);

    let t0 = Instant::now();
    let out = run(&spec, &entries, initial, a.quiet).map_err(|e| e.to_string())?;
    let elapsed = t0.elapsed().as_secs_f64();
    let rows = out.state.rows();
    let hands: u64 = out
        .rounds
        .iter()
        .flat_map(|r| &r.results)
        .map(|t| t.hands_played)
        .sum();
    let tables: usize = out.rounds.iter().map(|r| r.results.len()).sum();

    let run_id = format!("{}-s{}", utc_timestamp(), spec.seed);
    let mut cfg_text = text;
    if !overrides.is_empty() {
        cfg_text.push_str(&format!("\n# CLI overrides: {}\n", overrides.join(", ")));
    }
    let results_base = a
        .results_dir
        .unwrap_or_else(|| PathBuf::from(&cfg.run.results_dir));
    let dir = write_results_dir(&results_base, &run_id, &cfg_text, &keys, &out.rounds, &rows)
        .map_err(|e| format!("writing results: {e}"))?;
    lb.update(&rows, spec.table.reference_bb(), &run_id);
    lb.save(&lb_path)
        .map_err(|e| format!("{}: {e}", lb_path.display()))?;

    println!(
        "{} rounds, {tables} tables, {hands} hands in {elapsed:.2}s ({:.0} hands/s)\n",
        spec.rounds,
        hands as f64 / elapsed
    );
    print!("{}", render(&rows));
    println!(
        "\nresults: {}\nleaderboard: {}",
        dir.display(),
        lb_path.display()
    );
    Ok(())
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
    variant: BenchVariant,
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
    let (format, label) = match variant {
        BenchVariant::Nlhe => (GameFormat::nlhe(6), "6-max NLHE"),
        BenchVariant::Plo => (GameFormat::plo(6), "6-max PLO"),
    };
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
    };
    let factories = vec![bots::factory("random", "random", &serde_json::Value::Null)?];
    let schedule = Schedule {
        tables: vec![vec![0; 6]; tables],
    };
    let bar = arena::runner::ProgressBar::hidden();
    let measure = |pool: &rayon::ThreadPool| -> Result<f64, String> {
        let t = Instant::now();
        let res =
            run_tables(&spec, &factories, &schedule, 0, pool, &bar).map_err(|e| e.to_string())?;
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
