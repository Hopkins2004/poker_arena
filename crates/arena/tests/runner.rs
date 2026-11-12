//! Parallel runner: determinism across thread counts, external table execution, persistence.

mod common;

use arena::runner::{run_table_external, table_seed};
use arena::{
    run, BotEntry, LeaderboardFile, MatchmakingConfig, RatingConfig, RunSpec, RunState, TableConfig,
};
use engine::GameFormat;

fn entries() -> Vec<BotEntry> {
    let mk = |name: &str, factory| BotEntry {
        name: name.into(),
        version: None,
        factory,
        anchor: None,
    };
    vec![
        mk("raiser", common::raiser("raiser")),
        mk(
            "random_loose",
            common::random_w("random_loose", 0.5, 2.0, 1.0),
        ),
        mk("random", common::random("random")),
        mk(
            "random_aggro",
            common::random_w("random_aggro", 1.0, 1.0, 2.0),
        ),
        mk("station", common::station("station")),
    ]
}

fn spec(threads: usize) -> RunSpec {
    RunSpec {
        format: GameFormat::nlhe(6),
        table: TableConfig::cash(100, 1, 2, 100),
        matchmaking: MatchmakingConfig::default(),
        rating: RatingConfig::default(),
        rounds: 3,
        seed: 1234,
        threads: Some(threads),
        replay_hands_per_bot: 0,
    }
}

fn serialized(threads: usize) -> String {
    let bots = entries();
    let out = run(&spec(threads), &bots, vec![1500.0; bots.len()], true).unwrap();
    let rounds = serde_json::to_string(&out.rounds).unwrap();
    let rows = serde_json::to_string(&out.state.rows()).unwrap();
    format!("{rounds}\n{rows}\n{:?}", out.state.ratings())
}

#[test]
fn identical_output_for_1_and_8_threads() {
    let a = serialized(1);
    let b = serialized(8);
    assert_eq!(
        a, b,
        "run must be bit-for-bit identical across thread counts"
    );
    assert_eq!(a, serialized(3));
}

#[test]
fn run_produces_ratings_and_balanced_appearances() {
    let bots = entries();
    let out = run(&spec(4), &bots, vec![1500.0; bots.len()], true).unwrap();
    assert_eq!(out.rounds.len(), 3);
    for r in &out.rounds {
        // 5 bots, 6 seats, duplicates allowed: T = max(5, ceil(4*5/6)) = 5 tables.
        assert_eq!(r.results.len(), 5);
        let app = r.schedule.appearances(5);
        assert_eq!(app, vec![6; 5]);
    }
    let rows = out.state.rows();
    assert_eq!(rows.len(), 5);
    let total: f64 = out.state.ratings().iter().sum();
    assert!(
        (total - 7500.0).abs() < 1e-6,
        "pairwise Elo is zero-sum: {total}"
    );
    assert!(rows.iter().all(|r| r.hands == 3 * 6 * 100));
    assert!(rows.windows(2).all(|w| w[0].rating >= w[1].rating));
    assert!(rows.iter().all(|r| r.ci95.is_some()));
}

#[test]
fn external_table_execution_matches_run() {
    // Tables executed one by one outside the pool (as the Python driver does) and merged in
    // table-id order give exactly the same state.
    let bots = entries();
    let keys: Vec<String> = bots.iter().map(|b| b.key()).collect();
    let factories: Vec<_> = bots.iter().map(|b| b.factory.clone()).collect();
    let sp = spec(2);
    let mut st = RunState::new(sp.clone(), keys, vec![1500.0; 5]).unwrap();
    for round in 0..sp.rounds {
        let sched = st.schedule(round).unwrap();
        // Run in reverse order, then sort back into table-id order before merging.
        let mut res: Vec<(usize, _)> = (0..sched.tables.len())
            .rev()
            .map(|t| {
                assert_eq!(st.table_seed(round, t), table_seed(sp.seed, round, t));
                (
                    t,
                    run_table_external(&sp, &factories, &sched, round, t).unwrap(),
                )
            })
            .collect();
        res.sort_by_key(|x| x.0);
        let res: Vec<_> = res.into_iter().map(|x| x.1).collect();
        st.apply_round(&res);
    }
    let out = run(&sp, &bots, vec![1500.0; 5], true).unwrap();
    assert_eq!(st.ratings(), out.state.ratings());
    assert_eq!(st.rows(), out.state.rows());
}

#[test]
fn leaderboard_persistence_roundtrip() {
    let dir = std::env::temp_dir().join(format!("arena_lb_{}", std::process::id()));
    let path = dir.join("leaderboard.json");
    let _ = std::fs::remove_dir_all(&dir);
    let bots = entries();
    let keys: Vec<String> = bots.iter().map(|b| b.key()).collect();

    let lb = LeaderboardFile::load(&path).unwrap();
    assert!(lb.entries.is_empty());
    let init = lb.initial_ratings(&keys, 1500.0);
    let out = run(&spec(2), &bots, init, true).unwrap();
    let mut lb = lb;
    lb.update(&out.state.rows(), 2, "run1");
    lb.save(&path).unwrap();

    let lb2 = LeaderboardFile::load(&path).unwrap();
    assert_eq!(lb, lb2);
    let init2 = lb2.initial_ratings(&keys, 1500.0);
    assert_eq!(init2, out.state.ratings());
    // A second run continues from the stored ratings.
    let out2 = run(&spec(2), &bots, init2.clone(), true).unwrap();
    let rows2 = out2.state.rows();
    for r in &rows2 {
        let b = keys.iter().position(|k| k == &r.key).unwrap();
        assert!((r.rating - r.rating_change - init2[b]).abs() < 1e-9);
    }
    let mut lb3 = lb2;
    lb3.update(&rows2, 2, "run2");
    assert!(lb3
        .entries
        .values()
        .all(|e| e.runs == 2 && e.hands == 2 * 3 * 6 * 100));
    assert!(lb3.render().contains("run2"));

    let rdir = arena::leaderboard::write_results_dir(
        &dir.join("results"),
        "test_run",
        "# config\n",
        &keys,
        &out2.rounds,
        &rows2,
    )
    .unwrap();
    for f in [
        "config.toml",
        "schedule.json",
        "tables.jsonl",
        "leaderboard.json",
        "leaderboard.txt",
    ] {
        assert!(rdir.join(f).exists(), "{f}");
    }
    let lines = std::fs::read_to_string(rdir.join("tables.jsonl")).unwrap();
    assert_eq!(lines.lines().count(), 15);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn utc_timestamp_format() {
    let t = arena::leaderboard::utc_timestamp();
    assert_eq!(t.len(), 16, "{t}");
    assert!(t.starts_with("20") && t.ends_with('Z') && t.as_bytes()[8] == b'T');
}

#[test]
fn cancellation_stops_promptly_and_cleanly() {
    use arena::runner::{run_with_progress, RunError, RunProgress};
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    let bots = entries();
    let mut sp = spec(4);
    sp.table.hands = 2_000_000; // would take far longer than the test allows
    let progress = RunProgress::hidden();
    let p2 = progress.clone();
    let worker = std::thread::spawn(move || run_with_progress(&sp, &bots, vec![1500.0; 5], &p2));
    let t0 = Instant::now();
    while progress.hands_done.load(Ordering::Relaxed) < 1000 {
        assert!(t0.elapsed() < Duration::from_secs(20), "run never started");
        std::thread::sleep(Duration::from_millis(1));
    }
    let t_cancel = Instant::now();
    progress.request_cancel();
    let res = worker.join().unwrap();
    let stop_latency = t_cancel.elapsed();
    assert!(
        matches!(res, Err(RunError::Cancelled)),
        "expected Cancelled"
    );
    assert!(
        stop_latency < Duration::from_millis(500),
        "stop took {stop_latency:?}"
    );
    // No further hands after the run returned.
    let done = progress.hands_done.load(Ordering::Relaxed);
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(done, progress.hands_done.load(Ordering::Relaxed));

    // Cancelled before starting: nothing is played.
    let pre = RunProgress::hidden();
    pre.request_cancel();
    let res = run_with_progress(&spec(2), &entries(), vec![1500.0; 5], &pre);
    assert!(matches!(res, Err(RunError::Cancelled)));
    assert_eq!(pre.hands_done.load(Ordering::Relaxed), 0);
}

#[test]
fn hooks_do_not_change_results() {
    use arena::{run_table, run_table_with_hooks, TableConfig, TableHooks};
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    let f: Vec<_> = entries().into_iter().map(|e| e.factory).collect();
    let fmt = GameFormat::nlhe(6);
    let cfg = TableConfig::cash(300, 1, 2, 100);
    let plain = run_table(&fmt, &[0, 1, 2, 3, 4, 0], &f, &cfg, 9).unwrap();
    let hands = AtomicU64::new(0);
    let cancel = AtomicBool::new(false);
    let calls = AtomicU64::new(0);
    let on_hand = || {
        calls.fetch_add(1, Ordering::Relaxed);
    };
    let hooked = run_table_with_hooks(
        &fmt,
        &[0, 1, 2, 3, 4, 0],
        &f,
        &cfg,
        9,
        TableHooks {
            hands_done: Some(&hands),
            cancel: Some(&cancel),
            on_hand: Some(&on_hand),
            record: None,
        },
    )
    .unwrap();
    assert_eq!(plain, hooked);
    assert_eq!(hands.load(Ordering::Relaxed), 300);
    assert_eq!(calls.load(Ordering::Relaxed), 300);
}

#[test]
fn anchored_bots_keep_their_rating() {
    let mut bots = entries();
    bots[2].anchor = Some(850.0); // random
    bots[4].anchor = Some(400.0); // station
    let out = run(&spec(4), &bots, vec![1500.0; 5], true).unwrap();
    assert_eq!(out.state.ratings()[2], 850.0);
    assert_eq!(out.state.ratings()[4], 400.0);
    let rows = out.state.rows();
    for r in &rows {
        assert_eq!(r.anchored, r.key == "random" || r.key == "station");
        if r.anchored {
            assert_eq!(r.rating_change, 0.0);
        }
    }
    // The others moved relative to the anchors.
    assert!(out.state.ratings()[0] != 1500.0);
    let mut lb = LeaderboardFile::default();
    lb.update(&rows, 2, "r1");
    assert_eq!(lb.entries["station"].rating, 400.0);
    // Anchors override stored ratings on the next run.
    let mut bots2 = entries();
    bots2[4].anchor = Some(400.0);
    let keys: Vec<String> = bots2.iter().map(|b| b.key()).collect();
    let mut init = lb.initial_ratings(&keys, 1500.0);
    init[4] = 999.0;
    let out2 = run(&spec(2), &bots2, init, true).unwrap();
    assert_eq!(out2.state.ratings()[4], 400.0);
}

#[test]
fn hand_replays_sample_record_and_write() {
    use arena::replays::write_replays;
    let bots = entries();
    let mut sp = spec(3);
    sp.replay_hands_per_bot = 20;
    let out = run(&sp, &bots, vec![1500.0; 5], true).unwrap();
    let plan = &out.replay_plan;
    for (b, picks) in plan.per_bot.iter().enumerate() {
        assert_eq!(picks.len(), 20, "bot {b}");
        for &(r, t, h) in picks {
            let res = &out.rounds[r].results[t];
            assert!(
                res.seats.iter().any(|s| s.bot == b),
                "sampled a table without the bot"
            );
            let rec = res
                .recorded_hands
                .iter()
                .find(|x| x.hand_id == h)
                .expect("hand recorded");
            assert!(
                rec.text.contains(&bots[b].name),
                "bot not in its sampled hand"
            );
        }
    }
    // Only sampled hands are recorded.
    let recorded: usize = out
        .rounds
        .iter()
        .flat_map(|r| &r.results)
        .map(|t| t.recorded_hands.len())
        .sum();
    let planned: usize = plan.per_table.values().map(|v| v.len()).sum();
    assert_eq!(recorded, planned);
    // Recording does not change results, and is identical across thread counts.
    let plain = run(&spec(3), &bots, vec![1500.0; 5], true).unwrap();
    assert_eq!(plain.state.rows(), out.state.rows());
    let mut sp8 = sp.clone();
    sp8.threads = Some(8);
    let out8 = run(&sp8, &bots, vec![1500.0; 5], true).unwrap();
    assert_eq!(
        serde_json::to_string(&out8.rounds).unwrap(),
        serde_json::to_string(&out.rounds).unwrap()
    );

    let dir = std::env::temp_dir().join(format!("arena_replays_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let keys: Vec<String> = bots.iter().map(|b| b.key()).collect();
    let d = write_replays(&dir, "run1", &keys, plan, &out.rounds).unwrap();
    for k in &keys {
        let text = std::fs::read_to_string(d.join(format!("{k}.txt"))).unwrap();
        assert_eq!(text.matches("### Round").count(), 20, "{k}");
        assert_eq!(text.matches("Hand #").count(), 20, "{k}");
    }
    assert!(!dir.join(".run1.partial").exists());
    let _ = std::fs::remove_dir_all(&dir);
}
