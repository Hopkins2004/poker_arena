//! Parallel runner: determinism across thread counts, external table execution, persistence.

use arena::runner::{run_table_external, table_seed};
use arena::{
    run, BotEntry, LeaderboardFile, MatchmakingConfig, RatingConfig, RunSpec, RunState, TableConfig,
};
use engine::GameFormat;
use serde_json::json;

fn entries() -> Vec<BotEntry> {
    let mk = |name: &str, kind: &str, params: serde_json::Value| BotEntry {
        name: name.into(),
        version: None,
        factory: bots::factory(kind, name, &params).unwrap(),
    };
    vec![
        mk("tag", "tag", json!({"samples": 60})),
        mk(
            "tag_loose",
            "tag",
            json!({"samples": 60, "tightness": -2.0}),
        ),
        mk("random", "random", json!(null)),
        mk(
            "random_aggro",
            "random",
            json!({"fold": 1, "call": 1, "raise": 2}),
        ),
        mk("station", "calling_station", json!(null)),
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
