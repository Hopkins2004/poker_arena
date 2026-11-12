//! Config files and end-to-end CLI smoke tests.

use arena::config::ConfigFile;
use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn shipped_configs_are_valid() {
    for name in ["6max_nlh", "hu_nlh", "6max_plo", "tournament_6max"] {
        let path = repo().join("configs").join(format!("{name}.toml"));
        let cfg = ConfigFile::load(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
        cfg.run_spec().unwrap_or_else(|e| panic!("{name}: {e}"));
        for b in &cfg.bots {
            bots::factory(&b.kind, &b.name, &b.params)
                .unwrap_or_else(|e| panic!("{name}/{}: {e}", b.name));
        }
    }
    let demo = ConfigFile::load(&repo().join("configs/6max_nlh.toml")).unwrap();
    assert_eq!(demo.bots.len(), 7);
    assert_eq!(demo.table.hands_per_table, 1000);
    assert_eq!(demo.run.rounds, 5);
}

#[test]
fn run_and_leaderboard_end_to_end() {
    let tmp = std::env::temp_dir().join(format!("arena_cli_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();
    let cfg = tmp.join("c.toml");
    std::fs::write(
        &cfg,
        r#"
[run]
rounds = 2
seed = 3
[format]
variant = "nlhe"
max_seats = 6
[table]
hands_per_table = 50
blinds = { sb = 1, bb = 2 }
hand_history = true
[[bots]]
name = "r"
kind = "random"
[[bots]]
name = "s"
kind = "calling_station"
[[bots]]
name = "t"
version = "1"
kind = "tag"
params = { samples = 50 }
"#,
    )
    .unwrap();
    let lb = tmp.join("lb.json");
    let res = tmp.join("results");
    let bin = env!("CARGO_BIN_EXE_arena");
    let run = |extra: &[&str]| {
        let out = Command::new(bin)
            .args(["run", "--config", cfg.to_str().unwrap(), "--quiet"])
            .args([
                "--leaderboard",
                lb.to_str().unwrap(),
                "--results-dir",
                res.to_str().unwrap(),
            ])
            .args(extra)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    };
    let first = run(&["--threads", "2"]);
    assert!(first.contains("t@1"), "{first}");
    let second = run(&["--rounds", "1"]);
    assert!(second.contains("hands"));
    let out = Command::new(bin)
        .args(["leaderboard", "--file", lb.to_str().unwrap()])
        .output()
        .unwrap();
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(
        text.contains("t@1") && text.contains("r") && text.contains("s"),
        "{text}"
    );
    let stored = arena::LeaderboardFile::load(&lb).unwrap();
    assert!(stored.entries.values().all(|e| e.runs == 2));
    // --fresh resets stored history for the bots in the run.
    run(&["--fresh"]);
    let stored = arena::LeaderboardFile::load(&lb).unwrap();
    assert!(stored.entries.values().all(|e| e.runs == 1));
    let dirs: Vec<_> = std::fs::read_dir(&res).unwrap().collect();
    assert!(!dirs.is_empty());
    let d = dirs[0].as_ref().unwrap().path();
    assert!(d.join("hand_histories").exists());

    // Python bots are rejected by the Rust binary with a helpful message.
    let py = tmp.join("py.toml");
    std::fs::write(
        &py,
        std::fs::read_to_string(&cfg)
            .unwrap()
            .replace("kind = \"calling_station\"", "kind = \"python:mybots.Bot\""),
    )
    .unwrap();
    let out = Command::new(bin)
        .args(["run", "--config", py.to_str().unwrap(), "--quiet"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("pokerarena.run"));
    let _ = std::fs::remove_dir_all(&tmp);
}
