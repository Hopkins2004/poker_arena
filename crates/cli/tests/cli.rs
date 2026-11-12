//! Shipped configs and CLI smoke tests.

use arena::config::ConfigFile;
use std::path::PathBuf;
use std::process::Command;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn shipped_configs_are_valid_and_python_only() {
    let mut n = 0;
    for entry in std::fs::read_dir(repo().join("configs")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "toml") {
            continue;
        }
        let cfg = ConfigFile::load(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        cfg.run_spec()
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        for b in &cfg.bots {
            assert!(
                b.python_target().is_some(),
                "{}: bot `{}` is not a Python bot",
                path.display(),
                b.name
            );
        }
        n += 1;
    }
    assert!(n >= 4);
    let demo = ConfigFile::load(&repo().join("configs/6max_nlh.toml")).unwrap();
    let anchors: Vec<(String, Option<f64>)> = demo
        .bots
        .iter()
        .map(|b| (b.name.clone(), b.anchor))
        .collect();
    for (name, a) in [("random", 850.0), ("station", 400.0), ("tag", 1300.0)] {
        assert!(
            anchors.contains(&(name.to_string(), Some(a))),
            "{name} anchored at {a}"
        );
    }
}

#[test]
fn bench_and_leaderboard_smoke() {
    let bin = env!("CARGO_BIN_EXE_arena");
    let out = Command::new(bin)
        .args(["bench", "--tables", "4", "--hands", "500", "--threads", "2"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(
        text.contains("evals/s") && text.contains("hands/s"),
        "{text}"
    );

    let tmp = std::env::temp_dir().join(format!("arena_cli_lb_{}.json", std::process::id()));
    let mut lb = arena::LeaderboardFile::default();
    lb.entries.insert(
        "tag".into(),
        arena::leaderboard::StoredEntry {
            rating: 1300.0,
            hands: 10,
            net_bb: 1.0,
            tables: 1,
            runs: 1,
            last_run: "r".into(),
        },
    );
    lb.save(&tmp).unwrap();
    let out = Command::new(bin)
        .args(["leaderboard", "--file", tmp.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(String::from_utf8_lossy(&out.stdout).contains("1300.0"));
    let _ = std::fs::remove_file(&tmp);
    // `run` no longer exists: bots are Python-only.
    let out = Command::new(bin)
        .args(["run", "--config", "x"])
        .output()
        .unwrap();
    assert!(!out.status.success());
}
