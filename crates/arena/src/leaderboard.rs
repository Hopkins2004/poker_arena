//! Leaderboard rows, persistence (`leaderboard.json`), text rendering and the per-run
//! results directory.

use crate::runner::RoundRecord;
use crate::stats::SeatStats;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// One leaderboard row for a run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LeaderboardRow {
    /// Bot key (`name` or `name@version`).
    pub key: String,
    /// Elo after the run.
    pub rating: f64,
    /// Elo change during the run.
    pub rating_change: f64,
    /// Hands played.
    pub hands: u64,
    /// Net chips.
    pub net_chips: i64,
    /// Hand-weighted bb/100.
    pub bb_per_100: f64,
    /// Half-width of the 95% CI of bb/100, from the variance of per-table bb/100 (the bot's
    /// seats at a table pooled; `1.96·sd/√n`); `None` with fewer than two samples.
    pub ci95: Option<f64>,
    /// Tables played.
    pub tables: usize,
    /// VPIP rate.
    pub vpip: f64,
    /// PFR rate.
    pub pfr: f64,
    /// Showdown rate (showdowns / hands).
    pub showdown: f64,
    /// Illegal actions.
    pub illegal: u64,
    /// Mean decision time in µs (0 unless timing was recorded).
    pub mean_decision_us: f64,
    /// Mean finishing place (tournaments only).
    pub avg_place: Option<f64>,
}

impl LeaderboardRow {
    pub(crate) fn build(
        key: &str,
        rating: f64,
        initial: f64,
        st: &SeatStats,
        samples: &[f64],
        places: &[u32],
        bb: u64,
    ) -> LeaderboardRow {
        let n = samples.len();
        let ci95 = (n >= 2).then(|| {
            let mean = samples.iter().sum::<f64>() / n as f64;
            let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            1.96 * (var / n as f64).sqrt()
        });
        LeaderboardRow {
            key: key.to_string(),
            rating,
            rating_change: rating - initial,
            hands: st.hands,
            net_chips: st.net,
            bb_per_100: st.bb_per_100(bb),
            ci95,
            tables: n,
            vpip: st.vpip_rate(),
            pfr: st.pfr_rate(),
            showdown: st.showdown_rate(),
            illegal: st.illegal,
            mean_decision_us: st.mean_decision_us(),
            avg_place: (!places.is_empty())
                .then(|| places.iter().map(|&p| p as f64).sum::<f64>() / places.len() as f64),
        }
    }
}

/// Renders rows as an aligned text table.
pub fn render(rows: &[LeaderboardRow]) -> String {
    let w = rows.iter().map(|r| r.key.len()).max().unwrap_or(3).max(3);
    let tournament = rows.iter().any(|r| r.avg_place.is_some());
    let mut o = String::new();
    let _ = write!(
        o,
        "{:>3}  {:<w$}  {:>7}  {:>7}  {:>10}  {:>16}  {:>6}  {:>6}  {:>6}  {:>7}  {:>9}",
        "#",
        "bot",
        "elo",
        "Δelo",
        "hands",
        "bb/100 (95% CI)",
        "vpip",
        "pfr",
        "sd",
        "illegal",
        "µs/dec"
    );
    if tournament {
        let _ = write!(o, "  {:>6}", "place");
    }
    o.push('\n');
    for (i, r) in rows.iter().enumerate() {
        let ci = match r.ci95 {
            Some(c) => format!("{:+.1} ± {:.1}", r.bb_per_100, c),
            None => format!("{:+.1}", r.bb_per_100),
        };
        let _ = write!(
            o,
            "{:>3}  {:<w$}  {:>7.1}  {:>+7.1}  {:>10}  {:>16}  {:>5.1}%  {:>5.1}%  {:>5.1}%  {:>7}  {:>9.2}",
            i + 1,
            r.key,
            r.rating,
            r.rating_change,
            r.hands,
            ci,
            100.0 * r.vpip,
            100.0 * r.pfr,
            100.0 * r.showdown,
            r.illegal,
            r.mean_decision_us
        );
        if let Some(p) = r.avg_place {
            let _ = write!(o, "  {:>6.2}", p);
        }
        o.push('\n');
    }
    o
}

/// A persisted leaderboard entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredEntry {
    /// Current Elo.
    pub rating: f64,
    /// Cumulative hands over all runs.
    pub hands: u64,
    /// Cumulative net result in big blinds over all runs.
    pub net_bb: f64,
    /// Cumulative rated tables.
    #[serde(alias = "seats")]
    pub tables: u64,
    /// Number of runs.
    pub runs: u64,
    /// Run id of the last update.
    pub last_run: String,
}

/// The persisted leaderboard, keyed by bot key.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LeaderboardFile {
    /// Entries by bot key.
    pub entries: BTreeMap<String, StoredEntry>,
}

impl LeaderboardFile {
    /// Loads a leaderboard; a missing file yields an empty one.
    pub fn load(path: &Path) -> std::io::Result<LeaderboardFile> {
        match std::fs::read_to_string(path) {
            Ok(s) => serde_json::from_str(&s)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(LeaderboardFile::default()),
            Err(e) => Err(e),
        }
    }

    /// Writes the leaderboard as pretty JSON.
    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)?;
        }
        // Write-then-rename so an interrupted save never leaves a truncated file.
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp = PathBuf::from(tmp);
        std::fs::write(&tmp, serde_json::to_string_pretty(self).unwrap() + "\n")?;
        std::fs::rename(&tmp, path)
    }

    /// Starting ratings for `keys` (stored rating, or `default` for unseen bots).
    pub fn initial_ratings(&self, keys: &[String], default: f64) -> Vec<f64> {
        keys.iter()
            .map(|k| self.entries.get(k).map_or(default, |e| e.rating))
            .collect()
    }

    /// Records a run's rows. `bb` is the reference big blind for `net_bb`.
    pub fn update(&mut self, rows: &[LeaderboardRow], bb: u64, run_id: &str) {
        for r in rows {
            let e = self.entries.entry(r.key.clone()).or_insert(StoredEntry {
                rating: r.rating,
                hands: 0,
                net_bb: 0.0,
                tables: 0,
                runs: 0,
                last_run: String::new(),
            });
            e.rating = r.rating;
            e.hands += r.hands;
            e.net_bb += r.net_chips as f64 / bb as f64;
            e.tables += r.tables as u64;
            e.runs += 1;
            e.last_run = run_id.to_string();
        }
    }

    /// Rows for display (rating order), without run-specific fields.
    pub fn render(&self) -> String {
        let mut v: Vec<(&String, &StoredEntry)> = self.entries.iter().collect();
        v.sort_by(|a, b| b.1.rating.total_cmp(&a.1.rating).then(a.0.cmp(b.0)));
        let w = v.iter().map(|e| e.0.len()).max().unwrap_or(3).max(3);
        let mut o = format!(
            "{:>3}  {:<w$}  {:>7}  {:>10}  {:>8}  {:>5}  last run\n",
            "#", "bot", "elo", "hands", "bb/100", "runs"
        );
        for (i, (k, e)) in v.iter().enumerate() {
            let bb100 = if e.hands > 0 {
                e.net_bb / e.hands as f64 * 100.0
            } else {
                0.0
            };
            let _ = writeln!(
                o,
                "{:>3}  {:<w$}  {:>7.1}  {:>10}  {:>+8.2}  {:>5}  {}",
                i + 1,
                k,
                e.rating,
                e.hands,
                bb100,
                e.runs,
                e.last_run
            );
        }
        o
    }
}

/// A UTC timestamp like `20261112T153000Z` (no external date crate needed).
pub fn utc_timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!(
        "{y:04}{m:02}{d:02}T{:02}{:02}{:02}Z",
        rem / 3600,
        (rem / 60) % 60,
        rem % 60
    )
}

/// Writes `results/<run_id>/`: the config text, all schedules, per-table results (JSON
/// lines), hand histories (if any), and the leaderboard (JSON and text).
pub fn write_results_dir(
    base: &Path,
    run_id: &str,
    config_text: &str,
    keys: &[String],
    rounds: &[RoundRecord],
    rows: &[LeaderboardRow],
) -> std::io::Result<PathBuf> {
    // Build the directory under a hidden temporary name and rename it into place at the
    // end, so an interrupted write never leaves a half-written results directory.
    let final_dir = base.join(run_id);
    let dir = base.join(format!(".{run_id}.partial"));
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    std::fs::create_dir_all(&dir)?;
    std::fs::write(dir.join("config.toml"), config_text)?;

    let schedules: Vec<serde_json::Value> = rounds
        .iter()
        .map(|r| {
            serde_json::json!({
                "round": r.round,
                "tables": r.schedule.tables.iter()
                    .map(|t| t.iter().map(|&b| keys[b].clone()).collect::<Vec<_>>())
                    .collect::<Vec<_>>(),
            })
        })
        .collect();
    std::fs::write(
        dir.join("schedule.json"),
        serde_json::to_string_pretty(&schedules).unwrap() + "\n",
    )?;

    let mut lines = String::new();
    for r in rounds {
        for (t, res) in r.results.iter().enumerate() {
            let mut res = res.clone();
            if let Some(hh) = res.hand_history.take() {
                let hh_dir = dir.join("hand_histories");
                std::fs::create_dir_all(&hh_dir)?;
                std::fs::write(
                    hh_dir.join(format!("round{:03}_table{:04}.txt", r.round, t)),
                    hh,
                )?;
            }
            let line = serde_json::json!({
                "round": r.round,
                "table": t,
                "bots": r.schedule.tables[t].iter().map(|&b| keys[b].clone()).collect::<Vec<_>>(),
                "result": res,
            });
            lines.push_str(&line.to_string());
            lines.push('\n');
        }
    }
    std::fs::write(dir.join("tables.jsonl"), lines)?;
    std::fs::write(
        dir.join("leaderboard.json"),
        serde_json::to_string_pretty(rows).unwrap() + "\n",
    )?;
    std::fs::write(dir.join("leaderboard.txt"), render(rows))?;
    std::fs::rename(&dir, &final_dir)?;
    Ok(final_dir)
}
