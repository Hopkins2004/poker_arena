//! Hand replays: a random sample of hands per bot, written as text hand histories.
//!
//! For every bot, `per_bot` hands are drawn uniformly without replacement from all
//! `(round, table, hand)` slots where that bot is seated. The sample depends only on the
//! master seed and the schedules, so every process (the parent and Python workers)
//! computes the same plan. Only sampled hands are recorded (see
//! [`crate::table::TableHooks::record`]), so replays cost no measurable throughput.
//!
//! Tournament tables can end early, so a sampled hand may never be played; such hands are
//! simply missing from the replay file.

use crate::matchmaking::Schedule;
use crate::runner::RoundRecord;
use engine::rng::mix_seed;
use rand_xoshiro::rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const STREAM_REPLAYS: u64 = 0x05EE_D4E9_1A75;

/// Which hands to record and whose replay file each one goes to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ReplayPlan {
    /// Per bot (index into the bot list): sampled `(round, table, hand_id)`, sorted.
    pub per_bot: Vec<Vec<(usize, usize, u64)>>,
    /// Per `(round, table)`: sorted hand ids to record (union over bots).
    pub per_table: BTreeMap<(usize, usize), Vec<u64>>,
}

impl ReplayPlan {
    /// Hand ids to record at a table (empty if none).
    pub fn record_for(&self, round: usize, table: usize) -> &[u64] {
        self.per_table
            .get(&(round, table))
            .map_or(&[][..], |v| v.as_slice())
    }

    /// True if nothing is to be recorded.
    pub fn is_empty(&self) -> bool {
        self.per_table.is_empty()
    }
}

/// Number of hand ids a table produces: duplicate mode rounds the hand count down to a
/// multiple of the seat count (at least one deal).
pub fn hands_at_table(hands_per_table: u64, seats: usize, duplicate: bool) -> u64 {
    if duplicate {
        (hands_per_table / seats as u64).max(1) * seats as u64
    } else {
        hands_per_table
    }
}

/// Uniform integer in `0..n` for `n > 0` (multiply-shift; bias < 2^-40 for any n here).
fn below_u64(rng: &mut Xoshiro256PlusPlus, n: u64) -> u64 {
    ((rng.next_u64() as u128 * n as u128) >> 64) as u64
}

/// Draws the replay plan for `k` bots. `per_bot = 0` gives an empty plan.
pub fn plan_replays(
    schedules: &[Schedule],
    k: usize,
    hands_per_table: u64,
    duplicate: bool,
    per_bot: usize,
    seed: u64,
) -> ReplayPlan {
    let mut plan = ReplayPlan {
        per_bot: vec![Vec::new(); k],
        per_table: BTreeMap::new(),
    };
    if per_bot == 0 {
        return plan;
    }
    let mut tables_union: BTreeMap<(usize, usize), BTreeSet<u64>> = BTreeMap::new();
    for bot in 0..k {
        // Tables where this bot sits (once per table even with several seats), with sizes.
        let mut slots: Vec<(usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        for (r, s) in schedules.iter().enumerate() {
            for (t, seats) in s.tables.iter().enumerate() {
                if seats.contains(&bot) {
                    let h = hands_at_table(hands_per_table, seats.len(), duplicate);
                    slots.push((r, t, total));
                    total += h;
                }
            }
        }
        if total == 0 {
            continue;
        }
        // Floyd's algorithm: `want` distinct indices in 0..total.
        let want = (per_bot as u64).min(total);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(mix_seed(seed, STREAM_REPLAYS, bot as u64));
        let mut chosen: BTreeSet<u64> = BTreeSet::new();
        for j in (total - want)..total {
            let x = below_u64(&mut rng, j + 1);
            if !chosen.insert(x) {
                chosen.insert(j);
            }
        }
        for idx in chosen {
            // Locate the table slot containing global index `idx`.
            let pos = slots.partition_point(|s| s.2 <= idx) - 1;
            let (r, t, start) = slots[pos];
            let hand = idx - start;
            plan.per_bot[bot].push((r, t, hand));
            tables_union.entry((r, t)).or_default().insert(hand);
        }
    }
    plan.per_table = tables_union
        .into_iter()
        .map(|(key, set)| (key, set.into_iter().collect()))
        .collect();
    plan
}

fn file_name(key: &str) -> String {
    let safe: String = key
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "-_.@".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{safe}.txt")
}

/// Writes `<dir>/<run_id>/<bot>.txt`, one file per bot with its sampled hands in play order.
/// The directory is built under a temporary name and renamed into place at the end.
pub fn write_replays(
    dir: &Path,
    run_id: &str,
    keys: &[String],
    plan: &ReplayPlan,
    rounds: &[RoundRecord],
) -> std::io::Result<PathBuf> {
    let final_dir = dir.join(run_id);
    let tmp = dir.join(format!(".{run_id}.partial"));
    if tmp.exists() {
        std::fs::remove_dir_all(&tmp)?;
    }
    std::fs::create_dir_all(&tmp)?;
    let lookup = |r: usize, t: usize, h: u64| -> Option<&str> {
        let res = rounds.get(r)?.results.get(t)?;
        let i = res
            .recorded_hands
            .binary_search_by_key(&h, |x| x.hand_id)
            .ok()?;
        Some(res.recorded_hands[i].text.as_str())
    };
    for (bot, key) in keys.iter().enumerate() {
        let sample = &plan.per_bot[bot];
        let found = sample
            .iter()
            .filter(|&&(r, t, h)| lookup(r, t, h).is_some())
            .count();
        let mut out = format!(
            "# Hand replays for {key}: {found} random hands from run {run_id}\n\
             # (all hole cards are shown; bots never saw these during play)\n\n"
        );
        for &(r, t, h) in sample {
            if let Some(text) = lookup(r, t, h) {
                let _ = writeln!(
                    out,
                    "### Round {} · table {} · hand {}",
                    r + 1,
                    t + 1,
                    h + 1
                );
                out.push_str(text);
            }
        }
        std::fs::write(tmp.join(file_name(key)), out)?;
    }
    if final_dir.exists() {
        std::fs::remove_dir_all(&final_dir)?;
    }
    std::fs::rename(&tmp, &final_dir)?;
    Ok(final_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sched(tables: Vec<Vec<usize>>) -> Schedule {
        Schedule { tables }
    }

    #[test]
    fn plan_is_deterministic_exact_and_scoped_to_the_bots_tables() {
        let s = vec![
            sched(vec![vec![0, 1, 2], vec![2, 3, 0]]),
            sched(vec![vec![1, 3, 3], vec![0, 2, 1]]),
        ];
        let p = plan_replays(&s, 4, 1000, false, 100, 7);
        assert_eq!(p, plan_replays(&s, 4, 1000, false, 100, 7));
        assert_ne!(p, plan_replays(&s, 4, 1000, false, 100, 8));
        for (bot, picks) in p.per_bot.iter().enumerate() {
            assert_eq!(picks.len(), 100);
            let mut uniq = picks.clone();
            uniq.dedup();
            assert_eq!(uniq.len(), 100, "no duplicates");
            for &(r, t, h) in picks {
                assert!(s[r].tables[t].contains(&bot) && h < 1000);
                assert!(p.record_for(r, t).binary_search(&h).is_ok());
            }
        }
        // Fewer slots than requested: take them all.
        let small = plan_replays(&s, 4, 10, false, 100, 1);
        assert_eq!(small.per_bot[3].len(), 20); // bot 3 sits at two tables of 10 hands
        assert!(plan_replays(&s, 4, 1000, false, 0, 1).is_empty());
    }

    #[test]
    fn duplicate_mode_hand_ids_stay_in_range() {
        let s = vec![sched(vec![vec![0, 1, 2]])];
        assert_eq!(hands_at_table(1000, 3, true), 999);
        let p = plan_replays(&s, 3, 1000, true, 999, 3);
        assert!(p.per_bot[0].iter().all(|&(_, _, h)| h < 999));
        assert_eq!(p.per_bot[0].len(), 999);
    }
}
