//! Matchmaking: builds a round's schedule of tables with balanced appearances.
//!
//! Inputs: `k` bots, a seats-per-table target `s` (default: the format's max seats),
//! `min_appearances`, `allow_duplicate_bot_at_table` and a seed.
//!
//! 1. **Effective seats** `s'`: `min(s, k)` if duplicates are not allowed, else `s` (when
//!    `k < s` tables are filled with independent instances of the same bots; when `k >= s`
//!    tables still never repeat a bot).
//! 2. **Tables** `T = max(k, ⌈min_appearances·k / s'⌉, min_tables)`.
//! 3. Concatenate independent random permutations of the bots and cut the first `T·s'`
//!    entries into tables, so every bot appears `⌊T·s'/k⌋` or `⌈T·s'/k⌉` times.
//! 4. **Repair** tables holding a bot more than `cap = ⌈s'/k⌉` times (cap = 1 when `s' ≤ k`)
//!    by swapping with a slot in another table where the swap creates no new violation.
//! 5. Optionally **balance pairs**: swap-based local search that lowers the variance of how
//!    often each pair of bots shares a table.
//! 6. Shuffle the seat order within each table.

use engine::rng::below;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};

/// Matchmaking settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MatchmakingConfig {
    /// Seats per table; `None` = the format's max seats.
    pub seats_per_table: Option<usize>,
    /// Minimum number of table appearances per bot per round.
    pub min_appearances: usize,
    /// Allow several instances of the same bot at one table (only used when `k < s`).
    pub allow_duplicate_bot_at_table: bool,
    /// Lower bound on the number of tables per round.
    pub min_tables: usize,
    /// Run the pair-balancing local search.
    pub balance_pairs: bool,
    /// Number of local-search steps for `balance_pairs`.
    pub balance_steps: usize,
}

impl Default for MatchmakingConfig {
    fn default() -> Self {
        MatchmakingConfig {
            seats_per_table: None,
            min_appearances: 4,
            allow_duplicate_bot_at_table: true,
            min_tables: 0,
            balance_pairs: false,
            balance_steps: 500,
        }
    }
}

/// A round's schedule: `tables[t][seat]` is a bot index.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Schedule {
    /// Seat assignments per table.
    pub tables: Vec<Vec<usize>>,
}

impl Schedule {
    /// Number of appearances of each of `k` bots.
    pub fn appearances(&self, k: usize) -> Vec<usize> {
        let mut c = vec![0; k];
        for t in &self.tables {
            for &b in t {
                c[b] += 1;
            }
        }
        c
    }
}

/// Matchmaking error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MatchmakingError(pub String);
impl std::fmt::Display for MatchmakingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "matchmaking: {}", self.0)
    }
}
impl std::error::Error for MatchmakingError {}

fn shuffle<T>(v: &mut [T], rng: &mut Xoshiro256PlusPlus) {
    for i in (1..v.len()).rev() {
        let j = below(rng, (i + 1) as u32) as usize;
        v.swap(i, j);
    }
}

/// Effective seats per table for `k` bots.
pub fn effective_seats(k: usize, max_seats: usize, cfg: &MatchmakingConfig) -> usize {
    let s = cfg.seats_per_table.unwrap_or(max_seats);
    if cfg.allow_duplicate_bot_at_table {
        s
    } else {
        s.min(k)
    }
}

/// Builds a schedule for `k` bots at tables of a format with `max_seats` seats.
pub fn make_schedule(
    k: usize,
    max_seats: usize,
    cfg: &MatchmakingConfig,
    seed: u64,
) -> Result<Schedule, MatchmakingError> {
    let err = |m: String| Err(MatchmakingError(m));
    if k == 0 {
        return err("no bots".into());
    }
    let s = cfg.seats_per_table.unwrap_or(max_seats);
    if s < 2 || s > max_seats {
        return err(format!("seats_per_table {s} must be in 2..={max_seats}"));
    }
    if cfg.min_appearances == 0 {
        return err("min_appearances must be >= 1".into());
    }
    let sp = effective_seats(k, max_seats, cfg);
    if sp < 2 {
        return err(format!(
            "{k} bot(s) without duplicates allowed cannot fill a table"
        ));
    }
    let t = k
        .max((cfg.min_appearances * k).div_ceil(sp))
        .max(cfg.min_tables);
    let n = t * sp;
    let cap = sp.div_ceil(k);

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut seq = Vec::with_capacity(n + k);
    while seq.len() < n {
        let mut p: Vec<usize> = (0..k).collect();
        shuffle(&mut p, &mut rng);
        seq.extend(p);
    }
    seq.truncate(n);
    let mut tables: Vec<Vec<usize>> = seq.chunks(sp).map(|c| c.to_vec()).collect();

    repair(&mut tables, k, cap, &mut rng)?;
    if cfg.balance_pairs && k > 2 {
        balance_pairs(&mut tables, k, cap, cfg.balance_steps, &mut rng);
    }
    for tb in &mut tables {
        shuffle(tb, &mut rng);
    }
    Ok(Schedule { tables })
}

fn count(table: &[usize], b: usize) -> usize {
    table.iter().filter(|&&x| x == b).count()
}

fn repair(
    tables: &mut [Vec<usize>],
    k: usize,
    cap: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> Result<(), MatchmakingError> {
    let n_tables = tables.len();
    for t in 0..n_tables {
        while let Some(b) = (0..k).find(|&b| count(&tables[t], b) > cap) {
            let i = tables[t].iter().position(|&x| x == b).unwrap();
            let mut others: Vec<usize> = (0..n_tables).filter(|&u| u != t).collect();
            shuffle(&mut others, rng);
            let mut fixed = false;
            'search: for u in others {
                for j in 0..tables[u].len() {
                    let c = tables[u][j];
                    if c != b && count(&tables[t], c) < cap && count(&tables[u], b) < cap {
                        tables[t][i] = c;
                        tables[u][j] = b;
                        fixed = true;
                        break 'search;
                    }
                }
            }
            if !fixed {
                return Err(MatchmakingError(format!(
                    "could not repair table {t}: bot {b} appears more than {cap} times \
                     and no valid swap exists ({k} bots, {} seats)",
                    tables[t].len()
                )));
            }
        }
    }
    Ok(())
}

/// Pair co-occurrence statistics: `c[a*k+b]` = number of tables where distinct bots `a < b`
/// both sit.
struct PairStats {
    k: usize,
    c: Vec<i64>,
    sum: i64,
    sumsq: i64,
}

impl PairStats {
    fn new(tables: &[Vec<usize>], k: usize) -> PairStats {
        let mut p = PairStats {
            k,
            c: vec![0; k * k],
            sum: 0,
            sumsq: 0,
        };
        for t in tables {
            p.apply(t, 1);
        }
        p
    }
    fn apply(&mut self, table: &[usize], sign: i64) {
        let mut present = vec![false; self.k];
        for &b in table {
            present[b] = true;
        }
        for a in 0..self.k {
            if !present[a] {
                continue;
            }
            for b in a + 1..self.k {
                if present[b] {
                    let x = &mut self.c[a * self.k + b];
                    self.sumsq += 2 * *x * sign + 1;
                    *x += sign;
                    self.sum += sign;
                }
            }
        }
    }
    /// Variance of pair counts (times the squared number of pairs, which is constant).
    fn objective(&self) -> i128 {
        let pairs = (self.k * (self.k - 1) / 2) as i128;
        pairs * self.sumsq as i128 - (self.sum as i128) * (self.sum as i128)
    }
}

/// Variance of pair co-occurrence counts over all bot pairs.
pub fn pair_variance(s: &Schedule, k: usize) -> f64 {
    if k < 2 {
        return 0.0;
    }
    let p = PairStats::new(&s.tables, k);
    let pairs = (k * (k - 1) / 2) as f64;
    p.objective() as f64 / (pairs * pairs)
}

fn balance_pairs(
    tables: &mut [Vec<usize>],
    k: usize,
    cap: usize,
    steps: usize,
    rng: &mut Xoshiro256PlusPlus,
) {
    let n_tables = tables.len();
    if n_tables < 2 {
        return;
    }
    let mut ps = PairStats::new(tables, k);
    let mut best = ps.objective();
    for _ in 0..steps {
        let t = below(rng, n_tables as u32) as usize;
        let mut u = below(rng, n_tables as u32 - 1) as usize;
        if u >= t {
            u += 1;
        }
        let i = below(rng, tables[t].len() as u32) as usize;
        let j = below(rng, tables[u].len() as u32) as usize;
        let (b, c) = (tables[t][i], tables[u][j]);
        if b == c || count(&tables[t], c) >= cap || count(&tables[u], b) >= cap {
            continue;
        }
        ps.apply(&tables[t], -1);
        ps.apply(&tables[u], -1);
        tables[t][i] = c;
        tables[u][j] = b;
        ps.apply(&tables[t], 1);
        ps.apply(&tables[u], 1);
        let obj = ps.objective();
        if obj < best {
            best = obj;
        } else {
            // Revert.
            ps.apply(&tables[t], -1);
            ps.apply(&tables[u], -1);
            tables[t][i] = b;
            tables[u][j] = c;
            ps.apply(&tables[t], 1);
            ps.apply(&tables[u], 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(k: usize, max_seats: usize, s: Option<usize>, m: usize, dup: bool, seed: u64) {
        let cfg = MatchmakingConfig {
            seats_per_table: s,
            min_appearances: m,
            allow_duplicate_bot_at_table: dup,
            ..Default::default()
        };
        let sp = effective_seats(k, max_seats, &cfg);
        if sp < 2 {
            assert!(make_schedule(k, max_seats, &cfg, seed).is_err());
            return;
        }
        let sch = make_schedule(k, max_seats, &cfg, seed).unwrap();
        let ctx = format!("k={k} s={s:?} m={m} dup={dup} seed={seed}");
        assert!(sch.tables.len() >= k, "T >= k: {ctx}");
        let app = sch.appearances(k);
        let (lo, hi) = (*app.iter().min().unwrap(), *app.iter().max().unwrap());
        assert!(lo >= m, "min appearances: {ctx} {app:?}");
        assert!(hi - lo <= 1, "balanced: {ctx} {app:?}");
        let cap = sp.div_ceil(k);
        for t in &sch.tables {
            assert_eq!(t.len(), sp, "{ctx}");
            for b in 0..k {
                assert!(count(t, b) <= cap, "duplicate cap: {ctx} {t:?}");
            }
            if sp <= k {
                let mut d = t.clone();
                d.sort();
                d.dedup();
                assert_eq!(d.len(), t.len(), "no duplicates: {ctx} {t:?}");
            }
        }
        assert_eq!(
            sch,
            make_schedule(k, max_seats, &cfg, seed).unwrap(),
            "deterministic"
        );
    }

    #[test]
    fn grid() {
        for k in [1, 2, 3, 4, 5, 6, 7, 9, 13, 25, 40] {
            for s in [2, 3, 6, 9] {
                for m in [1, 4, 7] {
                    for dup in [true, false] {
                        for seed in 0..3 {
                            check(k, 10, Some(s), m, dup, seed);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn default_seats_from_format() {
        check(7, 6, None, 4, true, 1);
        let s = make_schedule(7, 6, &MatchmakingConfig::default(), 1).unwrap();
        assert_eq!(s.tables.len(), 7); // max(7, ceil(28/6)=5) = 7
        assert!(s.tables.iter().all(|t| t.len() == 6));
    }

    #[test]
    fn min_tables_raises_table_count() {
        let cfg = MatchmakingConfig {
            min_tables: 20,
            ..Default::default()
        };
        let s = make_schedule(4, 6, &cfg, 3).unwrap();
        assert_eq!(s.tables.len(), 20);
    }

    #[test]
    fn heads_up_large_k() {
        let cfg = MatchmakingConfig::default();
        let s = make_schedule(100, 2, &cfg, 9).unwrap();
        assert_eq!(s.tables.len(), 200);
        assert!(s.appearances(100).iter().all(|&a| a == 4));
    }

    #[test]
    fn balance_pairs_reduces_variance() {
        let base = MatchmakingConfig {
            seats_per_table: Some(4),
            ..Default::default()
        };
        let balanced = MatchmakingConfig {
            balance_pairs: true,
            balance_steps: 2000,
            ..base.clone()
        };
        let mut worse = 0;
        for seed in 0..10 {
            let a = make_schedule(12, 6, &base, seed).unwrap();
            let b = make_schedule(12, 6, &balanced, seed).unwrap();
            let (va, vb) = (pair_variance(&a, 12), pair_variance(&b, 12));
            if vb > va {
                worse += 1;
            }
            assert_eq!(a.appearances(12), b.appearances(12));
        }
        assert_eq!(worse, 0);
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(400))]
        #[test]
        fn random_combinations(
            k in 1usize..60,
            s in 2usize..=10,
            m in 1usize..8,
            dup in proptest::prelude::any::<bool>(),
            seed in proptest::prelude::any::<u64>(),
        ) {
            check(k, 10, Some(s), m, dup, seed);
        }
    }
}
