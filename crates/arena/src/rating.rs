//! Ratings: a [`RatingSystem`] trait and pairwise multiplayer Elo.
//!
//! **Pairwise Elo.** Every seat at a table is its own participant; seats are never pooled.
//! For seat `i` (bot `a`) and every seat `j` held by a *different* bot `b` (same-bot pairs
//! are skipped):
//!
//! * expected score `E_ij = 1 / (1 + 10^((R_b − R_a)/400))`, using ratings frozen at the
//!   start of the round;
//! * actual score `S_ij`:
//!   * cash games compare per-seat **bb/100**: `binary` gives 1, ½ or 0; `margin` gives
//!     `½ + ½·tanh((w_i − w_j)/c)`;
//!   * tournaments compare **finishing places** (1, ½ or 0); `margin` falls back to this.
//! * `Δ_i = K/(N−1) · Σ_j (S_ij − E_ij)` where `N` is the number of seats at the table.
//!
//! A bot's round delta is the sum of `Δ_i` over all its seats at all tables in the round,
//! applied once at round end, so the result does not depend on table order. With one seat
//! per bot (`N = n` distinct bots) this is exactly the `K/(n−1) · Σ_j (S_ij − E_ij)` formula.
//! Using the same divisor for every seat at a table keeps each pair's update symmetric, so
//! the system stays zero-sum even when a bot holds several seats.

use crate::table::TableResult;
use serde::{Deserialize, Serialize};

/// How the actual score of a pair is derived.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ScoreMode {
    /// 1 / ½ / 0 by comparing results.
    #[default]
    Binary,
    /// `½ + ½·tanh(Δbb/100 / c)` (cash only).
    Margin,
}

/// Rating settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RatingConfig {
    /// Initial rating of unseen bots.
    pub initial: f64,
    /// K-factor.
    pub k: f64,
    /// Score mode.
    pub score_mode: ScoreMode,
    /// Margin scale `c` in bb/100 for [`ScoreMode::Margin`].
    pub margin_c: f64,
}

impl Default for RatingConfig {
    fn default() -> Self {
        RatingConfig {
            initial: 1500.0,
            k: 16.0,
            score_mode: ScoreMode::Binary,
            margin_c: 30.0,
        }
    }
}

/// One seat's result, as seen by the rating system.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SeatOutcome {
    /// Bot index.
    pub bot: usize,
    /// Result where higher is better: bb/100 in cash games, `-place` in tournaments.
    pub value: f64,
}

/// A table's outcome for rating purposes.
#[derive(Clone, Debug, PartialEq)]
pub struct TableOutcome {
    /// Per-seat outcomes.
    pub seats: Vec<SeatOutcome>,
    /// True if values are finishing places (margin scoring does not apply).
    pub placement: bool,
}

impl TableOutcome {
    /// Extracts per-seat outcomes: bb/100 for cash tables, finishing place for tournaments.
    pub fn from_result(r: &TableResult) -> TableOutcome {
        TableOutcome {
            placement: r.tournament,
            seats: r
                .seats
                .iter()
                .map(|s| SeatOutcome {
                    bot: s.bot,
                    value: if r.tournament {
                        -(s.place.unwrap_or(u32::MAX) as f64)
                    } else {
                        s.stats.bb_per_100(r.bb)
                    },
                })
                .collect(),
        }
    }
}

/// A rating system updated in rounds. Implementations must make the end-of-round result
/// independent of the order in which tables are added.
pub trait RatingSystem: Send {
    /// Starts accumulating a round.
    fn begin_round(&mut self);
    /// Adds one table's outcome to the current round.
    fn add_table(&mut self, outcome: &TableOutcome);
    /// Applies the accumulated updates.
    fn end_round(&mut self);
    /// Current ratings, indexed by bot.
    fn ratings(&self) -> &[f64];
}

/// Pairwise multiplayer Elo. See the module docs.
#[derive(Clone, Debug)]
pub struct PairwiseElo {
    cfg: RatingConfig,
    ratings: Vec<f64>,
    pending: Vec<f64>,
}

impl PairwiseElo {
    /// Creates the system with the given starting ratings (one per bot).
    pub fn new(cfg: RatingConfig, initial: Vec<f64>) -> PairwiseElo {
        let n = initial.len();
        PairwiseElo {
            cfg,
            ratings: initial,
            pending: vec![0.0; n],
        }
    }

    /// Expected score of a player rated `ra` against one rated `rb`.
    pub fn expected(ra: f64, rb: f64) -> f64 {
        1.0 / (1.0 + 10f64.powf((rb - ra) / 400.0))
    }

    fn actual(&self, wi: f64, wj: f64, placement: bool) -> f64 {
        match self.cfg.score_mode {
            ScoreMode::Margin if !placement => 0.5 + 0.5 * ((wi - wj) / self.cfg.margin_c).tanh(),
            _ => {
                if wi > wj {
                    1.0
                } else if wi < wj {
                    0.0
                } else {
                    0.5
                }
            }
        }
    }
}

impl RatingSystem for PairwiseElo {
    fn begin_round(&mut self) {
        self.pending.iter_mut().for_each(|p| *p = 0.0);
    }

    fn add_table(&mut self, o: &TableOutcome) {
        let n = o.seats.len();
        if n < 2 {
            return;
        }
        let scale = self.cfg.k / (n - 1) as f64;
        for (i, si) in o.seats.iter().enumerate() {
            let mut sum = 0.0;
            for (j, sj) in o.seats.iter().enumerate() {
                if i == j || si.bot == sj.bot {
                    continue;
                }
                let e = Self::expected(self.ratings[si.bot], self.ratings[sj.bot]);
                sum += self.actual(si.value, sj.value, o.placement) - e;
            }
            self.pending[si.bot] += scale * sum;
        }
    }

    fn end_round(&mut self) {
        for (r, p) in self.ratings.iter_mut().zip(&mut self.pending) {
            *r += *p;
            *p = 0.0;
        }
    }

    fn ratings(&self) -> &[f64] {
        &self.ratings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(v: &[(usize, f64)]) -> TableOutcome {
        TableOutcome {
            seats: v
                .iter()
                .map(|&(bot, value)| SeatOutcome { bot, value })
                .collect(),
            placement: false,
        }
    }

    #[test]
    fn heads_up_binary() {
        let mut e = PairwiseElo::new(RatingConfig::default(), vec![1500.0, 1500.0]);
        e.begin_round();
        e.add_table(&table(&[(0, 10.0), (1, -10.0)]));
        e.end_round();
        assert_eq!(e.ratings(), &[1508.0, 1492.0]);
    }

    #[test]
    fn multiway_formula_and_zero_sum() {
        let mut e = PairwiseElo::new(RatingConfig::default(), vec![1500.0, 1600.0, 1400.0]);
        e.begin_round();
        e.add_table(&table(&[(0, 5.0), (1, 0.0), (2, -5.0)]));
        e.end_round();
        let r = e.ratings().to_vec();
        let e01 = PairwiseElo::expected(1500.0, 1600.0);
        let e02 = PairwiseElo::expected(1500.0, 1400.0);
        let d0 = 16.0 / 2.0 * ((1.0 - e01) + (1.0 - e02));
        assert!((r[0] - 1500.0 - d0).abs() < 1e-9);
        assert!((r.iter().sum::<f64>() - 4500.0).abs() < 1e-9, "zero-sum");
    }

    #[test]
    fn order_independent_within_round() {
        let tables = [
            table(&[(0, 5.0), (1, 0.0), (2, -5.0)]),
            table(&[(2, 50.0), (0, -50.0)]),
            table(&[(1, 1.0), (2, 2.0), (0, 3.0)]),
        ];
        let run = |order: &[usize]| {
            let mut e = PairwiseElo::new(RatingConfig::default(), vec![1500.0, 1520.0, 1480.0]);
            e.begin_round();
            for &i in order {
                e.add_table(&tables[i]);
            }
            e.end_round();
            e.ratings().to_vec()
        };
        let a = run(&[0, 1, 2]);
        let b = run(&[2, 0, 1]);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-9);
        }
    }

    #[test]
    fn same_bot_pairs_excluded_and_seats_not_pooled() {
        // Bot 0 holds two seats: one beats bot 1, one loses to it. Each seat is scored on its
        // own against bot 1 only (K/(N-1) = 8): +4 and -4, so bot 0 nets zero, as does bot 1.
        let mut e = PairwiseElo::new(RatingConfig::default(), vec![1500.0, 1500.0]);
        e.begin_round();
        e.add_table(&table(&[(0, 20.0), (0, -30.0), (1, 0.0)]));
        e.end_round();
        assert_eq!(e.ratings(), &[1500.0, 1500.0]);
        // A table of a single bot changes nothing.
        e.begin_round();
        e.add_table(&table(&[(0, 20.0), (0, -20.0)]));
        e.end_round();
        assert_eq!(e.ratings(), &[1500.0, 1500.0]);
    }

    #[test]
    fn zero_sum_with_duplicate_seats() {
        let mut e = PairwiseElo::new(RatingConfig::default(), vec![1500.0, 1450.0, 1600.0]);
        e.begin_round();
        e.add_table(&table(&[
            (0, 20.0),
            (0, -30.0),
            (1, 0.0),
            (2, 5.0),
            (1, 9.0),
            (2, -1.0),
        ]));
        e.add_table(&table(&[(2, 1.0), (1, 2.0), (1, 3.0)]));
        e.end_round();
        assert!((e.ratings().iter().sum::<f64>() - 4550.0).abs() < 1e-9);
    }

    #[test]
    fn margin_mode_and_placement_fallback() {
        let cfg = RatingConfig {
            score_mode: ScoreMode::Margin,
            ..Default::default()
        };
        let mut e = PairwiseElo::new(cfg.clone(), vec![1500.0, 1500.0]);
        e.begin_round();
        e.add_table(&table(&[(0, 30.0), (1, 0.0)]));
        e.end_round();
        let expect = 16.0 * (0.5 + 0.5 * 1f64.tanh() - 0.5);
        assert!((e.ratings()[0] - 1500.0 - expect).abs() < 1e-9);

        let mut e = PairwiseElo::new(cfg, vec![1500.0, 1500.0]);
        e.begin_round();
        let mut t = table(&[(0, -1.0), (1, -2.0)]); // places 1 and 2
        t.placement = true;
        e.add_table(&t);
        e.end_round();
        assert_eq!(e.ratings(), &[1508.0, 1492.0]);
    }
}
