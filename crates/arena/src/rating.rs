//! Ratings: a [`RatingSystem`] trait and pairwise multiplayer Elo.
//!
//! **Pooling.** When a bot holds several seats at a table (duplicates allowed and `k < s`),
//! its seats are pooled into one result for that table: in cash games its bb/100 over all
//! its seats (total net / total hands), in tournaments the mean finishing place of its
//! instances. Rating then works on the `n` distinct bots at the table.
//!
//! **Pairwise Elo.** For each pair of distinct bots `(i, j)` at a table:
//!
//! * expected score `E_ij = 1 / (1 + 10^((R_j − R_i)/400))`, using ratings frozen at the
//!   start of the round;
//! * actual score `S_ij`:
//!   * cash games compare pooled **bb/100**: `binary` gives 1, ½ or 0; `margin` gives
//!     `½ + ½·tanh((w_i − w_j)/c)`;
//!   * tournaments compare (mean) **finishing places** (1, ½ or 0); `margin` falls back
//!     to this.
//! * `Δ_i = K/(n−1) · Σ_j (S_ij − E_ij)`.
//!
//! Deltas are accumulated over the whole round and applied once at round end, so the result
//! does not depend on table order. The update is symmetric per pair, hence zero-sum.
//!
//! **Anchors.** A bot can be anchored at a fixed rating ([`PairwiseElo::anchor`]): it still
//! enters every expected-score computation, but its own rating never changes. Anchors pin the
//! scale (e.g. random = 850, calling station = 400, tight-aggressive = 1300), so ratings stay
//! comparable across runs and leaderboards. With anchors the system is no longer zero-sum.

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

/// One bot's (pooled) result at a table, as seen by the rating system.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BotOutcome {
    /// Bot index.
    pub bot: usize,
    /// Result where higher is better: pooled bb/100 in cash games, `-mean place` in
    /// tournaments.
    pub value: f64,
}

/// A table's outcome for rating purposes: one entry per distinct bot.
#[derive(Clone, Debug, PartialEq)]
pub struct TableOutcome {
    /// Per-bot outcomes (distinct bots, in order of first seat).
    pub bots: Vec<BotOutcome>,
    /// True if values are finishing places (margin scoring does not apply).
    pub placement: bool,
}

impl TableOutcome {
    /// Pools a table's seats by bot: bb/100 over all of a bot's seats (total net / total
    /// hands) for cash tables, mean finishing place for tournaments.
    pub fn from_result(r: &TableResult) -> TableOutcome {
        // (bot, net, hands, place sum, instances)
        let mut acc: Vec<(usize, i64, u64, f64, u32)> = Vec::new();
        for s in &r.seats {
            let i = match acc.iter().position(|a| a.0 == s.bot) {
                Some(i) => i,
                None => {
                    acc.push((s.bot, 0, 0, 0.0, 0));
                    acc.len() - 1
                }
            };
            acc[i].1 += s.stats.net;
            acc[i].2 += s.stats.hands;
            acc[i].3 += s.place.unwrap_or(u32::MAX) as f64;
            acc[i].4 += 1;
        }
        TableOutcome {
            placement: r.tournament,
            bots: acc
                .into_iter()
                .map(|(bot, net, hands, place_sum, n)| BotOutcome {
                    bot,
                    value: if r.tournament {
                        -(place_sum / n as f64)
                    } else if hands == 0 {
                        0.0
                    } else {
                        net as f64 / r.bb as f64 / hands as f64 * 100.0
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
    fixed: Vec<bool>,
}

impl PairwiseElo {
    /// Creates the system with the given starting ratings (one per bot).
    pub fn new(cfg: RatingConfig, initial: Vec<f64>) -> PairwiseElo {
        let n = initial.len();
        PairwiseElo {
            cfg,
            ratings: initial,
            pending: vec![0.0; n],
            fixed: vec![false; n],
        }
    }

    /// Fixes `bot`'s rating at `rating` for good (see the module docs on anchors).
    pub fn anchor(&mut self, bot: usize, rating: f64) {
        self.ratings[bot] = rating;
        self.fixed[bot] = true;
    }

    /// True if `bot` is anchored.
    pub fn is_anchored(&self, bot: usize) -> bool {
        self.fixed[bot]
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
        debug_assert!(
            o.bots
                .iter()
                .enumerate()
                .all(|(i, a)| o.bots[..i].iter().all(|b| b.bot != a.bot)),
            "TableOutcome must list distinct bots (use TableOutcome::from_result to pool)"
        );
        let n = o.bots.len();
        if n < 2 {
            return;
        }
        let scale = self.cfg.k / (n - 1) as f64;
        for (i, si) in o.bots.iter().enumerate() {
            let mut sum = 0.0;
            for (j, sj) in o.bots.iter().enumerate() {
                if i == j {
                    continue;
                }
                let e = Self::expected(self.ratings[si.bot], self.ratings[sj.bot]);
                sum += self.actual(si.value, sj.value, o.placement) - e;
            }
            self.pending[si.bot] += scale * sum;
        }
    }

    fn end_round(&mut self) {
        for ((r, p), fixed) in self
            .ratings
            .iter_mut()
            .zip(&mut self.pending)
            .zip(&self.fixed)
        {
            if !fixed {
                *r += *p;
            }
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
            bots: v
                .iter()
                .map(|&(bot, value)| BotOutcome { bot, value })
                .collect(),
            placement: false,
        }
    }

    fn result(seats: &[(usize, i64, Option<u32>)], tournament: bool) -> TableResult {
        use crate::stats::SeatStats;
        use crate::table::SeatResult;
        TableResult {
            hands_played: 100,
            bb: 2,
            tournament,
            hand_history: None,
            recorded_hands: Vec::new(),
            seats: seats
                .iter()
                .enumerate()
                .map(|(i, &(bot, net, place))| SeatResult {
                    bot,
                    seat: i,
                    stats: SeatStats {
                        hands: 100,
                        net,
                        ..Default::default()
                    },
                    final_stack: 0,
                    place,
                })
                .collect(),
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
    fn duplicate_seats_are_pooled() {
        // Bot 0 holds two seats (+20 and -30 chips over 100 hands each): pooled net -10 over
        // 200 hands = -2.5 bb/100, which loses to bot 1 (0). n = 2 distinct bots, so K/(n-1)
        // = 16 and bot 0 loses 8.
        let o = TableOutcome::from_result(&result(
            &[(0, 20, None), (0, -30, None), (1, 0, None)],
            false,
        ));
        assert_eq!(o.bots.len(), 2);
        assert_eq!(
            o.bots[0],
            BotOutcome {
                bot: 0,
                value: -2.5
            }
        );
        let mut e = PairwiseElo::new(RatingConfig::default(), vec![1500.0, 1500.0]);
        e.begin_round();
        e.add_table(&o);
        e.end_round();
        assert_eq!(e.ratings(), &[1492.0, 1508.0]);
        // A table of a single bot changes nothing.
        e.begin_round();
        e.add_table(&TableOutcome::from_result(&result(
            &[(0, 5, None), (0, -5, None)],
            false,
        )));
        e.end_round();
        assert_eq!(e.ratings(), &[1492.0, 1508.0]);
    }

    #[test]
    fn tournament_pools_mean_place() {
        let o = TableOutcome::from_result(&result(
            &[
                (0, 0, Some(1)),
                (0, 0, Some(4)),
                (1, 0, Some(2)),
                (2, 0, Some(3)),
            ],
            true,
        ));
        assert!(o.placement);
        let v: Vec<f64> = o.bots.iter().map(|b| b.value).collect();
        assert_eq!(v, vec![-2.5, -2.0, -3.0]);
    }

    #[test]
    fn zero_sum_with_duplicate_seats() {
        let mut e = PairwiseElo::new(RatingConfig::default(), vec![1500.0, 1450.0, 1600.0]);
        e.begin_round();
        e.add_table(&TableOutcome::from_result(&result(
            &[
                (0, 20, None),
                (0, -30, None),
                (1, 0, None),
                (2, 5, None),
                (1, 9, None),
                (2, -1, None),
            ],
            false,
        )));
        e.add_table(&TableOutcome::from_result(&result(
            &[(2, 1, None), (1, 2, None), (1, -3, None)],
            false,
        )));
        e.end_round();
        assert!((e.ratings().iter().sum::<f64>() - 4550.0).abs() < 1e-9);
    }

    #[test]
    fn anchored_bots_never_move_but_set_expectations() {
        let mut e = PairwiseElo::new(RatingConfig::default(), vec![1500.0, 1300.0, 400.0]);
        e.anchor(1, 1300.0);
        e.anchor(2, 400.0);
        e.begin_round();
        // Bot 0 loses to the 1300 anchor and beats the 400 anchor.
        e.add_table(&table(&[(0, 0.0), (1, 5.0), (2, -5.0)]));
        e.end_round();
        assert_eq!(&e.ratings()[1..], &[1300.0, 400.0]);
        let e01 = PairwiseElo::expected(1500.0, 1300.0);
        let e02 = PairwiseElo::expected(1500.0, 400.0);
        let d0 = 16.0 / 2.0 * ((0.0 - e01) + (1.0 - e02));
        assert!((e.ratings()[0] - 1500.0 - d0).abs() < 1e-9);
        assert!(e.is_anchored(1) && !e.is_anchored(0));
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
