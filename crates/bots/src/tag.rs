//! Tight-aggressive rule bot.
//!
//! * **Preflop (Hold'em):** hands are scored with the Chen formula (a standard 169-class
//!   ranking). It opens with a position-dependent threshold, 3-bets premium hands, calls
//!   with strong ones, and continues only with the top of the range against 3-bets.
//! * **Preflop (Omaha) and postflop:** Monte Carlo equity against the live opponents. It
//!   value-bets strong equity, raises very strong equity, and calls when equity beats pot
//!   odds plus a margin.

use crate::equity::equity;
use crate::simple::{clamp_raise, pot_fraction_to};
use arena::bot::{Action, ActionKind, Bot, Observation, Street, Variant};
use engine::eval::Evaluator;
use engine::Card;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Parameters of [`TightAggressive`].
#[derive(Clone, Debug, PartialEq)]
pub struct TagParams {
    /// Monte Carlo samples per postflop decision.
    pub samples: u32,
    /// Added to every preflop Chen threshold (positive = tighter).
    pub tightness: f64,
    /// Required equity edge over pot odds to call.
    pub call_margin: f64,
}

impl Default for TagParams {
    fn default() -> Self {
        TagParams {
            samples: 300,
            tightness: 0.0,
            call_margin: 0.05,
        }
    }
}

/// Tight-aggressive rule bot. See the module docs.
pub struct TightAggressive {
    name: String,
    p: TagParams,
    rng: Xoshiro256PlusPlus,
    ev: Evaluator,
}

impl TightAggressive {
    /// Creates the bot.
    pub fn new(name: impl Into<String>, p: TagParams, seed: u64) -> Self {
        TightAggressive {
            name: name.into(),
            p,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
            ev: Evaluator::get(),
        }
    }

    fn preflop_holdem(&mut self, obs: &Observation) -> Action {
        let la = &obs.legal;
        let chen = chen_score(obs.hole[0], obs.hole[1]) - self.p.tightness;
        let raises = obs
            .history
            .iter()
            .filter(|r| r.street == Street::Preflop && r.kind == ActionKind::Raise)
            .count();
        let fold_or_check = if la.can_check {
            Action::Check
        } else {
            Action::Fold
        };
        let behind = players_behind_preflop(obs);
        match raises {
            0 => {
                let open = match behind {
                    0 | 1 => 7.0,
                    2 | 3 => 8.0,
                    _ => 9.5,
                };
                if chen >= open && la.can_raise {
                    let limpers = obs
                        .history
                        .iter()
                        .filter(|r| r.street == Street::Preflop && r.kind == ActionKind::Call)
                        .count() as u64;
                    clamp_raise(la, obs.bb * (3 + limpers))
                } else {
                    fold_or_check
                }
            }
            1 => {
                if chen >= 12.0 && la.can_raise {
                    clamp_raise(la, obs.current_bet * 3)
                } else if chen >= 9.0 {
                    let my_stack = obs.stacks[obs.seat] + obs.street_contrib[obs.seat];
                    if la.call_amount * 6 <= my_stack {
                        Action::Call
                    } else {
                        fold_or_check
                    }
                } else {
                    fold_or_check
                }
            }
            _ => {
                if chen >= 15.0 {
                    if la.can_raise {
                        clamp_raise(la, la.max_raise_to)
                    } else {
                        Action::Call
                    }
                } else if chen >= 12.0 {
                    Action::Call
                } else {
                    fold_or_check
                }
            }
        }
    }

    fn by_equity(&mut self, obs: &Observation) -> Action {
        let la = &obs.legal;
        let opponents = obs
            .status
            .iter()
            .enumerate()
            .filter(|&(s, st)| s != obs.seat && st.is_live())
            .count();
        let samples = if obs.street == Street::Preflop {
            (self.p.samples / 2).max(50)
        } else {
            self.p.samples
        };
        let variant = obs.format.variant;
        let e = equity(
            self.ev,
            variant,
            obs.hole,
            obs.board,
            opponents,
            samples,
            &mut self.rng,
        );
        let fair = 1.0 / (opponents as f64 + 1.0);
        let strong = (1.6 * fair).clamp(0.55, 0.85);
        let very_strong = (strong + 0.15).min(0.92);
        if la.can_check {
            if e >= very_strong && la.can_raise {
                clamp_raise(la, pot_fraction_to(obs, 0.75))
            } else if e >= strong && la.can_raise {
                clamp_raise(la, pot_fraction_to(obs, 0.5))
            } else {
                Action::Check
            }
        } else {
            let odds = la.call_amount as f64 / (obs.pot + la.call_amount) as f64;
            if e >= very_strong && la.can_raise {
                clamp_raise(la, pot_fraction_to(obs, 0.8))
            } else if e >= odds + self.p.call_margin {
                Action::Call
            } else {
                Action::Fold
            }
        }
    }
}

impl Bot for TightAggressive {
    fn name(&self) -> &str {
        &self.name
    }
    fn act(&mut self, obs: &Observation) -> Action {
        if obs.street == Street::Preflop && obs.format.variant != Variant::PLO {
            self.preflop_holdem(obs)
        } else {
            self.by_equity(obs)
        }
    }
}

/// Number of players still to act after `obs.seat` in the first preflop orbit (0 for the
/// big blind).
fn players_behind_preflop(obs: &Observation) -> usize {
    let n = obs.n_seats;
    let bb_seat = obs
        .history
        .iter()
        .find(|r| r.kind == ActionKind::PostBigBlind)
        .map_or(obs.button, |r| r.seat as usize);
    let mut behind = 0;
    let mut s = obs.seat;
    while s != bb_seat {
        s = (s + 1) % n;
        if obs.status[s].is_live() {
            behind += 1;
        }
    }
    behind
}

/// Chen formula score of a Hold'em starting hand (higher is better; AA = 20, 72o ≈ -1).
pub fn chen_score(a: Card, b: Card) -> f64 {
    let (hi, lo) = if a.rank() >= b.rank() { (a, b) } else { (b, a) };
    let card_pts = |r: u8| match r {
        12 => 10.0,
        11 => 8.0,
        10 => 7.0,
        9 => 6.0,
        r => (r as f64 + 2.0) / 2.0,
    };
    let mut s = card_pts(hi.rank());
    if hi.rank() == lo.rank() {
        return (s * 2.0).max(5.0);
    }
    if hi.suit() == lo.suit() {
        s += 2.0;
    }
    let gap = hi.rank() - lo.rank() - 1;
    s -= match gap {
        0 => 0.0,
        1 => 1.0,
        2 => 2.0,
        3 => 4.0,
        _ => 5.0,
    };
    if gap <= 1 && hi.rank() < 10 {
        s += 1.0;
    }
    s.ceil()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn c(s: &str) -> Card {
        s.parse().unwrap()
    }
    #[test]
    fn chen_examples() {
        assert_eq!(chen_score(c("As"), c("Ah")), 20.0);
        assert_eq!(chen_score(c("Ks"), c("Kh")), 16.0);
        assert_eq!(chen_score(c("As"), c("Ks")), 12.0);
        assert_eq!(chen_score(c("2s"), c("2h")), 5.0);
        assert_eq!(chen_score(c("7s"), c("2h")), -1.0);
        assert_eq!(chen_score(c("Ts"), c("9s")), 8.0);
    }
}
