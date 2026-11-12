//! Simple baseline bots: random-legal and calling station.

use arena::bot::{Action, Bot, LegalActions, Observation};
use engine::rng::{below, unit_f64};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Clamps a raise-to amount into the legal range.
pub(crate) fn clamp_raise(la: &LegalActions, to: u64) -> Action {
    Action::RaiseTo(to.clamp(la.min_raise_to, la.max_raise_to))
}

/// Raise-to amount for a bet of `frac` × (pot after calling) on top of the current bet.
pub(crate) fn pot_fraction_to(obs: &Observation, frac: f64) -> u64 {
    let pot_after_call = obs.pot + obs.legal.call_amount;
    obs.current_bet + (pot_after_call as f64 * frac).round() as u64
}

/// Parameters of [`RandomLegal`].
#[derive(Clone, Debug, PartialEq)]
pub struct RandomLegalParams {
    /// Relative weight of folding when facing a bet (checking when not).
    pub fold: f64,
    /// Relative weight of calling (checking when not facing a bet).
    pub call: f64,
    /// Relative weight of betting/raising.
    pub raise: f64,
    /// Bet/raise sizes as fractions of the pot after calling; one is picked uniformly.
    pub raise_sizes: Vec<f64>,
}

impl Default for RandomLegalParams {
    fn default() -> Self {
        RandomLegalParams {
            fold: 1.0,
            call: 2.0,
            raise: 1.0,
            raise_sizes: vec![0.5, 1.0],
        }
    }
}

/// Picks a random legal action: fold / call / raise with configurable weights, raise sizes as
/// pot fractions (clamped to the legal range). When it can check, the fold weight is spent on
/// checking instead (it never folds for free).
pub struct RandomLegal {
    name: String,
    p: RandomLegalParams,
    rng: Xoshiro256PlusPlus,
}

impl RandomLegal {
    /// Creates the bot.
    pub fn new(name: impl Into<String>, p: RandomLegalParams, seed: u64) -> Self {
        RandomLegal {
            name: name.into(),
            p,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
    }
}

impl Bot for RandomLegal {
    fn name(&self) -> &str {
        &self.name
    }
    fn act(&mut self, obs: &Observation) -> Action {
        let la = &obs.legal;
        let raise_w = if la.can_raise { self.p.raise } else { 0.0 };
        let total = self.p.fold + self.p.call + raise_w;
        let u = unit_f64(&mut self.rng) * total;
        if u < self.p.fold + self.p.call || total <= 0.0 {
            if la.can_check {
                Action::Check
            } else if u < self.p.fold {
                Action::Fold
            } else {
                Action::Call
            }
        } else {
            let sizes = &self.p.raise_sizes;
            let frac = if sizes.is_empty() {
                1.0
            } else {
                sizes[below(&mut self.rng, sizes.len() as u32) as usize]
            };
            clamp_raise(la, pot_fraction_to(obs, frac))
        }
    }
}

/// Always checks or calls; never folds or raises.
pub struct CallingStation {
    name: String,
}

impl CallingStation {
    /// Creates the bot.
    pub fn new(name: impl Into<String>) -> Self {
        CallingStation { name: name.into() }
    }
}

impl Bot for CallingStation {
    fn name(&self) -> &str {
        &self.name
    }
    fn act(&mut self, obs: &Observation) -> Action {
        if obs.legal.can_check {
            Action::Check
        } else {
            Action::Call
        }
    }
}
