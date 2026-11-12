//! Test-only bots for exercising the engine and runner from Rust. They are fixtures, not
//! baselines: the arena's baseline bots are Python (`pokerarena.baselines`).

#![allow(dead_code)]

use arena::bot::{Action, Bot, BotFactory, Observation};
use engine::rng::unit_f64;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::sync::Arc;

/// Random legal play with fold / call / raise weights; raises to a random size in range.
pub struct Random {
    name: String,
    w: (f64, f64, f64),
    rng: Xoshiro256PlusPlus,
}

impl Bot for Random {
    fn name(&self) -> &str {
        &self.name
    }
    fn act(&mut self, obs: &Observation) -> Action {
        let la = obs.legal;
        let (f, c, r) = self.w;
        let r = if la.can_raise { r } else { 0.0 };
        let u = unit_f64(&mut self.rng) * (f + c + r);
        if u < f + c {
            if la.can_check {
                Action::Check
            } else if u < f {
                Action::Fold
            } else {
                Action::Call
            }
        } else {
            let span = la.max_raise_to - la.min_raise_to;
            let x = (unit_f64(&mut self.rng) * unit_f64(&mut self.rng) * span as f64) as u64;
            Action::RaiseTo(la.min_raise_to + x)
        }
    }
}

/// Always checks or calls.
pub struct Station(String);
impl Bot for Station {
    fn name(&self) -> &str {
        &self.0
    }
    fn act(&mut self, obs: &Observation) -> Action {
        if obs.legal.can_check {
            Action::Check
        } else {
            Action::Call
        }
    }
}

/// Min-raises whenever it can, otherwise calls.
pub struct Raiser(String);
impl Bot for Raiser {
    fn name(&self) -> &str {
        &self.0
    }
    fn act(&mut self, obs: &Observation) -> Action {
        if obs.legal.can_raise {
            Action::RaiseTo(obs.legal.min_raise_to)
        } else {
            Action::Call
        }
    }
}

pub fn random_w(name: &str, fold: f64, call: f64, raise: f64) -> BotFactory {
    let name = name.to_string();
    Arc::new(move |seed| {
        Box::new(Random {
            name: name.clone(),
            w: (fold, call, raise),
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        })
    })
}
pub fn random(name: &str) -> BotFactory {
    random_w(name, 1.0, 2.0, 1.0)
}
pub fn station(name: &str) -> BotFactory {
    let name = name.to_string();
    Arc::new(move |_| Box::new(Station(name.clone())))
}
pub fn raiser(name: &str) -> BotFactory {
    let name = name.to_string();
    Arc::new(move |_| Box::new(Raiser(name.clone())))
}
