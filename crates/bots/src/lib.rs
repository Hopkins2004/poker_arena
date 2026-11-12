//! Baseline bots and a registry that builds bot factories from a kind name plus parameters.
//!
//! Registered kinds:
//! * `random` — [`RandomLegal`]. Params: `fold`, `call`, `raise` (weights), `raise_sizes`
//!   (pot fractions).
//! * `calling_station` — [`CallingStation`]. No params.
//! * `tag` — [`TightAggressive`]. Params: `samples`, `tightness`, `call_margin`.

#![warn(missing_docs)]

pub mod equity;
pub mod simple;
pub mod tag;

use arena::bot::BotFactory;
use serde_json::Value;
use std::sync::Arc;

pub use simple::{CallingStation, RandomLegal, RandomLegalParams};
pub use tag::{TagParams, TightAggressive};

/// Names of the registered Rust bot kinds.
pub const KINDS: &[&str] = &["random", "calling_station", "tag"];

fn get_f64(params: &Value, key: &str, default: f64) -> Result<f64, String> {
    match params.get(key) {
        None => Ok(default),
        Some(v) => v
            .as_f64()
            .ok_or_else(|| format!("param `{key}` must be a number")),
    }
}

fn check_keys(params: &Value, allowed: &[&str]) -> Result<(), String> {
    match params {
        Value::Null => Ok(()),
        Value::Object(m) => {
            for k in m.keys() {
                if !allowed.contains(&k.as_str()) {
                    return Err(format!("unknown param `{k}` (allowed: {allowed:?})"));
                }
            }
            Ok(())
        }
        _ => Err("params must be a table".into()),
    }
}

/// Builds a factory for a registered bot `kind`, named `name`, with `params` (a JSON object
/// or null). Each call of the factory creates a fresh, independently seeded instance.
pub fn factory(kind: &str, name: &str, params: &Value) -> Result<BotFactory, String> {
    let name = name.to_string();
    match kind {
        "random" => {
            check_keys(params, &["fold", "call", "raise", "raise_sizes"])?;
            let d = RandomLegalParams::default();
            let raise_sizes = match params.get("raise_sizes") {
                None => d.raise_sizes.clone(),
                Some(Value::Array(a)) => a
                    .iter()
                    .map(|v| v.as_f64().ok_or("raise_sizes must be numbers"))
                    .collect::<Result<_, _>>()?,
                Some(_) => return Err("raise_sizes must be an array".into()),
            };
            let p = RandomLegalParams {
                fold: get_f64(params, "fold", d.fold)?,
                call: get_f64(params, "call", d.call)?,
                raise: get_f64(params, "raise", d.raise)?,
                raise_sizes,
            };
            if p.fold < 0.0 || p.call < 0.0 || p.raise < 0.0 || p.fold + p.call + p.raise <= 0.0 {
                return Err("weights must be >= 0 with a positive sum".into());
            }
            Ok(Arc::new(move |seed| {
                Box::new(RandomLegal::new(name.clone(), p.clone(), seed))
            }))
        }
        "calling_station" => {
            check_keys(params, &[])?;
            Ok(Arc::new(move |_| {
                Box::new(CallingStation::new(name.clone()))
            }))
        }
        "tag" => {
            check_keys(params, &["samples", "tightness", "call_margin"])?;
            let d = TagParams::default();
            let p = TagParams {
                samples: get_f64(params, "samples", d.samples as f64)? as u32,
                tightness: get_f64(params, "tightness", d.tightness)?,
                call_margin: get_f64(params, "call_margin", d.call_margin)?,
            };
            if p.samples == 0 {
                return Err("samples must be > 0".into());
            }
            Ok(Arc::new(move |seed| {
                Box::new(TightAggressive::new(name.clone(), p.clone(), seed))
            }))
        }
        other => Err(format!("unknown bot kind `{other}` (known: {KINDS:?})")),
    }
}
