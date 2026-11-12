//! Table-runner tests using the baseline bots.

use arena::bot::{Action, Bot, Observation};
use arena::{
    run_table, BlindLevel, BotFactory, IllegalActionPolicy, StackPolicy, TableConfig, TableError,
    TableMode,
};
use engine::GameFormat;
use serde_json::json;
use std::sync::Arc;

fn f(kind: &str, name: &str) -> BotFactory {
    bots::factory(kind, name, &serde_json::Value::Null).unwrap()
}

fn lineup() -> Vec<BotFactory> {
    vec![
        f("random", "random"),
        f("calling_station", "station"),
        f("tag", "tag"),
    ]
}

#[test]
fn cash_table_conserves_chips_and_is_deterministic() {
    let fmt = GameFormat::nlhe(6);
    let cfg = TableConfig::cash(500, 1, 2, 100);
    let seats = [0, 1, 2, 0, 1, 2];
    let a = run_table(&fmt, &seats, &lineup(), &cfg, 5).unwrap();
    let b = run_table(&fmt, &seats, &lineup(), &cfg, 5).unwrap();
    let c = run_table(&fmt, &seats, &lineup(), &cfg, 6).unwrap();
    assert_eq!(a, b, "same seed must give identical results");
    assert_ne!(a, c, "different seeds should differ");
    assert_eq!(a.hands_played, 500);
    let total: i64 = a.seats.iter().map(|s| s.stats.net).sum();
    assert_eq!(total, 0, "net chips must sum to zero");
    for s in &a.seats {
        assert_eq!(s.stats.hands, 500);
        assert_eq!(s.stats.illegal, 0, "baseline bots must act legally");
        assert!(s.stats.pfr <= s.stats.vpip);
    }
    // Calling station never raises; it is in every showdown it reaches.
    assert!(a
        .seats
        .iter()
        .filter(|s| s.bot == 1)
        .all(|s| s.stats.pfr == 0));
}

#[test]
fn stack_policies_conserve_chips() {
    let fmt = GameFormat::nlhe(6);
    for policy in [StackPolicy::RebuyWhenBusted, StackPolicy::TopUpToStart] {
        let mut cfg = TableConfig::cash(400, 1, 2, 20);
        if let TableMode::Cash { stack_policy, .. } = &mut cfg.mode {
            *stack_policy = policy;
        }
        let r = run_table(&fmt, &[0, 1, 0, 1], &lineup(), &cfg, 11).unwrap();
        let total: i64 = r.seats.iter().map(|s| s.stats.net).sum();
        assert_eq!(total, 0);
        for s in &r.seats {
            assert_eq!(s.stats.hands, 400);
        }
    }
}

#[test]
fn plo_and_flhe_tables_run() {
    let cfg = TableConfig::cash(300, 1, 2, 100);
    for fmt in [GameFormat::plo(6), GameFormat::flhe(6, 2, 4)] {
        let r = run_table(&fmt, &[0, 1, 2, 2], &lineup(), &cfg, 3).unwrap();
        assert_eq!(r.seats.iter().map(|s| s.stats.net).sum::<i64>(), 0);
        assert!(r.seats.iter().all(|s| s.stats.illegal == 0));
    }
}

#[test]
fn tournament_eliminates_until_one_left() {
    let fmt = GameFormat::nlhe(6);
    let cfg = TableConfig {
        hands: 100_000,
        starting_stack_bb: 50,
        mode: TableMode::Tournament {
            levels: vec![
                BlindLevel {
                    sb: 1,
                    bb: 2,
                    ante: 0,
                    hands: 30,
                },
                BlindLevel {
                    sb: 2,
                    bb: 4,
                    ante: 1,
                    hands: 30,
                },
                BlindLevel {
                    sb: 5,
                    bb: 10,
                    ante: 1,
                    hands: 30,
                },
                BlindLevel {
                    sb: 10,
                    bb: 20,
                    ante: 2,
                    hands: 30,
                },
                BlindLevel {
                    sb: 25,
                    bb: 50,
                    ante: 5,
                    hands: 30,
                },
            ],
        },
        ..TableConfig::cash(1, 1, 2, 1)
    };
    let r = run_table(&fmt, &[0, 1, 2, 0, 1, 2], &lineup(), &cfg, 21).unwrap();
    assert!(r.tournament);
    assert!(r.hands_played < 100_000);
    let total_chips: u64 = r.seats.iter().map(|s| s.final_stack).sum();
    assert_eq!(total_chips, 6 * 100);
    let mut places: Vec<u32> = r.seats.iter().map(|s| s.place.unwrap()).collect();
    places.sort();
    assert_eq!(places[0], 1);
    assert_eq!(places[1], 2, "exactly one winner");
    let winner = r.seats.iter().find(|s| s.place == Some(1)).unwrap();
    assert_eq!(winner.final_stack, 600);
    assert!(places.iter().all(|&p| (1..=6).contains(&p)));
}

#[test]
fn duplicate_mode_identical_bots_net_zero() {
    // With identical deterministic bots, every instance plays every seat's cards of every
    // deal under the same circumstances, so each instance's net is exactly zero.
    let fmt = GameFormat::nlhe(6);
    let mut cfg = TableConfig::cash(600, 1, 2, 100);
    cfg.duplicate = true;
    let r = run_table(&fmt, &[1, 1, 1, 1], &lineup(), &cfg, 8).unwrap();
    assert_eq!(r.hands_played, 600);
    for s in &r.seats {
        assert_eq!(s.stats.hands, 600);
        assert_eq!(s.stats.net, 0);
    }
    // Different bots: nets need not be zero but must still sum to zero.
    let r = run_table(&fmt, &[0, 1, 2], &lineup(), &cfg, 8).unwrap();
    assert_eq!(r.hands_played, 600);
    assert_eq!(r.seats.iter().map(|s| s.stats.net).sum::<i64>(), 0);
}

#[test]
fn duplicate_mode_rejects_carry_over_stacks() {
    let mut cfg = TableConfig::cash(10, 1, 2, 100);
    cfg.duplicate = true;
    if let TableMode::Cash { stack_policy, .. } = &mut cfg.mode {
        *stack_policy = StackPolicy::RebuyWhenBusted;
    }
    let e = run_table(&GameFormat::nlhe(6), &[0, 1], &lineup(), &cfg, 1).unwrap_err();
    assert!(matches!(e, TableError::Config(_)));
}

struct Cheater;
impl Bot for Cheater {
    fn name(&self) -> &str {
        "cheater"
    }
    fn act(&mut self, _obs: &Observation) -> Action {
        Action::RaiseTo(1)
    }
}

#[test]
fn illegal_actions_counted_or_rejected() {
    let fmt = GameFormat::nlhe(6);
    let mut facts = lineup();
    facts.push(Arc::new(|_| Box::new(Cheater)));
    let cfg = TableConfig::cash(50, 1, 2, 100);
    let r = run_table(&fmt, &[3, 1], &facts, &cfg, 2).unwrap();
    let cheater = &r.seats[0];
    assert!(cheater.stats.illegal > 0);
    assert_eq!(cheater.stats.illegal, cheater.stats.decisions);
    assert_eq!(r.seats[1].stats.illegal, 0);

    let mut strict = cfg.clone();
    strict.illegal_action_policy = IllegalActionPolicy::Error;
    let e = run_table(&fmt, &[3, 1], &facts, &strict, 2).unwrap_err();
    assert!(matches!(e, TableError::IllegalAction { .. }), "{e}");
}

#[test]
fn timing_and_hand_history() {
    let fmt = GameFormat::nlhe(6);
    let mut cfg = TableConfig::cash(20, 1, 2, 100);
    cfg.record_timing = true;
    cfg.hand_history = true;
    let r = run_table(&fmt, &[0, 1, 2], &lineup(), &cfg, 4).unwrap();
    assert!(r.seats.iter().all(|s| s.stats.decision_ns > 0));
    let hh = r.hand_history.unwrap();
    assert_eq!(hh.matches("Hand #").count(), 20);
    assert!(hh.contains("*** HOLE CARDS ***"));
    assert!(hh.contains("posts big blind 2"));
}

#[test]
fn registry_params() {
    assert!(bots::factory(
        "random",
        "r",
        &json!({"fold": 0, "call": 1, "raise": 3, "raise_sizes": [0.33, 2.0]})
    )
    .is_ok());
    assert!(bots::factory("random", "r", &json!({"bogus": 1})).is_err());
    assert!(bots::factory("tag", "t", &json!({"samples": 100, "tightness": 1.5})).is_ok());
    assert!(bots::factory("nope", "x", &json!(null)).is_err());
}
