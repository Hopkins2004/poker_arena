//! The example bot from BOTS.md, compiled and run here so the guide stays correct.

use std::sync::Arc;

use arena::bot::{Action, ActionKind, Bot, BotFactory, HandStart, Observation, PublicEvent};
use arena::{run, BotEntry, MatchmakingConfig, RatingConfig, RunSpec, TableConfig};
use bots::equity::equity;
use engine::eval::Evaluator;
use engine::GameFormat;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

// --- BEGIN GUIDE EXAMPLE ---------------------------------------------------------------
/// Calls when its Monte Carlo equity beats the pot odds, value-raises strong hands,
/// and plays tighter when the hand has already been raised twice.
pub struct PotOdds {
    name: String,
    samples: u32,
    rng: Xoshiro256PlusPlus,
    ev: Evaluator,
    raises_this_hand: u32, // reset in new_hand: never carry hand state across hands
}

impl PotOdds {
    pub fn new(name: &str, samples: u32, seed: u64) -> Self {
        PotOdds {
            name: name.to_string(),
            samples,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed), // all randomness from the seed
            ev: Evaluator::get(),
            raises_this_hand: 0,
        }
    }

    /// A factory: the arena calls it once per seat per table with a derived seed.
    pub fn factory(name: &str, samples: u32) -> BotFactory {
        let name = name.to_string();
        Arc::new(move |seed| Box::new(PotOdds::new(&name, samples, seed)))
    }
}

impl Bot for PotOdds {
    fn name(&self) -> &str {
        &self.name
    }

    fn new_hand(&mut self, _info: &HandStart) {
        self.raises_this_hand = 0;
    }

    fn observe(&mut self, event: &PublicEvent) {
        if let PublicEvent::Action(rec) = event {
            if rec.kind == ActionKind::Raise {
                self.raises_this_hand += 1;
            }
        }
    }

    fn act(&mut self, obs: &Observation) -> Action {
        let la = obs.legal;
        // Opponents still in the hand (status Active or AllIn, excluding me).
        let opponents = obs
            .status
            .iter()
            .enumerate()
            .filter(|&(s, st)| s != obs.seat && st.is_live())
            .count();
        let e = equity(
            self.ev,
            obs.format.variant,
            obs.hole,
            obs.board,
            opponents,
            self.samples,
            &mut self.rng,
        );
        let threshold = if self.raises_this_hand >= 2 {
            0.85
        } else {
            0.70
        };
        if la.can_raise && e >= threshold {
            // Raise-to = current bet + 2/3 of the pot after calling, clamped to the legal range.
            let to = obs.current_bet + (obs.pot + la.call_amount) * 2 / 3;
            return Action::RaiseTo(to.clamp(la.min_raise_to, la.max_raise_to));
        }
        if la.can_check {
            return Action::Check;
        }
        let pot_odds = la.call_amount as f64 / (obs.pot + la.call_amount) as f64;
        if e >= pot_odds {
            Action::Call
        } else {
            Action::Fold
        }
    }
}
// --- END GUIDE EXAMPLE -----------------------------------------------------------------

#[test]
fn guide_example_bot_runs_legally() {
    let bots = vec![
        BotEntry {
            name: "pot_odds".into(),
            version: None,
            factory: PotOdds::factory("pot_odds", 100),
        },
        BotEntry {
            name: "tag".into(),
            version: None,
            factory: bots::factory("tag", "tag", &serde_json::Value::Null).unwrap(),
        },
        BotEntry {
            name: "station".into(),
            version: None,
            factory: bots::factory("calling_station", "station", &serde_json::Value::Null).unwrap(),
        },
    ];
    let spec = RunSpec {
        format: GameFormat::nlhe(6),
        table: TableConfig::cash(300, 1, 2, 100),
        matchmaking: MatchmakingConfig::default(),
        rating: RatingConfig::default(),
        rounds: 2,
        seed: 1,
        threads: Some(2),
    };
    let out = run(&spec, &bots, vec![1500.0; 3], true).unwrap();
    let rows = out.state.rows();
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|r| r.illegal == 0 && r.hands > 0));
    // It should comfortably beat a calling station.
    let get = |k: &str| rows.iter().find(|r| r.key == k).unwrap().bb_per_100;
    assert!(get("pot_odds") > get("station"));
}

#[test]
fn guide_unit_test_snippet() {
    // --- BEGIN GUIDE UNIT TEST ---------------------------------------------------------
    use engine::{DeckBuilder, GameFormat, HandSetup, HandState, MAX_SEATS};

    let fmt = GameFormat::nlhe(6);
    let mut stacks = [0u64; MAX_SEATS];
    stacks[..2].fill(200); // two players, 100 bb each
    let setup = HandSetup {
        n_seats: 2,
        button: 0,
        stacks,
        sb: 1,
        bb: 2,
        ante: 0,
    };
    // Deal order starts left of the button: seat 1 then seat 0.
    let deck = DeckBuilder::new()
        .hole(0, "AsAh")
        .hole(1, "7c2d")
        .build(&[1, 0], 2);
    let mut st = HandState::new();
    st.start(&fmt, &setup, &deck).unwrap();

    let mut bot = PotOdds::new("pot_odds", 500, 42);
    let seat = st.to_act().unwrap(); // heads-up, the button (seat 0) acts first preflop
    let action = bot.act(&st.observation(seat));
    assert!(
        st.validate(action).is_ok(),
        "bot chose an illegal action: {action:?}"
    );
    assert!(
        matches!(action, Action::RaiseTo(_)),
        "aces should raise, got {action:?}"
    );
    // --- END GUIDE UNIT TEST -----------------------------------------------------------
}
