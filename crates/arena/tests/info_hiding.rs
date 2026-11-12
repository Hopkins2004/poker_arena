//! Inspects everything bots receive across many hands and asserts that no other seat's hole
//! cards are visible before showdown, and that only live players are revealed at showdown.

use arena::bot::{Action, Bot, Card, HandStart, Observation, PublicEvent};
use arena::{run_table, BotFactory, HandResult, TableConfig};
use engine::action::ActionKind;
use engine::rng::below;
use engine::GameFormat;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

#[derive(Default, Debug)]
struct HandLog {
    seat: usize,
    hole: Vec<Card>,
    /// Every card seen through observations, street events and hand start before showdown.
    seen_before_showdown: HashSet<Card>,
    obs_holes: Vec<Vec<Card>>,
    folded: HashSet<usize>,
    revealed: Vec<(usize, Vec<Card>)>,
    showdown_flag: bool,
    saw_showdown_event: bool,
}

type Log = Arc<Mutex<HashMap<(usize, u64), HandLog>>>;

struct Spy {
    id: usize,
    log: Log,
    hand: u64,
    rng: Xoshiro256PlusPlus,
    cur: HandLog,
}

impl Bot for Spy {
    fn name(&self) -> &str {
        "spy"
    }
    fn new_hand(&mut self, info: &HandStart) {
        self.hand = info.hand_id;
        self.cur = HandLog {
            seat: info.seat,
            hole: info.hole.to_vec(),
            ..Default::default()
        };
        self.cur.seen_before_showdown.extend(info.hole);
    }
    fn act(&mut self, obs: &Observation) -> Action {
        assert_eq!(obs.seat, self.cur.seat);
        self.cur.obs_holes.push(obs.hole.to_vec());
        self.cur.seen_before_showdown.extend(obs.hole);
        self.cur.seen_before_showdown.extend(obs.board);
        let la = obs.legal;
        match below(&mut self.rng, 5) {
            0 if !la.can_check => Action::Fold,
            1 if la.can_raise => Action::RaiseTo(la.min_raise_to),
            2 if la.can_raise => Action::AllIn,
            _ => Action::Call,
        }
    }
    fn observe(&mut self, ev: &PublicEvent) {
        match ev {
            PublicEvent::Action(r) if r.kind == ActionKind::Fold => {
                self.cur.folded.insert(r.seat as usize);
            }
            PublicEvent::Street { board, .. } => {
                assert!(!self.cur.saw_showdown_event);
                self.cur.seen_before_showdown.extend(*board);
            }
            PublicEvent::Showdown { seat, hole, .. } => {
                self.cur.saw_showdown_event = true;
                self.cur.revealed.push((*seat, hole.to_vec()));
            }
            _ => {}
        }
    }
    fn end_hand(&mut self, r: &HandResult) {
        self.cur.showdown_flag = r.showdown;
        let cur = std::mem::take(&mut self.cur);
        self.log.lock().unwrap().insert((self.id, self.hand), cur);
    }
}

fn check(format: GameFormat, n: usize, cfg: TableConfig) {
    let log: Log = Arc::default();
    let counter = Arc::new(Mutex::new(0usize));
    let factory: BotFactory = {
        let log = log.clone();
        Arc::new(move |seed| {
            let mut c = counter.lock().unwrap();
            *c += 1;
            Box::new(Spy {
                id: *c - 1,
                log: log.clone(),
                hand: 0,
                rng: Xoshiro256PlusPlus::seed_from_u64(seed),
                cur: HandLog::default(),
            })
        })
    };
    let res = run_table(&format, &vec![0; n], &[factory], &cfg, 99).unwrap();
    let log = log.lock().unwrap();
    let mut showdowns = 0;
    for h in 0..res.hands_played {
        let logs: Vec<&HandLog> = (0..n).filter_map(|i| log.get(&(i, h))).collect();
        assert_eq!(logs.len(), n, "hand {h}");
        for me in &logs {
            for oh in &me.obs_holes {
                assert_eq!(oh, &me.hole, "observation shows a different hole");
            }
            for other in logs.iter().filter(|o| o.seat != me.seat) {
                for c in &other.hole {
                    assert!(
                        !me.seen_before_showdown.contains(c),
                        "hand {h}: seat {} saw seat {}'s card {c} before showdown",
                        me.seat,
                        other.seat
                    );
                }
            }
            assert_eq!(me.saw_showdown_event, me.showdown_flag);
            if me.showdown_flag {
                showdowns += 1;
            }
            for (seat, cards) in &me.revealed {
                assert!(!me.folded.contains(seat), "folded seat {seat} was revealed");
                let owner = logs.iter().find(|o| o.seat == *seat).unwrap();
                assert_eq!(&owner.hole, cards);
            }
        }
    }
    assert!(showdowns > 0, "test should exercise showdowns");
}

#[test]
fn no_hole_card_leaks_nlhe() {
    check(GameFormat::nlhe(6), 6, TableConfig::cash(400, 1, 2, 100));
}

#[test]
fn no_hole_card_leaks_plo_heads_up() {
    check(GameFormat::plo(6), 2, TableConfig::cash(300, 1, 2, 100));
}

#[test]
fn no_hole_card_leaks_duplicate_mode() {
    let mut cfg = TableConfig::cash(300, 1, 2, 100);
    cfg.duplicate = true;
    check(GameFormat::nlhe(6), 3, cfg);
}
