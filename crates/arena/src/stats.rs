//! Per-seat statistics collected by the table runner.

use engine::action::{ActionKind, ActionRecord, Street};
use serde::{Deserialize, Serialize};

/// Statistics for one seat (one bot instance) at one table.
///
/// Money is kept in integer chips; bb/100 is derived for reporting only.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeatStats {
    /// Hands dealt in.
    pub hands: u64,
    /// Net chips won (+) or lost (−).
    pub net: i64,
    /// Hands where the player voluntarily put chips in preflop (call, bet or raise).
    pub vpip: u64,
    /// Hands where the player bet or raised preflop.
    pub pfr: u64,
    /// Hands that reached showdown with this player still in.
    pub showdowns: u64,
    /// Showdowns where the player won chips.
    pub showdowns_won: u64,
    /// Illegal actions submitted (replaced per the illegal-action policy).
    pub illegal: u64,
    /// Decisions made (calls to `act`).
    pub decisions: u64,
    /// Total decision time in nanoseconds (0 unless timing is recorded). Not deterministic.
    pub decision_ns: u64,
}

impl SeatStats {
    /// Adds another set of statistics into this one.
    pub fn merge(&mut self, o: &SeatStats) {
        self.hands += o.hands;
        self.net += o.net;
        self.vpip += o.vpip;
        self.pfr += o.pfr;
        self.showdowns += o.showdowns;
        self.showdowns_won += o.showdowns_won;
        self.illegal += o.illegal;
        self.decisions += o.decisions;
        self.decision_ns += o.decision_ns;
    }

    /// Big blinds won per 100 hands, relative to `bb` chips.
    pub fn bb_per_100(&self, bb: u64) -> f64 {
        if self.hands == 0 {
            0.0
        } else {
            self.net as f64 / bb as f64 / self.hands as f64 * 100.0
        }
    }

    /// Fraction of hands with VPIP.
    pub fn vpip_rate(&self) -> f64 {
        ratio(self.vpip, self.hands)
    }
    /// Fraction of hands with a preflop raise.
    pub fn pfr_rate(&self) -> f64 {
        ratio(self.pfr, self.hands)
    }
    /// Fraction of hands that reached showdown.
    pub fn showdown_rate(&self) -> f64 {
        ratio(self.showdowns, self.hands)
    }
    /// Mean decision time in microseconds (0 if not recorded).
    pub fn mean_decision_us(&self) -> f64 {
        if self.decisions == 0 {
            0.0
        } else {
            self.decision_ns as f64 / self.decisions as f64 / 1000.0
        }
    }

    /// Records the preflop VPIP/PFR flags of `seat` from a hand's action history.
    pub(crate) fn record_preflop(&mut self, seat: usize, history: &[ActionRecord]) {
        let (mut vpip, mut pfr) = (false, false);
        for r in history {
            if r.street != Street::Preflop {
                break;
            }
            if r.seat as usize == seat {
                match r.kind {
                    ActionKind::Call => vpip = true,
                    ActionKind::Bet | ActionKind::Raise => {
                        vpip = true;
                        pfr = true;
                    }
                    _ => {}
                }
            }
        }
        self.vpip += vpip as u64;
        self.pfr += pfr as u64;
    }
}

fn ratio(a: u64, b: u64) -> f64 {
    if b == 0 {
        0.0
    } else {
        a as f64 / b as f64
    }
}
