//! The table runner: plays N hands at one table between bot instances.
//!
//! One table is one unit of parallel work. It owns its bots, its engine state and its RNG,
//! and shares no mutable state with other tables.

use crate::bot::{
    Action, BotFactory, HandResult, HandStart, MatchInfo, PublicEvent, SeatStatus, Street,
};
use crate::hand_history::HandHistoryWriter;
use crate::stats::SeatStats;
use engine::rng::mix_seed;
use engine::{Deck, GameFormat, HandSetup, HandState, MAX_SEATS};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// What happens to stacks between hands in cash mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum StackPolicy {
    /// Every hand starts with the starting stack (lowest variance; required by duplicate mode).
    #[default]
    ResetEachHand,
    /// Stacks carry over; a busted player is reloaded to the starting stack.
    RebuyWhenBusted,
    /// Stacks carry over; any stack below the starting stack is topped up to it.
    TopUpToStart,
}

/// What happens when a bot returns an illegal action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum IllegalActionPolicy {
    /// Replace with check if possible, otherwise fold; count it in stats.
    #[default]
    FoldOrCheck,
    /// Abort the table with an error (strict mode).
    Error,
}

/// Fixed blinds for cash games.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Blinds {
    /// Small blind.
    pub sb: u64,
    /// Big blind.
    pub bb: u64,
    /// Per-player ante.
    #[serde(default)]
    pub ante: u64,
}

/// One tournament blind level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlindLevel {
    /// Small blind.
    pub sb: u64,
    /// Big blind.
    pub bb: u64,
    /// Per-player ante.
    #[serde(default)]
    pub ante: u64,
    /// Hands played at this level (the last level lasts forever).
    pub hands: u64,
}

/// Cash game or tournament.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "mode")]
pub enum TableMode {
    /// Fixed blinds, stacks maintained per `stack_policy`.
    Cash {
        /// Blinds and ante.
        blinds: Blinds,
        /// Stack handling between hands.
        stack_policy: StackPolicy,
    },
    /// Increasing blinds; busted players are eliminated and the table shrinks.
    Tournament {
        /// Blind levels in order.
        levels: Vec<BlindLevel>,
    },
}

/// Table settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableConfig {
    /// Hands to play (cash), or the maximum number of hands (tournament). In duplicate mode
    /// this counts hands, rounded down to a multiple of the seat count (at least one deal).
    pub hands: u64,
    /// Starting stack in big blinds (of the first level in tournaments).
    pub starting_stack_bb: u64,
    /// Cash or tournament.
    pub mode: TableMode,
    /// Replay each deal with bots rotated through all seats (cash, reset stacks only).
    #[serde(default)]
    pub duplicate: bool,
    /// Illegal action handling.
    #[serde(default)]
    pub illegal_action_policy: IllegalActionPolicy,
    /// Record per-decision wall-clock time.
    #[serde(default)]
    pub record_timing: bool,
    /// Produce a text hand history for the table.
    #[serde(default)]
    pub hand_history: bool,
}

impl TableConfig {
    /// A cash config with reset stacks.
    pub fn cash(hands: u64, sb: u64, bb: u64, starting_stack_bb: u64) -> TableConfig {
        TableConfig {
            hands,
            starting_stack_bb,
            mode: TableMode::Cash {
                blinds: Blinds { sb, bb, ante: 0 },
                stack_policy: StackPolicy::ResetEachHand,
            },
            duplicate: false,
            illegal_action_policy: IllegalActionPolicy::FoldOrCheck,
            record_timing: false,
            hand_history: false,
        }
    }

    /// Reference big blind used for bb/100 (the first level's in tournaments).
    pub fn reference_bb(&self) -> u64 {
        match &self.mode {
            TableMode::Cash { blinds, .. } => blinds.bb,
            TableMode::Tournament { levels } => levels.first().map_or(1, |l| l.bb),
        }
    }

    /// Starting stack in chips.
    pub fn starting_stack(&self) -> u64 {
        self.starting_stack_bb * self.reference_bb()
    }

    /// True in tournament mode.
    pub fn is_tournament(&self) -> bool {
        matches!(self.mode, TableMode::Tournament { .. })
    }

    /// Checks consistency (duplicate mode needs reset stacks, tournaments need levels, ...).
    pub fn validate(&self) -> Result<(), TableError> {
        let err = |m: &str| Err(TableError::Config(m.to_string()));
        if self.hands == 0 {
            return err("hands must be > 0");
        }
        if self.starting_stack_bb == 0 {
            return err("starting_stack_bb must be > 0");
        }
        match &self.mode {
            TableMode::Cash {
                blinds,
                stack_policy,
            } => {
                if blinds.bb == 0 || blinds.sb > blinds.bb {
                    return err("need 0 <= sb <= bb and bb > 0");
                }
                if self.duplicate && *stack_policy != StackPolicy::ResetEachHand {
                    return err("duplicate mode requires stack_policy = reset_each_hand");
                }
            }
            TableMode::Tournament { levels } => {
                if levels.is_empty() {
                    return err("tournament needs at least one blind level");
                }
                if levels.iter().any(|l| l.bb == 0 || l.sb > l.bb) {
                    return err("every level needs 0 <= sb <= bb and bb > 0");
                }
                if self.duplicate {
                    return err("duplicate mode is not supported in tournaments");
                }
            }
        }
        Ok(())
    }
}

/// Result for one seat (one bot instance) at a table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeatResult {
    /// Index of the bot (into the factory list).
    pub bot: usize,
    /// Seat index (the initial seat in duplicate mode).
    pub seat: usize,
    /// Statistics.
    pub stats: SeatStats,
    /// Stack after the last hand.
    pub final_stack: u64,
    /// Finishing place in tournaments (1 = winner; tied players share a place).
    pub place: Option<u32>,
}

/// Result of a table.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TableResult {
    /// Hands played.
    pub hands_played: u64,
    /// Reference big blind for bb/100.
    pub bb: u64,
    /// True if this was a tournament table.
    pub tournament: bool,
    /// One entry per seat, in seat order.
    pub seats: Vec<SeatResult>,
    /// Text hand history, if enabled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hand_history: Option<String>,
    /// Hands captured through [`TableHooks::record`], in hand order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recorded_hands: Vec<RecordedHand>,
}

/// One captured hand history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecordedHand {
    /// Hand number within the table.
    pub hand_id: u64,
    /// PokerStars-like text.
    pub text: String,
}

/// Errors from running a table.
#[derive(Clone, Debug, PartialEq)]
pub enum TableError {
    /// Invalid configuration.
    Config(String),
    /// The engine rejected a hand setup.
    Setup(String),
    /// A bot made an illegal action under the strict policy.
    IllegalAction {
        /// Seat.
        seat: usize,
        /// Bot name.
        bot: String,
        /// Action as debug text.
        action: String,
        /// Why it was rejected.
        reason: String,
    },
    /// Stopped through [`TableHooks::cancel`].
    Cancelled,
}

impl std::fmt::Display for TableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TableError::Config(m) => write!(f, "table config: {m}"),
            TableError::Setup(m) => write!(f, "hand setup: {m}"),
            TableError::IllegalAction {
                seat,
                bot,
                action,
                reason,
            } => write!(
                f,
                "illegal action by {bot} (seat {seat}): {action} ({reason})"
            ),
            TableError::Cancelled => write!(f, "cancelled"),
        }
    }
}
impl std::error::Error for TableError {}

/// Seed stream identifiers for [`mix_seed`].
const STREAM_DECK: u64 = 0xDEC;
const STREAM_BOT: u64 = 0xB07;

struct Table<'a> {
    format: GameFormat,
    cfg: &'a TableConfig,
    bots: Vec<Box<dyn crate::bot::Bot>>,
    stats: Vec<SeatStats>,
    st: HandState,
    deck: Deck,
    rng: Xoshiro256PlusPlus,
    hh: Option<HandHistoryWriter>,
    names: Vec<String>,
    hooks: TableHooks<'a>,
    recorded: Vec<RecordedHand>,
}

/// Plays a table. `seats[i]` is the index into `factories` of the bot in seat `i`.
///
/// Bot instances are created fresh from their factories with seeds derived from `seed`, and
/// the deck RNG is seeded from `seed` too, so the result is a pure function of the inputs
/// (apart from `decision_ns` when timing is recorded).
pub fn run_table(
    format: &GameFormat,
    seats: &[usize],
    factories: &[BotFactory],
    cfg: &TableConfig,
    seed: u64,
) -> Result<TableResult, TableError> {
    run_table_with_hooks(format, seats, factories, cfg, seed, TableHooks::default())
}

/// Optional observers and controls for a running table. They never change results: a table
/// either completes identically to [`run_table`] or stops with [`TableError::Cancelled`].
#[derive(Clone, Copy, Default)]
pub struct TableHooks<'a> {
    /// Incremented after every finished hand.
    pub hands_done: Option<&'a AtomicU64>,
    /// Checked before every hand; when set the table stops with [`TableError::Cancelled`].
    pub cancel: Option<&'a AtomicBool>,
    /// Called after every finished hand (after `hands_done` is incremented).
    pub on_hand: Option<&'a (dyn Fn() + Sync)>,
    /// Sorted hand ids whose text hand history is captured into
    /// [`TableResult::recorded_hands`] (used for hand replays).
    pub record: Option<&'a [u64]>,
}

/// [`run_table`] with progress reporting and cooperative cancellation (see [`TableHooks`]).
pub fn run_table_with_hooks(
    format: &GameFormat,
    seats: &[usize],
    factories: &[BotFactory],
    cfg: &TableConfig,
    seed: u64,
    hooks: TableHooks<'_>,
) -> Result<TableResult, TableError> {
    cfg.validate()?;
    format
        .validate()
        .map_err(|e| TableError::Config(e.to_string()))?;
    let n = seats.len();
    if n < 2 || n > format.max_seats as usize {
        return Err(TableError::Config(format!(
            "{n} seats not allowed for a {}-max format",
            format.max_seats
        )));
    }
    if let Some(&b) = seats.iter().find(|&&b| b >= factories.len()) {
        return Err(TableError::Config(format!("unknown bot index {b}")));
    }

    let bots: Vec<Box<dyn crate::bot::Bot>> = seats
        .iter()
        .enumerate()
        .map(|(i, &b)| factories[b](mix_seed(seed, STREAM_BOT, i as u64)))
        .collect();
    let names = bots.iter().map(|b| b.name().to_string()).collect();
    let mut t = Table {
        format: *format,
        cfg,
        bots,
        stats: vec![SeatStats::default(); n],
        st: HandState::new(),
        deck: Deck::new(),
        rng: Xoshiro256PlusPlus::seed_from_u64(mix_seed(seed, STREAM_DECK, 0)),
        hh: cfg.hand_history.then(HandHistoryWriter::new),
        names,
        hooks,
        recorded: Vec::new(),
    };

    let start = cfg.starting_stack();
    let (sb, bb, ante) = match &cfg.mode {
        TableMode::Cash { blinds, .. } => (blinds.sb, blinds.bb, blinds.ante),
        TableMode::Tournament { levels } => (levels[0].sb, levels[0].bb, levels[0].ante),
    };
    for (i, b) in t.bots.iter_mut().enumerate() {
        b.new_match(&MatchInfo {
            format: *format,
            n_seats: n,
            seat: i,
            sb,
            bb,
            ante,
            starting_stack: start,
            hands: cfg.hands,
            tournament: cfg.is_tournament(),
            duplicate: cfg.duplicate,
        });
    }

    let (hands_played, final_stacks, places) = match &cfg.mode {
        TableMode::Cash {
            blinds,
            stack_policy,
        } => {
            let (h, s) = if cfg.duplicate {
                t.run_duplicate(n, *blinds, start)?
            } else {
                t.run_cash(n, *blinds, *stack_policy, start)?
            };
            (h, s, None)
        }
        TableMode::Tournament { levels } => {
            let (h, s, p) = t.run_tournament(n, levels, start)?;
            (h, s, Some(p))
        }
    };

    let seats_out = (0..n)
        .map(|i| SeatResult {
            bot: seats[i],
            seat: i,
            stats: t.stats[i].clone(),
            final_stack: final_stacks[i],
            place: places.as_ref().map(|p: &Vec<u32>| p[i]),
        })
        .collect();
    Ok(TableResult {
        hands_played,
        bb: cfg.reference_bb(),
        tournament: cfg.is_tournament(),
        seats: seats_out,
        hand_history: t.hh.map(|h| h.finish()),
        recorded_hands: t.recorded,
    })
}

impl Table<'_> {
    fn deal(&mut self, n_players: usize) {
        let cards = n_players * self.format.hole_cards as usize + 5;
        self.deck.shuffle_prefix(&mut self.rng, cards);
    }

    fn run_cash(
        &mut self,
        n: usize,
        blinds: Blinds,
        policy: StackPolicy,
        start: u64,
    ) -> Result<(u64, Vec<u64>), TableError> {
        let mut stacks = [0u64; MAX_SEATS];
        stacks[..n].fill(start);
        let identity: [usize; MAX_SEATS] = std::array::from_fn(|i| i);
        for h in 0..self.cfg.hands {
            for s in &mut stacks[..n] {
                match policy {
                    StackPolicy::ResetEachHand => *s = start,
                    StackPolicy::RebuyWhenBusted if *s == 0 => *s = start,
                    StackPolicy::TopUpToStart if *s < start => *s = start,
                    _ => {}
                }
            }
            let setup = HandSetup {
                n_seats: n,
                button: (h % n as u64) as usize,
                stacks,
                sb: blinds.sb,
                bb: blinds.bb,
                ante: blinds.ante,
            };
            self.deal(n);
            self.play_hand(h, &setup, &identity)?;
            stacks[..n].copy_from_slice(self.st.stacks());
        }
        Ok((self.cfg.hands, stacks[..n].to_vec()))
    }

    /// Duplicate mode: each deal is replayed `n` times; in replay `r` seat `s` is played by
    /// instance `(s + r) % n`, so every instance plays every seat's cards. Cards and button
    /// stay fixed within a deal; stacks are reset every hand.
    fn run_duplicate(
        &mut self,
        n: usize,
        blinds: Blinds,
        start: u64,
    ) -> Result<(u64, Vec<u64>), TableError> {
        let deals = (self.cfg.hands / n as u64).max(1);
        let mut stacks = [0u64; MAX_SEATS];
        stacks[..n].fill(start);
        let mut hand_id = 0;
        for d in 0..deals {
            let setup = HandSetup {
                n_seats: n,
                button: (d % n as u64) as usize,
                stacks,
                sb: blinds.sb,
                bb: blinds.bb,
                ante: blinds.ante,
            };
            self.deal(n);
            for r in 0..n {
                let map: [usize; MAX_SEATS] = std::array::from_fn(|s| (s + r) % n);
                self.play_hand(hand_id, &setup, &map)?;
                hand_id += 1;
            }
        }
        Ok((hand_id, vec![start; n]))
    }

    fn run_tournament(
        &mut self,
        n: usize,
        levels: &[BlindLevel],
        start: u64,
    ) -> Result<(u64, Vec<u64>, Vec<u32>), TableError> {
        let mut stacks = [0u64; MAX_SEATS];
        stacks[..n].fill(start);
        let mut place = vec![0u32; n];
        let identity: [usize; MAX_SEATS] = std::array::from_fn(|i| i);
        let mut button = 0usize;
        let (mut level, mut level_hands) = (0usize, 0u64);
        let mut h = 0;
        while h < self.cfg.hands {
            let alive = stacks[..n].iter().filter(|&&s| s > 0).count();
            if alive <= 1 {
                break;
            }
            if level + 1 < levels.len() && level_hands >= levels[level].hands {
                level += 1;
                level_hands = 0;
            }
            let lv = levels[level];
            if h > 0 {
                // Simplified moving button: next seat that still has chips.
                button = (1..=n)
                    .map(|k| (button + k) % n)
                    .find(|&s| stacks[s] > 0)
                    .unwrap();
            } else if stacks[button] == 0 {
                button = (0..n).find(|&s| stacks[s] > 0).unwrap();
            }
            let setup = HandSetup {
                n_seats: n,
                button,
                stacks,
                sb: lv.sb,
                bb: lv.bb,
                ante: lv.ante,
            };
            self.deal(alive);
            self.play_hand(h, &setup, &identity)?;
            let before = stacks;
            stacks[..n].copy_from_slice(self.st.stacks());
            // Players busted this hand take places alive_after+1 ..= alive_before, ordered by
            // their stack at the start of the hand (larger = better); equal stacks tie.
            let alive_after = stacks[..n].iter().filter(|&&s| s > 0).count() as u32;
            for s in 0..n {
                if before[s] > 0 && stacks[s] == 0 {
                    let better = (0..n)
                        .filter(|&o| before[o] > 0 && stacks[o] == 0 && before[o] > before[s])
                        .count() as u32;
                    place[s] = alive_after + 1 + better;
                }
            }
            h += 1;
            level_hands += 1;
        }
        // Survivors are ranked by chips (ties share a place).
        for s in 0..n {
            if stacks[s] > 0 {
                let better = (0..n).filter(|&o| stacks[o] > stacks[s]).count() as u32;
                place[s] = 1 + better;
            }
        }
        Ok((h, stacks[..n].to_vec(), place))
    }

    /// Plays one hand. `map[seat]` is the bot instance playing that seat.
    fn play_hand(
        &mut self,
        hand_id: u64,
        setup: &HandSetup,
        map: &[usize; MAX_SEATS],
    ) -> Result<(), TableError> {
        if self.hooks.cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return Err(TableError::Cancelled);
        }
        let n = setup.n_seats;
        self.st
            .start(&self.format, setup, &self.deck)
            .map_err(|e| TableError::Setup(e.to_string()))?;
        let in_hand = |st: &HandState, s: usize| st.status(s) != SeatStatus::Empty;
        for s in 0..n {
            if in_hand(&self.st, s) {
                self.bots[map[s]].new_hand(&HandStart {
                    hand_id,
                    seat: s,
                    hole: self.st.hole_cards(s),
                    button: setup.button,
                    stacks: &setup.stacks[..n],
                    sb: setup.sb,
                    bb: setup.bb,
                    ante: setup.ante,
                });
            }
        }
        let mut sent = 0;
        let mut board_sent = 0;
        self.broadcast_progress(&mut sent, &mut board_sent, map);

        while let Some(s) = self.st.to_act() {
            let inst = map[s];
            let obs = self.st.observation(s);
            let action = if self.cfg.record_timing {
                let t0 = Instant::now();
                let a = self.bots[inst].act(&obs);
                self.stats[inst].decision_ns += t0.elapsed().as_nanos() as u64;
                a
            } else {
                self.bots[inst].act(&obs)
            };
            self.stats[inst].decisions += 1;
            if let Err(e) = self.st.apply(action) {
                self.stats[inst].illegal += 1;
                match self.cfg.illegal_action_policy {
                    IllegalActionPolicy::Error => {
                        return Err(TableError::IllegalAction {
                            seat: s,
                            bot: self.names[inst].clone(),
                            action: format!("{action:?}"),
                            reason: e.to_string(),
                        })
                    }
                    IllegalActionPolicy::FoldOrCheck => {
                        let fallback = if self.st.legal_actions().can_check {
                            Action::Check
                        } else {
                            Action::Fold
                        };
                        self.st.apply(fallback).expect("fold/check is always legal");
                    }
                }
            }
            self.broadcast_progress(&mut sent, &mut board_sent, map);
        }

        // End of hand: uncalled bet, showdown reveals, payouts.
        let st = &self.st;
        let emit = |bots: &mut [Box<dyn crate::bot::Bot>], ev: &PublicEvent| {
            for s in 0..n {
                if in_hand(st, s) {
                    bots[map[s]].observe(ev);
                }
            }
        };
        if let Some((seat, amount)) = st.uncalled_return() {
            emit(
                &mut self.bots,
                &PublicEvent::UncalledReturned { seat, amount },
            );
        }
        if st.went_to_showdown() {
            for s in 0..n {
                if let Some(hole) = st.shown_cards(s) {
                    let ev = PublicEvent::Showdown {
                        seat: s,
                        hole,
                        strength: st.strength(s),
                    };
                    emit(&mut self.bots, &ev);
                }
            }
        }
        for s in 0..n {
            if st.won(s) > 0 {
                let ev = PublicEvent::PotAwarded {
                    seat: s,
                    amount: st.won(s),
                };
                emit(&mut self.bots, &ev);
            }
        }
        let mut net = [0i64; MAX_SEATS];
        for (s, v) in net.iter_mut().enumerate().take(n) {
            *v = st.net(s);
        }
        let result = HandResult {
            hand_id,
            net: &net[..n],
            stacks: st.stacks(),
            board: st.board(),
            showdown: st.went_to_showdown(),
        };
        for s in 0..n {
            if in_hand(st, s) {
                let inst = map[s];
                self.bots[inst].end_hand(&result);
                let stats = &mut self.stats[inst];
                stats.hands += 1;
                stats.net += net[s];
                stats.record_preflop(s, st.history());
                if st.went_to_showdown() && st.status(s).is_live() {
                    stats.showdowns += 1;
                    if st.won(s) > 0 {
                        stats.showdowns_won += 1;
                    }
                }
            }
        }
        if let Some(hh) = &mut self.hh {
            let names: [&str; MAX_SEATS] = std::array::from_fn(|s| {
                if s < n {
                    self.names[map[s]].as_str()
                } else {
                    ""
                }
            });
            hh.write_hand(hand_id, st, setup, &names[..n]);
        }
        if self
            .hooks
            .record
            .is_some_and(|r| r.binary_search(&hand_id).is_ok())
        {
            let names: [&str; MAX_SEATS] = std::array::from_fn(|s| {
                if s < n {
                    self.names[map[s]].as_str()
                } else {
                    ""
                }
            });
            let mut w = HandHistoryWriter::new();
            w.write_hand(hand_id, &self.st, setup, &names[..n]);
            self.recorded.push(RecordedHand {
                hand_id,
                text: w.finish(),
            });
        }
        if let Some(p) = self.hooks.hands_done {
            p.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(f) = self.hooks.on_hand {
            f();
        }
        Ok(())
    }

    /// Sends newly recorded actions and newly dealt streets to every bot in the hand.
    fn broadcast_progress(
        &mut self,
        sent: &mut usize,
        board_sent: &mut usize,
        map: &[usize; MAX_SEATS],
    ) {
        let st = &self.st;
        let n = st.n_seats();
        let hist = st.history();
        for rec in &hist[*sent..] {
            let ev = PublicEvent::Action(*rec);
            for s in 0..n {
                if st.status(s) != SeatStatus::Empty {
                    self.bots[map[s]].observe(&ev);
                }
            }
        }
        *sent = hist.len();
        let board = st.board();
        while *board_sent < board.len() {
            let len = if *board_sent == 0 { 3 } else { *board_sent + 1 };
            let street = match len {
                3 => Street::Flop,
                4 => Street::Turn,
                _ => Street::River,
            };
            let ev = PublicEvent::Street {
                street,
                board: &board[..len],
            };
            for s in 0..n {
                if st.status(s) != SeatStatus::Empty {
                    self.bots[map[s]].observe(&ev);
                }
            }
            *board_sent = len;
        }
    }
}
