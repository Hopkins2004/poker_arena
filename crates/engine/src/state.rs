//! The hand state machine: blinds and antes, dealing, betting rounds, all-in runouts, side
//! pots, showdown and pot distribution.
//!
//! A [`HandState`] is reusable: call [`HandState::start`] for every hand; it resets its fixed
//! arrays and clears (without freeing) its history buffer, so the hot path does not allocate.
//!
//! Everything that differs between formats (showdown evaluation, raise sizes, caps, action
//! validation) is delegated to [`crate::rules`]; this file is the format-independent flow.
//!
//! Rules implemented (each is covered by scripted tests in `tests/scripted.rs`):
//! * Heads-up, the button posts the small blind, acts first preflop and last postflop.
//! * Minimum raise: the increment must be at least the largest previous full bet/raise
//!   increment on the street; preflop the big blind counts as the opening bet.
//! * Short all-ins: an all-in raise below a full raise does not reopen betting for players
//!   who already acted, unless the total they face since they last acted is at least a full
//!   raise (TDA cumulative rule).
//! * Uncalled bets are returned before showdown.
//! * Side pots are built from the all-in contribution levels; each pot is awarded among the
//!   eligible non-folded players. Odd chips go to the first eligible winner left of the button.
//! * When at most one player can still bet, the board is run out with no further action.
//! * Pot limit: maximum raise-to = current bet + (pot after calling).
//! * Fixed limit: fixed sizes per street, capped number of bets + raises.

use crate::action::{Action, ActionKind, ActionRecord, IllegalAction, LegalActions, Street};
use crate::card::Card;
use crate::deck::Deck;
use crate::eval::{Evaluator, Strength};
use crate::format::{GameFormat, MAX_HOLE, MAX_SEATS};
use crate::rules::actions::{legal_actions, resolve, PlayerView, Resolved};
use crate::rules::betting::{BetContext, BettingRules};

/// Per-seat status within a hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum SeatStatus {
    /// No player in this seat for this hand (empty, or zero stack).
    #[default]
    Empty,
    /// In the hand with chips behind.
    Active,
    /// In the hand with no chips behind.
    AllIn,
    /// Folded.
    Folded,
}

impl SeatStatus {
    /// In the hand and not folded.
    #[inline]
    pub fn is_live(self) -> bool {
        matches!(self, SeatStatus::Active | SeatStatus::AllIn)
    }
}

/// Parameters of one hand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandSetup {
    /// Number of seats at the table (`2..=format.max_seats`).
    pub n_seats: usize,
    /// Button seat; must hold a player.
    pub button: usize,
    /// Starting stack per seat; zero means the seat is empty / sitting out.
    pub stacks: [u64; MAX_SEATS],
    /// Small blind.
    pub sb: u64,
    /// Big blind (> 0).
    pub bb: u64,
    /// Per-player ante (0 for none).
    pub ante: u64,
}

/// Error starting a hand.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SetupError(pub String);
impl std::fmt::Display for SetupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid hand setup: {}", self.0)
    }
}
impl std::error::Error for SetupError {}

/// One (main or side) pot after settlement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Pot {
    /// Chips in the pot.
    pub amount: u64,
    /// Bitmask of seats eligible to win it.
    pub eligible: u16,
    /// Bitmask of seats that won (split) it.
    pub winners: u16,
}

/// A player's view of the hand. Built only by [`HandState::observation`], which exposes the
/// observing seat's own hole cards and public information only.
#[derive(Clone, Copy, Debug)]
pub struct Observation<'a> {
    /// Observing seat.
    pub seat: usize,
    /// Seats at the table.
    pub n_seats: usize,
    /// Button seat.
    pub button: usize,
    /// Game format.
    pub format: GameFormat,
    /// Current street.
    pub street: Street,
    /// The observer's own hole cards.
    pub hole: &'a [Card],
    /// Visible board cards.
    pub board: &'a [Card],
    /// Chips behind for every seat.
    pub stacks: &'a [u64],
    /// Contribution of every seat on the current street.
    pub street_contrib: &'a [u64],
    /// Contribution of every seat over the whole hand (antes and blinds included).
    pub total_contrib: &'a [u64],
    /// Status of every seat.
    pub status: &'a [SeatStatus],
    /// Total chips in the pot, including current-street bets.
    pub pot: u64,
    /// Street total that must be matched to call.
    pub current_bet: u64,
    /// Small blind, big blind and ante for this hand.
    pub sb: u64,
    /// Big blind.
    pub bb: u64,
    /// Ante.
    pub ante: u64,
    /// Legal actions (meaningful when `seat` is the player to act).
    pub legal: LegalActions,
    /// Public action history of the hand so far, including blinds and antes.
    pub history: &'a [ActionRecord],
}

/// State of a single hand. See the module docs for the rules implemented.
#[derive(Clone, Debug)]
pub struct HandState {
    fmt: GameFormat,
    ev: Evaluator,
    n_seats: usize,
    button: usize,
    sb_seat: usize,
    bb_seat: usize,
    sb: u64,
    bb: u64,
    ante: u64,

    start_stacks: [u64; MAX_SEATS],
    stacks: [u64; MAX_SEATS],
    street_contrib: [u64; MAX_SEATS],
    total_contrib: [u64; MAX_SEATS],
    /// Running sum of `total_contrib` (kept in sync wherever chips move; checked in debug).
    pot_total: u64,
    status: [SeatStatus; MAX_SEATS],
    hole: [[Card; MAX_HOLE]; MAX_SEATS],
    board: [Card; 5],
    board_len: usize,

    street: Street,
    current_bet: u64,
    /// Size of the last full bet/raise increment on this street.
    last_raise: u64,
    /// Number of full bets + raises this street (fixed-limit cap).
    bets_this_street: u8,
    acted: [bool; MAX_SEATS],
    /// `current_bet` right after the seat last acted on this street.
    acted_at: [u64; MAX_SEATS],
    needs_action: [bool; MAX_SEATS],
    to_act: Option<usize>,
    complete: bool,

    history: Vec<ActionRecord>,

    // Results, valid once `complete`.
    pots: [Pot; MAX_SEATS],
    n_pots: usize,
    showdown: bool,
    strengths: [Strength; MAX_SEATS],
    uncalled: Option<(usize, u64)>,
    won: [u64; MAX_SEATS],
}

impl Default for HandState {
    fn default() -> Self {
        Self::new()
    }
}

impl HandState {
    /// Creates an idle state with a preallocated history buffer.
    pub fn new() -> HandState {
        HandState {
            fmt: GameFormat::nlhe(MAX_SEATS as u8),
            ev: Evaluator::get(),
            n_seats: 0,
            button: 0,
            sb_seat: 0,
            bb_seat: 0,
            sb: 0,
            bb: 0,
            ante: 0,
            start_stacks: [0; MAX_SEATS],
            stacks: [0; MAX_SEATS],
            street_contrib: [0; MAX_SEATS],
            total_contrib: [0; MAX_SEATS],
            pot_total: 0,
            status: [SeatStatus::Empty; MAX_SEATS],
            hole: [[Card(0); MAX_HOLE]; MAX_SEATS],
            board: [Card(0); 5],
            board_len: 0,
            street: Street::Preflop,
            current_bet: 0,
            last_raise: 0,
            bets_this_street: 0,
            acted: [false; MAX_SEATS],
            acted_at: [0; MAX_SEATS],
            needs_action: [false; MAX_SEATS],
            to_act: None,
            complete: true,
            history: Vec::with_capacity(256),
            pots: [Pot::default(); MAX_SEATS],
            n_pots: 0,
            showdown: false,
            strengths: [0; MAX_SEATS],
            uncalled: None,
            won: [0; MAX_SEATS],
        }
    }

    /// Starts a new hand: deals from `deck` (see the deal convention in [`crate::deck`]),
    /// posts antes and blinds, and positions the action on the first player to act.
    pub fn start(
        &mut self,
        fmt: &GameFormat,
        setup: &HandSetup,
        deck: &Deck,
    ) -> Result<(), SetupError> {
        let n = setup.n_seats;
        if n < 2 || n > fmt.max_seats as usize || n > MAX_SEATS {
            return Err(SetupError(format!("n_seats {n} invalid for format")));
        }
        if setup.bb == 0 || setup.sb > setup.bb {
            return Err(SetupError("need 0 <= sb <= bb and bb > 0".into()));
        }
        if setup.button >= n || setup.stacks[setup.button] == 0 {
            return Err(SetupError("button seat must hold a player".into()));
        }
        let occupied = setup.stacks[..n].iter().filter(|&&s| s > 0).count();
        if occupied < 2 {
            return Err(SetupError("need at least two players with chips".into()));
        }

        self.fmt = *fmt;
        self.n_seats = n;
        self.button = setup.button;
        self.sb = setup.sb;
        self.bb = setup.bb;
        self.ante = setup.ante;
        self.start_stacks = [0; MAX_SEATS];
        self.start_stacks[..n].copy_from_slice(&setup.stacks[..n]);
        self.stacks = self.start_stacks;
        self.street_contrib = [0; MAX_SEATS];
        self.total_contrib = [0; MAX_SEATS];
        self.pot_total = 0;
        self.acted = [false; MAX_SEATS];
        self.acted_at = [0; MAX_SEATS];
        self.needs_action = [false; MAX_SEATS];
        self.status = [SeatStatus::Empty; MAX_SEATS];
        for s in 0..n {
            if self.stacks[s] > 0 {
                self.status[s] = SeatStatus::Active;
            }
        }
        self.history.clear();
        self.street = Street::Preflop;
        self.board_len = 0;
        self.complete = false;
        self.to_act = None;
        self.n_pots = 0;
        self.showdown = false;
        self.uncalled = None;
        self.won = [0; MAX_SEATS];
        self.strengths = [0; MAX_SEATS];

        // Deal: round-robin from the first player left of the button, no burns.
        let hc = fmt.hole_cards as usize;
        let mut pos = 0;
        let mut seat = self.button;
        for _ in 0..n {
            seat = (seat + 1) % n;
            if self.status[seat] == SeatStatus::Empty {
                continue;
            }
            for k in 0..hc {
                self.hole[seat][k] = deck.at(k * occupied + pos);
            }
            pos += 1;
        }
        for i in 0..5 {
            self.board[i] = deck.at(occupied * hc + i);
        }

        // Antes, in deal order.
        if self.ante > 0 {
            let mut seat = self.button;
            for _ in 0..n {
                seat = (seat + 1) % n;
                if self.status[seat] != SeatStatus::Empty {
                    self.post(seat, self.ante, ActionKind::PostAnte, false);
                }
            }
        }

        // Blinds. Heads-up the button posts the small blind.
        if occupied == 2 {
            self.sb_seat = self.button;
            self.bb_seat = self.next_occupied(self.button);
        } else {
            self.sb_seat = self.next_occupied(self.button);
            self.bb_seat = self.next_occupied(self.sb_seat);
        }
        if self.sb > 0 {
            self.post(self.sb_seat, self.sb, ActionKind::PostSmallBlind, true);
        }
        self.post(self.bb_seat, self.bb, ActionKind::PostBigBlind, true);

        // The big blind is the opening bet: players must call the full blind even if the
        // big blind is all-in for less, and the minimum raise increment is one big blind.
        self.current_bet = self.bb;
        self.last_raise = self.fmt.betting.opening_increment(Street::Preflop, self.bb);
        self.bets_this_street = 1;
        self.begin_betting_round();
        let from = self.bb_seat;
        self.advance(from);
        self.debug_check_conservation();
        Ok(())
    }

    fn next_occupied(&self, from: usize) -> usize {
        let mut s = from;
        for _ in 0..self.n_seats {
            s = (s + 1) % self.n_seats;
            if self.status[s] != SeatStatus::Empty {
                return s;
            }
        }
        unreachable!("no occupied seat")
    }

    /// Posts a forced bet (capped at the stack).
    fn post(&mut self, seat: usize, amount: u64, kind: ActionKind, live: bool) {
        let a = amount.min(self.stacks[seat]);
        self.stacks[seat] -= a;
        self.total_contrib[seat] += a;
        self.pot_total += a;
        if live {
            self.street_contrib[seat] += a;
        }
        let all_in = self.stacks[seat] == 0;
        if all_in {
            self.status[seat] = SeatStatus::AllIn;
        }
        self.history.push(ActionRecord {
            seat: seat as u8,
            street: Street::Preflop,
            kind,
            amount: a,
            to: self.street_contrib[seat],
            all_in,
        });
    }

    fn count_active(&self) -> usize {
        self.status[..self.n_seats]
            .iter()
            .filter(|s| **s == SeatStatus::Active)
            .count()
    }

    fn count_live(&self) -> usize {
        self.status[..self.n_seats]
            .iter()
            .filter(|s| s.is_live())
            .count()
    }

    /// Marks who must act at the start of a betting round. With two or more players able to
    /// bet, all of them must act; a lone player with chips only acts if facing a bet.
    fn begin_betting_round(&mut self) {
        let n_active = self.count_active();
        let max_contrib = self.street_contrib[..self.n_seats]
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        for s in 0..self.n_seats {
            self.needs_action[s] = self.status[s] == SeatStatus::Active
                && (n_active >= 2 || self.street_contrib[s] < max_contrib);
        }
    }

    /// Moves to the next player needing action after `from`, closing streets (and running
    /// out the board) as needed, and settling the hand when it is over.
    fn advance(&mut self, mut from: usize) {
        loop {
            if self.count_live() == 1 {
                self.settle();
                return;
            }
            let mut s = from;
            for _ in 0..self.n_seats {
                s = (s + 1) % self.n_seats;
                if self.needs_action[s] && self.status[s] == SeatStatus::Active {
                    self.to_act = Some(s);
                    return;
                }
            }
            // Street complete.
            match self.street.next() {
                None => {
                    self.settle();
                    return;
                }
                Some(next) => {
                    self.street = next;
                    self.board_len = next.board_len();
                    self.street_contrib = [0; MAX_SEATS];
                    self.current_bet = 0;
                    self.bets_this_street = 0;
                    self.acted = [false; MAX_SEATS];
                    self.acted_at = [0; MAX_SEATS];
                    self.last_raise = self.fmt.betting.opening_increment(next, self.bb);
                    // If fewer than two players can bet, this marks nobody and the loop
                    // runs the board out street by street.
                    self.begin_betting_round();
                    from = self.button;
                }
            }
        }
    }

    /// The seat to act, or `None` once the hand is complete.
    #[inline]
    pub fn to_act(&self) -> Option<usize> {
        self.to_act
    }

    /// True once the hand has been settled.
    #[inline]
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Total chips in the pot (all streets, including current bets).
    #[inline]
    pub fn pot(&self) -> u64 {
        self.pot_total
    }

    /// Legal actions for `seat` (all-false/zero if it is not that seat's turn). The betting
    /// structure decides raise sizes and caps ([`crate::rules::betting`]); the rest is
    /// structure-independent ([`crate::rules::actions::legal_actions`]).
    pub fn legal_actions_for(&self, seat: usize) -> LegalActions {
        if self.to_act != Some(seat) {
            return LegalActions::default();
        }
        let (ctx, player) = self.bet_view(seat);
        legal_actions(&self.fmt.betting, &ctx, &player)
    }

    fn bet_view(&self, seat: usize) -> (BetContext, PlayerView) {
        let contrib = self.street_contrib[seat];
        let ctx = BetContext {
            street: self.street,
            bb: self.bb,
            current_bet: self.current_bet,
            last_raise: self.last_raise,
            bets_this_street: self.bets_this_street,
            pot: self.pot(),
            to_call: self.current_bet.saturating_sub(contrib),
        };
        let player = PlayerView {
            street_contrib: contrib,
            stack: self.stacks[seat],
            acted: self.acted[seat],
            acted_at: self.acted_at[seat],
            others_can_respond: (0..self.n_seats)
                .any(|s| s != seat && self.status[s] == SeatStatus::Active),
        };
        (ctx, player)
    }

    /// Legal actions for the player to act.
    #[inline]
    pub fn legal_actions(&self) -> LegalActions {
        match self.to_act {
            Some(s) => self.legal_actions_for(s),
            None => LegalActions::default(),
        }
    }

    /// Checks whether `action` is legal for the player to act, without applying it.
    pub fn validate(&self, action: Action) -> Result<(), IllegalAction> {
        self.resolve(action).map(|_| ())
    }

    /// Validates an action ([`crate::rules::actions::resolve`]).
    fn resolve(&self, action: Action) -> Result<Resolved, IllegalAction> {
        let Some(seat) = self.to_act else {
            return Err(IllegalAction::NotToAct);
        };
        let la = self.legal_actions_for(seat);
        let allin_to = self.street_contrib[seat] + self.stacks[seat];
        resolve(action, &la, allin_to, self.current_bet)
    }

    /// Applies an action for the player to act. Illegal actions are rejected without
    /// changing the state.
    pub fn apply(&mut self, action: Action) -> Result<(), IllegalAction> {
        let r = self.resolve(action)?;
        let seat = self.to_act.expect("resolved implies a player to act");
        let street = self.street;
        match r {
            Resolved::Fold => {
                self.status[seat] = SeatStatus::Folded;
                self.push(seat, street, ActionKind::Fold, 0, false);
            }
            Resolved::Check => {
                self.push(seat, street, ActionKind::Check, 0, false);
            }
            Resolved::Call => {
                let add = (self.current_bet - self.street_contrib[seat]).min(self.stacks[seat]);
                let all_in = self.move_chips(seat, add);
                self.push(seat, street, ActionKind::Call, add, all_in);
            }
            Resolved::RaiseTo(to) => {
                let kind = if self.current_bet == 0 {
                    ActionKind::Bet
                } else {
                    ActionKind::Raise
                };
                let add = to - self.street_contrib[seat];
                let increment = to - self.current_bet;
                // Only a full raise (increment >= last full increment) changes the minimum
                // raise and counts toward the fixed-limit cap. Short all-ins don't.
                if increment >= self.last_raise {
                    self.last_raise = increment;
                    self.bets_this_street += 1;
                }
                self.current_bet = to;
                let all_in = self.move_chips(seat, add);
                for s in 0..self.n_seats {
                    if s != seat && self.status[s] == SeatStatus::Active {
                        self.needs_action[s] = true;
                    }
                }
                self.push(seat, street, kind, add, all_in);
            }
        }
        self.acted[seat] = true;
        self.acted_at[seat] = self.current_bet;
        self.needs_action[seat] = false;
        self.to_act = None;
        self.advance(seat);
        self.debug_check_conservation();
        Ok(())
    }

    fn move_chips(&mut self, seat: usize, add: u64) -> bool {
        debug_assert!(add <= self.stacks[seat]);
        self.stacks[seat] -= add;
        self.street_contrib[seat] += add;
        self.total_contrib[seat] += add;
        self.pot_total += add;
        if self.stacks[seat] == 0 {
            self.status[seat] = SeatStatus::AllIn;
            true
        } else {
            false
        }
    }

    fn push(&mut self, seat: usize, street: Street, kind: ActionKind, amount: u64, all_in: bool) {
        self.history.push(ActionRecord {
            seat: seat as u8,
            street,
            kind,
            amount,
            to: self.street_contrib[seat],
            all_in,
        });
    }

    /// Returns the uncalled bet, builds side pots, evaluates the showdown and pays out.
    fn settle(&mut self) {
        let n = self.n_seats;
        self.to_act = None;
        self.complete = true;

        // Uncalled bet: the top contributor's excess over the second-highest contribution was
        // never matched and goes back before showdown.
        let mut top = 0;
        for s in 1..n {
            if self.total_contrib[s] > self.total_contrib[top] {
                top = s;
            }
        }
        let second = (0..n)
            .filter(|&s| s != top)
            .map(|s| self.total_contrib[s])
            .max()
            .unwrap_or(0);
        let excess = self.total_contrib[top] - second;
        if excess > 0 {
            self.total_contrib[top] -= excess;
            self.pot_total -= excess;
            self.stacks[top] += excess;
            self.uncalled = Some((top, excess));
        }

        let live = self.count_live();
        self.showdown = live >= 2;
        if self.showdown {
            // All live players reach showdown; make sure the full board is visible.
            self.board_len = 5;
            self.street = Street::River;
            let board = self.board;
            let hc = self.fmt.hole_cards as usize;
            let rules = self.fmt.rules();
            for s in 0..n {
                if self.status[s].is_live() {
                    let hole = &self.hole[s][..hc];
                    self.strengths[s] = rules.strength(self.ev, hole, &board);
                }
            }
        }

        // Side pots from the distinct contribution levels of live players.
        let mut levels = [0u64; MAX_SEATS];
        let mut n_levels = 0;
        for s in 0..n {
            let c = self.total_contrib[s];
            if self.status[s].is_live() && c > 0 && !levels[..n_levels].contains(&c) {
                levels[n_levels] = c;
                n_levels += 1;
            }
        }
        levels[..n_levels].sort_unstable();
        let mut prev = 0u64;
        self.n_pots = 0;
        for &lvl in &levels[..n_levels] {
            let mut amount = 0;
            let mut eligible = 0u16;
            for s in 0..n {
                let c = self.total_contrib[s];
                amount += c.min(lvl) - c.min(prev);
                if self.status[s].is_live() && c >= lvl {
                    eligible |= 1 << s;
                }
            }
            self.pots[self.n_pots] = Pot {
                amount,
                eligible,
                winners: 0,
            };
            self.n_pots += 1;
            prev = lvl;
        }
        // Folded money above the top live level (only possible in odd edge cases) joins the
        // last pot so that no chip is lost.
        let leftover: u64 = (0..n)
            .map(|s| self.total_contrib[s] - self.total_contrib[s].min(prev))
            .sum();
        if leftover > 0 {
            if self.n_pots == 0 {
                // No live contributor at all (e.g. a walk where the big blind posted 0):
                // give everything to the remaining live player.
                let w = (0..n).find(|&s| self.status[s].is_live()).unwrap();
                self.pots[0] = Pot {
                    amount: 0,
                    eligible: 1 << w,
                    winners: 0,
                };
                self.n_pots = 1;
            }
            self.pots[self.n_pots - 1].amount += leftover;
        }

        // Award each pot.
        for p in 0..self.n_pots {
            let pot = self.pots[p];
            let winners = if self.showdown {
                let mut best = 0;
                let mut mask = 0u16;
                for s in 0..n {
                    if pot.eligible & (1 << s) != 0 {
                        let st = self.strengths[s];
                        if st > best {
                            best = st;
                            mask = 1 << s;
                        } else if st == best {
                            mask |= 1 << s;
                        }
                    }
                }
                mask
            } else {
                pot.eligible
            };
            self.pots[p].winners = winners;
            let w = winners.count_ones() as u64;
            let share = pot.amount / w;
            let mut odd = pot.amount % w;
            // Odd chips go one at a time to winners starting left of the button.
            let mut s = self.button;
            for _ in 0..n {
                s = (s + 1) % n;
                if winners & (1 << s) != 0 {
                    let extra = if odd > 0 {
                        odd -= 1;
                        1
                    } else {
                        0
                    };
                    self.won[s] += share + extra;
                    self.stacks[s] += share + extra;
                }
            }
        }
        self.total_contrib = [0; MAX_SEATS];
        self.pot_total = 0;
        self.street_contrib = [0; MAX_SEATS];
        self.debug_check_conservation();
    }

    #[inline]
    fn debug_check_conservation(&self) {
        #[cfg(debug_assertions)]
        {
            let start: u64 = self.start_stacks[..self.n_seats].iter().sum();
            let now: u64 = self.stacks[..self.n_seats].iter().sum::<u64>() + self.pot();
            assert_eq!(start, now, "chip conservation violated");
            let summed: u64 = self.total_contrib[..self.n_seats].iter().sum();
            assert_eq!(self.pot_total, summed, "running pot total out of sync");
        }
    }

    /// Builds the observation for `seat`: its own hole cards plus public information only.
    pub fn observation(&self, seat: usize) -> Observation<'_> {
        let hc = self.fmt.hole_cards as usize;
        Observation {
            seat,
            n_seats: self.n_seats,
            button: self.button,
            format: self.fmt,
            street: self.street,
            hole: &self.hole[seat][..hc],
            board: &self.board[..self.board_len],
            stacks: &self.stacks[..self.n_seats],
            street_contrib: &self.street_contrib[..self.n_seats],
            total_contrib: &self.total_contrib[..self.n_seats],
            status: &self.status[..self.n_seats],
            pot: self.pot(),
            current_bet: self.current_bet,
            sb: self.sb,
            bb: self.bb,
            ante: self.ante,
            legal: self.legal_actions_for(seat),
            history: &self.history,
        }
    }

    /// A seat's hole cards. For the table runner to deliver to *that seat only* (and for
    /// hand histories); never pass these to another seat.
    pub fn hole_cards(&self, seat: usize) -> &[Card] {
        &self.hole[seat][..self.fmt.hole_cards as usize]
    }

    /// Hole cards revealed at showdown: `Some` only for live players once a showdown happened.
    pub fn shown_cards(&self, seat: usize) -> Option<&[Card]> {
        (self.complete && self.showdown && self.status[seat].is_live())
            .then(|| self.hole_cards(seat))
    }

    /// Visible board cards.
    #[inline]
    pub fn board(&self) -> &[Card] {
        &self.board[..self.board_len]
    }
    /// Current street (River once settled after a showdown).
    #[inline]
    pub fn street(&self) -> Street {
        self.street
    }
    /// Public action history.
    #[inline]
    pub fn history(&self) -> &[ActionRecord] {
        &self.history
    }
    /// Seats at the table.
    #[inline]
    pub fn n_seats(&self) -> usize {
        self.n_seats
    }
    /// Button seat.
    #[inline]
    pub fn button(&self) -> usize {
        self.button
    }
    /// Small-blind seat (the button when heads-up).
    #[inline]
    pub fn sb_seat(&self) -> usize {
        self.sb_seat
    }
    /// Big-blind seat.
    #[inline]
    pub fn bb_seat(&self) -> usize {
        self.bb_seat
    }
    /// Seat status.
    #[inline]
    pub fn status(&self, seat: usize) -> SeatStatus {
        self.status[seat]
    }
    /// Chips behind.
    #[inline]
    pub fn stack(&self, seat: usize) -> u64 {
        self.stacks[seat]
    }
    /// Stacks of all seats.
    #[inline]
    pub fn stacks(&self) -> &[u64] {
        &self.stacks[..self.n_seats]
    }
    /// Stack at the start of the hand.
    #[inline]
    pub fn start_stack(&self, seat: usize) -> u64 {
        self.start_stacks[seat]
    }
    /// Current street bet to match.
    #[inline]
    pub fn current_bet(&self) -> u64 {
        self.current_bet
    }
    /// Contribution of a seat on the current street.
    #[inline]
    pub fn street_contrib(&self, seat: usize) -> u64 {
        self.street_contrib[seat]
    }
    /// Game format of the current hand.
    #[inline]
    pub fn format(&self) -> &GameFormat {
        &self.fmt
    }

    /// Net chips won (+) or lost (−) by `seat` this hand. Valid once complete.
    #[inline]
    pub fn net(&self, seat: usize) -> i64 {
        self.stacks[seat] as i64 - self.start_stacks[seat] as i64
    }
    /// Chips each seat collected from pots. Valid once complete.
    #[inline]
    pub fn won(&self, seat: usize) -> u64 {
        self.won[seat]
    }
    /// True if the hand went to showdown. Valid once complete.
    #[inline]
    pub fn went_to_showdown(&self) -> bool {
        self.showdown
    }
    /// Showdown strength of a live seat (0 if no showdown).
    #[inline]
    pub fn strength(&self, seat: usize) -> Strength {
        self.strengths[seat]
    }
    /// Pots awarded. Valid once complete.
    #[inline]
    pub fn pots(&self) -> &[Pot] {
        &self.pots[..self.n_pots]
    }
    /// The uncalled bet returned before showdown, if any: `(seat, chips)`.
    #[inline]
    pub fn uncalled_return(&self) -> Option<(usize, u64)> {
        self.uncalled
    }
}
