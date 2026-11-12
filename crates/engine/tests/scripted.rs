//! Scripted hands: a fixed deck plus a fixed action list, with asserted final stacks.
//!
//! Every action is paired with the seat expected to act, so turn order is checked too.

use engine::action::{Action, IllegalAction};
use engine::{DeckBuilder, GameFormat, HandSetup, HandState, SeatStatus, MAX_SEATS};
use Action::*;

const NL6: GameFormat = GameFormat::nlhe(6);

struct Game {
    st: HandState,
}

fn setup(stacks: &[u64], button: usize, sb: u64, bb: u64, ante: u64) -> HandSetup {
    let mut s = [0u64; MAX_SEATS];
    s[..stacks.len()].copy_from_slice(stacks);
    HandSetup {
        n_seats: stacks.len(),
        button,
        stacks: s,
        sb,
        bb,
        ante,
    }
}

/// Seats in the hand, in deal order (left of the button first).
fn deal_order(su: &HandSetup) -> Vec<usize> {
    (1..=su.n_seats)
        .map(|k| (su.button + k) % su.n_seats)
        .filter(|&s| su.stacks[s] > 0)
        .collect()
}

impl Game {
    fn new(fmt: GameFormat, su: HandSetup, deck: DeckBuilder) -> Game {
        let d = deck.build(&deal_order(&su), fmt.hole_cards as usize);
        let mut st = HandState::new();
        st.start(&fmt, &su, &d).expect("setup");
        Game { st }
    }
    fn act(&mut self, seat: usize, a: Action) -> &mut Self {
        assert_eq!(
            self.st.to_act(),
            Some(seat),
            "expected seat {seat} to act before {a:?}"
        );
        self.st
            .apply(a)
            .unwrap_or_else(|e| panic!("seat {seat} {a:?} rejected: {e:?}"));
        self
    }
    fn script(&mut self, actions: &[(usize, Action)]) -> &mut Self {
        for &(s, a) in actions {
            self.act(s, a);
        }
        self
    }
    fn illegal(&mut self, a: Action, why: IllegalAction) -> &mut Self {
        let before = self.st.history().len();
        let to_act = self.st.to_act();
        assert_eq!(self.st.apply(a), Err(why), "{a:?}");
        assert_eq!(self.st.history().len(), before);
        assert_eq!(self.st.to_act(), to_act);
        self
    }
    fn check_down(&mut self) -> &mut Self {
        while let Some(s) = self.st.to_act() {
            self.act(s, Check);
        }
        self
    }
    fn stacks(&self, expected: &[u64]) {
        assert!(self.st.is_complete(), "hand not complete");
        assert_eq!(self.st.stacks(), expected);
    }
}

// ---------------------------------------------------------------- heads-up and positions

#[test]
fn hu_button_posts_sb_acts_first_preflop_last_postflop() {
    let mut g = Game::new(
        NL6,
        setup(&[200, 200], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "AsAd")
            .hole(1, "KsKd")
            .board("2c7d9hJcQh"),
    );
    assert_eq!(g.st.sb_seat(), 0);
    assert_eq!(g.st.bb_seat(), 1);
    g.script(&[(0, Call), (1, Check)]); // preflop: button first
    g.script(&[(1, Check), (0, BetTo(4)), (1, Call)]); // flop: BB first
    g.script(&[(1, Check), (0, Check), (1, Check), (0, Check)]); // turn, river
    assert!(g.st.went_to_showdown());
    g.stacks(&[206, 194]);
}

#[test]
fn hu_sb_folds_bb_gets_blind() {
    let mut g = Game::new(NL6, setup(&[200, 200], 1, 1, 2, 0), DeckBuilder::new());
    assert_eq!(g.st.sb_seat(), 1);
    g.act(1, Fold);
    assert!(!g.st.went_to_showdown());
    assert_eq!(g.st.uncalled_return(), Some((0, 1)));
    g.stacks(&[201, 199]);
    assert_eq!(g.st.shown_cards(0), None);
}

#[test]
fn hu_min_reraise_sizes() {
    let mut g = Game::new(
        NL6,
        setup(&[200, 200], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "QsQd")
            .hole(1, "JsJd")
            .board("2c7d9h3c4h"),
    );
    assert_eq!(g.st.legal_actions().min_raise_to, 4);
    g.act(0, RaiseTo(6)); // increment 4
    assert_eq!(g.st.legal_actions().min_raise_to, 10);
    g.act(1, RaiseTo(16)); // increment 10
    let la = g.st.legal_actions();
    assert_eq!((la.min_raise_to, la.max_raise_to), (26, 200));
    g.illegal(RaiseTo(25), IllegalAction::BelowMinRaise);
    g.illegal(RaiseTo(201), IllegalAction::AboveMaxRaise);
    g.act(0, RaiseTo(26)).act(1, Call).check_down();
    g.stacks(&[226, 174]);
}

#[test]
fn three_handed_positions() {
    let mut g = Game::new(
        NL6,
        setup(&[200, 200, 200], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "2c3d")
            .hole(1, "AhAd")
            .hole(2, "7h8h")
            .board("KcQs4d5s9c"),
    );
    assert_eq!((g.st.sb_seat(), g.st.bb_seat()), (1, 2));
    g.script(&[(0, Call), (1, Call), (2, Check)]);
    g.script(&[(1, Check), (2, Check), (0, Check)]); // flop: SB first, button last
    g.check_down();
    g.stacks(&[198, 204, 198]);
}

#[test]
fn empty_seat_is_skipped_for_blinds_and_action() {
    // Seat 3 is empty; button 2 => SB seat 0, BB seat 1, first to act seat 2.
    let mut g = Game::new(
        NL6,
        setup(&[200, 200, 200, 0], 2, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "AsAd")
            .hole(1, "2c3d")
            .hole(2, "7c8d")
            .board("KcQs4d5s9h"),
    );
    assert_eq!((g.st.sb_seat(), g.st.bb_seat()), (0, 1));
    assert_eq!(g.st.status(3), SeatStatus::Empty);
    g.script(&[(2, Call), (0, Call), (1, Check)]);
    g.script(&[(0, Check), (1, Check), (2, Check)]);
    g.check_down();
    g.stacks(&[204, 198, 198, 0]);
}

#[test]
fn walk_to_big_blind_six_max() {
    let mut g = Game::new(NL6, setup(&[200; 6], 0, 1, 2, 0), DeckBuilder::new());
    g.script(&[(3, Fold), (4, Fold), (5, Fold), (0, Fold), (1, Fold)]);
    g.stacks(&[200, 199, 201, 200, 200, 200]);
}

#[test]
fn six_way_limp_to_showdown() {
    let mut g = Game::new(
        NL6,
        setup(&[200; 6], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "2c3d")
            .hole(1, "4c5d")
            .hole(2, "6c7d")
            .hole(3, "AcAd")
            .hole(4, "8c9h")
            .hole(5, "2h3h")
            .board("KsQsJd4s8d"),
    );
    g.script(&[
        (3, Call),
        (4, Call),
        (5, Call),
        (0, Call),
        (1, Call),
        (2, Check),
    ]);
    for _ in 0..3 {
        g.script(&[
            (1, Check),
            (2, Check),
            (3, Check),
            (4, Check),
            (5, Check),
            (0, Check),
        ]);
    }
    g.stacks(&[198, 198, 198, 210, 198, 198]);
    for s in 0..6 {
        assert!(g.st.shown_cards(s).is_some());
    }
}

// ---------------------------------------------------------------- raises and reopening

#[test]
fn min_raise_preflop_three_handed() {
    let mut g = Game::new(NL6, setup(&[200, 200, 200], 0, 1, 2, 0), DeckBuilder::new());
    g.illegal(RaiseTo(3), IllegalAction::BelowMinRaise);
    g.act(0, RaiseTo(4));
    g.illegal(RaiseTo(5), IllegalAction::BelowMinRaise);
    g.act(1, RaiseTo(10));
    assert_eq!(g.st.legal_actions().min_raise_to, 16);
    g.illegal(RaiseTo(15), IllegalAction::BelowMinRaise);
    g.script(&[(2, Fold), (0, Fold)]);
    assert_eq!(g.st.uncalled_return(), Some((1, 6)));
    g.stacks(&[196, 206, 198]);
}

#[test]
fn bb_option_to_raise_after_limps() {
    let mut g = Game::new(
        NL6,
        setup(&[200, 200, 200], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "AsAd")
            .hole(1, "2c3d")
            .hole(2, "7c8d")
            .board("KcQs4d5s9h"),
    );
    g.script(&[(0, Call), (1, Call)]);
    let la = g.st.legal_actions();
    assert!(la.can_check && la.can_raise);
    assert_eq!(la.min_raise_to, 4);
    g.script(&[(2, RaiseTo(6)), (0, Call), (1, Fold)]);
    g.script(&[(2, Check), (0, Check)]); // flop: seat 2 first (seat 1 folded)
    g.check_down();
    g.stacks(&[208, 198, 194]);
}

#[test]
fn short_all_in_does_not_reopen_betting() {
    // BB has 7 chips. Seat 0 raises to 6, seat 1 calls, BB shoves to 7 (increment 1 < 4).
    let mut g = Game::new(
        NL6,
        setup(&[200, 200, 7], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "2c3d")
            .hole(1, "4c5d")
            .hole(2, "AhAd")
            .board("KsQsJd9c8h"),
    );
    g.script(&[(0, RaiseTo(6)), (1, Call), (2, AllIn)]);
    let la = g.st.legal_actions();
    assert!(!la.can_raise, "short all-in must not reopen for seat 0");
    assert_eq!(la.call_amount, 1);
    g.illegal(RaiseTo(20), IllegalAction::RaiseNotAllowed);
    g.illegal(AllIn, IllegalAction::RaiseNotAllowed);
    g.act(0, Call);
    assert!(!g.st.legal_actions().can_raise);
    g.act(1, Call);
    g.check_down();
    g.stacks(&[193, 193, 21]);
}

#[test]
fn full_raise_all_in_reopens_betting() {
    // BB has 12: shove to 12 is an increment of 6 >= 4, a full raise.
    let mut g = Game::new(
        NL6,
        setup(&[200, 200, 12], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "AhAd")
            .hole(1, "4c5d")
            .hole(2, "2c3d")
            .board("KsQsJd9c8h"),
    );
    g.script(&[(0, RaiseTo(6)), (1, Call), (2, AllIn)]);
    let la = g.st.legal_actions();
    assert!(la.can_raise);
    assert_eq!(la.min_raise_to, 18);
    g.script(&[(0, RaiseTo(18)), (1, Fold)]);
    assert!(g.st.is_complete(), "runout without further action");
    assert_eq!(g.st.board().len(), 5);
    assert_eq!(g.st.uncalled_return(), Some((0, 6)));
    g.stacks(&[218, 194, 0]);
}

#[test]
fn cumulative_short_all_ins_reopen_for_original_bettor() {
    // Flop: s1 bets 10, s2 shoves 14 (short), s3 shoves 20 (short). s1 now faces 10 more
    // since it acted, a full raise in total, so it may re-raise (TDA cumulative rule).
    let mut g = Game::new(
        NL6,
        setup(&[200, 200, 16, 22], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "2c3d")
            .hole(1, "4c5d")
            .hole(2, "AhAd")
            .hole(3, "KhKd")
            .board("Ks7s2d9c8h"),
    );
    g.script(&[(3, Call), (0, Call), (1, Call), (2, Check)]);
    g.script(&[(1, BetTo(10)), (2, AllIn), (3, AllIn)]);
    assert_eq!(g.st.legal_actions().min_raise_to, 30);
    g.act(0, Call);
    assert!(g.st.legal_actions().can_raise);
    g.act(1, Call);
    g.check_down();
    // Main pot 64 (16 x 4) and side pot 18 (6 x 3): s3's trip kings beat s2's aces and win both.
    let pots = g.st.pots();
    assert_eq!(pots.len(), 2);
    assert_eq!((pots[0].amount, pots[1].amount), (64, 18));
    g.stacks(&[178, 178, 0, 82]);
}

#[test]
fn bet_below_min_only_as_all_in() {
    // Postflop seat 1 has 1 chip behind: a bet of 1 (< BB) is legal only as an all-in.
    let mut g = Game::new(
        NL6,
        setup(&[200, 3, 200], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "2c3d")
            .hole(1, "AhAd")
            .hole(2, "7c8d")
            .board("KsQs4d9c5h"),
    );
    g.script(&[(0, Call), (1, Call), (2, Check)]);
    let la = g.st.legal_actions();
    assert_eq!((la.min_raise_to, la.max_raise_to), (1, 1));
    g.act(1, BetTo(1));
    assert_eq!(g.st.legal_actions().min_raise_to, 3);
    g.script(&[(2, Call), (0, Call)]);
    g.check_down();
    // Main pot: 3*3 = 9 to seat 1.
    g.stacks(&[197, 9, 197]);
}

// ---------------------------------------------------------------- side pots, all-ins, splits

#[test]
fn three_side_pots_all_in_preflop() {
    let mut g = Game::new(
        NL6,
        setup(&[200, 150, 100, 50], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(3, "AhAd") // best
            .hole(2, "KhKd") // second
            .hole(1, "QhQd") // third
            .hole(0, "2c3d") // worst
            .board("9s8s4c7d2h"),
    );
    g.script(&[(3, AllIn), (0, AllIn), (1, AllIn), (2, AllIn)]);
    assert!(g.st.is_complete());
    assert_eq!(g.st.uncalled_return(), Some((0, 50)));
    let amounts: Vec<u64> = g.st.pots().iter().map(|p| p.amount).collect();
    assert_eq!(amounts, vec![200, 150, 100]);
    g.stacks(&[50, 100, 150, 200]);
}

#[test]
fn heads_up_all_in_preflop_runs_out_board() {
    let mut g = Game::new(
        NL6,
        setup(&[100, 300], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "7c2d")
            .hole(1, "AsKs")
            .board("7h7d2s3c4c"),
    );
    g.script(&[(0, AllIn), (1, Call)]);
    assert!(g.st.is_complete());
    assert_eq!(g.st.board().len(), 5);
    g.stacks(&[200, 200]);
}

#[test]
fn everyone_all_in_preflop_with_split_and_side_pot() {
    // Seats 1 and 2 tie (board straight) for the main pot; the side pot between 0 and 1
    // also ties. Seat 0's extra is returned uncalled.
    let mut g = Game::new(
        NL6,
        setup(&[300, 200, 100], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "2c2d")
            .hole(1, "3c3d")
            .hole(2, "4c4d")
            .board("AsKdQhJcTs"),
    );
    g.script(&[(0, AllIn), (1, AllIn), (2, AllIn)]);
    // Main pot 300 split 3 ways, side pot 200 split 2 ways.
    g.stacks(&[300, 200, 100]);
}

#[test]
fn odd_chip_goes_left_of_button() {
    // Pot 5 split between seats 0 and 2; seat 1 (first left of button) folded, so seat 2
    // gets the odd chip.
    let mut g = Game::new(
        NL6,
        setup(&[200, 200, 200], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "2c3d")
            .hole(1, "6h7h")
            .hole(2, "4c5d")
            .board("AsKdQhJcTs"),
    );
    g.script(&[(0, Call), (1, Fold), (2, Check)]);
    g.check_down();
    g.stacks(&[200, 199, 201]);
}

#[test]
fn odd_chips_three_way_split() {
    // Pot 8 split three ways (seats 0, 2, 3): 2 each and two odd chips to the first two
    // winners left of the button (seats 2 and 3).
    let mut g = Game::new(
        NL6,
        setup(&[200, 200, 200, 200], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "2c3d")
            .hole(1, "6h7h")
            .hole(2, "4c5d")
            .hole(3, "2h3c")
            .board("AsKdQhJcTs"),
    );
    g.script(&[(3, Call), (0, Call), (1, Call), (2, Check)]);
    g.act(1, Fold); // folding when a check is available is legal
    g.check_down();
    g.stacks(&[200, 198, 201, 201]);
}

#[test]
fn uncalled_bet_returned_to_shover() {
    let mut g = Game::new(NL6, setup(&[200, 200], 0, 1, 2, 0), DeckBuilder::new());
    g.script(&[(0, RaiseTo(6)), (1, AllIn), (0, Fold)]);
    assert_eq!(g.st.uncalled_return(), Some((1, 194)));
    g.stacks(&[194, 206]);
}

#[test]
fn uncalled_river_bet_returned() {
    let mut g = Game::new(
        NL6,
        setup(&[200, 200], 0, 1, 2, 0),
        DeckBuilder::new().hole(0, "2c3d").hole(1, "AhAd"),
    );
    g.script(&[(0, Call), (1, Check)]);
    g.script(&[(1, Check), (0, Check), (1, Check), (0, Check)]);
    g.script(&[(1, BetTo(50)), (0, Fold)]);
    assert_eq!(g.st.uncalled_return(), Some((1, 50)));
    g.stacks(&[198, 202]);
}

#[test]
fn folded_money_stays_in_side_pot() {
    // s0 all-in 10 preflop; s1 and s2 keep betting; s2 folds on the turn. s0 wins the main
    // pot, s1 the side pot including s2's folded flop call.
    let mut g = Game::new(
        NL6,
        setup(&[10, 200, 200], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "AhAd")
            .hole(1, "KhKd")
            .hole(2, "QhQd")
            .board("2c7s9d3c4h"),
    );
    g.script(&[(0, AllIn), (1, Call), (2, Call)]);
    g.script(&[(1, BetTo(20)), (2, Call)]);
    g.script(&[(1, BetTo(50)), (2, Fold)]);
    assert!(g.st.is_complete());
    assert_eq!(g.st.uncalled_return(), Some((1, 50)));
    g.stacks(&[30, 210, 170]);
}

#[test]
fn antes_and_walk() {
    let mut g = Game::new(NL6, setup(&[200, 200, 200], 0, 1, 2, 1), DeckBuilder::new());
    assert_eq!(g.st.pot(), 6);
    g.script(&[(0, Fold), (1, Fold)]);
    g.stacks(&[199, 198, 203]);
}

#[test]
fn ante_all_in_only_eligible_for_ante_pot() {
    let mut g = Game::new(
        NL6,
        setup(&[1, 200, 200], 0, 1, 2, 1),
        DeckBuilder::new()
            .hole(0, "AhAd")
            .hole(1, "KhKd")
            .hole(2, "2c3d")
            .board("Qs7s9d3c4h"),
    );
    assert_eq!(g.st.status(0), SeatStatus::AllIn);
    g.script(&[(1, Call), (2, Check)]);
    g.check_down();
    g.stacks(&[3, 201, 197]);
}

#[test]
fn partial_big_blind_all_in() {
    // BB has 1 chip (< BB of 2): others must still call the full 2.
    let mut g = Game::new(
        NL6,
        setup(&[200, 200, 1], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "KhKd")
            .hole(1, "2c3d")
            .hole(2, "AhAd")
            .board("Qs7s9d3c4h"),
    );
    assert_eq!(g.st.current_bet(), 2);
    assert_eq!(g.st.legal_actions().call_amount, 2);
    g.script(&[(0, Call), (1, Call)]);
    g.check_down();
    // Main 3 -> seat 2; side 2 -> seat 0.
    g.stacks(&[200, 198, 3]);
}

#[test]
fn partial_small_blind_all_in_heads_up_no_action() {
    // Button/SB has 1 chip and posts it all-in; BB has nothing to do. BB's extra is returned.
    let g = Game::new(
        NL6,
        setup(&[1, 200], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "AhAd")
            .hole(1, "2c3d")
            .board("Qs7s9d3c4h"),
    );
    assert!(g.st.is_complete());
    assert_eq!(g.st.uncalled_return(), Some((1, 1)));
    g.stacks(&[2, 199]);
}

#[test]
fn illegal_actions_rejected_without_state_change() {
    let mut g = Game::new(NL6, setup(&[200, 200, 200], 0, 1, 2, 0), DeckBuilder::new());
    g.illegal(Check, IllegalAction::CannotCheck);
    g.illegal(RaiseTo(201), IllegalAction::AboveMaxRaise);
    g.illegal(BetTo(1), IllegalAction::BelowMinRaise);
    g.act(0, Call);
    // Call with nothing to call (BB) is treated as a check.
    g.act(1, Call).act(2, Call);
    assert_eq!(g.st.street(), engine::Street::Flop);
}

// ---------------------------------------------------------------- PLO

#[test]
fn plo_pot_size_raises() {
    let plo = GameFormat::plo(6);
    let mut g = Game::new(
        plo,
        setup(&[200; 6], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(3, "AsKdQdJc")
            .hole(4, "2s3s8h8c")
            .board("KsQsJs9s4d"),
    );
    // UTG: pot 3, call 2 => max raise-to 2 + 3 + 2 = 7.
    let la = g.st.legal_actions();
    assert_eq!((la.min_raise_to, la.max_raise_to), (4, 7));
    g.illegal(RaiseTo(8), IllegalAction::AboveMaxRaise);
    g.act(3, RaiseTo(7));
    // Facing 7: pot 10, call 7 => 7 + 10 + 7 = 24.
    assert_eq!(g.st.legal_actions().max_raise_to, 24);
    g.act(4, RaiseTo(24));
    g.script(&[(5, Fold), (0, Fold), (1, Fold), (2, Fold), (3, Call)]);
    // Flop: pot 51. A pot-size bet is 51.
    assert_eq!(g.st.legal_actions().max_raise_to, 51);
    g.act(3, BetTo(51));
    // Facing 51 into 102: pot-limit max is 204, but seat 4 only has 176.
    let la = g.st.legal_actions();
    assert_eq!(la.max_raise_to, 176);
    g.act(4, AllIn).act(3, Call);
    // Seat 3 has one spade (As): no flush in Omaha. Seat 4 has 2s3s: flush.
    assert_eq!(
        engine::eval::category(g.st.strength(4)),
        engine::HandCategory::Flush
    );
    assert_ne!(
        engine::eval::category(g.st.strength(3)),
        engine::HandCategory::Flush
    );
    g.stacks(&[200, 199, 198, 0, 403, 200]);
}

#[test]
fn plo_all_in_not_allowed_above_pot_limit() {
    let plo = GameFormat::plo(6);
    let mut g = Game::new(plo, setup(&[200, 200], 0, 1, 2, 0), DeckBuilder::new());
    // HU button: pot 3, call 1 => max raise-to 2 + 3 + 1 = 6.
    assert_eq!(g.st.legal_actions().max_raise_to, 6);
    g.illegal(AllIn, IllegalAction::AboveMaxRaise);
    g.act(0, RaiseTo(6)).act(1, Fold);
    g.stacks(&[202, 198]);
}

// ---------------------------------------------------------------- FLHE

#[test]
fn flhe_cap_and_bet_sizes() {
    let fl = GameFormat::flhe(6, 2, 4);
    let mut g = Game::new(
        fl,
        setup(&[200, 200], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "AhAd")
            .hole(1, "KhKd")
            .board("2c7s9d3c4h"),
    );
    let la = g.st.legal_actions();
    assert_eq!((la.min_raise_to, la.max_raise_to), (4, 4));
    g.illegal(RaiseTo(6), IllegalAction::AboveMaxRaise);
    g.script(&[(0, RaiseTo(4)), (1, RaiseTo(6)), (0, RaiseTo(8))]);
    // Bet + 3 raises = cap of 4.
    assert!(!g.st.legal_actions().can_raise);
    g.illegal(RaiseTo(10), IllegalAction::RaiseNotAllowed);
    g.act(1, Call);
    // Flop: small bet.
    let la = g.st.legal_actions();
    assert_eq!((la.min_raise_to, la.max_raise_to), (2, 2));
    g.script(&[(1, BetTo(2)), (0, Call)]);
    // Turn: big bet.
    let la = g.st.legal_actions();
    assert_eq!((la.min_raise_to, la.max_raise_to), (4, 4));
    g.script(&[(1, BetTo(4)), (0, RaiseTo(8)), (1, Call)]);
    g.check_down();
    g.stacks(&[218, 182]);
}

#[test]
fn flhe_short_stack_all_in_raise() {
    // Seat 1 has 3 chips: its raise to 3 (< full raise to 4) is legal only as an all-in.
    let fl = GameFormat::flhe(6, 2, 4);
    let mut g = Game::new(
        fl,
        setup(&[200, 200, 3], 0, 1, 2, 0),
        DeckBuilder::new()
            .hole(0, "KhKd")
            .hole(1, "2c3d")
            .hole(2, "AhAd")
            .board("Qs7s9d5c4h"),
    );
    g.script(&[(0, Call), (1, Call)]);
    let la = g.st.legal_actions();
    assert_eq!((la.min_raise_to, la.max_raise_to), (3, 3));
    g.act(2, AllIn);
    // Short all-in: seats 0 and 1 already acted and may only call.
    assert!(!g.st.legal_actions().can_raise);
    g.script(&[(0, Call), (1, Call)]);
    g.check_down();
    g.stacks(&[197, 197, 9]);
}
