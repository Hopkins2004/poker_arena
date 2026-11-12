//! Property tests: random stacks, seats, formats and random (legal and illegal) actions.
//!
//! Checked after every step: chip conservation, the hand terminates within a bound, illegal
//! actions are rejected without changing the state, legal actions are accepted, and the
//! engine's acceptance matches an independent reading of `LegalActions`.

use engine::action::Action;
use engine::{Deck, GameFormat, HandSetup, HandState, LegalActions, SeatStatus, MAX_SEATS};
use proptest::prelude::*;
use rand_xoshiro::rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

fn format_for(kind: u8, seats: u8) -> GameFormat {
    match kind % 3 {
        0 => GameFormat::nlhe(seats),
        1 => GameFormat::plo(seats),
        _ => GameFormat::flhe(seats, 2, 4),
    }
}

/// Independent legality check derived only from `LegalActions` and public state.
fn expected_legal(a: Action, la: &LegalActions, allin_to: u64, current_bet: u64) -> bool {
    match a {
        Action::Fold | Action::Call => true,
        Action::Check => la.can_check,
        Action::BetTo(x) | Action::RaiseTo(x) => {
            la.can_raise && la.min_raise_to <= x && x <= la.max_raise_to
        }
        Action::AllIn => allin_to <= current_bet || (la.can_raise && allin_to <= la.max_raise_to),
    }
}

fn random_action<R: Rng>(rng: &mut R, la: &LegalActions, folds: bool) -> Action {
    let pick = rng.next_u32() % 12;
    let span = la.max_raise_to.saturating_sub(la.min_raise_to) + 1;
    match pick {
        0 if folds => Action::Fold,
        0 => Action::Call,
        1 | 2 => Action::Check,
        3..=5 => Action::Call,
        6 => Action::AllIn,
        7 | 8 => Action::RaiseTo(la.min_raise_to + rng.next_u64() % span),
        9 => Action::BetTo(la.min_raise_to.saturating_sub(1 + rng.next_u64() % 3)),
        10 => Action::RaiseTo(la.max_raise_to + 1 + rng.next_u64() % 5),
        _ => Action::RaiseTo(rng.next_u64() % 1000),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(3000))]

    #[test]
    fn random_hands_respect_invariants(
        seed in any::<u64>(),
        folds in any::<bool>(),
        kind in 0u8..3,
        n_seats in 2usize..=10,
        stacks in prop::collection::vec(0u64..=1000, 10),
        button_pick in 0usize..10,
        sb in 0u64..=3,
        bb_extra in 0u64..=3,
        ante in 0u64..=2,
    ) {
        let fmt = format_for(kind, MAX_SEATS as u8);
        let n = n_seats;
        let mut st_arr = [0u64; MAX_SEATS];
        // Make sure at least two seats have chips; sprinkle tiny and empty stacks.
        st_arr[..n].copy_from_slice(&stacks[..n]);
        if st_arr[..n].iter().filter(|&&s| s > 0).count() < 2 {
            st_arr[0] = st_arr[0].max(1);
            st_arr[1] = st_arr[1].max(1);
        }
        let occupied: Vec<usize> = (0..n).filter(|&s| st_arr[s] > 0).collect();
        let button = occupied[button_pick % occupied.len()];
        let bb = sb.max(1) + bb_extra;
        let setup = HandSetup { n_seats: n, button, stacks: st_arr, sb: sb.min(bb), bb, ante };

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut deck = Deck::new();
        deck.shuffle(&mut rng);
        let mut st = HandState::new();
        st.start(&fmt, &setup, &deck).unwrap();
        let total: u64 = st_arr.iter().sum();

        let mut steps = 0;
        while let Some(seat) = st.to_act() {
            steps += 1;
            prop_assert!(steps < 2000, "hand did not terminate");
            prop_assert_eq!(st.status(seat), SeatStatus::Active);
            let la = st.legal_actions();
            prop_assert!(la.call_amount <= st.stack(seat));
            if la.can_raise {
                prop_assert!(la.min_raise_to <= la.max_raise_to);
                prop_assert!(la.min_raise_to > st.current_bet());
                prop_assert!(la.max_raise_to <= st.street_contrib(seat) + st.stack(seat));
            }
            let a = random_action(&mut rng, &la, folds);
            let allin_to = st.street_contrib(seat) + st.stack(seat);
            let expect = expected_legal(a, &la, allin_to, st.current_bet());
            let hist = st.history().len();
            let stacks_before = st.stacks().to_vec();
            let r = st.apply(a);
            prop_assert_eq!(r.is_ok(), expect, "{:?} with {:?}: {:?}", a, la, r);
            if r.is_err() {
                prop_assert_eq!(st.history().len(), hist);
                prop_assert_eq!(st.to_act(), Some(seat));
                prop_assert_eq!(st.stacks(), &stacks_before[..]);
            }
            let in_play: u64 = st.stacks().iter().sum::<u64>() + st.pot();
            prop_assert_eq!(in_play, total, "chip conservation");
        }
        prop_assert!(st.is_complete());
        prop_assert_eq!(st.pot(), 0);
        prop_assert_eq!(st.stacks().iter().sum::<u64>(), total);
        let net: i64 = (0..n).map(|s| st.net(s)).sum();
        prop_assert_eq!(net, 0);
        let awarded: u64 = st.pots().iter().map(|p| p.amount).sum();
        let won: u64 = (0..n).map(|s| st.won(s)).sum();
        prop_assert_eq!(awarded, won);
        // Only live players can win, and folded seats never reveal cards.
        for s in 0..n {
            if st.won(s) > 0 {
                prop_assert!(st.status(s).is_live());
            }
            if st.status(s) == SeatStatus::Folded {
                prop_assert!(st.shown_cards(s).is_none());
            }
        }
        if st.went_to_showdown() {
            prop_assert_eq!(st.board().len(), 5);
        }
    }
}
