use super::reference::{self, HandCategory};
use super::*;
use crate::card::parse_cards;
use rand_xoshiro::rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

fn s(cards: &str) -> Strength {
    eval(&parse_cards(cards).unwrap())
}
fn cat(cards: &str) -> HandCategory {
    category(s(cards))
}
fn omaha(hole: &str, board: &str) -> Strength {
    let h = parse_cards(hole).unwrap();
    let b: [Card; 5] = parse_cards(board).unwrap().try_into().unwrap();
    Evaluator::get().eval_omaha(&h, &b)
}

fn random_cards<R: Rng>(rng: &mut R, n: usize) -> Vec<Card> {
    let mut d = crate::deck::Deck::new();
    d.shuffle_prefix(rng, n);
    d.cards()[..n].to_vec()
}

/// Checks that `fast` induces exactly the same order (including ties) as `slow`.
fn assert_same_order(mut pairs: Vec<(u32, Strength)>) {
    pairs.sort_unstable();
    for w in pairs.windows(2) {
        let ((r0, f0), (r1, f1)) = (w[0], w[1]);
        if r0 == r1 {
            assert_eq!(f0, f1, "tie in reference but not in fast eval");
        } else {
            assert!(
                f0 < f1,
                "order mismatch: ref {r0:#x}<{r1:#x} but fast {f0}>={f1}"
            );
        }
        assert_eq!(reference::category_of(r1), category(f1));
    }
}

#[test]
fn categories_basic() {
    assert_eq!(cat("AsKsQsJsTs2c3d"), HandCategory::StraightFlush);
    assert_eq!(cat("AsAhAdAc2s"), HandCategory::FourOfAKind);
    assert_eq!(cat("KsKhKd2c2s"), HandCategory::FullHouse);
    assert_eq!(cat("2s5s9sJsKs"), HandCategory::Flush);
    assert_eq!(cat("9c8d7h6s5c"), HandCategory::Straight);
    assert_eq!(cat("7c7d7h2sKc"), HandCategory::ThreeOfAKind);
    assert_eq!(cat("7c7d2h2sKc"), HandCategory::TwoPair);
    assert_eq!(cat("7c7d3h2sKc"), HandCategory::OnePair);
    assert_eq!(cat("7c9d3h2sKc"), HandCategory::HighCard);
}

#[test]
fn wheel_is_lowest_straight() {
    let wheel = s("As2d3h4c5s");
    let six_high = s("2d3h4c5s6s");
    assert_eq!(category(wheel), HandCategory::Straight);
    assert!(wheel < six_high);
    // Ace-high straight beats king-high.
    assert!(s("AsKdQhJcTs") > s("KdQhJcTs9s"));
    // Steel wheel is the lowest straight flush, but beats quads.
    let steel = s("Ah2h3h4h5h");
    assert_eq!(category(steel), HandCategory::StraightFlush);
    assert!(steel < s("2h3h4h5h6h"));
    assert!(steel > s("AsAhAdAcKs"));
    // No wrap-around straights.
    assert_eq!(cat("QsKdAh2c3s"), HandCategory::HighCard);
    // Wheel recognized in 7 cards alongside higher unpaired cards.
    assert_eq!(cat("As2d3h4c5sJdQc"), HandCategory::Straight);
    assert_eq!(s("As2d3h4c5sJdQc"), wheel);
    // 6-high straight beats a wheel when both are present.
    assert_eq!(s("As2d3h4c5s6dQc"), six_high);
}

#[test]
fn board_plays() {
    // Board is a broadway straight; both players' hole cards are irrelevant.
    let a = s("2c3d AsKdQhJcTs");
    let b = s("4h4s AsKdQhJcTs");
    assert_eq!(a, b);
    // Board quads with ace kicker on board: nobody improves.
    assert_eq!(s("2c3d 9s9h9d9cAs"), s("KcQd 9s9h9d9cAs"));
    // But a higher kicker in hand plays when the board kicker is low.
    assert!(s("Kc2d 9s9h9d9c3s") > s("Qc2d 9s9h9d9c3s"));
}

#[test]
fn kickers() {
    // Same pair, kicker decides.
    assert!(s("AsAd Kc 7h 5s 3c 2d") > s("AsAd Qc 7h 5s 3c 2d"));
    // Fifth kicker matters; sixth and seventh do not.
    assert!(s("AsKdQc9h7s") > s("AsKdQc9h6s"));
    assert_eq!(s("AsKdQc9h7s 3c 2d"), s("AsKdQc9h7s 4c 2d"));
    // Two pair: kicker decides.
    assert!(s("KsKdQcQh9s") > s("KsKdQcQh8s"));
    // Trips kicker.
    assert!(s("7c7d7hAsKc") > s("7c7d7hAsQc"));
    // Full house: trips rank first.
    assert!(s("3c3d3h2s2c") > s("2c2d2hAsAc"));
    // Flush compared card by card.
    assert!(s("As9s7s5s3s") > s("As9s7s5s2s"));
}

#[test]
fn counterfeited_two_pair() {
    // Hole 4-4 with board K-K-Q-Q-8: player plays KKQQ8; the 4s are counterfeited.
    let counterfeited = s("4c4d KsKhQdQs8c");
    let board_only_kicker = s("2c3d KsKhQdQs8c");
    assert_eq!(counterfeited, board_only_kicker);
    // A hole ace still plays as kicker.
    assert!(s("Ac3d KsKhQdQs8c") > counterfeited);
    // Three pairs: best two plus best remaining card.
    assert_eq!(s("AcAd KsKhQdQs2c"), s("AcAd KsKhQdJs2c"));
}

#[test]
fn flush_picks_top_five() {
    // 7 spades: best five of them.
    assert_eq!(s("As Ks 9s 7s 5s 3s 2s"), s("AsKs9s7s5s"));
    // 6 spades with a straight flush inside.
    assert_eq!(category(s("9s8s7s6s5s2sAs")), HandCategory::StraightFlush);
    assert_eq!(s("9s8s7s6s5s2sAs"), s("9s8s7s6s5s"));
    // Flush beats straight on the same 7 cards.
    assert_eq!(cat("9s8s7s6d5s2sAs"), HandCategory::Flush);
    // Full house beats a flush (needs ≤ 7 cards; flush impossible here anyway).
    assert!(s("KsKhKd2c2s") > s("AsQs9s5s3s"));
}

#[test]
fn five_six_seven_consistent() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
    let ev = Evaluator::get();
    for _ in 0..20_000 {
        let c = random_cards(&mut rng, 7);
        let s5 = ev.eval(&c[..5]);
        let s6 = ev.eval(&c[..6]);
        let s7 = ev.eval(&c);
        assert!(s5 <= s6 && s6 <= s7);
        let arr: [Card; 7] = c.clone().try_into().unwrap();
        assert_eq!(ev.eval7(&arr), s7);
    }
}

fn cross_check(n_hands: usize, seed: u64) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let ev = Evaluator::get();
    let mut pairs = Vec::with_capacity(n_hands);
    for i in 0..n_hands {
        let n = 5 + i % 3;
        let c = random_cards(&mut rng, n);
        pairs.push((reference::eval_best(&c), ev.eval(&c)));
    }
    assert_same_order(pairs);
}

#[test]
fn fast_matches_reference_random() {
    cross_check(60_000, 1);
}

#[test]
#[ignore = "slow in debug; run with --release -- --ignored"]
fn fast_matches_reference_random_1m() {
    cross_check(1_000_000, 2);
}

#[test]
fn all_5_card_hands_give_7462_classes() {
    // Exhaustive over C(52,5) = 2,598,960 hands: exactly the 7462 distinct 5-card classes,
    // with the standard per-category class counts.
    let ev = Evaluator::get();
    let mut seen = std::collections::HashSet::new();
    let c = |i: usize| Card(i as u8);
    for a in 0..52 {
        for b in a + 1..52 {
            for d in b + 1..52 {
                for e in d + 1..52 {
                    for f in e + 1..52 {
                        seen.insert(ev.eval(&[c(a), c(b), c(d), c(e), c(f)]));
                    }
                }
            }
        }
    }
    assert_eq!(seen.len(), 7462);
    let mut per_cat = [0usize; 9];
    for s in &seen {
        per_cat[category(*s) as usize] += 1;
    }
    assert_eq!(per_cat, [1277, 2860, 858, 858, 10, 1277, 156, 156, 10]);
}

#[test]
fn omaha_must_use_two_hole_cards() {
    // Four spades on board + one spade in hand: no flush in Omaha.
    assert_ne!(
        category(omaha("As Kd 7c 2h", "Ks Qs Js 3s 9d")),
        HandCategory::Flush
    );
    // Two spades in hand makes the flush.
    assert_eq!(
        category(omaha("As 2s 7c 2h", "Ks Qs Js 3d 9d")),
        HandCategory::Flush
    );
    // Board straight does not play by itself: hole cards must contribute two.
    assert_eq!(
        category(omaha("2c 2d 3h 3s", "Ts Jd Qh Kc Ac")),
        HandCategory::OnePair
    );
    // Quads on board: best is board trips + 2 hole cards (full house with a hole pair).
    assert_eq!(
        category(omaha("Kc Kd 2h 3s", "9s 9h 9d 9c 4d")),
        HandCategory::FullHouse
    );
    // Ac with an A-A-K-K board: Hold'em would be aces full, but Omaha must use two hole
    // cards and three board cards, so the best is trip aces.
    assert_eq!(
        category(omaha("Ac 7d 8h 2s", "As Ah Kd Kc 3d")),
        HandCategory::ThreeOfAKind
    );
}

#[test]
fn omaha_matches_reference() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
    let ev = Evaluator::get();
    let mut pairs = Vec::new();
    for _ in 0..5_000 {
        let c = random_cards(&mut rng, 9);
        let board: [Card; 5] = c[4..].try_into().unwrap();
        pairs.push((
            reference::eval_omaha_best(&c[..4], &board),
            ev.eval_omaha(&c[..4], &board),
        ));
    }
    assert_same_order(pairs);
}

#[test]
#[ignore = "exhaustive: 133,784,560 hands; run with --release -- --ignored"]
fn enumerate_all_7_card_hands() {
    let ev = Evaluator::get();
    let mut counts = [0u64; 9];
    let c = |i: usize| Card(i as u8);
    for a in 0..52 {
        let ha = HandAcc::EMPTY.with(c(a));
        for b in a + 1..52 {
            let hb = ha.with(c(b));
            for d in b + 1..52 {
                let hd = hb.with(c(d));
                for e in d + 1..52 {
                    let he = hd.with(c(e));
                    for f in e + 1..52 {
                        let hf = he.with(c(f));
                        for g in f + 1..52 {
                            let hg = hf.with(c(g));
                            for h in g + 1..52 {
                                let s = ev.eval_acc(hg.with(c(h)));
                                counts[((s >> 12) - 1) as usize] += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    let expected = [
        23_294_460u64, // high card
        58_627_800,    // one pair
        31_433_400,    // two pair
        6_461_620,     // three of a kind
        6_180_020,     // straight
        4_047_644,     // flush
        3_473_184,     // full house
        224_848,       // four of a kind
        41_584,        // straight flush (incl. royal)
    ];
    assert_eq!(counts, expected);
    assert_eq!(counts.iter().sum::<u64>(), 133_784_560);
}
