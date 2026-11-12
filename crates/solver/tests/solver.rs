//! Terminal evaluation vs brute force, and solver invariants on real river spots.

use engine::card::parse_cards;
use engine::rng::below;
use engine::Card;
use rand_xoshiro::rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use solver::terminal::{
    compat_mass, compat_mass_naive, showdown_net, showdown_net_naive, TermScratch,
};
use solver::{solve_river, Abstraction, HandSet, Params, RiverHands, N_COMBOS};

fn rng(seed: u64) -> Xoshiro256PlusPlus {
    Xoshiro256PlusPlus::seed_from_u64(seed)
}

fn board(s: &str) -> [Card; 5] {
    parse_cards(s).unwrap().try_into().unwrap()
}

fn random_board(r: &mut Xoshiro256PlusPlus) -> [Card; 5] {
    let mut deck: Vec<u8> = (0..52).collect();
    for i in 0..5 {
        let j = i + below(r, (52 - i) as u32) as usize;
        deck.swap(i, j);
    }
    [0, 1, 2, 3, 4].map(|i| Card(deck[i]))
}

/// Random range with a random share of exact zeros.
fn random_range(r: &mut Xoshiro256PlusPlus, n: usize) -> Vec<f32> {
    let zero = below(r, 4) as u64;
    (0..n)
        .map(|_| {
            if below(r, 4) < zero as u32 {
                0.0
            } else {
                (r.next_u32() >> 8) as f32 / (1 << 24) as f32
            }
        })
        .collect()
}

fn assert_close(fast: &[f32], slow: &[f64], tol: f64, what: &str) {
    for (i, (&f, &s)) in fast.iter().zip(slow).enumerate() {
        assert!(
            (f as f64 - s).abs() <= tol * (1.0 + s.abs()),
            "{what}[{i}]: fast {f} vs naive {s}"
        );
    }
}

/// Both O(n) paths (sweeps and the solver's gather index) against the O(n²) reference.
fn check_terminals(hs: &HandSet, q: &[f32]) {
    let n = hs.len();
    let (mass, net) = (compat_mass_naive(hs, q), showdown_net_naive(hs, q));
    let mut out = vec![0f32; n];
    compat_mass(hs, q, &mut out);
    assert_close(&out, &mass, 1e-5, "fold mass (sweep)");
    showdown_net(hs, q, &mut out);
    assert_close(&out, &net, 1e-5, "showdown net (sweep)");
    let mut sc = TermScratch::default();
    hs.index.prepare(q, &mut sc);
    hs.index.compat(q, &sc, &mut out);
    assert_close(&out, &mass, 1e-5, "fold mass (index)");
    hs.index.net(&sc, &mut out);
    assert_close(&out, &net, 1e-5, "showdown net (index)");
}

#[test]
fn terminals_match_brute_force_on_random_and_tie_heavy_boards() {
    let mut r = rng(1);
    let mut boards: Vec<[Card; 5]> = (0..12).map(|_| random_board(&mut r)).collect();
    // Tie-heavy: royal flush, quads + kicker, a straight, full house on board.
    for b in [
        "AsKsQsJsTs",
        "7c7d7h7sKd",
        "5c6d7h8s9c",
        "QcQdQh4s4d",
        "2c3d4h5s6c",
    ] {
        boards.push(board(b));
    }
    for b in &boards {
        let rh = RiverHands::new(b);
        let q = random_range(&mut r, rh.set.len());
        check_terminals(&rh.set, &q);
        check_terminals(&rh.set, &vec![0.0; rh.set.len()]);
    }
}

#[test]
fn terminals_match_brute_force_on_abstract_sets() {
    let mut r = rng(2);
    for trial in 0..50 {
        let n = 2 + below(&mut r, 40) as usize;
        let strength: Vec<u32> = (0..n).map(|_| below(&mut r, 6)).collect(); // many ties
        let cards: Option<Vec<[u8; 2]>> = if trial % 2 == 0 {
            None
        } else {
            // Distinct random card pairs.
            let mut set = std::collections::BTreeSet::new();
            while set.len() < n {
                let a = below(&mut r, 52) as u8;
                let b = below(&mut r, 52) as u8;
                if a != b {
                    set.insert((a.max(b), a.min(b)));
                }
            }
            Some(set.into_iter().map(|(a, b)| [a, b]).collect())
        };
        let (hs, _) = HandSet::new(&strength, cards.as_deref());
        let q = random_range(&mut r, n);
        check_terminals(&hs, &q);
    }
}

fn spot(seed: u64) -> ([Card; 5], f32, Vec<f32>, Vec<f32>) {
    let mut r = rng(seed);
    let b = random_board(&mut r);
    let stack = [0.5f32, 1.0, 2.0, 4.0, 10.0][below(&mut r, 5) as usize];
    let r1 = random_range(&mut r, N_COMBOS);
    let r2 = random_range(&mut r, N_COMBOS);
    (b, stack, r1, r2)
}

#[test]
fn zero_sum_identity_and_determinism() {
    for seed in 0..4 {
        let (b, s, r1, r2) = spot(seed);
        let params = Params {
            max_iters: 200,
            ..Params::default()
        };
        let a = solve_river(&b, s, [&r1, &r2], &Abstraction::default(), &params).unwrap();
        let rs = [a.hands.gather(&r1), a.hands.gather(&r2)];
        let tot: Vec<f64> = rs
            .iter()
            .map(|v| v.iter().map(|&x| x as f64).sum())
            .collect();
        let lhs: f64 = (0..2)
            .map(|p| {
                a.solution.cfv[p]
                    .iter()
                    .zip(&rs[p])
                    .map(|(&v, &r)| v as f64 * r as f64 / tot[p])
                    .sum::<f64>()
            })
            .sum();
        assert!(lhs.abs() < 1e-5, "zero-sum residual {lhs}");
        let again = solve_river(&b, s, [&r1, &r2], &Abstraction::default(), &params).unwrap();
        assert_eq!(a.solution.value, again.solution.value);
        assert_eq!(a.solution.avg, again.solution.avg);
        assert_eq!(a.solution.expl, again.solution.expl);
    }
}

#[test]
fn exploitability_falls_below_target() {
    let mut fails = 0;
    for seed in 10..20 {
        let (b, s, r1, r2) = spot(seed);
        let params = Params {
            max_iters: 1000,
            target_expl: 0.0,
            ..Params::default()
        };
        let sol = solve_river(&b, s, [&r1, &r2], &Abstraction::default(), &params)
            .unwrap()
            .solution;
        let h: Vec<f32> = sol.history.iter().map(|x| x.1).collect();
        // Roughly monotone: never more than 20% above the running minimum after the start.
        let mut best = f32::INFINITY;
        for (i, &e) in h.iter().enumerate() {
            assert!(e >= -1e-5, "negative exploitability {e}");
            if i > 1 {
                assert!(e <= best * 1.2 + 1e-4, "expl jumped: {h:?}");
            }
            best = best.min(e);
        }
        let at500 = sol.history.iter().find(|x| x.0 == 500).unwrap().1;
        if at500 >= 0.002 {
            fails += 1;
        }
    }
    assert!(
        fails <= 1,
        "{fails}/10 spots above 0.2% of pot at 500 iterations"
    );
}
