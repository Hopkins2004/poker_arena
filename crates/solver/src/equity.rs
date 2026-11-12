//! Range-vs-range equity on the flop or turn (used by the training-range generator).
//!
//! For every hand, the share of the compatible opponent range it beats (ties count half) on
//! random completions of the board. The turn enumerates every river; the flop uses `samples`
//! seeded turn+river runouts (the same runouts for all hands). Each completed board is
//! evaluated once per hand and scored with the solver's O(n) showdown sweep.

use crate::hands::{combo_cards, HandSet, N_COMBOS};
use crate::terminal::{compat_mass, showdown_net};
use engine::rng::below;
use engine::{Card, Evaluator, HandAcc};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

/// Equity of every combo against `opp` (1326 weights) on `board` (3 or 4 cards). Combos
/// blocked by the board, or with no compatible opponent mass, get 0.
pub fn range_equity(board: &[Card], opp: &[f32], samples: u32, seed: u64) -> Vec<f32> {
    assert!(board.len() == 3 || board.len() == 4, "flop or turn board");
    assert_eq!(opp.len(), N_COMBOS);
    let mut dead = 0u64;
    for c in board {
        dead |= 1 << c.0;
    }
    let live: Vec<u8> = (0..52u8).filter(|&c| dead & (1 << c) == 0).collect();
    let runouts: Vec<Vec<u8>> = if board.len() == 4 {
        live.iter().map(|&c| vec![c]).collect()
    } else {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        (0..samples.max(1))
            .map(|_| {
                let i = below(&mut rng, live.len() as u32) as usize;
                let mut j = below(&mut rng, live.len() as u32 - 1) as usize;
                if j >= i {
                    j += 1;
                }
                vec![live[i], live[j]]
            })
            .collect()
    };
    let ev = Evaluator::get();
    let table = combo_cards();
    let mut num = vec![0f64; N_COMBOS];
    let mut den = vec![0f64; N_COMBOS];
    let base = HandAcc::from_cards(board);
    let mut strength = Vec::with_capacity(N_COMBOS);
    let mut cards = Vec::with_capacity(N_COMBOS);
    let mut idx = Vec::with_capacity(N_COMBOS);
    for ro in &runouts {
        let mut d = dead;
        let mut acc = base;
        for &c in ro {
            d |= 1 << c;
            acc = acc.with(Card(c));
        }
        strength.clear();
        cards.clear();
        idx.clear();
        for (i, &[a, b]) in table.iter().enumerate() {
            if d & (1 << a) != 0 || d & (1 << b) != 0 {
                continue;
            }
            strength.push(ev.eval_acc(acc.with(Card(a)).with(Card(b))) as u32);
            cards.push([a, b]);
            idx.push(i);
        }
        let (hs, order) = HandSet::new(&strength, Some(&cards));
        let q: Vec<f32> = order.iter().map(|&k| opp[idx[k]]).collect();
        let mut net = vec![0f32; hs.len()];
        let mut mass = vec![0f32; hs.len()];
        showdown_net(&hs, &q, &mut net);
        compat_mass(&hs, &q, &mut mass);
        for (s, &k) in order.iter().enumerate() {
            // win − loss = net, win + loss + tie = mass: equity = (mass + net) / 2 / mass.
            let g = idx[k];
            num[g] += 0.5 * (mass[s] as f64 + net[s] as f64);
            den[g] += mass[s] as f64;
        }
    }
    (0..N_COMBOS)
        .map(|g| {
            if den[g] > 0.0 {
                (num[g] / den[g]) as f32
            } else {
                0.0
            }
        })
        .collect()
}
