//! Solve-time benchmark: `cargo run --release -p solver --example bench_solve [spots]`.
//! Random boards, SPR in {0.5, 1, 2, 4, 10}, random ranges; 500 iterations, single thread.

use engine::rng::below;
use engine::Card;
use rand_xoshiro::rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use solver::{solve_river, Abstraction, Params, Tree, N_COMBOS};
use std::time::Instant;

fn main() {
    let spots: u64 = std::env::args().nth(1).map_or(40, |s| s.parse().unwrap());
    let mut r = Xoshiro256PlusPlus::seed_from_u64(7);
    let mut times = vec![];
    let mut to_target = vec![];
    for _ in 0..spots {
        let mut deck: Vec<u8> = (0..52).collect();
        for i in 0..5 {
            let j = i + below(&mut r, (52 - i) as u32) as usize;
            deck.swap(i, j);
        }
        let board = [0, 1, 2, 3, 4].map(|i| Card(deck[i]));
        let stack = [0.5f32, 1.0, 2.0, 4.0, 10.0][below(&mut r, 5) as usize];
        let mut range = || -> Vec<f32> {
            (0..N_COMBOS)
                .map(|_| {
                    if below(&mut r, 3) == 0 {
                        0.0
                    } else {
                        (r.next_u32() >> 8) as f32
                    }
                })
                .collect()
        };
        let (r1, r2) = (range(), range());
        let fixed = Params {
            max_iters: 500,
            target_expl: 0.0,
            ..Params::default()
        };
        let t0 = Instant::now();
        let s = solve_river(&board, stack, [&r1, &r2], &Abstraction::default(), &fixed).unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1e3);
        let hit = s
            .solution
            .history
            .iter()
            .find(|x| x.1 < 0.002)
            .map_or(u32::MAX, |x| x.0);
        to_target.push(hit);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    to_target.sort();
    let pct = |v: &[f64], p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    println!(
        "nodes at SPR 0.5/1/2/4/10: {:?}",
        [0.5f32, 1., 2., 4., 10.].map(|s| Tree::river(s, &Abstraction::default()).len())
    );
    println!(
        "500 iters: median {:.1} ms, p90 {:.1} ms over {} spots",
        pct(&times, 0.5),
        pct(&times, 0.9),
        spots
    );
    let f: Vec<f64> = to_target.iter().map(|&x| x as f64).collect();
    println!(
        "iterations to 0.2% pot (checked every 50): median {}, p90 {}, never {}",
        pct(&f, 0.5),
        pct(&f, 0.9),
        to_target.iter().filter(|&&x| x == u32::MAX).count()
    );
}
