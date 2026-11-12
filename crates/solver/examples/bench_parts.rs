//! Cost of the solver's pieces on one river hand set (n = 1081).
use engine::card::parse_cards;
use solver::terminal::TermScratch;
use solver::{Abstraction, NodeKind, Params, RiverHands, Solver, Tree};
use std::time::Instant;
fn main() {
    let b: [engine::Card; 5] = parse_cards("Kd8s5h3c2d").unwrap().try_into().unwrap();
    let rh = RiverHands::new(&b);
    let n = rh.set.len();
    let q: Vec<f32> = (0..n).map(|i| ((i * 7919) % 13) as f32 / 13.0).collect();
    let mut out = vec![0f32; n];
    let reps = 20000;
    let t = Instant::now();
    let mut sc = TermScratch::default();
    for _ in 0..reps {
        rh.set.index.prepare(&q, &mut sc);
    }
    let prep = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
    let t = Instant::now();
    for _ in 0..reps {
        rh.set.index.compat(&q, &sc, &mut out);
    }
    let fold = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
    let t = Instant::now();
    for _ in 0..reps {
        rh.set.index.net(&sc, &mut out);
    }
    let sd = t.elapsed().as_secs_f64() / reps as f64 * 1e6;
    println!("prepare {prep:.2} µs");
    let tree = Tree::river(10.0, &Abstraction::default());
    let (mut nf, mut ns, mut nd) = (0, 0, 0);
    for k in &tree.kind {
        match k {
            NodeKind::Fold => nf += 1,
            NodeKind::Showdown => ns += 1,
            _ => nd += 1,
        }
    }
    let r = vec![1.0 / n as f32; n];
    let mut s = Solver::new(&tree, &rh.set, [&r, &r]);
    let p = Params::default();
    let t = Instant::now();
    for _ in 0..100 {
        s.iterate(&p);
    }
    let it = t.elapsed().as_secs_f64() / 100.0 * 1e6;
    println!("n={n}: fold {fold:.2} µs, showdown {sd:.2} µs; SPR 10 tree: {nd} decisions, {nf} folds, {ns} showdowns");
    let term = 2.0 * (nf as f64 * fold + ns as f64 * sd + (nf + ns) as f64 * 0.6 * prep);
    println!(
        "iteration {it:.0} µs; terminals ≈ {term:.0} µs of it ({:.0}%)",
        100.0 * term / it
    );
}
