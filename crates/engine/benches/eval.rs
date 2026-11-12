//! Evaluator throughput: random 7-card hands (non-incremental), single-threaded.
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use engine::{Card, Deck, Evaluator};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use std::hint::black_box;

const N: usize = 1 << 20;

fn hands(seed: u64) -> Vec<[Card; 7]> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut d = Deck::new();
    (0..N)
        .map(|_| {
            d.shuffle_prefix(&mut rng, 7);
            d.cards()[..7].try_into().unwrap()
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let ev = Evaluator::get();
    let hs = hands(1);
    let mut g = c.benchmark_group("eval");
    g.throughput(Throughput::Elements(N as u64));
    g.bench_function("eval7_random", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for h in &hs {
                acc = acc.wrapping_add(ev.eval7(black_box(h)) as u32);
            }
            acc
        })
    });
    let omaha: Vec<([Card; 4], [Card; 5])> = hs
        .iter()
        .zip(hands(2))
        .map(|(a, b)| ([a[0], a[1], a[2], a[3]], [b[0], b[1], b[2], b[3], b[4]]))
        .filter(|(h, b)| h.iter().all(|c| !b.contains(c)))
        .take(1 << 16)
        .collect();
    g.throughput(Throughput::Elements(omaha.len() as u64));
    g.bench_function("omaha_random", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for (h, bd) in &omaha {
                acc = acc.wrapping_add(ev.eval_omaha(black_box(h), black_box(bd)) as u32);
            }
            acc
        })
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
