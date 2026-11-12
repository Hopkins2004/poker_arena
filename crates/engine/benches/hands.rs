//! Full-hand throughput: 6-max NLHE, 200bb stacks reset every hand, random-legal policy.
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use engine::rng::{below, unit_f64};
use engine::{Action, Deck, GameFormat, HandSetup, HandState, LegalActions, MAX_SEATS};
use rand_xoshiro::rand_core::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

/// Random-legal policy: facing a bet fold 30% / call 50% / raise 20%; otherwise check 60% /
/// bet 40%. Bet sizes are 0.5, 1 or 2 pots (clamped to the legal range).
fn policy<R: Rng>(rng: &mut R, la: &LegalActions, pot: u64, current_bet: u64) -> Action {
    let u = unit_f64(rng);
    let raise = |rng: &mut R| {
        let frac = [0.5, 1.0, 2.0][below(rng, 3) as usize];
        let to = current_bet + ((pot + la.call_amount) as f64 * frac) as u64;
        Action::RaiseTo(to.clamp(la.min_raise_to, la.max_raise_to))
    };
    if la.can_check {
        if u < 0.6 || !la.can_raise {
            Action::Check
        } else {
            raise(rng)
        }
    } else if u < 0.3 {
        Action::Fold
    } else if u < 0.8 || !la.can_raise {
        Action::Call
    } else {
        raise(rng)
    }
}

fn play_hands(st: &mut HandState, rng: &mut Xoshiro256PlusPlus, deck: &mut Deck, n: u64) -> u64 {
    let fmt = GameFormat::nlhe(6);
    let mut setup = HandSetup {
        n_seats: 6,
        button: 0,
        stacks: [0; MAX_SEATS],
        sb: 1,
        bb: 2,
        ante: 0,
    };
    let mut actions = 0;
    for h in 0..n {
        setup.button = (h % 6) as usize;
        setup.stacks[..6].fill(400);
        deck.shuffle_prefix(rng, 6 * 2 + 5);
        st.start(&fmt, &setup, deck).unwrap();
        while st.to_act().is_some() {
            let a = policy(rng, &st.legal_actions(), st.pot(), st.current_bet());
            st.apply(a).unwrap();
            actions += 1;
        }
    }
    actions
}

fn bench(c: &mut Criterion) {
    const N: u64 = 20_000;
    let mut st = HandState::new();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
    let mut deck = Deck::new();
    let actions = play_hands(&mut st, &mut rng, &mut deck, N);
    println!("avg actions/hand: {:.2}", actions as f64 / N as f64);
    let mut g = c.benchmark_group("hands");
    g.throughput(Throughput::Elements(N));
    g.bench_function("nlhe_6max_random_legal", |b| {
        b.iter(|| play_hands(&mut st, &mut rng, &mut deck, N))
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
