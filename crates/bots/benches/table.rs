//! Table-runner throughput with real bot instances (dyn Bot, event broadcast, stats).
use arena::{run_table, TableConfig};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use engine::GameFormat;
use serde_json::Value;

fn bench(c: &mut Criterion) {
    let fmt = GameFormat::nlhe(6);
    let mut g = c.benchmark_group("table");
    g.sample_size(20);
    for (label, kind, hands) in [
        ("random_6max", "random", 20_000u64),
        ("tag_6max", "tag", 1_000),
    ] {
        let f = vec![bots::factory(kind, kind, &Value::Null).unwrap()];
        let cfg = TableConfig::cash(hands, 1, 2, 100);
        g.throughput(Throughput::Elements(hands));
        g.bench_function(label, |b| {
            b.iter(|| run_table(&fmt, &[0; 6], &f, &cfg, 1).unwrap())
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
