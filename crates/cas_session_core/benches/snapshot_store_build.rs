use std::hint::black_box;

use cas_ast::{Context, ExprId};
use cas_parser::parse;
use cas_session_core::store::SessionStore;
use cas_session_core::store_snapshot::{
    snapshot_from_store_with, EntryKindSnapshot, SimplifiedCacheSnapshot,
};
use cas_session_core::types::{CacheConfig, EntryKind};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

#[derive(Debug, Clone)]
struct BenchCache {
    key: u64,
    expr: ExprId,
}

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(25);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(2));
}

fn build_store(multiplier: usize) -> SessionStore<(), BenchCache> {
    let seeds = [
        "x + 1",
        "2 * 3 + 4",
        "sqrt(12*x^3)",
        "((5*x + 8)^2)^(1/2)",
        "(2*x + 2*y)/(4*x + 4*y)",
        "((x+y)*(a+b))/((x+y)*(c+d))",
        "sin(2*x + 1)^2 + cos(1 + 2*x)^2",
        "log(x^2, x^6)",
    ];

    let mut ctx = Context::new();
    let mut store = SessionStore::<(), BenchCache>::with_cache_config(CacheConfig {
        max_cached_entries: 256,
        max_cached_steps: 10_000,
        light_cache_threshold: Some(8),
    });

    for n in 0..multiplier {
        for seed in seeds {
            let expr = parse(seed, &mut ctx).expect("parse fixture expr");
            let id = store.push(EntryKind::Expr(expr), format!("{seed} // {n}"));
            if id.is_multiple_of(2) {
                store.update_simplified(id, BenchCache { key: id, expr });
            }
        }
    }

    store
}

fn bench_snapshot_store_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot_store_build");
    configure_group(&mut group);

    for (name, multiplier) in [("medium", 8usize), ("large", 32usize)] {
        let store = build_store(multiplier);
        group.bench_with_input(BenchmarkId::new("store", name), &store, |b, store| {
            b.iter(|| {
                black_box(snapshot_from_store_with(
                    store,
                    |kind| match kind {
                        EntryKind::Expr(expr) => EntryKindSnapshot::Expr(expr.index() as u32),
                        EntryKind::Eq { lhs, rhs } => EntryKindSnapshot::Eq {
                            lhs: lhs.index() as u32,
                            rhs: rhs.index() as u32,
                        },
                    },
                    |cache: &BenchCache| {
                        Some(SimplifiedCacheSnapshot {
                            key: cache.key,
                            expr: cache.expr.index() as u32,
                        })
                    },
                ))
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_snapshot_store_build);
criterion_main!(benches);
