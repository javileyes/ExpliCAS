use std::hint::black_box;

use cas_ast::Context;
use cas_parser::parse;
use cas_session_core::context_snapshot::ContextSnapshot;
use cas_session_core::snapshot_io::{load_bincode, save_bincode_atomic};
use cas_session_core::store_snapshot::{
    CacheConfigSnapshot, EntryKindSnapshot, EntrySnapshot, SessionStoreSnapshot,
    SimplifiedCacheSnapshot,
};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use serde::{Deserialize, Serialize};
use tempfile::tempdir;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SnapshotFixture {
    context: ContextSnapshot,
    session: SessionStoreSnapshot<u64>,
}

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(25);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(2));
}

fn build_fixture(multiplier: usize) -> SnapshotFixture {
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
    let mut expr_ids = Vec::new();
    let mut texts = Vec::new();
    for n in 0..multiplier {
        for seed in seeds {
            let expr_id = parse(seed, &mut ctx).expect("parse fixture expr");
            expr_ids.push(expr_id);
            texts.push(format!("{seed} // {n}"));
        }
    }

    let entries = expr_ids
        .iter()
        .zip(texts)
        .enumerate()
        .map(|(i, (expr_id, raw_text))| EntrySnapshot {
            id: i as u64 + 1,
            raw_text,
            kind: EntryKindSnapshot::Expr(expr_id.index() as u32),
            simplified: (i % 2 == 0).then(|| SimplifiedCacheSnapshot {
                key: i as u64 + 1,
                expr: expr_id.index() as u32,
            }),
        })
        .collect::<Vec<_>>();

    let cache_order = (1..=entries.len() as u64).rev().collect::<Vec<_>>();

    SnapshotFixture {
        context: ContextSnapshot::from_context(&ctx),
        session: SessionStoreSnapshot {
            next_id: entries.len() as u64 + 1,
            entries,
            cache_order,
            cache_config: CacheConfigSnapshot {
                max_cached_entries: 256,
                max_cached_steps: 10_000,
                light_cache_threshold: Some(8),
            },
            cached_steps_count: multiplier * 64,
        },
    }
}

fn bench_snapshot_io(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot_io");
    configure_group(&mut group);

    for (name, multiplier) in [("medium", 8usize), ("large", 32usize)] {
        let fixture = build_fixture(multiplier);

        group.bench_with_input(BenchmarkId::new("save", name), &fixture, |b, fixture| {
            b.iter_batched(
                || {
                    let tmp = tempdir().expect("tempdir failed");
                    let path = tmp.path().join("session.bin");
                    (tmp, path, fixture.clone())
                },
                |(_tmp, path, fixture)| {
                    save_bincode_atomic(&fixture, &path).unwrap();
                    black_box(())
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("load", name), &fixture, |b, fixture| {
            b.iter_batched(
                || {
                    let tmp = tempdir().expect("tempdir failed");
                    let path = tmp.path().join("session.bin");
                    save_bincode_atomic(fixture, &path).expect("seed snapshot");
                    (tmp, path)
                },
                |(_tmp, path)| black_box(load_bincode::<SnapshotFixture>(&path).unwrap()),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_snapshot_io);
criterion_main!(benches);
