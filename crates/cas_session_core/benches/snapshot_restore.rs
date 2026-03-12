use std::hint::black_box;

use cas_ast::{Context, ExprId};
use cas_parser::parse;
use cas_session_core::context_snapshot::ContextSnapshot;
use cas_session_core::store::SessionStore;
use cas_session_core::store_snapshot::{
    restore_store_from_snapshot_with, CacheConfigSnapshot, EntryKindSnapshot, EntrySnapshot,
    SessionStoreSnapshot, SimplifiedCacheSnapshot,
};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(25);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(2));
}

#[derive(Clone)]
struct RestoreFixture {
    context: ContextSnapshot,
    session: SessionStoreSnapshot<u64>,
}

fn build_fixture(multiplier: usize) -> RestoreFixture {
    let mut ctx = Context::new();
    let mut expr_ids = Vec::new();
    let mut texts = Vec::new();
    for n in 0..multiplier {
        let seeds = [
            format!("x{n} + 1"),
            "2 * 3 + 4".to_string(),
            format!("sqrt(12*x{n}^3)"),
            format!("((5*x{n} + 8)^2)^(1/2)"),
            format!("(2*x{n} + 2*y{n})/(4*x{n} + 4*y{n})"),
            format!("((x{n}+y{n})*(a{n}+b{n}))/((x{n}+y{n})*(c{n}+d{n}))"),
            format!("sin(2*x{n} + 1)^2 + cos(1 + 2*x{n})^2"),
            format!("log(x{n}^2, x{n}^6)"),
        ];
        for seed in seeds {
            let expr_id = parse(&seed, &mut ctx).expect("parse fixture expr");
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

    RestoreFixture {
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

fn restore_store(snapshot: SessionStoreSnapshot<u64>) -> SessionStore<(), u64> {
    restore_store_from_snapshot_with(
        snapshot,
        |kind| match kind {
            EntryKindSnapshot::Expr(expr) => {
                cas_session_core::types::EntryKind::Expr(ExprId::from_raw(expr))
            }
            EntryKindSnapshot::Eq { lhs, rhs } => cas_session_core::types::EntryKind::Eq {
                lhs: ExprId::from_raw(lhs),
                rhs: ExprId::from_raw(rhs),
            },
        },
        |cache| cache.key,
        SessionStore::with_cache_config,
    )
}

fn bench_snapshot_restore(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot_restore");
    configure_group(&mut group);

    for (name, multiplier) in [("medium", 8usize), ("large", 32usize)] {
        let fixture = build_fixture(multiplier);

        group.bench_with_input(BenchmarkId::new("context", name), &fixture, |b, fixture| {
            b.iter_batched(
                || fixture.context.clone(),
                |snapshot| black_box(snapshot.into_context()),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("store", name), &fixture, |b, fixture| {
            b.iter_batched(
                || fixture.session.clone(),
                |snapshot| black_box(restore_store(snapshot)),
                BatchSize::SmallInput,
            )
        });

        group.bench_with_input(BenchmarkId::new("bundle", name), &fixture, |b, fixture| {
            b.iter_batched(
                || fixture.clone(),
                |fixture| {
                    let ctx = fixture.context.into_context();
                    let store = restore_store(fixture.session);
                    black_box((ctx, store))
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_snapshot_restore);
criterion_main!(benches);
