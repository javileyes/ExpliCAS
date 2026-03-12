use std::hint::black_box;
use std::time::Duration;

use cas_ast::Context;
use cas_session_core::store::SessionStore;
use cas_session_core::types::EntryKind;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(25);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
}

fn make_store(size: usize) -> SessionStore<(), ()> {
    let mut ctx = Context::new();
    let x = ctx.var("x");
    let mut store = SessionStore::<(), ()>::new();
    for i in 0..size {
        let expr = if i % 2 == 0 {
            x
        } else {
            ctx.num((i % 13) as i64)
        };
        store.push(EntryKind::Expr(expr), format!("expr_{i}"));
    }
    store
}

fn bench_store_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("store_lookup");
    configure_group(&mut group);

    for size in [32usize, 256, 1024, 4096] {
        let store = make_store(size);
        let head = 1u64;
        let tail = size as u64;
        let miss = tail + 1;

        group.bench_with_input(BenchmarkId::new("get/head", size), &store, |b, store| {
            b.iter(|| black_box(store.get(head)))
        });
        group.bench_with_input(BenchmarkId::new("get/tail", size), &store, |b, store| {
            b.iter(|| black_box(store.get(tail)))
        });
        group.bench_with_input(BenchmarkId::new("get/miss", size), &store, |b, store| {
            b.iter(|| black_box(store.get(miss)))
        });
        group.bench_with_input(
            BenchmarkId::new("contains/head", size),
            &store,
            |b, store| b.iter(|| black_box(store.contains(head))),
        );
        group.bench_with_input(
            BenchmarkId::new("contains/tail", size),
            &store,
            |b, store| b.iter(|| black_box(store.contains(tail))),
        );
        group.bench_with_input(
            BenchmarkId::new("contains/miss", size),
            &store,
            |b, store| b.iter(|| black_box(store.contains(miss))),
        );
    }

    group.finish();
}

criterion_group!(benches, bench_store_lookup);
criterion_main!(benches);
