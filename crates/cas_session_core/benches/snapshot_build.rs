use std::hint::black_box;

use cas_ast::Context;
use cas_parser::parse;
use cas_session_core::context_snapshot::ContextSnapshot;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

fn configure_group(group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>) {
    group.sample_size(25);
    group.warm_up_time(std::time::Duration::from_secs(1));
    group.measurement_time(std::time::Duration::from_secs(2));
}

fn build_context(multiplier: usize) -> Context {
    let mut ctx = Context::new();
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
            parse(&seed, &mut ctx).expect("parse fixture expr");
        }
    }
    ctx
}

fn bench_snapshot_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot_build");
    configure_group(&mut group);

    for (name, multiplier) in [("medium", 8usize), ("large", 32usize)] {
        let ctx = build_context(multiplier);
        group.bench_with_input(BenchmarkId::new("context", name), &ctx, |b, ctx| {
            b.iter_batched(
                || ctx.clone(),
                |ctx| black_box(ContextSnapshot::from_context(&ctx)),
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_snapshot_build);
criterion_main!(benches);
