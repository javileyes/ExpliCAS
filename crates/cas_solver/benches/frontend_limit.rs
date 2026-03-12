mod common;

use std::hint::black_box;

use cas_solver::command_api::limit::{
    evaluate_limit_subcommand, LimitCommandApproach, LimitCommandPreSimplify,
};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_frontend_limit(c: &mut Criterion) {
    let cases = [
        (
            "light/rational_infinity",
            "(x^2+1)/(2*x^2-3)",
            "x",
            LimitCommandApproach::Infinity,
            LimitCommandPreSimplify::Off,
        ),
        (
            "safe/subtraction_cancel",
            "(x-x)/x",
            "x",
            LimitCommandApproach::Infinity,
            LimitCommandPreSimplify::Safe,
        ),
        (
            "safe/nested_add_zero",
            "((x+0)+0)/x",
            "x",
            LimitCommandApproach::Infinity,
            LimitCommandPreSimplify::Safe,
        ),
        (
            "safe/irrational_constant",
            "1/(1+sqrt(2))",
            "x",
            LimitCommandApproach::Infinity,
            LimitCommandPreSimplify::Safe,
        ),
    ];

    let mut group = c.benchmark_group("frontend_limit");
    common::configure_standard_group(&mut group);

    for (name, expr, var, approach, presimplify) in &cases {
        group.bench_with_input(
            BenchmarkId::new("wire", name),
            &(*expr, *var, *approach, *presimplify),
            |b, (expr, var, approach, presimplify)| {
                b.iter(|| {
                    black_box(evaluate_limit_subcommand(
                        expr,
                        var,
                        *approach,
                        *presimplify,
                        true,
                    ))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("text", name),
            &(*expr, *var, *approach, *presimplify),
            |b, (expr, var, approach, presimplify)| {
                b.iter(|| {
                    black_box(evaluate_limit_subcommand(
                        expr,
                        var,
                        *approach,
                        *presimplify,
                        false,
                    ))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_frontend_limit);
criterion_main!(benches);
