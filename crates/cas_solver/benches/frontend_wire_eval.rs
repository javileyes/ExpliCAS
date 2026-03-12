mod common;

use std::hint::black_box;

use cas_solver::wire::eval_str_to_wire;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_frontend_wire_eval(c: &mut Criterion) {
    let exprs = [
        ("light/x_plus_1", "x + 1"),
        ("gcd/scalar_multiple_fraction", "(2*x + 2*y)/(4*x + 4*y)"),
        ("trig/pythagorean_chain", "sin(2*x + 1)^2 + cos(1 + 2*x)^2"),
    ];
    let opts = [
        ("default_compact", "{}"),
        ("default_spaced", "{ }"),
        ("pretty_compact", r#"{"pretty":true}"#),
        ("pretty_spaced", r#"{ "pretty": true }"#),
        ("steps_compact", r#"{"steps":true}"#),
        ("steps_spaced", r#"{ "steps": true }"#),
        ("budget_cli_compact", r#"{"budget":{"preset":"cli"}}"#),
        ("budget_cli_spaced", r#"{ "budget": { "preset": "cli" } }"#),
        (
            "budget_strict_compact",
            r#"{"budget":{"preset":"cli","mode":"strict"}}"#,
        ),
        (
            "budget_strict_spaced",
            r#"{ "budget": { "preset": "cli", "mode": "strict" } }"#,
        ),
    ];

    let mut group = c.benchmark_group("frontend_wire_eval");
    common::configure_standard_group(&mut group);

    for (expr_name, expr) in &exprs {
        for (opts_name, opts_json) in &opts {
            let bench_name = format!("{expr_name}/{opts_name}");
            group.bench_with_input(
                BenchmarkId::new("eval_str_to_wire", bench_name),
                &(*expr, *opts_json),
                |b, (expr, opts_json)| {
                    b.iter(|| black_box(eval_str_to_wire(expr, opts_json)));
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_frontend_wire_eval);
criterion_main!(benches);
