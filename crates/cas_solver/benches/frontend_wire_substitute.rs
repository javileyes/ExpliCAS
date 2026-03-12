mod common;

use std::hint::black_box;

use cas_solver::wire::substitute_str_to_wire;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_frontend_wire_substitute(c: &mut Criterion) {
    let inputs = [
        ("light/simple_exact", ("x^2 + 1", "x^2", "y")),
        ("light/power_aware", ("x^4 + x^2 + 1", "x^2", "y")),
    ];
    let opts = [
        ("none", None),
        ("default_compact", Some("{}")),
        ("default_spaced", Some("{ }")),
        ("steps_compact", Some(r#"{"steps":true}"#)),
        ("steps_spaced", Some(r#"{ "steps": true }"#)),
        ("exact_compact", Some(r#"{"mode":"exact"}"#)),
        ("exact_spaced", Some(r#"{ "mode": "exact" }"#)),
        ("power_compact", Some(r#"{"mode":"power"}"#)),
        ("power_spaced", Some(r#"{ "mode": "power" }"#)),
        (
            "pretty_steps_compact",
            Some(r#"{"steps":true,"pretty":true}"#),
        ),
        (
            "pretty_steps_spaced",
            Some(r#"{ "steps": true, "pretty": true }"#),
        ),
    ];

    let mut group = c.benchmark_group("frontend_wire_substitute");
    common::configure_standard_group(&mut group);

    for (input_name, (expr, target, with_expr)) in &inputs {
        for (opts_name, opts_json) in &opts {
            let bench_name = format!("{input_name}/{opts_name}");
            group.bench_with_input(
                BenchmarkId::new("substitute_str_to_wire", bench_name),
                &(*expr, *target, *with_expr, *opts_json),
                |b, (expr, target, with_expr, opts_json)| {
                    b.iter(|| {
                        black_box(substitute_str_to_wire(expr, target, with_expr, *opts_json))
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(benches, bench_frontend_wire_substitute);
criterion_main!(benches);
