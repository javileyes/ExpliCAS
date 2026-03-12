mod common;

use common::configure_standard_group;
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_repl_frontend(c: &mut Criterion) {
    let mut group = c.benchmark_group("frontend_repl");
    configure_standard_group(&mut group);

    let parse_cases = [
        ("eval/light/x_plus_1", "x + 1"),
        ("session/show", "show 12"),
        ("session/reset_full", "reset full"),
        ("analysis/solve", "solve x + 1 = 2"),
        ("analysis/solve_system", "solve_system x+y=3; x-y=1; x; y"),
        ("algebra/expand_log", "expand_log ln(x*y)"),
    ];

    for (name, input) in parse_cases {
        group.bench_function(format!("parse/{name}"), |b| {
            b.iter(|| cas_solver::session_api::repl::parse_repl_command_input(input))
        });
    }

    let preprocess_cases = [
        ("unchanged/eval", "x + 1"),
        ("function/simplify", "simplify(x^2 + 1)"),
        ("function/solve", "solve(x + 1 = 2, x)"),
    ];

    for (name, input) in preprocess_cases {
        group.bench_function(format!("preprocess/{name}"), |b| {
            b.iter(|| cas_solver::session_api::repl::preprocess_repl_function_syntax(input))
        });
    }

    let split_cases = [
        ("plain/multi", "let a = 1; let b = 2; a + b"),
        ("solve_system/raw", "solve_system x+y=3; x-y=1; x; y"),
    ];

    for (name, input) in split_cases {
        group.bench_function(format!("split/{name}"), |b| {
            b.iter(|| cas_solver::session_api::repl::split_repl_statements(input))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_repl_frontend);
criterion_main!(benches);
