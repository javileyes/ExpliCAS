mod common;

use std::hint::black_box;

use cas_ast::Context;
use cas_parser::{parse, parse_statement};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};

fn bench_frontend_parse(c: &mut Criterion) {
    let expr_inputs = [
        ("light/x_plus_1", "x + 1"),
        ("light/numeric_add_chain", "2 * 3 + 4"),
        ("gcd/scalar_multiple_fraction", "(2*x + 2*y)/(4*x + 4*y)"),
        ("gcd/common_factor_fraction", "((x+y)*(a+b))/((x+y)*(c+d))"),
        ("heavy/nested_root", "sqrt(12*x^3)"),
        ("heavy/abs_square", "((5*x + 8)^2)^(1/2)"),
        ("complex/gaussian_div", "(3 + 4*i)/(1 + 2*i)"),
        ("trig/pythagorean_chain", "sin(2*x + 1)^2 + cos(1 + 2*x)^2"),
    ];

    let statement_inputs = [
        ("solve/linear_eq", "2*x + 3 = 7"),
        ("solve/quadratic_eq", "x^2 - 1 = 0"),
        ("solve/fraction_eq", "(x^2 - y^2)/(x - y) = x + y"),
        ("solve/trig_eq", "sin(x)^2 + cos(x)^2 = 1"),
        ("relation/strict_less", "x < 3"),
    ];

    let mut group = c.benchmark_group("frontend_parse");
    common::configure_standard_group(&mut group);

    group.bench_function("context/new", |b| {
        b.iter(|| black_box(Context::new()));
    });

    group.bench_function("expr_batch/standard_8", |b| {
        b.iter(|| {
            for (_, input) in &expr_inputs {
                let mut ctx = Context::new();
                black_box(parse(input, &mut ctx).expect("expr parse failed"));
            }
        })
    });

    group.bench_function("statement_batch/solve_5", |b| {
        b.iter(|| {
            for (_, input) in &statement_inputs {
                let mut ctx = Context::new();
                black_box(parse_statement(input, &mut ctx).expect("statement parse failed"));
            }
        })
    });

    for (name, input) in expr_inputs {
        group.bench_with_input(BenchmarkId::new("expr", name), &input, |b, input| {
            b.iter_batched(
                Context::new,
                |mut ctx| black_box(parse(input, &mut ctx).expect("expr parse failed")),
                BatchSize::SmallInput,
            )
        });
    }

    for (name, input) in statement_inputs {
        group.bench_with_input(BenchmarkId::new("statement", name), &input, |b, input| {
            b.iter_batched(
                Context::new,
                |mut ctx| {
                    black_box(parse_statement(input, &mut ctx).expect("statement parse failed"))
                },
                BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_frontend_parse);
criterion_main!(benches);
