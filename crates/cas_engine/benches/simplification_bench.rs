use cas_engine::Simplifier;
use cas_parser::parse;
use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn setup_bench(input_str: &str) -> (Simplifier, cas_ast::ExprId) {
    let mut simplifier = Simplifier::with_default_rules();
    let expr = parse(input_str, &mut simplifier.context).unwrap();
    (simplifier, expr)
}

fn benchmark_parser(c: &mut Criterion) {
    let mut group = c.benchmark_group("parser");

    group.bench_function("parse_complex_poly", |b| {
        b.iter(|| {
            let mut ctx = cas_ast::Context::new();
            black_box(parse("(((x+1)^2 + 1)^2 + 1)^2", &mut ctx).unwrap());
        })
    });

    group.finish();
}

fn benchmark_polynomial_simplification(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial");

    group.bench_function("expand_binomial_power_10", |b| {
        b.iter(|| {
            let (mut simplifier, input) = setup_bench("(x+1)*(x+2)*(x+3)");
            black_box(simplifier.simplify(input));
        })
    });

    group.bench_function("combine_like_terms_large", |b| {
        b.iter(|| {
            // x + 2x + 3x + ... + 10x
            let mut s = "x".to_string();
            for i in 2..=20 {
                s.push_str(&format!(" + {}*x", i));
            }
            let (mut simplifier, input) = setup_bench(&s);
            black_box(simplifier.simplify(input));
        })
    });

    group.finish();
}

fn benchmark_trig_simplification(c: &mut Criterion) {
    let mut group = c.benchmark_group("trigonometry");

    group.sample_size(10);
    group.bench_function("pythagorean_identity_nested", |b| {
        b.iter(|| {
            // sin^2(x) + cos^2(x) + sin^2(2x) + cos^2(2x) + ...
            let mut s = String::new();
            for i in 1..=5 {
                if i > 1 {
                    s.push_str(" + ");
                }
                s.push_str(&format!("sin({}*x)^2 + cos({}*x)^2", i, i));
            }
            let (mut simplifier, input) = setup_bench(&s);
            black_box(simplifier.simplify(input));
        })
    });

    group.finish();
}

fn benchmark_rational_simplification(c: &mut Criterion) {
    let mut group = c.benchmark_group("rational");

    group.bench_function("sum_fractions_10", |b| {
        b.iter(|| {
            // 1/x + 1/(x+1) + ...
            let mut s = "1/x".to_string();
            for i in 1..=10 {
                s.push_str(&format!(" + 1/(x+{})", i));
            }
            let (mut simplifier, input) = setup_bench(&s);
            black_box(simplifier.simplify(input));
        })
    });

    group.finish();
}

fn benchmark_calculus_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("calculus");

    group.bench_function("diff_nested_trig_exp", |b| {
        b.iter(|| {
            let (mut simplifier, input) = setup_bench("diff(exp(sin(x^2)), x)");
            black_box(simplifier.simplify(input));
        })
    });

    group.bench_function("integrate_trig_product", |b| {
        b.iter(|| {
            let (mut simplifier, input) = setup_bench("integrate(sin(x)*cos(x), x)");
            black_box(simplifier.simplify(input));
        })
    });

    group.finish();
}

fn benchmark_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver");

    group.bench_function("solve_quadratic", |b| {
        b.iter(|| {
            let (mut simplifier, input) = setup_bench("solve(x^2 + 5*x + 6, x)");
            black_box(simplifier.simplify(input));
        })
    });

    group.finish();
}

fn benchmark_stress(c: &mut Criterion) {
    let mut group = c.benchmark_group("stress");
    group.sample_size(10); // Lower sample size for slow tests

    group.bench_function("deeply_nested_poly", |b| {
        b.iter(|| {
            let (mut simplifier, input) = setup_bench("(((x+1)^2 + 1)^2 + 1)^2");
            black_box(simplifier.simplify(input));
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_parser,
    benchmark_polynomial_simplification,
    benchmark_trig_simplification,
    benchmark_rational_simplification,
    benchmark_calculus_operations,
    benchmark_solver,
    benchmark_stress
);
criterion_main!(benches);
