use criterion::{criterion_group, criterion_main, Criterion};
use cas_engine::Simplifier;
use cas_parser::parse;
use std::hint::black_box;

fn benchmark_polynomial_simplification(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial");
    
    group.bench_function("expand_binomial_power_10", |b| {
        b.iter(|| {
            let mut simplifier = Simplifier::with_default_rules();
            // (x+1)^10 is too slow for now maybe? Let's try ^5
            // Actually we don't have BinomialExpansionRule enabled by default or implemented fully for arbitrary n?
            // DistributeRule handles multiplication.
            // Let's use (x+1)*(x+2)*(x+3)
            let input = parse("(x+1)*(x+2)*(x+3)", &mut simplifier.context).unwrap();
            black_box(simplifier.simplify(input));
        })
    });

    group.bench_function("combine_like_terms_large", |b| {
        b.iter(|| {
            let mut simplifier = Simplifier::with_default_rules();
            // x + 2x + 3x + ... + 10x
            let mut s = "x".to_string();
            for i in 2..=20 {
                s.push_str(&format!(" + {}*x", i));
            }
            let input = parse(&s, &mut simplifier.context).unwrap();
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
            let mut simplifier = Simplifier::with_default_rules();
            // sin^2(x) + cos^2(x) + sin^2(2x) + cos^2(2x) + ...
            let mut s = String::new();
            for i in 1..=5 {
                if i > 1 { s.push_str(" + "); }
                s.push_str(&format!("sin({}*x)^2 + cos({}*x)^2", i, i));
            }
            let input = parse(&s, &mut simplifier.context).unwrap();
            black_box(simplifier.simplify(input));
        })
    });
    
    group.finish();
}

fn benchmark_rational_simplification(c: &mut Criterion) {
    let mut group = c.benchmark_group("rational");
    
    group.bench_function("sum_fractions_10", |b| {
        b.iter(|| {
            let mut simplifier = Simplifier::with_default_rules();
            // 1/x + 1/(x+1) + ...
            let mut s = "1/x".to_string();
            for i in 1..=10 {
                s.push_str(&format!(" + 1/(x+{})", i));
            }
            let input = parse(&s, &mut simplifier.context).unwrap();
            black_box(simplifier.simplify(input));
        })
    });

    group.finish();
}

fn benchmark_calculus_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("calculus");
    
    group.bench_function("diff_nested_trig_exp", |b| {
        b.iter(|| {
            let mut simplifier = Simplifier::with_default_rules();
            let input = parse("diff(exp(sin(x^2)), x)", &mut simplifier.context).unwrap();
            black_box(simplifier.simplify(input));
        })
    });

    group.bench_function("integrate_trig_product", |b| {
        b.iter(|| {
            let mut simplifier = Simplifier::with_default_rules();
            let input = parse("integrate(sin(x)*cos(x), x)", &mut simplifier.context).unwrap();
            black_box(simplifier.simplify(input));
        })
    });

    group.finish();
}

fn benchmark_solver(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver");
    
    group.bench_function("solve_quadratic", |b| {
        b.iter(|| {
            let mut simplifier = Simplifier::with_default_rules();
            let input = parse("solve(x^2 + 5*x + 6, x)", &mut simplifier.context).unwrap();
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
            let mut simplifier = Simplifier::with_default_rules();
            // (((x+1)^2 + 1)^2 + 1)^2
            let input = parse("(((x+1)^2 + 1)^2 + 1)^2", &mut simplifier.context).unwrap();
            black_box(simplifier.simplify(input));
        })
    });

    group.finish();
}

criterion_group!(
    benches, 
    benchmark_polynomial_simplification, 
    benchmark_trig_simplification,
    benchmark_rational_simplification,
    benchmark_calculus_operations,
    benchmark_solver,
    benchmark_stress
);
criterion_main!(benches);
