use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use cas_ast::Context;
use cas_parser::parse;

use cas_engine::options::{BranchMode, ComplexMode, ContextMode, EvalOptions};
use cas_engine::profile_cache::ProfileCache;
use cas_engine::Simplifier;

fn build_expr(input: &str) -> (Context, cas_ast::ExprId) {
    let mut ctx = Context::new();
    let id = parse(input, &mut ctx).expect("parse failed");
    (ctx, id)
}

fn bench_profile_build(c: &mut Criterion) {
    let opts = EvalOptions {
        branch_mode: BranchMode::Strict,
        context_mode: ContextMode::Standard,
        complex_mode: ComplexMode::Auto,
    };

    c.bench_function("profile_build/uncached", |b| {
        b.iter(|| {
            // "uncached": crear un cache vacío equivale a "construir el perfil cada vez"
            let mut cache = ProfileCache::new();
            let profile = cache.get_or_build(black_box(&opts));
            black_box(profile);
        })
    });

    c.bench_function("profile_build/cached_hit", |b| {
        let mut cache = ProfileCache::new();
        let _ = cache.get_or_build(&opts); // warm
        b.iter(|| {
            let profile = cache.get_or_build(black_box(&opts)); // hit
            black_box(profile);
        })
    });
}

fn bench_simplify_cached_vs_uncached(c: &mut Criterion) {
    let cases = [
        // "ligeras": aquí el cache suele notarse mucho
        ("light/x_plus_1", "x + 1"),
        ("light/pythagorean", "sin(2*x + 1)^2 + cos(1 + 2*x)^2"),
        // "pesadas": aquí el coste de simplificar domina y el cache se nota menos
        ("heavy/nested_root", "sqrt(12*x^3)"),
        ("heavy/abs_square", "((5*x + 8/3)*(5*x + 8/3))^(1/2)"),
    ];

    let opts = EvalOptions {
        branch_mode: BranchMode::Strict,
        context_mode: ContextMode::Standard,
        complex_mode: ComplexMode::Auto,
    };

    // Cached: perfil construido una vez
    let mut cached_cache = ProfileCache::new();
    let cached_profile = cached_cache.get_or_build(&opts);

    let mut group = c.benchmark_group("simplify_cached_vs_uncached");

    for (name, input) in cases {
        // 1) Cached
        group.bench_with_input(BenchmarkId::new("cached", name), &input, |b, input| {
            b.iter_batched(
                || build_expr(input),
                |(ctx, expr)| {
                    let mut s = Simplifier::from_profile(cached_profile.clone());
                    s.context = ctx;
                    let (out, _steps) = s.simplify(expr);
                    black_box(out);
                },
                criterion::BatchSize::SmallInput,
            )
        });

        // 2) Uncached (construye el profile cada iter)
        group.bench_with_input(BenchmarkId::new("uncached", name), &input, |b, input| {
            b.iter_batched(
                || {
                    let mut cache = ProfileCache::new(); // cache vacío ⇒ build
                    let profile = cache.get_or_build(&opts);
                    let (ctx, expr) = build_expr(input);
                    (profile, ctx, expr)
                },
                |(profile, ctx, expr)| {
                    let mut s = Simplifier::from_profile(profile);
                    s.context = ctx;
                    let (out, _steps) = s.simplify(expr);
                    black_box(out);
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_profile_build,
    bench_simplify_cached_vs_uncached
);
criterion_main!(benches);
