//! End-to-end REPL benchmark: parse → resolve → simplify → display
//! Measures the full user-facing latency with and without ProfileCache

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use cas_ast::Context;
use cas_formatter::DisplayExpr;
use cas_parser::parse;

use cas_engine::{
    BranchMode, ComplexMode, ContextMode, EvalOptions, ProfileCache, Simplifier, StepsMode,
};

/// Full REPL flow: parse → simplify → format result
fn full_eval(profile_cache: &mut ProfileCache, opts: &EvalOptions, input: &str) -> String {
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("parse failed");

    let profile = profile_cache.get_or_build(opts);
    let mut simplifier = Simplifier::from_profile(profile);
    simplifier.context = ctx;

    let (result, _steps) = simplifier.simplify(expr);

    format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    )
}

/// Full REPL flow with explicit StepsMode: parse → simplify → format result
/// Consumes domain_warnings to prevent optimizer from skipping side-channel work
fn full_eval_with_mode(
    profile_cache: &mut ProfileCache,
    opts: &EvalOptions,
    input: &str,
    steps_mode: StepsMode,
) -> (String, usize) {
    let mut ctx = Context::new();
    let expr = parse(input, &mut ctx).expect("parse failed");

    let profile = profile_cache.get_or_build(opts);
    let mut simplifier = Simplifier::from_profile(profile);
    simplifier.context = ctx;
    simplifier.set_steps_mode(steps_mode);

    let (result, steps) = simplifier.simplify(expr);
    let warnings = simplifier.take_domain_warnings(); // Consume to prevent optimization

    let display = format!(
        "{}",
        DisplayExpr {
            context: &simplifier.context,
            id: result
        }
    );
    (display, steps.len() + warnings.len())
}

fn bench_repl_end_to_end(c: &mut Criterion) {
    // Representative input set (mixed difficulty)
    let inputs = [
        // Light (overhead dominates)
        "x + 1",
        "2 * 3 + 4",
        "sin(x)^2 + cos(x)^2",
        // Medium (some simplification)
        "sqrt(12*x^3)",
        "((5*x + 8)^2)^(1/2)",
        "i^5",
        // GCD multivar
        "(2*x + 2*y)/(4*x + 4*y)",
        "(x^2 - y^2)/(x - y)",
        "((x+y)*(a+b))/((x+y)*(c+d))",
        // Complex
        "(3 + 4*i)/(1 + 2*i)",
        // Trig
        "sin(2*x + 1)^2 + cos(1 + 2*x)^2",
    ];

    let opts = EvalOptions {
        branch_mode: BranchMode::Strict,
        shared: cas_engine::phase::SharedSemanticConfig {
            context_mode: ContextMode::Standard,
            ..Default::default()
        },
        complex_mode: ComplexMode::Auto,
        steps_mode: StepsMode::On,
        ..Default::default()
    };

    let mut group = c.benchmark_group("repl_full_eval");

    // Cached: persistent ProfileCache across all inputs
    group.bench_function("cached/batch_11_inputs", |b| {
        b.iter(|| {
            let mut cache = ProfileCache::new();
            let _ = cache.get_or_build(&opts); // warm once
            for input in &inputs {
                black_box(full_eval(&mut cache, &opts, input));
            }
        })
    });

    // Uncached: new ProfileCache per input
    group.bench_function("uncached/batch_11_inputs", |b| {
        b.iter(|| {
            for input in &inputs {
                let mut cache = ProfileCache::new(); // cold cache each time
                black_box(full_eval(&mut cache, &opts, input));
            }
        })
    });

    group.finish();

    // Individual input benchmarks
    let mut individual = c.benchmark_group("repl_individual");

    // Pre-warm cache
    let mut warm_cache = ProfileCache::new();
    let _ = warm_cache.get_or_build(&opts);

    for (i, input) in inputs.iter().enumerate() {
        let name = format!(
            "{:02}_{}",
            i,
            &input[..input.len().min(20)].replace(' ', "_")
        );

        // Cached
        individual.bench_with_input(BenchmarkId::new("cached", &name), input, |b, input| {
            b.iter(|| black_box(full_eval(&mut warm_cache, &opts, input)))
        });

        // Uncached
        individual.bench_with_input(BenchmarkId::new("uncached", &name), input, |b, input| {
            b.iter(|| {
                let mut cold = ProfileCache::new();
                black_box(full_eval(&mut cold, &opts, input))
            })
        });
    }

    individual.finish();
}

/// Compare StepsMode::On vs Off vs Compact
/// Tests: batch + light + heavy expressions
fn bench_steps_mode_comparison(c: &mut Criterion) {
    // Representative input set (mixed difficulty)
    let inputs = [
        "x + 1",
        "2 * 3 + 4",
        "sin(x)^2 + cos(x)^2",
        "sqrt(12*x^3)",
        "((5*x + 8)^2)^(1/2)",
        "i^5",
        "(2*x + 2*y)/(4*x + 4*y)",
        "(x^2 - y^2)/(x - y)",
        "((x+y)*(a+b))/((x+y)*(c+d))",
        "(3 + 4*i)/(1 + 2*i)",
        "sin(2*x + 1)^2 + cos(1 + 2*x)^2",
    ];

    let opts = EvalOptions {
        branch_mode: BranchMode::Strict,
        shared: cas_engine::phase::SharedSemanticConfig {
            context_mode: ContextMode::Standard,
            ..Default::default()
        },
        complex_mode: ComplexMode::Auto,
        steps_mode: StepsMode::On,
        ..Default::default()
    };

    let mut group = c.benchmark_group("steps_mode_comparison");

    // ========== BATCH BENCHMARKS ==========
    // StepsMode::On - batch
    group.bench_function("batch_11/steps_on", |b| {
        b.iter(|| {
            let mut cache = ProfileCache::new();
            let _ = cache.get_or_build(&opts); // warm once
            for input in &inputs {
                black_box(full_eval_with_mode(&mut cache, &opts, input, StepsMode::On));
            }
        })
    });

    // StepsMode::Compact - batch
    group.bench_function("batch_11/steps_compact", |b| {
        b.iter(|| {
            let mut cache = ProfileCache::new();
            let _ = cache.get_or_build(&opts);
            for input in &inputs {
                black_box(full_eval_with_mode(
                    &mut cache,
                    &opts,
                    input,
                    StepsMode::Compact,
                ));
            }
        })
    });

    // StepsMode::Off - batch
    group.bench_function("batch_11/steps_off", |b| {
        b.iter(|| {
            let mut cache = ProfileCache::new();
            let _ = cache.get_or_build(&opts);
            for input in &inputs {
                black_box(full_eval_with_mode(
                    &mut cache,
                    &opts,
                    input,
                    StepsMode::Off,
                ));
            }
        })
    });

    // ========== LIGHT EXPRESSION (overhead-dominated) ==========
    let light_input = "i^12345";

    group.bench_function("light/steps_on", |b| {
        let mut cache = ProfileCache::new();
        let _ = cache.get_or_build(&opts);
        b.iter(|| {
            black_box(full_eval_with_mode(
                &mut cache,
                &opts,
                light_input,
                StepsMode::On,
            ))
        })
    });

    group.bench_function("light/steps_compact", |b| {
        let mut cache = ProfileCache::new();
        let _ = cache.get_or_build(&opts);
        b.iter(|| {
            black_box(full_eval_with_mode(
                &mut cache,
                &opts,
                light_input,
                StepsMode::Compact,
            ))
        })
    });

    group.bench_function("light/steps_off", |b| {
        let mut cache = ProfileCache::new();
        let _ = cache.get_or_build(&opts);
        b.iter(|| {
            black_box(full_eval_with_mode(
                &mut cache,
                &opts,
                light_input,
                StepsMode::Off,
            ))
        })
    });

    // ========== HEAVY EXPRESSION (algebra-dominated) ==========
    let heavy_input = "((x+y+z)*(x+2*y+3*z))/((x+y+z)*(2*x-y+z))";

    group.bench_function("heavy/steps_on", |b| {
        let mut cache = ProfileCache::new();
        let _ = cache.get_or_build(&opts);
        b.iter(|| {
            black_box(full_eval_with_mode(
                &mut cache,
                &opts,
                heavy_input,
                StepsMode::On,
            ))
        })
    });

    group.bench_function("heavy/steps_compact", |b| {
        let mut cache = ProfileCache::new();
        let _ = cache.get_or_build(&opts);
        b.iter(|| {
            black_box(full_eval_with_mode(
                &mut cache,
                &opts,
                heavy_input,
                StepsMode::Compact,
            ))
        })
    });

    group.bench_function("heavy/steps_off", |b| {
        let mut cache = ProfileCache::new();
        let _ = cache.get_or_build(&opts);
        b.iter(|| {
            black_box(full_eval_with_mode(
                &mut cache,
                &opts,
                heavy_input,
                StepsMode::Off,
            ))
        })
    });

    group.finish();
}

criterion_group!(benches, bench_repl_end_to_end, bench_steps_mode_comparison);
criterion_main!(benches);
