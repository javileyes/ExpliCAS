mod common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

use cas_ast::Context;
use cas_parser::parse;

use cas_engine::{
    BranchMode, ComplexMode, ContextMode, DomainMode, EvalOptions, ProfileCache, Simplifier,
    SimplifyOptions, StepsMode,
};

const SOLVE_PROFILE_FLAG: &str = "CAS_SOLVE_BENCH_PROFILE";
const SOLVE_PROFILE_MODE_VAR: &str = "CAS_SOLVE_BENCH_PROFILE_MODE";
const SOLVE_PROFILE_DETAIL_FLAG: &str = "CAS_SOLVE_BENCH_PROFILE_DETAIL";

fn build_expr(input: &str) -> (Context, cas_ast::ExprId) {
    let mut ctx = Context::new();
    let id = parse(input, &mut ctx).expect("parse failed");
    (ctx, id)
}

fn solve_profile_mode_filter() -> Option<String> {
    std::env::var(SOLVE_PROFILE_MODE_VAR)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty())
}

fn should_emit_solve_profile(mode_name: &str) -> bool {
    if !common::env_flag_enabled(SOLVE_PROFILE_FLAG) {
        return false;
    }

    match solve_profile_mode_filter() {
        Some(filter) => filter == mode_name,
        None => true,
    }
}

fn solve_profile_detail_enabled() -> bool {
    common::env_flag_enabled(SOLVE_PROFILE_DETAIL_FLAG)
}

fn emit_solve_profile_snapshot(
    mode_name: &str,
    options: &SimplifyOptions,
    inputs: &[&str],
    mut make_simplifier: impl FnMut(Context) -> Simplifier,
) {
    if !should_emit_solve_profile(mode_name) {
        return;
    }

    let mut total_len = 0usize;
    let mut aggregate = cas_engine::RuleProfiler::new(true);
    aggregate.enable_health();

    for input in inputs {
        let (ctx, expr) = build_expr(input);
        let mut simplifier = make_simplifier(ctx);
        simplifier.profiler.enable_health();
        simplifier.profiler.clear_run();

        let (out, _steps) = simplifier.simplify_with_options(expr, options.clone());
        total_len ^= format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: out
            }
        )
        .len();

        if solve_profile_detail_enabled() {
            let rendered = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: out
                }
            );
            println!("solve_profile_input[{mode_name}] input={input:?} output={rendered:?}");
            for &phase in cas_engine::SimplifyPhase::all() {
                let top = simplifier.profiler.top_applied_for_phase(phase, 5);
                if top.is_empty() {
                    continue;
                }

                let summary = top
                    .into_iter()
                    .map(|(rule, hits)| format!("{rule}={hits}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("  {:?}: {}", phase, summary);
            }
        }

        for &phase in cas_engine::SimplifyPhase::all() {
            for (rule_name, hits) in simplifier.profiler.top_applied_for_phase(phase, usize::MAX) {
                for _ in 0..hits {
                    aggregate.record(phase, &rule_name);
                }
            }
        }
    }

    println!("solve_profile[{mode_name}] total_output_len={total_len}");
    for &phase in cas_engine::SimplifyPhase::all() {
        let top = aggregate.top_applied_for_phase(phase, 5);
        if top.is_empty() {
            continue;
        }

        let summary = top
            .into_iter()
            .map(|(rule, hits)| format!("{rule}={hits}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!("  {:?}: {}", phase, summary);
    }
}

fn bench_profile_build(c: &mut Criterion) {
    let opts = EvalOptions {
        branch_mode: BranchMode::Strict,
        complex_mode: ComplexMode::Auto,
        steps_mode: StepsMode::On,
        shared: cas_engine::SharedSemanticConfig {
            context_mode: ContextMode::Standard,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut group = c.benchmark_group("profile_build");
    common::configure_standard_group(&mut group);

    group.bench_function("uncached", |b| {
        b.iter(|| {
            // "uncached": crear un cache vacío equivale a "construir el perfil cada vez"
            let mut cache = ProfileCache::new();
            let profile = cache.get_or_build(black_box(&opts));
            black_box(profile);
        })
    });

    group.bench_function("cached_hit", |b| {
        let mut cache = ProfileCache::new();
        let _ = cache.get_or_build(&opts); // warm
        b.iter(|| {
            let profile = cache.get_or_build(black_box(&opts)); // hit
            black_box(profile);
        })
    });

    group.finish();
}

fn bench_simplify_cached_vs_uncached(c: &mut Criterion) {
    let cases = [
        // "Light": cache overhead dominates
        ("light/x_plus_1", "x + 1"),
        ("light/pythagorean", "sin(2*x + 1)^2 + cos(1 + 2*x)^2"),
        // "Heavy": simplification cost dominates
        ("heavy/nested_root", "sqrt(12*x^3)"),
        ("heavy/abs_square", "((5*x + 8/3)*(5*x + 8/3))^(1/2)"),
        // GCD multivar: Layer 1 (monomial+content)
        ("gcd/layer1_content", "(2*x + 2*y)/(4*x + 4*y)"),
        // GCD multivar: Layer 2 (difference of squares)
        ("gcd/layer2_diff_squares", "(x^2 - y^2)/(x - y)"),
        // GCD multivar: Layer 2.5 candidate (multi-param factor)
        ("gcd/layer25_multiparam", "((x+y)*(a+b))/((x+y)*(c+d))"),
        // Complex numbers
        ("complex/gaussian_div", "(3 + 4*i)/(1 + 2*i)"),
        ("complex/i_power", "i^5"),
    ];

    let opts = EvalOptions {
        branch_mode: BranchMode::Strict,
        complex_mode: ComplexMode::Auto,
        steps_mode: StepsMode::On,
        shared: cas_engine::SharedSemanticConfig {
            context_mode: ContextMode::Standard,
            ..Default::default()
        },
        ..Default::default()
    };

    // Cached: perfil construido una vez
    let mut cached_cache = ProfileCache::new();
    let cached_profile = cached_cache.get_or_build(&opts);

    let mut group = c.benchmark_group("simplify_cached_vs_uncached");
    common::configure_standard_group(&mut group);

    for (name, input) in cases {
        // 1) Cached
        group.bench_with_input(BenchmarkId::new("cached", name), &input, |b, input| {
            b.iter_batched(
                || build_expr(input),
                |(ctx, expr)| {
                    let mut s = Simplifier::from_profile_with_context(cached_profile.clone(), ctx);
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
                    let mut s = Simplifier::from_profile_with_context(profile, ctx);
                    let (out, _steps) = s.simplify(expr);
                    black_box(out);
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_solve_modes_cached(c: &mut Criterion) {
    let inputs = [
        "(x^2 - y^2)/(x - y)",
        "(2*x + 2*y)/(4*x + 4*y)",
        "x/x",
        "exp(ln(x))",
        "(a^x)/a",
        "x^0",
    ];

    let profile_opts = EvalOptions {
        branch_mode: BranchMode::Strict,
        complex_mode: ComplexMode::Auto,
        steps_mode: StepsMode::Off,
        shared: cas_engine::SharedSemanticConfig {
            context_mode: ContextMode::Solve,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut cache = ProfileCache::new();
    let profile = cache.get_or_build(&profile_opts);

    let mut solve_prepass = SimplifyOptions::for_solve_prepass();
    solve_prepass.shared.context_mode = ContextMode::Solve;

    let mut tactic_strict = SimplifyOptions::for_solve_tactic(DomainMode::Strict);
    tactic_strict.collect_steps = false;
    tactic_strict.shared.context_mode = ContextMode::Solve;

    let mut tactic_generic = SimplifyOptions::for_solve_tactic(DomainMode::Generic);
    tactic_generic.collect_steps = false;
    tactic_generic.shared.context_mode = ContextMode::Solve;

    let mut tactic_assume = SimplifyOptions::for_solve_tactic(DomainMode::Assume);
    tactic_assume.collect_steps = false;
    tactic_assume.shared.context_mode = ContextMode::Solve;

    let mut group = c.benchmark_group("solve_modes_cached");
    common::configure_standard_group(&mut group);

    emit_solve_profile_snapshot("prepass", &solve_prepass, &inputs, |ctx| {
        Simplifier::from_profile_with_context(profile.clone(), ctx)
    });
    emit_solve_profile_snapshot("strict", &tactic_strict, &inputs, |ctx| {
        Simplifier::from_profile_with_context(profile.clone(), ctx)
    });
    emit_solve_profile_snapshot("generic", &tactic_generic, &inputs, |ctx| {
        Simplifier::from_profile_with_context(profile.clone(), ctx)
    });
    emit_solve_profile_snapshot("assume", &tactic_assume, &inputs, |ctx| {
        Simplifier::from_profile_with_context(profile.clone(), ctx)
    });

    group.bench_function("solve_prepass_batch", |b| {
        b.iter(|| {
            let mut total_len = 0usize;
            for input in &inputs {
                let (ctx, expr) = build_expr(input);
                let mut s = Simplifier::from_profile_with_context(profile.clone(), ctx);
                let (out, _steps) = s.simplify_with_options(expr, solve_prepass.clone());
                total_len ^= format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &s.context,
                        id: out
                    }
                )
                .len();
            }
            black_box(total_len)
        })
    });

    group.bench_function("solve_tactic_strict_batch", |b| {
        b.iter(|| {
            let mut total_len = 0usize;
            for input in &inputs {
                let (ctx, expr) = build_expr(input);
                let mut s = Simplifier::from_profile_with_context(profile.clone(), ctx);
                let (out, _steps) = s.simplify_with_options(expr, tactic_strict.clone());
                total_len ^= format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &s.context,
                        id: out
                    }
                )
                .len();
            }
            black_box(total_len)
        })
    });

    group.bench_function("solve_tactic_generic_batch", |b| {
        b.iter(|| {
            let mut total_len = 0usize;
            for input in &inputs {
                let (ctx, expr) = build_expr(input);
                let mut s = Simplifier::from_profile_with_context(profile.clone(), ctx);
                let (out, _steps) = s.simplify_with_options(expr, tactic_generic.clone());
                total_len ^= format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &s.context,
                        id: out
                    }
                )
                .len();
            }
            black_box(total_len)
        })
    });

    group.bench_function("solve_tactic_assume_batch", |b| {
        b.iter(|| {
            let mut total_len = 0usize;
            for input in &inputs {
                let (ctx, expr) = build_expr(input);
                let mut s = Simplifier::from_profile_with_context(profile.clone(), ctx);
                let (out, _steps) = s.simplify_with_options(expr, tactic_assume.clone());
                total_len ^= format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &s.context,
                        id: out
                    }
                )
                .len();
            }
            black_box(total_len)
        })
    });

    group.finish();
}

fn bench_solve_hotspots_cached(c: &mut Criterion) {
    let profile_opts = EvalOptions {
        branch_mode: BranchMode::Strict,
        complex_mode: ComplexMode::Auto,
        steps_mode: StepsMode::Off,
        shared: cas_engine::SharedSemanticConfig {
            context_mode: ContextMode::Solve,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut cache = ProfileCache::new();
    let profile = cache.get_or_build(&profile_opts);

    let mut tactic_generic = SimplifyOptions::for_solve_tactic(DomainMode::Generic);
    tactic_generic.collect_steps = false;
    tactic_generic.shared.context_mode = ContextMode::Solve;

    let mut tactic_assume = SimplifyOptions::for_solve_tactic(DomainMode::Assume);
    tactic_assume.collect_steps = false;
    tactic_assume.shared.context_mode = ContextMode::Solve;

    let cases = [
        (
            "generic/scalar_multiple_fraction",
            "(2*x + 2*y)/(4*x + 4*y)",
            tactic_generic.clone(),
        ),
        ("generic/x_over_x", "x/x", tactic_generic.clone()),
        ("generic/exp_ln_x", "exp(ln(x))", tactic_generic.clone()),
        ("assume/x_over_x", "x/x", tactic_assume.clone()),
        ("assume/exp_ln_x", "exp(ln(x))", tactic_assume.clone()),
    ];

    let mut group = c.benchmark_group("solve_hotspots_cached");
    common::configure_standard_group(&mut group);

    for (name, input, options) in cases {
        let input = input;
        let options = options.clone();
        group.bench_function(name, |b| {
            b.iter_batched(
                || build_expr(input),
                |(ctx, expr)| {
                    let mut s = Simplifier::from_profile_with_context(profile.clone(), ctx);
                    let (out, _steps) = s.simplify_with_options(expr, options.clone());
                    let output_len = format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &s.context,
                            id: out
                        }
                    )
                    .len();
                    black_box(output_len);
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
    bench_simplify_cached_vs_uncached,
    bench_solve_modes_cached,
    bench_solve_hotspots_cached
);
criterion_main!(benches);
