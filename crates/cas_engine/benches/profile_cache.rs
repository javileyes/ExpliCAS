mod common;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cas_ast::views::RationalFnView;
use cas_ast::Context;
use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_parser::parse;

use cas_engine::rules::algebra::SimplifyFractionRule;
use cas_engine::{
    BranchMode, ComplexMode, ContextMode, DomainMode, EvalConfig, EvalOptions, ProfileCache, Rule,
    Simplifier, SimplifyOptions, StepListener, StepsMode,
};
use cas_math::fraction_gcd_plan_support::try_plan_fraction_gcd_rewrite;
use cas_solver::{verify_solution_set, VerifySummary};
use cas_solver_core::engine_events::EngineEvent;

const SOLVE_PROFILE_FLAG: &str = "CAS_SOLVE_BENCH_PROFILE";
const SOLVE_PROFILE_MODE_VAR: &str = "CAS_SOLVE_BENCH_PROFILE_MODE";
const SOLVE_PROFILE_DETAIL_FLAG: &str = "CAS_SOLVE_BENCH_PROFILE_DETAIL";
const SOLVE_PROFILE_PROBE_FLAG: &str = "CAS_SOLVE_BENCH_PROFILE_PROBE";
const SOLVE_PROFILE_PROBE_ITERS_VAR: &str = "CAS_SOLVE_BENCH_PROFILE_PROBE_ITERS";

fn build_expr(input: &str) -> (Context, cas_ast::ExprId) {
    let mut ctx = Context::new();
    let id = parse(input, &mut ctx).expect("parse failed");
    (ctx, id)
}

fn build_div_expr(input: &str) -> (Context, cas_ast::ExprId, cas_ast::ExprId, cas_ast::ExprId) {
    let (ctx, expr) = build_expr(input);
    let (num, den) = match ctx.get(expr) {
        Expr::Div(num, den) => (*num, *den),
        other => panic!("expected division, got {other:?}"),
    };
    (ctx, expr, num, den)
}

fn build_equation_with_solutions(
    lhs: &str,
    rhs: &str,
    var: &str,
    solutions: &[&str],
) -> (Simplifier, Equation, String, SolutionSet) {
    let mut simplifier = Simplifier::with_default_rules();
    let equation = Equation {
        lhs: parse(lhs, &mut simplifier.context).expect("lhs parse failed"),
        rhs: parse(rhs, &mut simplifier.context).expect("rhs parse failed"),
        op: RelOp::Eq,
    };
    let solutions = SolutionSet::Discrete(
        solutions
            .iter()
            .map(|expr| parse(expr, &mut simplifier.context).expect("solution parse failed"))
            .collect(),
    );
    (simplifier, equation, var.to_string(), solutions)
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

fn solve_profile_probe_enabled() -> bool {
    common::env_flag_enabled(SOLVE_PROFILE_PROBE_FLAG)
}

fn solve_profile_probe_iters() -> usize {
    std::env::var(SOLVE_PROFILE_PROBE_ITERS_VAR)
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|iters| *iters > 0)
        .unwrap_or(2_000)
}

fn summarize_rule_bucket(rules: &[Arc<dyn cas_engine::Rule>], limit: usize) -> String {
    let shown = rules
        .iter()
        .take(limit)
        .map(|rule| rule.name())
        .collect::<Vec<_>>()
        .join(", ");
    if rules.len() > limit {
        format!("{shown}, ... (+{} more)", rules.len() - limit)
    } else {
        shown
    }
}

fn print_no_hit_bucket_candidates(simplifier: &Simplifier, ctx: &Context, expr: cas_ast::ExprId) {
    let target_kind = cas_ast::target_kind::TargetKind::from_expr(ctx.get(expr));
    let rules = simplifier.get_rules_clone();
    let Some(bucket) = rules.get(&target_kind) else {
        println!("  target_kind={target_kind}");
        println!("  candidates: (no specific bucket)");
        return;
    };

    println!("  target_kind={target_kind}");
    for &phase in cas_engine::SimplifyPhase::all() {
        let phase_mask = phase.mask();
        let phase_bucket: Vec<_> = bucket
            .iter()
            .filter(|rule| rule.allowed_phases().contains(phase_mask))
            .cloned()
            .collect();
        if phase_bucket.is_empty() {
            continue;
        }

        let limit = if solve_profile_detail_enabled() {
            usize::MAX
        } else {
            8
        };
        println!(
            "  candidates[{phase:?}/{target_kind}; count={}]: {}",
            phase_bucket.len(),
            summarize_rule_bucket(&phase_bucket, limit)
        );
    }
}

fn build_probe_parent_ctx(
    ctx: &Context,
    expr: cas_ast::ExprId,
    options: &SimplifyOptions,
) -> cas_engine::ParentContext {
    let semantics = options.shared.semantics;
    cas_engine::ParentContext::root()
        .with_domain_mode(semantics.domain_mode)
        .with_value_domain(semantics.value_domain)
        .with_inv_trig(semantics.inv_trig)
        .with_goal(options.goal)
        .with_context_mode(options.shared.context_mode)
        .with_simplify_purpose(options.simplify_purpose)
        .with_autoexpand_binomials(options.shared.autoexpand_binomials)
        .with_heuristic_poly(options.shared.heuristic_poly)
        .with_expand_mode_flag(options.expand_mode)
        .with_root_expr(ctx, expr)
}

fn build_fraction_rule_parent_ctx(
    ctx: &Context,
    expr: cas_ast::ExprId,
    domain_mode: DomainMode,
) -> cas_engine::ParentContext {
    let mut opts = SimplifyOptions::for_solve_tactic(domain_mode);
    opts.collect_steps = false;
    build_probe_parent_ctx(ctx, expr, &opts)
}

fn print_rule_probes(simplifier: &Simplifier, options: &SimplifyOptions, expr: cas_ast::ExprId) {
    if !solve_profile_probe_enabled() {
        return;
    }

    let target_kind = cas_ast::target_kind::TargetKind::from_expr(simplifier.context.get(expr));
    let rules = simplifier.get_rules_clone();
    let Some(bucket) = rules.get(&target_kind) else {
        return;
    };

    let probe_parent_ctx = build_probe_parent_ctx(&simplifier.context, expr, options);
    let probe_iters = solve_profile_probe_iters();
    println!("  probe[target_kind={target_kind}; iters={probe_iters}]");

    for &phase in cas_engine::SimplifyPhase::all() {
        let phase_mask = phase.mask();
        let phase_bucket: Vec<_> = bucket
            .iter()
            .filter(|rule| rule.allowed_phases().contains(phase_mask))
            .cloned()
            .collect();
        if phase_bucket.is_empty() {
            continue;
        }

        let mut results = Vec::with_capacity(phase_bucket.len());
        for rule in phase_bucket {
            let mut probe_ctx = simplifier.context.clone();
            let start = Instant::now();
            let mut hits = 0usize;
            for _ in 0..probe_iters {
                if rule
                    .apply(&mut probe_ctx, expr, &probe_parent_ctx)
                    .is_some()
                {
                    hits += 1;
                }
            }
            let avg_us = start.elapsed().as_secs_f64() * 1_000_000.0 / probe_iters as f64;
            results.push((avg_us, hits, rule.name().to_string()));
        }

        results.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (avg_us, hits, rule_name) in results {
            println!(
                "  probe[{phase:?}/{target_kind}] rule={rule_name:?} avg_us={avg_us:.3} hits={hits}"
            );
        }
    }
}

#[derive(Clone)]
struct BenchCapturingListener {
    sink: Arc<Mutex<Vec<EngineEvent>>>,
}

impl BenchCapturingListener {
    fn new(sink: Arc<Mutex<Vec<EngineEvent>>>) -> Self {
        Self { sink }
    }
}

impl StepListener for BenchCapturingListener {
    fn on_event(&mut self, event: &EngineEvent) {
        self.sink
            .lock()
            .expect("bench listener sink poisoned")
            .push(event.clone());
    }
}

fn emit_solve_profile_snapshot(
    mode_name: &str,
    options: &SimplifyOptions,
    inputs: &[&str],
    mut make_simplifier: impl FnMut(Context) -> Simplifier,
    mut on_no_rule_hits: impl FnMut(&Context, cas_ast::ExprId, &Simplifier),
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
        let event_sink = if solve_profile_detail_enabled() {
            let sink = Arc::new(Mutex::new(Vec::new()));
            simplifier.set_step_listener(Some(Box::new(BenchCapturingListener::new(sink.clone()))));
            Some(sink)
        } else {
            None
        };
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
            if let Some(sink) = event_sink {
                let events = sink.lock().expect("bench listener sink poisoned");
                for (idx, event) in events.iter().enumerate() {
                    let EngineEvent::RuleApplied {
                        rule_name,
                        after,
                        global_after,
                        is_chained,
                        ..
                    } = event;
                    let rendered_after = global_after.unwrap_or(*after);
                    let rendered_after = format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: rendered_after
                        }
                    );
                    println!(
                        "  event[{mode_name}#{idx}] rule={rule_name:?} chained={is_chained} after={rendered_after:?}"
                    );
                }
            }

            let rendered = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: out
                }
            );
            println!("solve_profile_input[{mode_name}] input={input:?} output={rendered:?}");
            let mut printed_phase_hits = false;
            for &phase in cas_engine::SimplifyPhase::all() {
                let top = simplifier.profiler.top_applied_for_phase(phase, 5);
                if top.is_empty() {
                    continue;
                }
                printed_phase_hits = true;

                let summary = top
                    .into_iter()
                    .map(|(rule, hits)| format!("{rule}={hits}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("  {:?}: {}", phase, summary);
            }
            if !printed_phase_hits {
                println!("  (no rule hits)");
                on_no_rule_hits(&simplifier.context, expr, &simplifier);
            }

            print_rule_probes(&simplifier, options, expr);
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
    let steps_mode = profile_opts.steps_mode;

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

    emit_solve_profile_snapshot(
        "prepass",
        &solve_prepass,
        &inputs,
        |ctx| {
            let mut simplifier = Simplifier::from_profile_with_context(profile.clone(), ctx);
            simplifier.set_steps_mode(steps_mode);
            simplifier
        },
        |_ctx, _expr, _simplifier| {},
    );
    emit_solve_profile_snapshot(
        "strict",
        &tactic_strict,
        &inputs,
        |ctx| {
            let mut simplifier = Simplifier::from_profile_with_context(profile.clone(), ctx);
            simplifier.set_steps_mode(steps_mode);
            simplifier
        },
        |_ctx, _expr, _simplifier| {},
    );
    emit_solve_profile_snapshot(
        "generic",
        &tactic_generic,
        &inputs,
        |ctx| {
            let mut simplifier = Simplifier::from_profile_with_context(profile.clone(), ctx);
            simplifier.set_steps_mode(steps_mode);
            simplifier
        },
        |_ctx, _expr, _simplifier| {},
    );
    emit_solve_profile_snapshot(
        "assume",
        &tactic_assume,
        &inputs,
        |ctx| {
            let mut simplifier = Simplifier::from_profile_with_context(profile.clone(), ctx);
            simplifier.set_steps_mode(steps_mode);
            simplifier
        },
        |_ctx, _expr, _simplifier| {},
    );

    group.bench_function("solve_prepass_batch", |b| {
        b.iter(|| {
            let mut total_len = 0usize;
            for input in &inputs {
                let (ctx, expr) = build_expr(input);
                let mut s = Simplifier::from_profile_with_context(profile.clone(), ctx);
                s.set_steps_mode(steps_mode);
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
                s.set_steps_mode(steps_mode);
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
                s.set_steps_mode(steps_mode);
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
                s.set_steps_mode(steps_mode);
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
    let steps_mode = profile_opts.steps_mode;

    let mut tactic_generic = SimplifyOptions::for_solve_tactic(DomainMode::Generic);
    tactic_generic.collect_steps = false;
    tactic_generic.shared.context_mode = ContextMode::Solve;

    let mut tactic_assume = SimplifyOptions::for_solve_tactic(DomainMode::Assume);
    tactic_assume.collect_steps = false;
    tactic_assume.shared.context_mode = ContextMode::Solve;

    let generic_inputs = [
        "(2*x + 2*y)/(4*x + 4*y)",
        "x/x",
        "exp(ln(x))",
        "(a^x)/a",
        "x^0",
    ];
    let cube_inputs = ["(x^3 - y^3)/(x - y)", "(x^3 + y^3)/(x + y)"];
    let binomial_inputs = [
        "(x^2 + 2*x*y + y^2)/(x + y)^2",
        "(x^2 - 2*x*y + y^2)/(x - y)",
    ];
    let assume_inputs = generic_inputs;

    emit_solve_profile_snapshot(
        "hotspots-generic",
        &tactic_generic,
        &generic_inputs,
        |ctx| {
            let mut simplifier = Simplifier::from_profile_with_context(profile.clone(), ctx);
            simplifier.set_steps_mode(steps_mode);
            simplifier
        },
        |_ctx, expr, simplifier| {
            print_no_hit_bucket_candidates(simplifier, &simplifier.context, expr);
            print_rule_probes(simplifier, &tactic_generic, expr);
        },
    );
    emit_solve_profile_snapshot(
        "hotspots-assume",
        &tactic_assume,
        &assume_inputs,
        |ctx| {
            let mut simplifier = Simplifier::from_profile_with_context(profile.clone(), ctx);
            simplifier.set_steps_mode(steps_mode);
            simplifier
        },
        |_ctx, expr, simplifier| {
            print_no_hit_bucket_candidates(simplifier, &simplifier.context, expr);
            print_rule_probes(simplifier, &tactic_assume, expr);
        },
    );
    emit_solve_profile_snapshot(
        "hotspots-cubes",
        &tactic_generic,
        &cube_inputs,
        |ctx| {
            let mut simplifier = Simplifier::from_profile_with_context(profile.clone(), ctx);
            simplifier.set_steps_mode(steps_mode);
            simplifier
        },
        |_ctx, expr, simplifier| {
            print_no_hit_bucket_candidates(simplifier, &simplifier.context, expr);
            print_rule_probes(simplifier, &tactic_generic, expr);
        },
    );
    emit_solve_profile_snapshot(
        "hotspots-binomials",
        &tactic_generic,
        &binomial_inputs,
        |ctx| {
            let mut simplifier = Simplifier::from_profile_with_context(profile.clone(), ctx);
            simplifier.set_steps_mode(steps_mode);
            simplifier
        },
        |_ctx, expr, simplifier| {
            print_no_hit_bucket_candidates(simplifier, &simplifier.context, expr);
            print_rule_probes(simplifier, &tactic_generic, expr);
        },
    );

    let cases = [
        (
            "generic/scalar_multiple_fraction",
            "(2*x + 2*y)/(4*x + 4*y)",
            tactic_generic.clone(),
        ),
        (
            "generic/difference_of_squares_fraction",
            "(x^2 - y^2)/(x - y)",
            tactic_generic.clone(),
        ),
        (
            "generic/perfect_square_minus_fraction",
            "(x^2 - 2*x*y + y^2)/(x - y)",
            tactic_generic.clone(),
        ),
        (
            "generic/difference_of_cubes_fraction",
            "(x^3 - y^3)/(x - y)",
            tactic_generic.clone(),
        ),
        (
            "generic/sum_of_cubes_fraction",
            "(x^3 + y^3)/(x + y)",
            tactic_generic.clone(),
        ),
        (
            "generic/power_quotient_fraction",
            "x^4/x^2",
            tactic_generic.clone(),
        ),
        (
            "generic/binomial_square_fraction",
            "(x^2 + 2*x*y + y^2)/(x + y)^2",
            tactic_generic.clone(),
        ),
        ("generic/x_over_x", "x/x", tactic_generic.clone()),
        ("generic/exp_ln_x", "exp(ln(x))", tactic_generic.clone()),
        (
            "generic/log_power_base",
            "log(x^2, x^6)",
            tactic_generic.clone(),
        ),
        ("generic/a_pow_x_over_a", "(a^x)/a", tactic_generic.clone()),
        ("generic/x_pow_0", "x^0", tactic_generic.clone()),
        (
            "assume/scalar_multiple_fraction",
            "(2*x + 2*y)/(4*x + 4*y)",
            tactic_assume.clone(),
        ),
        ("assume/x_over_x", "x/x", tactic_assume.clone()),
        ("assume/exp_ln_x", "exp(ln(x))", tactic_assume.clone()),
        (
            "assume/log_power_base",
            "log(x^2, x^6)",
            tactic_assume.clone(),
        ),
        ("assume/a_pow_x_over_a", "(a^x)/a", tactic_assume.clone()),
        ("assume/x_pow_0", "x^0", tactic_assume.clone()),
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
                    s.set_steps_mode(steps_mode);
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

fn bench_solve_eval_hotspots_cached(c: &mut Criterion) {
    let profiles = [
        (
            "eval-strict",
            "strict/scalar_multiple_fraction",
            "(2*x + 2*y)/(4*x + 4*y)",
            EvalOptions {
                branch_mode: BranchMode::Strict,
                complex_mode: ComplexMode::Auto,
                steps_mode: StepsMode::Off,
                shared: cas_engine::SharedSemanticConfig {
                    context_mode: ContextMode::Solve,
                    semantics: EvalConfig {
                        domain_mode: DomainMode::Strict,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "eval-generic",
            "generic/scalar_multiple_fraction",
            "(2*x + 2*y)/(4*x + 4*y)",
            EvalOptions {
                branch_mode: BranchMode::Strict,
                complex_mode: ComplexMode::Auto,
                steps_mode: StepsMode::Off,
                shared: cas_engine::SharedSemanticConfig {
                    context_mode: ContextMode::Solve,
                    semantics: EvalConfig {
                        domain_mode: DomainMode::Generic,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "eval-generic",
            "generic/difference_of_squares_fraction",
            "(x^2 - y^2)/(x - y)",
            EvalOptions {
                branch_mode: BranchMode::Strict,
                complex_mode: ComplexMode::Auto,
                steps_mode: StepsMode::Off,
                shared: cas_engine::SharedSemanticConfig {
                    context_mode: ContextMode::Solve,
                    semantics: EvalConfig {
                        domain_mode: DomainMode::Generic,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "eval-generic",
            "generic/perfect_square_minus_fraction",
            "(x^2 - 2*x*y + y^2)/(x - y)",
            EvalOptions {
                branch_mode: BranchMode::Strict,
                complex_mode: ComplexMode::Auto,
                steps_mode: StepsMode::Off,
                shared: cas_engine::SharedSemanticConfig {
                    context_mode: ContextMode::Solve,
                    semantics: EvalConfig {
                        domain_mode: DomainMode::Generic,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "eval-generic",
            "generic/difference_of_cubes_fraction",
            "(x^3 - y^3)/(x - y)",
            EvalOptions {
                branch_mode: BranchMode::Strict,
                complex_mode: ComplexMode::Auto,
                steps_mode: StepsMode::Off,
                shared: cas_engine::SharedSemanticConfig {
                    context_mode: ContextMode::Solve,
                    semantics: EvalConfig {
                        domain_mode: DomainMode::Generic,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "eval-generic",
            "generic/sum_of_cubes_fraction",
            "(x^3 + y^3)/(x + y)",
            EvalOptions {
                branch_mode: BranchMode::Strict,
                complex_mode: ComplexMode::Auto,
                steps_mode: StepsMode::Off,
                shared: cas_engine::SharedSemanticConfig {
                    context_mode: ContextMode::Solve,
                    semantics: EvalConfig {
                        domain_mode: DomainMode::Generic,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "eval-generic",
            "generic/power_quotient_fraction",
            "x^4/x^2",
            EvalOptions {
                branch_mode: BranchMode::Strict,
                complex_mode: ComplexMode::Auto,
                steps_mode: StepsMode::Off,
                shared: cas_engine::SharedSemanticConfig {
                    context_mode: ContextMode::Solve,
                    semantics: EvalConfig {
                        domain_mode: DomainMode::Generic,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "eval-generic",
            "generic/binomial_square_fraction",
            "(x^2 + 2*x*y + y^2)/(x + y)^2",
            EvalOptions {
                branch_mode: BranchMode::Strict,
                complex_mode: ComplexMode::Auto,
                steps_mode: StepsMode::Off,
                shared: cas_engine::SharedSemanticConfig {
                    context_mode: ContextMode::Solve,
                    semantics: EvalConfig {
                        domain_mode: DomainMode::Generic,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
        (
            "eval-assume",
            "assume/scalar_multiple_fraction",
            "(2*x + 2*y)/(4*x + 4*y)",
            EvalOptions {
                branch_mode: BranchMode::Strict,
                complex_mode: ComplexMode::Auto,
                steps_mode: StepsMode::Off,
                shared: cas_engine::SharedSemanticConfig {
                    context_mode: ContextMode::Solve,
                    semantics: EvalConfig {
                        domain_mode: DomainMode::Assume,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        ),
    ];

    let mut cache = ProfileCache::new();
    let mut group = c.benchmark_group("solve_eval_hotspots_cached");
    common::configure_standard_group(&mut group);

    for (profile_mode_name, name, input, opts) in profiles {
        let profile = cache.get_or_build(&opts);
        let simplify_opts = opts.to_simplify_options();
        emit_solve_profile_snapshot(
            profile_mode_name,
            &simplify_opts,
            &[input],
            |ctx| {
                let mut simplifier = Simplifier::from_profile_with_context(profile.clone(), ctx);
                simplifier.set_steps_mode(opts.steps_mode);
                simplifier
            },
            |_ctx, _expr, _simplifier| {},
        );
        let steps_mode = opts.steps_mode;
        group.bench_function(name, |b| {
            b.iter_batched(
                || build_expr(input),
                |(ctx, expr)| {
                    let mut s = Simplifier::from_profile_with_context(profile.clone(), ctx);
                    s.set_steps_mode(steps_mode);
                    let (out, _steps) = s.simplify_with_options(expr, simplify_opts.clone());
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

fn bench_fraction_gcd_planner_direct(c: &mut Criterion) {
    let cases = [
        (
            "plain/scalar_multiple_fraction",
            "(2*x + 2*y)/(4*x + 4*y)",
            false,
        ),
        (
            "trace/scalar_multiple_fraction",
            "(2*x + 2*y)/(4*x + 4*y)",
            true,
        ),
        ("plain/x_over_x", "x/x", false),
        ("plain/a_pow_x_over_a", "(a^x)/a", false),
        (
            "plain/difference_of_squares_fraction",
            "(x^2 - y^2)/(x - y)",
            false,
        ),
    ];

    let mut group = c.benchmark_group("fraction_gcd_planner_direct");
    common::configure_standard_group(&mut group);

    for (name, input, include_factored_form) in cases {
        group.bench_function(name, |b| {
            b.iter_batched(
                || build_div_expr(input),
                |(mut ctx, expr, num, den)| {
                    let plan =
                        try_plan_fraction_gcd_rewrite(&mut ctx, expr, num, den, include_factored_form);
                    let metric = plan
                        .map(|plan| {
                            let route_score = match plan.route {
                                cas_math::fraction_gcd_plan_support::FractionGcdRoute::StructuralScalarMultiple => 11usize,
                                cas_math::fraction_gcd_plan_support::FractionGcdRoute::Multivar { .. } => 17usize,
                                cas_math::fraction_gcd_plan_support::FractionGcdRoute::Univar => 23usize,
                            };
                            route_score ^ cas_formatter::render_expr(&ctx, plan.forms.result_norm).len()
                        })
                        .unwrap_or(0);
                    black_box(metric);
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_fraction_rule_direct(c: &mut Criterion) {
    let mut group = c.benchmark_group("fraction_rule_direct");
    common::configure_standard_group(&mut group);

    group.bench_function("rational_fn_view/scalar_multiple_fraction", |b| {
        b.iter_batched(
            || build_div_expr("(2*x + 2*y)/(4*x + 4*y)"),
            |(mut ctx, expr, _num, _den)| {
                let metric = RationalFnView::from(&mut ctx, expr)
                    .map(|view| view.num.index() ^ view.den.index())
                    .unwrap_or(0);
                black_box(metric);
            },
            criterion::BatchSize::SmallInput,
        )
    });

    let direct_cases = [
        (
            "apply/generic/scalar_multiple_fraction",
            "(2*x + 2*y)/(4*x + 4*y)",
            DomainMode::Generic,
        ),
        ("apply/generic/x_over_x", "x/x", DomainMode::Generic),
        (
            "apply/generic/a_pow_x_over_a",
            "(a^x)/a",
            DomainMode::Generic,
        ),
        (
            "apply/generic/difference_of_squares_fraction",
            "(x^2 - y^2)/(x - y)",
            DomainMode::Generic,
        ),
        (
            "apply/generic/perfect_square_minus_fraction",
            "(x^2 - 2*x*y + y^2)/(x - y)",
            DomainMode::Generic,
        ),
        (
            "apply/generic/power_quotient_fraction",
            "x^4/x^2",
            DomainMode::Generic,
        ),
        (
            "apply/generic/binomial_square_fraction",
            "(x^2 + 2*x*y + y^2)/(x + y)^2",
            DomainMode::Generic,
        ),
        (
            "apply/generic/difference_of_cubes_fraction",
            "(x^3 - y^3)/(x - y)",
            DomainMode::Generic,
        ),
        (
            "apply/generic/sum_of_cubes_fraction",
            "(x^3 + y^3)/(x + y)",
            DomainMode::Generic,
        ),
    ];
    let rule = SimplifyFractionRule;

    for (name, input, domain_mode) in direct_cases {
        group.bench_function(name, |b| {
            b.iter_batched(
                || build_div_expr(input),
                |(mut ctx, expr, _num, _den)| {
                    let parent_ctx = build_fraction_rule_parent_ctx(&ctx, expr, domain_mode);
                    let metric = rule
                        .apply(&mut ctx, expr, &parent_ctx)
                        .map(|rw| {
                            rw.new_expr.index()
                                ^ rw.required_conditions.len()
                                ^ rw.chained.len()
                                ^ rw.substeps.len()
                        })
                        .unwrap_or(0);
                    black_box(metric);
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    let single_rule_cases = [
        (
            "single_rule_engine/generic/scalar_multiple_fraction",
            "(2*x + 2*y)/(4*x + 4*y)",
            DomainMode::Generic,
        ),
        (
            "single_rule_engine/generic/x_over_x",
            "x/x",
            DomainMode::Generic,
        ),
        (
            "single_rule_engine/generic/a_pow_x_over_a",
            "(a^x)/a",
            DomainMode::Generic,
        ),
        (
            "single_rule_engine/generic/perfect_square_minus_fraction",
            "(x^2 - 2*x*y + y^2)/(x - y)",
            DomainMode::Generic,
        ),
        (
            "single_rule_engine/generic/power_quotient_fraction",
            "x^4/x^2",
            DomainMode::Generic,
        ),
        (
            "single_rule_engine/generic/difference_of_cubes_fraction",
            "(x^3 - y^3)/(x - y)",
            DomainMode::Generic,
        ),
        (
            "single_rule_engine/generic/binomial_square_fraction",
            "(x^2 + 2*x*y + y^2)/(x + y)^2",
            DomainMode::Generic,
        ),
        (
            "single_rule_engine/generic/sum_of_cubes_fraction",
            "(x^3 + y^3)/(x + y)",
            DomainMode::Generic,
        ),
    ];

    for (name, input, domain_mode) in single_rule_cases {
        group.bench_function(name, |b| {
            b.iter_batched(
                || {
                    let mut simplifier = Simplifier::new();
                    simplifier.add_rule(Box::new(SimplifyFractionRule));
                    simplifier.set_steps_mode(StepsMode::Off);
                    let expr = parse(input, &mut simplifier.context).expect("parse failed");
                    let mut opts = SimplifyOptions::for_solve_tactic(domain_mode);
                    opts.collect_steps = false;
                    (simplifier, expr, opts)
                },
                |(mut simplifier, expr, opts)| {
                    let (out, steps) = simplifier.simplify_with_options(expr, opts);
                    let metric = out.index() ^ steps.len();
                    black_box(metric);
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

fn bench_solve_phase_subset_cached(c: &mut Criterion) {
    let inputs = [
        ("scalar_multiple_fraction", "(2*x + 2*y)/(4*x + 4*y)"),
        ("a_pow_x_over_a", "(a^x)/a"),
        ("x_over_x", "x/x"),
        (
            "perfect_square_minus_fraction",
            "(x^2 - 2*x*y + y^2)/(x - y)",
        ),
        ("power_quotient_fraction", "x^4/x^2"),
        ("binomial_square_fraction", "(x^2 + 2*x*y + y^2)/(x + y)^2"),
        ("difference_of_cubes_fraction", "(x^3 - y^3)/(x - y)"),
        ("sum_of_cubes_fraction", "(x^3 + y^3)/(x + y)"),
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

    let mut solve_generic_full = SimplifyOptions::for_solve_tactic(DomainMode::Generic);
    solve_generic_full.collect_steps = false;
    solve_generic_full.shared.context_mode = ContextMode::Solve;

    let mut solve_generic_no_transform = solve_generic_full.clone();
    solve_generic_no_transform.enable_transform = false;

    let mut solve_generic_coreish = solve_generic_no_transform.clone();
    solve_generic_coreish.rationalize.auto_level = cas_engine::AutoRationalizeLevel::Off;

    let option_cases = [
        ("generic/full", solve_generic_full),
        ("generic/no_transform", solve_generic_no_transform),
        ("generic/no_transform_no_rationalize", solve_generic_coreish),
    ];

    let mut group = c.benchmark_group("solve_phase_subset_cached");
    common::configure_standard_group(&mut group);

    for (input_name, input) in inputs {
        for (opts_name, opts) in &option_cases {
            let bench_name = format!("{input_name}/{opts_name}");
            group.bench_function(BenchmarkId::from_parameter(bench_name), |b| {
                b.iter_batched(
                    || build_expr(input),
                    |(ctx, expr)| {
                        let mut simplifier =
                            Simplifier::from_profile_with_context(profile.clone(), ctx);
                        simplifier.set_steps_mode(StepsMode::Off);
                        let (out, _steps, stats) =
                            simplifier.simplify_with_stats(expr, opts.clone());
                        let metric = out.index()
                            ^ stats.core.iters_used
                            ^ stats.transform.iters_used
                            ^ stats.rationalize.iters_used
                            ^ stats.post_cleanup.iters_used;
                        black_box(metric);
                    },
                    criterion::BatchSize::SmallInput,
                )
            });
        }
    }

    group.finish();
}

fn bench_solve_prepass_inherited_steps_cached(c: &mut Criterion) {
    let inputs = [
        "(2*x + 2*y)/(4*x + 4*y)",
        "(x^2 - y^2)/(x - y)",
        "exp(ln(x))",
        "(a^x)/a",
        "x^0",
    ];

    let profile_opts = EvalOptions {
        branch_mode: BranchMode::Strict,
        complex_mode: ComplexMode::Auto,
        steps_mode: StepsMode::On,
        shared: cas_engine::SharedSemanticConfig {
            context_mode: ContextMode::Solve,
            ..Default::default()
        },
        ..Default::default()
    };

    let mut cache = ProfileCache::new();
    let profile = cache.get_or_build(&profile_opts);

    let mut group = c.benchmark_group("solve_prepass_inherited_steps_cached");
    common::configure_standard_group(&mut group);

    group.bench_function("steps_on_batch", |b| {
        b.iter(|| {
            let mut total_len = 0usize;
            for input in &inputs {
                let (ctx, expr) = build_expr(input);
                let mut s = Simplifier::from_profile_with_context(profile.clone(), ctx);
                s.set_steps_mode(StepsMode::On);
                let out = s.simplify_for_solve(expr);
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

fn bench_solver_verification_inherited_steps(c: &mut Criterion) {
    let mut group = c.benchmark_group("solver_verification_inherited_steps");
    common::configure_standard_group(&mut group);

    group.bench_function("quadratic_two_roots_steps_on", |b| {
        b.iter_batched(
            || build_equation_with_solutions("x^2 - 5*x + 6", "0", "x", &["2", "3"]),
            |(mut simplifier, equation, var, solutions)| {
                simplifier.set_steps_mode(StepsMode::On);
                let result = verify_solution_set(&mut simplifier, &equation, &var, &solutions);
                let verified = matches!(result.summary, VerifySummary::AllVerified);
                black_box(verified);
            },
            criterion::BatchSize::SmallInput,
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_profile_build,
    bench_simplify_cached_vs_uncached,
    bench_solve_modes_cached,
    bench_solve_hotspots_cached,
    bench_solve_eval_hotspots_cached,
    bench_fraction_gcd_planner_direct,
    bench_fraction_rule_direct,
    bench_solve_phase_subset_cached,
    bench_solve_prepass_inherited_steps_cached,
    bench_solver_verification_inherited_steps
);
criterion_main!(benches);
