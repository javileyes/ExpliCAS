mod common;

use std::hint::black_box;

use cas_api_models::{
    EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
    EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalSessionRunConfig,
    EvalStepsMode, EvalValueDomain, StepWire,
};
use cas_session::eval::{
    evaluate_eval_command_pretty_with_session, evaluate_eval_command_with_session,
    evaluate_eval_text_command_with_session,
};
use cas_session::repl::{build_repl_core_with_config, CasConfig};
use cas_session::SessionState;
use cas_solver::runtime::Engine;
use cas_solver::session_api::eval::evaluate_eval_text_simplify_with_session;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use tempfile::tempdir;

fn eval_config<'a>(expr: &'a str, auto_store: bool) -> EvalSessionRunConfig<'a> {
    EvalSessionRunConfig {
        expr,
        auto_store,
        max_chars: 2000,
        steps_mode: EvalStepsMode::Off,
        budget_preset: EvalBudgetPreset::Standard,
        strict: false,
        domain: EvalDomainMode::Generic,
        context_mode: EvalContextMode::Auto,
        branch_mode: EvalBranchMode::Strict,
        expand_policy: EvalExpandPolicy::Off,
        complex_mode: EvalComplexMode::Auto,
        const_fold: EvalConstFoldMode::Off,
        value_domain: EvalValueDomain::Real,
        complex_branch: cas_api_models::EvalBranchMode::Principal,
        inv_trig: EvalInvTrigPolicy::Strict,
        assume_scope: EvalAssumeScope::Real,
    }
}

fn no_steps(
    _steps: &[cas_engine::Step],
    _events: &[cas_solver_core::engine_events::EngineEvent],
    _context: &cas_ast::Context,
    _steps_mode: &str,
) -> Vec<StepWire> {
    Vec::new()
}

fn bench_frontend_session(c: &mut Criterion) {
    let eval_inputs = [
        ("light/x_plus_1", "x + 1"),
        ("gcd/scalar_multiple_fraction", "(2*x + 2*y)/(4*x + 4*y)"),
        ("trig/pythagorean_chain", "sin(2*x + 1)^2 + cos(1 + 2*x)^2"),
    ];

    let mut group = c.benchmark_group("frontend_session");
    common::configure_standard_group(&mut group);

    group.bench_function("repl/build/default", |b| {
        let config = CasConfig::default();
        b.iter(|| black_box(build_repl_core_with_config(&config)));
    });

    for (name, input) in eval_inputs {
        group.bench_with_input(
            BenchmarkId::new("eval_runtime/in_memory", name),
            &input,
            |b, input| {
                b.iter_batched(
                    || {
                        (
                            Engine::new(),
                            SessionState::new(),
                            eval_config(input, false),
                        )
                    },
                    |(mut engine, mut state, config)| {
                        black_box(
                            cas_solver::session_api::eval::evaluate_eval_with_session(
                                &mut engine,
                                &mut state,
                                config,
                                no_steps,
                            )
                            .expect("in-memory eval failed"),
                        )
                    },
                    BatchSize::SmallInput,
                )
            },
        );

        group.bench_with_input(
            BenchmarkId::new("eval_wire/no_session", name),
            &input,
            |b, input| {
                b.iter(|| {
                    black_box(evaluate_eval_command_with_session(
                        None,
                        eval_config(input, false),
                        no_steps,
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("eval_pretty/no_session", name),
            &input,
            |b, input| {
                b.iter(|| {
                    black_box(evaluate_eval_command_pretty_with_session(
                        None,
                        eval_config(input, false),
                        no_steps,
                    ))
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("eval_text/no_session", name),
            &input,
            |b, input| {
                b.iter(|| {
                    black_box(evaluate_eval_text_command_with_session(
                        None, "generic", input, false,
                    ))
                })
            },
        );
    }

    group.bench_function("eval_wire/persisted/light/x_plus_1", |b| {
        b.iter_batched(
            || {
                let tmp = tempdir().expect("tempdir failed");
                let session_path = tmp.path().join("session.toml");
                let _ = evaluate_eval_command_with_session(
                    Some(&session_path),
                    eval_config("x + 1", true),
                    no_steps,
                );
                (tmp, session_path)
            },
            |(_tmp, session_path)| {
                black_box(evaluate_eval_command_with_session(
                    Some(&session_path),
                    eval_config("x + 1", true),
                    no_steps,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("eval_wire/persisted_no_store/light/x_plus_1", |b| {
        b.iter_batched(
            || {
                let tmp = tempdir().expect("tempdir failed");
                let session_path = tmp.path().join("session.toml");
                let _ = evaluate_eval_command_with_session(
                    Some(&session_path),
                    eval_config("x + 1", true),
                    no_steps,
                );
                (tmp, session_path)
            },
            |(_tmp, session_path)| {
                black_box(evaluate_eval_command_with_session(
                    Some(&session_path),
                    eval_config("x + 1", false),
                    no_steps,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("eval_wire/persisted_no_store/cache_hit/ref_1", |b| {
        b.iter_batched(
            || {
                let tmp = tempdir().expect("tempdir failed");
                let session_path = tmp.path().join("session.toml");
                let _ = evaluate_eval_command_with_session(
                    Some(&session_path),
                    eval_config("x + 1", true),
                    no_steps,
                );
                (tmp, session_path)
            },
            |(_tmp, session_path)| {
                black_box(evaluate_eval_command_with_session(
                    Some(&session_path),
                    eval_config("#1", false),
                    no_steps,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("eval_text/persisted/light/x_plus_1", |b| {
        b.iter_batched(
            || {
                let tmp = tempdir().expect("tempdir failed");
                let session_path = tmp.path().join("session.toml");
                let _ = evaluate_eval_text_command_with_session(
                    Some(&session_path),
                    "generic",
                    "x + 1",
                    true,
                );
                (tmp, session_path)
            },
            |(_tmp, session_path)| {
                black_box(evaluate_eval_text_command_with_session(
                    Some(&session_path),
                    "generic",
                    "x + 1",
                    true,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("eval_text/persisted_no_store/light/x_plus_1", |b| {
        b.iter_batched(
            || {
                let tmp = tempdir().expect("tempdir failed");
                let session_path = tmp.path().join("session.toml");
                let _ = evaluate_eval_text_command_with_session(
                    Some(&session_path),
                    "generic",
                    "x + 1",
                    true,
                );
                (tmp, session_path)
            },
            |(_tmp, session_path)| {
                black_box(evaluate_eval_text_command_with_session(
                    Some(&session_path),
                    "generic",
                    "x + 1",
                    false,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("eval_text/persisted_no_store/cache_hit/ref_1", |b| {
        b.iter_batched(
            || {
                let tmp = tempdir().expect("tempdir failed");
                let session_path = tmp.path().join("session.toml");
                let _ = evaluate_eval_text_command_with_session(
                    Some(&session_path),
                    "generic",
                    "x + 1",
                    true,
                );
                (tmp, session_path)
            },
            |(_tmp, session_path)| {
                black_box(evaluate_eval_text_command_with_session(
                    Some(&session_path),
                    "generic",
                    "#1",
                    false,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("session_phase/load_or_new/persisted/cache_hit_seed", |b| {
        b.iter_batched(
            || {
                let tmp = tempdir().expect("tempdir failed");
                let session_path = tmp.path().join("session.toml");
                let _ = evaluate_eval_command_with_session(
                    Some(&session_path),
                    eval_config("x + 1", true),
                    no_steps,
                );
                (tmp, session_path)
            },
            |(_tmp, session_path)| {
                let key = cas_session::cache::SimplifyCacheKey::from_domain_flag("generic");
                black_box(cas_session::SessionState::load_compatible_snapshot(
                    &session_path,
                    &key,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("session_phase/engine_with_context/cache_hit_seed", |b| {
        b.iter_batched(
            || {
                let tmp = tempdir().expect("tempdir failed");
                let session_path = tmp.path().join("session.toml");
                let _ = evaluate_eval_command_with_session(
                    Some(&session_path),
                    eval_config("x + 1", true),
                    no_steps,
                );
                let key = cas_session::cache::SimplifyCacheKey::from_domain_flag("generic");
                let (context, _state) =
                    cas_session::SessionState::load_compatible_snapshot(&session_path, &key)
                        .expect("load compatible snapshot")
                        .expect("compatible snapshot");
                ((tmp, session_path), context)
            },
            |(_seed, context)| black_box(Engine::with_context(context)),
            BatchSize::SmallInput,
        )
    });

    group.bench_function("session_phase/run_loaded/cache_hit/ref_1", |b| {
        b.iter_batched(
            || {
                let tmp = tempdir().expect("tempdir failed");
                let session_path = tmp.path().join("session.toml");
                let _ = evaluate_eval_command_with_session(
                    Some(&session_path),
                    eval_config("x + 1", true),
                    no_steps,
                );
                let key = cas_session::cache::SimplifyCacheKey::from_domain_flag("generic");
                let (context, state) =
                    cas_session::SessionState::load_compatible_snapshot(&session_path, &key)
                        .expect("load compatible snapshot")
                        .expect("compatible snapshot");
                ((tmp, session_path), Engine::with_context(context), state)
            },
            |(_seed, mut engine, mut state)| {
                black_box(cas_solver::session_api::eval::evaluate_eval_with_session(
                    &mut engine,
                    &mut state,
                    eval_config("#1", false),
                    no_steps,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("session_phase_text/run_loaded/cache_hit/ref_1", |b| {
        b.iter_batched(
            || {
                let tmp = tempdir().expect("tempdir failed");
                let session_path = tmp.path().join("session.toml");
                let _ = evaluate_eval_text_command_with_session(
                    Some(&session_path),
                    "generic",
                    "x + 1",
                    true,
                );
                let key = cas_session::cache::SimplifyCacheKey::from_domain_flag("generic");
                let (context, state) =
                    cas_session::SessionState::load_compatible_snapshot(&session_path, &key)
                        .expect("load compatible snapshot")
                        .expect("compatible snapshot");
                ((tmp, session_path), Engine::with_context(context), state)
            },
            |(_seed, mut engine, mut state)| {
                black_box(evaluate_eval_text_simplify_with_session(
                    &mut engine,
                    &mut state,
                    "#1",
                    false,
                ))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("session_phase/save_snapshot/dirty_seed", |b| {
        b.iter_batched(
            || {
                let tmp = tempdir().expect("tempdir failed");
                let session_path = tmp.path().join("session.toml");
                let mut engine = Engine::new();
                let mut state = SessionState::new();
                let _ = cas_solver::session_api::eval::evaluate_eval_with_session(
                    &mut engine,
                    &mut state,
                    eval_config("x + 1", true),
                    no_steps,
                )
                .expect("seed eval failed");
                let key = cas_session::cache::SimplifyCacheKey::from_domain_flag("generic");
                (tmp, session_path, key, engine, state)
            },
            |(_tmp, session_path, key, engine, state)| {
                black_box(state.save_snapshot(&engine.simplifier.context, &session_path, key))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("session_phase/save_snapshot/overwrite_dirty_seed", |b| {
        b.iter_batched(
            || {
                let tmp = tempdir().expect("tempdir failed");
                let session_path = tmp.path().join("session.toml");
                let mut engine = Engine::new();
                let mut state = SessionState::new();
                let _ = cas_solver::session_api::eval::evaluate_eval_with_session(
                    &mut engine,
                    &mut state,
                    eval_config("x + 1", true),
                    no_steps,
                )
                .expect("seed eval failed");
                let key = cas_session::cache::SimplifyCacheKey::from_domain_flag("generic");
                state
                    .save_snapshot(&engine.simplifier.context, &session_path, key.clone())
                    .expect("seed snapshot save failed");
                (tmp, session_path, key, engine, state)
            },
            |(_tmp, session_path, key, engine, state)| {
                black_box(state.save_snapshot(&engine.simplifier.context, &session_path, key))
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function(
        "session_phase/save_snapshot/overwrite_after_mutation_seed",
        |b| {
            b.iter_batched(
                || {
                    let tmp = tempdir().expect("tempdir failed");
                    let session_path = tmp.path().join("session.toml");
                    let mut engine = Engine::new();
                    let mut state = SessionState::new();
                    let _ = cas_solver::session_api::eval::evaluate_eval_with_session(
                        &mut engine,
                        &mut state,
                        eval_config("x + 1", true),
                        no_steps,
                    )
                    .expect("seed eval failed");
                    let key = cas_session::cache::SimplifyCacheKey::from_domain_flag("generic");
                    state
                        .save_snapshot(&engine.simplifier.context, &session_path, key.clone())
                        .expect("seed snapshot save failed");
                    let _ = cas_solver::session_api::eval::evaluate_eval_with_session(
                        &mut engine,
                        &mut state,
                        eval_config("x + 2", true),
                        no_steps,
                    )
                    .expect("mutation eval failed");
                    (tmp, session_path, key, engine, state)
                },
                |(_tmp, session_path, key, engine, state)| {
                    black_box(state.save_snapshot(&engine.simplifier.context, &session_path, key))
                },
                BatchSize::SmallInput,
            )
        },
    );

    group.finish();
}

criterion_group!(benches, bench_frontend_session);
criterion_main!(benches);
