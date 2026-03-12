mod common;

use std::hint::black_box;

use cas_api_models::{
    EvalAssumeScope, EvalBranchMode, EvalBudgetPreset, EvalComplexMode, EvalConstFoldMode,
    EvalContextMode, EvalDomainMode, EvalExpandPolicy, EvalInvTrigPolicy, EvalSessionRunConfig,
    EvalStepsMode, EvalValueDomain, StepWire,
};
use cas_session::cache::SimplifyCacheKey;
use cas_session::eval_api::evaluate_eval_command_with_session;
use cas_session::state_api::SessionState;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use tempfile::tempdir;

fn eval_config<'a>(expr: &'a str, domain: EvalDomainMode) -> EvalSessionRunConfig<'a> {
    EvalSessionRunConfig {
        expr,
        auto_store: true,
        max_chars: 2000,
        steps_mode: EvalStepsMode::Off,
        budget_preset: EvalBudgetPreset::Standard,
        strict: false,
        domain,
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

fn seed_snapshot(path: &std::path::Path, domain: EvalDomainMode, count: usize) {
    for i in 0..count {
        let expr = format!("(2*x{i} + 2*y{i})/(4*x{i} + 4*y{i})");
        let _ =
            evaluate_eval_command_with_session(Some(path), eval_config(&expr, domain), no_steps);
    }
}

fn bench_snapshot_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("snapshot_load");
    common::configure_standard_group(&mut group);

    for (name, count) in [("medium", 8usize), ("large", 32usize)] {
        group.bench_function(BenchmarkId::new("compatible", name), |b| {
            b.iter_batched(
                || {
                    let tmp = tempdir().expect("tempdir failed");
                    let path = tmp.path().join("session.bin");
                    seed_snapshot(&path, EvalDomainMode::Generic, count);
                    (tmp, path)
                },
                |(_tmp, path)| {
                    black_box(
                        SessionState::load_compatible_snapshot(
                            &path,
                            &SimplifyCacheKey::from_domain_flag("generic"),
                        )
                        .expect("load compatible"),
                    )
                },
                criterion::BatchSize::SmallInput,
            )
        });

        group.bench_function(BenchmarkId::new("incompatible", name), |b| {
            b.iter_batched(
                || {
                    let tmp = tempdir().expect("tempdir failed");
                    let path = tmp.path().join("session.bin");
                    seed_snapshot(&path, EvalDomainMode::Generic, count);
                    (tmp, path)
                },
                |(_tmp, path)| {
                    black_box(
                        SessionState::load_compatible_snapshot(
                            &path,
                            &SimplifyCacheKey::from_domain_flag("strict"),
                        )
                        .expect("load incompatible"),
                    )
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

criterion_group!(benches, bench_snapshot_load);
criterion_main!(benches);
