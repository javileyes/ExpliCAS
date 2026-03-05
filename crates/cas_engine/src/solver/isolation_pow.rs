use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{
    context_render_expr, medium_step, simplifier_collect_steps, simplifier_context,
    simplifier_context_mut, simplifier_simplify_expr, SolveCtx, SolveStep, SolverOptions,
};
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_power::execute_pow_isolation_with_kernel_config_and_unified_step_mapper_for_var_with_state;

use super::isolation::isolate;
use super::isolation_entrypoints::{isolate_equation, isolate_equation_solutions};

type PowIsolationConfig =
    cas_solver_core::strategy_options::PowIsolationRuntimeConfig<crate::SimplifyOptions>;

/// Handle isolation for `Pow(b, e)`: `B^E = RHS`
#[allow(clippy::too_many_arguments)]
pub(super) fn isolate_pow(
    lhs: ExprId,
    b: ExprId,
    e: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let config = build_pow_isolation_config(simplifier, opts);
    execute_isolation_pow(
        lhs, b, e, rhs, op, var, simplifier, opts, steps, ctx, config,
    )
}

fn build_pow_isolation_config(
    simplifier: &mut Simplifier,
    opts: SolverOptions,
) -> PowIsolationConfig {
    cas_solver_core::strategy_options::pow_runtime_config_with(
        opts.core_domain_mode(),
        opts.wildcard_scope(),
        opts.value_domain == crate::ValueDomain::RealOnly,
        opts.budget,
        || simplifier.collect_steps(),
        || crate::SimplifyOptions::for_solve_tactic(opts.domain_mode),
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_isolation_pow(
    lhs: ExprId,
    b: ExprId,
    e: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &SolveCtx,
    config: PowIsolationConfig,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    execute_pow_isolation_with_kernel_config_and_unified_step_mapper_for_var_with_state(
        simplifier,
        lhs,
        b,
        e,
        rhs,
        op,
        var,
        config.kernel,
        steps,
        simplifier_context,
        simplifier_context_mut,
        context_render_expr,
        |simplifier, iso_lhs, iso_rhs, iso_op| {
            isolate(iso_lhs, iso_rhs, iso_op, var, simplifier, opts, ctx)
        },
        simplifier_simplify_expr,
        |_simplifier| crate::clear_blocked_hints(),
        |simplifier, expr| {
            simplifier
                .simplify_with_options(expr, config.tactic_opts.clone())
                .0
        },
        simplifier_collect_steps,
        |simplifier, tactic_base, tactic_rhs| {
            cas_solver_core::log_domain::classify_log_solve_with_env_and_tri_prover(
                &simplifier.context,
                tactic_base,
                tactic_rhs,
                opts.value_domain == crate::ValueDomain::RealOnly,
                opts.core_domain_mode(),
                &ctx.domain_env,
                |core_ctx, expr| {
                    crate::helpers::prove_positive_core(core_ctx, expr, opts.value_domain)
                },
            )
        },
        |simplifier, shortcut_rhs, shortcut_op| {
            isolate(e, shortcut_rhs, shortcut_op, var, simplifier, opts, ctx)
        },
        medium_step,
        |local_var, message| CasError::IsolationError(local_var.to_string(), message.to_string()),
        |core_ctx, assumption| {
            ctx.note_assumption(
                cas_solver_core::assumption_model::assumption_event_from_log_assumption(
                    core_ctx, assumption, b, rhs,
                ),
            );
        },
        |simplifier, equation| isolate_equation_solutions(equation, var, simplifier, opts, ctx),
        |core_ctx, hint| {
            crate::register_blocked_hint(cas_solver_core::assumption_model::map_log_blocked_hint(
                core_ctx, hint,
            ));
        },
        |message| CasError::UnsupportedInRealDomain(message.to_string()),
        |simplifier, equation| isolate_equation(equation, var, simplifier, opts, ctx),
    )
}
