use crate::engine::Simplifier;
use crate::error::CasError;
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::isolation_strategy::{
    execute_collect_terms_strategy_with_default_kernel_and_unified_step_mapper_with_state,
    execute_isolation_strategy_with_default_routing_and_unified_step_mapper_with_state,
    execute_unwrap_strategy_with_default_route_and_residual_hint_and_unified_step_mapper_with_state,
};

use super::isolation::isolate;
use super::solve_entrypoints::solve_with_ctx_and_options;
use super::{
    context_render_expr, medium_step, simplifier_context, simplifier_context_mut,
    simplifier_render_expr, simplifier_simplify_expr, SolveCtx, SolveStep, SolverOptions,
};

pub(super) fn apply_isolation_strategy(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let include_item = simplifier.collect_steps();
    execute_isolation_strategy_with_default_routing_and_unified_step_mapper_with_state(
        simplifier,
        equation,
        var,
        include_item,
        simplifier_context,
        |simplifier, next_eq, solve_var| {
            isolate(
                next_eq.lhs,
                next_eq.rhs,
                next_eq.op.clone(),
                solve_var,
                simplifier,
                *opts,
                ctx,
            )
        },
        medium_step,
        |missing_var| CasError::VariableNotFound(missing_var.to_string()),
    )
}

pub(super) fn apply_collect_terms_strategy(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let include_item = simplifier.collect_steps();
    execute_collect_terms_strategy_with_default_kernel_and_unified_step_mapper_with_state(
        simplifier,
        equation,
        var,
        include_item,
        simplifier_context_mut,
        simplifier_simplify_expr,
        simplifier_render_expr,
        |simplifier, next_eq, solve_var| {
            solve_with_ctx_and_options(next_eq, solve_var, simplifier, *opts, ctx)
        },
        medium_step,
    )
}

pub(super) fn apply_unwrap_strategy(
    equation: &Equation,
    var: &str,
    simplifier: &mut Simplifier,
    opts: &SolverOptions,
    ctx: &SolveCtx,
) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
    let mode = opts.core_domain_mode();
    let wildcard_scope = opts.wildcard_scope();

    let include_item = simplifier.collect_steps();
    execute_unwrap_strategy_with_default_route_and_residual_hint_and_unified_step_mapper_with_state(
        simplifier,
        equation,
        var,
        include_item,
        simplifier_context_mut,
        mode,
        wildcard_scope,
        |core_ctx, base, other_side| {
            cas_solver_core::log_domain::classify_log_solve_with_env_and_tri_prover(
                core_ctx,
                base,
                other_side,
                opts.value_domain == crate::ValueDomain::RealOnly,
                opts.core_domain_mode(),
                &ctx.domain_env,
                |inner_ctx, expr| {
                    crate::helpers::prove_positive_core(inner_ctx, expr, opts.value_domain)
                },
            )
        },
        context_render_expr,
        |simplifier, record| {
            ctx.note_assumption(
                cas_solver_core::assumption_model::assumption_event_from_log_assumption(
                    &simplifier.context,
                    record.assumption,
                    record.base,
                    record.other_side,
                ),
            );
        },
        |simplifier, next_eq, solve_var| {
            solve_with_ctx_and_options(next_eq, solve_var, simplifier, *opts, ctx)
        },
        medium_step,
    )
}
