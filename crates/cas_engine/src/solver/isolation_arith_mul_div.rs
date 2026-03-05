use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_entrypoints::solve_with_ctx_and_options;
use crate::solver::{
    context_render_expr, medium_step, simplifier_context, simplifier_context_mut,
    simplifier_is_known_negative, simplifier_prove_nonzero_status, simplifier_simplify_expr,
    SolveCtx, SolveStep, SolverOptions,
};
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_arithmetic::{
    execute_div_isolation_pipeline_with_default_reciprocal_fallback_and_unified_step_mapper_with_state,
    execute_mul_isolation_pipeline_with_default_additive_linear_collect_and_unified_step_mapper_with_state,
};

use super::isolation::isolate;

/// Handle isolation for `Mul(l, r)`: `A * B = RHS`
#[allow(clippy::too_many_arguments)]
pub(super) fn isolate_mul(
    lhs: ExprId,
    l: ExprId,
    r: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_item = simplifier.collect_steps();
    execute_mul_isolation_pipeline_with_default_additive_linear_collect_and_unified_step_mapper_with_state(
        simplifier,
        lhs,
        l,
        r,
        rhs,
        op,
        var,
        include_item,
        steps,
        simplifier_context,
        simplifier_context_mut,
        |simplifier, equation| solve_with_ctx_and_options(equation, var, simplifier, opts, ctx),
        simplifier_simplify_expr,
        simplifier_prove_nonzero_status,
        simplifier_is_known_negative,
        context_render_expr,
        |simplifier, equation| {
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op,
                var,
                simplifier,
                opts,
                ctx,
            )
        },
        medium_step,
    )
}

/// Handle isolation for `Div(l, r)`: `A / B = RHS`
#[allow(clippy::too_many_arguments)]
pub(super) fn isolate_div(
    lhs: ExprId,
    l: ExprId,
    r: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_numerator_items = simplifier.collect_steps();
    let include_denominator_items = simplifier.collect_steps();
    execute_div_isolation_pipeline_with_default_reciprocal_fallback_and_unified_step_mapper_with_state(
        simplifier,
        lhs,
        l,
        r,
        rhs,
        op,
        var,
        include_numerator_items,
        include_denominator_items,
        steps,
        simplifier_context,
        simplifier_context_mut,
        simplifier_is_known_negative,
        context_render_expr,
        simplifier_simplify_expr,
        |simplifier, equation| {
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )
        },
        simplifier_prove_nonzero_status,
        medium_step,
    )
}
