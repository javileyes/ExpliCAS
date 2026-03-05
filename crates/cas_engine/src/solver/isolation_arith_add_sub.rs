use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{
    context_render_expr, medium_step, simplifier_context, simplifier_context_mut,
    simplifier_prove_nonzero_status, simplifier_simplify_expr, SolveCtx, SolveStep, SolverOptions,
};
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_arithmetic::{
    execute_add_isolation_pipeline_with_default_factored_linear_collect_and_unified_step_mapper_with_state,
    execute_sub_isolation_pipeline_with_default_plan_and_unified_step_mapper_with_state,
};

use super::isolation::isolate;

/// Handle isolation for `Add(l, r)`: `(A + B) = RHS`
#[allow(clippy::too_many_arguments)]
pub(super) fn isolate_add(
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
    execute_add_isolation_pipeline_with_default_factored_linear_collect_and_unified_step_mapper_with_state(
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
        context_render_expr,
        simplifier_simplify_expr,
        simplifier_prove_nonzero_status,
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

/// Handle isolation for `Sub(l, r)`: `(A - B) = RHS`
#[allow(clippy::too_many_arguments)]
pub(super) fn isolate_sub(
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
    execute_sub_isolation_pipeline_with_default_plan_and_unified_step_mapper_with_state(
        simplifier,
        l,
        r,
        rhs,
        op,
        var,
        include_item,
        steps,
        simplifier_context,
        simplifier_context_mut,
        context_render_expr,
        simplifier_simplify_expr,
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
