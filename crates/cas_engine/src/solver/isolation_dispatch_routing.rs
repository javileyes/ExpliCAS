use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{
    context_render_expr, medium_step, simplifier_collect_steps, simplifier_context,
    simplifier_context_mut, simplifier_prove_nonzero_status, simplifier_simplify_expr, SolveCtx,
    SolveStep, SolverOptions,
};
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_dispatch::execute_isolation_dispatch_with_default_isolated_and_negated_entries_and_default_linear_collect_kernels_for_var_and_unified_step_mapper_with_state;

use super::isolation::isolate;
use super::isolation_arith::{isolate_add, isolate_div, isolate_mul, isolate_sub};
use super::isolation_function::isolate_function;
use super::isolation_pow::isolate_pow;

#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_isolation(
    lhs: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    execute_isolation_dispatch_with_default_isolated_and_negated_entries_and_default_linear_collect_kernels_for_var_and_unified_step_mapper_with_state(
        simplifier,
        lhs,
        rhs,
        op.clone(),
        var,
        simplifier_context,
        simplifier_context_mut,
        simplifier_simplify_expr,
        simplifier_collect_steps,
        simplifier_simplify_expr,
        simplifier_prove_nonzero_status,
        context_render_expr,
        |simplifier, left, right| {
            isolate_add(
                lhs,
                left,
                right,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, left, right| {
            isolate_sub(
                left,
                right,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, left, right| {
            isolate_mul(
                lhs,
                left,
                right,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, left, right| {
            isolate_div(
                lhs,
                left,
                right,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, base, exponent| {
            isolate_pow(
                lhs,
                base,
                exponent,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        |simplifier, fn_id, args| {
            isolate_function(
                fn_id,
                args,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                Vec::new(),
                ctx,
            )
        },
        simplifier_collect_steps,
        |simplifier, equation, solve_var| {
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op,
                solve_var,
                simplifier,
                opts,
                ctx,
            )
        },
        medium_step,
        |_simplifier, lhs_expr| {
            CasError::IsolationError(
                var.to_string(),
                format!("Cannot isolate from {:?}", lhs_expr),
            )
        },
    )
}
