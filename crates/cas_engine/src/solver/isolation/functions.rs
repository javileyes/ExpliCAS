use super::isolate;
use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{medium_step, render_expr as solver_render_expr, SolveStep, SolverOptions};
use cas_ast::symbol::SymbolId;
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_functions::{
    execute_abs_function_isolation_with_default_plan_and_finalizer_with_state,
    execute_function_isolation_route_for_var_with_state,
    execute_log_function_isolation_with_default_plan_with_state,
    execute_unary_function_isolation_with_default_plan_with_state,
};
use cas_solver_core::solve_outcome::AbsSplitExecutionItem;

/// Handle isolation for `Function(fn_id, args)`: abs, log, ln, exp, sqrt, trig
#[allow(clippy::too_many_arguments)]
pub(super) fn isolate_function(
    fn_id: SymbolId,
    args: Vec<ExprId>,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    execute_function_isolation_route_for_var_with_state(
        simplifier,
        |simplifier| &simplifier.context,
        fn_id,
        &args,
        var,
        |simplifier, arg| {
            isolate_abs(
                arg,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                steps.clone(),
                ctx,
            )
        },
        |simplifier, base, arg| {
            isolate_log(
                base,
                arg,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                steps.clone(),
                ctx,
            )
        },
        |simplifier, arg| {
            isolate_unary_function(
                fn_id,
                arg,
                rhs,
                op.clone(),
                var,
                simplifier,
                opts,
                steps.clone(),
                ctx,
            )
        },
        |_simplifier| CasError::VariableNotFound(var.to_string()),
        |simplifier| {
            CasError::IsolationError(
                var.to_string(),
                format!(
                    "Cannot invert function '{}' with {} arguments",
                    simplifier.context.sym_name(fn_id),
                    args.len()
                ),
            )
        },
    )
}

/// Handle `|A| = RHS` (absolute value isolation)
///
/// Soundness invariant: `|A| = B` requires `B ≥ 0` (absolute values are
/// non-negative). When `B` is a symbolic expression containing the solve
/// variable, we attach `NonNegative(rhs)` as a condition guard rather than
/// attempting to solve a guard inequality recursively.
#[allow(clippy::too_many_arguments)]
fn isolate_abs(
    arg: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_items = simplifier.collect_steps();
    execute_abs_function_isolation_with_default_plan_and_finalizer_with_state(
        simplifier,
        arg,
        rhs,
        op,
        var,
        include_items,
        steps,
        |simplifier, expr| solver_render_expr(&simplifier.context, expr),
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
        |item: AbsSplitExecutionItem| medium_step(item.description().to_string(), item.equation),
        |simplifier| &mut simplifier.context,
        |simplifier| &simplifier.context,
    )
}

/// Handle `log(base, arg) = RHS`
#[allow(clippy::too_many_arguments)]
fn isolate_log(
    base: ExprId,
    arg: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_item = simplifier.collect_steps();
    execute_log_function_isolation_with_default_plan_with_state(
        simplifier,
        base,
        arg,
        rhs,
        var,
        op,
        include_item,
        steps,
        |simplifier| &mut simplifier.context,
        solver_render_expr,
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
        |item| medium_step(item.description().to_string(), item.equation),
        |_simplifier| {
            CasError::IsolationError(
                var.to_string(),
                "Cannot isolate from log function".to_string(),
            )
        },
    )
}

/// Handle single-argument functions: ln, exp, sqrt, sin, cos, tan
#[allow(clippy::too_many_arguments)]
fn isolate_unary_function(
    fn_id: SymbolId,
    arg: ExprId,
    rhs: ExprId,
    op: RelOp,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let fn_name = simplifier.context.sym_name(fn_id).to_string();
    let include_items = simplifier.collect_steps();
    execute_unary_function_isolation_with_default_plan_with_state(
        simplifier,
        &fn_name,
        arg,
        rhs,
        op,
        true,
        include_items,
        steps,
        |simplifier| &mut simplifier.context,
        |simplifier, rhs_expr| {
            let (simplified_rhs, sim_steps) = simplifier.simplify(rhs_expr);
            let entries = sim_steps
                .into_iter()
                .map(|step| (step.description, step.after))
                .collect::<Vec<_>>();
            (simplified_rhs, entries)
        },
        |simplifier, lhs_expr, rhs_expr, inner_op| {
            isolate(lhs_expr, rhs_expr, inner_op, var, simplifier, opts, ctx)
        },
        |item: cas_solver_core::function_inverse::UnaryInverseSolveExecutionItem| {
            medium_step(item.description().to_string(), item.equation)
        },
        |_simplifier| CasError::UnknownFunction(fn_name.clone()),
    )
}
