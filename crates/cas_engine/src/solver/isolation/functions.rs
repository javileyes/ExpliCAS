use super::isolate;
use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{medium_step, render_expr as solver_render_expr, SolveStep, SolverOptions};
use cas_ast::symbol::SymbolId;
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::function_inverse::{
    derive_function_isolation_route, plan_unary_inverse_isolation_step, FunctionIsolationRoute,
    FunctionIsolationRouteError,
};
use cas_solver_core::log_isolation::plan_log_isolation_step_with;
use cas_solver_core::solve_outcome::{
    execute_abs_isolation_plan_with_rhs_sign_pipeline_with_optional_items_and_solver,
    execute_log_isolation_result_pipeline_with_plan_and_error_and_merge_with_existing_steps,
    execute_unary_inverse_result_pipeline_or_else_with_and_merge_with_existing_steps_with,
    finalize_abs_split_solution_set_for_rhs, plan_abs_isolation_with_rhs_sign,
};
use std::cell::RefCell;

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
    match derive_function_isolation_route(&simplifier.context, fn_id, &args, var) {
        Ok(FunctionIsolationRoute::AbsUnary { arg }) => {
            isolate_abs(arg, rhs, op, var, simplifier, opts, steps, ctx)
        }
        Ok(FunctionIsolationRoute::LogBinary { base, arg }) => {
            isolate_log(base, arg, rhs, op, var, simplifier, opts, steps, ctx)
        }
        Ok(FunctionIsolationRoute::UnaryInvertible { arg }) => {
            isolate_unary_function(fn_id, arg, rhs, op, var, simplifier, opts, steps, ctx)
        }
        Err(FunctionIsolationRouteError::VariableNotFoundInUnaryArg) => {
            Err(CasError::VariableNotFound(var.to_string()))
        }
        Err(FunctionIsolationRouteError::UnsupportedArity) => Err(CasError::IsolationError(
            var.to_string(),
            format!(
                "Cannot invert function '{}' with {} arguments",
                simplifier.context.sym_name(fn_id),
                args.len()
            ),
        )),
    }
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
    let finalize_op = op.clone();
    let simplifier_ref = RefCell::new(simplifier);
    execute_abs_isolation_plan_with_rhs_sign_pipeline_with_optional_items_and_solver(
        || {
            let mut simplifier = simplifier_ref.borrow_mut();
            plan_abs_isolation_with_rhs_sign(&mut simplifier.context, arg, rhs, op.clone())
        },
        arg,
        include_items,
        steps,
        |expr| {
            let simplifier = simplifier_ref.borrow();
            solver_render_expr(&simplifier.context, expr)
        },
        |equation| {
            let mut simplifier = simplifier_ref.borrow_mut();
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op.clone(),
                var,
                &mut simplifier,
                opts,
                ctx,
            )
        },
        |item| medium_step(item.description().to_string(), item.equation),
        |positive_set, negative_set| {
            let simplifier = simplifier_ref.borrow();
            finalize_abs_split_solution_set_for_rhs(
                &simplifier.context,
                finalize_op.clone(),
                rhs,
                var,
                positive_set,
                negative_set,
            )
        },
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
    let rewrite = plan_log_isolation_step_with(
        &mut simplifier.context,
        base,
        arg,
        rhs,
        var,
        op.clone(),
        solver_render_expr,
    );
    let include_item = simplifier.collect_steps();
    execute_log_isolation_result_pipeline_with_plan_and_error_and_merge_with_existing_steps(
        include_item,
        steps,
        rewrite,
        |equation| {
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
        CasError::IsolationError(
            var.to_string(),
            "Cannot isolate from log function".to_string(),
        ),
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
    let simplifier_ref = RefCell::new(simplifier);
    execute_unary_inverse_result_pipeline_or_else_with_and_merge_with_existing_steps_with(
        &fn_name,
        arg,
        rhs,
        op.clone(),
        true,
        include_items,
        steps,
        |name, lhs_expr, rhs_expr, rel_op, is_lhs| {
            let mut simplifier = simplifier_ref.borrow_mut();
            plan_unary_inverse_isolation_step(
                &mut simplifier.context,
                name,
                lhs_expr,
                rhs_expr,
                rel_op,
                is_lhs,
            )
        },
        |rhs_expr| {
            let mut simplifier = simplifier_ref.borrow_mut();
            let (simplified_rhs, sim_steps) = simplifier.simplify(rhs_expr);
            let entries = sim_steps
                .into_iter()
                .map(|step| (step.description, step.after))
                .collect::<Vec<_>>();
            (simplified_rhs, entries)
        },
        |lhs, rhs_expr, inner_op| {
            let mut simplifier = simplifier_ref.borrow_mut();
            isolate(lhs, rhs_expr, inner_op, var, &mut simplifier, opts, ctx)
        },
        |item| medium_step(item.description().to_string(), item.equation),
        || CasError::UnknownFunction(fn_name.clone()),
    )
}
