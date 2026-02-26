use super::isolate;
use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{medium_step, render_expr as solver_render_expr, SolveStep, SolverOptions};
use cas_ast::symbol::SymbolId;
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::function_inverse::{
    derive_function_isolation_route, execute_unary_inverse_with, plan_unary_inverse_isolation_step,
    solve_unary_inverse_execution_pipeline_with_items, FunctionIsolationRoute,
    FunctionIsolationRouteError,
};
use cas_solver_core::log_isolation::plan_log_isolation_step_with;
use cas_solver_core::solve_outcome::{
    execute_abs_isolation_plan_pipeline_with_optional_items_and_solver,
    execute_log_isolation_result_pipeline_with_plan_and_error_and_merge_with_existing_steps,
    finalize_abs_split_solution_set_for_rhs, merge_solved_with_existing_steps_prepend,
    plan_abs_isolation_with_rhs_sign,
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
    let abs_plan = plan_abs_isolation_with_rhs_sign(&mut simplifier.context, arg, rhs, op.clone());
    let include_items = simplifier.collect_steps();
    let render_ctx = RefCell::new(simplifier.context.clone());
    let finalize_op = op.clone();
    execute_abs_isolation_plan_pipeline_with_optional_items_and_solver(
        abs_plan,
        arg,
        include_items,
        steps,
        |expr| {
            let snapshot = render_ctx.borrow();
            solver_render_expr(&snapshot, expr)
        },
        |equation| {
            let solved = isolate(
                equation.lhs,
                equation.rhs,
                equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            );
            *render_ctx.borrow_mut() = simplifier.context.clone();
            solved
        },
        |item| medium_step(item.description().to_string(), item.equation),
        |positive_set, negative_set| {
            let snapshot = render_ctx.borrow();
            finalize_abs_split_solution_set_for_rhs(
                &snapshot,
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
            let (solution_set, solved_steps) = isolate(
                equation.lhs,
                equation.rhs,
                equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )?;
            Ok::<(SolutionSet, Vec<SolveStep>), CasError>((solution_set, solved_steps))
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
    let unary_plan = plan_unary_inverse_isolation_step(
        &mut simplifier.context,
        &fn_name,
        arg,
        rhs,
        op.clone(),
        true,
    );
    let include_items = simplifier.collect_steps();
    let execution = execute_unary_inverse_with(
        &fn_name,
        arg,
        rhs,
        op.clone(),
        true,
        |_fn_name, _arg, _other, _op, _is_lhs| unary_plan.clone(),
        |rhs_expr| {
            let (simplified_rhs, sim_steps) = simplifier.simplify(rhs_expr);
            let entries = sim_steps
                .into_iter()
                .map(|step| (step.description, step.after))
                .collect::<Vec<_>>();
            (simplified_rhs, entries)
        },
    )
    .ok_or_else(|| CasError::UnknownFunction(fn_name.clone()))?;

    let solved = solve_unary_inverse_execution_pipeline_with_items(
        execution,
        include_items,
        |lhs, rhs_expr, inner_op| isolate(lhs, rhs_expr, inner_op, var, simplifier, opts, ctx),
        |item| medium_step(item.description().to_string(), item.equation),
    )?;

    Ok(merge_solved_with_existing_steps_prepend(
        (solved.solution_set, solved.steps),
        steps,
    ))
}
