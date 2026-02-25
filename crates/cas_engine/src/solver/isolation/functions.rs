use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{medium_step, render_expr as solver_render_expr, SolveStep, SolverOptions};
use cas_ast::symbol::SymbolId;
use cas_ast::{Equation, ExprId, RelOp, SolutionSet};
use cas_solver_core::function_inverse::{
    derive_function_isolation_route, execute_unary_inverse_pipeline_with_items_with,
    plan_unary_inverse_isolation_step, FunctionIsolationRoute, FunctionIsolationRouteError,
};
use cas_solver_core::isolation_utils::{contains_var, numeric_sign};
use cas_solver_core::log_isolation::{
    plan_log_isolation_step_with, solve_log_isolation_rewrite_pipeline_with_item,
};
use cas_solver_core::solve_outcome::{
    finalize_abs_split_solution_set, plan_abs_isolation, solve_abs_isolation_plan_with,
    solve_abs_split_pipeline_with_optional_items, AbsIsolationSolved,
};

use super::{isolate, prepend_steps};

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
    let rhs_sign = numeric_sign(&simplifier.context, rhs);
    let abs_plan = plan_abs_isolation(&mut simplifier.context, arg, rhs, op.clone(), rhs_sign);
    let dispatched_abs =
        solve_abs_isolation_plan_with(abs_plan, Ok::<Equation, CasError>, |positive, negative| {
            Ok::<(Equation, Equation), CasError>((positive, negative))
        })?;

    match dispatched_abs {
        AbsIsolationSolved::ReturnedEmptySet => Ok((SolutionSet::Empty, steps)),
        AbsIsolationSolved::IsolatedSingle(equation) => isolate(
            equation.lhs,
            equation.rhs,
            equation.op,
            var,
            simplifier,
            opts,
            ctx,
        ),
        AbsIsolationSolved::Split((positive, negative)) => {
            let include_items = simplifier.collect_steps();
            let solved = {
                let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
                solve_abs_split_pipeline_with_optional_items(
                    positive,
                    negative,
                    arg,
                    include_items,
                    &steps,
                    |expr| {
                        let simplifier_ref = runtime_cell.borrow();
                        solver_render_expr(&simplifier_ref.context, expr)
                    },
                    |equation| {
                        let mut simplifier_ref = runtime_cell.borrow_mut();
                        isolate(
                            equation.lhs,
                            equation.rhs,
                            equation.op.clone(),
                            var,
                            *simplifier_ref,
                            opts,
                            ctx,
                        )
                    },
                    |item| medium_step(item.description().to_string(), item.equation),
                )?
            };
            let final_set = finalize_abs_split_solution_set(
                &simplifier.context,
                op,
                contains_var(&simplifier.context, rhs, var),
                rhs,
                solved.positive_set,
                solved.negative_set,
            );
            Ok((final_set, solved.steps))
        }
    }
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
    )
    .ok_or_else(|| {
        CasError::IsolationError(
            var.to_string(),
            "Cannot isolate from log function".to_string(),
        )
    })?;
    let include_item = simplifier.collect_steps();
    let solved = solve_log_isolation_rewrite_pipeline_with_item(
        rewrite,
        include_item,
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
    )?;
    let (solution_set, solved_steps) = solved.solved;
    prepend_steps((solution_set, solved_steps), steps)
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
    let solved_execution = {
        let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
        execute_unary_inverse_pipeline_with_items_with(
            &fn_name,
            arg,
            rhs,
            op.clone(),
            true,
            include_items,
            |inner_fn_name, inner_arg, inner_other, inner_op, inner_is_lhs| {
                let mut simplifier_ref = runtime_cell.borrow_mut();
                plan_unary_inverse_isolation_step(
                    &mut simplifier_ref.context,
                    inner_fn_name,
                    inner_arg,
                    inner_other,
                    inner_op,
                    inner_is_lhs,
                )
            },
            |rhs_expr| {
                let mut simplifier_ref = runtime_cell.borrow_mut();
                let (simplified_rhs, sim_steps) = simplifier_ref.simplify(rhs_expr);
                let entries = sim_steps
                    .into_iter()
                    .map(|step| (step.description, step.after))
                    .collect::<Vec<_>>();
                (simplified_rhs, entries)
            },
            |lhs, rhs, inner_op| {
                let mut simplifier_ref = runtime_cell.borrow_mut();
                isolate(lhs, rhs, inner_op, var, *simplifier_ref, opts, ctx)
            },
            |item| medium_step(item.description().to_string(), item.equation),
        )
        .ok_or_else(|| CasError::UnknownFunction(fn_name.clone()))??
    };
    prepend_steps(
        (solved_execution.solution_set, solved_execution.steps),
        steps,
    )
}
