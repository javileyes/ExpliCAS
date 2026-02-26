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
    build_abs_split_execution_with, collect_abs_split_execution_items,
    execute_log_isolation_result_pipeline_with_plan_and_error_and_merge_with_existing_steps,
    finalize_abs_split_solution_set_for_rhs, materialize_abs_split_execution,
    plan_abs_isolation_with_rhs_sign, AbsIsolationPlan,
};

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
    match abs_plan {
        AbsIsolationPlan::ReturnEmptySet => Ok((SolutionSet::Empty, steps)),
        AbsIsolationPlan::IsolateSingleEquation { equation } => {
            let (solution_set, mut solved_steps) = isolate(
                equation.lhs,
                equation.rhs,
                equation.op,
                var,
                simplifier,
                opts,
                ctx,
            )?;
            solved_steps.extend(steps);
            Ok((solution_set, solved_steps))
        }
        AbsIsolationPlan::SplitBranches { positive, negative } => {
            let execution = if include_items {
                build_abs_split_execution_with(positive, negative, arg, |expr| {
                    solver_render_expr(&simplifier.context, expr)
                })
            } else {
                materialize_abs_split_execution(positive, negative)
            };
            let mut split_items = if include_items {
                collect_abs_split_execution_items(&execution).into_iter()
            } else {
                Vec::new().into_iter()
            };

            let mut positive_steps = steps.clone();
            if let Some(item) = split_items.next() {
                positive_steps.push(medium_step(item.description().to_string(), item.equation));
            }
            let (positive_set, mut positive_sub_steps) = isolate(
                execution.positive_equation.lhs,
                execution.positive_equation.rhs,
                execution.positive_equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )?;
            positive_steps.append(&mut positive_sub_steps);

            let mut negative_steps = steps;
            if let Some(item) = split_items.next() {
                negative_steps.push(medium_step(item.description().to_string(), item.equation));
            }
            let (negative_set, mut negative_sub_steps) = isolate(
                execution.negative_equation.lhs,
                execution.negative_equation.rhs,
                execution.negative_equation.op,
                var,
                simplifier,
                opts,
                ctx,
            )?;
            negative_steps.append(&mut negative_sub_steps);

            let final_set = finalize_abs_split_solution_set_for_rhs(
                &simplifier.context,
                op,
                rhs,
                var,
                positive_set,
                negative_set,
            );
            positive_steps.extend(negative_steps);
            Ok((final_set, positive_steps))
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
        |lhs, rhs, inner_op| isolate(lhs, rhs, inner_op, var, simplifier, opts, ctx),
        |item| medium_step(item.description().to_string(), item.equation),
    )?;

    let mut solved_steps = solved.steps;
    solved_steps.extend(steps);
    Ok((solved.solution_set, solved_steps))
}
