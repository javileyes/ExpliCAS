use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{medium_step, render_expr as solver_render_expr, SolveStep, SolverOptions};
use cas_ast::symbol::SymbolId;
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::function_inverse::{
    derive_function_isolation_route, execute_unary_inverse_with_runtime,
    solve_unary_inverse_execution_pipeline_with_items, FunctionIsolationRoute,
    FunctionIsolationRouteError, UnaryInverseRuntime,
};
use cas_solver_core::isolation_utils::{contains_var, numeric_sign};
use cas_solver_core::log_isolation::{
    plan_log_isolation_step_with, solve_log_isolation_rewrite_with_item,
};
use cas_solver_core::solve_outcome::{
    finalize_abs_split_solution_set, plan_abs_isolation, solve_abs_isolation_plan_with,
    solve_abs_split_pipeline_with_optional_items, AbsIsolationSolved,
};
use std::cell::RefCell;

use super::{isolate, prepend_steps};

fn identity_equation(equation: cas_ast::Equation) -> Result<cas_ast::Equation, CasError> {
    Ok(equation)
}

fn pair_equations(
    positive: cas_ast::Equation,
    negative: cas_ast::Equation,
) -> Result<(cas_ast::Equation, cas_ast::Equation), CasError> {
    Ok((positive, negative))
}

struct EngineUnaryInverseRuntime<'a> {
    simplifier: &'a mut Simplifier,
}

impl UnaryInverseRuntime for EngineUnaryInverseRuntime<'_> {
    fn context(&mut self) -> &mut cas_ast::Context {
        &mut self.simplifier.context
    }

    fn simplify_rhs_with_entries(&mut self, rhs: ExprId) -> (ExprId, Vec<(String, ExprId)>) {
        let (simplified_rhs, sim_steps) = self.simplifier.simplify(rhs);
        let entries = sim_steps
            .into_iter()
            .map(|step| (step.description, step.after))
            .collect::<Vec<_>>();
        (simplified_rhs, entries)
    }
}

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
        solve_abs_isolation_plan_with(abs_plan, identity_equation, pair_equations)?;

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
            let simplifier_cell = RefCell::new(simplifier);
            let solved = solve_abs_split_pipeline_with_optional_items(
                positive,
                negative,
                arg,
                include_items,
                &steps,
                |id| {
                    let s_ref = simplifier_cell.borrow();
                    solver_render_expr(&s_ref.context, id)
                },
                |equation| {
                    let mut s_ref = simplifier_cell.borrow_mut();
                    isolate(
                        equation.lhs,
                        equation.rhs,
                        equation.op.clone(),
                        var,
                        &mut s_ref,
                        opts,
                        ctx,
                    )
                },
                |item| medium_step(item.description().to_string(), item.equation),
            )?;
            let s_ref = simplifier_cell.borrow();
            let final_set = finalize_abs_split_solution_set(
                &s_ref.context,
                op,
                contains_var(&s_ref.context, rhs, var),
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
        |render_ctx, id| solver_render_expr(render_ctx, id),
    )
    .ok_or_else(|| {
        CasError::IsolationError(
            var.to_string(),
            "Cannot isolate from log function".to_string(),
        )
    })?;
    let include_item = simplifier.collect_steps();
    let solved = solve_log_isolation_rewrite_with_item(rewrite, |item, equation| {
        isolate(
            equation.lhs,
            equation.rhs,
            equation.op.clone(),
            var,
            simplifier,
            opts,
            ctx,
        )
        .map(|isolated| (item, isolated))
    })?;
    let (item, (solution_set, mut solved_steps)) = solved.solved;
    let mut merged_steps =
        Vec::with_capacity(solved_steps.len() + usize::from(include_item && item.is_some()));
    if include_item {
        if let Some(item) = item {
            merged_steps.push(medium_step(item.description, item.equation));
        }
    }
    merged_steps.append(&mut solved_steps);
    prepend_steps((solution_set, merged_steps), steps)
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
    let execution = {
        let mut runtime = EngineUnaryInverseRuntime { simplifier };
        execute_unary_inverse_with_runtime(&mut runtime, &fn_name, arg, rhs, op.clone(), true)
            .ok_or_else(|| CasError::UnknownFunction(fn_name.clone()))?
    };
    let include_items = simplifier.collect_steps();
    let solved_execution = solve_unary_inverse_execution_pipeline_with_items(
        execution,
        include_items,
        |rewritten_lhs, rewritten_rhs, rewritten_op| {
            isolate(
                rewritten_lhs,
                rewritten_rhs,
                rewritten_op,
                var,
                simplifier,
                opts,
                ctx,
            )
        },
        |item| medium_step(item.description().to_string(), item.equation),
    )?;
    prepend_steps(
        (solved_execution.solution_set, solved_execution.steps),
        steps,
    )
}
