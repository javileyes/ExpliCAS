use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::{medium_step, render_expr as solver_render_expr, SolveStep, SolverOptions};
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_arithmetic::{
    execute_add_isolation_pipeline_with_default_operands_and_plan_with_state,
    execute_div_denominator_isolation_pipeline_with_default_plan_with_state,
    execute_div_numerator_isolation_pipeline_with_default_plan_with_state,
    execute_mul_isolation_pipeline_with_default_operands_and_plan_with_state,
    execute_sub_isolation_pipeline_with_default_plan_with_state,
};
use cas_solver_core::isolation_utils::{is_known_negative, should_try_reciprocal_solve};
use cas_solver_core::solve_outcome::{
    derive_div_isolation_route, finalize_product_zero_inequality_solved_sets, DivIsolationRoute,
};

use super::isolate;

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
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_item = simplifier.collect_steps();
    execute_add_isolation_pipeline_with_default_operands_and_plan_with_state(
        simplifier,
        lhs,
        l,
        r,
        rhs,
        op,
        var,
        include_item,
        steps,
        |simplifier| &simplifier.context,
        |simplifier| &mut simplifier.context,
        solver_render_expr,
        |simplifier, local_lhs, local_rhs, var_name| {
            crate::solver::linear_collect::try_linear_collect(
                local_lhs, local_rhs, var_name, simplifier,
            )
        },
        |simplifier, expr| simplifier.simplify(expr).0,
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
        |item| medium_step(item.description, item.equation),
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
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_item = simplifier.collect_steps();
    execute_sub_isolation_pipeline_with_default_plan_with_state(
        simplifier,
        l,
        r,
        rhs,
        op,
        var,
        include_item,
        steps,
        |simplifier| &simplifier.context,
        |simplifier| &mut simplifier.context,
        solver_render_expr,
        |simplifier, expr| simplifier.simplify(expr).0,
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
        |item| medium_step(item.description, item.equation),
    )
}

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
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let include_item = simplifier.collect_steps();
    execute_mul_isolation_pipeline_with_default_operands_and_plan_with_state(
        simplifier,
        lhs,
        l,
        r,
        rhs,
        op,
        var,
        include_item,
        steps,
        |simplifier| &simplifier.context,
        |simplifier| &mut simplifier.context,
        |simplifier, equation| solve_with_ctx_and_options(equation, var, simplifier, opts, ctx),
        |simplifier| simplifier.context.clone(),
        finalize_product_zero_inequality_solved_sets,
        |simplifier, local_lhs, local_rhs, var_name| {
            crate::solver::linear_collect::try_linear_collect_v2(
                local_lhs, local_rhs, var_name, simplifier,
            )
        },
        |simplifier, expr| is_known_negative(&simplifier.context, expr),
        solver_render_expr,
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
        |item| medium_step(item.description, item.equation),
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
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let div_route = derive_div_isolation_route(&simplifier.context, l, var);
    if matches!(div_route, DivIsolationRoute::VariableInNumerator) {
        let include_item = simplifier.collect_steps();
        execute_div_numerator_isolation_pipeline_with_default_plan_with_state(
            simplifier,
            l,
            r,
            rhs,
            op,
            var,
            include_item,
            steps,
            |simplifier| &mut simplifier.context,
            |simplifier, expr| is_known_negative(&simplifier.context, expr),
            |simplifier, expr| solver_render_expr(&simplifier.context, expr),
            |simplifier, expr| simplifier.simplify(expr).0,
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
            |item| medium_step(item.description, item.equation),
            |item| medium_step(item.description, item.equation),
            |simplifier, solved_sets| {
                cas_solver_core::solve_outcome::finalize_division_denominator_sign_split_solved_sets(
                    &simplifier.context,
                    solved_sets,
                )
            },
        )
    } else {
        let include_items = simplifier.collect_steps();
        execute_div_denominator_isolation_pipeline_with_default_plan_with_state(
            simplifier,
            lhs,
            l,
            r,
            rhs,
            op,
            var,
            include_items,
            steps,
            |simplifier| &mut simplifier.context,
            |simplifier, lhs_expr, local_op, var_name| {
                should_try_reciprocal_solve(&simplifier.context, lhs_expr, local_op, var_name)
            },
            |simplifier, lhs_expr, local_rhs, var_name| {
                crate::solver::reciprocal_solve::try_reciprocal_solve(
                    lhs_expr, local_rhs, var_name, simplifier,
                )
            },
            |simplifier, expr| solver_render_expr(&simplifier.context, expr),
            |simplifier, expr| simplifier.simplify(expr).0,
            |simplifier, expr| simplifier.simplify(expr).0,
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
            |item| medium_step(item.description, item.equation),
            |simplifier, solved_sets| {
                cas_solver_core::solve_outcome::finalize_isolated_denominator_sign_split_solved_sets(
                    &mut simplifier.context,
                    solved_sets,
                )
            },
        )
    }
}
