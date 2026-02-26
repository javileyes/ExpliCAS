use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::{medium_step, render_expr as solver_render_expr, SolveStep, SolverOptions};
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::{is_known_negative, should_try_reciprocal_solve};
use cas_solver_core::solve_outcome::{
    derive_add_isolation_operands, derive_div_isolation_route, derive_mul_isolation_operands,
    derive_sub_isolation_operands,
    execute_division_denominator_sign_split_or_term_isolation_plan_with_optional_items_and_merge_with_existing_steps_with,
    execute_isolated_denominator_sign_split_or_division_denominator_plan_with_optional_items_and_merge_with_existing_steps_with,
    execute_product_zero_inequality_split_pipeline_with_existing_steps,
    execute_term_isolation_plan_and_merge_with_existing_steps_with,
    finalize_product_zero_inequality_solved_sets,
    merge_optional_solved_with_existing_steps_append_mut, mul_rhs_contains_variable,
    plan_add_operand_isolation_step_with, plan_div_denominator_isolation_with_zero_rhs_guard,
    plan_division_denominator_sign_split_or_div_numerator_isolation_with,
    plan_isolated_denominator_sign_split_or_division_denominator,
    plan_mul_factor_isolation_step_with, plan_product_zero_inequality_split_if_applicable,
    plan_sub_isolation_step_with, resolve_div_denominator_isolation_rhs_with, AddIsolationRoute,
    DivIsolationRoute,
};
use std::cell::RefCell;

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
    mut steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // PEDAGOGICAL IMPROVEMENT: If BOTH addends contain the variable,
    // try linear_collect directly to avoid circular "subtract" steps.
    let add_operands = derive_add_isolation_operands(&simplifier.context, l, r, var);

    if matches!(add_operands.route, AddIsolationRoute::BothOperands) {
        if let Some(merged) = merge_optional_solved_with_existing_steps_append_mut(
            crate::solver::linear_collect::try_linear_collect(lhs, rhs, var, simplifier),
            &mut steps,
        ) {
            return Ok(merged);
        }
    }

    let add_moved_desc = solver_render_expr(&simplifier.context, add_operands.moved_addend);
    let mut add_plan = plan_add_operand_isolation_step_with(
        &mut simplifier.context,
        add_operands.isolated_addend,
        add_operands.moved_addend,
        rhs,
        op.clone(),
        |_| add_moved_desc.clone(),
    );
    add_plan.equation.rhs = simplifier.simplify(add_plan.equation.rhs).0;
    let include_item = simplifier.collect_steps();
    execute_term_isolation_plan_and_merge_with_existing_steps_with(
        add_plan,
        include_item,
        false,
        steps,
        |expr| expr,
        |equation| {
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
    let sub_operands = derive_sub_isolation_operands(&simplifier.context, l, r, var);
    let sub_moved = sub_operands.moved_term;
    let sub_moved_desc = solver_render_expr(&simplifier.context, sub_moved);
    let mut sub_plan =
        plan_sub_isolation_step_with(&mut simplifier.context, l, r, rhs, op.clone(), var, |_| {
            sub_moved_desc.clone()
        });
    sub_plan.equation.rhs = simplifier.simplify(sub_plan.equation.rhs).0;
    let include_item = simplifier.collect_steps();
    execute_term_isolation_plan_and_merge_with_existing_steps_with(
        sub_plan,
        include_item,
        false,
        steps,
        |expr| expr,
        |equation| {
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
    mut steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // CRITICAL: For inequalities with products, need sign analysis
    // Product inequality split: A * B op 0
    {
        let split_plan = plan_product_zero_inequality_split_if_applicable(
            &mut simplifier.context,
            l,
            r,
            rhs,
            op.clone(),
            var,
        );
        if let Some(split_plan) = split_plan {
            let split_ctx = RefCell::new(simplifier.context.clone());
            return execute_product_zero_inequality_split_pipeline_with_existing_steps(
                &split_plan,
                steps,
                |equation| {
                    let solved = solve_with_ctx_and_options(equation, var, simplifier, opts, ctx);
                    *split_ctx.borrow_mut() = simplifier.context.clone();
                    solved
                },
                |solved_sets| {
                    let snapshot = split_ctx.borrow();
                    finalize_product_zero_inequality_solved_sets(&snapshot, solved_sets)
                },
            );
        }
    }

    // Default behavior: divide by one factor
    if mul_rhs_contains_variable(&simplifier.context, rhs, var) {
        if let Some(merged) = merge_optional_solved_with_existing_steps_append_mut(
            crate::solver::linear_collect::try_linear_collect_v2(lhs, rhs, var, simplifier),
            &mut steps,
        ) {
            return Ok(merged);
        }
    }

    let mul_operands = derive_mul_isolation_operands(&simplifier.context, l, r, var);
    let moved_is_negative = is_known_negative(&simplifier.context, mul_operands.moved_factor);
    let mul_moved_desc = solver_render_expr(&simplifier.context, mul_operands.moved_factor);
    let mul_plan = plan_mul_factor_isolation_step_with(
        &mut simplifier.context,
        mul_operands.isolated_factor,
        mul_operands.moved_factor,
        rhs,
        op.clone(),
        moved_is_negative,
        |_| mul_moved_desc.clone(),
    );
    let include_item = simplifier.collect_steps();
    execute_term_isolation_plan_and_merge_with_existing_steps_with(
        mul_plan,
        include_item,
        false,
        steps,
        |expr| expr,
        |equation| {
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
    mut steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let div_route = derive_div_isolation_route(&simplifier.context, l, var);
    if matches!(div_route, DivIsolationRoute::VariableInNumerator) {
        let denominator_is_negative = is_known_negative(&simplifier.context, r);
        let denominator_desc = solver_render_expr(&simplifier.context, r);
        let include_item = simplifier.collect_steps();
        let (split_plan, term_plan) =
            plan_division_denominator_sign_split_or_div_numerator_isolation_with(
                &mut simplifier.context,
                l,
                r,
                rhs,
                op.clone(),
                var,
                denominator_is_negative,
                |_| denominator_desc.clone(),
            );
        let simplifier_ref = RefCell::new(simplifier);
        execute_division_denominator_sign_split_or_term_isolation_plan_with_optional_items_and_merge_with_existing_steps_with(
            split_plan,
            r,
            op.clone(),
            l,
            include_item,
            term_plan,
            false,
            steps,
            |expr| simplifier_ref.borrow_mut().simplify(expr).0,
            move |expr| {
                if expr == r {
                    denominator_desc.clone()
                } else {
                    format!("#{expr}")
                }
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
                .map(|(solution_set, _)| solution_set)
            },
            |item| medium_step(item.description, item.equation),
            |item| medium_step(item.description, item.equation),
            |solved_sets| {
                let simplifier = simplifier_ref.borrow();
                cas_solver_core::solve_outcome::finalize_division_denominator_sign_split_solved_sets(
                    &simplifier.context,
                    solved_sets,
                )
            },
        )
    } else {
        // B = A / RHS (variable in denominator)

        // PEDAGOGICAL IMPROVEMENT: If LHS is 1/var, use reciprocal solve
        if should_try_reciprocal_solve(&simplifier.context, lhs, &op, var) {
            if let Some(merged) = merge_optional_solved_with_existing_steps_append_mut(
                crate::solver::reciprocal_solve::try_reciprocal_solve(lhs, rhs, var, simplifier),
                &mut steps,
            ) {
                return Ok(merged);
            }
        }

        let isolation_plan = plan_div_denominator_isolation_with_zero_rhs_guard(
            &mut simplifier.context,
            r,
            l,
            rhs,
            op.clone(),
        );
        let (_isolated_eq, sim_rhs) =
            resolve_div_denominator_isolation_rhs_with(isolation_plan, |expr| {
                simplifier.simplify(expr).0
            });

        let include_items = simplifier.collect_steps();
        let (split_plan, didactic_plan) =
            plan_isolated_denominator_sign_split_or_division_denominator(
                &mut simplifier.context,
                l,
                r,
                rhs,
                sim_rhs,
                op.clone(),
                var,
            );
        let multiply_by = didactic_plan.multiply_by;
        let divide_by = didactic_plan.divide_by;
        let denominator_desc = solver_render_expr(&simplifier.context, r);
        let multiply_by_desc = solver_render_expr(&simplifier.context, multiply_by);
        let divide_by_desc = solver_render_expr(&simplifier.context, divide_by);
        let simplifier_ref = RefCell::new(simplifier);
        execute_isolated_denominator_sign_split_or_division_denominator_plan_with_optional_items_and_merge_with_existing_steps_with(
            split_plan,
            r,
            op.clone(),
            include_items,
            didactic_plan,
            steps,
            |expr| simplifier_ref.borrow_mut().simplify(expr).0,
            move |expr| {
                if expr == r {
                    denominator_desc.clone()
                } else if expr == multiply_by {
                    multiply_by_desc.clone()
                } else if expr == divide_by {
                    divide_by_desc.clone()
                } else {
                    format!("#{expr}")
                }
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
            |item| medium_step(item.description, item.equation),
            |solved_sets| {
                let mut simplifier = simplifier_ref.borrow_mut();
                cas_solver_core::solve_outcome::finalize_isolated_denominator_sign_split_solved_sets(
                    &mut simplifier.context,
                    solved_sets,
                )
            },
        )
    }
}
