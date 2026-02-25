use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::{medium_step, render_expr as solver_render_expr, SolveStep, SolverOptions};
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::{
    is_known_negative, should_split_division_denominator_sign_cases,
    should_split_isolated_denominator_variable, should_split_product_zero_inequality,
    should_try_reciprocal_solve,
};
use cas_solver_core::solve_outcome::{
    derive_add_isolation_operands, derive_div_isolation_route, derive_mul_isolation_operands,
    derive_sub_isolation_route, execute_division_denominator_plan_with_optional_items,
    execute_division_denominator_sign_split_pipeline_with_optional_items,
    execute_isolated_denominator_sign_split_pipeline_with_optional_items,
    execute_term_isolation_plan_with, finalize_division_denominator_sign_split_solved_sets,
    finalize_isolated_denominator_sign_split_solved_sets,
    finalize_product_zero_inequality_solved_sets, mul_rhs_contains_variable,
    plan_add_operand_isolation_step_with, plan_div_denominator_isolation_with_zero_rhs_guard,
    plan_div_numerator_isolation_step_with, plan_division_denominator,
    plan_division_denominator_sign_split, plan_isolated_denominator_sign_split,
    plan_mul_factor_isolation_step_with, plan_product_zero_inequality_split,
    plan_sub_isolation_step_with, solve_product_zero_inequality_split_execution_with,
    AddIsolationRoute, DivDenominatorIsolationRoute, DivIsolationRoute, SubIsolationRoute,
};

use super::{isolate, prepend_steps};

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
    // PEDAGOGICAL IMPROVEMENT: If BOTH addends contain the variable,
    // try linear_collect directly to avoid circular "subtract" steps.
    let add_operands = derive_add_isolation_operands(&simplifier.context, l, r, var);

    if matches!(add_operands.route, AddIsolationRoute::BothOperands) {
        if let Some((solution_set, linear_steps)) =
            crate::solver::linear_collect::try_linear_collect(lhs, rhs, var, simplifier)
        {
            let mut all_steps = steps;
            all_steps.extend(linear_steps);
            return Ok((solution_set, all_steps));
        }
    }

    let add_moved_desc = solver_render_expr(&simplifier.context, add_operands.moved_addend);
    let include_item = simplifier.collect_steps();
    let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
    let solved = execute_term_isolation_plan_with(
        || {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            plan_add_operand_isolation_step_with(
                &mut simplifier_ref.context,
                add_operands.isolated_addend,
                add_operands.moved_addend,
                rhs,
                op.clone(),
                |_| add_moved_desc.clone(),
            )
        },
        include_item,
        true,
        |expr| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            simplifier_ref.simplify(expr).0
        },
        |equation| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op,
                var,
                *simplifier_ref,
                opts,
                ctx,
            )
        },
        |item| medium_step(item.description, item.equation),
    )?;
    prepend_steps(solved, steps)
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
    let sub_moved = if matches!(
        derive_sub_isolation_route(&simplifier.context, l, var),
        SubIsolationRoute::Minuend
    ) {
        r
    } else {
        l
    };
    let sub_moved_desc = solver_render_expr(&simplifier.context, sub_moved);
    let include_item = simplifier.collect_steps();
    let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
    let solved = execute_term_isolation_plan_with(
        || {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            plan_sub_isolation_step_with(
                &mut simplifier_ref.context,
                l,
                r,
                rhs,
                op.clone(),
                var,
                |_| sub_moved_desc.clone(),
            )
        },
        include_item,
        true,
        |expr| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            simplifier_ref.simplify(expr).0
        },
        |equation| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op,
                var,
                *simplifier_ref,
                opts,
                ctx,
            )
        },
        |item| medium_step(item.description, item.equation),
    )?;
    prepend_steps(solved, steps)
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
    // CRITICAL: For inequalities with products, need sign analysis
    if should_split_product_zero_inequality(&simplifier.context, l, r, rhs, &op, var) {
        // Product inequality split: A * B op 0
        if let Some(plan) =
            plan_product_zero_inequality_split(&mut simplifier.context, l, r, op.clone())
        {
            let solved = solve_product_zero_inequality_split_execution_with(&plan, |equation| {
                solve_with_ctx_and_options(equation, var, simplifier, opts, ctx)
            })?;
            let final_set = finalize_product_zero_inequality_solved_sets(
                &simplifier.context,
                solved.solved_sets,
            );
            return prepend_steps((final_set, solved.steps), steps);
        }
    }

    // Default behavior: divide by one factor
    if mul_rhs_contains_variable(&simplifier.context, rhs, var) {
        if let Some((solution_set, linear_steps)) =
            crate::solver::linear_collect::try_linear_collect_v2(lhs, rhs, var, simplifier)
        {
            let mut all_steps = steps;
            all_steps.extend(linear_steps);
            return Ok((solution_set, all_steps));
        }
    }

    let mul_operands = derive_mul_isolation_operands(&simplifier.context, l, r, var);
    let moved_is_negative = is_known_negative(&simplifier.context, mul_operands.moved_factor);
    let mul_moved_desc = solver_render_expr(&simplifier.context, mul_operands.moved_factor);
    let include_item = simplifier.collect_steps();
    let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
    let solved = execute_term_isolation_plan_with(
        || {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            plan_mul_factor_isolation_step_with(
                &mut simplifier_ref.context,
                mul_operands.isolated_factor,
                mul_operands.moved_factor,
                rhs,
                op.clone(),
                moved_is_negative,
                |_| mul_moved_desc.clone(),
            )
        },
        include_item,
        false,
        |expr| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            simplifier_ref.simplify(expr).0
        },
        |equation| {
            let mut simplifier_ref = runtime_cell.borrow_mut();
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op,
                var,
                *simplifier_ref,
                opts,
                ctx,
            )
        },
        |item| medium_step(item.description, item.equation),
    )?;
    prepend_steps(solved, steps)
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
        if should_split_division_denominator_sign_cases(&simplifier.context, l, r, &op, var) {
            // Denominator contains variable. Split into cases.
            let split_plan = plan_division_denominator_sign_split(
                &mut simplifier.context,
                l,
                r,
                rhs,
                op.clone(),
            )
            .expect("inequality branch requires denominator sign cases");
            let (sim_rhs, _) = simplifier.simplify(split_plan.positive_equation.rhs);
            let include_items = simplifier.collect_steps();
            let solved = {
                let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
                execute_division_denominator_sign_split_pipeline_with_optional_items(
                    split_plan,
                    r,
                    l,
                    op.clone(),
                    sim_rhs,
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
                    |equation| {
                        let mut simplifier_ref = runtime_cell.borrow_mut();
                        let (set, _) =
                            solve_with_ctx_and_options(equation, var, *simplifier_ref, opts, ctx)?;
                        Ok(set)
                    },
                    |item| medium_step(item.description().to_string(), item.equation),
                    |solved_cases| {
                        let simplifier_ref = runtime_cell.borrow();
                        finalize_division_denominator_sign_split_solved_sets(
                            &simplifier_ref.context,
                            solved_cases,
                        )
                    },
                )?
            };
            Ok(solved)
        } else {
            // A = RHS * B
            let denominator_is_negative = is_known_negative(&simplifier.context, r);
            let denominator_desc = solver_render_expr(&simplifier.context, r);
            let include_item = simplifier.collect_steps();
            let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
            let solved = execute_term_isolation_plan_with(
                || {
                    let mut simplifier_ref = runtime_cell.borrow_mut();
                    plan_div_numerator_isolation_step_with(
                        &mut simplifier_ref.context,
                        l,
                        r,
                        rhs,
                        op.clone(),
                        denominator_is_negative,
                        |_| denominator_desc.clone(),
                    )
                },
                include_item,
                false,
                |expr| {
                    let mut simplifier_ref = runtime_cell.borrow_mut();
                    simplifier_ref.simplify(expr).0
                },
                |equation| {
                    let mut simplifier_ref = runtime_cell.borrow_mut();
                    isolate(
                        equation.lhs,
                        equation.rhs,
                        equation.op,
                        var,
                        *simplifier_ref,
                        opts,
                        ctx,
                    )
                },
                |item| medium_step(item.description, item.equation),
            )?;
            prepend_steps(solved, steps)
        }
    } else {
        // B = A / RHS (variable in denominator)

        // PEDAGOGICAL IMPROVEMENT: If LHS is 1/var, use reciprocal solve
        if should_try_reciprocal_solve(&simplifier.context, lhs, &op, var) {
            if let Some((solution_set, reciprocal_steps)) =
                crate::solver::reciprocal_solve::try_reciprocal_solve(lhs, rhs, var, simplifier)
            {
                return Ok((solution_set, reciprocal_steps));
            }
        }

        let isolation_plan = plan_div_denominator_isolation_with_zero_rhs_guard(
            &mut simplifier.context,
            r,
            l,
            rhs,
            op.clone(),
        );
        let isolated_eq = isolation_plan.equation;
        let sim_rhs = if matches!(
            isolation_plan.route,
            DivDenominatorIsolationRoute::RhsZeroToInfinity
        ) {
            isolated_eq.rhs
        } else {
            let (simplified, _) = simplifier.simplify(isolated_eq.rhs);
            simplified
        };

        // Check if denominator is just the variable (simple case)
        if should_split_isolated_denominator_variable(&simplifier.context, r, &op, var) {
            // Split into x > 0 and x < 0
            let split_plan = plan_isolated_denominator_sign_split(r, sim_rhs, op.clone())
                .expect("inequality branch requires denominator sign cases");
            let include_items = simplifier.collect_steps();
            let solved = {
                let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
                execute_isolated_denominator_sign_split_pipeline_with_optional_items(
                    split_plan,
                    r,
                    op.clone(),
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
                    |solved_cases| {
                        let mut simplifier_ref = runtime_cell.borrow_mut();
                        finalize_isolated_denominator_sign_split_solved_sets(
                            &mut simplifier_ref.context,
                            solved_cases,
                        )
                    },
                )?
            };
            return Ok(solved);
        }

        // PEDAGOGICAL: Decompose denominator isolation into two explicit steps
        // only when step collection is enabled.
        let include_items = simplifier.collect_steps();
        let didactic_plan =
            plan_division_denominator(&mut simplifier.context, l, r, rhs, sim_rhs, op.clone());
        let solved = {
            let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
            execute_division_denominator_plan_with_optional_items(
                didactic_plan,
                include_items,
                |expr| {
                    let mut simplifier_ref = runtime_cell.borrow_mut();
                    simplifier_ref.simplify(expr).0
                },
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
        prepend_steps(solved, steps)
    }
}
