use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::{medium_step, render_expr as solver_render_expr, SolveStep, SolverOptions};
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::{is_known_negative, should_try_reciprocal_solve};
use cas_solver_core::solve_outcome::{
    build_division_denominator_execution_with,
    build_division_denominator_sign_split_execution_with,
    build_isolated_denominator_sign_split_execution_with, derive_add_isolation_operands,
    derive_div_isolation_route, derive_mul_isolation_operands, derive_sub_isolation_operands,
    finalize_division_denominator_sign_split_solved_sets,
    finalize_isolated_denominator_sign_split_solved_sets,
    finalize_product_zero_inequality_solved_sets, first_term_isolation_rewrite_execution_item,
    materialize_division_denominator_sign_split_execution,
    materialize_isolated_denominator_sign_split_execution,
    merge_optional_solved_with_existing_steps_append_mut, mul_rhs_contains_variable,
    plan_add_operand_isolation_step_with, plan_div_denominator_isolation_with_zero_rhs_guard,
    plan_division_denominator_sign_split_or_div_numerator_isolation_with,
    plan_isolated_denominator_sign_split_or_division_denominator,
    plan_mul_factor_isolation_step_with, plan_product_zero_inequality_split_if_applicable,
    plan_sub_isolation_step_with, resolve_div_denominator_isolation_rhs_with,
    solve_division_denominator_execution_pipeline_with_items,
    solve_division_denominator_sign_split_execution_pipeline_with_single_solver_with_items,
    solve_isolated_denominator_sign_split_execution_pipeline_with_items,
    solve_product_zero_inequality_split_execution_with, AddIsolationRoute, DivIsolationRoute,
    DivisionDenominatorSignSplitSolvedCases, IsolatedDenominatorSignSplitSolvedCases,
    TermIsolationRewritePlan,
};

use super::isolate;

#[allow(clippy::too_many_arguments)]
fn run_term_isolation_plan(
    plan: TermIsolationRewritePlan,
    include_item: bool,
    simplify_rhs_before_solve: bool,
    existing_steps: Vec<SolveStep>,
    var: &str,
    simplifier: &mut Simplifier,
    opts: SolverOptions,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    let mut steps = Vec::new();
    if include_item {
        if let Some(item) = first_term_isolation_rewrite_execution_item(&plan) {
            steps.push(medium_step(item.description, item.equation));
        }
    }

    let mut rewritten = plan.equation;
    if simplify_rhs_before_solve {
        rewritten.rhs = simplifier.simplify(rewritten.rhs).0;
    }

    let (solution_set, mut solved_steps) = isolate(
        rewritten.lhs,
        rewritten.rhs,
        rewritten.op,
        var,
        simplifier,
        opts,
        ctx,
    )?;
    steps.append(&mut solved_steps);
    steps.extend(existing_steps);
    Ok((solution_set, steps))
}

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
    let add_plan = plan_add_operand_isolation_step_with(
        &mut simplifier.context,
        add_operands.isolated_addend,
        add_operands.moved_addend,
        rhs,
        op.clone(),
        |_| add_moved_desc.clone(),
    );
    let include_item = simplifier.collect_steps();
    run_term_isolation_plan(
        add_plan,
        include_item,
        true,
        steps,
        var,
        simplifier,
        opts,
        ctx,
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
    let sub_plan =
        plan_sub_isolation_step_with(&mut simplifier.context, l, r, rhs, op.clone(), var, |_| {
            sub_moved_desc.clone()
        });
    let include_item = simplifier.collect_steps();
    run_term_isolation_plan(
        sub_plan,
        include_item,
        true,
        steps,
        var,
        simplifier,
        opts,
        ctx,
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
            let solved =
                solve_product_zero_inequality_split_execution_with(&split_plan, |equation| {
                    solve_with_ctx_and_options(equation, var, simplifier, opts, ctx)
                })?;
            let final_set = finalize_product_zero_inequality_solved_sets(
                &simplifier.context,
                solved.solved_sets,
            );
            let mut merged_steps = solved.steps;
            merged_steps.extend(steps);
            return Ok((final_set, merged_steps));
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
    run_term_isolation_plan(
        mul_plan,
        include_item,
        false,
        steps,
        var,
        simplifier,
        opts,
        ctx,
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
        if let Some(split_plan) = split_plan {
            let simplified_rhs = simplifier.simplify(split_plan.positive_equation.rhs).0;
            let execution = if include_item {
                build_division_denominator_sign_split_execution_with(
                    split_plan,
                    r,
                    l,
                    op.clone(),
                    simplified_rhs,
                    |expr| solver_render_expr(&simplifier.context, expr),
                )
            } else {
                materialize_division_denominator_sign_split_execution(split_plan, simplified_rhs)
            };
            let solved =
                solve_division_denominator_sign_split_execution_pipeline_with_single_solver_with_items(
                    &execution,
                    include_item,
                    &steps,
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
                    |item| medium_step(item.description, item.equation),
                )?;
            let final_set = finalize_division_denominator_sign_split_solved_sets(
                &simplifier.context,
                DivisionDenominatorSignSplitSolvedCases {
                    positive_branch: solved.positive_set,
                    negative_branch: solved.negative_set,
                    positive_domain: solved.positive_domain_set,
                    negative_domain: solved.negative_domain_set,
                },
            );
            return Ok((final_set, solved.steps));
        }

        run_term_isolation_plan(
            term_plan,
            include_item,
            false,
            steps,
            var,
            simplifier,
            opts,
            ctx,
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
        if let Some(split_plan) = split_plan {
            let execution = if include_items {
                build_isolated_denominator_sign_split_execution_with(
                    split_plan,
                    r,
                    op.clone(),
                    |expr| solver_render_expr(&simplifier.context, expr),
                )
            } else {
                materialize_isolated_denominator_sign_split_execution(split_plan)
            };
            let solved = solve_isolated_denominator_sign_split_execution_pipeline_with_items(
                &execution,
                include_items,
                &steps,
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
                |item| medium_step(item.description, item.equation),
            )?;
            let final_set = finalize_isolated_denominator_sign_split_solved_sets(
                &mut simplifier.context,
                IsolatedDenominatorSignSplitSolvedCases {
                    positive_branch: solved.positive_set,
                    negative_branch: solved.negative_set,
                },
            );
            return Ok((final_set, solved.steps));
        }

        if include_items {
            let simplified_multiply_rhs =
                simplifier.simplify(didactic_plan.multiply_equation.rhs).0;
            let execution = build_division_denominator_execution_with(
                didactic_plan,
                simplified_multiply_rhs,
                |expr| solver_render_expr(&simplifier.context, expr),
            );
            let solved = solve_division_denominator_execution_pipeline_with_items(
                execution,
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
                |item| medium_step(item.description, item.equation),
            )?;
            let mut merged_steps = solved.steps;
            merged_steps.extend(steps);
            return Ok((solved.solution_set, merged_steps));
        }

        let (solution_set, mut sub_steps) = isolate(
            didactic_plan.divide_equation.lhs,
            didactic_plan.divide_equation.rhs,
            didactic_plan.divide_equation.op,
            var,
            simplifier,
            opts,
            ctx,
        )?;
        sub_steps.extend(steps);
        Ok((solution_set, sub_steps))
    }
}
