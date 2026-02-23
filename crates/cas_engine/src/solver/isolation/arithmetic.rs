use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::{ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::{
    contains_var, is_known_negative, should_split_division_denominator_sign_cases,
    should_split_isolated_denominator_variable, should_split_product_zero_inequality,
    should_try_reciprocal_solve,
};
use cas_solver_core::solve_outcome::{
    build_division_denominator_execution_with,
    build_division_denominator_sign_split_execution_with,
    build_isolated_denominator_sign_split_execution_with, derive_add_isolation_route,
    division_denominator_sign_split_boundary_item,
    finalize_division_denominator_sign_split_solved_sets,
    finalize_isolated_denominator_sign_split_solved_sets,
    finalize_product_zero_inequality_solved_sets, first_term_isolation_rewrite_execution_item,
    isolated_denominator_sign_split_boundary_item,
    materialize_division_denominator_sign_split_execution,
    materialize_isolated_denominator_sign_split_execution, plan_add_operand_isolation_step_with,
    plan_div_denominator_isolation_with_zero_rhs_guard, plan_div_numerator_isolation_step_with,
    plan_division_denominator, plan_division_denominator_sign_split,
    plan_isolated_denominator_sign_split, plan_mul_factor_isolation_step_with,
    plan_product_zero_inequality_split, plan_sub_minuend_isolation_step_with,
    plan_sub_subtrahend_isolation_step_with, solve_division_denominator_execution_with_items,
    solve_division_denominator_sign_split_cases_with_items,
    solve_isolated_denominator_sign_split_cases_with_items,
    solve_product_zero_inequality_cases_with, solve_term_isolation_rewrite_with, AddIsolationRoute,
    DivDenominatorIsolationRoute, DivisionDenominatorSignSplitSolvedCases,
    IsolatedDenominatorSignSplitSolvedCases, TermIsolationRewritePlan,
};

use super::{isolate, prepend_steps};

fn append_term_isolation_rewrite_steps(
    steps: &mut Vec<SolveStep>,
    collect_steps: bool,
    plan: &TermIsolationRewritePlan,
) {
    if !collect_steps {
        return;
    }
    if let Some(item) = first_term_isolation_rewrite_execution_item(plan) {
        steps.push(SolveStep {
            description: item.description().to_string(),
            equation_after: item.equation,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        });
    }
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
    let add_route = derive_add_isolation_route(&simplifier.context, l, r, var);

    if matches!(add_route, AddIsolationRoute::BothOperands) {
        if let Some((solution_set, linear_steps)) =
            crate::solver::linear_collect::try_linear_collect(lhs, rhs, var, simplifier)
        {
            let mut all_steps = steps;
            all_steps.extend(linear_steps);
            return Ok((solution_set, all_steps));
        }
    }

    if matches!(
        add_route,
        AddIsolationRoute::LeftOperand | AddIsolationRoute::BothOperands
    ) {
        // A + B = RHS -> A = RHS - B
        let moved_desc = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: r
            }
        );
        let plan = plan_add_operand_isolation_step_with(
            &mut simplifier.context,
            l,
            r,
            rhs,
            op.clone(),
            |_| moved_desc.clone(),
        );
        let solved_rewrite = solve_term_isolation_rewrite_with(plan, |equation| {
            let (sim_rhs, _) = simplifier.simplify(equation.rhs);
            isolate(
                equation.lhs,
                sim_rhs,
                equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )
        })?;
        append_term_isolation_rewrite_steps(
            &mut steps,
            simplifier.collect_steps(),
            &solved_rewrite.rewrite,
        );
        prepend_steps(solved_rewrite.solved, steps)
    } else {
        // A + B = RHS -> B = RHS - A
        let moved_desc = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: l
            }
        );
        let plan = plan_add_operand_isolation_step_with(
            &mut simplifier.context,
            r,
            l,
            rhs,
            op.clone(),
            |_| moved_desc.clone(),
        );
        let solved_rewrite = solve_term_isolation_rewrite_with(plan, |equation| {
            let (sim_rhs, _) = simplifier.simplify(equation.rhs);
            isolate(
                equation.lhs,
                sim_rhs,
                equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )
        })?;
        append_term_isolation_rewrite_steps(
            &mut steps,
            simplifier.collect_steps(),
            &solved_rewrite.rewrite,
        );
        prepend_steps(solved_rewrite.solved, steps)
    }
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
    mut steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    if contains_var(&simplifier.context, l, var) {
        // A - B = RHS -> A = RHS + B
        let moved_desc = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: r
            }
        );
        let plan = plan_sub_minuend_isolation_step_with(
            &mut simplifier.context,
            l,
            r,
            rhs,
            op.clone(),
            |_| moved_desc.clone(),
        );
        let solved_rewrite = solve_term_isolation_rewrite_with(plan, |equation| {
            let (sim_rhs, _) = simplifier.simplify(equation.rhs);
            isolate(
                equation.lhs,
                sim_rhs,
                equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )
        })?;
        append_term_isolation_rewrite_steps(
            &mut steps,
            simplifier.collect_steps(),
            &solved_rewrite.rewrite,
        );
        prepend_steps(solved_rewrite.solved, steps)
    } else {
        // A - B = RHS -> B = A - RHS (multiply by -1 flips inequality)
        let moved_desc = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: l
            }
        );
        let plan =
            plan_sub_subtrahend_isolation_step_with(&mut simplifier.context, l, r, rhs, op, |_| {
                moved_desc.clone()
            });
        let solved_rewrite = solve_term_isolation_rewrite_with(plan, |equation| {
            let (sim_rhs, _) = simplifier.simplify(equation.rhs);
            isolate(
                equation.lhs,
                sim_rhs,
                equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )
        })?;
        append_term_isolation_rewrite_steps(
            &mut steps,
            simplifier.collect_steps(),
            &solved_rewrite.rewrite,
        );
        prepend_steps(solved_rewrite.solved, steps)
    }
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
    if should_split_product_zero_inequality(&simplifier.context, l, r, rhs, &op, var) {
        // Product inequality split: A * B op 0
        if let Some(plan) =
            plan_product_zero_inequality_split(&mut simplifier.context, l, r, op.clone())
        {
            let solved_sets = solve_product_zero_inequality_cases_with(&plan, |equation| {
                let (solution_set, mut sub_steps) = solve_with_ctx(equation, var, simplifier, ctx)?;
                steps.append(&mut sub_steps);
                Ok::<_, CasError>(solution_set)
            })?;
            let final_set =
                finalize_product_zero_inequality_solved_sets(&simplifier.context, solved_sets);
            return Ok((final_set, steps));
        }
    }

    // Default behavior: divide by one factor
    if contains_var(&simplifier.context, l, var) {
        // If RHS also contains var, try linear_collect
        if contains_var(&simplifier.context, rhs, var) {
            if let Some((solution_set, linear_steps)) =
                crate::solver::linear_collect::try_linear_collect_v2(lhs, rhs, var, simplifier)
            {
                let mut all_steps = steps;
                all_steps.extend(linear_steps);
                return Ok((solution_set, all_steps));
            }
        }

        // A = RHS / B
        let moved_is_negative = is_known_negative(&simplifier.context, r);
        let moved_desc = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: r
            }
        );
        let plan = plan_mul_factor_isolation_step_with(
            &mut simplifier.context,
            l,
            r,
            rhs,
            op,
            moved_is_negative,
            |_| moved_desc.clone(),
        );
        let solved_rewrite = solve_term_isolation_rewrite_with(plan, |equation| {
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )
        })?;
        append_term_isolation_rewrite_steps(
            &mut steps,
            simplifier.collect_steps(),
            &solved_rewrite.rewrite,
        );
        prepend_steps(solved_rewrite.solved, steps)
    } else {
        // B = RHS / A (r contains var, l doesn't)
        if contains_var(&simplifier.context, rhs, var) {
            if let Some((solution_set, linear_steps)) =
                crate::solver::linear_collect::try_linear_collect_v2(lhs, rhs, var, simplifier)
            {
                let mut all_steps = steps;
                all_steps.extend(linear_steps);
                return Ok((solution_set, all_steps));
            }
        }

        let moved_is_negative = is_known_negative(&simplifier.context, l);
        let moved_desc = format!(
            "{}",
            cas_formatter::DisplayExpr {
                context: &simplifier.context,
                id: l
            }
        );
        let plan = plan_mul_factor_isolation_step_with(
            &mut simplifier.context,
            r,
            l,
            rhs,
            op,
            moved_is_negative,
            |_| moved_desc.clone(),
        );
        let solved_rewrite = solve_term_isolation_rewrite_with(plan, |equation| {
            isolate(
                equation.lhs,
                equation.rhs,
                equation.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )
        })?;
        append_term_isolation_rewrite_steps(
            &mut steps,
            simplifier.collect_steps(),
            &solved_rewrite.rewrite,
        );
        prepend_steps(solved_rewrite.solved, steps)
    }
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
    if contains_var(&simplifier.context, l, var) {
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
            let split_execution = if simplifier.collect_steps() {
                build_division_denominator_sign_split_execution_with(
                    split_plan,
                    r,
                    l,
                    op.clone(),
                    sim_rhs,
                    |id| {
                        format!(
                            "{}",
                            cas_formatter::DisplayExpr {
                                context: &simplifier.context,
                                id
                            }
                        )
                    },
                )
            } else {
                materialize_division_denominator_sign_split_execution(split_plan, sim_rhs)
            };
            let mut precomputed_domains = {
                let (domain_pos_set, _) =
                    solve_with_ctx(&split_execution.positive_domain, var, simplifier, ctx)?;
                let (domain_neg_set, _) =
                    solve_with_ctx(&split_execution.negative_domain, var, simplifier, ctx)?;
                vec![domain_pos_set, domain_neg_set].into_iter()
            };
            let solved = solve_division_denominator_sign_split_cases_with_items(
                &split_execution,
                |item, equation| {
                    let mut case_steps = steps.clone();
                    if let Some(item) = item {
                        case_steps.push(SolveStep {
                            description: item.description().to_string(),
                            equation_after: item.equation,
                            importance: crate::step::ImportanceLevel::Medium,
                            substeps: vec![],
                        });
                    }
                    let results = isolate(
                        equation.lhs,
                        equation.rhs,
                        equation.op.clone(),
                        var,
                        simplifier,
                        opts,
                        ctx,
                    )?;
                    prepend_steps(results, case_steps)
                },
                |_domain_equation| {
                    Ok::<_, CasError>(
                        precomputed_domains
                            .next()
                            .expect("precomputed domain sets for both sign branches"),
                    )
                },
            )?;

            let DivisionDenominatorSignSplitSolvedCases {
                positive_branch: (set_pos, steps_pos),
                negative_branch: (set_neg, steps_neg),
                positive_domain: domain_pos_set,
                negative_domain: domain_neg_set,
            } = solved;
            let final_set = finalize_division_denominator_sign_split_solved_sets(
                &simplifier.context,
                DivisionDenominatorSignSplitSolvedCases {
                    positive_branch: set_pos,
                    negative_branch: set_neg,
                    positive_domain: domain_pos_set,
                    negative_domain: domain_neg_set,
                },
            );

            // Combine steps
            let mut all_steps = steps_pos;
            if let Some(item) = division_denominator_sign_split_boundary_item(&split_execution) {
                all_steps.push(SolveStep {
                    description: item.description().to_string(),
                    equation_after: item.equation,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            all_steps.extend(steps_neg);

            Ok((final_set, all_steps))
        } else {
            // A = RHS * B
            let denominator_is_negative = is_known_negative(&simplifier.context, r);
            let moved_desc = format!(
                "{}",
                cas_formatter::DisplayExpr {
                    context: &simplifier.context,
                    id: r
                }
            );
            let plan = plan_div_numerator_isolation_step_with(
                &mut simplifier.context,
                l,
                r,
                rhs,
                op,
                denominator_is_negative,
                |_| moved_desc.clone(),
            );
            let solved_rewrite = solve_term_isolation_rewrite_with(plan, |equation| {
                isolate(
                    equation.lhs,
                    equation.rhs,
                    equation.op.clone(),
                    var,
                    simplifier,
                    opts,
                    ctx,
                )
            })?;
            append_term_isolation_rewrite_steps(
                &mut steps,
                simplifier.collect_steps(),
                &solved_rewrite.rewrite,
            );
            prepend_steps(solved_rewrite.solved, steps)
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
            let split_execution = if simplifier.collect_steps() {
                build_isolated_denominator_sign_split_execution_with(
                    split_plan,
                    r,
                    op.clone(),
                    |id| {
                        format!(
                            "{}",
                            cas_formatter::DisplayExpr {
                                context: &simplifier.context,
                                id
                            }
                        )
                    },
                )
            } else {
                materialize_isolated_denominator_sign_split_execution(split_plan)
            };
            let solved = solve_isolated_denominator_sign_split_cases_with_items(
                &split_execution,
                |item, equation| {
                    let mut case_steps = steps.clone();
                    if let Some(item) = item {
                        case_steps.push(SolveStep {
                            description: item.description().to_string(),
                            equation_after: item.equation,
                            importance: crate::step::ImportanceLevel::Medium,
                            substeps: vec![],
                        });
                    }
                    let results = isolate(
                        equation.lhs,
                        equation.rhs,
                        equation.op.clone(),
                        var,
                        simplifier,
                        opts,
                        ctx,
                    )?;
                    prepend_steps(results, case_steps)
                },
            )?;

            let IsolatedDenominatorSignSplitSolvedCases {
                positive_branch: (set_pos, steps_pos),
                negative_branch: (set_neg, steps_neg),
            } = solved;
            let final_set = finalize_isolated_denominator_sign_split_solved_sets(
                &mut simplifier.context,
                IsolatedDenominatorSignSplitSolvedCases {
                    positive_branch: set_pos,
                    negative_branch: set_neg,
                },
            );

            // Combine steps
            let mut all_steps = steps_pos;
            if let Some(item) = isolated_denominator_sign_split_boundary_item(&split_execution) {
                all_steps.push(SolveStep {
                    description: item.description().to_string(),
                    equation_after: item.equation,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            all_steps.extend(steps_neg);

            return Ok((final_set, all_steps));
        }

        if simplifier.collect_steps() {
            // PEDAGOGICAL: Decompose denominator isolation into two explicit steps.
            let execution_plan =
                plan_division_denominator(&mut simplifier.context, l, r, rhs, sim_rhs, op.clone());
            let (rhs_times_r_simplified, _) =
                simplifier.simplify(execution_plan.multiply_equation.rhs);
            let execution = build_division_denominator_execution_with(
                execution_plan,
                rhs_times_r_simplified,
                |id| {
                    format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id
                        }
                    )
                },
            );
            let solved_execution =
                solve_division_denominator_execution_with_items(execution, |items, equation| {
                    for item in items {
                        steps.push(SolveStep {
                            description: item.description().to_string(),
                            equation_after: item.equation,
                            importance: crate::step::ImportanceLevel::Medium,
                            substeps: vec![],
                        });
                    }
                    isolate(
                        equation.lhs,
                        equation.rhs,
                        equation.op.clone(),
                        var,
                        simplifier,
                        opts,
                        ctx,
                    )
                })?;
            return prepend_steps(solved_execution.solved, steps);
        }
        let results = isolate(r, sim_rhs, op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
    }
}
