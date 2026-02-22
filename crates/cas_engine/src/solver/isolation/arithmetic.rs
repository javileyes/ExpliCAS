use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::{Equation, ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::{
    contains_var, is_known_negative, should_split_division_denominator_sign_cases,
    should_split_isolated_denominator_variable, should_split_product_zero_inequality,
    should_try_reciprocal_solve,
};
use cas_solver_core::solution_set::{
    intersect_solution_sets, open_negative_domain, open_positive_domain, union_solution_sets,
};
use cas_solver_core::solve_outcome::{
    build_add_operand_isolation_step_with, build_div_numerator_isolation_step_with,
    build_division_denominator_didactic_steps_with,
    build_division_denominator_sign_split_steps_with,
    build_isolated_denominator_sign_split_steps_with, build_mul_factor_isolation_step_with,
    build_sub_minuend_isolation_step_with, build_sub_subtrahend_isolation_step_with,
    plan_division_denominator_didactic, plan_division_denominator_sign_split,
    plan_isolated_denominator_sign_split, plan_product_zero_inequality_split,
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
    mut steps: Vec<SolveStep>,
    ctx: &super::super::SolveCtx,
) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
    // PEDAGOGICAL IMPROVEMENT: If BOTH addends contain the variable,
    // try linear_collect directly to avoid circular "subtract" steps.
    let l_has = contains_var(&simplifier.context, l, var);
    let r_has = contains_var(&simplifier.context, r, var);

    if l_has && r_has {
        if let Some((solution_set, linear_steps)) =
            crate::solver::linear_collect::try_linear_collect(lhs, rhs, var, simplifier)
        {
            let mut all_steps = steps;
            all_steps.extend(linear_steps);
            return Ok((solution_set, all_steps));
        }
    }

    if l_has {
        // A + B = RHS -> A = RHS - B
        let new_eq = cas_solver_core::equation_rewrite::isolate_add_operand(
            &mut simplifier.context,
            l,
            r,
            rhs,
            op.clone(),
        );
        let (sim_rhs, _) = simplifier.simplify(new_eq.rhs);
        if simplifier.collect_steps() {
            let step_payload = build_add_operand_isolation_step_with(new_eq.clone(), r, |id| {
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id
                    }
                )
            });
            steps.push(SolveStep {
                description: step_payload.description,
                equation_after: step_payload.equation_after,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(new_eq.lhs, sim_rhs, new_eq.op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
    } else {
        // A + B = RHS -> B = RHS - A
        let new_eq = cas_solver_core::equation_rewrite::isolate_add_operand(
            &mut simplifier.context,
            r,
            l,
            rhs,
            op.clone(),
        );
        let (sim_rhs, _) = simplifier.simplify(new_eq.rhs);
        if simplifier.collect_steps() {
            let step_payload = build_add_operand_isolation_step_with(new_eq.clone(), l, |id| {
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id
                    }
                )
            });
            steps.push(SolveStep {
                description: step_payload.description,
                equation_after: step_payload.equation_after,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(new_eq.lhs, sim_rhs, new_eq.op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
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
        let new_eq = cas_solver_core::equation_rewrite::isolate_sub_minuend(
            &mut simplifier.context,
            l,
            r,
            rhs,
            op.clone(),
        );
        let (sim_rhs, _) = simplifier.simplify(new_eq.rhs);
        if simplifier.collect_steps() {
            let step_payload = build_sub_minuend_isolation_step_with(new_eq.clone(), r, |id| {
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id
                    }
                )
            });
            steps.push(SolveStep {
                description: step_payload.description,
                equation_after: step_payload.equation_after,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(new_eq.lhs, sim_rhs, new_eq.op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
    } else {
        // A - B = RHS -> B = A - RHS (multiply by -1 flips inequality)
        let new_eq = cas_solver_core::equation_rewrite::isolate_sub_subtrahend(
            &mut simplifier.context,
            l,
            r,
            rhs,
            op,
        );
        let (sim_rhs, _) = simplifier.simplify(new_eq.rhs);
        let new_op = new_eq.op.clone();
        if simplifier.collect_steps() {
            let step_payload = build_sub_subtrahend_isolation_step_with(new_eq.clone(), l, |id| {
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id
                    }
                )
            });
            steps.push(SolveStep {
                description: step_payload.description,
                equation_after: step_payload.equation_after,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(new_eq.lhs, sim_rhs, new_op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
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
            let (set_a_case1, _) = solve_with_ctx(&plan.case1_left, var, simplifier, ctx)?;
            let (set_b_case1, _) = solve_with_ctx(&plan.case1_right, var, simplifier, ctx)?;
            let case_set1 = intersect_solution_sets(&simplifier.context, set_a_case1, set_b_case1);

            let (set_a_case2, _) = solve_with_ctx(&plan.case2_left, var, simplifier, ctx)?;
            let (set_b_case2, _) = solve_with_ctx(&plan.case2_right, var, simplifier, ctx)?;
            let case_set2 = intersect_solution_sets(&simplifier.context, set_a_case2, set_b_case2);

            let final_set = union_solution_sets(&simplifier.context, case_set1, case_set2);
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
        let new_eq = cas_solver_core::equation_rewrite::isolate_mul_factor(
            &mut simplifier.context,
            l,
            r,
            rhs,
            op,
            moved_is_negative,
        );
        let new_rhs = new_eq.rhs;
        let new_op = new_eq.op.clone();
        if simplifier.collect_steps() {
            let step_payload = build_mul_factor_isolation_step_with(new_eq.clone(), r, |id| {
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id
                    }
                )
            });
            steps.push(SolveStep {
                description: step_payload.description,
                equation_after: step_payload.equation_after,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(new_eq.lhs, new_rhs, new_op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
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
        let new_eq = cas_solver_core::equation_rewrite::isolate_mul_factor(
            &mut simplifier.context,
            r,
            l,
            rhs,
            op,
            moved_is_negative,
        );
        let new_rhs = new_eq.rhs;
        let new_op = new_eq.op.clone();
        if simplifier.collect_steps() {
            let step_payload = build_mul_factor_isolation_step_with(new_eq.clone(), l, |id| {
                format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id
                    }
                )
            });
            steps.push(SolveStep {
                description: step_payload.description,
                equation_after: step_payload.equation_after,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(new_eq.lhs, new_rhs, new_op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
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
            let eq_pos = Equation {
                lhs: split_plan.positive_equation.lhs,
                rhs: sim_rhs,
                op: split_plan.positive_equation.op.clone(),
            };
            let eq_neg = Equation {
                lhs: split_plan.negative_equation.lhs,
                rhs: sim_rhs,
                op: split_plan.negative_equation.op.clone(),
            };
            let split_steps = simplifier.collect_steps().then(|| {
                build_division_denominator_sign_split_steps_with(
                    eq_pos.clone(),
                    eq_neg.clone(),
                    r,
                    l,
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
            });

            if let Some(split_steps) = split_steps.as_ref() {
                steps.push(SolveStep {
                    description: split_steps.positive_case.description.clone(),
                    equation_after: split_steps.positive_case.equation_after.clone(),
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            let results_pos = isolate(
                eq_pos.lhs,
                eq_pos.rhs,
                eq_pos.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )?;
            let (set_pos, steps_pos) = prepend_steps(results_pos, steps.clone())?;

            // Domain: r > 0
            let (domain_pos_set, _) =
                solve_with_ctx(&split_plan.positive_domain, var, simplifier, ctx)?;
            let final_pos = intersect_solution_sets(&simplifier.context, set_pos, domain_pos_set);

            // Case 2: Denominator < 0
            if let Some(split_steps) = split_steps.as_ref() {
                steps.push(SolveStep {
                    description: split_steps.negative_case.description.clone(),
                    equation_after: split_steps.negative_case.equation_after.clone(),
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            let results_neg = isolate(
                eq_neg.lhs,
                eq_neg.rhs,
                eq_neg.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )?;
            let (set_neg, steps_neg) = prepend_steps(results_neg, steps.clone())?;

            // Domain: r < 0
            let (domain_neg_set, _) =
                solve_with_ctx(&split_plan.negative_domain, var, simplifier, ctx)?;
            let final_neg = intersect_solution_sets(&simplifier.context, set_neg, domain_neg_set);

            // Combine
            let final_set = union_solution_sets(&simplifier.context, final_pos, final_neg);

            // Combine steps
            let mut all_steps = steps_pos;
            if let Some(split_steps) = split_steps {
                all_steps.push(SolveStep {
                    description: split_steps.case_boundary.description,
                    equation_after: split_steps.case_boundary.equation_after,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            all_steps.extend(steps_neg);

            Ok((final_set, all_steps))
        } else {
            // A = RHS * B
            let denominator_is_negative = is_known_negative(&simplifier.context, r);
            let new_eq = cas_solver_core::equation_rewrite::isolate_div_numerator(
                &mut simplifier.context,
                l,
                r,
                rhs,
                op,
                denominator_is_negative,
            );
            let new_rhs = new_eq.rhs;
            let new_op = new_eq.op.clone();
            if simplifier.collect_steps() {
                let step_payload =
                    build_div_numerator_isolation_step_with(new_eq.clone(), r, |id| {
                        format!(
                            "{}",
                            cas_formatter::DisplayExpr {
                                context: &simplifier.context,
                                id
                            }
                        )
                    });
                steps.push(SolveStep {
                    description: step_payload.description,
                    equation_after: step_payload.equation_after,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            let results = isolate(new_eq.lhs, new_rhs, new_op, var, simplifier, opts, ctx)?;
            prepend_steps(results, steps)
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

        let (isolated_eq, isolation_kind) =
            cas_solver_core::equation_rewrite::isolate_div_denominator_with_zero_rhs_guard(
                &mut simplifier.context,
                r,
                l,
                rhs,
                op.clone(),
            );
        let sim_rhs = if matches!(
            isolation_kind,
            cas_solver_core::equation_rewrite::DivDenominatorIsolationKind::RhsZeroToInfinity
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
            let eq_pos = split_plan.positive_equation;
            let eq_neg = split_plan.negative_equation;
            let split_steps = simplifier.collect_steps().then(|| {
                build_isolated_denominator_sign_split_steps_with(
                    eq_pos.clone(),
                    eq_neg.clone(),
                    r,
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
            });

            let mut steps_case1 = steps.clone();
            if let Some(split_steps) = split_steps.as_ref() {
                steps_case1.push(SolveStep {
                    description: split_steps.positive_case.description.clone(),
                    equation_after: split_steps.positive_case.equation_after.clone(),
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            let results_pos = isolate(
                eq_pos.lhs,
                eq_pos.rhs,
                eq_pos.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )?;
            let (set_pos, steps_pos) = prepend_steps(results_pos, steps_case1)?;

            // Intersect with (0, inf)
            let domain_pos = open_positive_domain(&mut simplifier.context);
            let final_pos = intersect_solution_sets(&simplifier.context, set_pos, domain_pos);

            // Case 2: x < 0. Multiply by x (negative) -> Inequality flips.
            let mut steps_case2 = steps.clone();
            if let Some(split_steps) = split_steps.as_ref() {
                steps_case2.push(SolveStep {
                    description: split_steps.negative_case.description.clone(),
                    equation_after: split_steps.negative_case.equation_after.clone(),
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            let results_neg = isolate(
                eq_neg.lhs,
                eq_neg.rhs,
                eq_neg.op.clone(),
                var,
                simplifier,
                opts,
                ctx,
            )?;
            let (set_neg, steps_neg) = prepend_steps(results_neg, steps_case2)?;

            // Intersect with (-inf, 0)
            let domain_neg = open_negative_domain(&mut simplifier.context);
            let final_neg = intersect_solution_sets(&simplifier.context, set_neg, domain_neg);

            // Union
            let final_set = union_solution_sets(&simplifier.context, final_pos, final_neg);

            // Combine steps
            let mut all_steps = steps_pos;
            if let Some(split_steps) = split_steps {
                all_steps.push(SolveStep {
                    description: split_steps.case_boundary.description,
                    equation_after: split_steps.case_boundary.equation_after,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            all_steps.extend(steps_neg);

            return Ok((final_set, all_steps));
        }

        if simplifier.collect_steps() {
            // PEDAGOGICAL: Decompose denominator isolation into two explicit steps.
            let didactic_plan = plan_division_denominator_didactic(
                &mut simplifier.context,
                l,
                r,
                rhs,
                sim_rhs,
                op.clone(),
            );
            let (rhs_times_r_simplified, _) =
                simplifier.simplify(didactic_plan.multiply_equation.rhs);
            let multiply_equation_after = Equation {
                lhs: didactic_plan.multiply_equation.lhs,
                rhs: rhs_times_r_simplified,
                op: didactic_plan.multiply_equation.op.clone(),
            };
            let didactic_steps = build_division_denominator_didactic_steps_with(
                multiply_equation_after,
                didactic_plan.divide_equation,
                didactic_plan.multiply_by,
                didactic_plan.divide_by,
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

            steps.push(SolveStep {
                description: didactic_steps.multiply_step.description,
                equation_after: didactic_steps.multiply_step.equation_after,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
            steps.push(SolveStep {
                description: didactic_steps.divide_step.description,
                equation_after: didactic_steps.divide_step.equation_after,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(r, sim_rhs, op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
    }
}
