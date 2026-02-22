use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, ExprId, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::{
    contains_var, denominator_sign_case_ops, is_inequality_relop, is_known_negative,
    is_numeric_zero, isolated_denominator_variable_case_ops, product_zero_inequality_cases,
};
use cas_solver_core::solution_set::{
    intersect_solution_sets, open_negative_domain, open_positive_domain, pos_inf,
    union_solution_sets,
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
            steps.push(SolveStep {
                description: format!(
                    "Subtract {} from both sides",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: r
                    }
                ),
                equation_after: new_eq.clone(),
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
            steps.push(SolveStep {
                description: format!(
                    "Subtract {} from both sides",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: l
                    }
                ),
                equation_after: new_eq.clone(),
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
            steps.push(SolveStep {
                description: format!(
                    "Add {} to both sides",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: r
                    }
                ),
                equation_after: new_eq.clone(),
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
            steps.push(SolveStep {
                description: format!(
                    "Move {} and multiply by -1 (flips inequality)",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: l
                    }
                ),
                equation_after: new_eq.clone(),
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
    let both_have_var =
        contains_var(&simplifier.context, l, var) && contains_var(&simplifier.context, r, var);
    let rhs_is_zero = is_numeric_zero(&simplifier.context, rhs);

    if both_have_var && rhs_is_zero {
        // Product inequality split: A * B op 0
        if let Some((case1, case2)) = product_zero_inequality_cases(op.clone()) {
            // Case 1
            let (eq_a_case1, eq_b_case1) =
                cas_solver_core::equation_rewrite::build_product_zero_sign_case(
                    &mut simplifier.context,
                    l,
                    r,
                    &case1,
                );
            let (set_a_case1, _) = solve_with_ctx(&eq_a_case1, var, simplifier, ctx)?;
            let (set_b_case1, _) = solve_with_ctx(&eq_b_case1, var, simplifier, ctx)?;
            let case_set1 = intersect_solution_sets(&simplifier.context, set_a_case1, set_b_case1);

            // Case 2
            let (eq_a_case2, eq_b_case2) =
                cas_solver_core::equation_rewrite::build_product_zero_sign_case(
                    &mut simplifier.context,
                    l,
                    r,
                    &case2,
                );
            let (set_a_case2, _) = solve_with_ctx(&eq_a_case2, var, simplifier, ctx)?;
            let (set_b_case2, _) = solve_with_ctx(&eq_b_case2, var, simplifier, ctx)?;
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
            steps.push(SolveStep {
                description: format!(
                    "Divide both sides by {}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: r
                    }
                ),
                equation_after: new_eq.clone(),
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
            steps.push(SolveStep {
                description: format!(
                    "Divide both sides by {}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: l
                    }
                ),
                equation_after: new_eq.clone(),
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
        if contains_var(&simplifier.context, r, var) && is_inequality_relop(&op) {
            // Denominator contains variable. Split into cases.
            // Case 1: Denominator > 0
            let (op_pos, op_neg) = denominator_sign_case_ops(op.clone())
                .expect("inequality branch requires denominator sign cases");
            let new_rhs = simplifier.context.add(Expr::Mul(rhs, r));
            let (sim_rhs, _) = simplifier.simplify(new_rhs);

            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: format!(
                        "Case 1: Assume {} > 0. Multiply by positive denominator.",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: r
                        }
                    ),
                    equation_after: Equation {
                        lhs: l,
                        rhs: sim_rhs,
                        op: op_pos.clone(),
                    },
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            let results_pos = isolate(l, sim_rhs, op_pos, var, simplifier, opts, ctx)?;
            let (set_pos, steps_pos) = prepend_steps(results_pos, steps.clone())?;

            // Domain: r > 0
            let domain_eq = cas_solver_core::equation_rewrite::build_sign_domain_equation(
                &mut simplifier.context,
                r,
                true,
            );
            let (domain_pos_set, _) = solve_with_ctx(&domain_eq, var, simplifier, ctx)?;
            let final_pos = intersect_solution_sets(&simplifier.context, set_pos, domain_pos_set);

            // Case 2: Denominator < 0
            let temp_rhs = simplifier.context.add(Expr::Mul(rhs, r));
            let (sim_rhs, _) = simplifier.simplify(temp_rhs);

            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: format!(
                        "Case 2: Assume {} < 0. Multiply by negative denominator (flips inequality).",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: r
                        }
                    ),
                    equation_after: Equation {
                        lhs: l,
                        rhs: sim_rhs,
                        op: op_neg.clone(),
                    },
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            let results_neg = isolate(l, sim_rhs, op_neg, var, simplifier, opts, ctx)?;
            let (set_neg, steps_neg) = prepend_steps(results_neg, steps.clone())?;

            // Domain: r < 0
            let domain_eq_neg = cas_solver_core::equation_rewrite::build_sign_domain_equation(
                &mut simplifier.context,
                r,
                false,
            );
            let (domain_neg_set, _) = solve_with_ctx(&domain_eq_neg, var, simplifier, ctx)?;
            let final_neg = intersect_solution_sets(&simplifier.context, set_neg, domain_neg_set);

            // Combine
            let final_set = union_solution_sets(&simplifier.context, final_pos, final_neg);

            // Combine steps
            let mut all_steps = steps_pos;
            all_steps.push(SolveStep {
                description: "--- End of Case 1 ---".to_string(),
                equation_after: Equation {
                    lhs: l,
                    rhs: sim_rhs,
                    op,
                },
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
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
                steps.push(SolveStep {
                    description: format!(
                        "Multiply both sides by {}",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: r
                        }
                    ),
                    equation_after: new_eq.clone(),
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
        if matches!(op, RelOp::Eq)
            && cas_solver_core::isolation_utils::is_simple_reciprocal(&simplifier.context, lhs, var)
        {
            if let Some((solution_set, reciprocal_steps)) =
                crate::solver::reciprocal_solve::try_reciprocal_solve(lhs, rhs, var, simplifier)
            {
                return Ok((solution_set, reciprocal_steps));
            }
        }

        // CRITICAL FIX: Check if RHS is zero to avoid creating undefined (1/0)
        let is_rhs_zero = is_numeric_zero(&simplifier.context, rhs);

        let sim_rhs = if is_rhs_zero {
            pos_inf(&mut simplifier.context)
        } else {
            let isolated = cas_solver_core::equation_rewrite::isolate_div_denominator(
                &mut simplifier.context,
                r,
                l,
                rhs,
                op.clone(),
            );
            let (simplified, _) = simplifier.simplify(isolated.rhs);
            simplified
        };

        let new_eq = Equation {
            lhs: r,
            rhs: sim_rhs,
            op: op.clone(),
        };

        // Check if denominator is just the variable (simple case)
        if let Expr::Variable(sym_id) = simplifier.context.get(r) {
            if simplifier.context.sym_name(*sym_id) == var && is_inequality_relop(&op) {
                // Split into x > 0 and x < 0
                let (op_pos, op_neg) = isolated_denominator_variable_case_ops(op.clone())
                    .expect("inequality branch requires denominator sign cases");

                let mut steps_case1 = steps.clone();
                if simplifier.collect_steps() {
                    steps_case1.push(SolveStep {
                        description: format!(
                            "Case 1: Assume {} > 0. Multiply by {} (positive). Inequality direction preserved (flipped from isolation logic).",
                            cas_formatter::DisplayExpr { context: &simplifier.context, id: r },
                            cas_formatter::DisplayExpr { context: &simplifier.context, id: r }
                        ),
                        equation_after: Equation {
                            lhs: r,
                            rhs: sim_rhs,
                            op: op_pos.clone(),
                        },
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }

                let results_pos = isolate(r, sim_rhs, op_pos, var, simplifier, opts, ctx)?;
                let (set_pos, steps_pos) = prepend_steps(results_pos, steps_case1)?;

                // Intersect with (0, inf)
                let domain_pos = open_positive_domain(&mut simplifier.context);
                let final_pos = intersect_solution_sets(&simplifier.context, set_pos, domain_pos);

                // Case 2: x < 0. Multiply by x (negative) -> Inequality flips.
                let mut steps_case2 = steps.clone();
                if simplifier.collect_steps() {
                    steps_case2.push(SolveStep {
                        description: format!(
                            "Case 2: Assume {} < 0. Multiply by {} (negative). Inequality flips.",
                            cas_formatter::DisplayExpr {
                                context: &simplifier.context,
                                id: r
                            },
                            cas_formatter::DisplayExpr {
                                context: &simplifier.context,
                                id: r
                            }
                        ),
                        equation_after: Equation {
                            lhs: r,
                            rhs: sim_rhs,
                            op: op_neg.clone(),
                        },
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }

                let results_neg = isolate(r, sim_rhs, op_neg, var, simplifier, opts, ctx)?;
                let (set_neg, steps_neg) = prepend_steps(results_neg, steps_case2)?;

                // Intersect with (-inf, 0)
                let domain_neg = open_negative_domain(&mut simplifier.context);
                let final_neg = intersect_solution_sets(&simplifier.context, set_neg, domain_neg);

                // Union
                let final_set = union_solution_sets(&simplifier.context, final_pos, final_neg);

                // Combine steps
                let mut all_steps = steps_pos;
                all_steps.push(SolveStep {
                    description: "--- End of Case 1 ---".to_string(),
                    equation_after: Equation {
                        lhs: r,
                        rhs: sim_rhs,
                        op,
                    },
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
                all_steps.extend(steps_neg);

                return Ok((final_set, all_steps));
            }
        }

        if simplifier.collect_steps() {
            // PEDAGOGICAL: Decompose into 2 steps
            // Step 1: Multiply both sides by denominator (r)
            let rhs_times_r = simplifier.context.add(Expr::Mul(rhs, r));
            let (rhs_times_r_simplified, _) = simplifier.simplify(rhs_times_r);

            steps.push(SolveStep {
                description: format!(
                    "Multiply both sides by {}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: r
                    }
                ),
                equation_after: Equation {
                    lhs: l,
                    rhs: rhs_times_r_simplified,
                    op: op.clone(),
                },
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });

            // Step 2: Divide both sides by rhs
            steps.push(SolveStep {
                description: format!(
                    "Divide both sides by {}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: rhs
                    }
                ),
                equation_after: new_eq,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(r, sim_rhs, op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
    }
}
