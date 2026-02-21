use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solution_set::{intersect_solution_sets, neg_inf, pos_inf, union_solution_sets};
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::{SolveStep, SolverOptions};
use cas_ast::{BoundType, Equation, Expr, ExprId, Interval, RelOp, SolutionSet};
use num_traits::Zero;

use super::{contains_var, flip_inequality, is_known_negative, isolate, prepend_steps};

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
        // A = RHS - B
        let new_rhs = simplifier.context.add(Expr::Sub(rhs, r));
        let (sim_rhs, _) = simplifier.simplify(new_rhs);
        let new_eq = Equation {
            lhs: l,
            rhs: new_rhs,
            op: op.clone(),
        };
        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: format!(
                    "Subtract {} from both sides",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: r
                    }
                ),
                equation_after: new_eq,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(l, sim_rhs, op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
    } else {
        // B = RHS - A
        let new_rhs = simplifier.context.add(Expr::Sub(rhs, l));
        let (sim_rhs, _) = simplifier.simplify(new_rhs);
        let new_eq = Equation {
            lhs: r,
            rhs: new_rhs,
            op: op.clone(),
        };
        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: format!(
                    "Subtract {} from both sides",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: l
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
        // A = RHS + B
        let new_rhs = simplifier.context.add(Expr::Add(rhs, r));
        let (sim_rhs, _) = simplifier.simplify(new_rhs);
        let new_eq = Equation {
            lhs: l,
            rhs: new_rhs,
            op: op.clone(),
        };
        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: format!(
                    "Add {} to both sides",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: r
                    }
                ),
                equation_after: new_eq,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(l, sim_rhs, op, var, simplifier, opts, ctx)?;
        prepend_steps(results, steps)
    } else {
        // -B = RHS - A -> B = A - RHS
        // Multiply by -1 flips inequality
        let new_rhs = simplifier.context.add(Expr::Sub(l, rhs));
        let (sim_rhs, _) = simplifier.simplify(new_rhs);
        let new_op = flip_inequality(op);
        let new_eq = Equation {
            lhs: r,
            rhs: new_rhs,
            op: new_op.clone(),
        };
        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: format!(
                    "Move {} and multiply by -1 (flips inequality)",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: l
                    }
                ),
                equation_after: new_eq,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(r, sim_rhs, new_op, var, simplifier, opts, ctx)?;
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
    let is_inequality = matches!(op, RelOp::Lt | RelOp::Gt | RelOp::Leq | RelOp::Geq);

    let rhs_is_zero = matches!(simplifier.context.get(rhs), Expr::Number(n) if n.is_zero());

    if both_have_var && is_inequality && rhs_is_zero {
        // Product inequality: A * B > 0 (or < 0, etc.)
        let zero = simplifier.context.num(0);

        match op {
            RelOp::Gt | RelOp::Geq => {
                // A * B > 0: Both same sign
                // Case 1: Both positive
                let eq_a_pos = Equation {
                    lhs: l,
                    rhs: zero,
                    op: if matches!(op, RelOp::Gt) {
                        RelOp::Gt
                    } else {
                        RelOp::Geq
                    },
                };
                let eq_b_pos = Equation {
                    lhs: r,
                    rhs: zero,
                    op: if matches!(op, RelOp::Gt) {
                        RelOp::Gt
                    } else {
                        RelOp::Geq
                    },
                };

                let (set_a_pos, _) = solve_with_ctx(&eq_a_pos, var, simplifier, ctx)?;
                let (set_b_pos, _) = solve_with_ctx(&eq_b_pos, var, simplifier, ctx)?;
                let case_pos = intersect_solution_sets(&simplifier.context, set_a_pos, set_b_pos);

                // Case 2: Both negative
                let eq_a_neg = Equation {
                    lhs: l,
                    rhs: zero,
                    op: if matches!(op, RelOp::Gt) {
                        RelOp::Lt
                    } else {
                        RelOp::Leq
                    },
                };
                let eq_b_neg = Equation {
                    lhs: r,
                    rhs: zero,
                    op: if matches!(op, RelOp::Gt) {
                        RelOp::Lt
                    } else {
                        RelOp::Leq
                    },
                };

                let (set_a_neg, _) = solve_with_ctx(&eq_a_neg, var, simplifier, ctx)?;
                let (set_b_neg, _) = solve_with_ctx(&eq_b_neg, var, simplifier, ctx)?;
                let case_neg = intersect_solution_sets(&simplifier.context, set_a_neg, set_b_neg);

                let final_set = union_solution_sets(&simplifier.context, case_pos, case_neg);

                return Ok((final_set, steps));
            }
            RelOp::Lt | RelOp::Leq => {
                // A * B < 0: Different signs
                // Case 1: A positive, B negative
                let eq_a_pos = Equation {
                    lhs: l,
                    rhs: zero,
                    op: if matches!(op, RelOp::Lt) {
                        RelOp::Gt
                    } else {
                        RelOp::Geq
                    },
                };
                let eq_b_neg = Equation {
                    lhs: r,
                    rhs: zero,
                    op: if matches!(op, RelOp::Lt) {
                        RelOp::Lt
                    } else {
                        RelOp::Leq
                    },
                };

                let (set_a_pos, _) = solve_with_ctx(&eq_a_pos, var, simplifier, ctx)?;
                let (set_b_neg, _) = solve_with_ctx(&eq_b_neg, var, simplifier, ctx)?;
                let case1 = intersect_solution_sets(&simplifier.context, set_a_pos, set_b_neg);

                // Case 2: A negative, B positive
                let eq_a_neg = Equation {
                    lhs: l,
                    rhs: zero,
                    op: if matches!(op, RelOp::Lt) {
                        RelOp::Lt
                    } else {
                        RelOp::Leq
                    },
                };
                let eq_b_pos = Equation {
                    lhs: r,
                    rhs: zero,
                    op: if matches!(op, RelOp::Lt) {
                        RelOp::Gt
                    } else {
                        RelOp::Geq
                    },
                };

                let (set_a_neg, _) = solve_with_ctx(&eq_a_neg, var, simplifier, ctx)?;
                let (set_b_pos, _) = solve_with_ctx(&eq_b_pos, var, simplifier, ctx)?;
                let case2 = intersect_solution_sets(&simplifier.context, set_a_neg, set_b_pos);

                let final_set = union_solution_sets(&simplifier.context, case1, case2);

                return Ok((final_set, steps));
            }
            _ => {
                // Equality - fall through to regular division
            }
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
        let mut new_op = op;
        if is_known_negative(&simplifier.context, r) {
            new_op = flip_inequality(new_op);
        }

        let new_rhs = simplifier.context.add(Expr::Div(rhs, r));
        let new_eq = Equation {
            lhs: l,
            rhs: new_rhs,
            op: new_op.clone(),
        };
        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: format!(
                    "Divide both sides by {}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: r
                    }
                ),
                equation_after: new_eq,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(l, new_rhs, new_op, var, simplifier, opts, ctx)?;
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

        let mut new_op = op;
        if is_known_negative(&simplifier.context, l) {
            new_op = flip_inequality(new_op);
        }

        let new_rhs = simplifier.context.add(Expr::Div(rhs, l));
        let new_eq = Equation {
            lhs: r,
            rhs: new_rhs,
            op: new_op.clone(),
        };
        if simplifier.collect_steps() {
            steps.push(SolveStep {
                description: format!(
                    "Divide both sides by {}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: l
                    }
                ),
                equation_after: new_eq,
                importance: crate::step::ImportanceLevel::Medium,
                substeps: vec![],
            });
        }
        let results = isolate(r, new_rhs, new_op, var, simplifier, opts, ctx)?;
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
        if contains_var(&simplifier.context, r, var)
            && matches!(op, RelOp::Lt | RelOp::Gt | RelOp::Leq | RelOp::Geq)
        {
            // Denominator contains variable. Split into cases.
            // Case 1: Denominator > 0
            let op_pos = op.clone();
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
            let zero = simplifier.context.num(0);
            let domain_eq = Equation {
                lhs: r,
                rhs: zero,
                op: RelOp::Gt,
            };
            let (domain_pos_set, _) = solve_with_ctx(&domain_eq, var, simplifier, ctx)?;
            let final_pos = intersect_solution_sets(&simplifier.context, set_pos, domain_pos_set);

            // Case 2: Denominator < 0
            let op_neg = flip_inequality(op.clone());
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
            let zero = simplifier.context.num(0);
            let domain_eq_neg = Equation {
                lhs: r,
                rhs: zero,
                op: RelOp::Lt,
            };
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
            let mut new_op = op;
            if is_known_negative(&simplifier.context, r) {
                new_op = flip_inequality(new_op);
            }

            let new_rhs = simplifier.context.add(Expr::Mul(rhs, r));
            let new_eq = Equation {
                lhs: l,
                rhs: new_rhs,
                op: new_op.clone(),
            };
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: format!(
                        "Multiply both sides by {}",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: r
                        }
                    ),
                    equation_after: new_eq,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }
            let results = isolate(l, new_rhs, new_op, var, simplifier, opts, ctx)?;
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
        let is_rhs_zero = matches!(simplifier.context.get(rhs), Expr::Number(n) if n.is_zero());

        let sim_rhs = if is_rhs_zero {
            pos_inf(&mut simplifier.context)
        } else {
            let new_rhs = simplifier.context.add(Expr::Div(l, rhs));
            let (simplified, _) = simplifier.simplify(new_rhs);
            simplified
        };

        let new_eq = Equation {
            lhs: r,
            rhs: sim_rhs,
            op: op.clone(),
        };

        // Check if denominator is just the variable (simple case)
        if let Expr::Variable(sym_id) = simplifier.context.get(r) {
            if simplifier.context.sym_name(*sym_id) == var
                && matches!(op, RelOp::Lt | RelOp::Gt | RelOp::Leq | RelOp::Geq)
            {
                // Split into x > 0 and x < 0

                let op_pos = flip_inequality(op.clone());

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
                let domain_pos = SolutionSet::Continuous(Interval {
                    min: simplifier.context.num(0),
                    min_type: BoundType::Open,
                    max: pos_inf(&mut simplifier.context),
                    max_type: BoundType::Open,
                });
                let final_pos = intersect_solution_sets(&simplifier.context, set_pos, domain_pos);

                // Case 2: x < 0. Multiply by x (negative) -> Inequality flips.
                let op_neg = op.clone();

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
                let domain_neg = SolutionSet::Continuous(Interval {
                    min: neg_inf(&mut simplifier.context),
                    min_type: BoundType::Open,
                    max: simplifier.context.num(0),
                    max_type: BoundType::Open,
                });
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
