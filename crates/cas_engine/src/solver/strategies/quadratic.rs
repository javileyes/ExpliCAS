use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::{contains_var, is_numeric_zero, split_zero_product_factors};
use cas_solver_core::quadratic_formula::{
    discriminant, discriminant_expr, roots_from_a_b_and_sqrt, roots_from_a_b_delta, sqrt_expr,
};
use cas_solver_core::solution_set::{
    get_number, order_pair_by_value, quadratic_numeric_solution, sort_and_dedup_exprs,
};
use num_rational::BigRational;
use num_traits::Zero;

pub struct QuadraticStrategy;

impl SolverStrategy for QuadraticStrategy {
    fn name(&self) -> &str {
        "Quadratic Formula"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        _opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        let mut steps = Vec::new();

        // Move everything to LHS: lhs - rhs = 0
        let poly_expr = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
        // Simplify first (this might factor it)
        let (sim_poly_expr, _) = simplifier.simplify(poly_expr);

        // Check for Zero Product Property: A * B = 0 => A = 0 or B = 0
        // Only if RHS is 0 (which it is, we moved everything to LHS)
        // We need to be careful not to recurse infinitely if A or B are not simpler.
        // But solving A=0 and B=0 breaks it down.

        let zero = simplifier.context.num(0);

        if let Some(factors) = split_zero_product_factors(&simplifier.context, sim_poly_expr) {
            // We found factors.
            if simplifier.collect_steps() {
                steps.push(SolveStep {
                    description: format!(
                        "Factorized equation: {} = 0",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id: sim_poly_expr
                        }
                    ),
                    equation_after: Equation {
                        lhs: sim_poly_expr,
                        rhs: zero,
                        op: RelOp::Eq,
                    },
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            // For inequalities, splitting is complex (sign analysis).
            // For Eq, it's simple union.
            if eq.op == RelOp::Eq {
                let mut all_solutions = Vec::new();
                for factor in factors {
                    // Skip factors that don't contain the variable (e.g., constant multipliers)
                    // This fixes cases like x/10000 = 5/10000 where simplified is (x-5)/10000 = Mul(1/10000, x-5)
                    // The 1/10000 factor has no variable, so we skip it
                    if !contains_var(&simplifier.context, factor, var) {
                        continue;
                    }

                    if simplifier.collect_steps() {
                        steps.push(SolveStep {
                            description: format!(
                                "Solve factor: {} = 0",
                                cas_formatter::DisplayExpr {
                                    context: &simplifier.context,
                                    id: factor
                                }
                            ),
                            equation_after: Equation {
                                lhs: factor,
                                rhs: zero,
                                op: RelOp::Eq,
                            },
                            importance: crate::step::ImportanceLevel::Medium,
                            substeps: vec![],
                        });
                    }
                    let factor_eq = Equation {
                        lhs: factor,
                        rhs: zero,
                        op: RelOp::Eq,
                    };
                    // Recursive solve
                    // We need to be careful about depth.
                    match solve_with_ctx(&factor_eq, var, simplifier, ctx) {
                        Ok((sol_set, mut sub_steps)) => {
                            steps.append(&mut sub_steps);
                            match sol_set {
                                SolutionSet::Discrete(sols) => all_solutions.extend(sols),
                                SolutionSet::Empty => {
                                    // No solutions from this factor — skip
                                }
                                SolutionSet::AllReals => {
                                    // Factor is identically zero → entire equation AllReals
                                    return Some(Ok((SolutionSet::AllReals, steps)));
                                }
                                _ => {
                                    // Residual/Conditional/Interval: can't extract discrete
                                    // roots. Return Residual for the whole equation rather
                                    // than crashing with SolverError.
                                    let residual =
                                        simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
                                    let (sim, _) = simplifier.simplify(residual);
                                    return Some(Ok((SolutionSet::Residual(sim), steps)));
                                }
                            }
                        }
                        Err(e) => return Some(Err(e)),
                    }
                }
                // Remove duplicates
                sort_and_dedup_exprs(&simplifier.context, &mut all_solutions);

                return Some(Ok((SolutionSet::Discrete(all_solutions), steps)));
            }
        }

        // Ensure expanded form for coefficient extraction
        // QuadraticStrategy relies on A*x^2 + B*x + C structure (Add/Sub chain)
        let expanded_expr = crate::expand::expand(&mut simplifier.context, sim_poly_expr);

        if let Some((a, b, c)) = cas_solver_core::quadratic_coeffs::extract_quadratic_coefficients(
            &mut simplifier.context,
            expanded_expr,
            var,
        ) {
            // Check if it is actually quadratic (a != 0)
            // We need to simplify 'a' to check if it's zero.
            let (sim_a, _) = simplifier.simplify(a);
            let (sim_b, _) = simplifier.simplify(b);
            let (sim_c, _) = simplifier.simplify(c);

            // If a is zero, it's linear, not quadratic.
            // But if 'a' is symbolic, we might assume it's non-zero or return a conditional solution?
            // For now, if 'a' simplifies to explicit 0, we reject.
            if is_numeric_zero(&simplifier.context, sim_a) {
                return None;
            }

            if simplifier.collect_steps() {
                // Build didactic substeps showing completing-the-square derivation
                let is_real_only =
                    matches!(_opts.value_domain, crate::semantics::ValueDomain::RealOnly);
                let mut substeps = crate::solver::quadratic_steps::build_quadratic_substeps(
                    simplifier,
                    var,
                    sim_a,
                    sim_b,
                    sim_c,
                    is_real_only,
                );

                // Post-pass: apply didactic simplification to clean up expressions
                crate::solver::quadratic_steps::didactic_simplify_substeps(
                    simplifier,
                    &mut substeps,
                );

                // Main step with substeps attached
                let main_step = SolveStep {
                    description:
                        cas_solver_core::quadratic_didactic::QUADRATIC_FORMULA_MAIN_STEP_DESCRIPTION
                            .to_string(),
                    equation_after: Equation {
                        lhs: sim_poly_expr,
                        rhs: simplifier.context.num(0),
                        op: RelOp::Eq,
                    },
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps,
                };
                steps.push(main_step);
            }

            // Check if coefficients are all numeric to support inequalities
            let a_num = get_number(&simplifier.context, sim_a);
            let b_num = get_number(&simplifier.context, sim_b);
            let c_num = get_number(&simplifier.context, sim_c);

            if let (Some(a_val), Some(b_val), Some(c_val)) = (a_num, b_num, c_num) {
                // Use numeric logic for better inequality support
                // delta = b^2 - 4ac
                let delta = discriminant(&a_val, &b_val, &c_val);

                // We need to return solutions in terms of Expr
                // x = (-b +/- sqrt(delta)) / 2a

                let delta_expr = simplifier.context.add(Expr::Number(delta.clone()));
                let (sol1, sol2) =
                    roots_from_a_b_delta(&mut simplifier.context, sim_a, sim_b, delta_expr);

                let (sim_sol1, _) = simplifier.simplify(sol1);
                let (sim_sol2, _) = simplifier.simplify(sol2);

                // Ensure r1 <= r2
                let (r1, r2) = order_pair_by_value(&simplifier.context, sim_sol1, sim_sol2);

                // Determine parabola direction
                let opens_up = a_val > BigRational::zero();

                let result = quadratic_numeric_solution(
                    &mut simplifier.context,
                    eq.op.clone(),
                    &delta,
                    opens_up,
                    r1,
                    r2,
                );

                // Emit scope for display transforms (sqrt display in quadratic context)
                crate::solver::emit_scope(cas_formatter::display_transforms::ScopeTag::Rule(
                    "QuadraticFormula",
                ));

                return Some(Ok((result, steps)));
            }

            // Symbolic coefficients

            // delta = b^2 - 4ac
            let delta_raw = discriminant_expr(&mut simplifier.context, sim_a, sim_b, sim_c);

            // POST-SIMPLIFY: Expand then simplify discriminant for cleaner form
            // This converts "16 + 4*(y - 4)" → "4*y"
            let delta_expanded = crate::expand::expand(&mut simplifier.context, delta_raw);
            let (sim_delta, _) = simplifier.simplify(delta_expanded);

            // x = (-b +/- sqrt(delta)) / 2a

            let sqrt_delta_raw = sqrt_expr(&mut simplifier.context, sim_delta);

            // POST-SIMPLIFY: Pull perfect square numeric factors from sqrt
            // This converts sqrt(4*y) → 2*sqrt(y)
            let sqrt_delta = cas_solver_core::quadratic_sqrt::pull_square_from_sqrt(
                &mut simplifier.context,
                sqrt_delta_raw,
            );

            let (sol1_raw, sol2_raw) =
                roots_from_a_b_and_sqrt(&mut simplifier.context, sim_a, sim_b, sqrt_delta);

            // POST-SIMPLIFY: Expand and simplify solutions for cleaner form
            // This converts "(4 ± 2*sqrt(y)) / 2" → "2 ± sqrt(y)"
            let sol1_expanded = crate::expand::expand(&mut simplifier.context, sol1_raw);
            let (sim_sol1, _) = simplifier.simplify(sol1_expanded);

            let sol2_expanded = crate::expand::expand(&mut simplifier.context, sol2_raw);
            let (sim_sol2, _) = simplifier.simplify(sol2_expanded);

            // For symbolic solutions, we can't easily order them or determine intervals.
            // We just return them as a discrete set.
            // TODO: Handle inequalities with symbolic coefficients (requires assumptions/cases).
            // For now, only support Eq.
            if eq.op != RelOp::Eq {
                return Some(Err(CasError::SolverError(
                    "Inequalities with symbolic coefficients not yet supported".to_string(),
                )));
            }

            // Emit scope for display transforms (sqrt display in quadratic context)
            crate::solver::emit_scope(cas_formatter::display_transforms::ScopeTag::Rule(
                "QuadraticFormula",
            ));

            return Some(Ok((SolutionSet::Discrete(vec![sim_sol1, sim_sol2]), steps)));
        }

        None
    }

    fn should_verify(&self) -> bool {
        false
    }
}
