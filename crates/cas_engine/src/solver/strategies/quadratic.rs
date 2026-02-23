use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::{is_numeric_zero, split_zero_product_factors};
use cas_solver_core::quadratic_didactic::{
    aggregate_zero_product_factor_solution_sets, build_quadratic_main_with_substeps_execution_with,
    finalize_zero_product_factor_solution_set, first_factorized_zero_product_entry_execution_item,
    simplify_quadratic_substep_execution_items_with,
    solve_zero_product_factor_execution_with_items, ZeroProductFactorSolutionAggregate,
};
use cas_solver_core::quadratic_formula::{
    discriminant, discriminant_expr, roots_from_a_b_and_sqrt, roots_from_a_b_delta, sqrt_expr,
};
use cas_solver_core::solution_set::{get_number, order_pair_by_value, quadratic_numeric_solution};
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
            let factorized_execution =
                cas_solver_core::quadratic_didactic::build_factorized_zero_product_execution_with(
                    &simplifier.context,
                    sim_poly_expr,
                    &factors,
                    var,
                    zero,
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

            // We found factors.
            if simplifier.collect_steps() {
                if let Some(item) =
                    first_factorized_zero_product_entry_execution_item(&factorized_execution)
                {
                    steps.push(SolveStep {
                        description: item.description().to_string(),
                        equation_after: item.equation,
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
            }

            // For inequalities, splitting is complex (sign analysis).
            // For Eq, it's simple union.
            if eq.op == RelOp::Eq {
                let solved_factors = match solve_zero_product_factor_execution_with_items(
                    &factorized_execution.factors,
                    |item, factor_equation| {
                        if simplifier.collect_steps() {
                            if let Some(item) = item {
                                steps.push(SolveStep {
                                    description: item.description,
                                    equation_after: item.equation,
                                    importance: crate::step::ImportanceLevel::Medium,
                                    substeps: vec![],
                                });
                            }
                        }
                        // Recursive solve
                        // We need to be careful about depth.
                        solve_with_ctx(factor_equation, var, simplifier, ctx)
                    },
                ) {
                    Ok(solved) => solved,
                    Err(e) => return Some(Err(e)),
                };
                let mut factor_solution_sets = Vec::new();
                for (sol_set, mut sub_steps) in solved_factors {
                    steps.append(&mut sub_steps);
                    factor_solution_sets.push(sol_set);
                }
                let aggregate = aggregate_zero_product_factor_solution_sets(
                    &simplifier.context,
                    factor_solution_sets,
                );
                let residual_expr =
                    if matches!(aggregate, ZeroProductFactorSolutionAggregate::NonDiscrete) {
                        // Residual/Conditional/Interval: can't extract discrete roots.
                        // Keep whole equation as residual solve target.
                        let residual = simplifier.context.add(Expr::Sub(eq.lhs, eq.rhs));
                        let (sim, _) = simplifier.simplify(residual);
                        sim
                    } else {
                        // Unused for discrete/all-reals aggregates.
                        simplifier.context.num(0)
                    };
                let final_set = finalize_zero_product_factor_solution_set(aggregate, residual_expr);
                return Some(Ok((final_set, steps)));
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
                let is_real_only =
                    matches!(_opts.value_domain, crate::semantics::ValueDomain::RealOnly);

                let main_equation = Equation {
                    lhs: sim_poly_expr,
                    rhs: simplifier.context.num(0),
                    op: RelOp::Eq,
                };
                let mut execution = build_quadratic_main_with_substeps_execution_with(
                    &mut simplifier.context,
                    var,
                    sim_a,
                    sim_b,
                    sim_c,
                    is_real_only,
                    main_equation,
                    |core_ctx, id| {
                        format!(
                            "{}",
                            cas_formatter::DisplayExpr {
                                context: core_ctx,
                                id
                            }
                        )
                    },
                    |id| id,
                );

                // Simplify substep equations with step collection disabled to avoid polluting timeline.
                let was_collecting = simplifier.collect_steps();
                simplifier.set_collect_steps(false);
                simplify_quadratic_substep_execution_items_with(
                    &mut execution.substep_items,
                    |id| {
                        let (simplified, _) = simplifier.simplify(id);
                        simplified
                    },
                );
                simplifier.set_collect_steps(was_collecting);

                let substeps = execution
                    .substep_items
                    .into_iter()
                    .map(|item| crate::solver::SolveSubStep {
                        description: item.description,
                        equation_after: item.equation,
                        importance: crate::step::ImportanceLevel::Low,
                    })
                    .collect::<Vec<_>>();

                for item in execution.main_items {
                    steps.push(SolveStep {
                        description: item.description().to_string(),
                        equation_after: item.equation,
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: substeps.clone(),
                    });
                }
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
