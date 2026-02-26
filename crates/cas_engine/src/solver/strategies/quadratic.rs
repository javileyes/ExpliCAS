use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{medium_step, render_expr, SolveCtx, SolveStep, SolveSubStep, SolverOptions};
use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::{is_numeric_zero, split_zero_product_factors};
use cas_solver_core::quadratic_coeffs::extract_quadratic_coefficients;
use cas_solver_core::quadratic_didactic::{
    build_factorized_zero_product_execution_with_optional_items,
    build_quadratic_main_with_substeps_execution_with_optional_items,
    finalize_factorized_zero_product_strategy_solved,
    solve_factorized_zero_product_execution_result_pipeline_with_items,
    solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification,
};
use cas_solver_core::quadratic_formula::{
    build_quadratic_coefficient_solve_plan, roots_from_a_b_and_simplified_delta,
    QuadraticCoefficientSolvePlan, QuadraticCoefficientSolvePlanError,
};
use cas_solver_core::solution_set::{order_pair_by_value, quadratic_numeric_solution};
use std::cell::RefCell;

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
        opts: &SolverOptions,
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
            // For inequalities, splitting is complex (sign analysis).
            // For Eq, it's simple union.
            if eq.op == RelOp::Eq {
                let include_items = simplifier.collect_steps();
                let residual_expr = sim_poly_expr;
                let factorized_execution =
                    build_factorized_zero_product_execution_with_optional_items(
                        &simplifier.context,
                        sim_poly_expr,
                        &factors,
                        var,
                        zero,
                        include_items,
                        |expr| render_expr(&simplifier.context, expr),
                    );
                let factor_ctx = RefCell::new(simplifier.context.clone());
                let solved =
                    match solve_factorized_zero_product_execution_result_pipeline_with_items(
                        &factorized_execution,
                        include_items,
                        |equation| {
                            let solved =
                                solve_with_ctx_and_options(equation, var, simplifier, *opts, ctx);
                            *factor_ctx.borrow_mut() = simplifier.context.clone();
                            solved
                        },
                        |item| medium_step(item.description().to_string(), item.equation),
                        |item| medium_step(item.description, item.equation),
                        |solved| {
                            let snapshot = factor_ctx.borrow();
                            finalize_factorized_zero_product_strategy_solved(
                                &snapshot,
                                solved,
                                residual_expr,
                                zero,
                            )
                        },
                    ) {
                        Ok(solved) => solved,
                        Err(e) => return Some(Err(e)),
                    };
                return Some(Ok(solved));
            }
        }

        // Ensure expanded form for coefficient extraction
        // QuadraticStrategy relies on A*x^2 + B*x + C structure (Add/Sub chain)
        let expanded_expr = crate::expand::expand(&mut simplifier.context, sim_poly_expr);

        let coeffs = extract_quadratic_coefficients(&mut simplifier.context, expanded_expr, var)
            .and_then(|(a, b, c)| {
                let sim_a = simplifier.simplify(a).0;
                let sim_b = simplifier.simplify(b).0;
                let sim_c = simplifier.simplify(c).0;

                if is_numeric_zero(&simplifier.context, sim_a) {
                    None
                } else {
                    Some((sim_a, sim_b, sim_c))
                }
            });

        if let Some((sim_a, sim_b, sim_c)) = coeffs {
            let include_items = simplifier.collect_steps();
            let is_real_only = matches!(opts.value_domain, crate::semantics::ValueDomain::RealOnly);
            let main_equation = Equation {
                lhs: sim_poly_expr,
                rhs: simplifier.context.num(0),
                op: RelOp::Eq,
            };
            let mut execution = build_quadratic_main_with_substeps_execution_with_optional_items(
                &mut simplifier.context,
                var,
                sim_a,
                sim_b,
                sim_c,
                is_real_only,
                main_equation,
                include_items,
                render_expr,
            );
            let was_collecting = simplifier.collect_steps();
            simplifier.set_collect_steps(false);
            let didactic_steps =
                solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification(
                    &mut execution,
                    include_items,
                    |expr| simplifier.simplify(expr).0,
                    |item, substeps| {
                        medium_step(item.description().to_string(), item.equation)
                            .with_substeps(substeps)
                    },
                    |item| SolveSubStep {
                        description: item.description,
                        equation_after: item.equation,
                        importance: crate::step::ImportanceLevel::Low,
                    },
                );
            simplifier.set_collect_steps(was_collecting);
            steps.extend(didactic_steps);

            let plan = match build_quadratic_coefficient_solve_plan(
                &mut simplifier.context,
                eq.op.clone(),
                sim_a,
                sim_b,
                sim_c,
            ) {
                Ok(plan) => plan,
                Err(QuadraticCoefficientSolvePlanError::UnsupportedSymbolicInequality) => {
                    return Some(Err(CasError::SolverError(
                        "Inequalities with symbolic coefficients not yet supported".to_string(),
                    )));
                }
            };

            let solution_set = match plan {
                QuadraticCoefficientSolvePlan::Numeric {
                    delta,
                    roots: (sol1, sol2),
                    opens_up,
                } => {
                    let sim_sol1 = simplifier.simplify(sol1).0;
                    let sim_sol2 = simplifier.simplify(sol2).0;
                    let (r1, r2) = order_pair_by_value(&simplifier.context, sim_sol1, sim_sol2);
                    quadratic_numeric_solution(
                        &mut simplifier.context,
                        eq.op.clone(),
                        &delta,
                        opens_up,
                        r1,
                        r2,
                    )
                }
                QuadraticCoefficientSolvePlan::SymbolicEq { delta_expr } => {
                    let delta_expanded = crate::expand::expand(&mut simplifier.context, delta_expr);
                    let sim_delta = simplifier.simplify(delta_expanded).0;

                    let (sol1_raw, sol2_raw) = roots_from_a_b_and_simplified_delta(
                        &mut simplifier.context,
                        sim_a,
                        sim_b,
                        sim_delta,
                    );
                    let sol1_expanded = crate::expand::expand(&mut simplifier.context, sol1_raw);
                    let sim_sol1 = simplifier.simplify(sol1_expanded).0;
                    let sol2_expanded = crate::expand::expand(&mut simplifier.context, sol2_raw);
                    let sim_sol2 = simplifier.simplify(sol2_expanded).0;
                    SolutionSet::Discrete(vec![sim_sol1, sim_sol2])
                }
            };

            // Emit scope for display transforms (sqrt display in quadratic context)
            ctx.emit_scope(cas_formatter::display_transforms::ScopeTag::Rule(
                "QuadraticFormula",
            ));

            return Some(Ok((solution_set, steps)));
        }

        None
    }

    fn should_verify(&self) -> bool {
        false
    }
}
