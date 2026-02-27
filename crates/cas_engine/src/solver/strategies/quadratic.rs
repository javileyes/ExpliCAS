use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{medium_step, render_expr, SolveCtx, SolveStep, SolveSubStep, SolverOptions};
use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::is_numeric_zero;
use cas_solver_core::quadratic_coeffs::extract_quadratic_coefficients;
use cas_solver_core::quadratic_didactic::{
    execute_factorized_zero_product_strategy_if_applicable_with_state,
    execute_quadratic_main_didactic_pipeline_with_default_execution_with_state,
};
use cas_solver_core::quadratic_formula::{
    build_quadratic_coefficient_solve_plan, roots_from_a_b_and_simplified_delta,
    solve_quadratic_coefficient_solve_plan_with_state, QuadraticCoefficientSolvePlanError,
};
use cas_solver_core::solution_set::{order_pair_by_value, quadratic_numeric_solution};

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

        let include_items = simplifier.collect_steps();
        if let Some(outcome) = execute_factorized_zero_product_strategy_if_applicable_with_state(
            simplifier,
            eq.op.clone(),
            sim_poly_expr,
            var,
            zero,
            include_items,
            |simplifier| &simplifier.context,
            render_expr,
            |simplifier, equation| {
                solve_with_ctx_and_options(equation, var, simplifier, *opts, ctx)
            },
            |item| medium_step(item.description().to_string(), item.equation),
            |item| medium_step(item.description, item.equation),
        ) {
            return Some(outcome);
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
            let didactic_steps =
                execute_quadratic_main_didactic_pipeline_with_default_execution_with_state(
                    simplifier,
                    var,
                    sim_a,
                    sim_b,
                    sim_c,
                    is_real_only,
                    main_equation,
                    include_items,
                    include_items,
                    |simplifier| &mut simplifier.context,
                    |simplifier, collecting| simplifier.set_collect_steps(collecting),
                    |simplifier, expr| simplifier.simplify(expr).0,
                    render_expr,
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

            let solution_set = solve_quadratic_coefficient_solve_plan_with_state(
                simplifier,
                eq.op.clone(),
                sim_a,
                sim_b,
                plan,
                |simplifier, expr| crate::expand::expand(&mut simplifier.context, expr),
                |simplifier, expr| simplifier.simplify(expr).0,
                |simplifier, eq_op, delta, (sol1, sol2), opens_up| {
                    let (r1, r2) = order_pair_by_value(&simplifier.context, sol1, sol2);
                    quadratic_numeric_solution(
                        &mut simplifier.context,
                        eq_op,
                        &delta,
                        opens_up,
                        r1,
                        r2,
                    )
                },
                |simplifier, a, b, sim_delta| {
                    roots_from_a_b_and_simplified_delta(&mut simplifier.context, a, b, sim_delta)
                },
            );

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
