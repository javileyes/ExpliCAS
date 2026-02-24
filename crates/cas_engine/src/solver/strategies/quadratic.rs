use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, Expr, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::{is_numeric_zero, split_zero_product_factors};
use cas_solver_core::quadratic_didactic::{
    build_factorized_zero_product_execution_with_optional_items,
    build_quadratic_main_with_substeps_execution_with_optional_items,
    finalize_factorized_zero_product_strategy_solved,
    solve_factorized_zero_product_execution_pipeline_with_items_runtime,
    solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification_runtime,
    FactorizedZeroProductExecutionRuntime, QuadraticExecutionItem,
    QuadraticMainWithSubstepsRuntime, QuadraticSubstepExecutionItem,
    ZeroProductFactorExecutionItem,
};
use cas_solver_core::quadratic_formula::{
    discriminant, discriminant_expr, roots_from_a_b_and_sqrt, roots_from_a_b_delta, sqrt_expr,
};
use cas_solver_core::solution_set::{get_number, order_pair_by_value, quadratic_numeric_solution};
use num_rational::BigRational;
use num_traits::Zero;

pub struct QuadraticStrategy;

struct EngineQuadraticFactorizedRuntime<'a, 'b, 'c> {
    var: &'a str,
    simplifier: &'b mut Simplifier,
    solve_ctx: &'c SolveCtx,
}

impl FactorizedZeroProductExecutionRuntime<CasError, (SolutionSet, Vec<SolveStep>), SolveStep>
    for EngineQuadraticFactorizedRuntime<'_, '_, '_>
{
    fn solve_factor(
        &mut self,
        equation: &Equation,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        solve_with_ctx(equation, self.var, self.simplifier, self.solve_ctx)
    }

    fn map_entry_item_to_step(&mut self, item: QuadraticExecutionItem) -> SolveStep {
        SolveStep {
            description: item.description().to_string(),
            equation_after: item.equation,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        }
    }

    fn map_factor_item_to_step(&mut self, item: ZeroProductFactorExecutionItem) -> SolveStep {
        SolveStep {
            description: item.description,
            equation_after: item.equation,
            importance: crate::step::ImportanceLevel::Medium,
            substeps: vec![],
        }
    }
}

struct EngineQuadraticMainRuntime<'a> {
    simplifier: &'a mut Simplifier,
}

impl QuadraticMainWithSubstepsRuntime<SolveStep, crate::solver::SolveSubStep>
    for EngineQuadraticMainRuntime<'_>
{
    fn simplify_expr(&mut self, expr: cas_ast::ExprId) -> cas_ast::ExprId {
        let (simplified, _) = self.simplifier.simplify(expr);
        simplified
    }

    fn map_main_item_to_step(
        &mut self,
        item: QuadraticExecutionItem,
        substeps: Vec<crate::solver::SolveSubStep>,
    ) -> SolveStep {
        SolveStep {
            description: item.description().to_string(),
            equation_after: item.equation,
            importance: crate::step::ImportanceLevel::Medium,
            substeps,
        }
    }

    fn map_substep_item_to_step(
        &mut self,
        item: QuadraticSubstepExecutionItem,
    ) -> crate::solver::SolveSubStep {
        crate::solver::SolveSubStep {
            description: item.description,
            equation_after: item.equation,
            importance: crate::step::ImportanceLevel::Low,
        }
    }
}

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
            // For inequalities, splitting is complex (sign analysis).
            // For Eq, it's simple union.
            if eq.op == RelOp::Eq {
                let include_items = simplifier.collect_steps();
                let residual_expr = sim_poly_expr;
                let factorized_display = format!(
                    "{}",
                    cas_formatter::DisplayExpr {
                        context: &simplifier.context,
                        id: sim_poly_expr
                    }
                );
                let factor_displays = factors
                    .iter()
                    .copied()
                    .map(|id| {
                        (
                            id,
                            format!(
                                "{}",
                                cas_formatter::DisplayExpr {
                                    context: &simplifier.context,
                                    id
                                }
                            ),
                        )
                    })
                    .collect::<Vec<_>>();
                let factorized_execution =
                    build_factorized_zero_product_execution_with_optional_items(
                        &simplifier.context,
                        sim_poly_expr,
                        &factors,
                        var,
                        zero,
                        include_items,
                        move |id| {
                            if id == sim_poly_expr {
                                factorized_display.clone()
                            } else {
                                factor_displays
                                    .iter()
                                    .find(|(candidate, _)| *candidate == id)
                                    .map(|(_, display)| display.clone())
                                    .unwrap_or_else(|| id.to_string())
                            }
                        },
                    );
                let solved_factorized = {
                    let mut runtime = EngineQuadraticFactorizedRuntime {
                        var,
                        simplifier,
                        solve_ctx: ctx,
                    };
                    match solve_factorized_zero_product_execution_pipeline_with_items_runtime(
                        &factorized_execution,
                        include_items,
                        &mut runtime,
                    ) {
                        Ok(solved) => solved,
                        Err(e) => return Some(Err(e)),
                    }
                };
                let finalized = finalize_factorized_zero_product_strategy_solved(
                    &simplifier.context,
                    solved_factorized,
                    residual_expr,
                    zero,
                );
                steps.extend(finalized.steps);
                return Some(Ok((finalized.solution_set, steps)));
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

            let include_items = simplifier.collect_steps();
            let is_real_only =
                matches!(_opts.value_domain, crate::semantics::ValueDomain::RealOnly);
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
                |core_ctx, id| {
                    format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: core_ctx,
                            id
                        }
                    )
                },
            );
            let was_collecting = simplifier.collect_steps();
            if include_items {
                // Simplify substep equations with step collection disabled to avoid polluting timeline.
                simplifier.set_collect_steps(false);
            }
            let didactic_steps = {
                let mut runtime = EngineQuadraticMainRuntime { simplifier };
                solve_quadratic_main_with_substeps_execution_pipeline_with_optional_items_and_simplification_runtime(
                    &mut execution,
                    include_items,
                    &mut runtime,
                )
            };
            if include_items {
                simplifier.set_collect_steps(was_collecting);
            }
            steps.extend(didactic_steps);

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
