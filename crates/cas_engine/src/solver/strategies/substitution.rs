use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{medium_step, render_expr, SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, ExprId, SolutionSet};
use cas_solver_core::substitution::{
    plan_exponential_substitution_rewrite, solve_exponential_substitution_strategy_with_items,
    SubstitutionStrategyRuntime, SubstitutionStrategySolved,
};

pub struct SubstitutionStrategy;

struct EngineSubstitutionRuntime<'a, 'ctx> {
    simplifier: &'a mut Simplifier,
    solve_ctx: &'ctx SolveCtx,
}

impl SubstitutionStrategyRuntime<CasError, SolveStep> for EngineSubstitutionRuntime<'_, '_> {
    fn render_expr(&mut self, expr: ExprId) -> String {
        render_expr(&self.simplifier.context, expr)
    }

    fn solve_equation(
        &mut self,
        equation: &Equation,
        var: &str,
    ) -> Result<(SolutionSet, Vec<SolveStep>), CasError> {
        solve_with_ctx(equation, var, self.simplifier, self.solve_ctx)
    }

    fn map_step(&mut self, description: String, equation_after: Equation) -> SolveStep {
        medium_step(description, equation_after)
    }
}

impl SolverStrategy for SubstitutionStrategy {
    fn name(&self) -> &str {
        "Substitution"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        _opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        const SUB_VAR_NAME: &str = "u";

        if let Some(rewrite_plan) =
            plan_exponential_substitution_rewrite(&mut simplifier.context, eq, var, SUB_VAR_NAME)
        {
            let include_didactic_items = simplifier.collect_steps();
            let mut runtime = EngineSubstitutionRuntime {
                simplifier,
                solve_ctx: ctx,
            };
            let solved = solve_exponential_substitution_strategy_with_items(
                &mut runtime,
                eq.clone(),
                rewrite_plan,
                var,
                SUB_VAR_NAME,
                include_didactic_items,
            );

            return match solved {
                Ok(SubstitutionStrategySolved::SolvedDiscrete { solutions, steps }) => {
                    Some(Ok((SolutionSet::Discrete(solutions), steps)))
                }
                Ok(SubstitutionStrategySolved::UnsupportedSolutionSet { .. }) => {
                    Some(Err(CasError::SolverError(
                        "Substitution strategy currently only supports discrete solutions"
                            .to_string(),
                    )))
                }
                Err(e) => Some(Err(e)),
            };
        }
        None
    }
}
