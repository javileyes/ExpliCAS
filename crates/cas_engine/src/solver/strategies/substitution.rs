use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{medium_step, render_expr, SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::substitution::execute_exponential_substitution_strategy_result_pipeline_with_default_plan_with_state;

pub struct SubstitutionStrategy;

impl SolverStrategy for SubstitutionStrategy {
    fn name(&self) -> &str {
        "Substitution"
    }

    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        const SUB_VAR_NAME: &str = "u";
        let include_didactic_items = simplifier.collect_steps();
        execute_exponential_substitution_strategy_result_pipeline_with_default_plan_with_state(
            simplifier,
            eq,
            var,
            SUB_VAR_NAME,
            include_didactic_items,
            |simplifier| &mut simplifier.context,
            |simplifier, id| render_expr(&simplifier.context, id),
            |simplifier, equation, solve_var| {
                solve_with_ctx_and_options(equation, solve_var, simplifier, *opts, ctx)
            },
            medium_step,
        )
    }
}
