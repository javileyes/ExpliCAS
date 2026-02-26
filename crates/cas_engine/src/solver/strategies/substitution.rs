use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{medium_step, render_expr, SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::substitution::{
    execute_exponential_substitution_strategy_result_pipeline_with_items_and_plan_with,
    plan_exponential_substitution_rewrite,
};
use std::cell::RefCell;

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
        let rewrite_plan =
            plan_exponential_substitution_rewrite(&mut simplifier.context, eq, var, SUB_VAR_NAME);
        let render_ctx = RefCell::new(simplifier.context.clone());
        execute_exponential_substitution_strategy_result_pipeline_with_items_and_plan_with(
            eq,
            rewrite_plan,
            var,
            SUB_VAR_NAME,
            include_didactic_items,
            |id| {
                let snapshot = render_ctx.borrow();
                render_expr(&snapshot, id)
            },
            |equation, solve_var| {
                let solved =
                    solve_with_ctx_and_options(equation, solve_var, simplifier, *opts, ctx);
                *render_ctx.borrow_mut() = simplifier.context.clone();
                solved
            },
            medium_step,
        )
    }
}
