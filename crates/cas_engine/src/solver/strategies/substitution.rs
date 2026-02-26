use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{medium_step, render_expr, SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::substitution::{
    aggregate_back_substitution_solutions, build_back_substitution_solve_plan_with,
    build_exponential_substitution_execution_with, plan_exponential_substitution_rewrite,
    solve_back_substitution_plan_execution_pipeline_with_items,
    solve_exponential_substitution_execution_pipeline_with_items,
};

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
        let rewrite_plan = rewrite_plan?;

        let intro_execution =
            build_exponential_substitution_execution_with(eq.clone(), rewrite_plan, |id| {
                render_expr(&simplifier.context, id)
            });
        let solved_intro = match solve_exponential_substitution_execution_pipeline_with_items(
            intro_execution,
            include_didactic_items,
            SUB_VAR_NAME,
            |equation, solve_var| {
                solve_with_ctx_and_options(equation, solve_var, simplifier, *opts, ctx)
            },
            |item| medium_step(item.description, item.equation),
        ) {
            Ok(solved) => solved,
            Err(err) => return Some(Err(err)),
        };

        let u_solutions = solved_intro.solution_set;
        let mut steps = solved_intro.steps;
        match u_solutions {
            SolutionSet::Discrete(vals) => {
                let back_plan = build_back_substitution_solve_plan_with(
                    solved_intro.substitution_expr,
                    &vals,
                    include_didactic_items,
                    |id| render_expr(&simplifier.context, id),
                );
                let solved_back = match solve_back_substitution_plan_execution_pipeline_with_items(
                    back_plan,
                    include_didactic_items,
                    var,
                    |equation, solve_var| {
                        solve_with_ctx_and_options(equation, solve_var, simplifier, *opts, ctx)
                    },
                    |item| medium_step(item.description, item.equation),
                ) {
                    Ok(solved) => solved,
                    Err(err) => return Some(Err(err)),
                };

                let solution_set =
                    match aggregate_back_substitution_solutions(solved_back, &mut steps) {
                        Ok(final_solutions) => SolutionSet::Discrete(final_solutions),
                        Err(solution_set) => solution_set,
                    };
                Some(Ok((solution_set, steps)))
            }
            solution_set => Some(Ok((solution_set, steps))),
        }
    }
}
