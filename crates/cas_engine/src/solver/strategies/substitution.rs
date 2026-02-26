use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{medium_step, render_expr, SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::substitution::{
    aggregate_back_substitution_solutions, build_back_substitution_solve_plan_with,
    build_exponential_substitution_execution_with, plan_exponential_substitution_rewrite,
    solve_back_substitution_plan_with_items, solve_exponential_substitution_with_items,
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
        let solved_intro =
            match solve_exponential_substitution_with_items(intro_execution, |items, equation| {
                let mut steps = Vec::new();
                if include_didactic_items {
                    steps.extend(
                        items
                            .into_iter()
                            .map(|item| medium_step(item.description, item.equation)),
                    );
                }
                let (u_solutions, mut u_steps) =
                    solve_with_ctx_and_options(equation, SUB_VAR_NAME, simplifier, *opts, ctx)?;
                steps.append(&mut u_steps);
                Ok::<(SolutionSet, Vec<SolveStep>), CasError>((u_solutions, steps))
            }) {
                Ok(solved) => solved,
                Err(err) => return Some(Err(err)),
            };

        let (u_solutions, mut steps) = solved_intro.solved;
        match u_solutions {
            SolutionSet::Discrete(vals) => {
                let back_plan = build_back_substitution_solve_plan_with(
                    solved_intro.execution.substitution_expr,
                    &vals,
                    include_didactic_items,
                    |id| render_expr(&simplifier.context, id),
                );
                let solved_back =
                    match solve_back_substitution_plan_with_items(back_plan, |item, equation| {
                        let mut local_steps = Vec::new();
                        if include_didactic_items {
                            if let Some(item) = item {
                                local_steps.push(medium_step(item.description, item.equation));
                            }
                        }
                        let (x_solution_set, mut x_steps) =
                            solve_with_ctx_and_options(equation, var, simplifier, *opts, ctx)?;
                        local_steps.append(&mut x_steps);
                        Ok::<(SolutionSet, Vec<SolveStep>), CasError>((x_solution_set, local_steps))
                    }) {
                        Ok(solved) => solved,
                        Err(err) => return Some(Err(err)),
                    };

                let solution_set =
                    match aggregate_back_substitution_solutions(solved_back.solved, &mut steps) {
                        Ok(final_solutions) => SolutionSet::Discrete(final_solutions),
                        Err(solution_set) => solution_set,
                    };
                Some(Ok((solution_set, steps)))
            }
            solution_set => Some(Ok((solution_set, steps))),
        }
    }
}
