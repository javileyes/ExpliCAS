use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx_and_options;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{medium_step, render_expr, SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::substitution::{
    plan_exponential_substitution_rewrite, solve_exponential_substitution_strategy_with_items_with,
    SubstitutionStrategySolved,
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

        if let Some(rewrite_plan) =
            plan_exponential_substitution_rewrite(&mut simplifier.context, eq, var, SUB_VAR_NAME)
        {
            let include_didactic_items = simplifier.collect_steps();
            let solved = {
                let runtime_cell = std::cell::RefCell::new(&mut *simplifier);
                solve_exponential_substitution_strategy_with_items_with(
                    eq.clone(),
                    rewrite_plan,
                    var,
                    SUB_VAR_NAME,
                    include_didactic_items,
                    |expr| {
                        let simplifier_ref = runtime_cell.borrow();
                        render_expr(&simplifier_ref.context, expr)
                    },
                    |equation, solve_var| {
                        let mut simplifier_ref = runtime_cell.borrow_mut();
                        solve_with_ctx_and_options(equation, solve_var, *simplifier_ref, *opts, ctx)
                    },
                    medium_step,
                )
            };

            return match solved {
                Ok(SubstitutionStrategySolved::SolvedDiscrete { solutions, steps }) => {
                    Some(Ok((SolutionSet::Discrete(solutions), steps)))
                }
                Ok(SubstitutionStrategySolved::UnsupportedSolutionSet {
                    solution_set,
                    steps,
                }) => Some(Ok((solution_set, steps))),
                Err(e) => Some(Err(e)),
            };
        }
        None
    }
}
