use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::substitution::{
    build_back_substitution_solve_plan_with, build_exponential_substitution_execution_with,
    collect_substitution_intro_execution_items, plan_exponential_substitution_rewrite,
    solve_back_substitution_plan_with_items, solve_exponential_substitution_with,
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
        _opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>> {
        const SUB_VAR_NAME: &str = "u";

        if let Some(rewrite_plan) =
            plan_exponential_substitution_rewrite(&mut simplifier.context, eq, var, SUB_VAR_NAME)
        {
            let intro_execution =
                build_exponential_substitution_execution_with(eq.clone(), rewrite_plan, |id| {
                    format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id
                        }
                    )
                });
            let solved_intro =
                match solve_exponential_substitution_with(intro_execution, |new_eq| {
                    solve_with_ctx(new_eq, SUB_VAR_NAME, simplifier, ctx)
                }) {
                    Ok(solved) => solved,
                    Err(e) => return Some(Err(e)),
                };
            let intro_items = collect_substitution_intro_execution_items(&solved_intro.execution);
            let mut steps = Vec::new();
            if simplifier.collect_steps() {
                for item in &intro_items {
                    steps.push(SolveStep {
                        description: item.description().to_string(),
                        equation_after: item.equation.clone(),
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
            }

            // Solve for u
            let (u_solutions, mut u_steps) = solved_intro.solved;
            steps.append(&mut u_steps);

            // eprintln!("Substitution u_solutions: {:?}", u_solutions);

            // Now solve u = val for each solution
            match u_solutions {
                SolutionSet::Discrete(vals) => {
                    let mut final_solutions = Vec::new();
                    let back_plan = build_back_substitution_solve_plan_with(
                        solved_intro.execution.substitution_expr,
                        &vals,
                        simplifier.collect_steps(),
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
                    let solved_back =
                        match solve_back_substitution_plan_with_items(back_plan, |item, sub_eq| {
                            if let Some(item) = item {
                                steps.push(SolveStep {
                                    description: item.description().to_string(),
                                    equation_after: item.equation,
                                    importance: crate::step::ImportanceLevel::Medium,
                                    substeps: vec![],
                                });
                            }
                            solve_with_ctx(sub_eq, var, simplifier, ctx)
                        }) {
                            Ok(solved) => solved,
                            Err(e) => return Some(Err(e)),
                        };

                    for (x_sol, mut x_steps) in solved_back.solved {
                        steps.append(&mut x_steps);
                        if let SolutionSet::Discrete(xs) = x_sol {
                            final_solutions.extend(xs);
                        }
                    }
                    return Some(Ok((SolutionSet::Discrete(final_solutions), steps)));
                }
                _ => {
                    // Handle intervals? Too complex for now.
                    return Some(Err(CasError::SolverError(
                        "Substitution strategy currently only supports discrete solutions"
                            .to_string(),
                    )));
                }
            }
        }
        None
    }
}
