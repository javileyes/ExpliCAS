use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, SolutionSet};
use cas_solver_core::substitution::{
    build_back_substitution_execution_with, build_exponential_substitution_execution_with,
    collect_back_substitution_execution_items, collect_substitution_intro_execution_items,
    plan_back_substitution_equations, plan_exponential_substitution_rewrite,
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
            let intro_items = collect_substitution_intro_execution_items(&intro_execution);
            let mut steps = Vec::new();
            let new_eq = intro_items
                .last()
                .map(|item| item.equation.clone())
                .unwrap_or_else(|| intro_execution.equation.clone());

            if simplifier.collect_steps() {
                for item in &intro_items {
                    steps.push(SolveStep {
                        description: item.didactic.description.clone(),
                        equation_after: item.equation.clone(),
                        importance: crate::step::ImportanceLevel::Medium,
                        substeps: vec![],
                    });
                }
            }

            // Solve for u
            let (u_solutions, mut u_steps) =
                match solve_with_ctx(&new_eq, SUB_VAR_NAME, simplifier, ctx) {
                    Ok(res) => res,
                    Err(e) => return Some(Err(e)),
                };
            steps.append(&mut u_steps);

            // eprintln!("Substitution u_solutions: {:?}", u_solutions);

            // Now solve u = val for each solution
            match u_solutions {
                SolutionSet::Discrete(vals) => {
                    let mut final_solutions = Vec::new();
                    let back_plan =
                        plan_back_substitution_equations(intro_execution.substitution_expr, &vals);
                    let back_execution = simplifier.collect_steps().then(|| {
                        build_back_substitution_execution_with(back_plan.clone(), |id| {
                            format!(
                                "{}",
                                cas_formatter::DisplayExpr {
                                    context: &simplifier.context,
                                    id
                                }
                            )
                        })
                    });

                    if let Some(back_execution) = back_execution.as_ref() {
                        for item in collect_back_substitution_execution_items(back_execution) {
                            steps.push(SolveStep {
                                description: item.didactic.description,
                                equation_after: item.equation.clone(),
                                importance: crate::step::ImportanceLevel::Medium,
                                substeps: vec![],
                            });
                            let (x_sol, mut x_steps) =
                                match solve_with_ctx(&item.equation, var, simplifier, ctx) {
                                    Ok(res) => res,
                                    Err(e) => return Some(Err(e)),
                                };
                            steps.append(&mut x_steps);

                            if let SolutionSet::Discrete(xs) = x_sol {
                                final_solutions.extend(xs);
                            }
                        }
                    } else {
                        for sub_eq in back_plan.equations {
                            let (x_sol, mut x_steps) =
                                match solve_with_ctx(&sub_eq, var, simplifier, ctx) {
                                    Ok(res) => res,
                                    Err(e) => return Some(Err(e)),
                                };
                            steps.append(&mut x_steps);

                            if let SolutionSet::Discrete(xs) = x_sol {
                                final_solutions.extend(xs);
                            }
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
