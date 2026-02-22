use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, RelOp, SolutionSet};
use cas_solver_core::substitution::{
    build_back_substitute_step_with, build_detected_substitution_step_with,
    build_substituted_equation_step_with, plan_exponential_substitution_rewrite,
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
            let mut steps = Vec::new();
            if simplifier.collect_steps() {
                let detect_step = build_detected_substitution_step_with(
                    eq.clone(),
                    rewrite_plan.substitution_expr,
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
                steps.push(SolveStep {
                    description: detect_step.description,
                    equation_after: detect_step.equation_after,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            let new_eq = rewrite_plan.equation;

            if simplifier.collect_steps() {
                let substituted_step = build_substituted_equation_step_with(new_eq.clone(), |id| {
                    format!(
                        "{}",
                        cas_formatter::DisplayExpr {
                            context: &simplifier.context,
                            id
                        }
                    )
                });
                steps.push(SolveStep {
                    description: substituted_step.description,
                    equation_after: substituted_step.equation_after,
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
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
                    for val in vals {
                        // Solve sub_var_expr = val
                        let sub_eq = Equation {
                            lhs: rewrite_plan.substitution_expr,
                            rhs: val,
                            op: RelOp::Eq,
                        };
                        if simplifier.collect_steps() {
                            let back_sub_step =
                                build_back_substitute_step_with(sub_eq.clone(), |id| {
                                    format!(
                                        "{}",
                                        cas_formatter::DisplayExpr {
                                            context: &simplifier.context,
                                            id
                                        }
                                    )
                                });
                            steps.push(SolveStep {
                                description: back_sub_step.description,
                                equation_after: back_sub_step.equation_after,
                                importance: crate::step::ImportanceLevel::Medium,
                                substeps: vec![],
                            });
                        }
                        let (x_sol, mut x_steps) =
                            match solve_with_ctx(&sub_eq, var, simplifier, ctx) {
                                Ok(res) => res,
                                Err(e) => return Some(Err(e)),
                            };
                        // eprintln!("Back-substitute result for val {:?}: {:?}", val, x_sol);
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
