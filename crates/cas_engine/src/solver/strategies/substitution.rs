use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::solve_core::solve_with_ctx;
use crate::solver::strategy::SolverStrategy;
use crate::solver::{SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, RelOp, SolutionSet};
use cas_solver_core::isolation_utils::contains_var;
use cas_solver_core::substitution::{
    back_substitute_message, detected_substitution_message, substituted_equation_message,
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
        if let Some(sub_var_expr) = cas_solver_core::substitution::detect_exponential_substitution(
            &mut simplifier.context,
            eq.lhs,
            eq.rhs,
            var,
        ) {
            let mut steps = Vec::new();
            if simplifier.collect_steps() {
                let sub_expr_desc = format!("{:?}", sub_var_expr);
                steps.push(SolveStep {
                    description: detected_substitution_message(&sub_expr_desc),
                    equation_after: eq.clone(),
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            // Rewrite equation in terms of u
            let u_sym = "u";
            let u_var = simplifier.context.var(u_sym);
            let new_lhs = cas_solver_core::substitution::substitute_expr_pattern(
                &mut simplifier.context,
                eq.lhs,
                sub_var_expr,
                u_var,
            );
            let new_rhs = cas_solver_core::substitution::substitute_expr_pattern(
                &mut simplifier.context,
                eq.rhs,
                sub_var_expr,
                u_var,
            );

            let new_eq = Equation {
                lhs: new_lhs,
                rhs: new_rhs,
                op: eq.op.clone(),
            };

            // Safety net: if the substituted equation still contains the original variable,
            // the substitution was incomplete (mixed polynomial+exponential that slipped past
            // the detect_exponential_substitution guard). Bail out instead of invalid solve path.
            if contains_var(&simplifier.context, new_lhs, var)
                || contains_var(&simplifier.context, new_rhs, var)
            {
                return None;
            }

            if simplifier.collect_steps() {
                let lhs_desc = format!("{:?}", new_eq.lhs);
                let rhs_desc = format!("{:?}", new_eq.rhs);
                let op_desc = new_eq.op.to_string();
                steps.push(SolveStep {
                    description: substituted_equation_message(&lhs_desc, &op_desc, &rhs_desc),
                    equation_after: new_eq.clone(),
                    importance: crate::step::ImportanceLevel::Medium,
                    substeps: vec![],
                });
            }

            // Solve for u
            let (u_solutions, mut u_steps) = match solve_with_ctx(&new_eq, u_sym, simplifier, ctx) {
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
                            lhs: sub_var_expr,
                            rhs: val,
                            op: RelOp::Eq,
                        };
                        if simplifier.collect_steps() {
                            let lhs_desc = format!("{:?}", sub_var_expr);
                            let rhs_desc = format!("{:?}", val);
                            steps.push(SolveStep {
                                description: back_substitute_message(&lhs_desc, &rhs_desc),
                                equation_after: sub_eq.clone(),
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
