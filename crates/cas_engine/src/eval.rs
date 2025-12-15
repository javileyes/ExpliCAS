use crate::session::{EntryKind, ResolveError};
use crate::session_state::SessionState;
use crate::Simplifier;
use cas_ast::{Equation, Expr, ExprId, RelOp};

/// The central Engine struct that wraps the core Simplifier and potentially other components.
pub struct Engine {
    pub simplifier: Simplifier,
}

impl Engine {
    pub fn new() -> Self {
        Self {
            simplifier: Simplifier::with_default_rules(),
        }
    }
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug)]
pub enum EvalAction {
    Simplify,
    Expand,
    // Solve for a variable.
    Solve { var: String },
    // Check equivalence between two expressions
    Equiv { other: ExprId },
}

#[derive(Clone, Debug)]
pub struct EvalRequest {
    pub raw_input: String,
    pub parsed: ExprId,
    pub kind: EntryKind,
    pub action: EvalAction,
    pub auto_store: bool,
}

#[derive(Clone, Debug)]
pub enum EvalResult {
    Expr(ExprId),
    Set(Vec<ExprId>), // For Solve multiple roots
    Bool(bool),       // For Equiv
    None,             // For commands with no output
}

#[derive(Clone, Debug)]
pub struct EvalOutput {
    pub stored_id: Option<u64>,
    pub parsed: ExprId,
    pub resolved: ExprId,
    pub result: EvalResult,
    pub domain_warnings: Vec<String>,
    pub steps: Vec<crate::Step>,
    pub solve_steps: Vec<crate::solver::SolveStep>,
}

impl Engine {
    /// The main entry point for evaluating requests.
    /// Handles session storage, resolution, and action dispatch.
    pub fn eval(
        &mut self,
        state: &mut SessionState,
        req: EvalRequest,
    ) -> Result<EvalOutput, anyhow::Error> {
        // 1. Auto-store raw + parsed (unresolved)
        let stored_id = if req.auto_store {
            Some(state.store.push(req.kind, req.raw_input.clone()))
        } else {
            None
        };

        // 2. Resolve (state.resolve_all)
        // We resolve the parsed expression against the session state (Session refs #id and Environment vars)
        let resolved = match state.resolve_all(&mut self.simplifier.context, req.parsed) {
            Ok(r) => r,
            Err(ResolveError::CircularReference(msg)) => {
                return Err(anyhow::anyhow!("Circular reference detected: {}", msg))
            }
            Err(e) => return Err(anyhow::anyhow!("Resolution error: {}", e)),
        };

        // 3. Dispatch Action -> produce EvalResult
        let (result, domain_warnings, steps, solve_steps) = match req.action {
            EvalAction::Simplify => {
                let (res, steps) = self.simplifier.simplify(resolved);
                // Collect domain assumptions from steps
                let warnings: Vec<String> = steps
                    .iter()
                    .filter_map(|s| s.domain_assumption.as_ref().map(|d| d.to_string()))
                    .collect();
                (EvalResult::Expr(res), warnings, steps, vec![])
            }
            EvalAction::Expand => {
                // Treating Expand as Simplify for now, as Simplifier has no explicit expand mode yet exposed cleanly
                let (res, steps) = self.simplifier.simplify(resolved);
                let warnings: Vec<String> = steps
                    .iter()
                    .filter_map(|s| s.domain_assumption.as_ref().map(|d| d.to_string()))
                    .collect();
                (EvalResult::Expr(res), warnings, steps, vec![])
            }
            EvalAction::Solve { var } => {
                // Construct proper Equation for solver
                // If resolved is "Equal(lhs, rhs)", use that.
                // Otherwise assume "resolved == 0".

                // We must peek at the resolved expression structure
                let eq_to_solve = match self.simplifier.context.get(resolved) {
                    Expr::Function(name, args) if name == "Equal" && args.len() == 2 => {
                        Equation {
                            lhs: args[0],
                            rhs: args[1],
                            op: RelOp::Eq, // Assuming strict equality for Solve for now
                        }
                    }
                    Expr::Function(name, args) if name == "Less" && args.len() == 2 => Equation {
                        lhs: args[0],
                        rhs: args[1],
                        op: RelOp::Lt,
                    },
                    Expr::Function(name, args) if name == "Greater" && args.len() == 2 => {
                        Equation {
                            lhs: args[0],
                            rhs: args[1],
                            op: RelOp::Gt,
                        }
                    }
                    // Handle other ops if needed, or default to Expr = 0
                    _ => {
                        use num_traits::Zero;
                        let zero = self
                            .simplifier
                            .context
                            .add(Expr::Number(num_rational::BigRational::zero()));
                        Equation {
                            lhs: resolved,
                            rhs: zero,
                            op: RelOp::Eq,
                        }
                    }
                };

                // Call solver
                let sol_result = crate::solver::solve(&eq_to_solve, &var, &mut self.simplifier);

                match sol_result {
                    Ok((solution_set, solve_steps)) => {
                        // Convert SolutionSet to EvalResult::Set (of ExprIds)
                        // If Discrete, straightforward. If Continuous/AllReals, might need representation.
                        // EvalResult::Set expects Vec<ExprId>.
                        // SolutionSet::Discrete(Vec<ExprId>).
                        // SolutionSet::AllReals -> no standard Expr representation yet? Or maybe a special Expr?
                        // Let's rely on what solve produces.

                        use cas_ast::SolutionSet;
                        // For Solve, currently no domain warnings in steps, but we can collect descriptions if we wanted.
                        let warnings: Vec<String> = vec![];

                        let eval_res = match solution_set {
                            SolutionSet::Discrete(sols) => EvalResult::Set(sols),
                            SolutionSet::AllReals => {
                                // Return empty list or special token?
                                // For now let's return EvalResult::None or handle via text output in CLI?
                                // The user requested EvalResult::Set(Vec<ExprId>).
                                // Let's create a special expression "AllReals"?
                                let id = self
                                    .simplifier
                                    .context
                                    .add(Expr::Variable("All Reals".to_string())); // Hacky
                                EvalResult::Set(vec![id])
                            }
                            SolutionSet::Empty => EvalResult::Set(vec![]),
                            SolutionSet::Continuous(interval) => {
                                // Convert interval to Expr representation?
                                // Interval has min, max.
                                // Let's return min and max in result set for now?? No that's confusing.
                                // Let's return the simplified interval boundaries as expressions.
                                // Or create a Tuple expr.
                                // "EvalResult::Set" implies discrete solutions.
                                // But User said "EvalResult::Set(Vec<ExprId>)".
                                // I'll return the interval bounds as 2 elements if continuous.
                                EvalResult::Set(vec![interval.min, interval.max])
                            }
                            SolutionSet::Union(intervals) => {
                                let mut bounds = Vec::new();
                                for interval in intervals {
                                    bounds.push(interval.min);
                                    bounds.push(interval.max);
                                }
                                EvalResult::Set(bounds)
                            }
                        };
                        (eval_res, warnings, vec![], solve_steps)
                    }
                    Err(e) => return Err(anyhow::anyhow!("Solver error: {}", e)),
                }
            }
            EvalAction::Equiv { other } => {
                let resolved_other = match state.resolve_all(&mut self.simplifier.context, other) {
                    Ok(r) => r,
                    Err(ResolveError::CircularReference(msg)) => {
                        return Err(anyhow::anyhow!(
                            "Circular reference detected in other: {}",
                            msg
                        ))
                    }
                    Err(e) => return Err(anyhow::anyhow!("Resolution error in other: {}", e)),
                };

                let are_eq = self.simplifier.are_equivalent(resolved, resolved_other);
                (EvalResult::Bool(are_eq), vec![], vec![], vec![])
            }
        };

        Ok(EvalOutput {
            stored_id,
            parsed: req.parsed,
            resolved,
            result,
            domain_warnings,
            steps,
            solve_steps,
        })
    }
}
