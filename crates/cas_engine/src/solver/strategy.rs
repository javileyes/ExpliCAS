use cas_ast::{Equation, SolutionSet};
use crate::engine::Simplifier;
use crate::solver::SolveStep;
use crate::error::CasError;

pub trait SolverStrategy {
    fn name(&self) -> &str;
    
    /// Attempts to solve the equation using this strategy.
    /// Returns:
    /// - None: Strategy does not apply to this equation.
    /// - Some(Ok(result)): Strategy applied and solved the equation.
    /// - Some(Err(e)): Strategy applied but encountered an error.
    fn apply(&self, eq: &Equation, var: &str, simplifier: &mut Simplifier) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>>;
}
