use crate::engine::Simplifier;
use crate::error::CasError;
use crate::solver::{SolveCtx, SolveStep, SolverOptions};
use cas_ast::{Equation, SolutionSet};

pub trait SolverStrategy {
    fn name(&self) -> &str;

    /// Attempts to solve the equation using this strategy.
    /// Returns:
    /// - None: Strategy does not apply to this equation.
    /// - Some(Ok(result)): Strategy applied and solved the equation.
    /// - Some(Err(e)): Strategy applied but encountered an error.
    fn apply(
        &self,
        eq: &Equation,
        var: &str,
        simplifier: &mut Simplifier,
        opts: &SolverOptions,
        ctx: &SolveCtx,
    ) -> Option<Result<(SolutionSet, Vec<SolveStep>), CasError>>;

    /// Whether the solutions returned by this strategy should be verified by substitution.
    /// Defaults to true. Override to false if verification is known to be difficult or unreliable (e.g. symbolic roots).
    fn should_verify(&self) -> bool {
        true
    }
}
